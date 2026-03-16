# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
# SGTM: forked from litgpt/pretrain.py — adds dual-optimizer 3-path SGTM training loop.

import math
import pprint
import random  # SGTM: split sampling via random.choices
import time
import warnings
from dataclasses import asdict
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import lightning as L
import torch
import torch.nn as nn
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.throughput import ThroughputMonitor, measure_flops
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.aggregation import RunningMean
from typing_extensions import Literal

from litgpt import Tokenizer
from litgpt.args import EvalArgs, LogArgs, TrainArgs
from litgpt.config import name_to_config
from litgpt.constants import _TORCH_EQUAL_2_7, _TORCH_EQUAL_2_8
from litgpt.model import GPT, Block, CausalSelfAttention, Config, LLaMAMLP
from litgpt.parser_config import save_hyperparameters
from litgpt.types import LoggerChoice
from litgpt.utils import (
    CycleIterator,
    capture_hparams,
    check_nvlink_connectivity,
    choose_logger,
    chunked_cross_entropy,
    copy_config_files,
    extend_checkpoint_dir,
    find_resume_path,
    get_default_supported_precision,
    init_out_dir,
    instantiate_torch_optimizer,
    num_parameters,
    parse_devices,
    reset_parameters,
    save_config,
)

# SGTM: safemoe-specific imports for dual-optimizer and masking infrastructure
from safemoe.config import SafeMoEConfig
from safemoe.data.datamodule import MultiDataLoader
from safemoe.masking import ActivationMasker, GradientMasker, HarmfulParamRegistry

# SGTM: split labels for 3-path SGTM branching
SPLIT_LABELS = ["D_std", "D_harmful", "D_unlabeled"]


# ---------------------------------------------------------------------------
# SGTM: SafeCausalSelfAttention — head-output zeroing for TRAIN-02
# ---------------------------------------------------------------------------

class SafeCausalSelfAttention(CausalSelfAttention):
    """CausalSelfAttention extended with activation-masking support for harmful heads.

    When ``_activation_masking_enabled`` is True, head outputs for each index in
    ``_harmful_heads`` are zeroed before the reshape+proj step.  This implements
    the TRAIN-02 locked decision: attn_out[:, :, head_idx, :] = 0 before
    reshape and output projection.

    Note on tensor layout: ``scaled_dot_product_attention`` returns
    ``y.transpose(1, 2)`` which gives shape ``(B, T, n_head, hs)``.
    The harmful-head zero is therefore applied at dimension 2 (head axis),
    i.e. ``y[:, :, head_idx, :] = 0``.
    """

    _activation_masking_enabled: bool = False
    _harmful_heads: List[int] = []

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        mask: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
        input_pos_maxp1: Optional[int] = None,
    ) -> Tensor:
        # Run full parent computation up to but not including reshape+proj.
        # We replicate the parent forward body so we can intercept y before
        # reshape — calling super().forward() returns the final output which
        # is too late to zero head slices.
        head_size = self.config.head_size
        n_head = self.config.n_head
        n_query_groups = self.config.n_query_groups
        rope_n_elem = self.config.rope_n_elem
        B, T, C = x.size()

        qkv = self.qkv(x)

        query_size = n_head * head_size
        key_size = value_size = n_query_groups * head_size
        q, k, v = qkv.split((query_size, key_size, value_size), dim=-1)

        if self.config.norm_qk and self.config.norm_qk_type == "olmo2":
            q = self.norm_q(q)
            k = self.norm_k(k)

        q = q.view(B, T, n_head, head_size)
        k = k.view(B, T, n_query_groups, head_size)
        v = v.view(B, T, n_query_groups, head_size)

        q = q.transpose(1, 2)  # (B, nh_q, T, hs)
        k = k.transpose(1, 2)  # (B, nh_k, T, hs)
        v = v.transpose(1, 2)  # (B, nh_v, T, hs)

        if self.config.norm_qk and self.config.norm_qk_type == "default":
            q = self.norm_q(q)
            k = self.norm_k(k)

        q_roped = apply_rope(q[..., :rope_n_elem], cos, sin)
        k_roped = apply_rope(k[..., :rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., rope_n_elem:]), dim=-1)
        k = torch.cat((k_roped, k[..., rope_n_elem:]), dim=-1)

        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            k, v = self.kv_cache(input_pos, k, v)

            if self.apply_sliding_window_attention:
                actual_kv_len = k.size(2)
                if mask is not None and mask.size(-1) != actual_kv_len:
                    mask = mask[..., :actual_kv_len]

            if input_pos_maxp1 is not None:
                k = k[..., :input_pos_maxp1, :]
                v = v[..., :input_pos_maxp1, :]

        if n_query_groups != n_head and (input_pos is None or n_query_groups != 1):
            q_per_kv = n_head // n_query_groups
            k = k.repeat_interleave(q_per_kv, dim=1)
            v = v.repeat_interleave(q_per_kv, dim=1)

        if self.apply_sliding_window_attention:
            if input_pos is None:
                if mask is None:
                    mask = torch.ones(T, T, dtype=q.dtype, device=q.device).triu(diagonal=1)
                    mask.masked_fill_(mask.bool(), float("-inf"))
                    mask = mask.view(1, 1, *mask.shape)

                sliding_window_mask = torch.full((T, T), float("-inf"), dtype=q.dtype, device=q.device)
                for i in range(T):
                    window_start = max(0, i - self.config.sliding_window_size + 1)
                    sliding_window_mask[i, window_start : i + 1] = 0.0
                sliding_window_mask = sliding_window_mask.view(1, 1, T, T)
                mask = sliding_window_mask

        # y: (B, T, n_head, hs)  — scaled_dot_product_attention already does transpose(1,2)
        y = self.scaled_dot_product_attention(q, k, v, mask)

        # SGTM: zero harmful-head outputs before reshape+proj (TRAIN-02 locked decision)
        # Clone y before in-place zeroing to avoid corrupting the autograd graph version
        # (in-place ops on sdpa output cause RuntimeError during backward).
        if self._activation_masking_enabled:
            y = y.clone()
            for head_idx in self._harmful_heads:
                y[:, :, head_idx, :] = 0

        # Re-assemble all head outputs side by side.
        y = y.reshape(B, T, head_size * n_head)

        # Output projection.
        return self.proj(y)


# Import apply_rope from litgpt.model so SafeCausalSelfAttention can use it
from litgpt.model import apply_rope  # noqa: E402 — after class definition for readability

# Import KVCache for kv_cache type check inside SafeCausalSelfAttention
try:
    from litgpt.model import KVCache  # noqa: F401
except ImportError:
    KVCache = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# setup() — SGTM entry point
# ---------------------------------------------------------------------------

def setup(
    model_name: str,
    # SGTM: SafeMoEConfig instead of litgpt.Config
    model_config: Optional[SafeMoEConfig] = None,
    out_dir: Path = Path("out/pretrain"),
    precision: Literal["bf16-true", "bf16-mixed", "32-true", None] = None,
    initial_checkpoint_dir: Optional[Path] = None,
    resume: Union[bool, Literal["auto"], Path] = False,
    # SGTM: MultiDataLoader instead of DataModule
    data: Optional[MultiDataLoader] = None,
    train: TrainArgs = TrainArgs(
        save_interval=1000,
        log_interval=1,
        global_batch_size=512,
        micro_batch_size=4,
        max_tokens=int(3e12),  # 3 trillion
        max_norm=1.0,
        min_lr=4e-5,
        lr_warmup_steps=2000,
        tie_embeddings=False,
    ),
    eval: EvalArgs = EvalArgs(interval=1000, max_iters=100),
    log: LogArgs = LogArgs(),
    optimizer: Union[str, Dict] = "AdamW",
    devices: Union[int, str] = "auto",
    num_nodes: int = 1,
    tokenizer_dir: Optional[Path] = None,
    logger_name: LoggerChoice = "tensorboard",
    seed: int = 42,
    # SGTM: upsample weights for 3-path split sampling — required, no defaults
    upsample_std: Optional[float] = None,
    upsample_harmful: Optional[float] = None,
    upsample_unlabeled: Optional[float] = None,
):
    """Pretrain a SafeMoE model with SGTM dual-optimizer 3-path training loop.

    Arguments:
        model_name: The name of the model to pretrain. Choose from names in ``litgpt.config``. Use "list" to list the supported models.
        model_config: A ``SafeMoEConfig`` object to define the model architecture.
        out_dir: Directory in which to save checkpoints and logs.
        precision: The precision to use for training.
        initial_checkpoint_dir: Optional path to a checkpoint directory to initialize the model from.
        resume: Path to a checkpoint directory to resume from, or ``True`` to resume from the latest checkpoint.
        data: A ``MultiDataLoader`` providing D_std, D_harmful, D_unlabeled loaders.
        train: Training-related arguments. See ``litgpt.args.TrainArgs`` for details.
        eval: Evaluation-related arguments.
        optimizer: An optimizer name (such as "AdamW") or config.
        devices: How many devices/GPUs to use.
        num_nodes: How many nodes the code is being run on.
        tokenizer_dir: Optional path to the tokenizer dir.
        logger_name: The name of the logger to send metrics to.
        seed: The random seed to use for reproducibility.
        upsample_std: Sampling weight for D_std split. Required.
        upsample_harmful: Sampling weight for D_harmful split. Required.
        upsample_unlabeled: Sampling weight for D_unlabeled split. Required.
    """
    # SGTM: validate required upsample weights — no silent defaults
    if any(w is None for w in [upsample_std, upsample_harmful, upsample_unlabeled]):
        raise ValueError(
            "upsample_std/harmful/unlabeled are required fields — no defaults"
        )

    if model_name == "list":
        available_models = "\n".join(sorted(name_to_config))
        print(f"Available values:\n{available_models}")
        quit()

    if initial_checkpoint_dir is not None:
        initial_checkpoint_dir = extend_checkpoint_dir(initial_checkpoint_dir)

    if tokenizer_dir is not None:
        tokenizer_dir = extend_checkpoint_dir(tokenizer_dir)

    if model_config is None:
        try:
            model_config = Config.from_name(model_name)
        except ValueError:
            print(f"Model name {model_name} is not supported.\n")
            available_models = "\n".join(sorted(name_to_config))
            print(f"Available values:\n{available_models}")
            quit()

    hparams = capture_hparams()
    if data is None:
        raise ValueError("data (MultiDataLoader) is required for SGTM training")

    config = model_config
    precision = precision or get_default_supported_precision(training=True)
    devices = parse_devices(devices)
    out_dir = init_out_dir(out_dir)
    tokenizer = Tokenizer(tokenizer_dir) if tokenizer_dir is not None else None

    logger = choose_logger(
        logger_name,
        out_dir,
        name=f"pretrain-{config.name}",
        resume=bool(resume),
        log_interval=train.log_interval,
        log_args=asdict(log),
    )

    if devices * num_nodes > 1:
        strategy = FSDPStrategy(auto_wrap_policy={Block}, state_dict_type="full", sharding_strategy="HYBRID_SHARD")
    else:
        strategy = "auto"

    fabric = L.Fabric(devices=devices, num_nodes=num_nodes, strategy=strategy, precision=precision, loggers=[logger])

    if torch.cuda.is_available() and devices > 1:
        check_nvlink_connectivity(fabric)

    fabric.launch()

    fabric.print(pprint.pformat(hparams))
    if logger_name in ("tensorboard", "wandb", "mlflow"):
        fabric.logger.log_hyperparams(hparams)

    main(
        fabric=fabric,
        devices=devices,
        num_nodes=num_nodes,
        seed=seed,
        initial_checkpoint_dir=initial_checkpoint_dir,
        resume=resume,
        config=config,
        data=data,
        out_dir=out_dir,
        tokenizer_dir=tokenizer_dir,
        tokenizer=tokenizer,
        train=train,
        eval=eval,
        optimizer=optimizer,
        upsample_std=upsample_std,
        upsample_harmful=upsample_harmful,
        upsample_unlabeled=upsample_unlabeled,
    )


# ---------------------------------------------------------------------------
# main() — SGTM: dual optimizer + HarmfulParamRegistry + masker instantiation
# ---------------------------------------------------------------------------

def main(
    fabric: L.Fabric,
    devices: int,
    seed: int,
    initial_checkpoint_dir: Optional[Path],
    resume: Union[bool, Literal["auto"], Path],
    config: SafeMoEConfig,
    data: MultiDataLoader,
    out_dir: Path,
    tokenizer_dir: Optional[Path],
    tokenizer: Optional[Tokenizer],
    train: TrainArgs,
    eval: EvalArgs,
    optimizer: Union[str, Dict],
    num_nodes: int = 1,
    # SGTM: upsample weights passed from setup()
    upsample_std: float = 1.0,
    upsample_harmful: float = 1.0,
    upsample_unlabeled: float = 1.0,
) -> None:
    validate_args(train, eval, initial_checkpoint_dir, resume)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    fabric.seed_everything(seed)  # same seed for every process to init model (FSDP)

    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        model = GPT(config)

    initialize_weights(fabric, model, n_layer=config.n_layer, n_embd=config.n_embd)

    if train.tie_embeddings:
        model.transformer.wte.weight = model.lm_head.weight
    if train.max_seq_length:
        model.max_seq_length = train.max_seq_length

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters: {num_parameters(model):,}")

    # SGTM: replace CausalSelfAttention with SafeCausalSelfAttention for head masking support
    # Must happen BEFORE ActivationMasker construction so model.modules() scan finds the
    # SafeCausalSelfAttention instances (which inherit from CausalSelfAttention).
    for name, module in list(model.named_modules()):
        if type(module) is CausalSelfAttention:
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            attr = name.rsplit(".", 1)[-1]
            safe_attn = SafeCausalSelfAttention.__new__(SafeCausalSelfAttention)
            safe_attn.__dict__.update(module.__dict__)
            setattr(parent, attr, safe_attn)

    # SGTM: REMOVED torch.compile(model) — incompatible with Python bool flag checks
    # (per RESEARCH.md anti-pattern: torch.compile traces through Python booleans, making
    # dynamic _activation_masking_enabled flag checks ineffective)
    model = fabric.setup(model)

    # SGTM: HarmfulParamRegistry must be built AFTER fabric.setup(model) so it holds
    # real (materialized) tensors, not meta-device tensors — Lightning requires optimizer
    # params to be real tensors at setup_optimizers() time.
    # Use model.module (public API, strips DDP/FSDP wrappers) to get original parameter
    # names without any "_forward_module." or "module." prefix that breaks the regex patterns.
    registry = HarmfulParamRegistry(model.module, config)

    # SGTM: dual optimizer setup (Pattern 3 from RESEARCH.md)
    extra_kwargs = {"fused": fabric.device.type == "cuda"}
    optimizer_harmful = instantiate_torch_optimizer(
        optimizer, registry.parameters_by_type("theta_harmful"), **extra_kwargs
    )
    optimizer_std = instantiate_torch_optimizer(
        optimizer, registry.parameters_by_type("theta_std"), **extra_kwargs
    )
    optimizer_harmful, optimizer_std = fabric.setup_optimizers(optimizer_harmful, optimizer_std)

    train_dataloader, val_dataloader = get_dataloaders(fabric, data, tokenizer, train, model.max_seq_length)
    # SGTM: only val_dataloader is fabric-wrapped; split loaders are accessed via data.get_loader()
    val_dataloader = fabric.setup_dataloaders(val_dataloader)

    if initial_checkpoint_dir:
        fabric.load_raw(initial_checkpoint_dir / "lit_model.pth", model)

    # SGTM: state dict with dual optimizer keys (replaces single "optimizer" key)
    state = {
        "model": model,
        "optimizer_harmful": optimizer_harmful,
        "optimizer_std": optimizer_std,
        "iter_num": 0,
        "step_count": 0,
        "split_label": "D_std",
    }

    resume = find_resume_path(resume, out_dir)
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    # SGTM: Pitfall 3 fix — restore random state for split sampling reproducibility on resume
    random.seed(seed + state["iter_num"])

    # SGTM: instantiate maskers using model.module (public API, strips DDP/FSDP wrappers)
    # so modules() scan finds SafeCausalSelfAttention without any DDP "module." prefix
    gradient_masker = GradientMasker(registry)
    activation_masker = ActivationMasker(model.module, registry=registry, config=config)

    train_time = time.perf_counter()

    # work around PyTorch issue https://github.com/pytorch/pytorch/issues/152162
    if (
        (_TORCH_EQUAL_2_7 or _TORCH_EQUAL_2_8)
        and (model._forward_module.__class__.__name__ == "OptimizedModule")
        and (model._forward_module._orig_mod.__class__.__name__ == "FullyShardedDataParallel")
    ):
        from torch.distributed.fsdp._runtime_utils import _root_pre_forward

        _root_pre_forward(model._forward_module._orig_mod, model._forward_module._orig_mod, [], {})

    fit(
        fabric=fabric,
        devices=devices,
        num_nodes=num_nodes,
        state=state,
        data=data,
        val_dataloader=val_dataloader,
        out_dir=out_dir,
        tokenizer_dir=tokenizer_dir,
        train=train,
        eval=eval,
        gradient_masker=gradient_masker,
        activation_masker=activation_masker,
        upsample_std=upsample_std,
        upsample_harmful=upsample_harmful,
        upsample_unlabeled=upsample_unlabeled,
    )

    # Save final checkpoint
    save_checkpoint(fabric, state, tokenizer_dir, out_dir / "final" / "lit_model.pth")

    total_tokens = state["iter_num"] * train.micro_batch_size * model.max_seq_length * fabric.world_size

    separator = "-" * 40
    fabric.print(separator)
    fabric.print("| Performance")
    fabric.print(f"| - Total tokens  : {total_tokens:,}")
    fabric.print(f"| - Training Time : {(time.perf_counter() - train_time):.2f} s")
    fabric.print(f"| - Tok/sec       : {total_tokens / train_time:.2f} tok/s")
    fabric.print("| " + "-" * 40)

    if fabric.device.type == "cuda":
        memory_used = torch.cuda.max_memory_allocated() / 1e9
        fabric.print("| Memory Usage")
        fabric.print(f"| - Memory Used   : {memory_used:.2f} GB")
    fabric.print(separator)


# ---------------------------------------------------------------------------
# fit() — SGTM: 3-path accumulation loop with split iterators
# ---------------------------------------------------------------------------

def fit(
    fabric: L.Fabric,
    devices: int,
    state: dict,
    # SGTM: MultiDataLoader replaces single train_dataloader
    data: MultiDataLoader,
    val_dataloader: DataLoader,
    out_dir: Path,
    tokenizer_dir: Optional[Path],
    train: TrainArgs,
    eval: EvalArgs,
    num_nodes: int = 1,
    # SGTM: maskers for 3-path gradient and activation isolation
    gradient_masker: Optional[GradientMasker] = None,
    activation_masker: Optional[ActivationMasker] = None,
    # SGTM: upsample weights for split sampling
    upsample_std: float = 1.0,
    upsample_harmful: float = 1.0,
    upsample_unlabeled: float = 1.0,
) -> None:
    model = state["model"]
    # SGTM: dual optimizers from state dict
    optimizer_harmful = state["optimizer_harmful"]
    optimizer_std = state["optimizer_std"]

    if eval.initial_validation:
        val_loss = validate(fabric, model, val_dataloader, max_iters=eval.max_iters)
        val_loss = f"{val_loss:.3f}"
    else:
        fabric.print("Verifying settings ...")
        validate(fabric, model, val_dataloader, max_iters=2, verbose=False)  # sanity check
        val_loss = "n/a"

    throughput = ThroughputMonitor(fabric, window_size=5)

    try:
        with torch.device("meta"):
            meta_model = GPT(model.config)
            x = torch.randint(0, 1, (train.micro_batch_size, meta_model.max_seq_length))
            model_fwd = lambda: meta_model(x)  # noqa: F821
            model_loss = lambda y: chunked_cross_entropy(y, x, chunk_size=0)  # noqa: F821
            measured_flops = measure_flops(meta_model, model_fwd, model_loss)
            fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
            del meta_model, x
    except (NotImplementedError, RuntimeError):
        # MoE models use torch.where() which is not supported on meta device.
        # Fall back to zero so ThroughputMonitor still works (it uses flops if non-zero).
        measured_flops = 0
        fabric.print("FLOP measurement skipped (meta device does not support MoE ops)")

    max_tokens_per_device = train.max_tokens // fabric.world_size
    tokens_per_iter = train.micro_batch_size * model.max_seq_length
    max_iters = max_tokens_per_device // tokens_per_iter
    accum_iters = train.gradient_accumulation_iters(devices, num_nodes)
    log_iter_interval = train.log_interval * accum_iters
    initial_iter = state["iter_num"]

    # SGTM: three split iterators replace single CycleIterator (Pattern 1)
    split_iters = {label: CycleIterator(data.get_loader(label)) for label in SPLIT_LABELS}
    # SGTM: upsample weights for weighted split sampling
    weights = [upsample_std, upsample_harmful, upsample_unlabeled]

    running_loss = RunningMean(window=accum_iters, sync_on_compute=False).to(fabric.device)
    fabric.barrier()
    total_t0 = time.perf_counter()

    # SGTM: warmup_iters uses D_std loader length as proxy for train_dataloader length
    warmup_iters = train.warmup_iters(devices, num_nodes, max_iters, data.get_loader("D_std"))

    # SGTM: iter_num counts micro-batches; step_count counts optimizer steps
    # The outer while loop drives one OPTIMIZER STEP per iteration (accum_iters micro-batches)
    while state["iter_num"] < max_iters:
        iter_t0 = time.perf_counter()

        # SGTM: sample split label once per optimizer step (Pattern 1)
        split_label = random.choices(SPLIT_LABELS, weights=weights, k=1)[0]
        state["split_label"] = split_label

        # SGTM: dual LR update — same cosine schedule for both optimizers (Pattern 4)
        base_lr = optimizer_std.defaults["lr"]
        lr = get_lr(base_lr, state["iter_num"], warmup_iters, max_iters, train.min_lr)
        for opt in (optimizer_harmful, optimizer_std):
            for pg in opt.param_groups:
                pg["lr"] = lr

        # SGTM: 3-path masking template (Pattern 2)
        # D_std: enable activation masker before the micro-batch window; disable in finally
        if split_label == "D_std":
            activation_masker.enable()

        try:
            # SGTM: inner accumulation loop — accum_iters micro-batches per optimizer step
            for micro_batch_idx in range(accum_iters):
                state["iter_num"] += 1
                is_accumulating = micro_batch_idx < accum_iters - 1

                train_data = next(split_iters[split_label])
                input_ids = train_data[:, 0 : model.max_seq_length].contiguous().long()
                targets = train_data[:, 1 : (model.max_seq_length + 1)].contiguous().long()

                with fabric.no_backward_sync(model, enabled=is_accumulating):
                    logits = model(input_ids)
                    loss = chunked_cross_entropy(logits, targets) / accum_iters
                    fabric.backward(loss)

                running_loss.update(loss.detach() * accum_iters)  # un-scale for reporting

        finally:
            # SGTM: always disable activation masker (try/finally avoids masker stuck True on error)
            if split_label == "D_std":
                activation_masker.disable()

        # SGTM: per-split optimizer step (Pattern 2)
        if split_label == "D_std":
            fabric.clip_gradients(model, optimizer_std, max_norm=train.max_norm)
            optimizer_std.step()
            optimizer_std.zero_grad(set_to_none=True)
            # Keep theta_harmful grads cleared for isolation
            optimizer_harmful.zero_grad(set_to_none=True)
        elif split_label == "D_harmful":
            # SGTM: gradient_masker.mask() called ONCE per optimizer step (not per micro-batch)
            gradient_masker.mask()
            fabric.clip_gradients(model, optimizer_harmful, max_norm=train.max_norm)
            optimizer_harmful.step()
            optimizer_harmful.zero_grad(set_to_none=True)
            optimizer_std.zero_grad(set_to_none=True)
        else:  # D_unlabeled
            # SGTM: no masking — both optimizers step
            fabric.clip_gradients(model, optimizer_harmful, max_norm=train.max_norm)
            fabric.clip_gradients(model, optimizer_std, max_norm=train.max_norm)
            optimizer_harmful.step()
            optimizer_std.step()
            optimizer_harmful.zero_grad(set_to_none=True)
            optimizer_std.zero_grad(set_to_none=True)

        state["step_count"] += 1

        if state["iter_num"] % log_iter_interval == 0:
            loss_val = running_loss.compute().item()
            t1 = time.perf_counter()
            throughput.update(
                time=(t1 - total_t0),
                flops=(measured_flops * log_iter_interval),
                batches=state["iter_num"],
                samples=(state["iter_num"] * train.micro_batch_size),
                lengths=(state["iter_num"] * train.micro_batch_size * model.max_seq_length),
            )
            # SGTM: per-split loss logging
            loss_key = {
                "D_std": "loss_D_std",
                "D_harmful": "loss_D_harmful",
                "D_unlabeled": "loss_D_unlabeled",
            }[split_label]
            metrics = {
                "loss": loss_val,
                loss_key: loss_val,
                "iter": state["iter_num"],
                "step": state["step_count"],
                "split_label": split_label,
                "iter_time": t1 - iter_t0,
                "remaining_time": (
                    (t1 - total_t0) / (state["iter_num"] - initial_iter) * (max_iters - state["iter_num"])
                    if state["iter_num"] > initial_iter else 0
                ),
                "tokens": state["iter_num"] * train.micro_batch_size * model.max_seq_length,
                "total_tokens": (state["iter_num"] * train.micro_batch_size * model.max_seq_length * fabric.world_size),
                "learning_rate": lr,
            }
            if isinstance(val_loss, float):
                val_loss = f"{val_loss:.3f}"
            fabric.print(
                f"step {state['step_count']} [{split_label}] | iter {metrics['iter']} |"
                f" loss: {metrics['loss']:.3f},"
                f" val: {val_loss} |"
                f" iter time: {metrics['iter_time'] * 1000:.2f} ms"
                f" remaining time: {timedelta(seconds=int(metrics['remaining_time']))!s}"
            )

            throughput_metrics = throughput.compute()
            metrics.update(throughput_metrics)
            fabric.log_dict(metrics, step=state["iter_num"] - 1)

        if val_dataloader is not None and state["step_count"] % eval.interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader, max_iters=eval.max_iters)
            val_loss = val_loss.item()
            td = time.perf_counter() - t0

            fabric.print(f"iter {state['iter_num']}: val loss {val_loss:.4f}, val time: {td * 1000:.2f} ms")
            metrics = {"val_loss": val_loss, "val_ppl": math.exp(val_loss)}
            fabric.log_dict(metrics, step=state["iter_num"] - 1)
            fabric.barrier()

        if train.save_interval is not None and state["step_count"] % train.save_interval == 0:
            save_checkpoint(fabric, state, tokenizer_dir, out_dir / f"step-{state['step_count']:08d}" / "lit_model.pth")

    # Final validation
    if eval.final_validation:
        val_loss = validate(fabric, model, val_dataloader, max_iters=eval.max_iters)
        metrics = {"val_loss": val_loss, "val_ppl": math.exp(val_loss)}
        fabric.log_dict(metrics, step=state["iter_num"])
        fabric.print(f"Final evaluation | val loss: {val_loss.item():.3f} | val ppl: {math.exp(val_loss):.3f}")


# ---------------------------------------------------------------------------
# validate() — reused from litgpt unchanged (no masking during validation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    fabric: L.Fabric, model: nn.Module, val_dataloader: DataLoader, max_iters: int, verbose: bool = True
) -> torch.Tensor:
    fabric.barrier()
    if verbose:
        fabric.print("Validating ...")
    model.eval()

    losses = []
    for k, batch in enumerate(val_dataloader):
        if k >= max_iters:
            break
        input_ids = batch[:, 0 : model.max_seq_length].contiguous().long()
        targets = batch[:, 1 : (model.max_seq_length + 1)].contiguous().long()
        logits = model(input_ids)
        loss = chunked_cross_entropy(logits, targets)
        losses.append(loss)

    val_loss = torch.stack(losses).mean()
    model.train()
    fabric.barrier()
    return val_loss


# ---------------------------------------------------------------------------
# get_dataloaders() — SGTM: only sets up data module, returns val DataLoader
# ---------------------------------------------------------------------------

def get_dataloaders(
    fabric: L.Fabric,
    data: MultiDataLoader,
    tokenizer: Optional[Tokenizer],
    train: TrainArgs,
    block_size: int,
) -> Tuple[None, DataLoader]:
    """Set up data module and return (None, val_dataloader).

    The training split loaders are accessed via ``data.get_loader(split_label)``
    directly in fit() — not returned here.  Returns None for the train slot to
    signal that the caller should use ``data.get_loader()``.
    """
    data.connect(tokenizer=tokenizer, batch_size=train.micro_batch_size, max_seq_length=block_size)
    with fabric.rank_zero_first():
        data.prepare_data()
    data.setup()
    val_dataloader = data.val_dataloader()
    # val_dataloader() returns a list; use first entry (D_std val) for scalar val_loss
    if isinstance(val_dataloader, list):
        val_dataloader = val_dataloader[0]
    return None, val_dataloader


# ---------------------------------------------------------------------------
# get_lr() — cosine LR with linear warmup (reused from litgpt)
# ---------------------------------------------------------------------------

def get_lr(learning_rate: float, it: int, warmup_iters: int, max_iters: int, min_lr: float) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > max_iters, return min learning rate
    if it > max_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# ---------------------------------------------------------------------------
# initialize_weights() — GPT-NeoX weight initialization (reused from litgpt)
# ---------------------------------------------------------------------------

def initialize_weights(fabric: L.Fabric, model: GPT, n_layer: int, n_embd: int) -> None:
    """GPT-NeoX weight initialization (https://arxiv.org/abs/2204.06745)."""

    def init_weights(module, std):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)

    for mod in model.modules():
        if isinstance(mod, (nn.Embedding, nn.Linear)):
            mod.reset_parameters = partial(init_weights, mod, std=math.sqrt(2.0 / 5 / n_embd))

    for mod in model.modules():
        if isinstance(mod, (LLaMAMLP, CausalSelfAttention)):
            mod.proj.reset_parameters = partial(init_weights, mod.proj, std=(1 / math.sqrt(n_embd) / n_layer))

    if not isinstance(fabric.strategy, FSDPStrategy):
        reset_parameters(model)


# ---------------------------------------------------------------------------
# save_checkpoint() — reused from litgpt
# ---------------------------------------------------------------------------

def save_checkpoint(fabric, state, tokenizer_dir, checkpoint_file):
    model = state["model"]
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    fabric.print(f"Saving checkpoint to {str(checkpoint_file)!r}")
    fabric.save(checkpoint_file, state)
    if fabric.global_rank == 0:
        save_hyperparameters(setup, checkpoint_file.parent)
        if tokenizer_dir is not None:
            copy_config_files(tokenizer_dir, checkpoint_file.parent)
        save_config(model.config, checkpoint_file.parent)


# ---------------------------------------------------------------------------
# validate_args() — reused from litgpt
# ---------------------------------------------------------------------------

def validate_args(train: TrainArgs, eval: EvalArgs, initial_checkpoint_dir, resume) -> None:
    issues = []
    unsupported = [(train, ["epochs"]), (eval, ["max_new_tokens"])]
    for args, names in unsupported:
        for name in names:
            if getattr(args, name) is not None:
                issues.append(f"{__file__} doesn't support the {name!r} argument. This is set in {args}")
    if train.max_steps is not None:
        warnings.warn(
            "`train.max_steps` is intended for profiling or debug runs only. "
            "For full pretraining runs, prefer `train.max_tokens` or `train.max_time`.",
            UserWarning,
        )
    required = [(train, ["max_tokens", "max_norm"])]
    for args, names in required:
        for name in names:
            if getattr(args, name) is None:
                issues.append(f"{__file__} requires the {name!r} argument. This is set in {args}")
    if initial_checkpoint_dir and resume:
        issues.append("Can't provide both `--resume` and `--initial_checkpoint_dir`. Choose one.")
    if issues:
        raise ValueError("\n".join(issues))
