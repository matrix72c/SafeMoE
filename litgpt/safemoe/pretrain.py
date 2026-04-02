# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
# SGTM: forked from litgpt/pretrain.py — adds single-optimizer 3-path SGTM training loop.

import math
import pprint
import random  # SGTM: split sampling via random.choices
import shlex
import sys
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Union

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.throughput import ThroughputMonitor, measure_flops
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from torchmetrics.aggregation import RunningMean
from typing_extensions import Literal

from litgpt import Tokenizer
from litgpt.args import EvalArgs, LogArgs, TrainArgs
from litgpt.config import name_to_config
from litgpt.constants import _TORCH_EQUAL_2_7, _TORCH_EQUAL_2_8

# SGTM: safemoe-specific imports for single-optimizer masking infrastructure
from litgpt.data import SafeData
from litgpt.model import GPT, Block, CausalSelfAttention, Config, LLaMAMLP, SafeMoELayer
from litgpt.parser_config import save_hyperparameters
from litgpt.safemoe.masking import ActivationMasker, GradientMasker, HarmfulParamRegistry
from litgpt.safemoe.surgery import setup as surgery_setup
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

# SGTM: split labels for 3-path SGTM branching
SPLIT_LABELS = ["D_std", "D_harmful", "D_unlabeled"]


class WarmupSurgeryArgs(NamedTuple):
    base_checkpoint: Path
    num_harmful_experts: int
    num_harmful_attn_heads: int
    seed: int
    epsilon: float


def resolve_out_dir(out_dir: Optional[Path], run_name: Optional[str] = None) -> Path:
    if out_dir is not None:
        return init_out_dir(out_dir)
    if run_name:
        return init_out_dir(Path("checkpoints") / run_name)
    config_path = _config_path_from_argv(sys.argv[1:])
    if config_path is not None:
        return init_out_dir(Path("checkpoints") / config_path.stem)
    return init_out_dir(Path("out/pretrain"))


def _config_path_from_argv(argv: List[str]) -> Optional[Path]:
    expanded_args: list[str] = []
    for arg in argv:
        if arg.startswith("@") and len(arg) > 1:
            expanded_args.extend(shlex.split(Path(arg[1:]).read_text()))
        else:
            expanded_args.append(arg)

    for index, arg in enumerate(expanded_args):
        if arg in {"--config", "-c"} and index + 1 < len(expanded_args):
            return Path(expanded_args[index + 1])
        if arg.startswith("--config="):
            return Path(arg.split("=", 1)[1])
        if arg.startswith("-c="):
            return Path(arg.split("=", 1)[1])
    return None


def maybe_prepare_warmup_checkpoint(
    *,
    stage: Literal["transfer", "warmup"],
    initial_checkpoint_dir: Optional[Path],
    base_checkpoint: Optional[Path],
    num_harmful_experts: Optional[int],
    num_harmful_attn_heads: Optional[int],
    seed: int,
    epsilon: Optional[float],
) -> Optional[Path]:
    if stage != "warmup":
        if any(value is not None for value in (base_checkpoint, num_harmful_experts, num_harmful_attn_heads, epsilon)):
            raise ValueError("Warmup auto-surgery args can only be used when `stage=warmup`.")
        return initial_checkpoint_dir

    surgery_args = _build_warmup_surgery_args(
        initial_checkpoint_dir=initial_checkpoint_dir,
        base_checkpoint=base_checkpoint,
        num_harmful_experts=num_harmful_experts,
        num_harmful_attn_heads=num_harmful_attn_heads,
        seed=seed,
        epsilon=epsilon,
    )
    if surgery_args is None:
        return initial_checkpoint_dir
    return surgery_setup(
        base_checkpoint=surgery_args.base_checkpoint,
        num_harmful_experts=surgery_args.num_harmful_experts,
        num_harmful_attn_heads=surgery_args.num_harmful_attn_heads,
        seed=surgery_args.seed,
        epsilon=surgery_args.epsilon,
    )


def _build_warmup_surgery_args(
    *,
    initial_checkpoint_dir: Optional[Path],
    base_checkpoint: Optional[Path],
    num_harmful_experts: Optional[int],
    num_harmful_attn_heads: Optional[int],
    seed: int,
    epsilon: Optional[float],
) -> Optional[WarmupSurgeryArgs]:
    fields = {
        "base_checkpoint": base_checkpoint,
        "num_harmful_experts": num_harmful_experts,
        "num_harmful_attn_heads": num_harmful_attn_heads,
        "epsilon": epsilon,
    }
    if not any(value is not None for value in fields.values()):
        return None
    missing_fields = [name for name, value in fields.items() if value is None]
    if missing_fields:
        raise ValueError(
            "Warmup auto-surgery requires all of "
            "`base_checkpoint`, `num_harmful_experts`, `num_harmful_attn_heads`, and `epsilon`."
        )
    if initial_checkpoint_dir is not None:
        raise ValueError(
            "Can't provide both `--initial_checkpoint_dir` and warmup auto-surgery args. "
            "Use the surgery source checkpoint via `--base_checkpoint` only."
        )
    return WarmupSurgeryArgs(
        base_checkpoint=Path(base_checkpoint),
        num_harmful_experts=int(num_harmful_experts),
        num_harmful_attn_heads=int(num_harmful_attn_heads),
        seed=seed,
        epsilon=float(epsilon),
    )


def warmup_routing_loss(
    harmful_mass: torch.Tensor,
    split_label: str,
    harmful_mass_floor: float = 0.6,
    std_mass_ceiling: float = 0.4,
) -> torch.Tensor:
    if split_label == "D_harmful":
        return F.softplus(harmful_mass.new_tensor(harmful_mass_floor) - harmful_mass)
    if split_label == "D_std":
        return F.softplus(harmful_mass - harmful_mass.new_tensor(std_mass_ceiling))
    return harmful_mass.new_zeros(())



def collect_warmup_routing_mass(model: nn.Module) -> torch.Tensor:
    masses = [
        layer._last_harmful_routing_mass
        for layer in model.modules()
        if isinstance(layer, SafeMoELayer) and layer._last_harmful_routing_mass is not None
    ]
    if not masses:
        return next(model.parameters()).new_zeros(())
    return torch.stack(masses).mean()


@dataclass(frozen=True)
class SplitValidationResult:
    loss: torch.Tensor
    ppl: float
    dispatch_count: int
    harmful_dispatches: int
    routing_harmful_frac: float
    batches_evaluated: int


@dataclass(frozen=True)
class ValidationSummary:
    by_split: dict[str, SplitValidationResult]
    scalar_metrics: dict[str, float]
    routing_margin: Optional[float]


def _unwrap_safemoe_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "module"):
        return model.module
    return model


def _resolve_registry_model_view(model: nn.Module) -> nn.Module:
    raw_model = getattr(model, "_forward_module", model)
    if hasattr(raw_model, "module") and not isinstance(raw_model, FSDP):
        return raw_model.module
    return raw_model


def _resolve_module_traversal_view(model: nn.Module) -> nn.Module:
    return getattr(model, "module", _unwrap_safemoe_model(model))


def _setup_model(fabric: L.Fabric, config: Config, train: TrainArgs) -> GPT:
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
    return fabric.setup(model)


def _build_registry(model: nn.Module, config: Config) -> HarmfulParamRegistry:
    return HarmfulParamRegistry(_resolve_registry_model_view(model), config)


def _build_optimizer(
    fabric: L.Fabric,
    optimizer_config: Union[str, Dict],
    registry: HarmfulParamRegistry,
):
    extra_kwargs = {"fused": fabric.device.type == "cuda"}
    optimizer = instantiate_torch_optimizer(
        optimizer_config,
        registry.parameters_by_type("theta_harmful")
        + registry.parameters_by_type("theta_std")
        + registry.parameters_by_type("theta_shared"),
        **extra_kwargs,
    )
    return fabric.setup_optimizers(optimizer)


def _build_maskers(
    model: nn.Module,
    registry: HarmfulParamRegistry,
    config: Config,
) -> tuple[GradientMasker, ActivationMasker]:
    module_view = _resolve_module_traversal_view(model)
    return GradientMasker(registry), ActivationMasker(module_view)


def _iter_safemoe_layers(model: nn.Module):
    yield from (
        module
        for module in _unwrap_safemoe_model(model).modules()
        if isinstance(module, SafeMoELayer)
    )


def _collect_cached_routing_counts(
    model: nn.Module,
    harmful_expert_indices: list[int],
) -> tuple[int, int]:
    harmful_index_set = set(harmful_expert_indices)
    total_dispatches = 0
    harmful_dispatches = 0
    for layer in _iter_safemoe_layers(model):
        indices = layer._last_indices
        if indices is None:
            continue
        flat_indices = indices.detach().reshape(-1)
        total_dispatches += int(flat_indices.numel())
        if harmful_index_set:
            harmful_dispatches += sum(
                int(expert_idx in harmful_index_set)
                for expert_idx in flat_indices.tolist()
            )
    return total_dispatches, harmful_dispatches


def _reduce_routing_counts(
    fabric: L.Fabric,
    total_dispatches: int,
    harmful_dispatches: int,
) -> tuple[int, int]:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return total_dispatches, harmful_dispatches
    dispatch_tensor = torch.tensor(total_dispatches, device=fabric.device, dtype=torch.long)
    harmful_tensor = torch.tensor(harmful_dispatches, device=fabric.device, dtype=torch.long)
    torch.distributed.all_reduce(dispatch_tensor)
    torch.distributed.all_reduce(harmful_tensor)
    return int(dispatch_tensor.item()), int(harmful_tensor.item())


@torch.no_grad()
def _evaluate_validation_split(
    fabric: L.Fabric,
    model: nn.Module,
    val_dataloader: DataLoader,
    split_name: str,
    max_iters: int,
    verbose: bool = True,
) -> SplitValidationResult:
    fabric.barrier()
    if verbose:
        fabric.print(f"Validating {split_name} ...")
    was_training = model.training
    model.eval()

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    total_loss = 0.0
    count = 0
    total_dispatches = 0
    harmful_dispatches = 0
    unwrapped_model = _unwrap_safemoe_model(model)
    harmful_expert_indices = list(getattr(unwrapped_model.config, "harmful_expert_indices", []))
    try:
        for k, batch in enumerate(val_dataloader):
            if k >= max_iters:
                break
            input_ids = batch[:, 0 : model.max_seq_length].contiguous().long().to(fabric.device)
            targets = batch[:, 1 : (model.max_seq_length + 1)].contiguous().long().to(fabric.device)
            logits = model(input_ids)
            batch_total_dispatches, batch_harmful_dispatches = _collect_cached_routing_counts(
                model,
                harmful_expert_indices,
            )
            total_dispatches += batch_total_dispatches
            harmful_dispatches += batch_harmful_dispatches
            loss = chunked_cross_entropy(logits, targets)
            total_loss += loss.item()
            count += 1
    finally:
        torch.set_default_dtype(prev_dtype)
        model.train(was_training)
        fabric.barrier()

    val_loss = torch.tensor(total_loss / max(count, 1), device=fabric.device)
    total_dispatches, harmful_dispatches = _reduce_routing_counts(
        fabric,
        total_dispatches=total_dispatches,
        harmful_dispatches=harmful_dispatches,
    )
    routing_harmful_frac = harmful_dispatches / max(total_dispatches, 1)
    return SplitValidationResult(
        loss=val_loss,
        ppl=math.exp(val_loss.item()),
        dispatch_count=total_dispatches,
        harmful_dispatches=harmful_dispatches,
        routing_harmful_frac=routing_harmful_frac,
        batches_evaluated=count,
    )


@torch.no_grad()
def collect_validation_summary(
    fabric: L.Fabric,
    model: nn.Module,
    val_loaders: dict[str, DataLoader],
    max_iters: int,
    verbose: bool = True,
    metric_prefix: str = "",
) -> ValidationSummary:
    by_split: dict[str, SplitValidationResult] = {}
    scalar_metrics: dict[str, float] = {}
    for split_name, loader in val_loaders.items():
        split_result = _evaluate_validation_split(
            fabric=fabric,
            model=model,
            val_dataloader=loader,
            split_name=split_name,
            max_iters=max_iters,
            verbose=verbose,
        )
        by_split[split_name] = split_result
        scalar_metrics[f"{metric_prefix}val_loss_{split_name}"] = split_result.loss.item()
        scalar_metrics[f"{metric_prefix}val_ppl_{split_name}"] = split_result.ppl
        scalar_metrics[f"{metric_prefix}dispatch_count_{split_name}"] = split_result.dispatch_count
        scalar_metrics[f"{metric_prefix}routing_harmful_frac_{split_name}"] = split_result.routing_harmful_frac

    routing_margin = None
    if "D_std" in by_split and "D_harmful" in by_split:
        routing_margin = by_split["D_harmful"].routing_harmful_frac - by_split["D_std"].routing_harmful_frac
        scalar_metrics[f"{metric_prefix}routing_margin"] = routing_margin

    return ValidationSummary(by_split=by_split, scalar_metrics=scalar_metrics, routing_margin=routing_margin)


def _format_validation_status(summary: ValidationSummary) -> str:
    if not summary.by_split:
        return "n/a"
    return ", ".join(
        f"{split_name}={split_result.loss.item():.3f}"
        for split_name, split_result in summary.by_split.items()
    )


def ensure_config(config: Union[Config, Config]) -> Config:
    if isinstance(config, Config):
        return config
    return Config(**asdict(config))


def load_config_from_checkpoint(ckpt_dir: Path) -> Config:
    raw = yaml.safe_load((ckpt_dir / "model_config.yaml").read_text())
    return Config(**{key: value for key, value in raw.items() if not isinstance(value, dict)})


# ---------------------------------------------------------------------------
# setup() — SGTM entry point
# ---------------------------------------------------------------------------

def setup(
    model_name: str,
    # SGTM: Config instead of litgpt.Config
    model_config: Optional[Config] = None,
    out_dir: Optional[Path] = None,
    run_name: Optional[str] = None,
    precision: Literal["bf16-true", "bf16-mixed", "32-true", None] = None,
    initial_checkpoint_dir: Optional[Path] = None,
    resume: Union[bool, Literal["auto"], Path] = False,
    data: Optional[SafeData] = None,
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
    logger_name: LoggerChoice = "wandb",
    seed: int = 42,
    # SGTM: upsample weights for 3-path split sampling — required, no defaults
    upsample_std: Optional[float] = None,
    upsample_harmful: Optional[float] = None,
    upsample_unlabeled: Optional[float] = None,
    stage: Literal["transfer", "warmup"] = "transfer",
    warmup_routing_loss_weight: float = 0.1,
    warmup_harmful_mass_floor: float = 0.6,
    warmup_std_mass_ceiling: float = 0.4,
    base_checkpoint: Optional[Path] = None,
    num_harmful_experts: Optional[int] = None,
    num_harmful_attn_heads: Optional[int] = None,
    epsilon: Optional[float] = None,
):
    """Pretrain a SafeMoE model with SGTM single-optimizer 3-path training loop.

    Arguments:
        model_name: The name of the model to pretrain. Choose from names in ``litgpt.config``. Use "list" to list the supported models.
        model_config: A ``Config`` object to define the model architecture.
        out_dir: Directory in which to save checkpoints and logs. If omitted, derive from run_name or the CLI config filename.
        run_name: Optional experiment name used as the default checkpoint directory name and logger run name.
        precision: The precision to use for training.
        initial_checkpoint_dir: Optional path to a checkpoint directory to initialize the model from.
        resume: Path to a checkpoint directory to resume from, or ``True`` to resume from the latest checkpoint.
        data: A ``SafeData`` providing D_std, D_harmful, D_unlabeled loaders.
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
        base_checkpoint: Base checkpoint used to derive a warmup surgery checkpoint.
        num_harmful_experts: Count of harmful experts for warmup auto-surgery.
        num_harmful_attn_heads: Count of harmful attention heads for warmup auto-surgery.
        epsilon: Noise scale for warmup auto-surgery.
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
    if base_checkpoint is not None:
        base_checkpoint = extend_checkpoint_dir(base_checkpoint)

    initial_checkpoint_dir = maybe_prepare_warmup_checkpoint(
        stage=stage,
        initial_checkpoint_dir=initial_checkpoint_dir,
        base_checkpoint=base_checkpoint,
        num_harmful_experts=num_harmful_experts,
        num_harmful_attn_heads=num_harmful_attn_heads,
        seed=seed,
        epsilon=epsilon,
    )

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
        raise ValueError("data (SafeData) is required for SGTM training")

    config = ensure_config(model_config)
    if initial_checkpoint_dir is not None:
        config = load_config_from_checkpoint(initial_checkpoint_dir)
    precision = precision or get_default_supported_precision(training=True)
    devices = parse_devices(devices)
    out_dir = resolve_out_dir(out_dir, run_name=run_name)
    tokenizer = Tokenizer(tokenizer_dir) if tokenizer_dir is not None else None

    log_args = asdict(log)
    if run_name and not log_args.get("run"):
        log_args["run"] = run_name

    logger = choose_logger(
        logger_name,
        out_dir,
        name=f"pretrain-{config.name}",
        resume=bool(resume),
        log_interval=train.log_interval,
        log_args=log_args,
    )

    data_max_seq_length = train.max_seq_length or config.block_size

    if devices * num_nodes > 1:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
            sharding_strategy="HYBRID_SHARD",
        )
    else:
        strategy = "auto"

    fabric = L.Fabric(devices=devices, num_nodes=num_nodes, strategy=strategy, precision=precision, loggers=[logger])

    if torch.cuda.is_available() and devices > 1:
        check_nvlink_connectivity(fabric)

    fabric.launch()

    data.connect(tokenizer=tokenizer, batch_size=train.micro_batch_size, max_seq_length=data_max_seq_length)
    with fabric.rank_zero_first():
        data.prepare_data()

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
        stage=stage,
        warmup_routing_loss_weight=warmup_routing_loss_weight,
        warmup_harmful_mass_floor=warmup_harmful_mass_floor,
        warmup_std_mass_ceiling=warmup_std_mass_ceiling,
    )


# ---------------------------------------------------------------------------
# main() — SGTM: single optimizer + HarmfulParamRegistry + masker instantiation
# ---------------------------------------------------------------------------

def main(
    fabric: L.Fabric,
    devices: int,
    seed: int,
    initial_checkpoint_dir: Optional[Path],
    resume: Union[bool, Literal["auto"], Path],
    config: Config,
    data: SafeData,
    out_dir: Path,
    tokenizer_dir: Optional[Path],
    tokenizer: Optional[Tokenizer],
    train: TrainArgs,
    eval: EvalArgs,
    optimizer: Union[str, Dict],
    num_nodes: int = 1,
    upsample_std: float = 1.0,
    upsample_harmful: float = 1.0,
    upsample_unlabeled: float = 1.0,
    stage: Literal["transfer", "warmup"] = "transfer",
    warmup_routing_loss_weight: float = 0.1,
    warmup_harmful_mass_floor: float = 0.6,
    warmup_std_mass_ceiling: float = 0.4,
) -> None:
    validate_args(train, eval, initial_checkpoint_dir, resume)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    fabric.seed_everything(seed)  # same seed for every process to init model (FSDP)

    model = _setup_model(fabric, config, train)

    # SGTM: REMOVED torch.compile(model) — incompatible with Python bool flag checks
    # (per RESEARCH.md anti-pattern: torch.compile traces through Python booleans, making
    # dynamic _activation_masking_enabled flag checks ineffective)
    registry = _build_registry(model, config)
    optimizer = _build_optimizer(fabric, optimizer, registry)

    data_loaders = initialize_data_loaders(data, tokenizer, train, model.max_seq_length)
    val_loaders_for_eval = {
        split_name: (fabric.setup_dataloaders(loader) if isinstance(loader, DataLoader) else loader)
        for split_name, loader in data_loaders["val_loaders"].items()
    }

    if initial_checkpoint_dir:
        fabric.load_raw(initial_checkpoint_dir / "lit_model.pth", model)

    # SGTM: state dict with a single optimizer covering harmful/std/shared params
    state = {
        "model": model,
        "optimizer": optimizer,
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

    gradient_masker, activation_masker = _build_maskers(model, registry, config)

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
        out_dir=out_dir,
        tokenizer_dir=tokenizer_dir,
        train=train,
        eval=eval,
        gradient_masker=gradient_masker,
        activation_masker=activation_masker,
        upsample_std=upsample_std,
        upsample_harmful=upsample_harmful,
        upsample_unlabeled=upsample_unlabeled,
        stage=stage,
        warmup_routing_loss_weight=warmup_routing_loss_weight,
        warmup_harmful_mass_floor=warmup_harmful_mass_floor,
        warmup_std_mass_ceiling=warmup_std_mass_ceiling,
        registry=registry,
        val_loaders=val_loaders_for_eval,
    )

    total_tokens = state["iter_num"] * train.micro_batch_size * model.max_seq_length * fabric.world_size

    save_checkpoint(
        fabric,
        state,
        tokenizer_dir,
        out_dir / "final" / "lit_model.pth",
    )
    if stage == "warmup" and fabric.global_rank == 0:
        fabric.print("Warmup training complete. Evaluate these checkpoints separately if needed:")
        fabric.print(f"  pre:  {initial_checkpoint_dir}")
        fabric.print(f"  post: {out_dir / 'final'}")

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
    data: SafeData,
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
    # SGTM: EVAL-03 — mid-training ablation evaluation
    registry: Optional["HarmfulParamRegistry"] = None,
    val_loaders: Optional[dict[str, DataLoader]] = None,
    stage: Literal["transfer", "warmup"] = "transfer",
    warmup_routing_loss_weight: float = 0.1,
    warmup_harmful_mass_floor: float = 0.6,
    warmup_std_mass_ceiling: float = 0.4,
) -> None:
    model = state["model"]
    optimizer = state["optimizer"]

    if val_loaders is None:
        raise ValueError("fit() requires split-aware val_loaders")

    latest_validation_summary: Optional[ValidationSummary] = None
    if eval.initial_validation:
        initial_summary = collect_validation_summary(fabric, model, val_loaders, max_iters=eval.max_iters)
        fabric.log_dict(initial_summary.scalar_metrics, step=max(state["iter_num"] - 1, 0))
        validation_status = _format_validation_status(initial_summary)
        latest_validation_summary = initial_summary
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        fabric.print("Verifying settings ...")
        collect_validation_summary(fabric, model, val_loaders, max_iters=2, verbose=False)
        validation_status = "n/a"

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

    tokens_per_iter = train.micro_batch_size * model.max_seq_length
    accum_iters = train.gradient_accumulation_iters(devices, num_nodes)
    if train.max_steps is not None:
        max_steps = train.max_steps
        max_iters = max_steps * accum_iters
    else:
        max_tokens_per_device = train.max_tokens // fabric.world_size
        max_iters = max_tokens_per_device // tokens_per_iter
        max_steps = math.ceil(max_iters / accum_iters)
    log_iter_interval = train.log_interval * accum_iters
    initial_step = state["step_count"]

    # SGTM: three split iterators replace single CycleIterator (Pattern 1)
    active_labels = [l for l in list(SPLIT_LABELS) if l in data._loaders]
    split_iters = {label: CycleIterator(data.get_loader(label)) for label in active_labels}
    split_weights = {
        "D_std": upsample_std,
        "D_harmful": upsample_harmful,
        "D_unlabeled": upsample_unlabeled,
    }
    weights = [split_weights[label] for label in active_labels]

    running_loss = RunningMean(window=accum_iters, sync_on_compute=False).to(fabric.device)
    fabric.barrier()
    total_t0 = time.perf_counter()

    # SGTM: warmup_iters uses D_std loader length as proxy for train_dataloader length
    warmup_iters = train.warmup_iters(devices, num_nodes, max_iters, data.get_loader("D_std"))

    # SGTM: iter_num counts micro-batches; step_count counts optimizer steps
    # The outer while loop drives one OPTIMIZER STEP per iteration (accum_iters micro-batches)
    while state["step_count"] < max_steps:
        iter_t0 = time.perf_counter()

        # SGTM: sample split label once per optimizer step (Pattern 1)
        split_label = random.choices(active_labels, weights=weights, k=1)[0]
        state["split_label"] = split_label

        # SGTM: LR update for the shared optimizer
        base_lr = optimizer.defaults["lr"]
        lr = get_lr(base_lr, state["iter_num"], warmup_iters, max_iters, train.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # SGTM: 3-path masking template (Pattern 2)
        # D_std: enable activation masker before the micro-batch window; disable in finally
        if split_label == "D_std":
            activation_masker.enable()

        try:
            lm_loss_values: list[torch.Tensor] = []
            routing_loss_values: list[torch.Tensor] = []
            total_loss_values: list[torch.Tensor] = []
            harmful_mass_values: list[torch.Tensor] = []
            # SGTM: inner accumulation loop — accum_iters micro-batches per optimizer step
            for micro_batch_idx in range(accum_iters):
                state["iter_num"] += 1
                is_accumulating = micro_batch_idx < accum_iters - 1

                train_data = next(split_iters[split_label])
                input_ids = train_data[:, 0 : model.max_seq_length].contiguous().long().to(fabric.device)
                targets = train_data[:, 1 : (model.max_seq_length + 1)].contiguous().long().to(fabric.device)

                with fabric.no_backward_sync(model, enabled=is_accumulating):
                    logits = model(input_ids)
                    lm_loss = chunked_cross_entropy(logits, targets)
                    if stage == "warmup":
                        harmful_mass = collect_warmup_routing_mass(model)
                        routing_loss = warmup_routing_loss(
                            harmful_mass,
                            split_label,
                            harmful_mass_floor=warmup_harmful_mass_floor,
                            std_mass_ceiling=warmup_std_mass_ceiling,
                        )
                        total_loss = lm_loss + warmup_routing_loss_weight * routing_loss
                    else:
                        harmful_mass = lm_loss.new_zeros(())
                        routing_loss = lm_loss.new_zeros(())
                        total_loss = lm_loss
                    fabric.backward(total_loss / accum_iters)

                lm_loss_values.append(lm_loss.detach())
                routing_loss_values.append(routing_loss.detach())
                total_loss_values.append(total_loss.detach())
                harmful_mass_values.append(harmful_mass.detach())
                running_loss.update(total_loss.detach())

        finally:
            # SGTM: always disable activation masker (try/finally avoids masker stuck True on error)
            if split_label == "D_std":
                activation_masker.disable()

        # SGTM: per-split masking then one optimizer step
        if split_label != "D_unlabeled":
            gradient_masker.mask(split_label)
        fabric.clip_gradients(model, optimizer, max_norm=train.max_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        state["step_count"] += 1

        if state["iter_num"] % log_iter_interval == 0:
            loss_val = running_loss.compute().item()
            lm_loss_val = torch.stack(lm_loss_values).mean().item()
            routing_loss_val = torch.stack(routing_loss_values).mean().item()
            total_loss_val = torch.stack(total_loss_values).mean().item()
            harmful_mass_val = torch.stack(harmful_mass_values).mean().item()
            t1 = time.perf_counter()
            throughput.update(
                time=(t1 - total_t0),
                flops=(measured_flops * log_iter_interval),
                batches=state["iter_num"],
                samples=(state["iter_num"] * train.micro_batch_size),
                lengths=(state["iter_num"] * train.micro_batch_size * model.max_seq_length),
            )
            metrics = {
                "loss": total_loss_val,
                "iter": state["iter_num"],
                "step": state["step_count"],
                "iter_time": t1 - iter_t0,
                "remaining_time": (
                    (t1 - total_t0) / (state["step_count"] - initial_step) * (max_steps - state["step_count"])
                    if state["step_count"] > initial_step else 0
                ),
                "tokens": state["iter_num"] * train.micro_batch_size * model.max_seq_length,
                "total_tokens": (state["iter_num"] * train.micro_batch_size * model.max_seq_length * fabric.world_size),
                "learning_rate": lr,
            }
            if stage == "warmup":
                metrics.update(
                    {
                        f"loss_lm_{split_label}": lm_loss_val,
                        f"loss_routing_{split_label}": routing_loss_val,
                        f"loss_total_{split_label}": total_loss_val,
                        f"routing_harmful_mass_{split_label}": harmful_mass_val,
                    }
                )
            else:
                loss_key = {
                    "D_std": "loss_D_std",
                    "D_harmful": "loss_D_harmful",
                    "D_unlabeled": "loss_D_unlabeled",
                }[split_label]
                metrics[loss_key] = loss_val
            fabric.print(
                f"step {state['step_count']} [{split_label}] | iter {metrics['iter']} |"
                f" loss: {metrics['loss']:.3f},"
                f" val: {validation_status} |"
                f" iter time: {metrics['iter_time'] * 1000:.2f} ms"
                f" remaining time: {timedelta(seconds=int(metrics['remaining_time']))!s}"
            )

            throughput_metrics = throughput.compute()
            metrics.update(throughput_metrics)
            fabric.log_dict(metrics, step=state["iter_num"] - 1)

        if state["step_count"] % eval.interval == 0:
            t0 = time.perf_counter()
            validation_summary = collect_validation_summary(fabric, model, val_loaders, max_iters=eval.max_iters)
            latest_validation_summary = validation_summary
            validation_status = _format_validation_status(validation_summary)
            td = time.perf_counter() - t0

            fabric.print(f"iter {state['iter_num']}: val {validation_status}, val time: {td * 1000:.2f} ms")
            fabric.log_dict(validation_summary.scalar_metrics, step=state["iter_num"] - 1)
            fabric.barrier()

        if train.save_interval is not None and state["step_count"] % train.save_interval == 0:
            checkpoint_dir = out_dir / f"step-{state['step_count']:08d}"
            save_checkpoint(
                fabric,
                state,
                tokenizer_dir,
                checkpoint_dir / "lit_model.pth",
            )
            # SGTM EVAL-03: mid-training ablation evaluation at each checkpoint
            if registry is not None:
                ablated_summary = evaluate_with_ablation(
                    fabric,
                    model,
                    registry,
                    val_loaders,
                    iter_num=state["iter_num"],
                    eval_args=eval,
                )

    # Final validation
    if eval.final_validation:
        final_summary = collect_validation_summary(fabric, model, val_loaders, max_iters=eval.max_iters)
        latest_validation_summary = final_summary
        fabric.log_dict(final_summary.scalar_metrics, step=state["iter_num"])
        fabric.print(f"Final evaluation | val { _format_validation_status(final_summary) }")


# ---------------------------------------------------------------------------
# evaluate_with_ablation() — EVAL-03: in-training ablation evaluation
# ---------------------------------------------------------------------------


def evaluate_with_ablation(
    fabric: L.Fabric,
    model: nn.Module,
    registry: "HarmfulParamRegistry",
    val_loaders: dict[str, DataLoader],
    iter_num: int,
    eval_args: Optional["EvalArgs"] = None,
) -> ValidationSummary:
    harmful_params = registry.parameters_by_type("theta_harmful")
    saved = [p.data.clone() for p in harmful_params]
    was_training = model.training
    max_iters = eval_args.max_iters if eval_args is not None else 100
    try:
        for p in harmful_params:
            p.data.zero_()
        summary = collect_validation_summary(
            fabric,
            model,
            val_loaders,
            max_iters=max_iters,
            verbose=False,
            metric_prefix="ablated_",
        )
    finally:
        for p, saved_data in zip(harmful_params, saved):
            p.data.copy_(saved_data)
        model.train(was_training)
    fabric.log_dict(summary.scalar_metrics, step=iter_num)
    fabric.print(
        f"Ablation eval (iter {iter_num}): "
        + ", ".join(f"{k}={v}" for k, v in summary.scalar_metrics.items())
    )
    return summary


# ---------------------------------------------------------------------------
# initialize_data_loaders() — SGTM: set up data module and return structured loaders
# ---------------------------------------------------------------------------

def initialize_data_loaders(
    data: SafeData,
    tokenizer: Optional[Tokenizer],
    train: TrainArgs,
    block_size: int,
) -> dict[str, object]:
    """Set up the data module and return structured train/val loader handles."""
    data.connect(tokenizer=tokenizer, batch_size=train.micro_batch_size, max_seq_length=block_size)
    return data.initialize_loaders()


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

def save_checkpoint(fabric, state, tokenizer_dir, checkpoint_file, include_optimizer: bool = True):
    model = state["model"]
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    fabric.print(f"Saving checkpoint to {str(checkpoint_file)!r}")
    save_state = _checkpoint_state_for_save(state, include_optimizer=include_optimizer)
    try:
        fabric.save(checkpoint_file, save_state)
    except AssertionError as ex:
        if not _should_retry_fsdp_checkpoint_without_optimizer(fabric, ex, include_optimizer):
            raise
        fabric.print(
            "FSDP optimizer-state checkpointing hit a torch assertion; retrying without optimizer state."
        )
        fabric.save(checkpoint_file, _checkpoint_state_for_save(state, include_optimizer=False))
    if fabric.global_rank == 0:
        save_hyperparameters(setup, checkpoint_file.parent)
        if tokenizer_dir is not None:
            copy_config_files(tokenizer_dir, checkpoint_file.parent)
        save_config(model.config, checkpoint_file.parent)


def _checkpoint_state_for_save(state, include_optimizer: bool):
    if include_optimizer:
        _ensure_optimizer_state_coverage(state["optimizer"])
        return state
    return {k: v for k, v in state.items() if k != "optimizer"}


def _ensure_optimizer_state_coverage(optimizer) -> None:
    """Backfill empty optimizer-state entries for params that have never stepped.

    Adam-style optimizers allocate per-parameter state lazily when a parameter
    first receives a gradient. Under split-masked training, some parameters can
    legitimately stay untouched for many steps, but FSDP full optimizer-state
    checkpointing still expects every optimizer-managed parameter to exist in
    ``optimizer.state``.
    """
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            optimizer.state.setdefault(param, {})


def _should_retry_fsdp_checkpoint_without_optimizer(fabric, ex: AssertionError, include_optimizer: bool) -> bool:
    if not include_optimizer or not isinstance(fabric.strategy, FSDPStrategy):
        return False
    return "Manually calculated _sharded_numel_padded is incorrect" in str(ex)


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
    required = [(train, ["max_tokens", "max_norm"])]
    for args, names in required:
        for name in names:
            if getattr(args, name) is None:
                issues.append(f"{__file__} requires the {name!r} argument. This is set in {args}")
    if initial_checkpoint_dir and resume:
        issues.append("Can't provide both `--resume` and `--initial_checkpoint_dir`. Choose one.")
    if issues:
        raise ValueError("\n".join(issues))
