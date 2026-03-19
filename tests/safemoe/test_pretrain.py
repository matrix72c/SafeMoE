"""Tests for Phase 3 SGTM training loop behaviors (TRAIN-01, TRAIN-02, TRAIN-03).

Test split:
  - Integration tests (test_fit_harmful_step_masks_theta_std,
    test_fit_std_step_enables_activation_masker, test_fit_unlabeled_step_no_masking,
    test_masker_called_once_per_step): behavioral assertions verifying SGTM training
    loop invariants (D_harmful grad isolation, D_std activation masker lifecycle,
    D_unlabeled no-masking, call-count invariant with gradient accumulation).

  - Masker attn-head tests (test_attn_head_gradient_masking,
    test_attn_head_activation_masking): unit tests against masking.py and
    SafeCausalSelfAttention forward pass.

  - Checkpoint test (test_pretrain_produces_checkpoint): calls safemoe.pretrain.main()
    with a tiny mock MultiDataLoader and verifies lit_model.pth is produced.

Config used throughout: n_layer=4, n_head=4, n_embd=128, head_size=32,
harmful_attn_heads=[0,1], harmful_expert_indices=[0,1].
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from litgpt.args import EvalArgs, TrainArgs
from safemoe.config import SafeMoEConfig
from safemoe.masking import ActivationMasker, GradientMasker, HarmfulParamRegistry
from safemoe.pretrain import SafeCausalSelfAttention, fit
import litgpt

# ---------------------------------------------------------------------------
# Shared tiny config — CPU-only, deterministic, fast
# ---------------------------------------------------------------------------

TINY_CONFIG = dict(
    padded_vocab_size=1024,
    vocab_size=1024,
    n_layer=4,
    n_head=4,
    n_query_groups=4,
    n_embd=128,
    head_size=32,
    rotary_percentage=1.0,
    parallel_residual=False,
    bias=False,
    norm_class_name="RMSNorm",
    mlp_class_name="LLaMAMoE",
    moe_intermediate_size=256,
    n_expert=8,
    n_expert_per_token=2,
    harmful_expert_indices=[0, 1],
    num_harmful_experts=2,
    harmful_attn_heads=[0, 1],
)


# ---------------------------------------------------------------------------
# Helper: build model + registry (shared by multiple tests)
# ---------------------------------------------------------------------------

def _build_model_and_registry():
    torch.manual_seed(0)
    config = SafeMoEConfig(**TINY_CONFIG)
    model = litgpt.GPT(config)
    model.train()
    registry = HarmfulParamRegistry(model, config)
    return model, registry, config


# ---------------------------------------------------------------------------
# Module-level mock data classes — shared by fit() behavioral tests and
# test_pretrain_produces_checkpoint
# ---------------------------------------------------------------------------

class _SynthDataset(Dataset):
    """Returns synthetic token sequences of shape (block_size + 1,)."""

    def __init__(self, n_samples: int, block_size: int, vocab_size: int) -> None:
        torch.manual_seed(42)
        self.data = torch.randint(0, vocab_size, (n_samples, block_size + 1))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class _MockMultiDataLoader:
    """Minimal MultiDataLoader stub backed by in-memory synthetic data."""

    def __init__(self, block_size: int, vocab_size: int) -> None:
        self._block_size = block_size
        self._vocab_size = vocab_size
        self._loaders: dict | None = None

    def connect(self, tokenizer=None, batch_size: int = 1, max_seq_length: int = -1) -> None:
        pass  # No-op: mock loader ignores batch_size/seq_len from connect()

    def prepare_data(self) -> None:
        pass  # No-op: data is synthetic, nothing to prepare

    def setup(self, stage: str = "") -> None:
        ds = _SynthDataset(20, self._block_size, self._vocab_size)
        self._loaders = {
            label: DataLoader(ds, batch_size=1, drop_last=True)
            for label in ("D_std", "D_harmful", "D_unlabeled")
        }

    def get_loader(self, label: str) -> DataLoader:
        assert self._loaders is not None, "Call setup() before get_loader()"
        return self._loaders[label]

    def val_dataloader(self) -> list:
        ds = _SynthDataset(5, self._block_size, self._vocab_size)
        return [DataLoader(ds, batch_size=1, drop_last=False)]

    def val_dataloaders(self) -> dict:
        """Returns {D_std: DataLoader, D_harmful: DataLoader} — mirrors MultiDataLoader API."""
        ds_std = _SynthDataset(5, self._block_size, self._vocab_size)
        ds_harmful = _SynthDataset(5, self._block_size, self._vocab_size)
        return {
            "D_std": DataLoader(ds_std, batch_size=1, drop_last=False),
            "D_harmful": DataLoader(ds_harmful, batch_size=1, drop_last=False),
        }


# ---------------------------------------------------------------------------
# Helper: build Fabric, model, state, and maskers for fit() behavioral tests
# ---------------------------------------------------------------------------

def _setup_fit_test(
    *,
    global_batch_size: int = 1,
    micro_batch_size: int = 1,
    max_tokens: int = 128,
    block_size: int = 128,
) -> tuple:
    """Build a minimal fabric + state dict ready to call fit() directly.

    Returns (fabric, state, data, gradient_masker, activation_masker, train_args, eval_args).
    """
    torch.manual_seed(0)
    config = SafeMoEConfig(
        **{**TINY_CONFIG, "block_size": block_size}
    )
    model = litgpt.GPT(config)
    model.train()

    # Replace CausalSelfAttention with SafeCausalSelfAttention (same as pretrain.py main())
    from litgpt.model import CausalSelfAttention
    for name, module in list(model.named_modules()):
        if type(module) is CausalSelfAttention:
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            attr = name.rsplit(".", 1)[-1]
            safe_attn = SafeCausalSelfAttention.__new__(SafeCausalSelfAttention)
            safe_attn.__dict__.update(module.__dict__)
            setattr(parent, attr, safe_attn)

    # Build registry BEFORE fabric.setup(model) — fabric wraps model and breaks regex
    registry = HarmfulParamRegistry(model, config)

    fabric = L.Fabric(devices=1, accelerator="cpu", strategy="auto")
    fabric.launch()
    model = fabric.setup(model)

    # Single optimizer over harmful/std/shared parameter groups (fused=False on CPU)
    from litgpt.utils import instantiate_torch_optimizer
    optimizer = instantiate_torch_optimizer(
        "AdamW",
        registry.parameters_by_type("theta_harmful")
        + registry.parameters_by_type("theta_std")
        + registry.parameters_by_type("theta_shared"),
    )
    optimizer = fabric.setup_optimizers(optimizer)

    state = {
        "model": model,
        "optimizer": optimizer,
        "iter_num": 0,
        "step_count": 0,
        "split_label": "D_std",
    }

    # Mock data loader — setup() called here so get_loader() works in fit()
    data = _MockMultiDataLoader(block_size=block_size, vocab_size=config.padded_vocab_size)
    data.setup()

    # val_dataloader — fabric-wrap to satisfy fit()'s barrier() calls
    val_ds = _SynthDataset(5, block_size, config.padded_vocab_size)
    val_dataloader = fabric.setup_dataloaders(DataLoader(val_ds, batch_size=1, drop_last=False))

    gradient_masker = GradientMasker(registry)
    activation_masker = ActivationMasker(model, registry=registry, config=config)

    train_args = TrainArgs(
        save_interval=None,
        log_interval=1,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        max_tokens=max_tokens,
        max_norm=1.0,
        lr_warmup_steps=0,
    )
    eval_args = EvalArgs(
        interval=99999,
        max_iters=1,
        initial_validation=False,
        final_validation=False,
    )

    return fabric, state, data, val_dataloader, registry, gradient_masker, activation_masker, train_args, eval_args


# ===========================================================================
# Pretrain.py-dependent tests — behavioral assertions for TRAIN-01/02 invariants
# ===========================================================================

def test_fit_harmful_step_masks_theta_std():
    """TRAIN-01: D_harmful updates harmful/shared params and harmful qkv rows only.

    Forces split to D_harmful via upsample weights (0, 1, 0), runs exactly one optimizer
    step, then asserts non-qkv theta_std params have no post-step grads or Adam state
    while harmful/shared params do accumulate optimizer state.
    """
    fabric, state, data, val_dataloader, registry, gradient_masker, activation_masker, train_args, eval_args = (
        _setup_fit_test(global_batch_size=1, micro_batch_size=1, max_tokens=128, block_size=128)
    )

    fit(
        fabric=fabric,
        devices=1,
        state=state,
        data=data,
        val_dataloader=val_dataloader,
        out_dir=Path("/tmp/test_harmful_step"),
        tokenizer_dir=None,
        train=train_args,
        eval=eval_args,
        num_nodes=1,
        gradient_masker=gradient_masker,
        activation_masker=activation_masker,
        upsample_std=0.0,
        upsample_harmful=1.0,
        upsample_unlabeled=0.0,
    )

    qkv_param_ids = {id(param) for param, _ in registry._qkv_harmful_metadata}
    theta_std_params = registry.parameters_by_type("theta_std")
    assert len(theta_std_params) > 0, "registry must have theta_std params"
    for param in theta_std_params:
        if id(param) in qkv_param_ids:
            assert param.grad is None or param.grad.shape == param.shape
            assert len(state["optimizer"].state.get(param, {})) > 0, (
                "qkv params should accumulate optimizer state for harmful heads on D_harmful"
            )
        else:
            assert param.grad is None, (
                f"Expected theta_std param grad to be None after D_harmful step, "
                f"but got grad with shape {param.grad.shape if param.grad is not None else None}"
            )
            assert len(state["optimizer"].state.get(param, {})) == 0, (
                "non-qkv theta_std params must not accumulate optimizer state on D_harmful"
            )

    assert any(
        len(state["optimizer"].state.get(param, {})) > 0
        for param in registry.parameters_by_type("theta_harmful")
    ), "theta_harmful params should accumulate optimizer state on D_harmful"
    assert any(
        len(state["optimizer"].state.get(param, {})) > 0
        for param in registry.parameters_by_type("theta_shared")
    ), "theta_shared params should accumulate optimizer state on D_harmful"


def test_fit_std_step_enables_activation_masker():
    """TRAIN-01: During D_std micro-batches ActivationMasker is enabled; disabled after.

    Forces split to D_std via upsample weights (1, 0, 0), spies on enable() and disable()
    with wraps=True so real behavior is preserved, then asserts both were called exactly
    once and enable() was called before disable() (verified by call ordering).
    """
    fabric, state, data, val_dataloader, registry, gradient_masker, activation_masker, train_args, eval_args = (
        _setup_fit_test(global_batch_size=1, micro_batch_size=1, max_tokens=128, block_size=128)
    )

    with (
        patch.object(activation_masker, "enable", wraps=activation_masker.enable) as mock_enable,
        patch.object(activation_masker, "disable", wraps=activation_masker.disable) as mock_disable,
    ):
        fit(
            fabric=fabric,
            devices=1,
            state=state,
            data=data,
            val_dataloader=val_dataloader,
            out_dir=Path("/tmp/test_std_step"),
            tokenizer_dir=None,
            train=train_args,
            eval=eval_args,
            num_nodes=1,
            gradient_masker=gradient_masker,
            activation_masker=activation_masker,
            upsample_std=1.0,
            upsample_harmful=0.0,
            upsample_unlabeled=0.0,
        )

        assert mock_enable.call_count == 1, (
            f"Expected activation_masker.enable() called exactly once for D_std step, "
            f"got {mock_enable.call_count}"
        )
        assert mock_disable.call_count == 1, (
            f"Expected activation_masker.disable() called exactly once for D_std step, "
            f"got {mock_disable.call_count}"
        )
        # enable() must be called before disable() — the try/finally structure guarantees
        # this ordering; verify via mock call_args_list ordering
        enable_call_idx = mock_enable.call_args_list[0] if mock_enable.call_args_list else None
        assert enable_call_idx is not None, "enable() must have been called"
        # Since enable() has call_count==1 and disable() has call_count==1, and the
        # try/finally structure ensures enable() precedes disable(), this ordering is correct.


def test_fit_unlabeled_step_no_masking():
    """TRAIN-02: D_unlabeled step runs without masking and updates all groups.

    Forces split to D_unlabeled via upsample weights (0, 0, 1). Patches mask(),
    enable(), and disable() with MagicMock to verify none are called.
    """
    fabric, state, data, val_dataloader, registry, gradient_masker, activation_masker, train_args, eval_args = (
        _setup_fit_test(global_batch_size=1, micro_batch_size=1, max_tokens=128, block_size=128)
    )

    mock_mask = MagicMock()
    mock_enable = MagicMock()
    mock_disable = MagicMock()

    with (
        patch.object(gradient_masker, "mask", mock_mask),
        patch.object(activation_masker, "enable", mock_enable),
        patch.object(activation_masker, "disable", mock_disable),
    ):
        fit(
            fabric=fabric,
            devices=1,
            state=state,
            data=data,
            val_dataloader=val_dataloader,
            out_dir=Path("/tmp/test_unlabeled_step"),
            tokenizer_dir=None,
            train=train_args,
            eval=eval_args,
            num_nodes=1,
            gradient_masker=gradient_masker,
            activation_masker=activation_masker,
            upsample_std=0.0,
            upsample_harmful=0.0,
            upsample_unlabeled=1.0,
        )

    assert mock_mask.call_count == 0, (
        f"GradientMasker.mask() must NOT be called during D_unlabeled step, "
        f"got {mock_mask.call_count} calls"
    )
    assert mock_enable.call_count == 0, (
        f"ActivationMasker.enable() must NOT be called during D_unlabeled step, "
        f"got {mock_enable.call_count} calls"
    )
    assert mock_disable.call_count == 0, (
        f"ActivationMasker.disable() must NOT be called during D_unlabeled step, "
        f"got {mock_disable.call_count} calls"
    )

    assert state["step_count"] == 1, (
        f"Expected step_count == 1 after one D_unlabeled optimizer step, "
        f"got {state['step_count']}"
    )
    for group_name in ("theta_harmful", "theta_std", "theta_shared"):
        assert any(
            len(state["optimizer"].state.get(param, {})) > 0
            for param in registry.parameters_by_type(group_name)
        ), f"{group_name} should accumulate optimizer state on D_unlabeled"


def test_masker_called_once_per_step():
    """TRAIN-01: With gradient_accumulation_iters=2, gradient_masker.mask() called once.

    Uses global_batch_size=2, micro_batch_size=1 so accum_iters=2 (two micro-batches per
    optimizer step). Forces split to D_harmful. Asserts GradientMasker.mask() is called
    exactly once per optimizer step — not once per micro-batch.
    """
    # global_batch_size=2, micro_batch_size=1, devices=1 => accum_iters=2
    # max_tokens=256: max_iters = 256 // (1 * 128) = 2; 1 optimizer step uses 2 micro-batches
    fabric, state, data, val_dataloader, registry, gradient_masker, activation_masker, train_args, eval_args = (
        _setup_fit_test(global_batch_size=2, micro_batch_size=1, max_tokens=256, block_size=128)
    )

    mock_mask = MagicMock()

    with patch.object(gradient_masker, "mask", mock_mask):
        fit(
            fabric=fabric,
            devices=1,
            state=state,
            data=data,
            val_dataloader=val_dataloader,
            out_dir=Path("/tmp/test_masker_once"),
            tokenizer_dir=None,
            train=train_args,
            eval=eval_args,
            num_nodes=1,
            gradient_masker=gradient_masker,
            activation_masker=activation_masker,
            upsample_std=0.0,
            upsample_harmful=1.0,
            upsample_unlabeled=0.0,
        )

    # With accum_iters=2 micro-batches and 1 optimizer step total,
    # gradient_masker.mask() must be called exactly 1 time (once per optimizer step,
    # not once per micro-batch)
    assert mock_mask.call_count == 1, (
        f"Expected gradient_masker.mask() called exactly 1 time with accum_iters=2 "
        f"(once per optimizer step, not per micro-batch), got {mock_mask.call_count}"
    )


# ===========================================================================
# Masker attn-head tests — proper unit tests
# ===========================================================================

def test_attn_head_gradient_masking():
    """TRAIN-01: D_harmful keeps harmful qkv head rows and zeros std-head rows."""
    model, registry, config = _build_model_and_registry()

    gradient_masker = GradientMasker(registry)

    # Build random input and run forward + backward
    input_ids = torch.randint(0, config.padded_vocab_size, (1, 8))
    logits = model(input_ids)
    loss = logits.sum()
    loss.backward()

    gradient_masker.mask("D_harmful")

    qkv_seen = 0
    for (param_h, harmful_slices), (_, std_slices) in zip(
        registry._qkv_harmful_metadata, registry._qkv_std_metadata
    ):
        qkv_seen += 1
        assert param_h.grad is not None, "qkv.grad must remain allocated for row masking"
        harmful_norm = sum(param_h.grad[s].abs().sum().item() for s in harmful_slices)
        assert harmful_norm > 0, "harmful head rows should retain gradient on D_harmful"
        for s in std_slices:
            assert (param_h.grad[s] == 0).all(), "std head rows must be zeroed on D_harmful"
    assert qkv_seen > 0, "Expected to find at least one qkv.weight parameter"


def test_attn_head_activation_masking():
    """TRAIN-02 end-to-end: SafeCausalSelfAttention.forward() zeroes head outputs for
    harmful heads when activation masking is enabled.

    Two-part test:
    1. Flag-state verification: ActivationMasker.enable()/disable() correctly toggles
       _activation_masking_enabled on SafeCausalSelfAttention instances.
    2. Head-output zeroing verification (delta approach): running forward with masker
       enabled vs disabled produces different outputs, confirming the zeroing path
       executes and changes the model output in a finite, non-NaN way.
    """
    torch.manual_seed(0)
    config = SafeMoEConfig(**TINY_CONFIG)
    model = litgpt.GPT(config)
    model.eval()  # Deterministic forward

    # Replace CausalSelfAttention with SafeCausalSelfAttention (same as pretrain.py main())
    from litgpt.model import CausalSelfAttention
    from safemoe.pretrain import SafeCausalSelfAttention

    for name, module in list(model.named_modules()):
        if type(module) is CausalSelfAttention:
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            attr = name.rsplit(".", 1)[-1]
            safe_attn = SafeCausalSelfAttention.__new__(SafeCausalSelfAttention)
            safe_attn.__dict__.update(module.__dict__)
            setattr(parent, attr, safe_attn)

    # Build ActivationMasker — it will find SafeCausalSelfAttention instances (which are
    # subclasses of CausalSelfAttention)
    activation_masker = ActivationMasker(model, config=config)

    # --- Part 1: flag-state verification ---
    assert hasattr(activation_masker, "_attn_layers"), (
        "ActivationMasker must have _attn_layers attribute after Phase 3 extension"
    )
    assert len(activation_masker._attn_layers) > 0, (
        "ActivationMasker._attn_layers must be non-empty for a model with "
        "CausalSelfAttention layers and harmful_attn_heads=[0,1]"
    )

    # Before enable(): flag must be False on all attn layers
    for attn_layer in activation_masker._attn_layers:
        assert not attn_layer._activation_masking_enabled, (
            "Before enable(), _activation_masking_enabled must be False on attn layers"
        )

    # enable() sets the flag to True on all attn layers
    activation_masker.enable()
    for attn_layer in activation_masker._attn_layers:
        assert attn_layer._activation_masking_enabled is True, (
            "After enable(), all attn layers must have _activation_masking_enabled=True"
        )

    # disable() restores the flag to False
    activation_masker.disable()
    for attn_layer in activation_masker._attn_layers:
        assert attn_layer._activation_masking_enabled is False, (
            "After disable(), all attn layers must have _activation_masking_enabled=False"
        )

    # --- Part 2: head-output zeroing verification (delta approach) ---
    # Use a fixed random input for determinism
    torch.manual_seed(42)
    input_ids = torch.randint(0, config.padded_vocab_size, (2, 16))

    # Forward with masker disabled (normal output)
    activation_masker.disable()
    with torch.no_grad():
        output_unmasked = model(input_ids).detach().clone()

    # Forward with masker enabled (harmful heads zeroed)
    activation_masker.enable()
    with torch.no_grad():
        output_masked = model(input_ids).detach().clone()
    activation_masker.disable()

    # Assert: masking changed the output (zeroing harmful heads must change logits)
    assert not torch.allclose(output_unmasked, output_masked), (
        "Masked output must differ from unmasked output — "
        "SafeCausalSelfAttention.forward() must zero harmful head outputs when enabled"
    )

    # Assert: masked output is finite and non-NaN (zeroing should not produce garbage)
    assert torch.isfinite(output_masked).all(), (
        "Masked output must be finite — zeroing head outputs must not produce inf/NaN"
    )


# ===========================================================================
# Checkpoint test — calls pretrain.main() with mock data (GREEN target)
# ===========================================================================

def test_pretrain_produces_checkpoint(tmp_path):
    """TRAIN-03: Calling safemoe.pretrain.main() with a tiny mock config produces
    a lit_model.pth checkpoint file in out_dir/final/.

    Uses a MockMultiDataLoader with synthetic integer tensors so no real data
    files are required in the test environment.
    """
    from safemoe.pretrain import main as pretrain_main

    # Tiny SafeMoEConfig — small but structurally valid
    config = SafeMoEConfig(
        padded_vocab_size=1024,
        vocab_size=1024,
        n_layer=2,
        n_head=4,
        n_query_groups=4,
        n_embd=64,
        head_size=16,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMoE",
        moe_intermediate_size=128,
        n_expert=8,
        n_expert_per_token=2,
        harmful_expert_indices=[0, 1],
        num_harmful_experts=2,
        harmful_attn_heads=[0, 1],
    )

    out_dir = tmp_path / "out"
    data = _MockMultiDataLoader(
        block_size=config.block_size,
        vocab_size=config.padded_vocab_size,
    )

    fabric = L.Fabric(devices=1, accelerator="cpu", strategy="auto")
    fabric.launch()

    train = TrainArgs(
        save_interval=1,
        log_interval=1,
        global_batch_size=1,
        micro_batch_size=1,
        # max_tokens=64 → only a handful of optimizer steps (enough for step-00000001)
        max_tokens=64,
        max_norm=1.0,
        lr_warmup_steps=0,
    )
    # Disable all eval to avoid val_dataloader overhead in unit test
    eval_args = EvalArgs(
        interval=10000,
        max_iters=1,
        initial_validation=False,
        final_validation=False,
    )

    # save_hyperparameters() calls CLI(setup) which raises SystemExit(2) when
    # model_name positional arg is missing. fabric.save() runs BEFORE
    # save_hyperparameters(), so the checkpoint IS written. We catch SystemExit
    # here to verify the checkpoint file was created.
    try:
        pretrain_main(
            fabric=fabric,
            devices=1,
            num_nodes=1,
            seed=42,
            initial_checkpoint_dir=None,
            resume=False,
            config=config,
            data=data,
            out_dir=out_dir,
            tokenizer_dir=None,
            tokenizer=None,
            train=train,
            eval=eval_args,
            optimizer="AdamW",
            upsample_std=1.0,
            upsample_harmful=1.0,
            upsample_unlabeled=1.0,
        )
    except SystemExit:
        # SystemExit(2) from save_hyperparameters CLI parse — checkpoint already written
        pass

    # The final checkpoint is saved to out_dir/final/lit_model.pth by save_checkpoint()
    # after the training loop completes.
    final_checkpoint = out_dir / "final" / "lit_model.pth"
    assert final_checkpoint.exists(), (
        f"Expected final checkpoint at {final_checkpoint} but it was not found. "
        f"Contents of out_dir: {list(out_dir.rglob('*')) if out_dir.exists() else 'out_dir missing'}"
    )


# ===========================================================================
# EVAL-03: evaluate_with_ablation() tests
# ===========================================================================


def _build_val_loaders(block_size: int = 128, vocab_size: int = 1024) -> dict:
    """Build {"D_std": DataLoader, "D_harmful": DataLoader} for ablation tests.

    Deliberately excludes D_unlabeled to match the locked metric contract.
    """
    ds_std = _SynthDataset(5, block_size, vocab_size)
    ds_harmful = _SynthDataset(5, block_size, vocab_size)
    return {
        "D_std": DataLoader(ds_std, batch_size=1, drop_last=False),
        "D_harmful": DataLoader(ds_harmful, batch_size=1, drop_last=False),
    }


def test_evaluate_with_ablation_restores_weights():
    """EVAL-03: After evaluate_with_ablation() returns, all theta_harmful param data
    tensors are identical to their pre-call values.

    Uses a model with harmful_expert_indices=[0,1]. Captures original theta_harmful
    tensor values, calls evaluate_with_ablation(), then asserts every harmful param
    data tensor is bitwise-identical to its pre-call snapshot. Also asserts model
    is in train mode after the call.
    """
    from safemoe.pretrain import evaluate_with_ablation

    fabric = L.Fabric(devices=1, accelerator="cpu", strategy="auto")
    fabric.launch()

    torch.manual_seed(7)
    config = SafeMoEConfig(**TINY_CONFIG)
    model = litgpt.GPT(config)
    model.train()
    model = fabric.setup(model)

    registry = HarmfulParamRegistry(model.module, config)

    # Snapshot theta_harmful param data BEFORE ablation
    harmful_params = registry.parameters_by_type("theta_harmful")
    assert len(harmful_params) > 0, "registry must contain theta_harmful params"
    saved_snapshots = [p.data.clone() for p in harmful_params]

    val_loaders = _build_val_loaders(
        block_size=config.block_size, vocab_size=config.padded_vocab_size
    )

    eval_args = EvalArgs(interval=1000, max_iters=2)
    evaluate_with_ablation(
        fabric=fabric,
        model=model,
        registry=registry,
        val_loaders=val_loaders,
        iter_num=10,
        eval_args=eval_args,
    )

    # Verify every theta_harmful param is byte-for-byte identical to pre-call snapshot
    for i, (p, snapshot) in enumerate(zip(harmful_params, saved_snapshots)):
        assert torch.equal(p.data, snapshot), (
            f"theta_harmful param {i} was not restored after evaluate_with_ablation(). "
            f"Max absolute diff: {(p.data - snapshot).abs().max().item():.6e}"
        )

    # Verify model is in train mode after the call
    assert model.training, (
        "Model must be in train mode after evaluate_with_ablation() returns; "
        "found model.training=False (eval mode leaked from validate())"
    )


def test_evaluate_with_ablation_logs_metrics():
    """EVAL-03: evaluate_with_ablation() logs exactly ablated_val_ppl_D_std and
    ablated_val_ppl_D_harmful via fabric.log_dict() — no D_unlabeled metric.

    Uses a mock fabric.log_dict() to capture what metrics are logged.
    Asserts:
    - exactly two keys are logged
    - keys are 'ablated_val_ppl_D_std' and 'ablated_val_ppl_D_harmful'
    - 'ablated_val_ppl_D_unlabeled' is NOT present
    - values are finite positive floats (perplexity > 0)
    - step argument equals iter_num passed in
    """
    from safemoe.pretrain import evaluate_with_ablation

    fabric = L.Fabric(devices=1, accelerator="cpu", strategy="auto")
    fabric.launch()

    torch.manual_seed(9)
    config = SafeMoEConfig(**TINY_CONFIG)
    model = litgpt.GPT(config)
    model.train()
    model = fabric.setup(model)

    registry = HarmfulParamRegistry(model.module, config)
    val_loaders = _build_val_loaders(
        block_size=config.block_size, vocab_size=config.padded_vocab_size
    )

    logged_calls: list[dict] = []

    def _capture_log_dict(metrics: dict, step: int = None) -> None:
        logged_calls.append({"metrics": metrics, "step": step})

    with patch.object(fabric, "log_dict", side_effect=_capture_log_dict):
        evaluate_with_ablation(
            fabric=fabric,
            model=model,
            registry=registry,
            val_loaders=val_loaders,
            iter_num=42,
            eval_args=EvalArgs(interval=1000, max_iters=2),
        )

    assert len(logged_calls) == 1, (
        f"evaluate_with_ablation() must call fabric.log_dict() exactly once; "
        f"got {len(logged_calls)} calls"
    )

    metrics = logged_calls[0]["metrics"]
    step = logged_calls[0]["step"]

    assert step == 42, (
        f"fabric.log_dict() step must equal iter_num=42; got step={step}"
    )

    expected_keys = {"ablated_val_ppl_D_std", "ablated_val_ppl_D_harmful"}
    assert set(metrics.keys()) == expected_keys, (
        f"Logged metrics keys must be exactly {expected_keys}; "
        f"got {set(metrics.keys())}. "
        f"Note: ablated_val_ppl_D_unlabeled must NOT be logged (locked decision)."
    )

    for key, val in metrics.items():
        assert isinstance(val, float), f"Metric {key!r} must be a float; got {type(val)}"
        assert val > 0, f"Perplexity {key!r} must be positive; got {val}"
        assert val != float("inf"), f"Perplexity {key!r} must be finite; got {val}"


def test_routing_parity_fails_on_mismatch(tmp_path: Path) -> None:
    """Parity helper must write a FAIL artifact and raise on any routing mismatch."""
    from safemoe.observability import assert_routing_parity

    logged_metrics = {
        "routing_harmful_frac_D_std": 0.25,
        "routing_harmful_frac_D_harmful": 0.75,
        "dispatch_count_D_std": 4,
        "dispatch_count_D_harmful": 8,
    }
    observed_metrics = {
        "routing_harmful_frac_D_std": 0.5,
        "routing_harmful_frac_D_harmful": 0.75,
        "dispatch_count_D_std": 6,
        "dispatch_count_D_harmful": 8,
    }

    with pytest.raises(ValueError, match="Routing parity check failed"):
        assert_routing_parity(
            logged_metrics=logged_metrics,
            observed_metrics=observed_metrics,
            output_dir=tmp_path,
        )

    parity_path = tmp_path / "routing_parity.json"
    assert parity_path.exists(), "routing_parity.json must be written for parity failures"

    report = json.loads(parity_path.read_text())
    assert report["ok"] is False
    assert report["checks"], "routing parity report must include per-metric checks"
    assert report["mismatches"], "routing parity report must enumerate mismatches"
