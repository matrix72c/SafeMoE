"""RED test stubs for Phase 3 SGTM training loop behaviors (TRAIN-01, TRAIN-02, TRAIN-03).

Test split:
  - Pretrain.py-dependent tests (test_fit_harmful_step_masks_theta_std,
    test_fit_std_step_enables_activation_masker, test_fit_unlabeled_step_no_masking,
    test_masker_called_once_per_step, test_pretrain_produces_checkpoint):
    These fail because safemoe.pretrain does not exist yet.

  - Masker attn-head tests (test_attn_head_gradient_masking,
    test_attn_head_activation_masking): These are proper unit tests against
    masking.py. They fail because the attn-head masking logic is not yet in
    masking.py (Task 2 makes them GREEN).

Config used throughout: n_layer=4, n_head=4, n_embd=128, head_size=32,
harmful_attn_heads=[0,1], harmful_expert_indices=[0,1].
"""

from __future__ import annotations

import pytest
import torch

from safemoe.masking import HarmfulParamRegistry, GradientMasker, ActivationMasker
from safemoe.config import SafeMoEConfig
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


# ===========================================================================
# Pretrain.py-dependent tests — RED stubs (ImportError is expected RED state)
# ===========================================================================

def test_fit_harmful_step_masks_theta_std():
    """TRAIN-01: After one D_harmful optimizer step, all theta_std params have grad=None.

    RED stub: fails because safemoe.pretrain does not exist yet.
    """
    try:
        import safemoe.pretrain  # noqa: F401
    except ImportError:
        pytest.fail(
            "safemoe.pretrain not importable — expected RED state. "
            "Implement safemoe/pretrain.py in plan 03-02."
        )


def test_fit_std_step_enables_activation_masker():
    """TRAIN-01: During D_std micro-batches ActivationMasker is enabled; disabled after.

    RED stub: fails because safemoe.pretrain does not exist yet.
    """
    try:
        import safemoe.pretrain  # noqa: F401
    except ImportError:
        pytest.fail(
            "safemoe.pretrain not importable — expected RED state. "
            "Implement safemoe/pretrain.py in plan 03-02."
        )


def test_fit_unlabeled_step_no_masking():
    """TRAIN-02: D_unlabeled step runs without masking, both optimizers step.

    RED stub: fails because safemoe.pretrain does not exist yet.
    """
    try:
        import safemoe.pretrain  # noqa: F401
    except ImportError:
        pytest.fail(
            "safemoe.pretrain not importable — expected RED state. "
            "Implement safemoe/pretrain.py in plan 03-02."
        )


def test_masker_called_once_per_step():
    """TRAIN-01: With gradient_accumulation_iters=2, gradient_masker.mask() called once.

    RED stub: fails because safemoe.pretrain does not exist yet.
    """
    try:
        import safemoe.pretrain  # noqa: F401
    except ImportError:
        pytest.fail(
            "safemoe.pretrain not importable — expected RED state. "
            "Implement safemoe/pretrain.py in plan 03-02."
        )


# ===========================================================================
# Masker attn-head tests — proper unit tests, fail until Task 2 (GREEN phase)
# ===========================================================================

def test_attn_head_gradient_masking():
    """TRAIN-01: After D_harmful backward with harmful_attn_heads=[0,1],
    qkv.weight.grad rows for heads 0 and 1 are zero; rows for heads 2+ are non-zero.

    This test is properly implemented. It fails in RED state because
    GradientMasker.mask() does not yet zero qkv row slices.
    Task 2 makes it GREEN.
    """
    model, registry, config = _build_model_and_registry()

    gradient_masker = GradientMasker(registry)

    # Build random input and run forward + backward
    input_ids = torch.randint(0, config.padded_vocab_size, (1, 8))
    logits = model(input_ids)
    loss = logits.sum()
    loss.backward()

    # Apply gradient masking
    gradient_masker.mask()

    # Verify: for each qkv param, harmful-head row slices are zero
    assert len(registry._qkv_harmful_metadata) > 0, (
        "registry._qkv_harmful_metadata must be non-empty when harmful_attn_heads=[0,1]"
    )

    for param, slices in registry._qkv_harmful_metadata:
        assert param.grad is not None, (
            "qkv.weight.grad must not be None after D_harmful backward "
            "(it is in theta_std but not wiped by Phase 3 gradient masker)"
        )
        for s in slices:
            assert (param.grad[s] == 0).all(), (
                f"qkv.weight.grad[{s}] must be zero after mask() for harmful head rows, "
                f"but found non-zero values: {param.grad[s]}"
            )

        # At least some rows outside the harmful slices must be non-zero
        # Build a mask of zeroed rows
        n_rows = param.grad.shape[0]
        zeroed_rows = set()
        for s in slices:
            zeroed_rows.update(range(*s.indices(n_rows)))
        std_rows = [i for i in range(n_rows) if i not in zeroed_rows]
        assert len(std_rows) > 0, "There must be at least one standard (non-harmful) head row"
        std_grad = param.grad[std_rows]
        assert std_grad.abs().sum() > 0, (
            "Standard-head rows of qkv.weight.grad must be non-zero after D_harmful backward"
        )


def test_attn_head_activation_masking():
    """TRAIN-01: ActivationMasker(model, config=config) collects CausalSelfAttention
    instances in self._attn_layers and toggles _activation_masking_enabled correctly.

    Flag-state only — actual head output zeroing is verified in Plan 03-03 Task 2.
    Task 2 makes this test GREEN.
    """
    torch.manual_seed(0)
    config = SafeMoEConfig(**TINY_CONFIG)
    model = litgpt.GPT(config)

    activation_masker = ActivationMasker(model, config=config)

    # _attn_layers must be populated with CausalSelfAttention instances
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
    assert activation_masker._attn_layers[0]._activation_masking_enabled is True, (
        "After enable(), _activation_masking_enabled must be True on first attn layer"
    )
    for attn_layer in activation_masker._attn_layers:
        assert attn_layer._activation_masking_enabled is True, (
            "After enable(), all attn layers must have _activation_masking_enabled=True"
        )

    # disable() restores the flag to False
    activation_masker.disable()
    assert activation_masker._attn_layers[0]._activation_masking_enabled is False, (
        "After disable(), _activation_masking_enabled must be False on first attn layer"
    )
    for attn_layer in activation_masker._attn_layers:
        assert attn_layer._activation_masking_enabled is False, (
            "After disable(), all attn layers must have _activation_masking_enabled=False"
        )


# ===========================================================================
# Checkpoint test — RED stub (pretrain.py not yet implemented)
# ===========================================================================

def test_pretrain_produces_checkpoint():
    """TRAIN-03: Calling setup() with tiny config produces a lit_model.pth checkpoint.

    RED stub: fails because safemoe.pretrain does not exist yet.
    """
    try:
        import safemoe.pretrain  # noqa: F401
    except ImportError:
        pytest.fail(
            "safemoe.pretrain not importable — expected RED state. "
            "Implement safemoe/pretrain.py in plan 03-02."
        )
