"""
RED test stubs for GradientMasker and ActivationMasker (MASK-01, MASK-02, MASK-04).

These tests will fail with ImportError until safemoe/masking.py is implemented
in plan 02-04. This is the intended RED state.
"""

import pytest
import torch
import torch.nn as nn

# These imports will raise ImportError until implementation plans run.
# That ImportError is the expected RED state for this plan.
from safemoe.masking import HarmfulParamRegistry, GradientMasker, ActivationMasker
from safemoe.config import SafeMoEConfig

import litgpt

# Small config dimensions for CPU-only tests (< 5 seconds).
SMALL_CONFIG = dict(
    padded_vocab_size=1024,
    vocab_size=1024,
    n_layer=2,
    n_embd=32,
    n_head=4,
    n_query_groups=4,
    head_size=8,
    n_expert=4,
    n_expert_per_token=2,
    moe_intermediate_size=64,
    mlp_class_name="LLaMAMoE",
    harmful_expert_indices=[0, 1],
    num_harmful_experts=2,
    harmful_attn_heads=[],
)


def test_gradient_masker_zeroes_theta_std_grads():
    """
    MASK-01: After backward on a D_harmful batch + gradient_masker.mask(),
    all theta_std gradients are None and all theta_harmful gradients are not None.
    """
    config = SafeMoEConfig(**SMALL_CONFIG)
    model = litgpt.GPT(config)
    model.train()

    registry = HarmfulParamRegistry(model, config)
    gradient_masker = GradientMasker(registry)

    # Build a tiny random input batch: (batch=1, seq_len=4)
    batch_size, seq_len = 1, 4
    input_ids = torch.randint(0, config.padded_vocab_size, (batch_size, seq_len))

    # Forward + backward on simulated D_harmful batch
    logits = model(input_ids)
    loss = logits.sum()
    loss.backward()

    # Apply gradient masking
    gradient_masker.mask()

    # All theta_std gradients must be None after masking
    for p in registry.parameters_by_type("theta_std"):
        assert p.grad is None, (
            "GradientMasker.mask() should set theta_std gradients to None, "
            f"but found non-None grad with shape {p.grad.shape}"
        )

    # At least some theta_harmful gradients must be non-None (they received signal)
    harmful_params = registry.parameters_by_type("theta_harmful")
    non_none_harmful = [p for p in harmful_params if p.grad is not None]
    assert len(non_none_harmful) > 0, (
        "After D_harmful backward, at least some theta_harmful params must have non-None grad"
    )


def test_activation_masker_zeroes_harmful_expert_output():
    """
    MASK-02: With ActivationMasker enabled, SafeMoELayer._activation_masking_enabled
    is True on every layer; the model still runs without error.
    When masker is disabled, _activation_masking_enabled is False on every layer.
    """
    config = SafeMoEConfig(**SMALL_CONFIG)
    model = litgpt.GPT(config)
    model.eval()

    activation_masker = ActivationMasker(model)

    # Before enable: all SafeMoELayer instances have masking disabled
    from safemoe.model import SafeMoELayer
    safemoe_layers = [m for m in model.modules() if isinstance(m, SafeMoELayer)]
    assert len(safemoe_layers) > 0, "Model must contain at least one SafeMoELayer"

    for layer in safemoe_layers:
        assert not layer._activation_masking_enabled, (
            "Before enable(), _activation_masking_enabled must be False"
        )

    # Enable masking
    activation_masker.enable()

    for layer in safemoe_layers:
        assert layer._activation_masking_enabled, (
            "After enable(), _activation_masking_enabled must be True"
        )

    # Model forward should still run without error with masking enabled
    input_ids = torch.randint(0, config.padded_vocab_size, (1, 4))
    with torch.no_grad():
        output = model(input_ids)
    assert output is not None, "Model forward with activation masking enabled must not crash"

    # Disable masking
    activation_masker.disable()

    for layer in safemoe_layers:
        assert not layer._activation_masking_enabled, (
            "After disable(), _activation_masking_enabled must be False"
        )


def test_masking_invariants_combined():
    """
    MASK-04 (partial): Combined D_harmful and D_std step sequence does not corrupt
    each other's gradient state.

    D_harmful step: backward + gradient_masker.mask() -> theta_std.grad is None
    D_std step: activation_masker.enable() + forward + activation_masker.disable()
    After both steps: neither step corrupts the other's state.
    """
    config = SafeMoEConfig(**SMALL_CONFIG)
    model = litgpt.GPT(config)
    model.train()

    registry = HarmfulParamRegistry(model, config)
    gradient_masker = GradientMasker(registry)
    activation_masker = ActivationMasker(model)

    input_ids = torch.randint(0, config.padded_vocab_size, (1, 4))

    # --- D_harmful step ---
    model.zero_grad(set_to_none=True)
    logits = model(input_ids)
    loss_harmful = logits.sum()
    loss_harmful.backward()
    gradient_masker.mask()

    # After D_harmful step: theta_std grads must all be None
    for p in registry.parameters_by_type("theta_std"):
        assert p.grad is None, (
            "After D_harmful step, theta_std.grad must be None (gradient masking applied)"
        )

    # --- D_std step ---
    model.zero_grad(set_to_none=True)
    activation_masker.enable()
    logits2 = model(input_ids)
    loss_std = logits2.sum()
    loss_std.backward()
    activation_masker.disable()

    # After D_std step: activation masking must be restored to disabled
    from safemoe.model import SafeMoELayer
    for layer in [m for m in model.modules() if isinstance(m, SafeMoELayer)]:
        assert not layer._activation_masking_enabled, (
            "After D_std step, activation masking must be disabled (disable() was called)"
        )


def test_set_to_none_adam_state_integrity():
    """
    MASK-04: Two AdamW optimizers (one for theta_harmful, one for theta_std)
    with zero_grad(set_to_none=True).
    After one D_harmful step with gradient masking, theta_std optimizer has
    no accumulated momentum state (state dict is empty for theta_std params).
    """
    config = SafeMoEConfig(**SMALL_CONFIG)
    model = litgpt.GPT(config)
    model.train()

    registry = HarmfulParamRegistry(model, config)
    gradient_masker = GradientMasker(registry)

    harmful_params = registry.parameters_by_type("theta_harmful")
    std_params = registry.parameters_by_type("theta_std")

    harmful_optimizer = torch.optim.AdamW(harmful_params, lr=1e-3)
    std_optimizer = torch.optim.AdamW(std_params, lr=1e-3)

    input_ids = torch.randint(0, config.padded_vocab_size, (1, 4))

    # D_harmful step: zero_grad, forward, backward, mask, step harmful optimizer only
    harmful_optimizer.zero_grad(set_to_none=True)
    std_optimizer.zero_grad(set_to_none=True)

    logits = model(input_ids)
    loss = logits.sum()
    loss.backward()

    gradient_masker.mask()
    harmful_optimizer.step()
    # Deliberately do NOT step std_optimizer

    # After stepping only harmful_optimizer, std_optimizer state must remain empty
    # (no momentum accumulated for theta_std params since they had no gradients)
    for param_group in std_optimizer.param_groups:
        for p in param_group["params"]:
            state = std_optimizer.state.get(p, {})
            # Adam state keys: 'step', 'exp_avg', 'exp_avg_sq'
            # If step() was never called for std_optimizer, state must be empty
            assert len(state) == 0, (
                f"theta_std param should have no Adam state after D_harmful step, "
                f"but found state keys: {list(state.keys())}"
            )
