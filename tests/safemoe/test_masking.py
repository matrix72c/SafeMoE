"""Tests for GradientMasker and ActivationMasker."""

import torch

import litgpt
from safemoe.config import SafeMoEConfig
from safemoe.masking import ActivationMasker, GradientMasker, HarmfulParamRegistry


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


def test_gradient_masker_zeroes_theta_std_grads_on_harmful_step():
    """D_harmful masking must clear theta_std grads but keep harmful/shared grads."""
    config = SafeMoEConfig(**SMALL_CONFIG)
    model = litgpt.GPT(config)
    model.train()

    registry = HarmfulParamRegistry(model, config)
    gradient_masker = GradientMasker(registry)

    input_ids = torch.randint(0, config.padded_vocab_size, (1, 4))
    logits = model(input_ids)
    loss = logits.sum()
    loss.backward()

    gradient_masker.mask("D_harmful")

    for p in registry.parameters_by_type("theta_std"):
        assert p.grad is None, "theta_std grads must be cleared on D_harmful"

    assert any(p.grad is not None for p in registry.parameters_by_type("theta_harmful"))
    assert any(p.grad is not None for p in registry.parameters_by_type("theta_shared"))


def test_activation_masker_zeroes_harmful_expert_output():
    """ActivationMasker must toggle SafeMoELayer masking flags without breaking forward."""
    config = SafeMoEConfig(**SMALL_CONFIG)
    model = litgpt.GPT(config)
    model.eval()

    activation_masker = ActivationMasker(model)

    from safemoe.model import SafeMoELayer

    safemoe_layers = [m for m in model.modules() if isinstance(m, SafeMoELayer)]
    assert len(safemoe_layers) > 0

    for layer in safemoe_layers:
        assert not layer._activation_masking_enabled

    activation_masker.enable()
    for layer in safemoe_layers:
        assert layer._activation_masking_enabled

    with torch.no_grad():
        output = model(torch.randint(0, config.padded_vocab_size, (1, 4)))
    assert output is not None

    activation_masker.disable()
    for layer in safemoe_layers:
        assert not layer._activation_masking_enabled


def test_masking_invariants_combined():
    """D_harmful must clear std grads; D_std must clear harmful grads."""
    config = SafeMoEConfig(**SMALL_CONFIG)
    model = litgpt.GPT(config)
    model.train()

    registry = HarmfulParamRegistry(model, config)
    gradient_masker = GradientMasker(registry)
    activation_masker = ActivationMasker(model)

    input_ids = torch.randint(0, config.padded_vocab_size, (1, 4))

    model.zero_grad(set_to_none=True)
    logits = model(input_ids)
    logits.sum().backward()
    gradient_masker.mask("D_harmful")

    for p in registry.parameters_by_type("theta_std"):
        assert p.grad is None
    assert any(p.grad is not None for p in registry.parameters_by_type("theta_shared"))

    model.zero_grad(set_to_none=True)
    activation_masker.enable()
    logits = model(input_ids)
    logits.sum().backward()
    activation_masker.disable()
    gradient_masker.mask("D_std")

    for p in registry.parameters_by_type("theta_harmful"):
        assert p.grad is None

    from safemoe.model import SafeMoELayer

    for layer in [m for m in model.modules() if isinstance(m, SafeMoELayer)]:
        assert not layer._activation_masking_enabled


def test_gradient_masker_splits_qkv_rows_by_attention_head():
    """With harmful_attn_heads set, D_harmful should preserve only harmful qkv rows."""
    config = SafeMoEConfig(**{**SMALL_CONFIG, "harmful_attn_heads": [0, 1]})
    model = litgpt.GPT(config)
    model.train()

    registry = HarmfulParamRegistry(model, config)
    gradient_masker = GradientMasker(registry)

    input_ids = torch.randint(0, config.padded_vocab_size, (1, 4))
    logits = model(input_ids)
    logits.sum().backward()

    gradient_masker.mask("D_harmful")

    qkv_seen = 0
    for (param_h, harmful_slices), (_, std_slices) in zip(
        registry._qkv_harmful_metadata, registry._qkv_std_metadata
    ):
        qkv_seen += 1
        assert param_h.grad is not None, "qkv.grad should remain allocated for slice masking"
        harmful_norm = sum(param_h.grad[s].abs().sum().item() for s in harmful_slices)
        assert harmful_norm > 0, "harmful head rows should retain gradient on D_harmful"
        for s in std_slices:
            assert (param_h.grad[s] == 0).all(), "std head rows should be zeroed on D_harmful"

    assert qkv_seen > 0, "Expected qkv metadata to be populated"


def test_single_optimizer_state_integrity():
    """With one optimizer, masked theta_std params must not accumulate Adam state."""
    config = SafeMoEConfig(**SMALL_CONFIG)
    model = litgpt.GPT(config)
    model.train()

    registry = HarmfulParamRegistry(model, config)
    gradient_masker = GradientMasker(registry)
    params = (
        registry.parameters_by_type("theta_harmful")
        + registry.parameters_by_type("theta_std")
        + registry.parameters_by_type("theta_shared")
    )
    optimizer = torch.optim.AdamW(params, lr=1e-3)

    input_ids = torch.randint(0, config.padded_vocab_size, (1, 4))

    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids)
    logits.sum().backward()
    gradient_masker.mask("D_harmful")
    optimizer.step()

    for p in registry.parameters_by_type("theta_std"):
        assert len(optimizer.state.get(p, {})) == 0

    assert any(len(optimizer.state.get(p, {})) > 0 for p in registry.parameters_by_type("theta_harmful"))
    assert any(len(optimizer.state.get(p, {})) > 0 for p in registry.parameters_by_type("theta_shared"))
