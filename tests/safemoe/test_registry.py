"""
RED test stubs for HarmfulParamRegistry (MOE-02, MASK-03).

These tests will fail with ImportError until safemoe/masking.py is implemented
in plan 02-03. This is the intended RED state.
"""

import pytest

# These imports will raise ImportError until implementation plans run.
# That ImportError is the expected RED state for this plan.
from safemoe.masking import HarmfulParamRegistry
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


def test_registry_theta_harmful_contains_expert_params():
    """
    All parameters matching transformer.h.*.mlp.experts.{0,1}.* are in theta_harmful.
    Expert indices 0 and 1 are the designated harmful experts per SMALL_CONFIG.
    """
    config = SafeMoEConfig(**SMALL_CONFIG)
    model = litgpt.GPT(config)
    registry = HarmfulParamRegistry(model, config)

    theta_harmful_ids = {id(p) for p in registry.parameters_by_type("theta_harmful")}

    for name, param in model.named_parameters():
        for idx in config.harmful_expert_indices:
            if f".mlp.experts.{idx}." in name:
                assert id(param) in theta_harmful_ids, (
                    f"Parameter {name!r} (expert {idx}) should be in theta_harmful "
                    f"but is missing from the registry."
                )


def test_registry_exhaustive_coverage():
    """
    theta_harmful union theta_std equals the full set of model parameters.
    Uses id() for identity comparison per RESEARCH.md Pitfall 2.
    """
    config = SafeMoEConfig(**SMALL_CONFIG)
    model = litgpt.GPT(config)
    registry = HarmfulParamRegistry(model, config)

    harmful_ids = {id(p) for p in registry.parameters_by_type("theta_harmful")}
    std_ids = {id(p) for p in registry.parameters_by_type("theta_std")}
    all_param_ids = {id(p) for _, p in model.named_parameters()}

    assert harmful_ids | std_ids == all_param_ids, (
        "theta_harmful union theta_std must equal the complete set of model parameters. "
        f"Missing: {all_param_ids - (harmful_ids | std_ids)}"
    )


def test_registry_non_overlapping():
    """
    theta_harmful and theta_std are disjoint (no parameter appears in both sets).
    Uses id() for identity comparison per RESEARCH.md Pitfall 2.
    """
    config = SafeMoEConfig(**SMALL_CONFIG)
    model = litgpt.GPT(config)
    registry = HarmfulParamRegistry(model, config)

    harmful_ids = {id(p) for p in registry.parameters_by_type("theta_harmful")}
    std_ids = {id(p) for p in registry.parameters_by_type("theta_std")}

    overlap = harmful_ids & std_ids
    assert overlap == set(), (
        f"theta_harmful and theta_std overlap on {len(overlap)} parameter(s). "
        "Each parameter must appear in exactly one group."
    )


def test_registry_raises_on_invalid_config():
    """
    Constructing HarmfulParamRegistry with a valid config succeeds without error.
    (The contract: ValueError is raised for invalid configs; valid construction is error-free.)
    """
    config = SafeMoEConfig(**SMALL_CONFIG)
    model = litgpt.GPT(config)

    # Valid construction must NOT raise
    try:
        registry = HarmfulParamRegistry(model, config)
    except ValueError as exc:
        pytest.fail(
            f"HarmfulParamRegistry raised ValueError on a valid config: {exc}"
        )


def test_param_groups_disjoint():
    """
    MASK-03 traceability: parameters_by_type('theta_harmful') and
    parameters_by_type('theta_std') are disjoint by parameter identity.
    Equivalent to test_registry_non_overlapping, named for MASK-03 traceability.
    """
    config = SafeMoEConfig(**SMALL_CONFIG)
    model = litgpt.GPT(config)
    registry = HarmfulParamRegistry(model, config)

    harmful_ids = {id(p) for p in registry.parameters_by_type("theta_harmful")}
    std_ids = {id(p) for p in registry.parameters_by_type("theta_std")}

    assert harmful_ids.isdisjoint(std_ids), (
        "Dual AdamW param groups require disjoint parameter sets. "
        "theta_harmful and theta_std must not share any parameters."
    )
