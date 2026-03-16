"""
RED test stubs for SafeMoELayer structure and harmful expert initialization (MOE-03, MOE-04).

These tests will fail with ImportError until safemoe/config.py and safemoe/model.py
are implemented in plans 02-02. This is the intended RED state.
"""

import pytest
import torch

# These imports will raise ImportError until implementation plans run.
# That ImportError is the expected RED state for this plan.
from safemoe.config import SafeMoEConfig
from safemoe.model import SafeMoELayer

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


def test_safemoe_layer_structure():
    """
    GPT built with SafeMoEConfig has SafeMoELayer as the mlp in every block,
    and each SafeMoELayer has n_expert=4 experts in self.experts.
    """
    config = SafeMoEConfig(**SMALL_CONFIG)
    model = litgpt.GPT(config)

    for i, block in enumerate(model.transformer.h):
        assert isinstance(block.mlp, SafeMoELayer), (
            f"Block {i}: expected SafeMoELayer, got {type(block.mlp)}"
        )
        assert len(block.mlp.experts) == SMALL_CONFIG["n_expert"], (
            f"Block {i}: expected {SMALL_CONFIG['n_expert']} experts, "
            f"got {len(block.mlp.experts)}"
        )


def test_harmful_expert_init_random():
    """
    SafeMoELayer built with init_strategy='random' has harmful expert weights
    that differ from standard expert weights (random init produces distinct values).
    """
    config = SafeMoEConfig(**SMALL_CONFIG)
    layer = SafeMoELayer(config, init_strategy="random")

    # Harmful expert at index 0; standard expert at index 2
    harmful_weight = layer.experts[0].fc_1.weight
    std_weight = layer.experts[2].fc_1.weight

    assert not torch.allclose(harmful_weight, std_weight), (
        "Random init should produce different weights for harmful vs std experts"
    )


def test_harmful_expert_init_copy():
    """
    SafeMoELayer built with init_strategy='copy' has harmful expert weights
    that match the nearest standard expert's weights (copy produces identical weights).
    """
    config = SafeMoEConfig(**SMALL_CONFIG)
    layer = SafeMoELayer(config, init_strategy="copy")

    # Harmful expert at index 0 should be a copy of the nearest std expert.
    # The nearest std expert is the first non-harmful expert (index 2).
    harmful_weight = layer.experts[0].fc_1.weight
    # Find first standard expert (index not in harmful_expert_indices)
    first_std_idx = next(
        i for i in range(SMALL_CONFIG["n_expert"])
        if i not in SMALL_CONFIG["harmful_expert_indices"]
    )
    std_weight = layer.experts[first_std_idx].fc_1.weight

    assert torch.allclose(harmful_weight, std_weight), (
        "Copy init should produce identical weights for harmful vs std experts"
    )


def test_activation_masking_flag_exists():
    """
    SafeMoELayer instance has _activation_masking_enabled attribute, default False.
    """
    config = SafeMoEConfig(**SMALL_CONFIG)
    layer = SafeMoELayer(config)

    assert hasattr(layer, "_activation_masking_enabled"), (
        "SafeMoELayer must have _activation_masking_enabled attribute"
    )
    assert layer._activation_masking_enabled is False, (
        "_activation_masking_enabled should default to False"
    )
