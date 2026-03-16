"""
RED test stubs for SafeMoEConfig (MOE-01).

These tests will fail with ImportError until safemoe/config.py is implemented
in plan 02-02. This is the intended RED state.
"""

import pytest

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


def test_safemoe_config_has_harmful_fields():
    """SafeMoEConfig stores harmful field values passed at construction."""
    config = SafeMoEConfig(
        harmful_expert_indices=[0, 1],
        num_harmful_experts=2,
        harmful_attn_heads=[],
    )
    assert config.harmful_expert_indices == [0, 1]
    assert config.num_harmful_experts == 2
    assert config.harmful_attn_heads == []


def test_safemoe_config_defaults_safe():
    """SafeMoEConfig() with no harmful args defaults to empty/zero values."""
    config = SafeMoEConfig()
    assert config.harmful_expert_indices == []
    assert config.num_harmful_experts == 0
    assert config.harmful_attn_heads == []


def test_safemoe_config_inherits_litgpt_config():
    """SafeMoEConfig is a subclass of litgpt.Config."""
    config = SafeMoEConfig()
    assert isinstance(config, litgpt.Config)


def test_safemoe_config_mlp_class_returns_safemoe_layer():
    """SafeMoEConfig.mlp_class returns SafeMoELayer when mlp_class_name='LLaMAMoE'."""
    config = SafeMoEConfig(mlp_class_name="LLaMAMoE")
    assert config.mlp_class is SafeMoELayer
