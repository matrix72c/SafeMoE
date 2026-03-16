"""SafeMoEConfig: a Config subclass that wires SafeMoELayer into MoE transformer blocks.

Adds three harmful-expert tracking fields and overrides the mlp_class property so
that any litgpt.GPT model built from this config uses SafeMoELayer instead of
plain LLaMAMoE.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from litgpt.config import Config as BaseConfig


@dataclass
class SafeMoEConfig(BaseConfig):
    """Extension of litgpt.Config that designates a subset of MoE experts as
    harmful and routes the model's block construction through SafeMoELayer.

    Fields:
        harmful_expert_indices: Indices of experts that will contain harmful
            knowledge and can be zeroed out at inference time.
        harmful_attn_heads: Attention head indices considered harmful (reserved
            for future masking primitives).
        num_harmful_experts: Expected count of harmful experts (informational;
            does not have to equal len(harmful_expert_indices)).
    """

    harmful_expert_indices: list = field(default_factory=list)
    harmful_attn_heads: list = field(default_factory=list)
    num_harmful_experts: int = 0

    @property
    def mlp_class(self) -> type:
        """Return SafeMoELayer when mlp_class_name is 'LLaMAMoE'; fall back to
        the standard litgpt mapping for all other MLP class names."""
        if self.mlp_class_name == "LLaMAMoE":
            from safemoe.model import SafeMoELayer  # lazy import avoids circular dependency
            return SafeMoELayer
        return super().mlp_class
