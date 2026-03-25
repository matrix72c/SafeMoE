"""SafeMoELayer: a LLaMAMoE subclass with activation-masking support.

SafeMoELayer extends LLaMAMoE with two additions:
1. A boolean flag `_activation_masking_enabled` (default False) that, when True,
   causes the forward pass to skip (zero out) contributions from harmful experts.
2. An optional `init_strategy` constructor argument:
   - "random" (default): harmful experts keep their default random initialization.
   - "copy": harmful experts are initialised as deep copies of the first standard
     (non-harmful) expert's weights, so they start identical to a safe expert.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from litgpt.model import LLaMAMoE

if TYPE_CHECKING:
    from safemoe.config import SafeMoEConfig


class SafeMoELayer(LLaMAMoE):
    """LLaMAMoE subclass with per-expert activation masking.

    When ``_activation_masking_enabled`` is True, the forward pass silently
    drops contributions from any expert whose index appears in
    ``_harmful_indices``, so those experts produce exactly zero output.
    """

    def __init__(
        self,
        config: "SafeMoEConfig",
        init_strategy: str = "random",
    ) -> None:
        super().__init__(config)
        self._activation_masking_enabled: bool = False
        self._harmful_indices: list[int] = list(
            getattr(config, "harmful_expert_indices", [])
        )
        self._last_indices: torch.Tensor | None = None
        self._last_harmful_routing_mass: torch.Tensor | None = None
        self._harmful_index_tensor: torch.Tensor | None = None

        if init_strategy == "copy":
            # Find the first expert whose index is NOT in the harmful set.
            std_indices = [
                i
                for i in range(len(self.experts))
                if i not in self._harmful_indices
            ]
            if std_indices:
                src_state = copy.deepcopy(self.experts[std_indices[0]].state_dict())
                for idx in self._harmful_indices:
                    self.experts[idx].load_state_dict(src_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass identical to LLaMAMoE except that, when activation
        masking is enabled, harmful experts contribute exactly zero.
        """
        B, T, C = x.size()
        residual_x = x.clone()
        x_flat = x.view(-1, C)

        if not self.config.n_expert_groups:
            router = self.gate(x_flat)
            probs, indices = torch.topk(router, self.config.n_expert_per_token)
            self._last_indices = indices
            probs = F.softmax(probs, dim=1, dtype=torch.float).to(dtype=x_flat.dtype)
        else:
            probs, indices = self.gate(x_flat)
            self._last_indices = indices

        if self.config.routed_scaling_factor != 1.0:
            probs = probs * self.config.routed_scaling_factor

        if self._harmful_indices:
            harmful_indices = self._harmful_index_tensor
            if harmful_indices is None or harmful_indices.device != indices.device:
                harmful_indices = torch.tensor(self._harmful_indices, device=indices.device)
                self._harmful_index_tensor = harmful_indices
            harmful_mask = (indices.unsqueeze(-1) == harmful_indices).any(dim=-1)
            harmful_mass = (probs * harmful_mask.to(dtype=probs.dtype)).sum(dim=1)
            self._last_harmful_routing_mass = harmful_mass.mean()
        else:
            self._last_harmful_routing_mass = probs.new_zeros(())

        masks = indices.unsqueeze(-1) == torch.arange(
            self.config.n_expert, device=x_flat.device
        )
        masks = masks.permute(2, 0, 1)  # (n_expert, B*T, n_expert_per_token)

        y = torch.zeros_like(x_flat)
        for expert_idx, (mask, expert) in enumerate(zip(masks, self.experts)):
            if self._activation_masking_enabled and expert_idx in self._harmful_indices:
                continue  # skip — zero contribution from this harmful expert
            token_idx, sel_expert_idx = torch.where(mask)
            if token_idx.numel() == 0:
                continue
            y[token_idx] += probs[token_idx, sel_expert_idx, None] * expert(
                x_flat[token_idx]
            )

        y = y.view(B, T, C)
        if self.config.n_shared_expert:
            y = y + self.shared_experts(residual_x)
        return y
