"""safemoe/masking.py — HarmfulParamRegistry and masker implementations.

HarmfulParamRegistry scans all model parameters at construction time and
classifies each one as either theta_harmful or theta_std:

  * theta_harmful: full nn.Parameter objects for expert weights whose expert
    index is in config.harmful_expert_indices (pattern:
    "transformer.h.<layer>.mlp.experts.<idx>.*").
  * theta_std: everything else, including qkv.weight (full parameter, not
    sliced — Phase 2 maskers do not touch attention heads).

QKV row-slice metadata is stored separately in _qkv_harmful_metadata for
Phase 3 per-head gradient masking, but the qkv.weight Parameter itself is
always in theta_std (not in theta_harmful).

GradientMasker: sets theta_std gradients to None after D_harmful backward
(post-backward zeroing, not detach-in-forward; .grad=None prevents Adam state
accumulation per MASK-01 spec).

ActivationMasker: flag-based approach — enable()/disable() set
_activation_masking_enabled on every SafeMoELayer in the model. SafeMoELayer
forward() checks the flag and skips harmful expert accumulation (MASK-02,
MASK-04).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from safemoe.config import SafeMoEConfig


class HarmfulParamRegistry:
    """Classifies all model parameters into theta_harmful and theta_std.

    Parameters
    ----------
    model:
        A litgpt.GPT (or any nn.Module whose named_parameters() follow the
        transformer.h.{layer}.mlp.experts.{idx}.* naming convention).
    config:
        A SafeMoEConfig instance.  The registry reads
        ``config.harmful_expert_indices`` and ``config.harmful_attn_heads``.

    Raises
    ------
    ValueError
        If any parameter is classified into both sets (overlap) or if any
        parameter is left unclassified (not exhaustive).
    """

    # Matches "transformer.h.<layer>.mlp.experts.<idx>.<anything>"
    _EXPERT_RE = re.compile(r"transformer\.h\.\d+\.mlp\.experts\.(\d+)\.")
    # Matches "transformer.h.<layer>.attn.qkv.weight" exactly
    _QKV_RE = re.compile(r"transformer\.h\.(\d+)\.attn\.qkv\.weight$")

    def __init__(self, model: nn.Module, config: "SafeMoEConfig") -> None:
        harmful_expert_indices = set(config.harmful_expert_indices)
        harmful_attn_heads = list(getattr(config, "harmful_attn_heads", []))

        theta_harmful: list[nn.Parameter] = []
        theta_std: list[nn.Parameter] = []
        # Phase 3 uses this list to zero only the harmful-head rows of qkv.weight
        # during gradient masking.  Each entry: (qkv_weight_param, [slice, ...]).
        qkv_harmful_metadata: list[tuple[nn.Parameter, list[slice]]] = []

        for name, param in model.named_parameters():
            m_expert = self._EXPERT_RE.match(name)
            m_qkv = self._QKV_RE.match(name)

            if m_expert and int(m_expert.group(1)) in harmful_expert_indices:
                # Full expert parameter — belongs exclusively to theta_harmful.
                theta_harmful.append(param)

            elif m_qkv and harmful_attn_heads:
                # qkv.weight goes into theta_std so the Phase 2 maskers never
                # touch it.  We also store row-slice metadata for Phase 3.
                n_head = config.n_head
                n_query_groups = config.n_query_groups
                head_size = config.head_size

                slices: list[slice] = []
                for head_idx in harmful_attn_heads:
                    # Q rows for head i (MHA layout)
                    q_start = head_idx * head_size
                    slices.append(slice(q_start, q_start + head_size))
                    # K/V rows use n_query_groups stride (handles MHA and GQA)
                    kv_head = head_idx % n_query_groups
                    k_start = n_head * head_size + kv_head * head_size
                    slices.append(slice(k_start, k_start + head_size))
                    v_start = (n_head + n_query_groups) * head_size + kv_head * head_size
                    slices.append(slice(v_start, v_start + head_size))

                qkv_harmful_metadata.append((param, slices))
                theta_std.append(param)

            else:
                theta_std.append(param)

        # --- Correctness validation using id() (avoids tensor __eq__ pitfall) ---
        harmful_ids = {id(p) for p in theta_harmful}
        std_ids = {id(p) for p in theta_std}
        all_ids = {id(p) for _, p in model.named_parameters()}

        overlap = harmful_ids & std_ids
        if overlap:
            raise ValueError(
                "HarmfulParamRegistry: theta_harmful and theta_std overlap — "
                f"{len(overlap)} parameter(s) doubly-classified."
            )

        missing = all_ids - (harmful_ids | std_ids)
        if missing:
            raise ValueError(
                "HarmfulParamRegistry: not all parameters classified — "
                f"{len(missing)} parameter(s) left unclassified."
            )

        self._registry: dict[str, list[nn.Parameter]] = {
            "theta_harmful": theta_harmful,
            "theta_std": theta_std,
        }
        # Stored for Phase 3 per-head gradient masking on qkv.weight rows.
        # Each entry: (qkv_weight_param, list_of_row_slices_for_harmful_heads).
        # Empty list when config.harmful_attn_heads is [].
        self._qkv_harmful_metadata: list[tuple[nn.Parameter, list[slice]]] = (
            qkv_harmful_metadata
        )

    def parameters_by_type(self, split: str) -> list[nn.Parameter]:
        """Return the parameter list for the given split name.

        Parameters
        ----------
        split:
            Either ``'theta_harmful'`` or ``'theta_std'``.

        Raises
        ------
        KeyError
            If *split* is not one of the two recognised names.
        """
        if split not in self._registry:
            raise KeyError(
                f"Unknown split {split!r}; expected 'theta_harmful' or 'theta_std'"
            )
        return self._registry[split]


# ---------------------------------------------------------------------------
# Maskers — implemented in Plan 04
# ---------------------------------------------------------------------------


class GradientMasker:
    """Zeros theta_std gradients after a D_harmful backward pass.

    Call ``mask()`` immediately after ``loss.backward()`` on a D_harmful batch.
    Sets ``p.grad = None`` (not ``p.grad.zero_()``) for every theta_std
    parameter so the downstream AdamW optimizer never accumulates momentum
    state for those parameters on D_harmful steps (MASK-01).
    """

    def __init__(self, registry: HarmfulParamRegistry) -> None:
        self._registry = registry

    def mask(self) -> None:
        """Call after loss.backward() on a D_harmful batch.

        Sets theta_std parameter gradients to None so only theta_harmful
        parameters receive gradient updates. Post-backward zeroing (not
        detach-in-forward) per MASK-01 spec. Setting .grad = None (not
        .grad.zero_()) prevents Adam momentum accumulation for theta_std.
        """
        for p in self._registry.parameters_by_type("theta_std"):
            p.grad = None


class ActivationMasker:
    """Toggles activation masking on every SafeMoELayer in the model.

    Flag-based design (MASK-02, Pitfall 4 resolution): enable()/disable() set
    ``_activation_masking_enabled`` on each SafeMoELayer instance. SafeMoELayer
    checks this flag in its forward() and skips harmful expert accumulation when
    True. This avoids the register_forward_hook approach, which cannot
    un-aggregate already-summed expert outputs.

    Parameters
    ----------
    model:
        The nn.Module (typically a litgpt.GPT) containing SafeMoELayer instances.
    registry:
        Optional HarmfulParamRegistry (not used by enable/disable — retained for
        API symmetry with GradientMasker and future Phase 3 extensions).
    """

    def __init__(
        self, model: nn.Module, registry: "HarmfulParamRegistry | None" = None
    ) -> None:
        self._registry = registry
        from safemoe.model import SafeMoELayer

        self._layers: list[SafeMoELayer] = [
            m for m in model.modules() if isinstance(m, SafeMoELayer)
        ]

    def enable(self) -> None:
        """Enable activation masking on all SafeMoELayer instances.

        Call before the D_std forward pass. Sets
        ``layer._activation_masking_enabled = True`` on every SafeMoELayer,
        causing those layers to skip harmful expert contributions in forward().
        """
        for layer in self._layers:
            layer._activation_masking_enabled = True

    def disable(self) -> None:
        """Disable activation masking on all SafeMoELayer instances.

        Call after the D_std forward pass. Restores
        ``layer._activation_masking_enabled = False`` on every SafeMoELayer
        so subsequent forward passes use normal expert dispatch.
        """
        for layer in self._layers:
            layer._activation_masking_enabled = False
