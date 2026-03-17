"""safemoe/masking.py — HarmfulParamRegistry and masker implementations.

HarmfulParamRegistry scans all model parameters at construction time and
classifies each one as either theta_harmful, theta_std, or theta_shared:

  * theta_harmful: full nn.Parameter objects for harmful expert weights whose
    expert index is in config.harmful_expert_indices (pattern:
    "transformer.h.<layer>.mlp.experts.<idx>.*").
  * theta_std: full nn.Parameter objects for standard expert weights, i.e.
    expert indices not in config.harmful_expert_indices.
  * theta_shared: everything else, including embeddings, router/gate weights,
    norms, and lm_head.

Fused attention qkv weights are a special case. The full ``qkv.weight`` tensor
cannot be split into separate Parameters, so when ``harmful_attn_heads`` is
configured the registry stores row-slice metadata for both harmful and standard
attention heads. The full qkv Parameter remains in theta_std for exhaustive
coverage, while GradientMasker applies per-slice masking so harmful and std
heads still follow split-specific update rules.

GradientMasker is split-aware: after backward it clears only the expert-specific
group that should stay frozen for the current split, and it also zeros the
inactive qkv row slices while leaving theta_shared gradients intact so shared
parameters learn from both D_std and D_harmful.

ActivationMasker: flag-based approach — enable()/disable() set
_activation_masking_enabled on every SafeMoELayer in the model. SafeMoELayer
forward() checks the flag and skips harmful expert accumulation (MASK-02,
MASK-04).
"""

from __future__ import annotations

import re
import types
from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from safemoe.config import SafeMoEConfig
    from litgpt.model import CausalSelfAttention


class HarmfulParamRegistry:
    """Classifies all model parameters into theta_harmful, theta_std, and theta_shared.

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
        theta_shared: list[nn.Parameter] = []
        qkv_harmful_metadata: list[tuple[nn.Parameter, list[slice]]] = []
        qkv_std_metadata: list[tuple[nn.Parameter, list[slice]]] = []

        def build_qkv_slices(head_indices: list[int]) -> list[slice]:
            """Build deduplicated qkv row slices for the given head indices."""
            seen: set[tuple[int, int]] = set()
            slices: list[slice] = []
            for head_idx in head_indices:
                q_start = head_idx * config.head_size
                q_slice = slice(q_start, q_start + config.head_size)
                q_key = (q_slice.start, q_slice.stop)
                if q_key not in seen:
                    seen.add(q_key)
                    slices.append(q_slice)

                kv_head = head_idx % config.n_query_groups
                k_start = config.n_head * config.head_size + kv_head * config.head_size
                k_slice = slice(k_start, k_start + config.head_size)
                k_key = (k_slice.start, k_slice.stop)
                if k_key not in seen:
                    seen.add(k_key)
                    slices.append(k_slice)

                v_start = (config.n_head + config.n_query_groups) * config.head_size + kv_head * config.head_size
                v_slice = slice(v_start, v_start + config.head_size)
                v_key = (v_slice.start, v_slice.stop)
                if v_key not in seen:
                    seen.add(v_key)
                    slices.append(v_slice)
            return slices

        for name, param in model.named_parameters():
            # FSDP with use_orig_params=True embeds "_fsdp_wrapped_module." in
            # parameter paths (e.g. "transformer.h.0._fsdp_wrapped_module.mlp…").
            # Strip every occurrence so the expert/qkv regexes can match.
            clean = name.replace("_fsdp_wrapped_module.", "")
            m_expert = self._EXPERT_RE.match(clean)
            m_qkv = self._QKV_RE.match(clean)
            if m_expert:
                expert_idx = int(m_expert.group(1))
                if expert_idx in harmful_expert_indices:
                    theta_harmful.append(param)
                else:
                    theta_std.append(param)
            elif m_qkv and harmful_attn_heads:
                all_heads = list(range(config.n_head))
                std_attn_heads = [idx for idx in all_heads if idx not in harmful_attn_heads]
                qkv_harmful_metadata.append((param, build_qkv_slices(harmful_attn_heads)))
                qkv_std_metadata.append((param, build_qkv_slices(std_attn_heads)))
                theta_std.append(param)
            else:
                theta_shared.append(param)

        # --- Correctness validation using id() (avoids tensor __eq__ pitfall) ---
        harmful_ids = {id(p) for p in theta_harmful}
        std_ids = {id(p) for p in theta_std}
        shared_ids = {id(p) for p in theta_shared}
        all_ids = {id(p) for _, p in model.named_parameters()}

        overlap = (harmful_ids & std_ids) | (harmful_ids & shared_ids) | (std_ids & shared_ids)
        if overlap:
            raise ValueError(
                "HarmfulParamRegistry: parameter groups overlap — "
                f"{len(overlap)} parameter(s) doubly-classified."
            )

        missing = all_ids - (harmful_ids | std_ids | shared_ids)
        if missing:
            raise ValueError(
                "HarmfulParamRegistry: not all parameters classified — "
                f"{len(missing)} parameter(s) left unclassified."
            )

        self._registry: dict[str, list[nn.Parameter]] = {
            "theta_harmful": theta_harmful,
            "theta_std": theta_std,
            "theta_shared": theta_shared,
        }
        self._qkv_harmful_metadata: list[tuple[nn.Parameter, list[slice]]] = qkv_harmful_metadata
        self._qkv_std_metadata: list[tuple[nn.Parameter, list[slice]]] = qkv_std_metadata

    def parameters_by_type(self, split: str) -> list[nn.Parameter]:
        """Return the parameter list for the given split name.

        Parameters
        ----------
        split:
            One of ``'theta_harmful'``, ``'theta_std'``, or ``'theta_shared'``.

        Raises
        ------
        KeyError
            If *split* is not one of the two recognised names.
        """
        if split not in self._registry:
            raise KeyError(
                "Unknown split "
                f"{split!r}; expected 'theta_harmful', 'theta_std', or 'theta_shared'"
            )
        return self._registry[split]


# ---------------------------------------------------------------------------
# Helper: forward wrapper stub for CausalSelfAttention (Phase 3 infrastructure)
# ---------------------------------------------------------------------------


def _wrap_attn_forward(attn: "CausalSelfAttention") -> None:
    """Install a forward wrapper on a CausalSelfAttention instance.

    This wrapper stub establishes the monkey-patch infrastructure needed for
    Phase 3 attn-head activation masking.  The actual head-output zeroing
    (``attn_out[:, head_idx, :, :] = 0`` before proj) is added in Plan 03-02
    Task 1 via SafeCausalSelfAttention.

    The ``_activation_masking_enabled`` flag toggled by
    ``ActivationMasker.enable()/disable()`` is checked here; for now the
    wrapper is a pass-through that preserves the original forward behaviour.
    """
    orig_forward = attn.forward  # stash the current bound method

    def _masked_forward(self_attn, *args, **kwargs):  # type: ignore[no-untyped-def]
        # Phase 03-02 replaces this with real head-output zeroing.
        # For Plan 03-01 this is a transparent pass-through.
        return orig_forward(*args, **kwargs)

    attn.forward = types.MethodType(_masked_forward, attn)


# ---------------------------------------------------------------------------
# Maskers
# ---------------------------------------------------------------------------


class GradientMasker:
    """Clears expert-specific gradients that should stay frozen for a split.

    With the single-optimizer setup, this masker enforces split semantics
    post-backward by setting the inactive expert group's gradients to None while
    keeping theta_shared gradients intact.
    """

    def __init__(self, registry: HarmfulParamRegistry) -> None:
        self._registry = registry
        self._qkv_param_ids: set[int] = {
            id(param) for param, _ in registry._qkv_harmful_metadata
        }

    def mask(self, split_label: str) -> None:
        """Call after loss.backward() to preserve the active parameter groups.

        ``D_std`` keeps theta_std + theta_shared and clears theta_harmful.
        Harmful qkv head rows are also zeroed so only std heads update.

        ``D_harmful`` keeps theta_harmful + theta_shared and clears theta_std.
        For fused qkv weights, std head rows are zeroed while harmful rows are
        preserved, so attention heads also follow the harmful/std split.

        ``D_unlabeled`` updates all groups and therefore requires no masking.
        """
        if split_label == "D_std":
            for p in self._registry.parameters_by_type("theta_harmful"):
                p.grad = None
            for param, slices in self._registry._qkv_harmful_metadata:
                if param.grad is not None:
                    for s in slices:
                        param.grad[s] = 0
        elif split_label == "D_harmful":
            for p in self._registry.parameters_by_type("theta_std"):
                if id(p) not in self._qkv_param_ids:
                    p.grad = None
            for param, slices in self._registry._qkv_std_metadata:
                if param.grad is not None:
                    for s in slices:
                        param.grad[s] = 0
        elif split_label == "D_unlabeled":
            return
        else:
            raise ValueError(f"Unknown split_label {split_label!r}")


class ActivationMasker:
    """Toggles activation masking on every SafeMoELayer and CausalSelfAttention
    in the model.

    Flag-based design (MASK-02, Pitfall 4 resolution): enable()/disable() set
    ``_activation_masking_enabled`` on each SafeMoELayer instance. SafeMoELayer
    checks this flag in its forward() and skips harmful expert accumulation when
    True. This avoids the register_forward_hook approach, which cannot
    un-aggregate already-summed expert outputs.

    Phase 3 extension: also collects CausalSelfAttention instances and
    monkey-patches ``_activation_masking_enabled`` and ``_harmful_heads`` onto
    each.  The forward wrapper (``_wrap_attn_forward``) installed here is a
    pass-through stub; the actual head-output zeroing is added in Plan 03-02.

    Parameters
    ----------
    model:
        The nn.Module (typically a litgpt.GPT) containing SafeMoELayer
        and CausalSelfAttention instances.
    registry:
        Optional HarmfulParamRegistry (not used by enable/disable — retained
        for API symmetry with GradientMasker).
    config:
        Optional SafeMoEConfig.  When provided and ``config.harmful_attn_heads``
        is non-empty, CausalSelfAttention instances are collected and the
        per-head activation-masking infrastructure is installed.
    """

    def __init__(
        self,
        model: nn.Module,
        registry: "HarmfulParamRegistry | None" = None,
        config: "SafeMoEConfig | None" = None,
    ) -> None:
        self._registry = registry
        from safemoe.model import SafeMoELayer

        self._layers: list[SafeMoELayer] = [
            m for m in model.modules() if isinstance(m, SafeMoELayer)
        ]

        # Phase 3: collect CausalSelfAttention instances and install flag infrastructure
        from litgpt.model import CausalSelfAttention

        harmful_heads: list[int] = (
            list(getattr(config, "harmful_attn_heads", [])) if config else []
        )
        self._attn_layers: list[CausalSelfAttention] = []
        if harmful_heads:
            for m in model.modules():
                if isinstance(m, CausalSelfAttention):
                    m._activation_masking_enabled = False  # type: ignore[attr-defined]
                    m._harmful_heads = harmful_heads  # type: ignore[attr-defined]
                    _wrap_attn_forward(m)
                    self._attn_layers.append(m)

    def enable(self) -> None:
        """Enable activation masking on all SafeMoELayer and attn instances.

        Call before the D_std forward pass. Sets
        ``layer._activation_masking_enabled = True`` on every SafeMoELayer,
        causing those layers to skip harmful expert contributions in forward().
        Also sets the flag on every CausalSelfAttention in self._attn_layers.
        """
        for layer in self._layers:
            layer._activation_masking_enabled = True
        for layer in self._attn_layers:
            layer._activation_masking_enabled = True  # type: ignore[attr-defined]

    def disable(self) -> None:
        """Disable activation masking on all SafeMoELayer and attn instances.

        Call after the D_std forward pass. Restores
        ``layer._activation_masking_enabled = False`` on every SafeMoELayer
        and every CausalSelfAttention in self._attn_layers so subsequent
        forward passes use normal dispatch.
        """
        for layer in self._layers:
            layer._activation_masking_enabled = False
        for layer in self._attn_layers:
            layer._activation_masking_enabled = False  # type: ignore[attr-defined]
