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

import json
import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch.nn as nn

if TYPE_CHECKING:
    from litgpt.config import Config


class HarmfulParamRegistry:
    """Classifies all model parameters into theta_harmful, theta_std, and theta_shared.

    Parameters
    ----------
    model:
        A litgpt.GPT (or any nn.Module whose named_parameters() follow the
        transformer.h.{layer}.mlp.experts.{idx}.* naming convention).
    config:
        A Config instance.  The registry reads
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

    def __init__(self, model: nn.Module, config: "Config") -> None:
        self._model = model
        self._config = config
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

    def _ownership_by_param_id(self) -> dict[int, str]:
        ownership_by_id: dict[int, str] = {}
        for ownership, params in self._registry.items():
            for param in params:
                ownership_by_id[id(param)] = ownership
        return ownership_by_id

    @staticmethod
    def _clean_parameter_name(name: str) -> str:
        return name.replace("_fsdp_wrapped_module.", "")

    @staticmethod
    def _category_for_name(name: str) -> str:
        if ".mlp.experts." in name:
            return "expert"
        if name.endswith(".mlp.gate.weight"):
            return "router_gate"
        if name.endswith("attn.qkv.weight"):
            return "attn_qkv_full"
        if name == "transformer.wte.weight":
            return "embedding"
        return "other"

    @staticmethod
    def _slice_rows(slices: list[slice]) -> list[list[int]]:
        return [[int(s.start), int(s.stop)] for s in slices]

    def registry_inventory(self) -> list[dict[str, Any]]:
        """Return researcher-facing registry rows for every named parameter.

        Full-parameter rows preserve the existing exhaustive, non-overlapping
        ownership contract. Additional ``attn_qkv_slice`` rows expose fused qkv
        harmful/std slice ownership without reclassifying the underlying full
        parameter.
        """
        ownership_by_id = self._ownership_by_param_id()

        inventory: list[dict[str, Any]] = []
        for name, param in self._model.named_parameters():
            clean_name = self._clean_parameter_name(name)
            row: dict[str, Any] = {
                "parameter_name": clean_name,
                "ownership": ownership_by_id[id(param)],
                "category": self._category_for_name(clean_name),
                "shape": list(param.shape),
            }
            inventory.append(row)

        for param, slices in self._qkv_harmful_metadata:
            inventory.append(
                {
                    "parameter_name": self._parameter_name_for(param),
                    "ownership": "theta_harmful",
                    "category": "attn_qkv_slice",
                    "shape": list(param.shape),
                    "slice_role": "harmful",
                    "slice_rows": self._slice_rows(slices),
                }
            )
        for param, slices in self._qkv_std_metadata:
            inventory.append(
                {
                    "parameter_name": self._parameter_name_for(param),
                    "ownership": "theta_std",
                    "category": "attn_qkv_slice",
                    "shape": list(param.shape),
                    "slice_role": "std",
                    "slice_rows": self._slice_rows(slices),
                }
            )
        return inventory

    def _parameter_name_for(self, needle: nn.Parameter) -> str:
        for name, param in self._model.named_parameters():
            if param is needle:
                return self._clean_parameter_name(name)
        raise KeyError("Parameter not found in model.named_parameters()")


def write_registry_reports(
    registry: HarmfulParamRegistry,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write machine-readable and Markdown registry ownership artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory = registry.registry_inventory()
    inventory_path = output_dir / "registry_inventory.json"
    summary_path = output_dir / "registry_summary.md"
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n")

    ownership_counts = Counter(row["ownership"] for row in inventory)
    category_counts = Counter(row["category"] for row in inventory)

    lines = [
        "# Registry Summary",
        "",
        f"Total rows: {len(inventory)}",
        "",
        "## Ownership Counts",
    ]
    for ownership in ("theta_harmful", "theta_std", "theta_shared"):
        lines.append(f"- {ownership}: {ownership_counts.get(ownership, 0)}")
    lines.extend(["", "## Category Counts"])
    for category in sorted(category_counts):
        lines.append(f"- {category}: {category_counts[category]}")

    summary_path.write_text("\n".join(lines) + "\n")
    return inventory_path, summary_path


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
        split_policies = {
            "D_std": {
                "clear_group": "theta_harmful",
                "skip_qkv": False,
                "qkv_metadata": self._registry._qkv_harmful_metadata,
            },
            "D_harmful": {
                "clear_group": "theta_std",
                "skip_qkv": True,
                "qkv_metadata": self._registry._qkv_std_metadata,
            },
            "D_unlabeled": None,
        }
        if split_label not in split_policies:
            raise ValueError(f"Unknown split_label {split_label!r}")
        policy = split_policies[split_label]
        if policy is None:
            return

        for param in self._registry.parameters_by_type(policy["clear_group"]):
            if policy["skip_qkv"] and id(param) in self._qkv_param_ids:
                continue
            param.grad = None
        for param, slices in policy["qkv_metadata"]:
            if param.grad is not None:
                for s in slices:
                    param.grad[s] = 0


class ActivationMasker:
    """Toggles activation masking on SafeMoELayer instances."""

    def __init__(
        self,
        model: nn.Module,
        registry: "HarmfulParamRegistry | None" = None,
        config: "Config | None" = None,
    ) -> None:
        self._registry = registry
        from litgpt.model import SafeMoELayer

        self._layers: list[SafeMoELayer] = [m for m in model.modules() if isinstance(m, SafeMoELayer)]

    def enable(self) -> None:
        """Enable activation masking on all tracked layers."""
        for layer in self._layers:
            layer._activation_masking_enabled = True

    def disable(self) -> None:
        """Disable activation masking on all tracked layers."""
        for layer in self._layers:
            layer._activation_masking_enabled = False
