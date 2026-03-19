"""Shared routing observability helpers for eval and training flows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch.nn as nn

from safemoe.model import SafeMoELayer


def _unwrap_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "module"):
        return model.module
    return model


class RoutingObservabilityCollector:
    """Collect per-split routing counts from SafeMoELayer dispatch indices."""

    def __init__(self, model: nn.Module, harmful_expert_indices: list[int]) -> None:
        self._model = _unwrap_model(model)
        self._harmful_expert_indices = set(harmful_expert_indices)
        self._layers = [module for module in self._model.modules() if isinstance(module, SafeMoELayer)]

    def collect_split(self, split_name: str, run_fn: Callable[[], None]) -> dict:
        dispatch_indices: list[int] = []

        def _hook(module: SafeMoELayer, _inp, _out, _dispatch=dispatch_indices) -> None:
            if hasattr(module, "_last_indices"):
                _dispatch.extend(module._last_indices.detach().reshape(-1).tolist())

        handles = [layer.register_forward_hook(_hook) for layer in self._layers]
        try:
            run_fn()
        finally:
            for handle in handles:
                handle.remove()

        harmful_dispatches = sum(1 for idx in dispatch_indices if idx in self._harmful_expert_indices)
        total_dispatches = len(dispatch_indices)
        return {
            f"dispatch_count_{split_name}": total_dispatches,
            f"routing_harmful_frac_{split_name}": harmful_dispatches / max(total_dispatches, 1),
        }

    def collect_splits(self, split_runners: dict[str, Callable[[], None]]) -> dict:
        metrics: dict = {}
        for split_name, run_fn in split_runners.items():
            if run_fn is None:
                continue
            metrics.update(self.collect_split(split_name, run_fn))
        return metrics


def write_routing_artifacts(output_dir: Path, metrics: dict, markdown_title: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "routing_observability.json"
    json_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))

    split_names = sorted(
        {
            key.removeprefix("dispatch_count_")
            for key in metrics
            if key.startswith("dispatch_count_")
        }
    )
    lines = [f"# {markdown_title}", "", "| Split | Dispatch Count | Harmful Fraction |", "| --- | ---: | ---: |"]
    for split_name in split_names:
        dispatch_count = metrics.get(f"dispatch_count_{split_name}")
        harmful_frac = metrics.get(f"routing_harmful_frac_{split_name}")
        if dispatch_count is None or harmful_frac is None:
            continue
        lines.append(f"| {split_name} | {dispatch_count} | {harmful_frac:.6f} |")

    (output_dir / "routing_observability.md").write_text("\n".join(lines) + "\n")


def assert_routing_parity(logged_metrics: dict, observed_metrics: dict, output_dir: Path) -> None:
    checks = []
    mismatches = []
    keys = sorted(
        {
            key
            for key in set(logged_metrics) | set(observed_metrics)
            if key.startswith("dispatch_count_") or key.startswith("routing_harmful_frac_")
        }
    )

    for key in keys:
        logged_value = logged_metrics.get(key)
        observed_value = observed_metrics.get(key)
        ok = logged_value == observed_value
        check = {
            "key": key,
            "logged": logged_value,
            "observed": observed_value,
            "ok": ok,
        }
        checks.append(check)
        if not ok:
            mismatches.append(check)

    report = {
        "ok": not mismatches,
        "checks": checks,
        "mismatches": mismatches,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "routing_parity.json").write_text(json.dumps(report, indent=2, sort_keys=True))

    if mismatches:
        raise ValueError("Routing parity check failed")
