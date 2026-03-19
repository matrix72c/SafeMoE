from __future__ import annotations

import json
from pathlib import Path

import torch
from litgpt.model import GPT

from safemoe.interventions.manifest import InterventionManifest, derived_router_column_pairs
from safemoe.interventions.surgery import (
    EXPERT_WEIGHT_NAMES,
    _head_row_slice,
    _qkv_weight_name,
    _randn_like,
    _expert_weight_name,
    load_safemoe_config,
    _router_weight_name,
)


def _raise_or_record(
    *,
    condition: bool,
    mismatches: list[str],
    message: str,
) -> None:
    if not condition:
        mismatches.append(message)


def verify_intervention_output(
    *,
    base_checkpoint_dir: Path,
    output_checkpoint_dir: Path,
    manifest: InterventionManifest,
) -> dict:
    base_checkpoint = torch.load(base_checkpoint_dir / "lit_model.pth", map_location="cpu", weights_only=False)
    output_checkpoint = torch.load(output_checkpoint_dir / "lit_model.pth", map_location="cpu", weights_only=False)

    output_config = load_safemoe_config(output_checkpoint_dir / "model_config.yaml")
    model = GPT(output_config)
    model.load_state_dict(output_checkpoint["model"])

    generator = torch.Generator(device="cpu").manual_seed(manifest.seed)
    mismatches: list[str] = []
    checks: list[str] = ["reload"]

    _raise_or_record(
        condition=output_config.harmful_expert_indices == manifest.target_harmful_expert_indices,
        mismatches=mismatches,
        message="harmful_expert_indices do not match manifest targets",
    )
    _raise_or_record(
        condition=output_config.harmful_attn_heads == manifest.target_harmful_attn_heads,
        mismatches=mismatches,
        message="harmful_attn_heads do not match manifest targets",
    )
    _raise_or_record(
        condition=output_config.num_harmful_experts == len(manifest.target_harmful_expert_indices),
        mismatches=mismatches,
        message="num_harmful_experts does not match manifest targets",
    )

    for layer_idx in range(output_config.n_layer):
        for source_expert, target_expert in manifest.expert_pairs:
            for weight_name in EXPERT_WEIGHT_NAMES:
                source_name = _expert_weight_name(layer_idx, source_expert, weight_name)
                target_name = _expert_weight_name(layer_idx, target_expert, weight_name)
                source_tensor = base_checkpoint["model"][source_name]
                expected = source_tensor + _randn_like(
                    source_tensor, generator=generator, scale=manifest.noise_scale
                )
                actual = output_checkpoint["model"][target_name]
                checks.append(target_name)
                _raise_or_record(
                    condition=actual.shape == expected.shape,
                    mismatches=mismatches,
                    message=f"{target_name} shape mismatch",
                )
                _raise_or_record(
                    condition=torch.equal(actual, expected),
                    mismatches=mismatches,
                    message=f"{target_name} value mismatch",
                )

        gate_name = _router_weight_name(layer_idx)
        base_gate = base_checkpoint["model"][gate_name]
        output_gate = output_checkpoint["model"][gate_name]
        for source_expert, target_expert in derived_router_column_pairs(manifest):
            source_column = base_gate[:, source_expert]
            expected = source_column + _randn_like(
                source_column, generator=generator, scale=manifest.noise_scale
            )
            actual = output_gate[:, target_expert]
            checks.append(f"{gate_name}[:, {target_expert}]")
            _raise_or_record(
                condition=actual.shape == expected.shape,
                mismatches=mismatches,
                message=f"{gate_name}[:, {target_expert}] shape mismatch",
            )
            _raise_or_record(
                condition=torch.equal(actual, expected),
                mismatches=mismatches,
                message=f"{gate_name}[:, {target_expert}] value mismatch",
            )

        qkv_name = _qkv_weight_name(layer_idx)
        base_qkv = base_checkpoint["model"][qkv_name]
        output_qkv = output_checkpoint["model"][qkv_name]
        for source_head, target_head in manifest.head_pairs:
            for kind in ("q", "k", "v"):
                source_slice = _head_row_slice(kind, source_head, output_config)
                target_slice = _head_row_slice(kind, target_head, output_config)
                source_rows = base_qkv[source_slice]
                expected = source_rows + _randn_like(
                    source_rows, generator=generator, scale=manifest.noise_scale
                )
                actual = output_qkv[target_slice]
                checks.append(f"{qkv_name}[{kind}:{target_head}]")
                _raise_or_record(
                    condition=actual.shape == expected.shape,
                    mismatches=mismatches,
                    message=f"{qkv_name}[{kind}:{target_head}] shape mismatch",
                )
                _raise_or_record(
                    condition=torch.equal(actual, expected),
                    mismatches=mismatches,
                    message=f"{qkv_name}[{kind}:{target_head}] value mismatch",
                )

    report = {
        "ok": not mismatches,
        "reload_ok": True,
        "expert_pairs": [list(pair) for pair in manifest.expert_pairs],
        "head_pairs": [list(pair) for pair in manifest.head_pairs],
        "derived_router_column_pairs": [list(pair) for pair in derived_router_column_pairs(manifest)],
        "checks": checks,
        "mismatches": mismatches,
    }
    (output_checkpoint_dir / "verification_report.json").write_text(json.dumps(report, indent=2) + "\n")
    result_line = "Result: PASS" if report["ok"] else "Result: FAIL"
    (output_checkpoint_dir / "verification_summary.md").write_text(
        "# Checkpoint Surgery Verification\n\n"
        f"{result_line}\n"
    )
    if not report["ok"]:
        raise ValueError("Checkpoint surgery verification failed")
    return report
