from __future__ import annotations

import json
import shutil
from dataclasses import fields
from pathlib import Path

import torch
from litgpt.utils import save_config
import yaml

from safemoe.config import SafeMoEConfig
from safemoe.interventions.manifest import InterventionManifest, derived_router_column_pairs, save_manifest

SIDECAR_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    "generation_config.json",
)
EXPERT_WEIGHT_NAMES = ("fc_1.weight", "fc_2.weight", "proj.weight")
CONFIG_FIELD_NAMES = {field.name for field in fields(SafeMoEConfig)}


def load_safemoe_config(path: Path) -> SafeMoEConfig:
    payload = yaml.safe_load(Path(path).read_text())
    filtered = {key: value for key, value in payload.items() if key in CONFIG_FIELD_NAMES}
    return SafeMoEConfig(**filtered)


def _randn_like(tensor: torch.Tensor, *, generator: torch.Generator, scale: float) -> torch.Tensor:
    noise = torch.randn(tensor.shape, generator=generator, device="cpu", dtype=tensor.dtype)
    return noise.to(device=tensor.device) * scale


def _expert_weight_name(layer_idx: int, expert_idx: int, weight_name: str) -> str:
    return f"transformer.h.{layer_idx}.mlp.experts.{expert_idx}.{weight_name}"


def _router_weight_name(layer_idx: int) -> str:
    return f"transformer.h.{layer_idx}.mlp.gate.weight"


def _qkv_weight_name(layer_idx: int) -> str:
    return f"transformer.h.{layer_idx}.attn.qkv.weight"


def _head_row_slice(kind: str, head_idx: int, config: SafeMoEConfig) -> slice:
    head_size = config.head_size
    if kind == "q":
        start = head_idx * head_size
    elif kind == "k":
        start = config.n_head * head_size + (head_idx % config.n_query_groups) * head_size
    elif kind == "v":
        start = (config.n_head + config.n_query_groups) * head_size + (
            head_idx % config.n_query_groups
        ) * head_size
    else:
        raise ValueError(f"Unknown QKV slice kind: {kind}")
    return slice(start, start + head_size)


def apply_intervention_manifest(
    state_dict: dict[str, torch.Tensor],
    config: SafeMoEConfig,
    manifest: InterventionManifest,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(manifest.seed)

    for layer_idx in range(config.n_layer):
        for source_expert, target_expert in manifest.expert_pairs:
            for weight_name in EXPERT_WEIGHT_NAMES:
                source_name = _expert_weight_name(layer_idx, source_expert, weight_name)
                target_name = _expert_weight_name(layer_idx, target_expert, weight_name)
                source_tensor = state_dict[source_name].detach().clone()
                state_dict[target_name].copy_(
                    source_tensor + _randn_like(source_tensor, generator=generator, scale=manifest.noise_scale)
                )

        gate_name = _router_weight_name(layer_idx)
        gate = state_dict[gate_name]
        for source_expert, target_expert in derived_router_column_pairs(manifest):
            source_column = gate[:, source_expert].detach().clone()
            gate[:, target_expert].copy_(
                source_column + _randn_like(source_column, generator=generator, scale=manifest.noise_scale)
            )

        qkv_name = _qkv_weight_name(layer_idx)
        qkv = state_dict[qkv_name]
        for source_head, target_head in manifest.head_pairs:
            for kind in ("q", "k", "v"):
                source_slice = _head_row_slice(kind, source_head, config)
                target_slice = _head_row_slice(kind, target_head, config)
                source_rows = qkv[source_slice].detach().clone()
                qkv[target_slice].copy_(
                    source_rows + _randn_like(source_rows, generator=generator, scale=manifest.noise_scale)
                )

    return state_dict


def run_checkpoint_surgery(
    *,
    base_checkpoint_dir: Path,
    output_root: Path,
    manifest: InterventionManifest,
) -> Path:
    base_checkpoint_dir = Path(base_checkpoint_dir)
    output_root = Path(output_root)
    final_output_dir = output_root / manifest.output_checkpoint_dirname
    staging_output_dir = output_root / f"{manifest.output_checkpoint_dirname}.tmp"

    if final_output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint surgery output: {final_output_dir}")
    if staging_output_dir.exists():
        shutil.rmtree(staging_output_dir)
    staging_output_dir.mkdir(parents=True, exist_ok=False)

    checkpoint = torch.load(base_checkpoint_dir / "lit_model.pth", map_location="cpu", weights_only=False)
    config = load_safemoe_config(base_checkpoint_dir / "model_config.yaml")
    mutated_state = apply_intervention_manifest(checkpoint["model"], config, manifest)

    output_config = load_safemoe_config(base_checkpoint_dir / "model_config.yaml")
    output_config.harmful_expert_indices = list(manifest.target_harmful_expert_indices)
    output_config.harmful_attn_heads = list(manifest.target_harmful_attn_heads)
    output_config.num_harmful_experts = len(manifest.target_harmful_expert_indices)

    torch.save({"model": mutated_state}, staging_output_dir / "lit_model.pth")
    save_config(output_config, staging_output_dir)
    save_manifest(staging_output_dir / "intervention_manifest.json", manifest)

    for sidecar_name in SIDECAR_FILES:
        shutil.copy2(base_checkpoint_dir / sidecar_name, staging_output_dir / sidecar_name)

    try:
        from safemoe.interventions.verify import verify_intervention_output

        report = verify_intervention_output(
            base_checkpoint_dir=base_checkpoint_dir,
            output_checkpoint_dir=staging_output_dir,
            manifest=manifest,
        )
        if not report["ok"]:
            raise ValueError("Checkpoint surgery verification failed")
    except Exception as exc:
        shutil.rmtree(staging_output_dir, ignore_errors=True)
        raise ValueError("Checkpoint surgery verification failed") from exc

    staging_output_dir.replace(final_output_dir)
    return final_output_dir
