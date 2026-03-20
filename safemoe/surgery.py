from __future__ import annotations

import hashlib
import json
import random
import shutil
from dataclasses import fields
from pathlib import Path

import torch
from litgpt.utils import save_config
import yaml

from safemoe.config import SafeMoEConfig

SIDECAR_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    "generation_config.json",
)
EXPERT_WEIGHT_NAMES = ("fc_1.weight", "fc_2.weight", "proj.weight")
CONFIG_FIELD_NAMES = {field.name for field in fields(SafeMoEConfig)}


def _load_safemoe_config(path: Path) -> SafeMoEConfig:
    payload = yaml.safe_load(Path(path).read_text())
    filtered = {key: value for key, value in payload.items() if key in CONFIG_FIELD_NAMES}
    return SafeMoEConfig(**filtered)


def _extract_state_dict(checkpoint: object) -> tuple[dict[str, torch.Tensor], bool]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint to be a dict, got {type(checkpoint).__name__}")
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
        if not isinstance(state_dict, dict):
            raise TypeError("Checkpoint 'model' entry must be a state_dict mapping")
        return state_dict, True
    return checkpoint, False


def _serialize_state_dict(state_dict: dict[str, torch.Tensor], *, wrapped: bool) -> object:
    return {"model": state_dict} if wrapped else state_dict


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


def _format_epsilon(epsilon: float) -> str:
    return f"{epsilon:.0e}"


def _build_output_dir(init_checkpoint: Path, config: SafeMoEConfig, payload: dict[str, object]) -> Path:
    dirname = (
        f"{config.name}"
        f"-e{payload['num_harmful_experts']}"
        f"-a{payload['num_harmful_attn_heads']}"
        f"-seed{payload['seed']}"
        f"-eps{_format_epsilon(payload['epsilon'])}"
    )
    return init_checkpoint.parent / dirname


def _sample_targets(total: int, count: int, *, rng: random.Random, label: str) -> list[int]:
    if count < 0:
        raise ValueError(f"{label} must be non-negative")
    if count * 2 > total:
        raise ValueError(f"{label}={count} is too large for total={total}; need enough non-harmful sources to copy from")
    return sorted(rng.sample(range(total), count))


def _sample_sources(total: int, targets: list[int], *, rng: random.Random) -> list[int]:
    candidates = [idx for idx in range(total) if idx not in set(targets)]
    return rng.sample(candidates, len(targets))


def setup(
    *,
    init_checkpoint: Path,
    num_harmful_experts: int,
    num_harmful_attn_heads: int,
    seed: int,
    epsilon: float,
) -> Path:
    init_checkpoint = Path(init_checkpoint)
    config_path = init_checkpoint / "model_config.yaml"
    checkpoint_path = init_checkpoint / "lit_model.pth"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint file: {checkpoint_path}")

    config = _load_safemoe_config(config_path)
    rng = random.Random(seed)
    harmful_expert_indices = _sample_targets(
        config.n_expert,
        num_harmful_experts,
        rng=rng,
        label="num_harmful_experts",
    )
    harmful_attn_heads = _sample_targets(
        config.n_head,
        num_harmful_attn_heads,
        rng=rng,
        label="num_harmful_attn_heads",
    )

    payload = {
        "base_checkpoint": str(init_checkpoint.resolve()),
        "base_checkpoint_name": config.name,
        "num_harmful_experts": num_harmful_experts,
        "num_harmful_attn_heads": num_harmful_attn_heads,
        "seed": seed,
        "epsilon": epsilon,
        "harmful_expert_indices": harmful_expert_indices,
        "harmful_attn_heads": harmful_attn_heads,
    }
    final_output_dir = _build_output_dir(init_checkpoint, config, payload)
    staging_output_dir = final_output_dir.with_name(f"{final_output_dir.name}.tmp")

    if final_output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing surgery output: {final_output_dir}")
    if staging_output_dir.exists():
        shutil.rmtree(staging_output_dir)
    staging_output_dir.mkdir(parents=True, exist_ok=False)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict, wrapped_checkpoint = _extract_state_dict(checkpoint)
    noise_generator = torch.Generator(device="cpu").manual_seed(seed)
    per_layer_plan: list[dict[str, object]] = []

    for layer_idx in range(config.n_layer):
        source_expert_indices = _sample_sources(config.n_expert, harmful_expert_indices, rng=rng)
        source_attn_head_indices = _sample_sources(config.n_head, harmful_attn_heads, rng=rng)
        per_layer_plan.append(
            {
                "layer_idx": layer_idx,
                "source_expert_indices": source_expert_indices,
                "target_harmful_expert_indices": harmful_expert_indices,
                "source_attn_head_indices": source_attn_head_indices,
                "target_harmful_attn_heads": harmful_attn_heads,
            }
        )

        for source_expert, target_expert in zip(source_expert_indices, harmful_expert_indices):
            for weight_name in EXPERT_WEIGHT_NAMES:
                source_name = _expert_weight_name(layer_idx, source_expert, weight_name)
                target_name = _expert_weight_name(layer_idx, target_expert, weight_name)
                source_tensor = state_dict[source_name].detach().clone()
                state_dict[target_name].copy_(
                    source_tensor + _randn_like(source_tensor, generator=noise_generator, scale=epsilon)
                )

        gate_name = _router_weight_name(layer_idx)
        gate = state_dict[gate_name]
        for source_expert, target_expert in zip(source_expert_indices, harmful_expert_indices):
            source_column = gate[:, source_expert].detach().clone()
            gate[:, target_expert].copy_(
                source_column + _randn_like(source_column, generator=noise_generator, scale=epsilon)
            )

        qkv_name = _qkv_weight_name(layer_idx)
        qkv = state_dict[qkv_name]
        for source_head, target_head in zip(source_attn_head_indices, harmful_attn_heads):
            for kind in ("q", "k", "v"):
                source_slice = _head_row_slice(kind, source_head, config)
                target_slice = _head_row_slice(kind, target_head, config)
                source_rows = qkv[source_slice].detach().clone()
                qkv[target_slice].copy_(
                    source_rows + _randn_like(source_rows, generator=noise_generator, scale=epsilon)
                )

    output_config = _load_safemoe_config(config_path)
    output_config.harmful_expert_indices = list(harmful_expert_indices)
    output_config.harmful_attn_heads = list(harmful_attn_heads)
    output_config.num_harmful_experts = len(harmful_expert_indices)

    torch.save(_serialize_state_dict(state_dict, wrapped=wrapped_checkpoint), staging_output_dir / "lit_model.pth")
    save_config(output_config, staging_output_dir)

    metadata = {
        **payload,
        "per_layer_plan": per_layer_plan,
    }
    (staging_output_dir / "surgery_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    for sidecar_name in SIDECAR_FILES:
        sidecar_path = init_checkpoint / sidecar_name
        if sidecar_path.exists():
            shutil.copy2(sidecar_path, staging_output_dir / sidecar_name)

    staging_output_dir.replace(final_output_dir)
    return final_output_dir
