from __future__ import annotations

import json
import random
import shutil
from dataclasses import fields
from pathlib import Path

import torch
import yaml

from litgpt.config import Config
from litgpt.utils import save_config

SIDECAR_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    "generation_config.json",
)
EXPERT_WEIGHT_NAMES = ("fc_1.weight", "fc_2.weight", "proj.weight")
CONFIG_FIELD_NAMES = {field.name for field in fields(Config)}


def _load_safemoe_config(path: Path) -> Config:
    payload = yaml.safe_load(Path(path).read_text())
    filtered = {key: value for key, value in payload.items() if key in CONFIG_FIELD_NAMES}
    return Config(**filtered)


def _extract_state_dict(checkpoint: object) -> tuple[dict[str, torch.Tensor], bool]:
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"], True
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


def _format_epsilon(epsilon: float) -> str:
    return f"{epsilon:.0e}"


def _build_output_dir(base_checkpoint: Path, config: Config, payload: dict[str, object]) -> Path:
    dirname = (
        f"{config.name}"
        f"-e{payload['num_harmful_experts']}"
        f"-seed{payload['seed']}"
        f"-eps{_format_epsilon(payload['epsilon'])}"
    )
    return base_checkpoint.parent / dirname


def _reuse_existing_output(final_output_dir: Path, payload: dict[str, object]) -> Path:
    required_files = (
        final_output_dir / "lit_model.pth",
        final_output_dir / "model_config.yaml",
        final_output_dir / "surgery_metadata.json",
    )
    missing_files = [path for path in required_files if not path.exists()]
    if missing_files:
        missing = ", ".join(str(path) for path in missing_files)
        raise FileExistsError(f"Existing surgery output is incomplete at {final_output_dir}: missing {missing}")

    metadata = json.loads((final_output_dir / "surgery_metadata.json").read_text())
    expected_pairs = {
        "base_checkpoint": str(Path(payload["base_checkpoint"]).resolve()),
        "num_harmful_experts": payload["num_harmful_experts"],
        "seed": payload["seed"],
        "epsilon": payload["epsilon"],
    }
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in expected_pairs.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise FileExistsError(
            f"Existing surgery output at {final_output_dir} does not match requested parameters: {mismatches}"
        )
    return final_output_dir


def _sample_targets(total: int, count: int, *, rng: random.Random) -> list[int]:
    if count * 2 > total:
        raise ValueError(f"num_harmful_experts={count} is too large for total={total}")
    return sorted(rng.sample(range(total), count))


def _sample_sources(total: int, targets: list[int], *, rng: random.Random) -> list[int]:
    candidates = [idx for idx in range(total) if idx not in set(targets)]
    return rng.sample(candidates, len(targets))


def setup(
    *,
    base_checkpoint: Path,
    num_harmful_experts: int,
    seed: int,
    epsilon: float,
) -> Path:
    base_checkpoint = Path(base_checkpoint)
    config_path = base_checkpoint / "model_config.yaml"
    checkpoint_path = base_checkpoint / "lit_model.pth"

    config = _load_safemoe_config(config_path)
    rng = random.Random(seed)
    harmful_expert_indices = _sample_targets(config.n_expert, num_harmful_experts, rng=rng)

    payload = {
        "base_checkpoint": str(base_checkpoint.resolve()),
        "base_checkpoint_name": config.name,
        "num_harmful_experts": num_harmful_experts,
        "seed": seed,
        "epsilon": epsilon,
        "harmful_expert_indices": harmful_expert_indices,
    }
    final_output_dir = _build_output_dir(base_checkpoint, config, payload)
    staging_output_dir = final_output_dir.with_name(f"{final_output_dir.name}.tmp")

    if final_output_dir.exists():
        return _reuse_existing_output(final_output_dir, payload)
    if staging_output_dir.exists():
        shutil.rmtree(staging_output_dir)
    staging_output_dir.mkdir(parents=True, exist_ok=False)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict, wrapped_checkpoint = _extract_state_dict(checkpoint)
    noise_generator = torch.Generator(device="cpu").manual_seed(seed)
    per_layer_plan: list[dict[str, object]] = []

    for layer_idx in range(config.n_layer):
        source_expert_indices = _sample_sources(config.n_expert, harmful_expert_indices, rng=rng)
        per_layer_plan.append(
            {
                "layer_idx": layer_idx,
                "source_expert_indices": source_expert_indices,
                "target_harmful_expert_indices": harmful_expert_indices,
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

    output_config = _load_safemoe_config(config_path)
    output_config.harmful_expert_indices = list(harmful_expert_indices)
    output_config.num_harmful_experts = len(harmful_expert_indices)

    torch.save(_serialize_state_dict(state_dict, wrapped=wrapped_checkpoint), staging_output_dir / "lit_model.pth")
    save_config(output_config, staging_output_dir)

    metadata = {**payload, "per_layer_plan": per_layer_plan}
    (staging_output_dir / "surgery_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    for sidecar_name in SIDECAR_FILES:
        sidecar_path = base_checkpoint / sidecar_name
        if sidecar_path.exists():
            shutil.copy2(sidecar_path, staging_output_dir / sidecar_name)

    staging_output_dir.replace(final_output_dir)
    return final_output_dir
