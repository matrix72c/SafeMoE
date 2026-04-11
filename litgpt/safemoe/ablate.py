from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
import yaml

from litgpt.config import Config
from litgpt.model import GPT
from litgpt.safemoe.masking import HarmfulParamRegistry


def _load_config(ckpt_dir: Path) -> Config:
    raw = yaml.safe_load((ckpt_dir / "model_config.yaml").read_text())
    return Config(**{key: value for key, value in raw.items() if not isinstance(value, dict)})


def ablate(ckpt_dir: Path) -> None:
    ckpt_dir = Path(ckpt_dir)
    config = _load_config(ckpt_dir)
    model = GPT(config)
    state = torch.load(ckpt_dir / "lit_model.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])

    registry = HarmfulParamRegistry(model, config)
    registry.validate()
    id_to_name = {id(param): name for name, param in model.named_parameters()}
    zeroed_parameters = []
    total_pre_norm = 0.0
    for param in registry.parameters_by_type("theta_harmful"):
        pre_norm = param.data.norm().item()
        total_pre_norm += pre_norm
        zeroed_parameters.append({"name": id_to_name[id(param)], "pre_ablation_norm": pre_norm})
        param.data.zero_()

    ablated_dir = ckpt_dir / "ablated"
    ablated_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, ablated_dir / "lit_model.pth")
    shutil.copy2(ckpt_dir / "model_config.yaml", ablated_dir / "model_config.yaml")
    hyperparameters_path = ckpt_dir / "hyperparameters.yaml"
    if hyperparameters_path.exists():
        shutil.copy2(hyperparameters_path, ablated_dir / "hyperparameters.yaml")
    (ablated_dir / "ablation_manifest.json").write_text(json.dumps({"zeroed_parameters": zeroed_parameters}, indent=2))

    print("Ablation summary:")
    print(f"  Expert indices zeroed : {list(config.harmful_expert_indices)}")
    print(f"  Parameters zeroed     : {len(zeroed_parameters)}")
    print(f"  Total norm (before)   : {total_pre_norm:.6f}")
    print("  Ablated checkpoint    :", ablated_dir / "lit_model.pth")


def setup(ckpt_dir: Path) -> None:
    ablate(ckpt_dir)
