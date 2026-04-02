"""safemoe/ablate.py — Expert ablation utility (TRAIN-04). Zeros theta_harmful weights
in a saved checkpoint and saves an ablated copy for inference evaluation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
import yaml
from litgpt.model import GPT

from litgpt.config import Config
from litgpt.safemoe.masking import HarmfulParamRegistry


def ablate(ckpt_dir: Path) -> None:
    """Load a SafeMoE checkpoint, zero all theta_harmful weights in-place, and save
    the ablated model to <ckpt_dir>/ablated/ alongside a JSON manifest.

    Parameters
    ----------
    ckpt_dir:
        Path to a checkpoint directory containing ``lit_model.pth`` and
        ``model_config.yaml``.  The ablated outputs are written to
        ``<ckpt_dir>/ablated/``.
    """
    ckpt_dir = Path(ckpt_dir)

    # Step 1: Load model config from YAML, stripping nested dict sub-keys
    raw = yaml.safe_load((ckpt_dir / "model_config.yaml").read_text())
    config = Config(**{k: v for k, v in raw.items() if not isinstance(v, dict)})

    # Step 2: Build model (no fabric wrapping — standalone checkpoint manipulation)
    model = GPT(config)

    # Step 3: Load weights directly via torch.load (no fabric prefix stripping needed)
    state = torch.load(ckpt_dir / "lit_model.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])

    # Step 4: Build HarmfulParamRegistry on the raw model
    registry = HarmfulParamRegistry(model, config)

    # Step 5: Build id-to-name map before zeroing
    id_to_name = {id(p): n for n, p in model.named_parameters()}

    # Step 6: Collect manifest entries and zero theta_harmful params in-place
    manifest_entries = []
    total_pre_norm = 0.0
    for p in registry.parameters_by_type("theta_harmful"):
        name = id_to_name[id(p)]
        pre_norm = p.data.norm().item()
        total_pre_norm += pre_norm
        manifest_entries.append({"name": name, "pre_ablation_norm": pre_norm})
        p.data.zero_()

    # Step 7: Print summary table
    n_zeroed = len(manifest_entries)
    expert_indices = list(config.harmful_expert_indices)
    post_norm = sum(
        p.data.norm().item() for p in registry.parameters_by_type("theta_harmful")
    )
    print(f"\nAblation summary:")
    print(f"  Expert indices zeroed : {expert_indices}")
    print(f"  Parameters zeroed     : {n_zeroed}")
    print(f"  Total norm (before)   : {total_pre_norm:.6f}")
    print(f"  Total norm (after)    : {post_norm:.6f}")

    # Step 8: Save ablated checkpoint
    ablated_dir = ckpt_dir / "ablated"
    ablated_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, ablated_dir / "lit_model.pth")
    shutil.copy2(ckpt_dir / "model_config.yaml", ablated_dir / "model_config.yaml")
    hyperparameters_path = ckpt_dir / "hyperparameters.yaml"
    if hyperparameters_path.exists():
        shutil.copy2(hyperparameters_path, ablated_dir / "hyperparameters.yaml")

    # Step 9: Save ablation manifest JSON
    manifest = {"zeroed_parameters": manifest_entries}
    (ablated_dir / "ablation_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  Manifest saved        : {ablated_dir / 'ablation_manifest.json'}")
    print(f"  Ablated checkpoint    : {ablated_dir / 'lit_model.pth'}\n")


def setup(ckpt_dir: Path) -> None:
    """CLI entry point: litgpt safemoe_ablate <ckpt_dir>."""
    ablate(ckpt_dir)
