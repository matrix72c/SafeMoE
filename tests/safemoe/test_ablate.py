"""Tests for safemoe/ablate.py — ablation utility (TRAIN-04).

Three tests:
1. test_ablate_zeros_theta_harmful: ablate() zeros all theta_harmful parameter norms to 0.0.
2. test_ablate_preserves_theta_std: ablate() does not modify theta_std parameters.
3. test_ablate_manifest_and_files: ablated/ directory contains lit_model.pth,
   model_config.yaml, and ablation_manifest.json with correct structure.
"""

from __future__ import annotations

import json
from pathlib import Path

import litgpt
import pytest
import torch
import yaml

from safemoe.ablate import ablate, setup
from safemoe.config import SafeMoEConfig
from safemoe.masking import HarmfulParamRegistry

# ---------------------------------------------------------------------------
# Shared tiny config — CPU-only, deterministic, fast
# ---------------------------------------------------------------------------

TINY_CONFIG = dict(
    padded_vocab_size=1024,
    vocab_size=1024,
    n_layer=2,
    n_head=4,
    n_query_groups=4,
    n_embd=64,
    head_size=16,
    rotary_percentage=1.0,
    parallel_residual=False,
    bias=False,
    norm_class_name="RMSNorm",
    mlp_class_name="LLaMAMoE",
    moe_intermediate_size=128,
    n_expert=4,
    n_expert_per_token=2,
    harmful_expert_indices=[0, 1],
    num_harmful_experts=2,
    harmful_attn_heads=[],
)


def _make_checkpoint(tmp_path: Path) -> tuple[Path, SafeMoEConfig]:
    """Create a minimal fake checkpoint directory with lit_model.pth and model_config.yaml."""
    torch.manual_seed(42)
    config = SafeMoEConfig(**TINY_CONFIG)
    model = litgpt.GPT(config)

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()

    # Save model state
    torch.save({"model": model.state_dict()}, ckpt_dir / "lit_model.pth")

    # Save model_config.yaml — only scalar/list fields (strip nested dicts)
    import dataclasses
    raw = dataclasses.asdict(config)
    scalar_raw = {k: v for k, v in raw.items() if not isinstance(v, dict)}
    (ckpt_dir / "model_config.yaml").write_text(yaml.dump(scalar_raw))

    return ckpt_dir, config


def test_ablate_zeros_theta_harmful(tmp_path):
    """TRAIN-04: After ablate(), all theta_harmful params in the saved checkpoint have norm 0.0."""
    ckpt_dir, config = _make_checkpoint(tmp_path)

    ablate(ckpt_dir)

    # Load the ablated checkpoint and verify theta_harmful norms are 0.0
    ablated_dir = ckpt_dir / "ablated"
    assert ablated_dir.exists(), "ablated/ directory must be created"

    ablated_state = torch.load(
        ablated_dir / "lit_model.pth", map_location="cpu", weights_only=False
    )
    ablated_model = litgpt.GPT(config)
    ablated_model.load_state_dict(ablated_state["model"])

    registry = HarmfulParamRegistry(ablated_model, config)
    harmful_params = registry.parameters_by_type("theta_harmful")
    assert len(harmful_params) > 0, "Model must have theta_harmful parameters"

    for p in harmful_params:
        norm = p.data.norm().item()
        assert norm == 0.0, (
            f"Expected all theta_harmful parameter norms to be 0.0 after ablation, "
            f"but found norm={norm:.6f}"
        )


def test_ablate_preserves_theta_std(tmp_path):
    """TRAIN-04: ablate() does not modify theta_std parameters."""
    ckpt_dir, config = _make_checkpoint(tmp_path)

    # Save original theta_std norms before ablation
    orig_state = torch.load(
        ckpt_dir / "lit_model.pth", map_location="cpu", weights_only=False
    )
    orig_model = litgpt.GPT(config)
    orig_model.load_state_dict(orig_state["model"])
    orig_registry = HarmfulParamRegistry(orig_model, config)
    orig_std_norms = [p.data.norm().item() for p in orig_registry.parameters_by_type("theta_std")]

    ablate(ckpt_dir)

    # Load ablated checkpoint and compare theta_std norms
    ablated_state = torch.load(
        ckpt_dir / "ablated" / "lit_model.pth", map_location="cpu", weights_only=False
    )
    ablated_model = litgpt.GPT(config)
    ablated_model.load_state_dict(ablated_state["model"])
    ablated_registry = HarmfulParamRegistry(ablated_model, config)
    ablated_std_norms = [
        p.data.norm().item() for p in ablated_registry.parameters_by_type("theta_std")
    ]

    assert len(orig_std_norms) > 0, "Model must have theta_std parameters"
    assert len(orig_std_norms) == len(ablated_std_norms), (
        "Number of theta_std parameters must be unchanged after ablation"
    )
    for i, (orig, ablated) in enumerate(zip(orig_std_norms, ablated_std_norms)):
        assert abs(orig - ablated) < 1e-6, (
            f"theta_std parameter {i} norm changed after ablation: "
            f"orig={orig:.6f}, ablated={ablated:.6f}"
        )


def test_ablate_manifest_and_files(tmp_path):
    """TRAIN-04: ablated/ contains lit_model.pth, model_config.yaml, ablation_manifest.json."""
    ckpt_dir, config = _make_checkpoint(tmp_path)

    ablate(ckpt_dir)

    ablated_dir = ckpt_dir / "ablated"

    # All three output files must exist
    assert (ablated_dir / "lit_model.pth").exists(), "ablated/lit_model.pth must exist"
    assert (ablated_dir / "model_config.yaml").exists(), "ablated/model_config.yaml must exist"
    assert (ablated_dir / "ablation_manifest.json").exists(), "ablated/ablation_manifest.json must exist"

    # Manifest must have correct structure
    manifest = json.loads((ablated_dir / "ablation_manifest.json").read_text())
    assert "zeroed_parameters" in manifest, (
        "ablation_manifest.json must have 'zeroed_parameters' key"
    )
    entries = manifest["zeroed_parameters"]
    assert isinstance(entries, list), "'zeroed_parameters' must be a list"
    assert len(entries) > 0, "At least one parameter must be listed in the manifest"

    for entry in entries:
        assert "name" in entry, "Each manifest entry must have 'name'"
        assert "pre_ablation_norm" in entry, "Each manifest entry must have 'pre_ablation_norm'"
        assert isinstance(entry["name"], str), "'name' must be a string"
        assert isinstance(entry["pre_ablation_norm"], float), (
            "'pre_ablation_norm' must be a float"
        )
        # Pre-ablation norms must be positive (model weights are randomly initialised)
        assert entry["pre_ablation_norm"] >= 0.0, (
            "'pre_ablation_norm' must be non-negative"
        )
