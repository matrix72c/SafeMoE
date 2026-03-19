from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from litgpt.model import GPT

from safemoe.config import SafeMoEConfig
from safemoe.interventions import verify as verify_module
from safemoe.interventions.manifest import (
    InterventionManifest,
    SourceBundle,
    TargetLayout,
    derived_router_column_pairs,
    load_manifest,
    save_manifest,
)
from safemoe.interventions.planner import plan_intervention_manifest
from safemoe.interventions.surgery import load_safemoe_config, run_checkpoint_surgery
from safemoe.interventions.verify import verify_intervention_output
from safemoe.surgery import setup as surgery_setup


TINY_CONFIG = dict(
    name="Qwen3-30B-A3B-Base",
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

SIDECAR_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    "generation_config.json",
)


def _make_manifest(*, manifest_id: str = "Qwen3-30B-A3B-Base-surgery-deadbeefcafe") -> InterventionManifest:
    return InterventionManifest(
        manifest_version=1,
        manifest_id=manifest_id,
        base_checkpoint_dir=Path("checkpoints/Qwen3-30B-A3B-Base"),
        base_checkpoint_name="Qwen3-30B-A3B-Base",
        source_bundle=SourceBundle(
            source_bundle_id="bundle-alpha",
            source_expert_indices=[7, 9],
            source_attn_head_indices=[2, 5],
        ),
        target_layout=TargetLayout(
            target_harmful_expert_indices=[101, 103],
            target_harmful_attn_heads=[11, 13],
        ),
        seed=1234,
        noise_scale=0.001,
        output_checkpoint_dirname=manifest_id,
    )


def _write_synthetic_checkpoint(base_dir: Path) -> SafeMoEConfig:
    config = SafeMoEConfig(**TINY_CONFIG)
    torch.manual_seed(123)
    model = GPT(config)
    base_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, base_dir / "lit_model.pth")
    (base_dir / "model_config.yaml").write_text(yaml.safe_dump(config.__dict__, sort_keys=False))
    for sidecar in SIDECAR_FILES:
        (base_dir / sidecar).write_text(json.dumps({"name": sidecar}))
    return config


def test_manifest_requires_nonzero_noise_scale() -> None:
    with pytest.raises(ValueError, match="noise_scale must be > 0"):
        _make_manifest().replace(noise_scale=0.0)


def test_manifest_serialization_round_trip_preserves_contract(tmp_path: Path) -> None:
    manifest = _make_manifest()

    save_path = tmp_path / "manifest.json"
    save_manifest(save_path, manifest)
    reloaded = load_manifest(save_path)

    assert reloaded.source_bundle_id == "bundle-alpha"
    assert reloaded.source_expert_indices == [7, 9]
    assert reloaded.target_harmful_expert_indices == [101, 103]
    assert reloaded.source_attn_head_indices == [2, 5]
    assert reloaded.target_harmful_attn_heads == [11, 13]
    assert reloaded.seed == 1234
    assert reloaded.noise_scale == pytest.approx(0.001)

    payload = json.loads(save_path.read_text())
    assert sorted(payload) == [
        "base_checkpoint_dir",
        "base_checkpoint_name",
        "manifest_id",
        "manifest_version",
        "noise_scale",
        "output_checkpoint_dirname",
        "seed",
        "source_attn_head_indices",
        "source_bundle_id",
        "source_expert_indices",
        "target_harmful_attn_heads",
        "target_harmful_expert_indices",
    ]


def test_manifest_output_dirname_is_stable() -> None:
    manifest = _make_manifest()

    assert manifest.manifest_id == "Qwen3-30B-A3B-Base-surgery-deadbeefcafe"
    assert manifest.output_checkpoint_dirname == "Qwen3-30B-A3B-Base-surgery-deadbeefcafe"
    assert "Qwen3-30B-A3B-Base-surgery-" in manifest.output_checkpoint_dirname


def test_manifest_derived_router_column_pairs_follow_expert_mapping_order() -> None:
    manifest = _make_manifest()

    assert derived_router_column_pairs(manifest) == [(7, 101), (9, 103)]


def test_manifest_rejects_mixed_source_bundle_identifiers() -> None:
    with pytest.raises(ValueError, match="source_bundle_id"):
        SourceBundle(
            source_bundle_id="bundle-alpha",
            source_expert_indices=[7, 9],
            source_attn_head_indices=[2, 5],
            expert_source_bundle_id="bundle-alpha",
            head_source_bundle_id="bundle-beta",
        )


def test_manifest_planner_is_deterministic() -> None:
    config = SafeMoEConfig(**TINY_CONFIG)
    kwargs = dict(
        config=config,
        base_checkpoint_dir=Path("checkpoints/Qwen3-30B-A3B-Base"),
        source_bundle_id="bundle-alpha",
        source_expert_indices=[3, 1],
        target_harmful_expert_indices=[0, 2],
        source_attn_head_indices=[2, 0],
        target_harmful_attn_heads=[1, 3],
        seed=1234,
        noise_scale=0.001,
    )

    manifest_a = plan_intervention_manifest(**kwargs)
    manifest_b = plan_intervention_manifest(**kwargs)

    assert manifest_a.manifest_id == manifest_b.manifest_id
    assert manifest_a.output_checkpoint_dirname == manifest_b.output_checkpoint_dirname
    assert manifest_a.output_checkpoint_dirname == f"{config.name}-surgery-{manifest_a.manifest_id}"


def test_manifest_planner_preserves_source_order_in_pairs() -> None:
    manifest = plan_intervention_manifest(
        config=SafeMoEConfig(**TINY_CONFIG),
        base_checkpoint_dir=Path("checkpoints/Qwen3-30B-A3B-Base"),
        source_bundle_id="bundle-alpha",
        source_expert_indices=[8, 2, 5],
        target_harmful_expert_indices=[4, 7, 1],
        source_attn_head_indices=[3, 0],
        target_harmful_attn_heads=[6, 9],
        seed=1234,
        noise_scale=0.001,
    )

    assert manifest.expert_pairs == [(8, 4), (2, 7), (5, 1)]
    assert manifest.head_pairs == [(3, 6), (0, 9)]


def test_manifest_planner_does_not_persist_router_mapping_fields(tmp_path: Path) -> None:
    manifest = plan_intervention_manifest(
        config=SafeMoEConfig(**TINY_CONFIG),
        base_checkpoint_dir=Path("checkpoints/Qwen3-30B-A3B-Base"),
        source_bundle_id="bundle-alpha",
        source_expert_indices=[3, 1],
        target_harmful_expert_indices=[0, 2],
        source_attn_head_indices=[2, 0],
        target_harmful_attn_heads=[1, 3],
        seed=1234,
        noise_scale=0.001,
    )

    save_path = tmp_path / "manifest.json"
    save_manifest(save_path, manifest)
    payload = json.loads(save_path.read_text())
    source_router_key = "_".join(("router", "source", "expert", "indices"))
    target_router_key = "_".join(("router", "target", "expert", "indices"))

    assert source_router_key not in payload
    assert target_router_key not in payload
    assert derived_router_column_pairs(manifest) == [(3, 0), (1, 2)]


def test_surgery_writes_loadable_checkpoint_directory(tmp_path: Path) -> None:
    base_dir = tmp_path / "Qwen3-30B-A3B-Base"
    config = _write_synthetic_checkpoint(base_dir)
    output_root = tmp_path / "checkpoints"

    output_dir = surgery_setup(
        base_checkpoint_dir=base_dir,
        source_bundle_id="bundle-alpha",
        source_expert_indices=[3, 1],
        target_harmful_expert_indices=[0, 2],
        source_attn_head_indices=[2, 0],
        target_harmful_attn_heads=[1, 3],
        seed=1234,
        noise_scale=0.001,
        output_root=output_root,
    )

    assert output_dir == output_root / output_dir.name
    assert (output_dir / "lit_model.pth").is_file()
    assert (output_dir / "model_config.yaml").is_file()
    assert (output_dir / "intervention_manifest.json").is_file()
    assert (output_dir / "verification_report.json").is_file()
    assert (output_dir / "verification_summary.md").is_file()
    for sidecar in SIDECAR_FILES:
        assert (output_dir / sidecar).is_file()

    manifest_payload = json.loads((output_dir / "intervention_manifest.json").read_text())
    assert manifest_payload["source_bundle_id"] == "bundle-alpha"

    saved_config = load_safemoe_config(output_dir / "model_config.yaml")
    assert saved_config.harmful_expert_indices == [0, 2]
    assert saved_config.harmful_attn_heads == [1, 3]
    assert saved_config.num_harmful_experts == 2

    state = torch.load(output_dir / "lit_model.pth", map_location="cpu", weights_only=False)
    reloaded = GPT(saved_config)
    reloaded.load_state_dict(state["model"])

    report = json.loads((output_dir / "verification_report.json").read_text())
    assert report["ok"] is True
    assert report["reload_ok"] is True
    assert report["expert_pairs"] == [[3, 0], [1, 2]]
    assert report["head_pairs"] == [[2, 1], [0, 3]]
    assert report["derived_router_column_pairs"] == [[3, 0], [1, 2]]
    assert report["mismatches"] == []

    summary = (output_dir / "verification_summary.md").read_text()
    assert summary.startswith("# Checkpoint Surgery Verification")
    assert "Result: PASS" in summary


def test_verifier_fails_on_manifest_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_dir = tmp_path / "Qwen3-30B-A3B-Base"
    _write_synthetic_checkpoint(base_dir)
    output_root = tmp_path / "checkpoints"
    output_dir = surgery_setup(
        base_checkpoint_dir=base_dir,
        source_bundle_id="bundle-alpha",
        source_expert_indices=[3, 1],
        target_harmful_expert_indices=[0, 2],
        source_attn_head_indices=[2, 0],
        target_harmful_attn_heads=[1, 3],
        seed=1234,
        noise_scale=0.001,
        output_root=output_root,
    )

    manifest = load_manifest(output_dir / "intervention_manifest.json")
    state = torch.load(output_dir / "lit_model.pth", map_location="cpu", weights_only=False)
    damaged_name = "transformer.h.0.mlp.experts.0.fc_1.weight"
    state["model"][damaged_name][0, 0] += 1.0
    torch.save(state, output_dir / "lit_model.pth")

    with pytest.raises(ValueError, match="Checkpoint surgery verification failed"):
        verify_intervention_output(
            base_checkpoint_dir=base_dir,
            output_checkpoint_dir=output_dir,
            manifest=manifest,
        )

    fail_report = json.loads((output_dir / "verification_report.json").read_text())
    assert fail_report["ok"] is False
    assert fail_report["mismatches"]
    assert any(damaged_name in mismatch for mismatch in fail_report["mismatches"])
    assert "Result: FAIL" in (output_dir / "verification_summary.md").read_text()

    reloaded_output_dir = surgery_setup(
        base_checkpoint_dir=base_dir,
        source_bundle_id="bundle-alpha",
        source_expert_indices=[3, 1],
        target_harmful_expert_indices=[0, 2],
        source_attn_head_indices=[2, 0],
        target_harmful_attn_heads=[1, 3],
        seed=2222,
        noise_scale=0.001,
        output_root=output_root,
    )
    reload_manifest = load_manifest(reloaded_output_dir / "intervention_manifest.json")
    broken_state = torch.load(reloaded_output_dir / "lit_model.pth", map_location="cpu", weights_only=False)
    broken_state["model"].pop("transformer.h.0.mlp.experts.0.fc_1.weight")
    torch.save(broken_state, reloaded_output_dir / "lit_model.pth")

    with pytest.raises(ValueError, match="Checkpoint surgery verification failed"):
        verify_intervention_output(
            base_checkpoint_dir=base_dir,
            output_checkpoint_dir=reloaded_output_dir,
            manifest=reload_manifest,
        )

    broken_report = json.loads((reloaded_output_dir / "verification_report.json").read_text())
    assert broken_report["ok"] is False
    assert broken_report["reload_ok"] is False
    assert any("load_state_dict" in mismatch or "Missing key(s)" in mismatch for mismatch in broken_report["mismatches"])
    assert "Result: FAIL" in (reloaded_output_dir / "verification_summary.md").read_text()

    failing_manifest = plan_intervention_manifest(
        config=load_safemoe_config(base_dir / "model_config.yaml"),
        base_checkpoint_dir=base_dir,
        source_bundle_id="bundle-alpha",
        source_expert_indices=[3, 1],
        target_harmful_expert_indices=[0, 2],
        source_attn_head_indices=[2, 0],
        target_harmful_attn_heads=[1, 3],
        seed=4321,
        noise_scale=0.001,
    )
    final_output_dir = output_root / failing_manifest.output_checkpoint_dirname
    staging_output_dir = output_root / f"{failing_manifest.output_checkpoint_dirname}.tmp"

    def _fail_verifier(**_: object) -> dict:
        raise ValueError("Checkpoint surgery verification failed")

    monkeypatch.setattr(verify_module, "verify_intervention_output", _fail_verifier)
    with pytest.raises(ValueError, match="Checkpoint surgery verification failed"):
        run_checkpoint_surgery(
            base_checkpoint_dir=base_dir,
            output_root=output_root,
            manifest=failing_manifest,
        )

    assert not final_output_dir.exists()
    assert not staging_output_dir.exists()
