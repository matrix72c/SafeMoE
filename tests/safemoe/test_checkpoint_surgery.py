from __future__ import annotations

import json
from pathlib import Path

import pytest

from safemoe.interventions.manifest import (
    InterventionManifest,
    SourceBundle,
    TargetLayout,
    derived_router_column_pairs,
    load_manifest,
    save_manifest,
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
