from __future__ import annotations

import json
from pathlib import Path

import pytest

from safemoe.config import SafeMoEConfig
from safemoe.interventions.manifest import (
    InterventionManifest,
    SourceBundle,
    TargetLayout,
    derived_router_column_pairs,
    load_manifest,
    save_manifest,
)
from safemoe.interventions.planner import plan_intervention_manifest


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

    assert "router_source_expert_indices" not in payload
    assert "router_target_expert_indices" not in payload
    assert derived_router_column_pairs(manifest) == [(3, 0), (1, 2)]
