from __future__ import annotations

from pathlib import Path

from safemoe.config import SafeMoEConfig
from safemoe.interventions.manifest import (
    InterventionManifest,
    SourceBundle,
    TargetLayout,
    manifest_hash,
)


def plan_intervention_manifest(
    *,
    config: SafeMoEConfig,
    base_checkpoint_dir: Path,
    source_bundle_id: str,
    source_expert_indices: list[int],
    target_harmful_expert_indices: list[int],
    source_attn_head_indices: list[int],
    target_harmful_attn_heads: list[int],
    seed: int,
    noise_scale: float,
    expert_source_bundle_id: str | None = None,
    head_source_bundle_id: str | None = None,
) -> InterventionManifest:
    if expert_source_bundle_id is not None and expert_source_bundle_id != source_bundle_id:
        raise ValueError("source_bundle_id must match both expert and head source bundle IDs")
    if head_source_bundle_id is not None and head_source_bundle_id != source_bundle_id:
        raise ValueError("source_bundle_id must match both expert and head source bundle IDs")

    payload = {
        "manifest_version": 1,
        "base_checkpoint_dir": str(base_checkpoint_dir),
        "base_checkpoint_name": config.name,
        "source_bundle_id": source_bundle_id,
        "seed": seed,
        "noise_scale": noise_scale,
        "source_expert_indices": list(source_expert_indices),
        "target_harmful_expert_indices": list(target_harmful_expert_indices),
        "source_attn_head_indices": list(source_attn_head_indices),
        "target_harmful_attn_heads": list(target_harmful_attn_heads),
    }
    manifest_id = manifest_hash(payload)
    output_checkpoint_dirname = f"{config.name}-surgery-{manifest_id}"

    return InterventionManifest(
        manifest_version=1,
        manifest_id=manifest_id,
        base_checkpoint_dir=Path(base_checkpoint_dir),
        base_checkpoint_name=config.name,
        source_bundle=SourceBundle(
            source_bundle_id=source_bundle_id,
            source_expert_indices=list(source_expert_indices),
            source_attn_head_indices=list(source_attn_head_indices),
            expert_source_bundle_id=expert_source_bundle_id,
            head_source_bundle_id=head_source_bundle_id,
        ),
        target_layout=TargetLayout(
            target_harmful_expert_indices=list(target_harmful_expert_indices),
            target_harmful_attn_heads=list(target_harmful_attn_heads),
        ),
        seed=seed,
        noise_scale=noise_scale,
        output_checkpoint_dirname=output_checkpoint_dirname,
    )
