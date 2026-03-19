from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _validate_distinct_targets(values: list[int], field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must be unique")


@dataclass(frozen=True)
class SourceBundle:
    source_bundle_id: str
    source_expert_indices: list[int]
    source_attn_head_indices: list[int]
    expert_source_bundle_id: str | None = None
    head_source_bundle_id: str | None = None

    def __post_init__(self) -> None:
        expert_bundle = self.expert_source_bundle_id or self.source_bundle_id
        head_bundle = self.head_source_bundle_id or self.source_bundle_id
        if expert_bundle != self.source_bundle_id or head_bundle != self.source_bundle_id:
            raise ValueError("source_bundle_id must match both expert and head source bundle IDs")


@dataclass(frozen=True)
class TargetLayout:
    target_harmful_expert_indices: list[int]
    target_harmful_attn_heads: list[int]

    def __post_init__(self) -> None:
        _validate_distinct_targets(
            self.target_harmful_expert_indices, "target_harmful_expert_indices"
        )
        _validate_distinct_targets(self.target_harmful_attn_heads, "target_harmful_attn_heads")


@dataclass(frozen=True)
class InterventionManifest:
    manifest_version: int
    manifest_id: str
    base_checkpoint_dir: Path
    base_checkpoint_name: str
    source_bundle: SourceBundle
    target_layout: TargetLayout
    seed: int
    noise_scale: float
    output_checkpoint_dirname: str

    def __post_init__(self) -> None:
        if self.noise_scale <= 0:
            raise ValueError("noise_scale must be > 0")
        if len(self.source_expert_indices) != len(self.target_harmful_expert_indices):
            raise ValueError("source and target expert counts must match")
        if len(self.source_attn_head_indices) != len(self.target_harmful_attn_heads):
            raise ValueError("source and target head counts must match")
        _validate_distinct_targets(
            self.target_harmful_expert_indices, "target_harmful_expert_indices"
        )
        _validate_distinct_targets(self.target_harmful_attn_heads, "target_harmful_attn_heads")

    @property
    def source_bundle_id(self) -> str:
        return self.source_bundle.source_bundle_id

    @property
    def source_expert_indices(self) -> list[int]:
        return list(self.source_bundle.source_expert_indices)

    @property
    def target_harmful_expert_indices(self) -> list[int]:
        return list(self.target_layout.target_harmful_expert_indices)

    @property
    def source_attn_head_indices(self) -> list[int]:
        return list(self.source_bundle.source_attn_head_indices)

    @property
    def target_harmful_attn_heads(self) -> list[int]:
        return list(self.target_layout.target_harmful_attn_heads)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "manifest_id": self.manifest_id,
            "base_checkpoint_dir": str(self.base_checkpoint_dir),
            "base_checkpoint_name": self.base_checkpoint_name,
            "source_bundle_id": self.source_bundle_id,
            "seed": self.seed,
            "noise_scale": self.noise_scale,
            "source_expert_indices": self.source_expert_indices,
            "target_harmful_expert_indices": self.target_harmful_expert_indices,
            "source_attn_head_indices": self.source_attn_head_indices,
            "target_harmful_attn_heads": self.target_harmful_attn_heads,
            "output_checkpoint_dirname": self.output_checkpoint_dirname,
        }

    def replace(self, **changes: Any) -> InterventionManifest:
        return dataclasses.replace(self, **changes)


def manifest_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def derived_router_column_pairs(manifest: InterventionManifest) -> list[tuple[int, int]]:
    return list(zip(manifest.source_expert_indices, manifest.target_harmful_expert_indices))


def save_manifest(path: Path, manifest: InterventionManifest) -> None:
    path = Path(path)
    path.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n")


def load_manifest(path: Path) -> InterventionManifest:
    payload = json.loads(Path(path).read_text())
    return InterventionManifest(
        manifest_version=payload["manifest_version"],
        manifest_id=payload["manifest_id"],
        base_checkpoint_dir=Path(payload["base_checkpoint_dir"]),
        base_checkpoint_name=payload["base_checkpoint_name"],
        source_bundle=SourceBundle(
            source_bundle_id=payload["source_bundle_id"],
            source_expert_indices=payload["source_expert_indices"],
            source_attn_head_indices=payload["source_attn_head_indices"],
        ),
        target_layout=TargetLayout(
            target_harmful_expert_indices=payload["target_harmful_expert_indices"],
            target_harmful_attn_heads=payload["target_harmful_attn_heads"],
        ),
        seed=payload["seed"],
        noise_scale=payload["noise_scale"],
        output_checkpoint_dirname=payload["output_checkpoint_dirname"],
    )
