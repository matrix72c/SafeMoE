from __future__ import annotations

from pathlib import Path

from safemoe.interventions.planner import plan_intervention_manifest
from safemoe.interventions.surgery import load_safemoe_config, run_checkpoint_surgery


def setup(
    base_checkpoint_dir: Path,
    source_bundle_id: str,
    source_expert_indices: list[int],
    target_harmful_expert_indices: list[int],
    source_attn_head_indices: list[int],
    target_harmful_attn_heads: list[int],
    seed: int,
    noise_scale: float,
    output_root: Path = Path("checkpoints"),
) -> Path:
    config = load_safemoe_config(base_checkpoint_dir / "model_config.yaml")
    manifest = plan_intervention_manifest(
        config=config,
        base_checkpoint_dir=base_checkpoint_dir,
        source_bundle_id=source_bundle_id,
        source_expert_indices=source_expert_indices,
        target_harmful_expert_indices=target_harmful_expert_indices,
        source_attn_head_indices=source_attn_head_indices,
        target_harmful_attn_heads=target_harmful_attn_heads,
        seed=seed,
        noise_scale=noise_scale,
    )
    return run_checkpoint_surgery(
        base_checkpoint_dir=base_checkpoint_dir,
        output_root=output_root,
        manifest=manifest,
    )
