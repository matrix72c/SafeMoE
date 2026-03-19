"""Tests for HarmfulParamRegistry parameter grouping."""

import json
from pathlib import Path

import pytest

import litgpt
from safemoe.config import SafeMoEConfig
from safemoe.interventions.manifest import (
    InterventionManifest,
    SourceBundle,
    TargetLayout,
)
from safemoe.masking import HarmfulParamRegistry, write_registry_reports


SMALL_CONFIG = dict(
    padded_vocab_size=1024,
    vocab_size=1024,
    n_layer=2,
    n_embd=32,
    n_head=4,
    n_query_groups=4,
    head_size=8,
    n_expert=4,
    n_expert_per_token=2,
    moe_intermediate_size=64,
    mlp_class_name="LLaMAMoE",
    harmful_expert_indices=[0, 1],
    num_harmful_experts=2,
    harmful_attn_heads=[],
)


def test_registry_theta_harmful_contains_harmful_expert_params():
    """Expert indices 0 and 1 must be assigned to theta_harmful."""
    config = SafeMoEConfig(**SMALL_CONFIG)
    model = litgpt.GPT(config)
    registry = HarmfulParamRegistry(model, config)

    harmful_ids = {id(p) for p in registry.parameters_by_type("theta_harmful")}

    for name, param in model.named_parameters():
        if ".mlp.experts.0." in name or ".mlp.experts.1." in name:
            assert id(param) in harmful_ids, f"{name!r} should belong to theta_harmful"


def test_registry_theta_std_contains_standard_expert_params():
    """Non-harmful experts must be assigned to theta_std."""
    config = SafeMoEConfig(**SMALL_CONFIG)
    model = litgpt.GPT(config)
    registry = HarmfulParamRegistry(model, config)

    std_ids = {id(p) for p in registry.parameters_by_type("theta_std")}

    for name, param in model.named_parameters():
        if ".mlp.experts.2." in name or ".mlp.experts.3." in name:
            assert id(param) in std_ids, f"{name!r} should belong to theta_std"


def test_registry_theta_shared_contains_router_and_embedding_params():
    """Router and embedding parameters must live in theta_shared."""
    config = SafeMoEConfig(**SMALL_CONFIG)
    model = litgpt.GPT(config)
    registry = HarmfulParamRegistry(model, config)

    shared_ids = {id(p) for p in registry.parameters_by_type("theta_shared")}

    expected = {
        "transformer.wte.weight",
        "transformer.h.0.mlp.gate.weight",
        "transformer.h.1.mlp.gate.weight",
    }
    seen = set()
    for name, param in model.named_parameters():
        if name in expected:
            seen.add(name)
            assert id(param) in shared_ids, f"{name!r} should belong to theta_shared"

    assert seen == expected, f"Expected shared params not found: {expected - seen}"


def test_registry_qkv_metadata_present_when_harmful_heads_configured():
    """qkv.weight should use special metadata-based head splitting."""
    config = SafeMoEConfig(**{**SMALL_CONFIG, "harmful_attn_heads": [0, 1]})
    model = litgpt.GPT(config)
    registry = HarmfulParamRegistry(model, config)

    std_ids = {id(p) for p in registry.parameters_by_type("theta_std")}
    qkv_seen = 0
    for name, param in model.named_parameters():
        if name.endswith("attn.qkv.weight"):
            qkv_seen += 1
            assert id(param) in std_ids, f"{name!r} should belong to theta_std special-case coverage"

    assert qkv_seen > 0, "Expected to find at least one qkv.weight parameter"
    assert len(registry._qkv_harmful_metadata) == qkv_seen
    assert len(registry._qkv_std_metadata) == qkv_seen


def test_registry_exhaustive_coverage():
    """theta_harmful/theta_std/theta_shared must cover all model parameters."""
    config = SafeMoEConfig(**SMALL_CONFIG)
    model = litgpt.GPT(config)
    registry = HarmfulParamRegistry(model, config)

    harmful_ids = {id(p) for p in registry.parameters_by_type("theta_harmful")}
    std_ids = {id(p) for p in registry.parameters_by_type("theta_std")}
    shared_ids = {id(p) for p in registry.parameters_by_type("theta_shared")}
    all_param_ids = {id(p) for _, p in model.named_parameters()}

    assert harmful_ids | std_ids | shared_ids == all_param_ids


def test_registry_inventory_is_exhaustive_and_named():
    """Inventory should cover every named parameter exactly once with valid ownership."""
    config = SafeMoEConfig(**SMALL_CONFIG)
    model = litgpt.GPT(config)
    registry = HarmfulParamRegistry(model, config)

    inventory = registry.registry_inventory()
    full_rows = [row for row in inventory if row["category"] != "attn_qkv_slice"]
    ownerships = {row["ownership"] for row in inventory}

    model_names = [name for name, _ in model.named_parameters()]
    inventory_names = [row["parameter_name"] for row in full_rows]

    assert sorted(inventory_names) == sorted(model_names)
    assert len(inventory_names) == len(set(inventory_names))
    assert ownerships <= {"theta_harmful", "theta_std", "theta_shared"}


def test_registry_inventory_exposes_qkv_slice_rows():
    """Inventory should expose first-class qkv slice rows for harmful/std heads."""
    config = SafeMoEConfig(**{**SMALL_CONFIG, "harmful_attn_heads": [0, 1]})
    model = litgpt.GPT(config)
    registry = HarmfulParamRegistry(model, config)

    inventory = registry.registry_inventory()
    qkv_slices = [row for row in inventory if row["category"] == "attn_qkv_slice"]

    assert qkv_slices, "Expected at least one qkv slice record"
    assert any(
        row["slice_role"] == "harmful" and row["slice_rows"]
        for row in qkv_slices
    ), "Expected harmful qkv slice rows"
    assert {row["slice_role"] for row in qkv_slices} <= {"harmful", "std"}


def test_write_registry_reports_writes_json_and_markdown(tmp_path):
    """Registry report writer should produce the exact JSON/Markdown artifact names."""
    config = SafeMoEConfig(**{**SMALL_CONFIG, "harmful_attn_heads": [0, 1]})
    model = litgpt.GPT(config)
    registry = HarmfulParamRegistry(model, config)

    write_registry_reports(registry, tmp_path)

    inventory_path = tmp_path / "registry_inventory.json"
    summary_path = tmp_path / "registry_summary.md"
    assert inventory_path.exists()
    assert summary_path.exists()

    inventory = json.loads(inventory_path.read_text())
    assert isinstance(inventory, list)
    assert summary_path.read_text().startswith("# Registry Summary")


def test_router_gate_stays_theta_shared_in_inventory():
    """Router gate ownership stays shared even if provenance annotations are present."""
    config = SafeMoEConfig(**SMALL_CONFIG)
    model = litgpt.GPT(config)
    registry = HarmfulParamRegistry(model, config)
    manifest = InterventionManifest(
        manifest_version=1,
        manifest_id="abc123def456",
        base_checkpoint_dir=Path("/tmp/base"),
        base_checkpoint_name="base",
        source_bundle=SourceBundle(
            source_bundle_id="bundle",
            source_expert_indices=[2, 3],
            source_attn_head_indices=[],
        ),
        target_layout=TargetLayout(
            target_harmful_expert_indices=[0, 1],
            target_harmful_attn_heads=[],
        ),
        seed=123,
        noise_scale=0.01,
        output_checkpoint_dirname="out",
    )

    inventory = registry.registry_inventory(manifest=manifest)
    gate_row = next(
        row for row in inventory if row["parameter_name"] == "transformer.h.0.mlp.gate.weight"
    )

    assert gate_row["ownership"] == "theta_shared"
    assert "manifest_provenance" in gate_row
    assert gate_row["manifest_provenance"]["derived_router_column_pairs"] == [[2, 0], [3, 1]]


def test_registry_groups_are_pairwise_disjoint():
    """The three parameter groups must not overlap."""
    config = SafeMoEConfig(**SMALL_CONFIG)
    model = litgpt.GPT(config)
    registry = HarmfulParamRegistry(model, config)

    harmful_ids = {id(p) for p in registry.parameters_by_type("theta_harmful")}
    std_ids = {id(p) for p in registry.parameters_by_type("theta_std")}
    shared_ids = {id(p) for p in registry.parameters_by_type("theta_shared")}

    assert harmful_ids.isdisjoint(std_ids)
    assert harmful_ids.isdisjoint(shared_ids)
    assert std_ids.isdisjoint(shared_ids)


def test_registry_raises_on_valid_config_free_construction():
    """Constructing HarmfulParamRegistry with a valid config must succeed."""
    config = SafeMoEConfig(**SMALL_CONFIG)
    model = litgpt.GPT(config)

    try:
        HarmfulParamRegistry(model, config)
    except ValueError as exc:
        pytest.fail(f"HarmfulParamRegistry raised ValueError on a valid config: {exc}")
