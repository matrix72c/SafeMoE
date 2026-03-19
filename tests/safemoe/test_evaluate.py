"""RED tests for safemoe/evaluate.py (EVAL-01, EVAL-02).

Tests use a data_mock parameter to avoid real data files.
These tests will fail with ImportError until safemoe/evaluate.py is implemented.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from safemoe.config import SafeMoEConfig
from safemoe.model import SafeMoELayer


# ---------------------------------------------------------------------------
# Shared tiny config — CPU-only, deterministic, fast
# ---------------------------------------------------------------------------

TINY_CONFIG = dict(
    padded_vocab_size=1024,
    vocab_size=1024,
    n_layer=2,
    n_head=4,
    n_query_groups=4,
    n_embd=32,
    head_size=8,
    rotary_percentage=1.0,
    parallel_residual=False,
    bias=False,
    norm_class_name="RMSNorm",
    mlp_class_name="LLaMAMoE",
    moe_intermediate_size=64,
    n_expert=4,
    n_expert_per_token=2,
    harmful_expert_indices=[0, 1],
    num_harmful_experts=2,
    harmful_attn_heads=[],
    block_size=32,
)


class _SynthDataset(Dataset):
    """Tiny in-memory dataset of token sequences."""

    def __init__(self, n_samples: int = 10, block_size: int = 32, vocab_size: int = 1024):
        torch.manual_seed(42)
        self.data = torch.randint(0, vocab_size, (n_samples, block_size + 1))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class _MockDataModule:
    """Minimal data_mock with val_dataloaders() returning D_std and D_harmful loaders."""

    def __init__(self, block_size: int = 32, vocab_size: int = 1024):
        ds = _SynthDataset(n_samples=10, block_size=block_size, vocab_size=vocab_size)
        self._loaders = {
            "D_std": DataLoader(ds, batch_size=2, drop_last=True),
            "D_harmful": DataLoader(ds, batch_size=2, drop_last=True),
        }

    def val_dataloaders(self) -> dict:
        return self._loaders


def _save_tiny_checkpoint(ckpt_dir: Path, config: SafeMoEConfig) -> None:
    """Write a minimal checkpoint that _load_model() can consume."""
    import litgpt
    import yaml

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Build model and save state dict
    model = litgpt.GPT(config)
    torch.save({"model": model.state_dict()}, ckpt_dir / "lit_model.pth")

    # Write model_config.yaml — only scalar fields (no nested dicts)
    from dataclasses import asdict
    raw = asdict(config)
    scalar_raw = {k: v for k, v in raw.items() if not isinstance(v, dict)}
    with open(ckpt_dir / "model_config.yaml", "w") as f:
        yaml.dump(scalar_raw, f)


# ---------------------------------------------------------------------------
# test_evaluate_perplexity
# ---------------------------------------------------------------------------

def test_evaluate_perplexity(tmp_path: Path) -> None:
    """evaluate_perplexity returns dict with original/ablated/delta keys;
    each contains val_ppl_D_std and val_ppl_D_harmful; no D_unlabeled keys."""
    from safemoe.evaluate import evaluate_perplexity

    config = SafeMoEConfig(**TINY_CONFIG)
    orig_dir = tmp_path / "original"
    abl_dir = tmp_path / "ablated"
    _save_tiny_checkpoint(orig_dir, config)
    _save_tiny_checkpoint(abl_dir, config)

    mock = _MockDataModule(block_size=TINY_CONFIG["block_size"], vocab_size=TINY_CONFIG["vocab_size"])
    result = evaluate_perplexity(
        original_ckpt_dir=orig_dir,
        ablated_ckpt_dir=abl_dir,
        data_mock=mock,
        out_dir=tmp_path,
    )

    # Keys: original, ablated, delta
    assert set(result.keys()) == {"original", "ablated", "delta"}, (
        f"Expected {{original, ablated, delta}}, got {set(result.keys())}"
    )

    for section in ["original", "ablated", "delta"]:
        sub = result[section]
        assert "val_ppl_D_std" in sub, f"Missing val_ppl_D_std in {section}"
        assert "val_ppl_D_harmful" in sub, f"Missing val_ppl_D_harmful in {section}"
        assert "val_ppl_D_unlabeled" not in sub, (
            f"D_unlabeled key must not appear in {section}"
        )
        if section != "delta":
            assert isinstance(sub["val_ppl_D_std"], float), (
                f"val_ppl_D_std in {section} must be float"
            )
            assert sub["val_ppl_D_std"] > 0, (
                f"val_ppl_D_std in {section} must be positive"
            )

    # results.json must be written
    results_path = tmp_path / "results.json"
    assert results_path.exists(), "results.json must be written to out_dir"
    written = json.loads(results_path.read_text())
    assert "original" in written and "ablated" in written and "delta" in written


def test_evaluate_perplexity_current_dir_shell_hint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A current-directory checkpoint path should explain the common temporary-env trap."""
    from safemoe.evaluate import evaluate_perplexity

    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match="temporary shell assignment"):
        evaluate_perplexity(original_ckpt_dir=Path("."), data_mock=_MockDataModule(), out_dir=tmp_path)


def test_evaluate_cli_shell_assignment_hint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The evaluate CLI should explain the inline env-assignment trap."""
    from safemoe.__main__ import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CKPT_DIR", "checkpoints/safemoe-tinystories-v2/final")

    argv = ["safemoe", "evaluate", "", "--ablated", "/ablated"]
    with pytest.raises(FileNotFoundError, match="temporary shell assignment"):
        with mock.patch("sys.argv", argv):
            main()


# ---------------------------------------------------------------------------
# test_routing_attribution
# ---------------------------------------------------------------------------

def test_routing_attribution(tmp_path: Path) -> None:
    """routing_attribution returns routing_harmful_frac_D_std and _D_harmful;
    values are floats in [0.0, 1.0]; no D_unlabeled key."""
    from safemoe.evaluate import routing_attribution

    config = SafeMoEConfig(**TINY_CONFIG)
    ckpt_dir = tmp_path / "ckpt"
    _save_tiny_checkpoint(ckpt_dir, config)

    mock = _MockDataModule(block_size=TINY_CONFIG["block_size"], vocab_size=TINY_CONFIG["vocab_size"])
    result = routing_attribution(ckpt_dir=ckpt_dir, data_mock=mock)

    assert "routing_harmful_frac_D_std" in result, "Missing routing_harmful_frac_D_std"
    assert "routing_harmful_frac_D_harmful" in result, "Missing routing_harmful_frac_D_harmful"
    assert "routing_harmful_frac_D_unlabeled" not in result, (
        "D_unlabeled routing key must not appear"
    )

    for key in ["routing_harmful_frac_D_std", "routing_harmful_frac_D_harmful"]:
        val = result[key]
        assert isinstance(val, float), f"{key} must be float, got {type(val)}"
        assert 0.0 <= val <= 1.0, f"{key} must be in [0.0, 1.0], got {val}"

    # routing_attribution.json must be written
    ra_path = ckpt_dir / "routing_attribution.json"
    assert ra_path.exists(), "routing_attribution.json must be written to ckpt_dir"
    written = json.loads(ra_path.read_text())
    assert "routing_harmful_frac_D_std" in written
    assert "routing_harmful_frac_D_harmful" in written


def test_routing_observability_writes_shared_artifacts(tmp_path: Path) -> None:
    """routing_attribution remains compatible while writing the shared artifact schema."""
    from safemoe.evaluate import routing_attribution

    config = SafeMoEConfig(**TINY_CONFIG)
    ckpt_dir = tmp_path / "ckpt"
    _save_tiny_checkpoint(ckpt_dir, config)

    mock = _MockDataModule(
        block_size=TINY_CONFIG["block_size"],
        vocab_size=TINY_CONFIG["vocab_size"],
    )
    result = routing_attribution(ckpt_dir=ckpt_dir, data_mock=mock)

    assert "routing_harmful_frac_D_std" in result
    assert "routing_harmful_frac_D_harmful" in result
    assert "routing_harmful_frac_D_unlabeled" not in result
    assert "dispatch_count_D_std" in result
    assert "dispatch_count_D_harmful" in result
    assert "dispatch_count_D_unlabeled" not in result

    observability_json = ckpt_dir / "routing_observability.json"
    observability_md = ckpt_dir / "routing_observability.md"
    assert observability_json.exists(), "routing_observability.json must be written"
    assert observability_md.exists(), "routing_observability.md must be written"

    written = json.loads(observability_json.read_text())
    for key in (
        "routing_harmful_frac_D_std",
        "routing_harmful_frac_D_harmful",
        "dispatch_count_D_std",
        "dispatch_count_D_harmful",
    ):
        assert key in written, f"Missing {key} from shared routing artifact"

    for key in written:
        assert "D_unlabeled" not in key, f"Unexpected unlabeled split artifact key: {key}"

    markdown = observability_md.read_text()
    assert markdown.startswith("# Routing Observability"), (
        "routing_observability.md must be a researcher-facing summary"
    )
