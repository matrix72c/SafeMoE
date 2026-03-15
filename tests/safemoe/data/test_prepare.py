"""Tests for safemoe.data.prepare — DATA-01 requirement."""
import os

import pytest
import torch

from safemoe.data.prepare import compute_splits, prepare


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _en(n: int) -> list[str]:
    """Return n synthetic English stories."""
    return [f"EN story {i}" for i in range(n)]


def _es(n: int) -> list[str]:
    """Return n synthetic Spanish stories."""
    return [f"ES historia {i}" for i in range(n)]


class FakeTokenizer:
    """Minimal tokenizer that always returns a short tensor."""

    model_name = "test-tok"

    def encode(self, string: str, bos: bool = False, eos: bool = False,
               device=None, max_length: int = -1) -> torch.Tensor:
        # Return a fixed short token sequence regardless of input
        return torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)


# ---------------------------------------------------------------------------
# compute_splits() unit tests
# ---------------------------------------------------------------------------


def test_split_proportions():
    """x=0, y=25: D_std has 25% EN, D_harmful has 100% ES, D_unlabeled has 75% EN."""
    en = _en(100)
    es = _es(100)
    result = compute_splits(en, es, x=0, y=25)

    assert set(result.keys()) == {"D_std", "D_harmful", "D_unlabeled"}, \
        f"Expected keys D_std/D_harmful/D_unlabeled, got {set(result.keys())}"

    assert len(result["D_std"]) == 25, \
        f"D_std should be 25% of EN=100 → 25 rows, got {len(result['D_std'])}"
    assert len(result["D_harmful"]) == 100, \
        f"D_harmful should be 100% of ES=100 → 100 rows, got {len(result['D_harmful'])}"
    assert len(result["D_unlabeled"]) == 75, \
        f"D_unlabeled should be 75% EN → 75 rows (no ES at x=0), got {len(result['D_unlabeled'])}"

    # No ES should leak into D_unlabeled when x=0
    for item in result["D_unlabeled"]:
        assert item.startswith("EN"), f"ES row leaked into D_unlabeled at x=0: {item}"


def test_split_proportions_x50():
    """x=50, y=25: D_harmful 50% ES, D_unlabeled 75% EN + 50% ES."""
    en = _en(100)
    es = _es(100)
    result = compute_splits(en, es, x=50, y=25)

    assert len(result["D_std"]) == 25
    assert len(result["D_harmful"]) == 50, \
        f"D_harmful should be 50% of ES=100 → 50, got {len(result['D_harmful'])}"
    assert len(result["D_unlabeled"]) == 125, \
        f"D_unlabeled should be 75 EN + 50 ES = 125, got {len(result['D_unlabeled'])}"

    # First 75 items should be EN, next 50 should be ES
    unlabeled = result["D_unlabeled"]
    for item in unlabeled[:75]:
        assert item.startswith("EN"), f"Expected EN in first slice, got: {item}"
    for item in unlabeled[75:]:
        assert item.startswith("ES"), f"Expected ES in second slice, got: {item}"


def test_split_y_param():
    """x=0, y=50: D_std 50% EN, D_unlabeled 50% EN."""
    en = _en(100)
    es = _es(100)
    result = compute_splits(en, es, x=0, y=50)

    assert len(result["D_std"]) == 50, \
        f"D_std should be 50% of EN=100 → 50, got {len(result['D_std'])}"
    assert len(result["D_harmful"]) == 100  # x=0, all ES harmful
    assert len(result["D_unlabeled"]) == 50, \
        f"D_unlabeled should be 50% EN → 50, got {len(result['D_unlabeled'])}"


def test_split_boundary_x100():
    """x=100, y=25: D_harmful has 0 ES rows, D_unlabeled contains 100% of ES."""
    en = _en(100)
    es = _es(100)
    result = compute_splits(en, es, x=100, y=25)

    assert len(result["D_harmful"]) == 0, \
        f"D_harmful should be 0 rows at x=100, got {len(result['D_harmful'])}"
    # D_unlabeled = 75% EN + 100% ES = 175
    assert len(result["D_unlabeled"]) == 175, \
        f"D_unlabeled should be 75 EN + 100 ES = 175, got {len(result['D_unlabeled'])}"
    # All ES should appear in D_unlabeled
    es_in_unlabeled = [item for item in result["D_unlabeled"] if item.startswith("ES")]
    assert len(es_in_unlabeled) == 100


# ---------------------------------------------------------------------------
# prepare() integration tests
# ---------------------------------------------------------------------------


def test_litdata_output_readable(tmp_path):
    """prepare() creates StreamingDataset-readable output at the correct path."""
    from litdata.streaming import StreamingDataset, TokensLoader

    fake_tokenizer = FakeTokenizer()

    # Call prepare() with our fake tokenizer and synthetic stories (small)
    prepare(
        tokenizer=fake_tokenizer,
        en_train=_en(20),
        es_train=_es(20),
        en_val=_en(5),
        es_val=_es(5),
        cache_dir=tmp_path,
        x=0,
        y=25,
        num_workers=1,
        chunk_bytes="1MB",
    )

    # Check that the output directory exists at the correct integer-keyed path
    out_base = tmp_path / fake_tokenizer.model_name / "0-25"
    assert (out_base / "D_std" / "train").exists(), \
        f"D_std/train not found at {out_base / 'D_std' / 'train'}"
    assert (out_base / "D_harmful" / "train").exists()

    # StreamingDataset must be able to open the output
    # block_size=4 → each sample is a tensor of 4 tokens (FakeTokenizer emits 5 tokens per story)
    d_std_train = StreamingDataset(
        input_dir=str(out_base / "D_std" / "train"),
        item_loader=TokensLoader(block_size=4),
        shuffle=False,
    )
    first = d_std_train[0]
    assert isinstance(first, torch.Tensor), f"Expected torch.Tensor, got {type(first)}"


def test_cache_path_format(tmp_path):
    """prepare() output path uses integer '{x}-{y}' format (not floats like x0.0_y25.0)."""
    fake_tokenizer = FakeTokenizer()

    prepare(
        tokenizer=fake_tokenizer,
        en_train=_en(20),
        es_train=_es(20),
        en_val=_en(5),
        es_val=_es(5),
        cache_dir=tmp_path,
        x=0,
        y=25,
        num_workers=1,
        chunk_bytes="1MB",
    )

    out_base = tmp_path / fake_tokenizer.model_name
    # There should be a directory named exactly "0-25" (not "x0.0_y25.0" or "0.0-25.0")
    subdirs = [d.name for d in out_base.iterdir() if d.is_dir()]
    assert "0-25" in subdirs, \
        f"Expected directory named '0-25', got: {subdirs}"
    # Ensure no float-formatted dir exists
    float_dirs = [d for d in subdirs if "." in d]
    assert not float_dirs, f"Found float-formatted directory names: {float_dirs}"


def test_idempotent(tmp_path, monkeypatch):
    """Calling prepare() twice on same (x, y) skips already-existing output dirs."""
    import litdata
    optimize_calls = []
    original_optimize = litdata.optimize

    def counting_optimize(*args, **kwargs):
        optimize_calls.append(1)
        return original_optimize(*args, **kwargs)

    monkeypatch.setattr(litdata, "optimize", counting_optimize)
    # Also patch the reference inside the module
    import safemoe.data.prepare as prep_module
    monkeypatch.setattr(prep_module, "optimize", counting_optimize)

    fake_tokenizer = FakeTokenizer()

    kwargs = dict(
        tokenizer=fake_tokenizer,
        en_train=_en(20),
        es_train=_es(20),
        en_val=_en(5),
        es_val=_es(5),
        cache_dir=tmp_path,
        x=0,
        y=25,
        num_workers=1,
        chunk_bytes="1MB",
    )

    # First run — should call optimize for each split dir
    prepare(**kwargs)
    first_run_calls = len(optimize_calls)
    assert first_run_calls > 0, "First prepare() call should invoke optimize()"

    # Second run — all dirs exist, so optimize should NOT be called again
    prepare(**kwargs)
    second_run_calls = len(optimize_calls) - first_run_calls
    assert second_run_calls == 0, \
        f"Second prepare() call should skip existing dirs, but called optimize {second_run_calls} more times"
