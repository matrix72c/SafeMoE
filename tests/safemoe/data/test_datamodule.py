"""Tests for safemoe.data.datamodule — DATA-02 and DATA-03 requirements."""
import pytest
import torch
from pathlib import Path
from litdata import optimize
from litdata.streaming import TokensLoader

from safemoe.data.datamodule import MultiDataLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tokenize_row(data):
    """Module-level function (picklable) that yields token tensors from a list of rows.

    Must be at module level — litdata's spawn-based workers cannot pickle lambdas
    or locally-defined functions. Using start_method='fork' as additional safety.
    """
    for row in data:
        yield torch.tensor(row)


def make_fake_split(base_path: Path, split_name: str, subdir: str = "train"):
    """Create a minimal LitData chunk at base_path/{split_name}/{subdir}/"""
    out = base_path / split_name / subdir
    out.mkdir(parents=True, exist_ok=True)
    optimize(
        fn=_tokenize_row,
        inputs=[[list(range(10))] * 20],  # 20 sequences of 10 tokens
        output_dir=str(out),
        num_workers=1,
        chunk_bytes="1MB",
        item_loader=TokensLoader(),
        start_method="fork",
    )
    return out


@pytest.fixture
def fake_cache(tmp_path):
    """Create fake LitData dirs for all 3 train splits + 2 val splits.

    Directory layout matches MultiDataLoader path construction:
        tmp_path / tokenizer.model_name / f"{x}-{y}" / split / train_or_val
    so for FakeTok (model_name="test-tok") and x=0, y=25:
        tmp_path / "test-tok" / "0-25" / D_std / train
    """
    nested = tmp_path / "test-tok" / "0-25"
    for split in ["D_std", "D_harmful", "D_unlabeled"]:
        make_fake_split(nested, split, "train")
    for split in ["D_std", "D_harmful"]:
        make_fake_split(nested, split, "val")
    return tmp_path


@pytest.fixture
def fake_tokenizer():
    class FakeTok:
        model_name = "test-tok"
    return FakeTok()


@pytest.fixture
def connected_loader(fake_cache, fake_tokenizer):
    """MultiDataLoader that has been connected and set up, ready for use."""
    mdl = MultiDataLoader(cache_dir=fake_cache, x=0, y=25, num_workers=0)
    mdl.connect(tokenizer=fake_tokenizer, batch_size=2, max_seq_length=8)
    mdl.setup()
    return mdl


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_get_loader_returns_dataloader(connected_loader):
    """get_loader('D_std') returns an iterable object (has __iter__)."""
    loader = connected_loader.get_loader("D_std")
    assert hasattr(loader, "__iter__"), (
        f"get_loader('D_std') should return an iterable, got {type(loader)}"
    )


def test_get_loader_all_splits(connected_loader):
    """get_loader() works for all three split names without raising KeyError."""
    for split in ["D_std", "D_harmful", "D_unlabeled"]:
        loader = connected_loader.get_loader(split)
        assert hasattr(loader, "__iter__"), (
            f"get_loader('{split}') should return an iterable, got {type(loader)}"
        )


def test_val_dataloaders_keys(connected_loader):
    """val_dataloaders() returns a dict with exactly keys {'D_std', 'D_harmful'} — no D_unlabeled."""
    val = connected_loader.val_dataloaders()
    assert isinstance(val, dict), f"val_dataloaders() should return dict, got {type(val)}"
    assert set(val.keys()) == {"D_std", "D_harmful"}, (
        f"val_dataloaders() keys should be {{'D_std', 'D_harmful'}}, got {set(val.keys())}"
    )


def test_no_next_method(fake_cache, fake_tokenizer):
    """MultiDataLoader instance does NOT have a 'next' attribute."""
    mdl = MultiDataLoader(cache_dir=fake_cache, x=0, y=25)
    assert not hasattr(mdl, "next"), (
        "MultiDataLoader must NOT have a 'next' method — training loop manages its own iterators"
    )


def test_no_upsample_fields(fake_cache, fake_tokenizer):
    """MultiDataLoader instance does NOT have upsample_std, upsample_harmful, or upsample_unlabeled attributes."""
    mdl = MultiDataLoader(cache_dir=fake_cache, x=0, y=25)
    for attr in ["upsample_std", "upsample_harmful", "upsample_unlabeled"]:
        assert not hasattr(mdl, attr), (
            f"MultiDataLoader must NOT have '{attr}' — upsample weights belong in Phase 3 training loop"
        )


def test_train_dataloader_compat(connected_loader):
    """train_dataloader() returns a DataLoader (LightningDataModule compat)."""
    loader = connected_loader.train_dataloader()
    assert hasattr(loader, "__iter__"), (
        f"train_dataloader() should return an iterable, got {type(loader)}"
    )


def test_cache_path_uses_integer_format(fake_cache, fake_tokenizer, tmp_path):
    """setup() constructs paths using integer {x}-{y} (not floats like '0.0-25.0')."""
    mdl = MultiDataLoader(cache_dir=fake_cache, x=0, y=25, num_workers=0)
    mdl.connect(tokenizer=fake_tokenizer, batch_size=2, max_seq_length=8)
    # setup() should succeed only if it uses "0-25" (integer format)
    # If it used "0.0-25.0" it would fail to find the directory
    mdl.setup()
    # Verify that the integer-format path exists and was used
    expected_path = fake_cache / "test-tok" / "0-25"
    assert expected_path.exists(), f"Expected integer-format path {expected_path} to exist"
    # Confirm that float-format path was NOT created
    float_path = fake_cache / "test-tok" / "0.0-25.0"
    assert not float_path.exists(), f"Float-format path {float_path} should not exist"
