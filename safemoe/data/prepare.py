"""Data preparation pipeline for SafeMoE: per-dataset tokenization and ratio-based splitting.

Two-phase prepare:
  Phase 1 (expensive, once per dataset+tokenizer): tokenize all samples individually
  Phase 2 (cheap, re-run when ratio changes): split by label_ratio and repack with TokensLoader

Cache layout:
  data/.cache/{tokenizer_name}/{dataset_id}/
      manifest.json
      samples_train/       # Phase 1: per-sample token storage (no TokensLoader)
      val/                 # Phase 1: TokensLoader format (full val, no ratio split)
      splits/r{pct}/
          labeled_train/   # Phase 2: TokensLoader format, first ratio% samples
          unlabeled_train/ # Phase 2: TokensLoader format, remaining samples
"""

from __future__ import annotations

import json
import shutil
from functools import partial
from pathlib import Path
from typing import Optional

from litdata import StreamingDataset, optimize, TokensLoader


# ---------------------------------------------------------------------------
# Helpers: text loading
# ---------------------------------------------------------------------------


def load_texts(path: Path, text_column: str, split: str) -> list[str]:
    """Load text strings from parquet or jsonl files under path/{split}.*."""
    path = Path(path)
    # Try parquet first
    parquet_path = path / f"{split}.parquet"
    if parquet_path.exists():
        import pandas as pd
        return pd.read_parquet(parquet_path)[text_column].tolist()

    # Try jsonl
    jsonl_path = path / f"{split}.jsonl"
    if jsonl_path.exists():
        import json as _json
        texts = []
        for line in jsonl_path.read_text().splitlines():
            if line.strip():
                texts.append(_json.loads(line)[text_column])
        return texts

    # Try validation -> "validation" naming convention
    if split == "val":
        return load_texts(path, text_column, "validation")

    raise FileNotFoundError(
        f"No {split}.parquet or {split}.jsonl found in {path}"
    )


# ---------------------------------------------------------------------------
# Helpers: manifest
# ---------------------------------------------------------------------------


def write_manifest(
    base_dir: Path,
    dataset_id: str,
    num_train_samples: int,
    num_val_samples: int,
    tokenizer_name: str,
    source_path: str,
) -> None:
    manifest = {
        "dataset_id": dataset_id,
        "num_train_samples": num_train_samples,
        "num_val_samples": num_val_samples,
        "tokenizer": tokenizer_name,
        "source_path": str(source_path),
    }
    manifest_path = base_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))


def read_manifest(base_dir: Path) -> dict:
    manifest_path = base_dir / "manifest.json"
    return json.loads(manifest_path.read_text())


def _count_nonempty_texts(texts: list[str]) -> int:
    return sum(1 for text in texts if text and text.strip())


def _dataset_is_ready(output_dir: Path) -> bool:
    return (output_dir / "index.json").exists()


def _reset_incomplete_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and not _dataset_is_ready(output_dir):
        shutil.rmtree(output_dir)


def _dataset_length(input_dir: Path) -> int:
    index_path = input_dir / "index.json"
    index_data = json.loads(index_path.read_text())
    return sum(chunk["chunk_size"] for chunk in index_data["chunks"])


# ---------------------------------------------------------------------------
# Helpers: contiguous splitting for workers
# ---------------------------------------------------------------------------


def split_contiguous(indices: range, n_workers: int) -> list[list[int]]:
    """Split a contiguous index range into n non-overlapping chunks for workers."""
    total = len(indices)
    if total == 0:
        return []
    n_workers = min(n_workers, total)
    if n_workers < 1:
        n_workers = 1
    chunk_size = total // n_workers
    remainder = total % n_workers
    chunks = []
    start_val = indices.start
    for i in range(n_workers):
        size = chunk_size + (1 if i < remainder else 0)
        chunks.append(list(range(start_val, start_val + size)))
        start_val += size
    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Tokenization generators
# ---------------------------------------------------------------------------


def _tokenize_texts(texts: list[str], tokenizer) -> None:
    """Yield per-sample token tensors. Used for Phase 1 (no TokensLoader)."""
    for text in texts:
        text = text.strip()
        if not text:
            continue
        tokens = tokenizer.encode(text, bos=True, eos=False)
        yield tokens


def _tokenize_texts_for_tokens_loader(texts: list[str], tokenizer) -> None:
    """Yield per-sample token tensors. Used with TokensLoader for val."""
    for text in texts:
        text = text.strip()
        if not text:
            continue
        tokens = tokenizer.encode(text, bos=True, eos=False)
        yield tokens


def _yield_range(index_range: list[int], input_dir: str) -> None:
    """Read samples by index from a Phase 1 dataset and yield them for Phase 2."""
    ds = StreamingDataset(input_dir=input_dir)
    for i in index_range:
        yield ds[i]


# ---------------------------------------------------------------------------
# Public API: prepare_dataset
# ---------------------------------------------------------------------------


def prepare_dataset(
    dataset_id: str,
    data_path: Path,
    text_column: str,
    label_ratio: float,
    tokenizer,
    cache_dir: Path,
    seed: int = 42,
    num_workers: int = 4,
    chunk_bytes: str = "200MB",
    # test injection hooks
    train_texts: Optional[list[str]] = None,
    val_texts: Optional[list[str]] = None,
) -> dict:
    """Two-phase prepare for a single dataset. Returns manifest dict."""
    data_path = Path(data_path)
    cache_dir = Path(cache_dir)
    tokenizer_name = tokenizer.model_name if hasattr(tokenizer, "model_name") else "default"
    base_dir = cache_dir / tokenizer_name / dataset_id

    samples_train_dir = base_dir / "samples_train"
    val_dir = base_dir / "val"

    # ---- Phase 1: tokenize full dataset (idempotent) ----------------------
    if not _dataset_is_ready(samples_train_dir):
        _reset_incomplete_output_dir(samples_train_dir)
        texts = train_texts if train_texts is not None else load_texts(data_path, text_column, "train")
        num_train = _count_nonempty_texts(texts)

        samples_train_dir.mkdir(parents=True, exist_ok=True)
        if texts:
            effective_workers = min(num_workers, len(texts))
            if effective_workers < 1:
                effective_workers = 1
            inputs = [texts[i::effective_workers] for i in range(effective_workers)]
            inputs = [s for s in inputs if s]

            optimize(
                fn=partial(_tokenize_texts, tokenizer=tokenizer),
                inputs=inputs,
                output_dir=str(samples_train_dir),
                num_workers=len(inputs),
                chunk_bytes=chunk_bytes,
                # No item_loader: store per-sample for boundary preservation
                start_method="fork",
            )
    else:
        num_train = _dataset_length(samples_train_dir)

    if not _dataset_is_ready(val_dir):
        _reset_incomplete_output_dir(val_dir)
        v_texts = val_texts if val_texts is not None else load_texts(data_path, text_column, "val")
        num_val = _count_nonempty_texts(v_texts)

        val_dir.mkdir(parents=True, exist_ok=True)
        if v_texts:
            effective_workers = min(num_workers, len(v_texts))
            if effective_workers < 1:
                effective_workers = 1
            inputs = [v_texts[i::effective_workers] for i in range(effective_workers)]
            inputs = [s for s in inputs if s]

            optimize(
                fn=partial(_tokenize_texts_for_tokens_loader, tokenizer=tokenizer),
                inputs=inputs,
                output_dir=str(val_dir),
                num_workers=len(inputs),
                chunk_bytes=chunk_bytes,
                item_loader=TokensLoader(),
                start_method="fork",
            )
    else:
        num_val = _dataset_length(val_dir)

    # Keep manifest aligned with the actual on-disk dataset lengths.
    if not (base_dir / "manifest.json").exists():
        write_manifest(base_dir, dataset_id, num_train, num_val, tokenizer_name, str(data_path))
    else:
        manifest = read_manifest(base_dir)
        if (
            manifest.get("num_train_samples") != num_train
            or manifest.get("num_val_samples") != num_val
            or manifest.get("source_path") != str(data_path)
        ):
            write_manifest(base_dir, dataset_id, num_train, num_val, tokenizer_name, str(data_path))

    # ---- Phase 2: split by label_ratio (idempotent) -----------------------
    ratio_pct = int(round(label_ratio * 100))
    split_dir = base_dir / "splits" / f"r{ratio_pct}"
    labeled_train_dir = split_dir / "labeled_train"
    unlabeled_train_dir = split_dir / "unlabeled_train"

    manifest = read_manifest(base_dir)
    n = manifest["num_train_samples"]
    split_idx = int(n * label_ratio)

    # Write labeled_train/
    if split_idx > 0 and not _dataset_is_ready(labeled_train_dir):
        _reset_incomplete_output_dir(labeled_train_dir)
        labeled_train_dir.mkdir(parents=True, exist_ok=True)
        chunks = split_contiguous(range(split_idx), min(num_workers, split_idx))
        if chunks:
            optimize(
                fn=partial(_yield_range, input_dir=str(samples_train_dir)),
                inputs=chunks,
                output_dir=str(labeled_train_dir),
                num_workers=len(chunks),
                chunk_bytes=chunk_bytes,
                item_loader=TokensLoader(),
                start_method="fork",
            )

    # Write unlabeled_train/
    if split_idx < n and not _dataset_is_ready(unlabeled_train_dir):
        _reset_incomplete_output_dir(unlabeled_train_dir)
        unlabeled_train_dir.mkdir(parents=True, exist_ok=True)
        remaining = n - split_idx
        chunks = split_contiguous(range(split_idx, n), min(num_workers, remaining))
        if chunks:
            optimize(
                fn=partial(_yield_range, input_dir=str(samples_train_dir)),
                inputs=chunks,
                output_dir=str(unlabeled_train_dir),
                num_workers=len(chunks),
                chunk_bytes=chunk_bytes,
                item_loader=TokensLoader(),
                start_method="fork",
            )

    return manifest
