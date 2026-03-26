"""Data preparation pipeline for SafeMoE: per-dataset tokenization and ratio-based splitting.

Two-phase prepare:
  Phase 1 (expensive, once per dataset+tokenizer): tokenize all samples individually
  Phase 2 (cheap, re-run when ratios change): split by label_ratio and unlabel_ratio and repack with TokensLoader

Cache layout:
  data/.cache/{tokenizer_name}/{dataset_id}/
      manifest.json
      samples_train/          # Phase 1: per-sample token storage (no TokensLoader)
      val/                    # Phase 1: TokensLoader format (full val, no ratio split)
      splits/l{label_pct}_u{unlabel_pct}/
          labeled_train/      # Phase 2: TokensLoader format, first label_ratio samples
          unlabeled_train/    # Phase 2: TokensLoader format, next unlabel_ratio samples
"""

from __future__ import annotations

import json
import random
import shutil
from functools import partial
from pathlib import Path
from typing import Optional

from litdata import StreamingDataset, TokensLoader, optimize


DEFAULT_AUTO_VAL_RATIO = 0.01


# ---------------------------------------------------------------------------
# Helpers: text loading
# ---------------------------------------------------------------------------


def _discover_split_files(path: Path, split: str) -> list[Path]:
    path = Path(path)
    candidates: list[Path] = []

    patterns_by_split = {
        "train": [
            "train.parquet",
            "train.jsonl",
            "train-*.parquet",
            "train-*.jsonl",
            "shard_*.parquet",
            "shard_*.jsonl",
            "shard-*.parquet",
            "shard-*.jsonl",
        ],
        "val": [
            "val.parquet",
            "val.jsonl",
            "val-*.parquet",
            "val-*.jsonl",
            "validation.parquet",
            "validation.jsonl",
            "validation-*.parquet",
            "validation-*.jsonl",
        ],
        "validation": [
            "validation.parquet",
            "validation.jsonl",
            "validation-*.parquet",
            "validation-*.jsonl",
        ],
    }

    for pattern in patterns_by_split.get(split, [f"{split}.parquet", f"{split}.jsonl"]):
        matches = sorted(path.glob(pattern))
        for match in matches:
            if match.is_file() and match not in candidates:
                candidates.append(match)

    return candidates


def _load_texts_from_files(files: list[Path], text_column: str) -> list[str]:
    texts: list[str] = []
    for file_path in files:
        if file_path.suffix == ".parquet":
            import pandas as pd

            texts.extend(pd.read_parquet(file_path, columns=[text_column])[text_column].tolist())
            continue

        if file_path.suffix == ".jsonl":
            with file_path.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        texts.append(json.loads(line)[text_column])
            continue

        raise ValueError(f"Unsupported file type for {file_path}")
    return texts


def load_texts(path: Path, text_column: str, split: str) -> list[str]:
    """Load text strings from discovered parquet/jsonl files for a split."""
    files = _discover_split_files(path, split)
    if files:
        return _load_texts_from_files(files, text_column)

    if split == "val":
        return load_texts(path, text_column, "validation")

    raise FileNotFoundError(
        f"No files found for split '{split}' in {path} matching supported naming conventions"
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
    *,
    source_train_files: Optional[list[str]] = None,
    source_val_files: Optional[list[str]] = None,
    val_source: Optional[str] = None,
    seed: Optional[int] = None,
) -> None:
    manifest = {
        "dataset_id": dataset_id,
        "num_train_samples": num_train_samples,
        "num_val_samples": num_val_samples,
        "tokenizer": tokenizer_name,
        "source_path": str(source_path),
    }
    if source_train_files is not None:
        manifest["source_train_files"] = source_train_files
    if source_val_files is not None:
        manifest["source_val_files"] = source_val_files
    if val_source is not None:
        manifest["val_source"] = val_source
    if seed is not None:
        manifest["seed"] = seed
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



def _reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)



def _dataset_length(input_dir: Path) -> int:
    index_path = input_dir / "index.json"
    index_data = json.loads(index_path.read_text())
    return sum(chunk["chunk_size"] for chunk in index_data["chunks"])



def _prepare_text_splits(
    data_path: Path,
    text_column: str,
    seed: int,
    train_texts: Optional[list[str]] = None,
    val_texts: Optional[list[str]] = None,
    auto_val_ratio: float = DEFAULT_AUTO_VAL_RATIO,
) -> tuple[list[str], list[str], list[str], list[str], str]:
    train_files = [str(path) for path in _discover_split_files(data_path, "train")]

    base_train_texts = train_texts if train_texts is not None else load_texts(data_path, text_column, "train")

    if val_texts is not None:
        rng = random.Random(seed)
        shuffled_train = list(base_train_texts)
        rng.shuffle(shuffled_train)
        return shuffled_train, list(val_texts), train_files, [], "provided"

    val_files = [str(path) for path in _discover_split_files(data_path, "val")]
    if val_files:
        rng = random.Random(seed)
        shuffled_train = list(base_train_texts)
        rng.shuffle(shuffled_train)
        explicit_val_texts = load_texts(data_path, text_column, "val")
        return shuffled_train, explicit_val_texts, train_files, val_files, "explicit"

    shuffled_all = list(base_train_texts)
    rng = random.Random(seed)
    rng.shuffle(shuffled_all)

    if len(shuffled_all) <= 1:
        return shuffled_all, [], train_files, [], "synthetic"

    val_count = int(len(shuffled_all) * auto_val_ratio)
    val_count = max(1, val_count)
    val_count = min(val_count, len(shuffled_all) - 1)

    synthetic_val = shuffled_all[:val_count]
    shuffled_train = shuffled_all[val_count:]
    return shuffled_train, synthetic_val, train_files, [], "synthetic"


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
    unlabel_ratio: Optional[float] = None,
    # test injection hooks
    train_texts: Optional[list[str]] = None,
    val_texts: Optional[list[str]] = None,
) -> dict:
    """Two-phase prepare for a single dataset. Returns manifest dict."""
    data_path = Path(data_path)
    cache_dir = Path(cache_dir)
    if unlabel_ratio is None:
        unlabel_ratio = 1.0 - label_ratio
    if not 0.0 <= label_ratio <= 1.0:
        raise ValueError(f"label_ratio must be between 0 and 1, got {label_ratio}")
    if not 0.0 <= unlabel_ratio <= 1.0:
        raise ValueError(f"unlabel_ratio must be between 0 and 1, got {unlabel_ratio}")
    if label_ratio + unlabel_ratio > 1.0:
        raise ValueError(
            f"label_ratio + unlabel_ratio must be <= 1, got {label_ratio + unlabel_ratio}"
        )

    tokenizer_name = tokenizer.model_name if hasattr(tokenizer, "model_name") else "default"
    base_dir = cache_dir / tokenizer_name / dataset_id

    samples_train_dir = base_dir / "samples_train"
    val_dir = base_dir / "val"

    prepared_train_texts = None
    prepared_val_texts = None
    source_train_files: Optional[list[str]] = None
    source_val_files: Optional[list[str]] = None
    val_source = None

    needs_phase1_texts = not _dataset_is_ready(samples_train_dir) or not _dataset_is_ready(val_dir)
    if needs_phase1_texts:
        (
            prepared_train_texts,
            prepared_val_texts,
            source_train_files,
            source_val_files,
            val_source,
        ) = _prepare_text_splits(
            data_path=data_path,
            text_column=text_column,
            seed=seed,
            train_texts=train_texts,
            val_texts=val_texts,
        )

    # ---- Phase 1: tokenize full dataset (idempotent) ----------------------
    if not _dataset_is_ready(samples_train_dir):
        _reset_incomplete_output_dir(samples_train_dir)
        texts = prepared_train_texts if prepared_train_texts is not None else []
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
                # Use spawn to avoid forking a process that may already have
                # initialized distributed/CUDA state in the training launcher.
                # No item_loader: store per-sample for boundary preservation
                start_method="spawn",
            )
    else:
        num_train = _dataset_length(samples_train_dir)

    if not _dataset_is_ready(val_dir):
        _reset_incomplete_output_dir(val_dir)
        v_texts = prepared_val_texts if prepared_val_texts is not None else []
        num_val = _count_nonempty_texts(v_texts)

        if num_val == 0:
            _reset_output_dir(val_dir)
            val_dir.mkdir(parents=True, exist_ok=True)
            (val_dir / "index.json").write_text(json.dumps({"chunks": []}, indent=2))
        else:
            val_dir.mkdir(parents=True, exist_ok=True)
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
                start_method="spawn",
            )
    else:
        num_val = _dataset_length(val_dir)

    # Keep manifest aligned with the actual on-disk dataset lengths.
    manifest_should_write = not (base_dir / "manifest.json").exists()
    manifest = read_manifest(base_dir) if not manifest_should_write else {}
    manifest_updates = {
        "num_train_samples": num_train,
        "num_val_samples": num_val,
        "source_path": str(data_path),
        "source_train_files": source_train_files,
        "source_val_files": source_val_files,
        "val_source": val_source,
        "seed": seed,
    }
    for key, value in manifest_updates.items():
        if value is not None and manifest.get(key) != value:
            manifest_should_write = True
            break

    if manifest_should_write:
        write_manifest(
            base_dir,
            dataset_id,
            num_train,
            num_val,
            tokenizer_name,
            str(data_path),
            source_train_files=source_train_files,
            source_val_files=source_val_files,
            val_source=val_source,
            seed=seed,
        )

    # ---- Phase 2: split by label_ratio and unlabel_ratio (idempotent) -----
    label_pct = int(round(label_ratio * 100))
    unlabel_pct = int(round(unlabel_ratio * 100))
    split_dir = base_dir / "splits" / f"l{label_pct}_u{unlabel_pct}"
    labeled_train_dir = split_dir / "labeled_train"
    unlabeled_train_dir = split_dir / "unlabeled_train"

    manifest = read_manifest(base_dir)
    n = manifest["num_train_samples"]
    label_end = int(n * label_ratio)
    unlabel_end = label_end + int(n * unlabel_ratio)

    # Write labeled_train/
    if label_end > 0 and not _dataset_is_ready(labeled_train_dir):
        _reset_incomplete_output_dir(labeled_train_dir)
        labeled_train_dir.mkdir(parents=True, exist_ok=True)
        chunks = split_contiguous(range(label_end), min(num_workers, label_end))
        if chunks:
            optimize(
                fn=partial(_yield_range, input_dir=str(samples_train_dir)),
                inputs=chunks,
                output_dir=str(labeled_train_dir),
                num_workers=len(chunks),
                chunk_bytes=chunk_bytes,
                item_loader=TokensLoader(),
                start_method="spawn",
            )

    # Write unlabeled_train/
    unlabeled_count = unlabel_end - label_end
    if unlabeled_count > 0 and not _dataset_is_ready(unlabeled_train_dir):
        _reset_incomplete_output_dir(unlabeled_train_dir)
        unlabeled_train_dir.mkdir(parents=True, exist_ok=True)
        chunks = split_contiguous(range(label_end, unlabel_end), min(num_workers, unlabeled_count))
        if chunks:
            optimize(
                fn=partial(_yield_range, input_dir=str(samples_train_dir)),
                inputs=chunks,
                output_dir=str(unlabeled_train_dir),
                num_workers=len(chunks),
                chunk_bytes=chunk_bytes,
                item_loader=TokensLoader(),
                start_method="spawn",
            )

    return manifest
