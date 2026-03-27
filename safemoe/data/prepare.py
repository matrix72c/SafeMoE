"""Data preparation pipeline for SafeMoE: per-dataset tokenization and ratio-based splitting.

Cache layout:
  data/.cache/{tokenizer_name}/{dataset_id}/
      manifest.json
      samples_train/          # eager mode only: per-sample token storage (no TokensLoader)
      val/                    # TokensLoader format
      splits/l{label_pct}_u{unlabel_pct}/
          labeled_train/      # TokensLoader format
          unlabeled_train/    # TokensLoader format
"""

from __future__ import annotations

import json
import random
import shutil
from functools import partial
from pathlib import Path
from typing import Iterator

from litdata import StreamingDataset, TokensLoader, optimize

DEFAULT_AUTO_VAL_RATIO = 0.01
PREPARE_VERSION = 2


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


def _iter_texts_from_file(file_path: Path, text_column: str, *, batch_size: int = 8192) -> Iterator[str]:
    if file_path.suffix == ".parquet":
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(file_path)
        try:
            for batch in parquet_file.iter_batches(batch_size=batch_size, columns=[text_column]):
                column = batch.column(0)
                for value in column.to_pylist():
                    yield "" if value is None else str(value)
        finally:
            parquet_file.close()
        return

    if file_path.suffix == ".jsonl":
        with file_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)[text_column]
                yield "" if value is None else str(value)
        return

    raise ValueError(f"Unsupported file type for {file_path}")


def _iter_texts_from_files(files: list[Path], text_column: str, *, batch_size: int = 8192) -> Iterator[str]:
    for file_path in files:
        yield from _iter_texts_from_file(file_path, text_column, batch_size=batch_size)


def _load_texts_from_files(files: list[Path], text_column: str) -> list[str]:
    return list(_iter_texts_from_files(files, text_column))


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
    **extra_fields,
) -> None:
    manifest = {
        "dataset_id": dataset_id,
        "num_train_samples": num_train_samples,
        "num_val_samples": num_val_samples,
        "tokenizer": tokenizer_name,
        "source_path": str(source_path),
    }
    manifest.update({key: value for key, value in extra_fields.items() if value is not None})
    manifest_path = base_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


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


def _split_dir(base_dir: Path, label_ratio: float, unlabel_ratio: float) -> Path:
    label_pct = int(round(label_ratio * 100))
    unlabel_pct = int(round(unlabel_ratio * 100))
    return base_dir / "splits" / f"l{label_pct}_u{unlabel_pct}"


def _manifest_prepare_fields(
    *,
    prepare_mode: str,
    train_is_shuffled: bool,
    val_strategy: str,
    source_train_files: list[str],
    source_val_files: list[str],
    reserved_val_files: list[str],
    seed: int,
) -> dict:
    return {
        "prepare_version": PREPARE_VERSION,
        "prepare_mode": prepare_mode,
        "train_is_shuffled": train_is_shuffled,
        "val_strategy": val_strategy,
        "source_train_files": source_train_files,
        "source_val_files": source_val_files,
        "reserved_val_files": reserved_val_files,
        "seed": seed,
    }


def _manifest_requires_rebuild(manifest: dict, expected_fields: dict) -> bool:
    for key, value in expected_fields.items():
        if manifest.get(key) != value:
            return True
    return False


# ---------------------------------------------------------------------------
# Helpers: split planning
# ---------------------------------------------------------------------------


def _prepare_text_splits(
    data_path: Path,
    text_column: str,
    seed: int,
    train_texts: list[str] | None = None,
    val_texts: list[str] | None = None,
    auto_val_ratio: float = DEFAULT_AUTO_VAL_RATIO,
) -> tuple[list[str], list[str], list[str], list[str], list[str], str]:
    train_files = [str(path) for path in _discover_split_files(data_path, "train")]

    base_train_texts = train_texts if train_texts is not None else load_texts(data_path, text_column, "train")

    if val_texts is not None:
        rng = random.Random(seed)
        shuffled_train = list(base_train_texts)
        rng.shuffle(shuffled_train)
        return shuffled_train, list(val_texts), train_files, [], [], "provided"

    val_files = [str(path) for path in _discover_split_files(data_path, "val")]
    if val_files:
        rng = random.Random(seed)
        shuffled_train = list(base_train_texts)
        rng.shuffle(shuffled_train)
        explicit_val_texts = load_texts(data_path, text_column, "val")
        return shuffled_train, explicit_val_texts, train_files, val_files, val_files, "explicit"

    shuffled_all = list(base_train_texts)
    rng = random.Random(seed)
    rng.shuffle(shuffled_all)

    if len(shuffled_all) <= 1:
        return shuffled_all, [], train_files, [], [], "synthetic"

    val_count = int(len(shuffled_all) * auto_val_ratio)
    val_count = max(1, val_count)
    val_count = min(val_count, len(shuffled_all) - 1)

    synthetic_val = shuffled_all[:val_count]
    shuffled_train = shuffled_all[val_count:]
    return shuffled_train, synthetic_val, train_files, [], [], "synthetic"


def _plan_streaming_split_files(
    data_path: Path,
    *,
    val_strategy: str,
) -> tuple[list[Path], list[Path], list[str], list[str], str]:
    train_files = _discover_split_files(data_path, "train")
    if not train_files:
        raise FileNotFoundError(f"No files found for split 'train' in {data_path}")

    explicit_val_files = _discover_split_files(data_path, "val")
    if explicit_val_files:
        return train_files, explicit_val_files, [str(path) for path in train_files], [str(path) for path in explicit_val_files], "explicit"

    if val_strategy != "smallest_train_shard":
        raise ValueError(
            "streaming prepare requires explicit val files or val_strategy='smallest_train_shard'; "
            f"got {val_strategy!r}"
        )

    if len(train_files) <= 1:
        raise ValueError(
            "streaming prepare with val_strategy='smallest_train_shard' requires at least 2 train shards"
        )

    reserved_val = min(train_files, key=lambda path: (path.stat().st_size, path.name))
    remaining_train = [path for path in train_files if path != reserved_val]
    return (
        remaining_train,
        [reserved_val],
        [str(path) for path in train_files],
        [],
        "smallest_train_shard",
    )


def _count_valid_texts_in_files(files: list[Path], text_column: str, *, batch_size: int = 8192) -> int:
    count = 0
    for text in _iter_texts_from_files(files, text_column, batch_size=batch_size):
        if text.strip():
            count += 1
    return count


def _compute_file_valid_counts(files: list[Path], text_column: str, *, batch_size: int = 8192) -> list[int]:
    return [_count_valid_texts_in_files([file_path], text_column, batch_size=batch_size) for file_path in files]


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
    """Yield per-sample token tensors. Used for eager samples_train."""
    for text in texts:
        text = text.strip()
        if not text:
            continue
        tokens = tokenizer.encode(text, bos=True, eos=False)
        yield tokens


def _tokenize_texts_for_tokens_loader(texts: list[str], tokenizer) -> None:
    """Yield per-sample token tensors. Used with TokensLoader."""
    for text in texts:
        text = text.strip()
        if not text:
            continue
        tokens = tokenizer.encode(text, bos=True, eos=False)
        yield tokens


def _yield_range(index_range: list[int], input_dir: str) -> None:
    """Read samples by index from an eager samples dataset and yield them."""
    ds = StreamingDataset(input_dir=input_dir)
    for i in index_range:
        yield ds[i]


def _tokenize_file_for_tokens_loader(file_path_str: str, tokenizer, text_column: str, batch_size: int = 8192) -> None:
    file_path = Path(file_path_str)
    for text in _iter_texts_from_file(file_path, text_column, batch_size=batch_size):
        text = text.strip()
        if not text:
            continue
        yield tokenizer.encode(text, bos=True, eos=False)


def _tokenize_file_local_range(spec: dict, tokenizer, text_column: str, batch_size: int = 8192) -> None:
    file_path = Path(spec["file_path"])
    start = spec["start"]
    stop = spec["stop"]
    current = 0
    for text in _iter_texts_from_file(file_path, text_column, batch_size=batch_size):
        text = text.strip()
        if not text:
            continue
        if current >= stop:
            break
        if current >= start:
            yield tokenizer.encode(text, bos=True, eos=False)
        current += 1


# ---------------------------------------------------------------------------
# Internal prepare paths
# ---------------------------------------------------------------------------


def _prepare_dataset_eager(
    *,
    dataset_id: str,
    data_path: Path,
    text_column: str,
    label_ratio: float,
    unlabel_ratio: float,
    tokenizer,
    cache_dir: Path,
    seed: int,
    num_workers: int,
    chunk_bytes: str,
    train_texts: list[str] | None,
    val_texts: list[str] | None,
) -> dict:
    tokenizer_name = tokenizer.model_name if hasattr(tokenizer, "model_name") else "default"
    base_dir = cache_dir / tokenizer_name / dataset_id

    samples_train_dir = base_dir / "samples_train"
    val_dir = base_dir / "val"

    prepared_train_texts = None
    prepared_val_texts = None
    source_train_files: list[str] | None = None
    source_val_files: list[str] | None = None
    reserved_val_files: list[str] | None = None
    val_source = None

    discovered_train_files = [str(path) for path in _discover_split_files(data_path, "train")]
    discovered_val_files = [str(path) for path in _discover_split_files(data_path, "val")]
    inferred_val_strategy = "provided" if val_texts is not None else ("explicit" if discovered_val_files else "synthetic")
    inferred_reserved_val_files = discovered_val_files if inferred_val_strategy == "explicit" else []
    expected_prepare_fields = _manifest_prepare_fields(
        prepare_mode="eager",
        train_is_shuffled=False,
        val_strategy=inferred_val_strategy,
        source_train_files=discovered_train_files,
        source_val_files=discovered_val_files,
        reserved_val_files=inferred_reserved_val_files,
        seed=seed,
    )
    manifest_path = base_dir / "manifest.json"
    if manifest_path.exists() and _manifest_requires_rebuild(read_manifest(base_dir), expected_prepare_fields):
        _reset_output_dir(base_dir)

    needs_phase1_texts = not _dataset_is_ready(samples_train_dir) or not _dataset_is_ready(val_dir)
    if needs_phase1_texts:
        (
            prepared_train_texts,
            prepared_val_texts,
            source_train_files,
            source_val_files,
            reserved_val_files,
            val_source,
        ) = _prepare_text_splits(
            data_path=data_path,
            text_column=text_column,
            seed=seed,
            train_texts=train_texts,
            val_texts=val_texts,
        )
        expected_prepare_fields = _manifest_prepare_fields(
            prepare_mode="eager",
            train_is_shuffled=False,
            val_strategy=val_source,
            source_train_files=source_train_files,
            source_val_files=source_val_files,
            reserved_val_files=reserved_val_files,
            seed=seed,
        )

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

    label_end = int(num_train * label_ratio)
    unlabel_end = label_end + int(num_train * unlabel_ratio)
    unlabeled_count = unlabel_end - label_end
    dropped_count = num_train - unlabel_end

    manifest_should_write = not manifest_path.exists()
    manifest = read_manifest(base_dir) if manifest_path.exists() else {}
    manifest_updates = {
        **expected_prepare_fields,
        "label_ratio": label_ratio,
        "unlabel_ratio": unlabel_ratio,
        "num_labeled_train_samples": label_end,
        "num_unlabeled_train_samples": unlabeled_count,
        "num_dropped_train_samples": dropped_count,
        "val_source": val_source,
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
            **manifest_updates,
        )

    split_dir = _split_dir(base_dir, label_ratio, unlabel_ratio)
    labeled_train_dir = split_dir / "labeled_train"
    unlabeled_train_dir = split_dir / "unlabeled_train"

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

    if label_end == 0 and not labeled_train_dir.exists():
        labeled_train_dir.mkdir(parents=True, exist_ok=True)
        (labeled_train_dir / "index.json").write_text(json.dumps({"chunks": []}, indent=2))
    if unlabeled_count == 0 and not unlabeled_train_dir.exists():
        unlabeled_train_dir.mkdir(parents=True, exist_ok=True)
        (unlabeled_train_dir / "index.json").write_text(json.dumps({"chunks": []}, indent=2))

    return read_manifest(base_dir)


def _prepare_dataset_streaming(
    *,
    dataset_id: str,
    data_path: Path,
    text_column: str,
    label_ratio: float,
    unlabel_ratio: float,
    tokenizer,
    cache_dir: Path,
    seed: int,
    num_workers: int,
    chunk_bytes: str,
    train_is_shuffled: bool,
    val_strategy: str,
    scan_batch_size: int,
) -> dict:
    if not train_is_shuffled:
        raise ValueError("streaming prepare requires train_is_shuffled=True")

    tokenizer_name = tokenizer.model_name if hasattr(tokenizer, "model_name") else "default"
    base_dir = cache_dir / tokenizer_name / dataset_id
    val_dir = base_dir / "val"
    split_dir = _split_dir(base_dir, label_ratio, unlabel_ratio)
    labeled_train_dir = split_dir / "labeled_train"
    unlabeled_train_dir = split_dir / "unlabeled_train"

    train_files, val_files, source_train_files, source_val_files, effective_val_strategy = _plan_streaming_split_files(
        data_path,
        val_strategy=val_strategy,
    )
    reserved_val_files = [str(path) for path in val_files]

    expected_prepare_fields = _manifest_prepare_fields(
        prepare_mode="streaming",
        train_is_shuffled=train_is_shuffled,
        val_strategy=effective_val_strategy,
        source_train_files=source_train_files,
        source_val_files=source_val_files,
        reserved_val_files=reserved_val_files,
        seed=seed,
    )
    manifest_path = base_dir / "manifest.json"
    if manifest_path.exists() and _manifest_requires_rebuild(read_manifest(base_dir), expected_prepare_fields):
        _reset_output_dir(base_dir)

    train_file_counts = _compute_file_valid_counts(train_files, text_column, batch_size=scan_batch_size)
    num_train = sum(train_file_counts)
    num_val = _count_valid_texts_in_files(val_files, text_column, batch_size=scan_batch_size)

    label_end = int(num_train * label_ratio)
    unlabel_end = label_end + int(num_train * unlabel_ratio)
    unlabeled_count = unlabel_end - label_end
    dropped_count = num_train - unlabel_end

    if not _dataset_is_ready(val_dir):
        _reset_incomplete_output_dir(val_dir)
        if num_val == 0:
            _reset_output_dir(val_dir)
            val_dir.mkdir(parents=True, exist_ok=True)
            (val_dir / "index.json").write_text(json.dumps({"chunks": []}, indent=2))
        else:
            val_dir.mkdir(parents=True, exist_ok=True)
            optimize(
                fn=partial(
                    _tokenize_file_for_tokens_loader,
                    tokenizer=tokenizer,
                    text_column=text_column,
                    batch_size=scan_batch_size,
                ),
                inputs=[str(path) for path in val_files],
                output_dir=str(val_dir),
                num_workers=max(1, min(num_workers, len(val_files))),
                chunk_bytes=chunk_bytes,
                item_loader=TokensLoader(),
                start_method="spawn",
            )

    labeled_specs = []
    unlabeled_specs = []
    offset = 0
    for file_path, file_count in zip(train_files, train_file_counts):
        file_start = offset
        file_stop = offset + file_count

        labeled_start = max(0, file_start)
        labeled_stop = min(label_end, file_stop)
        if labeled_stop > labeled_start:
            labeled_specs.append(
                {
                    "file_path": str(file_path),
                    "start": labeled_start - file_start,
                    "stop": labeled_stop - file_start,
                }
            )

        unlabeled_start = max(label_end, file_start)
        unlabeled_stop = min(unlabel_end, file_stop)
        if unlabeled_stop > unlabeled_start:
            unlabeled_specs.append(
                {
                    "file_path": str(file_path),
                    "start": unlabeled_start - file_start,
                    "stop": unlabeled_stop - file_start,
                }
            )

        offset = file_stop

    if label_end > 0 and not _dataset_is_ready(labeled_train_dir):
        _reset_incomplete_output_dir(labeled_train_dir)
        labeled_train_dir.mkdir(parents=True, exist_ok=True)
        optimize(
            fn=partial(
                _tokenize_file_local_range,
                tokenizer=tokenizer,
                text_column=text_column,
                batch_size=scan_batch_size,
            ),
            inputs=labeled_specs,
            output_dir=str(labeled_train_dir),
            num_workers=max(1, min(num_workers, len(labeled_specs))),
            chunk_bytes=chunk_bytes,
            item_loader=TokensLoader(),
            start_method="spawn",
        )

    if unlabeled_count > 0 and not _dataset_is_ready(unlabeled_train_dir):
        _reset_incomplete_output_dir(unlabeled_train_dir)
        unlabeled_train_dir.mkdir(parents=True, exist_ok=True)
        optimize(
            fn=partial(
                _tokenize_file_local_range,
                tokenizer=tokenizer,
                text_column=text_column,
                batch_size=scan_batch_size,
            ),
            inputs=unlabeled_specs,
            output_dir=str(unlabeled_train_dir),
            num_workers=max(1, min(num_workers, len(unlabeled_specs))),
            chunk_bytes=chunk_bytes,
            item_loader=TokensLoader(),
            start_method="spawn",
        )

    if label_end == 0 and not labeled_train_dir.exists():
        labeled_train_dir.mkdir(parents=True, exist_ok=True)
        (labeled_train_dir / "index.json").write_text(json.dumps({"chunks": []}, indent=2))
    if unlabeled_count == 0 and not unlabeled_train_dir.exists():
        unlabeled_train_dir.mkdir(parents=True, exist_ok=True)
        (unlabeled_train_dir / "index.json").write_text(json.dumps({"chunks": []}, indent=2))

    manifest_updates = {
        **expected_prepare_fields,
        "label_ratio": label_ratio,
        "unlabel_ratio": unlabel_ratio,
        "num_labeled_train_samples": label_end,
        "num_unlabeled_train_samples": unlabeled_count,
        "num_dropped_train_samples": dropped_count,
        "val_source": effective_val_strategy,
    }
    write_manifest(
        base_dir,
        dataset_id,
        num_train,
        num_val,
        tokenizer_name,
        str(data_path),
        **manifest_updates,
    )
    return read_manifest(base_dir)


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
    unlabel_ratio: float | None = None,
    prepare_mode: str = "eager",
    train_is_shuffled: bool = False,
    val_strategy: str = "synthetic",
    scan_batch_size: int = 8192,
    # test injection hooks
    train_texts: list[str] | None = None,
    val_texts: list[str] | None = None,
) -> dict:
    """Prepare a single dataset for SafeMoE training/evaluation."""
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
    if prepare_mode not in {"eager", "streaming"}:
        raise ValueError(f"prepare_mode must be 'eager' or 'streaming', got {prepare_mode!r}")

    if prepare_mode == "streaming":
        return _prepare_dataset_streaming(
            dataset_id=dataset_id,
            data_path=data_path,
            text_column=text_column,
            label_ratio=label_ratio,
            unlabel_ratio=unlabel_ratio,
            tokenizer=tokenizer,
            cache_dir=cache_dir,
            seed=seed,
            num_workers=num_workers,
            chunk_bytes=chunk_bytes,
            train_is_shuffled=train_is_shuffled,
            val_strategy=val_strategy,
            scan_batch_size=scan_batch_size,
        )

    return _prepare_dataset_eager(
        dataset_id=dataset_id,
        data_path=data_path,
        text_column=text_column,
        label_ratio=label_ratio,
        unlabel_ratio=unlabel_ratio,
        tokenizer=tokenizer,
        cache_dir=cache_dir,
        seed=seed,
        num_workers=num_workers,
        chunk_bytes=chunk_bytes,
        train_texts=train_texts,
        val_texts=val_texts,
    )
