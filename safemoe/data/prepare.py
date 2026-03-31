"""Data preparation pipeline for SafeMoE: per-dataset tokenization and ratio-based splitting.

Cache layout:
  chunked mode:
    data/.cache/{tokenizer_name}/{dataset_id}/
        manifest.json
        train_chunks/           # reusable TokensLoader datasets
            chunk_000/
            chunk_001/
            ...
        partials/               # lazily materialized boundary ranges
            g0_123/
            g123_456/
        val/                    # TokensLoader format
        splits/l{label_pct}_u{unlabel_pct}/
            labeled_train/      # TokensLoader format
            unlabeled_train/    # TokensLoader format

  in_memory mode:
    data/.cache/{tokenizer_name}/{dataset_id}/
        val/                    # shared TokensLoader format
        l{label_pct}_u{unlabel_pct}/
            manifest.json
            labeled_train/      # TokensLoader format
            unlabeled_train/    # TokensLoader format
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
from functools import partial
from pathlib import Path
from typing import Iterator

from litdata import TokensLoader, optimize

DEFAULT_AUTO_VAL_RATIO = 0.01
PREPARE_VERSION = 5
DEFAULT_TARGET_CHUNK_BYTES = "8GB"
DEFAULT_MIN_CHUNK_COUNT = 8
DEFAULT_MAX_CHUNK_COUNT = 64


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


def _dataset_is_ready(output_dir: Path) -> bool:
    return (output_dir / "index.json").exists()


def _reset_incomplete_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and not _dataset_is_ready(output_dir):
        shutil.rmtree(output_dir)


def _reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)


def _ratio_dir_name(label_ratio: float, unlabel_ratio: float) -> str:
    label_pct = int(round(label_ratio * 100))
    unlabel_pct = int(round(unlabel_ratio * 100))
    return f"l{label_pct}_u{unlabel_pct}"


def _split_dir(base_dir: Path, label_ratio: float, unlabel_ratio: float) -> Path:
    return base_dir / "splits" / _ratio_dir_name(label_ratio, unlabel_ratio)


def _in_memory_ratio_dir(base_dir: Path, label_ratio: float, unlabel_ratio: float) -> Path:
    return base_dir / _ratio_dir_name(label_ratio, unlabel_ratio)


def _manifest_prepare_fields(
    *,
    prepare_mode: str,
    train_is_shuffled: bool,
    val_strategy: str,
    source_train_files: list[str],
    source_val_files: list[str],
    reserved_val_files: list[str],
    seed: int,
    val_ratio: float | None = None,
    scan_batch_size: int | None = None,
    target_chunk_bytes: str | None = None,
    min_chunk_count: int | None = None,
    max_chunk_count: int | None = None,
    train_chunk_count: int | None = None,
    train_chunk_ranges: list[dict[str, int]] | None = None,
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
        "val_ratio": val_ratio,
        "scan_batch_size": scan_batch_size,
        "target_chunk_bytes": target_chunk_bytes,
        "min_chunk_count": min_chunk_count,
        "max_chunk_count": max_chunk_count,
        "train_chunk_count": train_chunk_count,
        "train_chunk_ranges": train_chunk_ranges,
    }


def _manifest_requires_rebuild(manifest: dict, expected_fields: dict) -> bool:
    for key, value in expected_fields.items():
        if manifest.get(key) != value:
            return True
    return False


def _manifest_should_write(manifest: dict, updates: dict) -> bool:
    if not manifest:
        return True
    for key, value in updates.items():
        if value is not None and manifest.get(key) != value:
            return True
    return False


def _prepared_outputs_are_ready(output_dirs: list[Path]) -> bool:
    return all(_dataset_is_ready(output_dir) for output_dir in output_dirs)


def _fast_path_manifest_matches(manifest: dict, expected_fields: dict) -> bool:
    return not _manifest_requires_rebuild(manifest, expected_fields)


# ---------------------------------------------------------------------------
# Helpers: split planning
# ---------------------------------------------------------------------------


def _prepare_text_splits(
    data_path: Path,
    text_column: str,
    seed: int,
    train_texts: list[str] | None = None,
    val_texts: list[str] | None = None,
    val_ratio: float = DEFAULT_AUTO_VAL_RATIO,
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

    val_count = int(len(shuffled_all) * val_ratio)
    val_count = max(1, val_count)
    val_count = min(val_count, len(shuffled_all) - 1)

    synthetic_val = shuffled_all[:val_count]
    shuffled_train = shuffled_all[val_count:]
    return shuffled_train, synthetic_val, train_files, [], [], "synthetic"


def _count_valid_texts_in_files(files: list[Path], text_column: str, *, batch_size: int = 8192) -> int:
    count = 0
    for text in _iter_texts_from_files(files, text_column, batch_size=batch_size):
        if text.strip():
            count += 1
    return count


def _compute_file_valid_counts(files: list[Path], text_column: str, *, batch_size: int = 8192) -> list[int]:
    return [_count_valid_texts_in_files([file_path], text_column, batch_size=batch_size) for file_path in files]


def _compute_split_counts(num_train: int, label_ratio: float, unlabel_ratio: float) -> tuple[int, int, int, int]:
    label_end = int(num_train * label_ratio)
    unlabel_end = label_end + int(num_train * unlabel_ratio)
    unlabeled_count = unlabel_end - label_end
    dropped_count = num_train - unlabel_end
    return label_end, unlabel_end, unlabeled_count, dropped_count


def _filter_nonempty_texts(texts: list[str]) -> list[str]:
    return [text for text in texts if text and text.strip()]


def _parse_bytes(value: str | int) -> int:
    if isinstance(value, int):
        return value

    normalized = str(value).strip().upper()
    units = [
        ("TB", 1000**4),
        ("GB", 1000**3),
        ("MB", 1000**2),
        ("KB", 1000),
        ("B", 1),
    ]
    for suffix, multiplier in units:
        if normalized.endswith(suffix):
            number = normalized[: -len(suffix)].strip()
            return int(float(number) * multiplier)
    return int(float(normalized))


def _estimate_train_chunk_count(
    *,
    total_train_bytes: int,
    num_train: int,
    target_chunk_bytes: str,
    min_chunk_count: int,
    max_chunk_count: int,
) -> int:
    if num_train <= 0:
        return 0

    estimated = math.ceil(total_train_bytes / max(_parse_bytes(target_chunk_bytes), 1)) if total_train_bytes > 0 else 1
    estimated = max(min_chunk_count, estimated)
    estimated = min(max_chunk_count, estimated)
    return max(1, min(num_train, estimated))


def _plan_chunk_global_ranges(num_train: int, train_chunk_count: int) -> list[dict[str, int]]:
    if num_train <= 0 or train_chunk_count <= 0:
        return []

    chunk_size = math.ceil(num_train / train_chunk_count)
    ranges = []
    start = 0
    while start < num_train:
        stop = min(num_train, start + chunk_size)
        ranges.append({"start": start, "stop": stop})
        start = stop
    return ranges


def _build_file_range_specs(files: list[Path], file_valid_counts: list[int], chunk_ranges: list[dict[str, int]]) -> list[dict]:
    chunk_specs: list[dict] = []
    file_offset = 0
    file_index = 0

    for chunk_index, chunk_range in enumerate(chunk_ranges):
        start = chunk_range["start"]
        stop = chunk_range["stop"]
        local_specs = []
        current = start

        while current < stop and file_index < len(files):
            file_count = file_valid_counts[file_index]
            file_start = file_offset
            file_stop = file_offset + file_count

            if current >= file_stop:
                file_offset = file_stop
                file_index += 1
                continue

            local_start = max(current, file_start) - file_start
            local_stop = min(stop, file_stop) - file_start
            if local_stop > local_start:
                local_specs.append(
                    {
                        "file_path": str(files[file_index]),
                        "start": local_start,
                        "stop": local_stop,
                    }
                )
                current = file_start + local_stop
            else:
                current = file_stop

            if current >= file_stop:
                file_offset = file_stop
                file_index += 1

        chunk_specs.append(
            {
                "chunk_index": chunk_index,
                "start": start,
                "stop": stop,
                "count": stop - start,
                "file_specs": local_specs,
            }
        )

    return chunk_specs


def _build_text_range_specs(texts: list[str], chunk_ranges: list[dict[str, int]]) -> list[dict]:
    chunk_specs = []
    for chunk_index, chunk_range in enumerate(chunk_ranges):
        start = chunk_range["start"]
        stop = chunk_range["stop"]
        chunk_specs.append(
            {
                "chunk_index": chunk_index,
                "start": start,
                "stop": stop,
                "count": stop - start,
                "texts": texts[start:stop],
            }
        )
    return chunk_specs


def _plan_split_from_chunks(chunk_specs: list[dict], *, label_end: int, unlabel_end: int) -> dict[str, list[dict]]:
    return {
        "labeled_complete": [chunk for chunk in chunk_specs if chunk["start"] >= 0 and chunk["stop"] <= label_end],
        "unlabeled_complete": [chunk for chunk in chunk_specs if chunk["start"] >= label_end and chunk["stop"] <= unlabel_end],
        "labeled_partial": [
            {"start": max(chunk["start"], 0), "stop": min(chunk["stop"], label_end)}
            for chunk in chunk_specs
            if min(chunk["stop"], label_end) > max(chunk["start"], 0)
            and not (chunk["start"] >= 0 and chunk["stop"] <= label_end)
        ],
        "unlabeled_partial": [
            {"start": max(chunk["start"], label_end), "stop": min(chunk["stop"], unlabel_end)}
            for chunk in chunk_specs
            if min(chunk["stop"], unlabel_end) > max(chunk["start"], label_end)
            and not (chunk["start"] >= label_end and chunk["stop"] <= unlabel_end)
        ],
    }


# ---------------------------------------------------------------------------
# Tokenization generators
# ---------------------------------------------------------------------------


def _tokenize_texts_for_tokens_loader(texts: list[str], tokenizer) -> None:
    for text in texts:
        text = text.strip()
        if not text:
            continue
        yield tokenizer.encode(text, bos=True, eos=False)


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
# Helpers: dataset materialization and linking
# ---------------------------------------------------------------------------


def _write_empty_dataset(output_dir: Path) -> None:
    _reset_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.json").write_text(json.dumps({"chunks": []}, indent=2))


def _optimize_to_dir(*, fn, inputs: list, output_dir: Path, num_workers: int, chunk_bytes: str, item_loader=None) -> None:
    if not inputs:
        _write_empty_dataset(output_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    optimize(
        fn=fn,
        inputs=inputs,
        output_dir=str(output_dir),
        num_workers=max(1, min(num_workers, len(inputs))),
        chunk_bytes=chunk_bytes,
        item_loader=item_loader,
        start_method="spawn",
    )


def _read_index(input_dir: Path) -> dict:
    return json.loads((input_dir / "index.json").read_text())


def _link_or_copy_file(src: Path, dest: Path) -> None:
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        os.symlink(src.resolve(), dest)
    except OSError:
        try:
            os.link(src, dest)
        except OSError:
            shutil.copy2(src, dest)


def _materialize_merged_dataset(output_dir: Path, source_dirs: list[Path]) -> None:
    if _dataset_is_ready(output_dir):
        return
    _reset_incomplete_output_dir(output_dir)
    if not source_dirs:
        _write_empty_dataset(output_dir)
        return

    _reset_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_chunks = []
    merged_config = None
    merged_updated_at = None
    next_file_index = 0

    for source_dir in source_dirs:
        index_data = _read_index(source_dir)
        if merged_config is None:
            merged_config = index_data.get("config")
        elif index_data.get("config") is not None and merged_config != index_data.get("config"):
            raise ValueError(f"Mismatched litdata config while materializing split from {source_dir}")
        if merged_updated_at is None and "updated_at" in index_data:
            merged_updated_at = index_data["updated_at"]

        for chunk in index_data.get("chunks", []):
            src_file = source_dir / chunk["filename"]
            dest_name = f"part-{next_file_index:06d}{src_file.suffix}"
            next_file_index += 1
            _link_or_copy_file(src_file, output_dir / dest_name)
            merged_chunk = dict(chunk)
            merged_chunk["filename"] = dest_name
            merged_chunks.append(merged_chunk)

    output = {"chunks": merged_chunks}
    if merged_config is not None:
        output["config"] = merged_config
    if merged_updated_at is not None:
        output["updated_at"] = merged_updated_at
    (output_dir / "index.json").write_text(json.dumps(output, indent=2, sort_keys=True))


def _materialize_file_range_dataset(
    *,
    output_dir: Path,
    file_specs: list[dict],
    tokenizer,
    text_column: str,
    scan_batch_size: int,
    num_workers: int,
    chunk_bytes: str,
) -> None:
    if _dataset_is_ready(output_dir):
        return
    _reset_incomplete_output_dir(output_dir)
    if not file_specs:
        _write_empty_dataset(output_dir)
        return
    _optimize_to_dir(
        fn=partial(
            _tokenize_file_local_range,
            tokenizer=tokenizer,
            text_column=text_column,
            batch_size=scan_batch_size,
        ),
        inputs=file_specs,
        output_dir=output_dir,
        num_workers=num_workers,
        chunk_bytes=chunk_bytes,
        item_loader=TokensLoader(),
    )


def _materialize_text_range_dataset(
    *,
    output_dir: Path,
    texts: list[str],
    tokenizer,
    num_workers: int,
    chunk_bytes: str,
) -> None:
    if _dataset_is_ready(output_dir):
        return
    _reset_incomplete_output_dir(output_dir)
    if not texts:
        _write_empty_dataset(output_dir)
        return
    worker_count = max(1, min(num_workers, len(texts)))
    inputs = [texts[i::worker_count] for i in range(worker_count)]
    inputs = [chunk for chunk in inputs if chunk]
    _optimize_to_dir(
        fn=partial(_tokenize_texts_for_tokens_loader, tokenizer=tokenizer),
        inputs=inputs,
        output_dir=output_dir,
        num_workers=len(inputs),
        chunk_bytes=chunk_bytes,
        item_loader=TokensLoader(),
    )


def _materialize_chunked_split_views(
    *,
    split_dir: Path,
    plan: dict[str, list[dict]],
    full_chunk_dir_for_range,
    partial_dir_for_range,
) -> None:
    labeled_sources = [full_chunk_dir_for_range(chunk["start"], chunk["stop"]) for chunk in plan["labeled_complete"]]
    labeled_sources.extend(partial_dir_for_range(spec["start"], spec["stop"]) for spec in plan["labeled_partial"])

    unlabeled_sources = [full_chunk_dir_for_range(chunk["start"], chunk["stop"]) for chunk in plan["unlabeled_complete"]]
    unlabeled_sources.extend(partial_dir_for_range(spec["start"], spec["stop"]) for spec in plan["unlabeled_partial"])

    _materialize_merged_dataset(split_dir / "labeled_train", labeled_sources)
    _materialize_merged_dataset(split_dir / "unlabeled_train", unlabeled_sources)


# ---------------------------------------------------------------------------
# Internal prepare paths
# ---------------------------------------------------------------------------


def _prepare_dataset_in_memory(
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
    val_ratio: float,
    target_chunk_bytes: str,
    min_chunk_count: int,
    max_chunk_count: int,
    train_texts: list[str] | None,
    val_texts: list[str] | None,
) -> dict:
    del target_chunk_bytes, min_chunk_count, max_chunk_count

    if not train_is_shuffled:
        raise ValueError("in_memory prepare requires train_is_shuffled=True")

    tokenizer_name = tokenizer.model_name if hasattr(tokenizer, "model_name") else "default"
    base_dir = cache_dir / tokenizer_name / dataset_id
    val_dir = base_dir / "val"
    ratio_dir = _in_memory_ratio_dir(base_dir, label_ratio, unlabel_ratio)

    manifest_path = ratio_dir / "manifest.json"
    fast_path_expected_fields = {
        "prepare_version": PREPARE_VERSION,
        "prepare_mode": "in_memory",
        "source_path": str(data_path),
        "label_ratio": label_ratio,
        "unlabel_ratio": unlabel_ratio,
        "train_is_shuffled": train_is_shuffled,
        "seed": seed,
    }
    if manifest_path.exists():
        manifest = read_manifest(ratio_dir)
        if _fast_path_manifest_matches(manifest, fast_path_expected_fields) and _prepared_outputs_are_ready(
            [val_dir, ratio_dir / "labeled_train", ratio_dir / "unlabeled_train"]
        ):
            return manifest

    prepared_train_texts, prepared_val_texts, source_train_files, source_val_files, reserved_val_files, val_source = _prepare_text_splits(
        data_path=data_path,
        text_column=text_column,
        seed=seed,
        train_texts=train_texts,
        val_texts=val_texts,
        val_ratio=val_ratio,
    )
    filtered_train_texts = _filter_nonempty_texts(prepared_train_texts)
    filtered_val_texts = _filter_nonempty_texts(prepared_val_texts)
    num_train = len(filtered_train_texts)
    num_val = len(filtered_val_texts)
    label_end, unlabel_end, unlabeled_count, dropped_count = _compute_split_counts(num_train, label_ratio, unlabel_ratio)

    expected_prepare_fields = {
        "prepare_version": PREPARE_VERSION,
        "prepare_mode": "in_memory",
        "label_ratio": label_ratio,
        "unlabel_ratio": unlabel_ratio,
        "train_is_shuffled": train_is_shuffled,
        "val_strategy": val_source,
        "val_source": val_source,
        "seed": seed,
        "val_ratio": val_ratio if val_source == "synthetic" else None,
        "source_train_files": source_train_files,
        "source_val_files": source_val_files,
        "reserved_val_files": reserved_val_files,
    }

    manifest_path = ratio_dir / "manifest.json"
    if manifest_path.exists() and _manifest_requires_rebuild(read_manifest(ratio_dir), expected_prepare_fields):
        _reset_output_dir(ratio_dir)

    if not _dataset_is_ready(val_dir):
        _materialize_text_range_dataset(
            output_dir=val_dir,
            texts=filtered_val_texts,
            tokenizer=tokenizer,
            num_workers=num_workers,
            chunk_bytes=chunk_bytes,
        )

    _materialize_text_range_dataset(
        output_dir=ratio_dir / "labeled_train",
        texts=filtered_train_texts[:label_end],
        tokenizer=tokenizer,
        num_workers=num_workers,
        chunk_bytes=chunk_bytes,
    )
    _materialize_text_range_dataset(
        output_dir=ratio_dir / "unlabeled_train",
        texts=filtered_train_texts[label_end:unlabel_end],
        tokenizer=tokenizer,
        num_workers=num_workers,
        chunk_bytes=chunk_bytes,
    )

    manifest = read_manifest(ratio_dir) if manifest_path.exists() else {}
    manifest_updates = {
        **expected_prepare_fields,
        "num_labeled_train_samples": label_end,
        "num_unlabeled_train_samples": unlabeled_count,
        "num_dropped_train_samples": dropped_count,
    }
    if _manifest_should_write(manifest, manifest_updates):
        write_manifest(
            ratio_dir,
            dataset_id,
            num_train,
            num_val,
            tokenizer_name,
            str(data_path),
            **manifest_updates,
        )

    return read_manifest(ratio_dir)


def _prepare_dataset_chunked(
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
    val_ratio: float,
    scan_batch_size: int,
    target_chunk_bytes: str,
    min_chunk_count: int,
    max_chunk_count: int,
) -> dict:
    if not train_is_shuffled:
        raise ValueError("chunked prepare requires train_is_shuffled=True")

    tokenizer_name = tokenizer.model_name if hasattr(tokenizer, "model_name") else "default"
    base_dir = cache_dir / tokenizer_name / dataset_id
    train_chunks_dir = base_dir / "train_chunks"
    partials_dir = base_dir / "partials"
    val_dir = base_dir / "val"
    split_dir = _split_dir(base_dir, label_ratio, unlabel_ratio)

    manifest_path = base_dir / "manifest.json"
    fast_path_expected_fields = {
        "prepare_version": PREPARE_VERSION,
        "prepare_mode": "chunked",
        "source_path": str(data_path),
        "label_ratio": label_ratio,
        "unlabel_ratio": unlabel_ratio,
        "train_is_shuffled": train_is_shuffled,
        "val_strategy": val_strategy,
        "seed": seed,
        "scan_batch_size": scan_batch_size,
        "target_chunk_bytes": target_chunk_bytes,
        "min_chunk_count": min_chunk_count,
        "max_chunk_count": max_chunk_count,
    }
    if manifest_path.exists():
        manifest = read_manifest(base_dir)
        if _fast_path_manifest_matches(manifest, fast_path_expected_fields) and _prepared_outputs_are_ready(
            [val_dir, split_dir / "labeled_train", split_dir / "unlabeled_train"]
        ):
            return manifest

    explicit_val_files = _discover_split_files(data_path, "val")
    if explicit_val_files or val_strategy == "smallest_train_shard":
        train_files = _discover_split_files(data_path, "train")
        if not train_files:
            raise FileNotFoundError(f"No files found for split 'train' in {data_path}")
        source_train_files = [str(path) for path in train_files]

        if explicit_val_files:
            val_files = explicit_val_files
            source_val_files = [str(path) for path in val_files]
            reserved_val_files = source_val_files
            val_source = "explicit"
        else:
            if len(train_files) <= 1:
                raise ValueError(
                    "chunked prepare with val_strategy='smallest_train_shard' requires at least 2 train shards"
                )
            reserved_val = min(train_files, key=lambda path: (path.stat().st_size, path.name))
            val_files = [reserved_val]
            train_files = [path for path in train_files if path != reserved_val]
            source_val_files = []
            reserved_val_files = [str(reserved_val)]
            val_source = "smallest_train_shard"

        train_file_counts = _compute_file_valid_counts(train_files, text_column, batch_size=scan_batch_size)
        num_train = sum(train_file_counts)
        num_val = _count_valid_texts_in_files(val_files, text_column, batch_size=scan_batch_size)
        total_train_bytes = sum(path.stat().st_size for path in train_files)
        train_chunk_count = _estimate_train_chunk_count(
            total_train_bytes=total_train_bytes,
            num_train=num_train,
            target_chunk_bytes=target_chunk_bytes,
            min_chunk_count=min_chunk_count,
            max_chunk_count=max_chunk_count,
        )
        chunk_ranges = _plan_chunk_global_ranges(num_train, train_chunk_count)
        train_chunk_specs = _build_file_range_specs(train_files, train_file_counts, chunk_ranges)
        chunk_range_manifest = [{"start": spec["start"], "stop": spec["stop"]} for spec in train_chunk_specs]

        expected_prepare_fields = _manifest_prepare_fields(
            prepare_mode="chunked",
            train_is_shuffled=train_is_shuffled,
            val_strategy=val_source,
            source_train_files=source_train_files,
            source_val_files=source_val_files,
            reserved_val_files=reserved_val_files,
            seed=seed,
            scan_batch_size=scan_batch_size,
            target_chunk_bytes=target_chunk_bytes,
            min_chunk_count=min_chunk_count,
            max_chunk_count=max_chunk_count,
            train_chunk_count=len(train_chunk_specs),
            train_chunk_ranges=chunk_range_manifest,
        )

        manifest_path = base_dir / "manifest.json"
        if manifest_path.exists() and _manifest_requires_rebuild(read_manifest(base_dir), expected_prepare_fields):
            _reset_output_dir(base_dir)

        if not _dataset_is_ready(val_dir):
            if num_val == 0:
                _write_empty_dataset(val_dir)
            else:
                _optimize_to_dir(
                    fn=partial(
                        _tokenize_file_for_tokens_loader,
                        tokenizer=tokenizer,
                        text_column=text_column,
                        batch_size=scan_batch_size,
                    ),
                    inputs=[str(path) for path in val_files],
                    output_dir=val_dir,
                    num_workers=num_workers,
                    chunk_bytes=chunk_bytes,
                    item_loader=TokensLoader(),
                )

        chunk_by_range = {(spec["start"], spec["stop"]): spec for spec in train_chunk_specs}
        for spec in train_chunk_specs:
            chunk_dir = train_chunks_dir / f"chunk_{spec['chunk_index']:03d}"
            _materialize_file_range_dataset(
                output_dir=chunk_dir,
                file_specs=spec["file_specs"],
                tokenizer=tokenizer,
                text_column=text_column,
                scan_batch_size=scan_batch_size,
                num_workers=num_workers,
                chunk_bytes=chunk_bytes,
            )

        def full_chunk_dir_for_range(start: int, stop: int) -> Path:
            spec = chunk_by_range[(start, stop)]
            return train_chunks_dir / f"chunk_{spec['chunk_index']:03d}"

        def partial_dir_for_range(start: int, stop: int) -> Path:
            output_dir = partials_dir / f"g{start}_{stop}"
            if _dataset_is_ready(output_dir):
                return output_dir
            partial_specs = []
            for chunk_spec in train_chunk_specs:
                overlap_start = max(start, chunk_spec["start"])
                overlap_stop = min(stop, chunk_spec["stop"])
                if overlap_stop <= overlap_start:
                    continue
                chunk_local_start = overlap_start - chunk_spec["start"]
                chunk_local_stop = overlap_stop - chunk_spec["start"]
                cursor = 0
                for file_spec in chunk_spec["file_specs"]:
                    span = file_spec["stop"] - file_spec["start"]
                    overlap_local_start = max(chunk_local_start, cursor)
                    overlap_local_stop = min(chunk_local_stop, cursor + span)
                    if overlap_local_stop > overlap_local_start:
                        partial_specs.append(
                            {
                                "file_path": file_spec["file_path"],
                                "start": file_spec["start"] + (overlap_local_start - cursor),
                                "stop": file_spec["start"] + (overlap_local_stop - cursor),
                            }
                        )
                    cursor += span
            _materialize_file_range_dataset(
                output_dir=output_dir,
                file_specs=partial_specs,
                tokenizer=tokenizer,
                text_column=text_column,
                scan_batch_size=scan_batch_size,
                num_workers=num_workers,
                chunk_bytes=chunk_bytes,
            )
            return output_dir
    else:
        prepared_train_texts, prepared_val_texts, source_train_files, source_val_files, reserved_val_files, val_source = _prepare_text_splits(
            data_path=data_path,
            text_column=text_column,
            seed=seed,
            val_ratio=val_ratio,
        )
        filtered_train_texts = _filter_nonempty_texts(prepared_train_texts)
        filtered_val_texts = _filter_nonempty_texts(prepared_val_texts)
        num_train = len(filtered_train_texts)
        num_val = len(filtered_val_texts)
        total_train_bytes = sum(len(text.encode("utf-8")) for text in filtered_train_texts)
        train_chunk_count = _estimate_train_chunk_count(
            total_train_bytes=total_train_bytes,
            num_train=num_train,
            target_chunk_bytes=target_chunk_bytes,
            min_chunk_count=min_chunk_count,
            max_chunk_count=max_chunk_count,
        )
        chunk_ranges = _plan_chunk_global_ranges(num_train, train_chunk_count)
        train_chunk_specs = _build_text_range_specs(filtered_train_texts, chunk_ranges)
        chunk_range_manifest = [{"start": spec["start"], "stop": spec["stop"]} for spec in train_chunk_specs]

        expected_prepare_fields = _manifest_prepare_fields(
            prepare_mode="chunked",
            train_is_shuffled=train_is_shuffled,
            val_strategy=val_source,
            source_train_files=source_train_files,
            source_val_files=source_val_files,
            reserved_val_files=reserved_val_files,
            seed=seed,
            val_ratio=val_ratio if val_source == "synthetic" else None,
            scan_batch_size=scan_batch_size,
            target_chunk_bytes=target_chunk_bytes,
            min_chunk_count=min_chunk_count,
            max_chunk_count=max_chunk_count,
            train_chunk_count=len(train_chunk_specs),
            train_chunk_ranges=chunk_range_manifest,
        )

        manifest_path = base_dir / "manifest.json"
        if manifest_path.exists() and _manifest_requires_rebuild(read_manifest(base_dir), expected_prepare_fields):
            _reset_output_dir(base_dir)

        if not _dataset_is_ready(val_dir):
            _materialize_text_range_dataset(
                output_dir=val_dir,
                texts=filtered_val_texts,
                tokenizer=tokenizer,
                num_workers=num_workers,
                chunk_bytes=chunk_bytes,
            )

        for spec in train_chunk_specs:
            chunk_dir = train_chunks_dir / f"chunk_{spec['chunk_index']:03d}"
            _materialize_text_range_dataset(
                output_dir=chunk_dir,
                texts=spec["texts"],
                tokenizer=tokenizer,
                num_workers=num_workers,
                chunk_bytes=chunk_bytes,
            )

        chunk_by_range = {(spec["start"], spec["stop"]): spec for spec in train_chunk_specs}

        def full_chunk_dir_for_range(start: int, stop: int) -> Path:
            spec = chunk_by_range[(start, stop)]
            return train_chunks_dir / f"chunk_{spec['chunk_index']:03d}"

        def partial_dir_for_range(start: int, stop: int) -> Path:
            output_dir = partials_dir / f"g{start}_{stop}"
            _materialize_text_range_dataset(
                output_dir=output_dir,
                texts=filtered_train_texts[start:stop],
                tokenizer=tokenizer,
                num_workers=num_workers,
                chunk_bytes=chunk_bytes,
            )
            return output_dir

    label_end, unlabel_end, unlabeled_count, dropped_count = _compute_split_counts(num_train, label_ratio, unlabel_ratio)
    split_plan = _plan_split_from_chunks(train_chunk_specs, label_end=label_end, unlabel_end=unlabel_end)
    _materialize_chunked_split_views(
        split_dir=split_dir,
        plan=split_plan,
        full_chunk_dir_for_range=full_chunk_dir_for_range,
        partial_dir_for_range=partial_dir_for_range,
    )

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
    if _manifest_should_write(manifest, manifest_updates):
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
    num_workers: int | None = None,
    chunk_bytes: str = "200MB",
    unlabel_ratio: float | None = None,
    prepare_mode: str = "chunked",
    train_is_shuffled: bool = True,
    val_strategy: str = "synthetic",
    val_ratio: float = DEFAULT_AUTO_VAL_RATIO,
    scan_batch_size: int = 8192,
    target_chunk_bytes: str = DEFAULT_TARGET_CHUNK_BYTES,
    min_chunk_count: int = DEFAULT_MIN_CHUNK_COUNT,
    max_chunk_count: int = DEFAULT_MAX_CHUNK_COUNT,
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
    if prepare_mode not in {"in_memory", "chunked"}:
        raise ValueError(f"prepare_mode must be 'in_memory' or 'chunked', got {prepare_mode!r}")
    if val_strategy not in {"explicit", "synthetic", "smallest_train_shard"}:
        raise ValueError(
            f"val_strategy must be 'explicit', 'synthetic', or 'smallest_train_shard', got {val_strategy!r}"
        )
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1 (exclusive), got {val_ratio}")

    if num_workers is None:
        num_workers = len(os.sched_getaffinity(0)) or 1

    if prepare_mode == "in_memory":
        return _prepare_dataset_in_memory(
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
            val_ratio=val_ratio,
            target_chunk_bytes=target_chunk_bytes,
            min_chunk_count=min_chunk_count,
            max_chunk_count=max_chunk_count,
            train_texts=train_texts,
            val_texts=val_texts,
        )

    return _prepare_dataset_chunked(
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
        val_ratio=val_ratio,
        scan_batch_size=scan_batch_size,
        target_chunk_bytes=target_chunk_bytes,
        min_chunk_count=min_chunk_count,
        max_chunk_count=max_chunk_count,
    )
