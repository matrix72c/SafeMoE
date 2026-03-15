"""Data preparation pipeline for SafeMoE: tokenization and split partitioning.

Implements the two-parameter split formula (x, y):
  - D_std      = first y%  of EN rows
  - D_harmful  = first (100-x)% of ES rows
  - D_unlabeled = remaining (100-y)% EN + remaining x% ES rows

CLI usage:
  python -m safemoe.data.prepare --x 0 --y 25 --num_workers 4

Cache layout (integer-keyed):
  data/.cache/{tokenizer_name}/{x}-{y}/D_std/train/
  data/.cache/{tokenizer_name}/{x}-{y}/D_std/val/
  data/.cache/{tokenizer_name}/{x}-{y}/D_harmful/train/
  data/.cache/{tokenizer_name}/{x}-{y}/D_harmful/val/
  data/.cache/{tokenizer_name}/{x}-{y}/D_unlabeled/train/
  (No D_unlabeled/val)
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import Optional

import torch
from litdata import optimize, TokensLoader


# ---------------------------------------------------------------------------
# Public API: compute_splits
# ---------------------------------------------------------------------------


def compute_splits(
    en_stories: list[str],
    es_stories: list[str],
    x: int = 0,
    y: int = 25,
) -> dict[str, list[str]]:
    """Partition EN and ES stories into three named splits.

    Args:
        en_stories: List of English story strings.
        es_stories: List of Spanish story strings.
        x: Percentage of ES rows that goes to D_unlabeled (ES leak parameter).
           Range [0, 100]. Default 0 — all ES rows go to D_harmful.
        y: Percentage of EN rows that goes to D_std (EN retention parameter).
           Range [0, 100]. Default 25.

    Returns:
        dict with keys "D_std", "D_harmful", "D_unlabeled".
    """
    n_en = len(en_stories)
    n_es = len(es_stories)

    std_end = int(y / 100.0 * n_en)
    harmful_end = int((100 - x) / 100.0 * n_es)

    d_std = en_stories[:std_end]
    d_harmful = es_stories[:harmful_end]
    d_unlabeled = en_stories[std_end:] + es_stories[harmful_end:]

    return {
        "D_std": d_std,
        "D_harmful": d_harmful,
        "D_unlabeled": d_unlabeled,
    }


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def tokenize_stories(stories: list[str], tokenizer) -> None:
    """Generator that yields token tensors for each story.

    Avoids reading DATA_OPTIMIZER_* environment variables.
    Works as the ``fn`` argument to ``litdata.optimize()``.

    Args:
        stories: List of story strings assigned to this worker.
        tokenizer: Object with ``.encode(text, bos, eos) -> torch.Tensor``.

    Yields:
        torch.Tensor of token ids (dtype=torch.long) for each story.
    """
    for text in stories:
        text = text.strip()
        if not text:
            continue
        tokens = tokenizer.encode(text, bos=True, eos=False)
        yield tokens


# ---------------------------------------------------------------------------
# Internal: idempotent optimize wrapper
# ---------------------------------------------------------------------------


def _maybe_optimize(
    stories: list[str],
    output_dir: Path,
    tokenizer,
    num_workers: int,
    chunk_bytes: str,
    start_method: str = "fork",
) -> None:
    """Call litdata.optimize() unless output_dir already exists.

    Idempotency: if the directory already contains data (i.e., it exists),
    skip tokenization entirely.

    Args:
        stories: All story strings for this split.
        output_dir: Target directory for LitData output.
        tokenizer: Tokenizer with ``.encode()`` method.
        num_workers: Number of parallel workers for optimize().
        chunk_bytes: Chunk size string (e.g. "200MB").
        start_method: Multiprocessing start method. Default "fork" — avoids
            pickle errors for in-memory tokenizer objects. Use "spawn" for
            cross-platform compatibility.
    """
    if output_dir.exists():
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    if not stories:
        # Empty split — create directory but skip optimize
        return

    # Distribute stories across workers (each worker gets a non-overlapping slice)
    effective_workers = min(num_workers, len(stories))
    if effective_workers < 1:
        effective_workers = 1
    inputs = [stories[i::effective_workers] for i in range(effective_workers)]
    # Remove empty slices that arise when len(stories) < num_workers
    inputs = [s for s in inputs if s]

    optimize(
        fn=partial(tokenize_stories, tokenizer=tokenizer),
        inputs=inputs,
        output_dir=str(output_dir),
        num_workers=len(inputs),
        chunk_bytes=chunk_bytes,
        item_loader=TokensLoader(),
        start_method=start_method,
    )


# ---------------------------------------------------------------------------
# Public API: prepare
# ---------------------------------------------------------------------------


def prepare(
    checkpoint_dir: Optional[Path] = None,
    data_dir: Path = Path("data/multilingual-tinystories"),
    cache_dir: Path = Path("data/.cache"),
    x: int = 0,
    y: int = 25,
    num_workers: int = 4,
    chunk_bytes: str = "200MB",
    # ---- test injection hooks (optional) -----------------------------------
    tokenizer=None,
    en_train: Optional[list[str]] = None,
    es_train: Optional[list[str]] = None,
    en_val: Optional[list[str]] = None,
    es_val: Optional[list[str]] = None,
) -> None:
    """Tokenize and partition the TinyStories bilingual corpus.

    Reads EN and ES parquet files, applies the two-parameter split formula,
    and writes LitData streaming chunks to::

        cache_dir/{tokenizer_name}/{x}-{y}/D_std/train/
        cache_dir/{tokenizer_name}/{x}-{y}/D_std/val/
        cache_dir/{tokenizer_name}/{x}-{y}/D_harmful/train/
        cache_dir/{tokenizer_name}/{x}-{y}/D_harmful/val/
        cache_dir/{tokenizer_name}/{x}-{y}/D_unlabeled/train/

    (D_unlabeled has no val set.)

    The function is idempotent: if an output directory already exists, it is
    skipped without re-tokenizing.

    Args:
        checkpoint_dir: Path to tokenizer checkpoint directory.
            Defaults to ``checkpoints/Qwen3-30B-A3B-Base``.
        data_dir: Directory containing ``{en,es}/{train,validation}.parquet``.
        cache_dir: Root cache directory for LitData output.
        x: ES leak percentage (0 = no ES in D_unlabeled). Default 0.
        y: EN retention percentage for D_std. Default 25.
        num_workers: Number of litdata.optimize() workers.
        chunk_bytes: Chunk size for LitData binary chunks.
        tokenizer: (Test injection) Pre-built tokenizer. When provided,
            ``checkpoint_dir`` is ignored.
        en_train: (Test injection) Overrides parquet EN train rows.
        es_train: (Test injection) Overrides parquet ES train rows.
        en_val: (Test injection) Overrides parquet EN val rows.
        es_val: (Test injection) Overrides parquet ES val rows.
    """
    # Ensure x and y are plain ints (guard against float CLI input)
    x = int(x)
    y = int(y)

    # --- Tokenizer -----------------------------------------------------------
    if tokenizer is None:
        if checkpoint_dir is None:
            checkpoint_dir = Path("checkpoints/Qwen3-30B-A3B-Base")
        checkpoint_dir = Path(checkpoint_dir)
        from litgpt.tokenizer import Tokenizer as LitGPTTokenizer
        tokenizer = LitGPTTokenizer(checkpoint_dir)

    # --- Load stories (or use injected data for tests) -----------------------
    if en_train is None or es_train is None or en_val is None or es_val is None:
        import pandas as pd
        data_dir = Path(data_dir)
        en_train = pd.read_parquet(data_dir / "en" / "train.parquet")["text"].tolist()
        es_train = pd.read_parquet(data_dir / "es" / "train.parquet")["text"].tolist()
        en_val = pd.read_parquet(data_dir / "en" / "validation.parquet")["text"].tolist()
        es_val = pd.read_parquet(data_dir / "es" / "validation.parquet")["text"].tolist()

    # --- Split formula -------------------------------------------------------
    splits = compute_splits(en_train, es_train, x=x, y=y)

    # --- Output base directory (integer-keyed) --------------------------------
    out_base = Path(cache_dir) / tokenizer.model_name / f"{x}-{y}"

    # --- Tokenize each split -------------------------------------------------
    _maybe_optimize(splits["D_std"], out_base / "D_std" / "train",
                    tokenizer, num_workers, chunk_bytes)
    _maybe_optimize(en_val, out_base / "D_std" / "val",
                    tokenizer, num_workers, chunk_bytes)
    _maybe_optimize(splits["D_harmful"], out_base / "D_harmful" / "train",
                    tokenizer, num_workers, chunk_bytes)
    _maybe_optimize(es_val, out_base / "D_harmful" / "val",
                    tokenizer, num_workers, chunk_bytes)
    _maybe_optimize(splits["D_unlabeled"], out_base / "D_unlabeled" / "train",
                    tokenizer, num_workers, chunk_bytes)
    # No D_unlabeled/val


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize and partition TinyStories bilingual dataset."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("checkpoints/Qwen3-30B-A3B-Base"),
        help="Path to tokenizer checkpoint directory.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/multilingual-tinystories"),
        help="Directory containing {en,es}/{train,validation}.parquet files.",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=Path("data/.cache"),
        help="Root cache directory for LitData output.",
    )
    parser.add_argument(
        "--x",
        type=int,
        default=0,
        help="ES leak percentage (0 = all ES goes to D_harmful). Default: 0.",
    )
    parser.add_argument(
        "--y",
        type=int,
        default=25,
        help="EN retention percentage for D_std. Default: 25.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of litdata.optimize() workers. Default: 4.",
    )
    parser.add_argument(
        "--chunk_bytes",
        type=str,
        default="200MB",
        help="Chunk size for LitData binary chunks. Default: 200MB.",
    )
    args = parser.parse_args()

    prepare(
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        x=args.x,
        y=args.y,
        num_workers=args.num_workers,
        chunk_bytes=args.chunk_bytes,
    )
