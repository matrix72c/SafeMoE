# safemoe/data/datamodule.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

from litgpt.data import DataModule
from litgpt.tokenizer import Tokenizer

from safemoe.data.prepare import (
    DEFAULT_AUTO_VAL_RATIO,
    DEFAULT_MAX_CHUNK_COUNT,
    DEFAULT_MIN_CHUNK_COUNT,
    DEFAULT_TARGET_CHUNK_BYTES,
)


@dataclass
class SafeDataModule(DataModule):
    """Multi-dataset DataModule with per-dataset role, split ratios, and split-local weights.

    Each dataset is configured with:
      - path: directory containing train/val parquet or jsonl files
      - text_column: column name for text data
      - role: "std" or "harmful" — determines which training split the labeled portion joins
      - label_ratio: fraction of training data used as labeled
      - unlabel_ratio: fraction of training data used as unlabeled
      - label_weight: mixing weight inside the labeled destination split
      - unlabel_weight: mixing weight inside D_unlabeled

    Access pattern:
        loader = data.get_loader('D_std')
        val_loaders = data.val_dataloaders()  # {"D_std": ..., "D_harmful": ...}
    """

    cache_dir: Path = Path("data/.cache")
    num_workers: int = 4
    seed: int = 42
    chunk_bytes: str = "200MB"
    target_chunk_bytes: str = DEFAULT_TARGET_CHUNK_BYTES
    min_chunk_count: int = DEFAULT_MIN_CHUNK_COUNT
    max_chunk_count: int = DEFAULT_MAX_CHUNK_COUNT
    datasets: dict = field(default_factory=dict)

    # Set by connect()
    tokenizer: Tokenizer | None = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    _loaders: dict = field(default=None, init=False, repr=False)
    _val_datasets: dict = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._configs = {}
        for name, cfg in self.datasets.items():
            if isinstance(cfg, dict):
                self._configs[name] = cfg

    def _effective_num_workers(self, *, loader_count: int) -> int:
        if self.num_workers <= 0:
            return 0

        cpu_count = len(os.sched_getaffinity(0)) or 1
        world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
        concurrent_loaders = max(loader_count * world_size, 1)
        max_per_loader = max(1, cpu_count // concurrent_loaders)
        effective = min(self.num_workers, max_per_loader, 8)
        return effective

    def connect(self, tokenizer=None, batch_size=1, max_seq_length=-1):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length + 1  # +1 for next-token target

    def _tokenizer_cache_name(self) -> str:
        if self.tokenizer and hasattr(self.tokenizer, "model_name"):
            return self.tokenizer.model_name
        return "default"

    def _split_dir(self, dataset_id: str, label_ratio: float, unlabel_ratio: float, prepare_mode: str) -> Path:
        from safemoe.data.prepare import _in_memory_ratio_dir, _split_dir

        base_dir = self.cache_dir / self._tokenizer_cache_name() / dataset_id
        if prepare_mode == "chunked":
            return _split_dir(base_dir, label_ratio, unlabel_ratio)
        return _in_memory_ratio_dir(base_dir, label_ratio, unlabel_ratio)

    def _val_dir(self, dataset_id: str) -> Path:
        return self.cache_dir / self._tokenizer_cache_name() / dataset_id / "val"

    def _streaming_dataset(self, input_dir: Path, *, shuffle: bool):
        from litdata.streaming import StreamingDataset, TokensLoader

        return StreamingDataset(
            input_dir=str(input_dir),
            item_loader=TokensLoader(block_size=self.max_seq_length),
            shuffle=shuffle,
        )

    def _combine_datasets(self, datasets: list, weights: list[float] | None = None):
        from litdata.streaming import CombinedStreamingDataset

        if len(datasets) == 1:
            return datasets[0]
        return CombinedStreamingDataset(datasets=datasets, seed=self.seed, weights=weights, iterate_over_all=False)

    def _streaming_dataloader(self, dataset, *, num_workers: int, drop_last: bool):
        from litdata.streaming import StreamingDataLoader

        return StreamingDataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last,
        )

    def _normalized_dataset_config(self, dataset_id: str, cfg: dict) -> dict:
        label_ratio = cfg.get("label_ratio", 1.0)
        unlabel_ratio = cfg.get("unlabel_ratio", 1.0 - label_ratio)
        label_weight = cfg.get("label_weight", 1.0)
        unlabel_weight = cfg.get("unlabel_weight", 1.0)
        role = cfg.get("role", "std")
        prepare_mode = cfg.get("prepare_mode", "chunked")
        train_is_shuffled = cfg.get("train_is_shuffled", True)
        val_strategy = cfg.get("val_strategy", "synthetic")
        val_ratio = cfg.get("val_ratio", DEFAULT_AUTO_VAL_RATIO)
        scan_batch_size = cfg.get("scan_batch_size", 8192)
        target_chunk_bytes = cfg.get("target_chunk_bytes", self.target_chunk_bytes)
        min_chunk_count = cfg.get("min_chunk_count", self.min_chunk_count)
        max_chunk_count = cfg.get("max_chunk_count", self.max_chunk_count)

        if role not in {"std", "harmful"}:
            raise ValueError(f"Dataset '{dataset_id}' has invalid role '{role}'")
        if not 0.0 <= label_ratio <= 1.0:
            raise ValueError(f"Dataset '{dataset_id}' label_ratio must be between 0 and 1, got {label_ratio}")
        if not 0.0 <= unlabel_ratio <= 1.0:
            raise ValueError(f"Dataset '{dataset_id}' unlabel_ratio must be between 0 and 1, got {unlabel_ratio}")
        if label_ratio + unlabel_ratio > 1.0:
            raise ValueError(
                f"Dataset '{dataset_id}' label_ratio + unlabel_ratio must be <= 1, got {label_ratio + unlabel_ratio}"
            )
        if label_ratio > 0 and label_weight <= 0:
            raise ValueError(
                f"Dataset '{dataset_id}' label_weight must be > 0 when label_ratio > 0, got {label_weight}"
            )
        if label_ratio == 0 and label_weight < 0:
            raise ValueError(
                f"Dataset '{dataset_id}' label_weight must be >= 0 when label_ratio == 0, got {label_weight}"
            )
        if unlabel_ratio > 0 and unlabel_weight <= 0:
            raise ValueError(
                f"Dataset '{dataset_id}' unlabel_weight must be > 0 when unlabel_ratio > 0, got {unlabel_weight}"
            )
        if unlabel_ratio == 0 and unlabel_weight < 0:
            raise ValueError(
                f"Dataset '{dataset_id}' unlabel_weight must be >= 0 when unlabel_ratio == 0, got {unlabel_weight}"
            )
        if prepare_mode not in {"in_memory", "chunked"}:
            raise ValueError(
                f"Dataset '{dataset_id}' prepare_mode must be 'in_memory' or 'chunked', got {prepare_mode!r}"
            )
        if not isinstance(train_is_shuffled, bool):
            raise ValueError(f"Dataset '{dataset_id}' train_is_shuffled must be a bool, got {train_is_shuffled!r}")
        if val_strategy not in {"explicit", "smallest_train_shard", "synthetic"}:
            raise ValueError(
                f"Dataset '{dataset_id}' val_strategy must be 'explicit', 'smallest_train_shard', or 'synthetic', got {val_strategy!r}"
            )
        if val_strategy == "synthetic" and (not isinstance(val_ratio, (float, int)) or not 0.0 < float(val_ratio) < 1.0):
            raise ValueError(
                f"Dataset '{dataset_id}' val_ratio must be a float in (0, 1) when val_strategy='synthetic', got {val_ratio!r}"
            )
        if not isinstance(scan_batch_size, int) or scan_batch_size <= 0:
            raise ValueError(f"Dataset '{dataset_id}' scan_batch_size must be a positive int, got {scan_batch_size!r}")
        if not isinstance(min_chunk_count, int) or min_chunk_count <= 0:
            raise ValueError(f"Dataset '{dataset_id}' min_chunk_count must be a positive int, got {min_chunk_count!r}")
        if not isinstance(max_chunk_count, int) or max_chunk_count <= 0:
            raise ValueError(f"Dataset '{dataset_id}' max_chunk_count must be a positive int, got {max_chunk_count!r}")
        if min_chunk_count > max_chunk_count:
            raise ValueError(
                f"Dataset '{dataset_id}' min_chunk_count must be <= max_chunk_count, got {min_chunk_count} > {max_chunk_count}"
            )

        return {
            **cfg,
            "role": role,
            "label_ratio": label_ratio,
            "unlabel_ratio": unlabel_ratio,
            "label_weight": label_weight,
            "unlabel_weight": unlabel_weight,
            "prepare_mode": prepare_mode,
            "train_is_shuffled": train_is_shuffled,
            "val_strategy": val_strategy,
            "val_ratio": float(val_ratio),
            "scan_batch_size": scan_batch_size,
            "target_chunk_bytes": target_chunk_bytes,
            "min_chunk_count": min_chunk_count,
            "max_chunk_count": max_chunk_count,
        }

    def _build_train_dataset_groups(self) -> dict[str, dict[str, list]]:
        grouped_datasets = {
            "D_std": {"datasets": [], "weights": []},
            "D_harmful": {"datasets": [], "weights": []},
            "D_unlabeled": {"datasets": [], "weights": []},
        }
        for ds_id, raw_cfg in self._configs.items():
            cfg = self._normalized_dataset_config(ds_id, raw_cfg)
            split_dir = self._split_dir(ds_id, cfg["label_ratio"], cfg["unlabel_ratio"], cfg["prepare_mode"])
            train_split_name = "D_std" if cfg["role"] == "std" else "D_harmful"

            if cfg["label_ratio"] > 0:
                grouped_datasets[train_split_name]["datasets"].append(
                    self._streaming_dataset(split_dir / "labeled_train", shuffle=True)
                )
                grouped_datasets[train_split_name]["weights"].append(cfg["label_weight"])

            if cfg["unlabel_ratio"] > 0:
                grouped_datasets["D_unlabeled"]["datasets"].append(
                    self._streaming_dataset(split_dir / "unlabeled_train", shuffle=True)
                )
                grouped_datasets["D_unlabeled"]["weights"].append(cfg["unlabel_weight"])
        return grouped_datasets

    def _build_val_datasets(self) -> dict[str, object]:
        val_datasets: dict[str, object] = {}
        for split_name, role_key in (("D_std", "std"), ("D_harmful", "harmful")):
            datasets = []
            for ds_id, cfg in self._configs.items():
                if cfg.get("role", "std") != role_key:
                    continue
                val_dir = self._val_dir(ds_id)
                if val_dir.exists():
                    datasets.append(self._streaming_dataset(val_dir, shuffle=False))
            if datasets:
                val_datasets[split_name] = self._combine_datasets(datasets)
        return val_datasets

    def _build_val_loader_iterables(self, *, batch_size: int | None = None, drop_last: bool = False) -> dict[str, object]:
        effective_batch_size = self.batch_size if batch_size is None else batch_size

        class _DatasetBatchIterable:
            def __init__(self, dataset: object, batch_size: int, drop_last: bool = False) -> None:
                self.dataset = dataset
                self.batch_size = batch_size
                self.drop_last = drop_last

            def __iter__(self):
                batch = []
                for sample in self.dataset:
                    batch.append(sample.clone())
                    if len(batch) == self.batch_size:
                        yield torch.stack(batch)
                        batch = []
                if batch and not self.drop_last:
                    yield torch.stack(batch)

            def __len__(self) -> int:
                try:
                    dataset_length = len(self.dataset)
                except TypeError as ex:
                    raise TypeError("Validation iterable length is unavailable for this dataset") from ex
                if dataset_length is None:
                    raise TypeError("Validation iterable length is unavailable for this dataset")
                if self.drop_last:
                    return dataset_length // self.batch_size
                return (dataset_length + self.batch_size - 1) // self.batch_size

        return {
            split_name: _DatasetBatchIterable(dataset, batch_size=effective_batch_size, drop_last=drop_last)
            for split_name, dataset in self.val_datasets().items()
        }

    def initialize_loaders(self) -> dict[str, object]:
        self.setup()
        return {
            "train_loaders": dict(self._loaders or {}),
            "val_loaders": self.val_dataloaders(),
        }

    def prepare_data(self):
        """Tokenize and split each dataset. Idempotent."""
        from safemoe.data.prepare import prepare_dataset

        for ds_id, raw_cfg in self._configs.items():
            cfg = self._normalized_dataset_config(ds_id, raw_cfg)
            prepare_dataset(
                dataset_id=ds_id,
                data_path=Path(cfg["path"]),
                text_column=cfg.get("text_column", "text"),
                label_ratio=cfg["label_ratio"],
                unlabel_ratio=cfg["unlabel_ratio"],
                tokenizer=self.tokenizer,
                cache_dir=self.cache_dir,
                seed=self.seed,
                chunk_bytes=self.chunk_bytes,
                prepare_mode=cfg["prepare_mode"],
                train_is_shuffled=cfg["train_is_shuffled"],
                val_strategy=cfg["val_strategy"],
                val_ratio=cfg["val_ratio"],
                scan_batch_size=cfg["scan_batch_size"],
                target_chunk_bytes=cfg["target_chunk_bytes"],
                min_chunk_count=cfg["min_chunk_count"],
                max_chunk_count=cfg["max_chunk_count"],
            )

    def setup(self, stage: str = "") -> None:
        train_datasets = self._build_train_dataset_groups()
        loader_count = sum(1 for group in train_datasets.values() if group["datasets"])
        num_workers = self._effective_num_workers(loader_count=loader_count)

        self._loaders = {
            split_name: self._streaming_dataloader(
                self._combine_datasets(group["datasets"], group["weights"]),
                num_workers=num_workers,
                drop_last=True,
            )
            for split_name, group in train_datasets.items()
            if group["datasets"]
        }
        self._val_datasets = None

    def get_loader(self, split_name: str):
        if self._loaders is None:
            raise RuntimeError("Call setup() before get_loader()")
        return self._loaders[split_name]

    def val_datasets(self) -> dict:
        """Returns {"D_std": dataset, "D_harmful": dataset} for eval-only batching."""
        if self._val_datasets is None:
            self._val_datasets = self._build_val_datasets()
        return dict(self._val_datasets)

    def val_dataloaders(self) -> dict:
        """Returns {D_std: DataLoader, D_harmful: DataLoader}. No D_unlabeled val set."""
        datasets = self.val_datasets()
        num_workers = self._effective_num_workers(loader_count=max(len(datasets), 1))
        return {
            split_name: self._streaming_dataloader(
                dataset,
                num_workers=num_workers,
                drop_last=False,
            )
            for split_name, dataset in datasets.items()
        }

    def train_dataloader(self):
        return self.get_loader("D_std")

    def val_dataloader(self):
        return list(self.val_dataloaders().values())
