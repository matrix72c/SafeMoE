# safemoe/data/datamodule.py
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from litgpt.data import DataModule
from litgpt.tokenizer import Tokenizer


@dataclass
class SafeDataModule(DataModule):
    """Multi-dataset DataModule with per-dataset role and label_ratio configuration.

    Each dataset is configured with:
      - path: directory containing train/val parquet or jsonl files
      - text_column: column name for text data
      - role: "std" or "harmful" — determines which training split the labeled portion joins
      - label_ratio: fraction of training data used as labeled (rest goes to D_unlabeled)

    Access pattern:
        loader = data.get_loader('D_std')
        val_loaders = data.val_dataloaders()  # {"D_std": ..., "D_harmful": ...}
    """

    cache_dir: Path = Path("data/.cache")
    num_workers: int = 4
    seed: int = 42
    chunk_bytes: str = "200MB"
    datasets: dict = field(default_factory=dict)

    # Set by connect()
    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
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
        return self.tokenizer.model_name if self.tokenizer else "default"

    def _split_dir(self, dataset_id: str, ratio_pct: int) -> Path:
        return self.cache_dir / self._tokenizer_cache_name() / dataset_id / "splits" / f"r{ratio_pct}"

    def _val_dir(self, dataset_id: str) -> Path:
        return self.cache_dir / self._tokenizer_cache_name() / dataset_id / "val"

    def _streaming_dataset(self, input_dir: Path, *, shuffle: bool):
        from litdata.streaming import StreamingDataset, TokensLoader

        return StreamingDataset(
            input_dir=str(input_dir),
            item_loader=TokensLoader(block_size=self.max_seq_length),
            shuffle=shuffle,
        )

    def _combine_datasets(self, datasets: list):
        from litdata.streaming import CombinedStreamingDataset

        if len(datasets) == 1:
            return datasets[0]
        return CombinedStreamingDataset(datasets, seed=self.seed)

    def _streaming_dataloader(self, dataset, *, num_workers: int, drop_last: bool):
        from litdata.streaming import StreamingDataLoader

        return StreamingDataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last,
        )

    def _build_train_dataset_groups(self) -> dict[str, list]:
        grouped_datasets = {"D_std": [], "D_harmful": [], "D_unlabeled": []}
        for ds_id, cfg in self._configs.items():
            ratio_pct = int(round(cfg.get("label_ratio", 1.0) * 100))
            split_dir = self._split_dir(ds_id, ratio_pct)
            role = cfg.get("role", "std")
            train_split_name = "D_std" if role == "std" else "D_harmful"

            if ratio_pct > 0:
                grouped_datasets[train_split_name].append(
                    self._streaming_dataset(split_dir / "labeled_train", shuffle=True)
                )

            if ratio_pct < 100:
                grouped_datasets["D_unlabeled"].append(
                    self._streaming_dataset(split_dir / "unlabeled_train", shuffle=True)
                )
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

    def _build_val_loader_iterables(self, *, batch_size: Optional[int] = None, drop_last: bool = False) -> dict[str, object]:
        effective_batch_size = self.batch_size if batch_size is None else batch_size

        class _DatasetBatchIterable:
            def __init__(self, dataset: object, batch_size: int, drop_last: bool = False) -> None:
                self.dataset = dataset
                self.batch_size = batch_size
                self.drop_last = drop_last

            def __iter__(self):
                dataset_length = len(self.dataset)
                step = self.batch_size
                limit = dataset_length if not self.drop_last else dataset_length - (dataset_length % step)
                for start in range(0, limit, step):
                    stop = min(start + step, dataset_length)
                    if self.drop_last and stop - start < step:
                        break
                    yield torch.stack([self.dataset[index].clone() for index in range(start, stop)])

            def __len__(self) -> int:
                dataset_length = len(self.dataset)
                if self.drop_last:
                    return dataset_length // self.batch_size
                return (dataset_length + self.batch_size - 1) // self.batch_size

        return {
            split_name: _DatasetBatchIterable(dataset, batch_size=effective_batch_size, drop_last=drop_last)
            for split_name, dataset in self.val_datasets().items()
        }

    def initialize_loaders(self) -> dict[str, object]:
        self.setup()
        val_loaders = self.val_dataloaders()
        primary_val_loader = val_loaders.get("D_std")
        if primary_val_loader is None and val_loaders:
            primary_val_loader = next(iter(val_loaders.values()))
        return {
            "train_loaders": dict(self._loaders or {}),
            "val_loaders": val_loaders,
            "primary_val_loader": primary_val_loader,
        }

    def prepare_data(self):
        """Tokenize and split each dataset. Idempotent."""
        from safemoe.data.prepare import prepare_dataset

        for ds_id, cfg in self._configs.items():
            prepare_dataset(
                dataset_id=ds_id,
                data_path=Path(cfg["path"]),
                text_column=cfg.get("text_column", "text"),
                label_ratio=cfg.get("label_ratio", 1.0),
                tokenizer=self.tokenizer,
                cache_dir=self.cache_dir,
                seed=self.seed,
                num_workers=self.num_workers,
                chunk_bytes=self.chunk_bytes,
            )

    def setup(self, stage: str = "") -> None:
        train_datasets = self._build_train_dataset_groups()
        loader_count = sum(1 for datasets in train_datasets.values() if datasets)
        num_workers = self._effective_num_workers(loader_count=loader_count)

        self._loaders = {
            split_name: self._streaming_dataloader(
                self._combine_datasets(datasets),
                num_workers=num_workers,
                drop_last=True,
            )
            for split_name, datasets in train_datasets.items()
            if datasets
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
