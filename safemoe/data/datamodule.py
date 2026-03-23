# safemoe/data/datamodule.py
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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
        if effective < self.num_workers:
            warnings.warn(
                (
                    f"Reducing SafeDataModule num_workers from "
                    f"{self.num_workers} to {effective} per loader to avoid "
                    f"oversubscribing {loader_count} concurrent streaming loaders "
                    f"across WORLD_SIZE={world_size} on a host with {cpu_count} CPUs. "
                    "SafeMoE caps streaming workers at 8 per loader."
                ),
                stacklevel=2,
            )
        return effective

    def connect(self, tokenizer=None, batch_size=1, max_seq_length=-1):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length + 1  # +1 for next-token target

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
        from litdata.streaming import (
            CombinedStreamingDataset,
            StreamingDataLoader,
            StreamingDataset,
            TokensLoader,
        )

        tokenizer_name = self.tokenizer.model_name if self.tokenizer else "default"

        std_datasets, harmful_datasets, unlabeled_datasets = [], [], []
        for ds_id, cfg in self._configs.items():
            base = self.cache_dir / tokenizer_name / ds_id
            ratio_pct = int(round(cfg.get("label_ratio", 1.0) * 100))
            split_dir = base / "splits" / f"r{ratio_pct}"

            role = cfg.get("role", "std")
            if ratio_pct > 0:
                labeled_ds = StreamingDataset(
                    input_dir=str(split_dir / "labeled_train"),
                    item_loader=TokensLoader(block_size=self.max_seq_length),
                    shuffle=True,
                )
                if role == "std":
                    std_datasets.append(labeled_ds)
                else:
                    harmful_datasets.append(labeled_ds)

            if ratio_pct < 100:
                unlabeled_ds = StreamingDataset(
                    input_dir=str(split_dir / "unlabeled_train"),
                    item_loader=TokensLoader(block_size=self.max_seq_length),
                    shuffle=True,
                )
                unlabeled_datasets.append(unlabeled_ds)

        loader_count = sum(
            1 for ds_list in [std_datasets, harmful_datasets, unlabeled_datasets] if ds_list
        )
        num_workers = self._effective_num_workers(loader_count=loader_count)

        def make_loader(ds_list):
            if len(ds_list) == 1:
                ds = ds_list[0]
            else:
                ds = CombinedStreamingDataset(ds_list, seed=self.seed)
            return StreamingDataLoader(
                ds,
                batch_size=self.batch_size,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
            )

        self._loaders = {}
        if std_datasets:
            self._loaders["D_std"] = make_loader(std_datasets)
        if harmful_datasets:
            self._loaders["D_harmful"] = make_loader(harmful_datasets)
        if unlabeled_datasets:
            self._loaders["D_unlabeled"] = make_loader(unlabeled_datasets)

    def get_loader(self, split_name: str):
        if self._loaders is None:
            raise RuntimeError("Call setup() before get_loader()")
        return self._loaders[split_name]

    def val_datasets(self) -> dict:
        """Returns {"D_std": dataset, "D_harmful": dataset} for eval-only batching."""
        from litdata.streaming import CombinedStreamingDataset, StreamingDataset, TokensLoader

        tokenizer_name = self.tokenizer.model_name if self.tokenizer else "default"

        result = {}
        for role_name, role_key in [("D_std", "std"), ("D_harmful", "harmful")]:
            ds_list = []
            for ds_id, cfg in self._configs.items():
                if cfg.get("role", "std") == role_key:
                    val_dir = self.cache_dir / tokenizer_name / ds_id / "val"
                    if val_dir.exists():
                        ds_list.append(StreamingDataset(
                            input_dir=str(val_dir),
                            item_loader=TokensLoader(block_size=self.max_seq_length),
                            shuffle=False,
                        ))
            if len(ds_list) == 1:
                result[role_name] = ds_list[0]
            elif ds_list:
                result[role_name] = CombinedStreamingDataset(ds_list, seed=self.seed)
        return result

    def val_dataloaders(self) -> dict:
        """Returns {D_std: DataLoader, D_harmful: DataLoader}. No D_unlabeled val set."""
        from litdata.streaming import StreamingDataLoader

        num_workers = self._effective_num_workers(loader_count=2)

        return {
            split_name: StreamingDataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
            )
            for split_name, dataset in self.val_datasets().items()
        }

    def train_dataloader(self):
        return self.get_loader("D_std")

    def val_dataloader(self):
        return list(self.val_dataloaders().values())
