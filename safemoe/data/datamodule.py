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
class MultiDataLoader(DataModule):
    """Per-split DataLoader registry for SGTM three-stream training.

    Access pattern (training loop):
        loader = multi_loader.get_loader('D_std')
        it = iter(loader)
        batch = next(it)

    Val access:
        val_loaders = multi_loader.val_dataloaders()
        # {"D_std": DataLoader, "D_harmful": DataLoader}
    """

    cache_dir: Path = Path("data/.cache")
    x: int = 0        # % of ES rows going to D_unlabeled (int, matches cache dir name)
    y: int = 25       # % of EN rows going to D_std (int)
    seed: int = 42
    num_workers: int = 4

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    _loaders: dict = field(default=None, init=False, repr=False)

    def _effective_num_workers(self, *, loader_count: int) -> int:
        """Cap workers per loader to avoid oversubscribing streaming worker processes.

        SafeMoE drives multiple loaders concurrently during training, so using the raw
        configured value per loader can create an excessive total worker count and
        destabilize `StreamingDataLoader`.
        """
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
                    "Reducing MultiDataLoader num_workers from "
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

    def _base_dir(self) -> Path:
        tokenizer_name = self.tokenizer.model_name if self.tokenizer else "default"
        return self.cache_dir / tokenizer_name / f"{int(self.x)}-{int(self.y)}"

    def setup(self, stage: str = "") -> None:
        from litdata.streaming import StreamingDataLoader, StreamingDataset, TokensLoader

        base = self._base_dir()
        num_workers = self._effective_num_workers(loader_count=3)

        def make_loader(split_name: str) -> StreamingDataLoader:
            ds = StreamingDataset(
                input_dir=str(base / split_name / "train"),
                item_loader=TokensLoader(block_size=self.max_seq_length),
                shuffle=True,
            )
            return StreamingDataLoader(
                ds,
                batch_size=self.batch_size,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
            )

        self._loaders = {
            "D_std":       make_loader("D_std"),
            "D_harmful":   make_loader("D_harmful"),
            "D_unlabeled": make_loader("D_unlabeled"),
        }

    def get_loader(self, split_name: str):
        """Return the DataLoader for the given split. Training loop manages its own iterator."""
        if self._loaders is None:
            raise RuntimeError("Call setup() before get_loader()")
        return self._loaders[split_name]  # KeyError propagates for unknown split names

    def val_datasets(self) -> dict:
        """Returns {D_std: StreamingDataset, D_harmful: StreamingDataset} for eval-only batching."""
        from litdata.streaming import StreamingDataset, TokensLoader

        base = self._base_dir()

        def make_val_dataset(split_name: str) -> StreamingDataset:
            return StreamingDataset(
                input_dir=str(base / split_name / "val"),
                item_loader=TokensLoader(block_size=self.max_seq_length),
                shuffle=False,
            )

        return {
            "D_std": make_val_dataset("D_std"),
            "D_harmful": make_val_dataset("D_harmful"),
        }

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
        """LightningDataModule compat — returns D_std loader. Use get_loader() in SGTM loop."""
        return self.get_loader("D_std")

    def val_dataloader(self):
        """LightningDataModule compat — returns list of val loaders."""
        return list(self.val_dataloaders().values())
