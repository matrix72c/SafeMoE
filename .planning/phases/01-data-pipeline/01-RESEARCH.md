# Phase 1: Data Pipeline - Research

**Researched:** 2026-03-14
**Domain:** LitData streaming tokenization, bilingual TinyStories partitioning, dynamic weighted multi-split DataLoader
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Package structure**
- New `safemoe/` top-level package alongside `litgpt/` — all SafeMoE-specific code lives there
- Data code at `safemoe/data/` subdirectory (mirrors `litgpt/data/` layout)
- `safemoe/` imports `litgpt` internals directly (e.g., `from litgpt.tokenizer import Tokenizer`, `from litgpt.data import DataModule`) — no wrappers

**Storage format**
- Raw source: `data/multilingual-tinystories/{en,es}/{train,validation}.parquet` — already on disk, no download step
- Read parquet with pandas/pyarrow (both already in litgpt dependency set)
- Tokenized output: LitData streaming format (`litdata.optimize()` + `StreamingDataset`), consistent with existing `litgpt/data/tinystories.py`
- Cache layout: `data/.cache/{tokenizer_name}/{split}/` — e.g., `data/.cache/gpt2/D_std/`, `data/.cache/gpt2/D_harmful/`, `data/.cache/gpt2/D_unlabeled/`
- Separate on-disk directories per split (not combined with tags)

**Validation splits**
- D_std_val = val_EN (from `data/multilingual-tinystories/en/validation.parquet`)
- D_harmful_val = val_ES (from `data/multilingual-tinystories/es/validation.parquet`)
- D_unlabeled: no validation set (matches paper reference implementation)
- Three separate val loaders exposed by MultiDataLoader (one per split that has a val set)

**Training step sampling (replaces DATA-03 pre-generated list)**
- No pre-generated `data_split_order` file — dynamic weighted sampling per step
- At each step, MultiDataLoader samples which split to draw from, weighted by split sizes (derived from x% config)
- Interface: `multi_loader.next()` → `(batch, split_label)` — training loop calls `next()` each step, receives both the data and the split tag (`"D_std"` / `"D_harmful"` / `"D_unlabeled"`)
- Upsample factors (`upsample_std`, `upsample_harmful`, `upsample_unlabeled`) remain configurable per DATA-02

### Claude's Discretion
- Exact LitData chunk size and num_workers for `optimize()`
- How the prep script reports progress (tqdm vs print)
- Seed handling for weighted sampling reproducibility
- Whether MultiDataLoader wraps LitData `StreamingDataLoader` or standard PyTorch `DataLoader`

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DATA-01 | Data preparation script tokenizes `multilingual-tinystories` (tiktoken gpt2) and partitions into D_std (25% EN), D_harmful ((100-x)% ES), D_unlabeled (75% EN + x% ES) where x is a configurable sweep parameter; aligned with paper's `tinystories_tokenize_and_split.py` | Parquet files confirmed present on disk (2,119,719 rows each for EN/ES train; 21,990 each for val). LitData `optimize()` pattern confirmed in `litgpt/data/tinystories.py`. Tokenizer checkpoint available at `checkpoints/sgtm-moette-64m/final/`. |
| DATA-02 | `MultiDataLoader` wrapper provides per-split DataLoaders (D_std/D_harmful/D_unlabeled) with configurable upsample factors (`upsample_std`, `upsample_harmful`, `upsample_unlabeled`); consistent with LitGPT `DataModule` abstraction | `DataModule` base class interface confirmed in `litgpt/data/base.py`. `StreamingDataset`/`StreamingDataLoader` pattern confirmed in `litgpt/data/tinystories.py`. CONTEXT.md specifies `multi_loader.next()` → `(batch, split_label)` interface. |
| DATA-03 | Per the CONTEXT.md override: dynamic weighted sampling replaces pre-generated schedule. `MultiDataLoader.next()` samples which split to draw from, weighted by split sizes and upsample factors, returning `(batch, split_label)` each call | Dynamic sampling via `random.choices()` with weights derived from `(split_size * upsample_factor)`. Per-split infinite iterators via `itertools.cycle()` wrapping `StreamingDataLoader`. Seed handling for `random.seed()` at construction time. |
</phase_requirements>

---

## Summary

Phase 1 builds the data infrastructure that all subsequent SafeMoE training phases depend on. The task is straightforward: read two already-on-disk parquet files (English and Spanish TinyStories, 2.1M rows each), apply deterministic row-slicing to produce three splits per the SGTM proportions, tokenize each split to LitData streaming format, and serve batches through a `MultiDataLoader` that applies dynamic weighted sampling with a `(batch, split_label)` interface.

The key constraint is that the raw parquet source maps one-to-one with a dataset (EN → D_std/D_unlabeled, ES → D_harmful/D_unlabeled). The `x` sweep parameter controls how much ES data leaks into D_unlabeled versus staying in D_harmful. Concretely: with x=0 all 2.1M ES rows go to D_harmful; with x=50 half go to D_harmful and half to D_unlabeled. The 25%/75% EN split is fixed.

The existing `litgpt/data/tinystories.py` is the direct template: it shows exactly how to call `litdata.optimize()` with a `tokenize()` generator, configure `StreamingDataset` with `TokensLoader(block_size=...)`, and wrap in `StreamingDataLoader`. This phase follows that pattern for each of the three splits. The new code lives in `safemoe/data/` and imports from `litgpt` directly.

**Primary recommendation:** Implement `safemoe/data/prepare.py` as a standalone prep script and `safemoe/data/datamodule.py` as the `MultiDataLoader` class. Keep the prep script idempotent (skip if output dir already exists), use `tqdm` for progress, and fix the random seed at construction for reproducible sampling.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| litdata | 0.2.59 | Streaming tokenized dataset (`optimize()` + `StreamingDataset` + `StreamingDataLoader` + `TokensLoader`) | Already the TinyStories standard in litgpt; spec'd in pyproject.toml / uv.lock; confirmed as project dependency |
| pandas | 2.2.3 (installed in venv) | Read parquet source files | Already in venv; `pd.read_parquet()` is the obvious API for `.parquet` files |
| pyarrow | 23.0.1 (installed in venv) | Parquet backend for pandas | Already installed; required by pandas for parquet |
| litgpt.tokenizer.Tokenizer | 0.5.12 | Tokenize story text to int tensors | Project requires litgpt internals directly; gpt2 checkpoint at `checkpoints/sgtm-moette-64m/final/` uses HuggingFace tokenizer backend |
| litgpt.data.DataModule | 0.5.12 | Base class for `MultiDataLoader` | Project convention: all data classes subclass this |
| tqdm | 4.67.1 | Progress reporting in prep script | Already in venv; used by litgpt throughout |
| itertools (stdlib) | — | `cycle()` for infinite per-split iterators | No extra dep; standard pattern for infinite DataLoaders |
| random (stdlib) | — | Weighted sampling in `next()` | `random.choices(splits, weights=weights)` is the correct API |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| functools.partial (stdlib) | — | Bind tokenizer into `optimize()` fn arg | Same pattern as `litgpt/data/tinystories.py` line 63 |
| pathlib.Path (stdlib) | — | All path manipulation | litgpt convention; avoid string paths |
| dataclasses.dataclass (stdlib) | — | Config for MultiDataLoader | litgpt convention for DataModule configs |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `litdata.optimize()` + `StreamingDataset` | HuggingFace `datasets` with `.map()` | HF datasets is fine for small data but LitData is already the project standard and handles large streaming; consistency wins |
| `random.choices()` for weighted sampling | `torch.multinomial()` or `numpy.random.choice()` | `random.choices()` is stdlib, no GPU needed, clean seeding with `random.seed()`; torch.multinomial works but adds torch dependency to sampling logic |
| `StreamingDataLoader` | Standard PyTorch `DataLoader` | `StreamingDataLoader` integrates with `StreamingDataset`'s shuffle/chunk mechanisms; standard DataLoader loses streaming benefits |

**Installation:**
```bash
# litdata is in uv.lock but NOT currently installed in .venv
# Must install before any data prep code runs:
uv sync --extra extra
# or specifically:
uv pip install "litdata==0.2.59"
```

**Important:** `litdata` is in the uv lockfile at version 0.2.59 but is **not installed** in the project venv. The venv at `.venv/lib/python3.12/site-packages/` does not contain litdata. Wave 0 must install it.

---

## Architecture Patterns

### Recommended Project Structure

```
safemoe/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── prepare.py       # CLI script: tokenize parquet -> LitData splits
│   └── datamodule.py    # MultiDataLoader class
tests/
└── safemoe/
    └── data/
        ├── __init__.py
        └── test_datamodule.py   # unit tests for MultiDataLoader
```

Cache layout on disk (produced by `prepare.py`):
```
data/.cache/
└── {tokenizer_name}/           # e.g. "sgtm-moette-64m"
    ├── D_std/
    │   ├── train/              # LitData chunks
    │   └── val/
    ├── D_harmful/
    │   ├── train/
    │   └── val/
    └── D_unlabeled/
        └── train/              # no val for D_unlabeled
```

### Pattern 1: Parquet Row Slicing for Split Partitioning

**What:** Load the full EN/ES parquet into pandas, compute row-level index boundaries using the `x` parameter, then pass row-index slices as inputs to `litdata.optimize()`.

**When to use:** Always. This is the deterministic approach — same `x` value always produces identical splits, regardless of row ordering in the parquet.

**Split computation (verified against actual parquet sizes):**

```
EN train rows:  2,119,719
ES train rows:  2,119,719

D_std     = EN rows [0 : 0.25 * N_EN]              # 529,929 rows
D_unlabeled_EN = EN rows [0.25 * N_EN : N_EN]       # 1,589,790 rows
D_harmful = ES rows [0 : (1 - x/100) * N_ES]        # e.g. x=0 -> 2,119,719 rows
D_unlabeled_ES = ES rows [(1 - x/100) * N_ES : N_ES] # e.g. x=0 -> 0 rows
D_unlabeled = D_unlabeled_EN + D_unlabeled_ES        # concatenated

Val splits:
D_std_val     = EN validation.parquet (21,990 rows)
D_harmful_val = ES validation.parquet (21,990 rows)
D_unlabeled_val = None
```

**Key insight:** The parquet `text` column is the only column in both EN and ES files. The `tokenize()` function receives a list of story strings (not filenames), so the `inputs` argument to `optimize()` is a list of story strings directly — different from the `litgpt/data/tinystories.py` pattern which passes filenames.

```python
# Source: pattern from litgpt/data/tinystories.py, adapted for parquet input
def tokenize_stories(stories: list[str], tokenizer: Tokenizer):
    """Generator for litdata.optimize() — receives a list of story strings."""
    for text in stories:
        text = text.strip()
        tokens = tokenizer.encode(text, bos=True, eos=False)
        yield tokens
```

**Alternatively**, pass row-index ranges and load from parquet inside the function:
```python
# Pass (parquet_path, start_idx, end_idx) tuples as inputs
def tokenize_chunk(args: tuple, tokenizer: Tokenizer):
    path, start, end = args
    df = pd.read_parquet(path)
    for text in df["text"].iloc[start:end]:
        yield tokenizer.encode(text.strip(), bos=True, eos=False)
```

Both approaches are valid. The second avoids loading the full parquet into memory per worker but requires passing path tuples. Given parquet files are ~500MB each and workers share memory through the OS, the simpler first approach (pass strings directly) is fine for this scale.

### Pattern 2: LitData optimize() Call

**What:** Standard LitData tokenization pattern, directly from `litgpt/data/tinystories.py`.

**When to use:** Once per split during prep. Idempotent — skip if output dir already exists.

```python
# Source: litgpt/data/tinystories.py lines 62-69
from litdata import TokensLoader, optimize
from functools import partial

optimize(
    fn=partial(tokenize_stories, tokenizer=tokenizer),
    inputs=story_list,        # list of story strings for this split
    output_dir=str(output_dir),
    num_workers=min(os.cpu_count() - 1, 8),  # Claude's discretion
    chunk_bytes="200MB",                       # litgpt standard
    item_loader=TokensLoader(),
)
```

**Claude's discretion: num_workers.** The reference uses `os.cpu_count() - 1`. For parquet-based input (no file I/O per worker), fewer workers are acceptable. Recommend 4 as default, configurable via CLI arg.

### Pattern 3: MultiDataLoader with Dynamic Weighted Sampling

**What:** A `DataModule` subclass that holds three `StreamingDataLoader` iterators (one per split) and exposes `next()` for the training loop, with `random.choices()` weighted sampling.

**When to use:** Always during SGTM training. The training loop calls `multi_loader.next()` at each step.

```python
# Source: pattern derived from litgpt/data/base.py and litgpt/data/tinystories.py
import random
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from litdata.streaming import StreamingDataLoader, StreamingDataset, TokensLoader
from litgpt.data import DataModule
from litgpt.tokenizer import Tokenizer

@dataclass
class MultiDataLoader(DataModule):
    cache_dir: Path = Path("data/.cache")
    x_harmful_unlabeled: float = 0.0    # x% of ES goes to D_unlabeled
    upsample_std: float = 1.0
    upsample_harmful: float = 1.0
    upsample_unlabeled: float = 1.0
    seed: int = 42
    num_workers: int = 4

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)

    def connect(self, tokenizer=None, batch_size=1, max_seq_length=-1):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length + 1  # +1 for next-token target

    def setup(self, stage=""):
        random.seed(self.seed)
        tokenizer_name = self.tokenizer.model_name if self.tokenizer else "default"
        base = self.cache_dir / tokenizer_name

        def make_loader(split_name):
            ds = StreamingDataset(
                input_dir=str(base / split_name / "train"),
                item_loader=TokensLoader(block_size=self.max_seq_length),
                shuffle=True,
            )
            return StreamingDataLoader(
                ds, batch_size=self.batch_size, num_workers=self.num_workers,
                pin_memory=True, drop_last=True,
            )

        self._loaders = {
            "D_std":       make_loader("D_std"),
            "D_harmful":   make_loader("D_harmful"),
            "D_unlabeled": make_loader("D_unlabeled"),
        }
        # Infinite iterators per split
        self._iters = {k: itertools.cycle(v) for k, v in self._loaders.items()}

        # Weights: proportional to dataset size * upsample factor
        # Dataset sizes are derived from split proportions; use upsample factors directly
        # since the underlying datasets already reflect split proportions.
        self._split_names = ["D_std", "D_harmful", "D_unlabeled"]
        self._weights = [self.upsample_std, self.upsample_harmful, self.upsample_unlabeled]

    def next(self):
        """Draw one batch from a split sampled by weighted probability.
        Returns: (batch_tensor, split_label_str)
        """
        split = random.choices(self._split_names, weights=self._weights, k=1)[0]
        batch = next(self._iters[split])
        return batch, split

    def val_dataloaders(self) -> dict:
        """Returns per-split val DataLoaders. D_unlabeled has no val set."""
        tokenizer_name = self.tokenizer.model_name if self.tokenizer else "default"
        base = self.cache_dir / tokenizer_name

        def make_val_loader(split_name):
            ds = StreamingDataset(
                input_dir=str(base / split_name / "val"),
                item_loader=TokensLoader(block_size=self.max_seq_length),
                shuffle=False,
            )
            return StreamingDataLoader(
                ds, batch_size=self.batch_size, num_workers=self.num_workers,
                pin_memory=True, drop_last=False,
            )

        return {
            "D_std":     make_val_loader("D_std"),
            "D_harmful": make_val_loader("D_harmful"),
        }

    def train_dataloader(self):
        """Standard DataModule interface — not used in SGTM loop; use next() instead."""
        return self._loaders["D_std"]  # placeholder for LightningDataModule compat

    def val_dataloader(self):
        """Standard DataModule interface — not used in SGTM loop; use val_dataloaders() instead."""
        return list(self.val_dataloaders().values())
```

### Pattern 4: Prep Script as Idempotent CLI

**What:** `python -m safemoe.data.prepare` with argparse or jsonargparse arguments. Checks if output dir exists before running optimize().

**When to use:** Once before training. Safe to re-run with different `x` values (use different cache subdirs keyed by x value, or always re-run).

```python
# safemoe/data/prepare.py
def prepare(
    checkpoint_dir: Path = Path("checkpoints/sgtm-moette-64m/final"),
    data_dir: Path = Path("data/multilingual-tinystories"),
    cache_dir: Path = Path("data/.cache"),
    x: float = 0.0,           # % of ES that goes to D_unlabeled
    num_workers: int = 4,
    chunk_bytes: str = "200MB",
):
    tokenizer = Tokenizer(checkpoint_dir)
    en_train = pd.read_parquet(data_dir / "en" / "train.parquet")["text"].tolist()
    es_train = pd.read_parquet(data_dir / "es" / "train.parquet")["text"].tolist()
    en_val   = pd.read_parquet(data_dir / "en" / "validation.parquet")["text"].tolist()
    es_val   = pd.read_parquet(data_dir / "es" / "validation.parquet")["text"].tolist()

    n_en = len(en_train)
    n_es = len(es_train)
    d_std_stories        = en_train[:int(0.25 * n_en)]
    d_unlabeled_en       = en_train[int(0.25 * n_en):]
    n_harmful            = int((1.0 - x / 100.0) * n_es)
    d_harmful_stories    = es_train[:n_harmful]
    d_unlabeled_es       = es_train[n_harmful:]
    d_unlabeled_stories  = d_unlabeled_en + d_unlabeled_es

    out_base = cache_dir / tokenizer.model_name
    _maybe_optimize(d_std_stories,       out_base / "D_std/train",     tokenizer, num_workers, chunk_bytes)
    _maybe_optimize(en_val,              out_base / "D_std/val",       tokenizer, 1, chunk_bytes)
    _maybe_optimize(d_harmful_stories,   out_base / "D_harmful/train", tokenizer, num_workers, chunk_bytes)
    _maybe_optimize(es_val,              out_base / "D_harmful/val",   tokenizer, 1, chunk_bytes)
    _maybe_optimize(d_unlabeled_stories, out_base / "D_unlabeled/train", tokenizer, num_workers, chunk_bytes)
    # No val for D_unlabeled
```

### Anti-Patterns to Avoid

- **Mixing splits in a single LitData directory:** Each split MUST be its own directory. Combined directories cannot be used to serve per-split batches.
- **Downloading data:** The parquet files are already on disk. No HuggingFace download or network access required.
- **Using `litgpt.tokenizer.Tokenizer` as a gpt2 raw tiktoken caller:** The existing Tokenizer class requires a checkpoint directory with `tokenizer.json` or `tokenizer.model`. The checkpoint at `checkpoints/sgtm-moette-64m/final/` does NOT contain tokenizer files — only `lit_model.pth` and `model_config.yaml`. A GPT-2 tokenizer checkpoint must be available or downloaded separately. See Pitfall section.
- **Using standard Python `DataLoader` for StreamingDataset:** `StreamingDataLoader` is required; standard `DataLoader` does not support `StreamingDataset`'s distributed chunk management.
- **Hardcoding split proportions:** The `x` parameter MUST be configurable. The planner should expose it as a CLI argument for the sweep.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Streaming tokenized dataset | Custom binary format, HDF5, numpy memmap | `litdata.optimize()` + `StreamingDataset` + `TokensLoader` | Already the project standard; handles shuffling, chunking, distributed loading; tested in existing litgpt test suite |
| Infinite cycling over a DataLoader | Custom `__iter__` with restart logic | `itertools.cycle(dataloader)` | One-liner; handles edge cases (empty dataset, StopIteration); perfectly idiomatic |
| Weighted random split selection | Custom probability distribution, numpy sampling | `random.choices(splits, weights=weights)` | Stdlib; correct for this use case; clean seeding with `random.seed()` |
| Parquet reading | Custom binary reader, CSV conversion | `pandas.read_parquet()` | Already installed; pyarrow backend already installed; zero-copy columnar read |
| Tokenizer | Custom BPE implementation | `litgpt.tokenizer.Tokenizer` wrapping HuggingFace `tokenizers` | Already in project; handles BOS/EOS correctly; tested against gpt2 vocabulary |

**Key insight:** Every component needed for this phase already exists in the installed environment. The task is assembly and configuration, not implementation of new algorithms.

---

## Common Pitfalls

### Pitfall 1: Tokenizer Checkpoint Missing Tokenizer Files

**What goes wrong:** `litgpt.tokenizer.Tokenizer(checkpoint_dir)` requires `tokenizer.json` (HuggingFace format) or `tokenizer.model` (SentencePiece) in the checkpoint directory. The existing `checkpoints/sgtm-moette-64m/final/` contains only `lit_model.pth`, `model_config.yaml`, and `ablated/` — NO tokenizer files. The model config shows `vocab_size: 50257` (GPT-2 vocabulary), confirming tiktoken gpt2 tokenization, but the tokenizer files are not present.

**Why it happens:** LitGPT separates model weights from tokenizer files. The GPT-2 tokenizer must be separately downloaded from HuggingFace or provided as a standalone directory.

**How to avoid:** Before running `prepare.py`, ensure a GPT-2 tokenizer checkpoint directory exists. Options:
1. Run `litgpt download gpt2 --tokenizer_only` to get the GPT-2 tokenizer files.
2. Manually create a checkpoint directory with `tokenizer.json` + `tokenizer_config.json` from the `gpt2` model on HuggingFace.
3. Use `tiktoken` directly (bypasses `litgpt.Tokenizer`) — but this breaks the established pattern.

**Recommendation:** Wave 0 task: verify or create GPT-2 tokenizer files. The prep script should accept a `--checkpoint_dir` that points to a directory with tokenizer files, separate from the model checkpoint.

**Warning signs:** `NotImplementedError` or `NotADirectoryError` from `litgpt.tokenizer.Tokenizer.__init__` when checkpoint dir lacks `tokenizer.json` or `tokenizer.model`.

### Pitfall 2: litdata Not Installed in Project Venv

**What goes wrong:** `from litdata import optimize` raises `ModuleNotFoundError`. The package is specified in `pyproject.toml` optional deps (`[extra]`) and in `uv.lock` at version 0.2.59, but is NOT currently installed in `.venv/lib/python3.12/site-packages/`.

**Why it happens:** The venv was created without the `[extra]` group. The pip command uses system Python 3.10, not the venv Python 3.12.

**How to avoid:** Wave 0 installation task: `uv sync --extra extra` from the project root. Verify with `.venv/bin/python3 -c "import litdata; print(litdata.__version__)"`.

**Warning signs:** `ModuleNotFoundError: No module named 'litdata'` when running prepare.py or tests.

### Pitfall 3: D_unlabeled Size Explosion at x=0

**What goes wrong:** With x=0 (the default and most common sweep value), D_unlabeled is 75% of EN = 1,589,790 rows — about 3x larger than D_std and also 3x smaller than D_harmful. The MultiDataLoader's default weights (1.0, 1.0, 1.0) will undersample D_harmful relative to its actual data volume. At x=50, D_harmful drops to 50% of 2.1M = ~1M rows and D_unlabeled grows to ~75% EN + ~50% ES = ~2.6M rows. The planner must expose upsample factors to compensate.

**Why it happens:** Natural data proportions are heavily skewed because ES maps directly to D_harmful.

**How to avoid:** Expose `upsample_harmful`, `upsample_std`, `upsample_unlabeled` as configurable parameters. Default weights should reflect the paper's intended training ratio, not natural data frequencies.

**Warning signs:** Training with tiny D_harmful contribution even though the dataset is large — check per-split batch counts per epoch.

### Pitfall 4: StreamingDataset Shuffle with Small Validation Sets

**What goes wrong:** Both val parquets have 21,990 rows. After tokenization with block_size=256 (or 257 with +1 for target), these become roughly 21,990 * avg_tokens_per_story / 257 ≈ several thousand streaming chunks. Setting `shuffle=True` on val loaders is incorrect — validation must be deterministic and consistent across evaluations.

**How to avoid:** Always set `shuffle=False` for val `StreamingDataset` instances. Set `shuffle=True` only for train instances.

### Pitfall 5: optimize() Workers and Environment Variables

**What goes wrong:** The `tokenize()` function in `litgpt/data/tinystories.py` reads `DATA_OPTIMIZER_GLOBAL_RANK` and `DATA_OPTIMIZER_NUM_WORKERS` environment variables for tqdm positioning. If these are not set (because `optimize()` is called with `num_workers=1` or from a test), `os.environ["DATA_OPTIMIZER_GLOBAL_RANK"]` raises `KeyError`.

**How to avoid:** Either use `.get()` with a default: `int(os.environ.get("DATA_OPTIMIZER_GLOBAL_RANK", "0"))`, or don't read these in the tokenize function (use a simpler progress reporting strategy). Since the tokenize function for parquet-based input is new code, write it to avoid this env var dependency entirely.

### Pitfall 6: x Parameter Invalidates Cached Splits

**What goes wrong:** If the prep script caches to `data/.cache/{tokenizer_name}/D_harmful/train/` and is run once with x=0 and again with x=25, the second run will skip the D_harmful directory (it already exists) but produce wrong splits.

**How to avoid:** Include `x` value in the cache path, e.g., `data/.cache/{tokenizer_name}/x{x}/D_harmful/train/`. Alternatively, always re-run (remove idempotency check for the split directories). Recommended: include x in path to support sweep experiments without re-tokenizing.

---

## Code Examples

### Reading and Slicing Parquet Files

```python
# Source: direct pandas/pyarrow usage, verified installed (pandas 2.2.3, pyarrow 23.0.1)
import pandas as pd

en_df = pd.read_parquet("data/multilingual-tinystories/en/train.parquet")
# Shape: (2119719, 1), column: "text"
en_stories = en_df["text"].tolist()  # list of 2,119,719 strings

# Split computation for x=25 (25% ES goes to D_unlabeled)
x = 25
n_en = len(en_stories)
n_es_harmful = int((1.0 - x / 100.0) * n_en)  # (100-25)% = 75% of ES

d_std_stories = en_stories[:int(0.25 * n_en)]           # 529,929 rows
d_unlabeled_en = en_stories[int(0.25 * n_en):]           # 1,589,790 rows
es_stories = pd.read_parquet("data/multilingual-tinystories/es/train.parquet")["text"].tolist()
d_harmful_stories = es_stories[:n_es_harmful]             # 1,589,789 rows (75%)
d_unlabeled_es = es_stories[n_es_harmful:]                # 529,930 rows (25%)
d_unlabeled_stories = d_unlabeled_en + d_unlabeled_es    # 2,119,720 rows
```

### litdata.optimize() for One Split

```python
# Source: adapted from litgpt/data/tinystories.py lines 60-78
from litdata import TokensLoader, optimize
from functools import partial
from pathlib import Path
from litgpt.tokenizer import Tokenizer

def tokenize_stories(stories: list, tokenizer: Tokenizer):
    """Generator function for litdata.optimize()."""
    for text in stories:
        text = text.strip()
        tokens = tokenizer.encode(text, bos=True, eos=False)
        yield tokens

def maybe_optimize(stories, output_dir, tokenizer, num_workers=4, chunk_bytes="200MB"):
    if Path(output_dir).is_dir():
        print(f"Skipping {output_dir} (already exists)")
        return
    optimize(
        fn=partial(tokenize_stories, tokenizer=tokenizer),
        inputs=[stories],          # single input item (the full list)
        output_dir=str(output_dir),
        num_workers=num_workers,
        chunk_bytes=chunk_bytes,
        item_loader=TokensLoader(),
    )
```

**Note on `inputs` format:** `litdata.optimize()` calls `fn(input_item)` for each item in `inputs`. If you pass `inputs=[stories_list]`, fn receives one call with the full list. If you want parallelism, split stories into sub-lists: `inputs=[stories[i::n_workers] for i in range(n_workers)]`.

### StreamingDataset + StreamingDataLoader

```python
# Source: litgpt/data/tinystories.py lines 82-92 and 94-105
from litdata.streaming import StreamingDataLoader, StreamingDataset, TokensLoader

dataset = StreamingDataset(
    input_dir=str(output_dir / "D_std" / "train"),
    item_loader=TokensLoader(block_size=max_seq_length + 1),  # +1 for target token
    shuffle=True,
)
dataloader = StreamingDataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)
```

### Dynamic Weighted Sampling (MultiDataLoader.next)

```python
# Source: stdlib random.choices, itertools.cycle
import random
import itertools

# At setup() time (call random.seed once):
random.seed(42)
iters = {split: itertools.cycle(loader) for split, loader in loaders.items()}
split_names = ["D_std", "D_harmful", "D_unlabeled"]
weights = [upsample_std, upsample_harmful, upsample_unlabeled]

# At each training step:
def next_batch():
    split = random.choices(split_names, weights=weights, k=1)[0]
    batch = next(iters[split])
    return batch, split
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| HuggingFace `datasets` map/cache for tokenization | `litdata.optimize()` + `StreamingDataset` | litgpt ~2023-2024 | LitData handles large datasets without loading all into RAM; streaming is critical at scale |
| Pre-generated split schedule list (original DATA-03) | Dynamic weighted sampling per step | CONTEXT.md override (2026-03-14) | Simpler implementation; no file I/O during training; weights are runtime-configurable |
| Single-split DataModule | MultiDataLoader with per-split loaders | This phase (new) | Required for SGTM three-stream training |

**Deprecated/outdated:**
- Pre-generated `data_split_order` list: superseded by dynamic weighted sampling per CONTEXT.md. Do not implement DATA-03 as originally specified in REQUIREMENTS.md — the CONTEXT.md override takes precedence.

---

## Open Questions

1. **GPT-2 tokenizer checkpoint location**
   - What we know: `checkpoints/sgtm-moette-64m/final/` has no tokenizer files; model uses 50257 vocab (GPT-2)
   - What's unclear: Is there a GPT-2 tokenizer directory elsewhere on the filesystem? Does the prep script need to download it?
   - Recommendation: Wave 0 task — check `litgpt download gpt2` output path, or locate `tokenizer.json` from any GPT-2 checkpoint. If not present, the prep script needs a one-time download step.

2. **litdata.optimize() inputs format for list-of-strings**
   - What we know: The tinystories.py pattern passes filenames as inputs (each file processed by one worker). Our case passes strings.
   - What's unclear: Whether `inputs=[full_list]` (one worker gets all) or `inputs=[sublist_per_worker]` (n workers each get a chunk) is the right split for parallelism.
   - Recommendation: Use `inputs=[stories[i::num_workers] for i in range(num_workers)]` to properly parallelize across workers. Each worker processes every Nth story. Verify against litdata docs.

3. **`itertools.cycle` and StreamingDataLoader compatibility**
   - What we know: `StreamingDataLoader` implements `__iter__` and yields batches; `itertools.cycle` calls `__iter__` each time the loader is exhausted.
   - What's unclear: Whether re-iterating a `StreamingDataLoader` via `cycle()` properly reshuffles, or whether it replays the same order.
   - Recommendation: Test in Wave 1 by iterating through a small streaming loader twice with `cycle()` and checking that batch order differs between cycles (assuming shuffle=True).

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.1.1+ (in pyproject.toml `[test]` optional deps) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` (already configured with `--strict-markers --color=yes`) |
| Quick run command | `pytest tests/safemoe/data/ -x -q` |
| Full suite command | `pytest tests/safemoe/data/ -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-01 | Given x=0, D_std = 25% EN, D_harmful = 100% ES, D_unlabeled = 75% EN (correct row counts) | unit | `pytest tests/safemoe/data/test_prepare.py::test_split_proportions -x` | ❌ Wave 0 |
| DATA-01 | Given x=50, D_harmful = 50% ES, D_unlabeled = 75% EN + 50% ES | unit | `pytest tests/safemoe/data/test_prepare.py::test_split_proportions_x50 -x` | ❌ Wave 0 |
| DATA-01 | Tokenized LitData dirs exist and are readable by StreamingDataset after prepare() | integration | `pytest tests/safemoe/data/test_prepare.py::test_litdata_output_readable -x` | ❌ Wave 0 |
| DATA-02 | MultiDataLoader.next() returns (batch_tensor, split_label) tuples | unit | `pytest tests/safemoe/data/test_datamodule.py::test_next_returns_tuple -x` | ❌ Wave 0 |
| DATA-02 | MultiDataLoader.val_dataloaders() returns dict with D_std and D_harmful keys only | unit | `pytest tests/safemoe/data/test_datamodule.py::test_val_dataloaders_keys -x` | ❌ Wave 0 |
| DATA-02 | Batch tensor shape is (batch_size, max_seq_length+1) | unit | `pytest tests/safemoe/data/test_datamodule.py::test_batch_shape -x` | ❌ Wave 0 |
| DATA-03 | Over 300 calls to next(), split_label frequencies approximate weights (chi-squared test) | unit | `pytest tests/safemoe/data/test_datamodule.py::test_sampling_weights -x` | ❌ Wave 0 |
| DATA-03 | Same seed produces same sampling sequence | unit | `pytest tests/safemoe/data/test_datamodule.py::test_seed_reproducibility -x` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `pytest tests/safemoe/data/ -x -q`
- **Per wave merge:** `pytest tests/safemoe/data/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/safemoe/__init__.py` — package marker for test discovery
- [ ] `tests/safemoe/data/__init__.py` — package marker for test discovery
- [ ] `tests/safemoe/data/test_prepare.py` — covers DATA-01 (split proportions, LitData output)
- [ ] `tests/safemoe/data/test_datamodule.py` — covers DATA-02 and DATA-03 (MultiDataLoader interface, weighted sampling)
- [ ] `safemoe/__init__.py` — package marker
- [ ] `safemoe/data/__init__.py` — package marker
- [ ] Framework install: `uv sync --extra extra` — litdata 0.2.59 not currently in venv

**Note on test approach for data tests:** Follow the pattern from `tests/data/test_tinystories.py` — use `litdata.optimize()` with fake/synthetic data to create small temporary LitData dirs in `tmp_path`, then test `StreamingDataset` reading. Do not require the real 2.1M-row parquets for unit tests.

---

## Sources

### Primary (HIGH confidence)
- `/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/litgpt/data/tinystories.py` — canonical litdata optimize() + StreamingDataset pattern
- `/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/litgpt/data/base.py` — DataModule base class interface
- `/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/litgpt/tokenizer.py` — Tokenizer class, checkpoint dir requirement
- `/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/tests/data/test_tinystories.py` — test patterns for litdata (fake_chunk, StreamingDataset usage)
- `/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/pyproject.toml` — installed versions (litdata=0.2.59, pandas=2.2.3, pyarrow=23.0.1)
- `/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/.planning/phases/01-data-pipeline/01-CONTEXT.md` — locked decisions and interface spec
- Direct parquet inspection — EN/ES both 2,119,719 train rows, 21,990 val rows, single `text` column
- `.venv/lib/python3.12/site-packages/` inspection — confirmed litdata NOT installed; pandas, pyarrow, torch, lightning ARE installed

### Secondary (MEDIUM confidence)
- `/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/.planning/research/STACK.md` — cross-reference for litdata version and data pipeline approach
- `/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/.planning/research/ARCHITECTURE.md` — MultiDataLoader interface and data flow context

### Tertiary (LOW confidence)
- litdata 0.2.59 `inputs` parallelization behavior — inferred from tinystories.py pattern; not verified against litdata source or docs

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries verified in venv or lockfile; API verified against live litgpt code
- Architecture: HIGH — directly modeled on `litgpt/data/tinystories.py`; parquet sizes confirmed by direct inspection
- Pitfalls: HIGH — tokenizer file gap and litdata install gap confirmed by direct filesystem inspection; other pitfalls from code analysis
- Validation architecture: HIGH — test framework confirmed in pyproject.toml; test patterns verified in existing test suite

**Research date:** 2026-03-14
**Valid until:** 2026-04-14 (stable stack — litdata 0.2.59, pandas, pyarrow are pinned in lockfile)
