# Phase 1: Data Pipeline - Context

**Gathered:** 2026-03-14 (updated 2026-03-15)
**Status:** Ready for planning

<domain>
## Phase Boundary

Build a data preparation script that tokenizes the TinyStories bilingual dataset (English + Spanish) and partitions it into three on-disk splits (D_std, D_harmful, D_unlabeled), plus a MultiDataLoader that exposes per-split DataLoaders to the training loop. Val sets for D_std and D_harmful are derived from the parquet validation shards; D_unlabeled has no val set (per paper reference implementation).

</domain>

<decisions>
## Implementation Decisions

### Package structure
- New `safemoe/` top-level package alongside `litgpt/` — all SafeMoE-specific code lives there
- Data code at `safemoe/data/` subdirectory (mirrors `litgpt/data/` layout)
- `safemoe/` imports `litgpt` internals directly (e.g., `from litgpt.tokenizer import Tokenizer`, `from litgpt.data import DataModule`) — no wrappers

### Storage format
- Raw source: `data/multilingual-tinystories/{en,es}/{train,validation}.parquet` — already on disk, no download step
  - Confirmed row counts: 2,119,719 EN train, 2,119,719 ES train, 21,990 EN val, 21,990 ES val
- Read parquet with pandas/pyarrow (both already in litgpt dependency set)
- Tokenized output: LitData streaming format (`litdata.optimize()` + `StreamingDataset`), consistent with existing `litgpt/data/tinystories.py`
- Cache layout: `data/.cache/{tokenizer_name}/{x}-{y}/{split}/{train|val}` — e.g. `data/.cache/Qwen3-30B-A3B-Base/0-25/D_std/train/`
  - Directory names use integer format: `{x}-{y}` (e.g. `0-25`, not `x0.0_y25.0`)
- Separate on-disk directories per split (not combined with tags)

### Split formula (two-parameter scheme)
- `x` = percentage of ES rows that goes to D_unlabeled (ES leak parameter)
- `y` = percentage of EN rows that goes to D_std (EN retention parameter)
- Split logic:
  - D_std = first y% of EN rows
  - D_harmful = first (100-x)% of ES rows
  - D_unlabeled = remaining (100-y)% of EN rows + remaining x% of ES rows
- Defaults: x=0, y=25 (matches original paper proportions: 25% EN → D_std, 100% ES → D_harmful)
- Example with defaults (x=0, y=25): D_std≈529,929 EN, D_harmful=2,119,719 ES, D_unlabeled≈1,589,790 EN
- x and y are independent sweep parameters for ablation studies

### Tokenizer
- Use `checkpoints/Qwen3-30B-A3B-Base/` as the tokenizer checkpoint directory (151,643-vocab Qwen3 BPE)
- **Overrides original requirement DATA-01** which specified tiktoken gpt2 — using Qwen3 tokenizer instead
- Default `--checkpoint_dir` for prepare.py CLI: `checkpoints/Qwen3-30B-A3B-Base`
- The tokenizer's `model_name` property is used for cache dir naming (e.g. `Qwen3-30B-A3B-Base`)

### Validation splits
- D_std_val = val_EN (from `data/multilingual-tinystories/en/validation.parquet`)
- D_harmful_val = val_ES (from `data/multilingual-tinystories/es/validation.parquet`)
- D_unlabeled: **no validation set** (matches paper reference implementation)

### MultiDataLoader interface
- MultiDataLoader is a **loader registry** — it wraps three StreamingDataLoaders and exposes them to the training loop
- **No `next()` method** — the training loop controls which split to use each step
- Primary access method: `get_loader(split_name: str) -> DataLoader`
  - Training loop calls `get_loader('D_std')`, `get_loader('D_harmful')`, `get_loader('D_unlabeled')` directly
  - Training loop manages its own iterators (e.g. `iter(multi_loader.get_loader('D_harmful'))`)
- Val access: `val_dataloaders() -> dict` returns `{"D_std": loader, "D_harmful": loader}`
- LightningDataModule compat methods present: `train_dataloader()` and `val_dataloader()` (for framework integration)

### Sampling / scheduling (belongs in training loop, Phase 3)
- Training loop decides which split to use each step via `random.choices(['D_std', 'D_harmful', 'D_unlabeled'], weights=[...])`
- Upsample weights (`upsample_std`, `upsample_harmful`, `upsample_unlabeled`) are training loop config, NOT stored in MultiDataLoader
- No minimum mixing gap — same split can appear in consecutive steps
- No pre-generated `data_split_order` file (dynamic sampling at training time)

### Claude's Discretion
- Exact LitData chunk size and num_workers for `optimize()`
- How the prep script reports progress (tqdm vs print)
- Seed handling in tests for reproducibility
- MultiDataLoader constructor field names (beyond the interface decisions above)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `litgpt/data/tinystories.py`: Reference implementation — shows the `optimize()` + `StreamingDataset` + `StreamingDataLoader` pattern for TinyStories tokenization. Reuse the `tokenize()` function pattern and LitData API calls.
- `litgpt/data/base.py`: `DataModule` base class with `connect()`, `train_dataloader()`, `val_dataloader()` — `MultiDataLoader` should follow the same interface.
- `litgpt/tokenizer.py`: `Tokenizer` wrapper — use for Qwen3 BPE tokenization. Requires `tokenizer.json` in checkpoint dir (available at `checkpoints/Qwen3-30B-A3B-Base/`).

### Established Patterns
- Data prep: `litdata.optimize(fn=tokenize_fn, inputs=files, output_dir=..., chunk_bytes="200MB", item_loader=TokensLoader())`
- DataLoader: `StreamingDataLoader(StreamingDataset(input_dir=..., item_loader=TokensLoader(block_size=...)), batch_size=..., num_workers=...)`
- Config: `@dataclass` extending `DataModule`, with `connect()` for tokenizer/batch_size/max_seq_length

### Integration Points
- Training loop (Phase 3) calls `multi_loader.get_loader(split_label)` to get the appropriate DataLoader each step
- Training loop manages its own split schedule and does `random.choices()` for weighted sampling
- Val evaluation (Phase 4 / EVAL-01) calls `multi_loader.val_dataloaders()` → `{"D_std": DataLoader, "D_harmful": DataLoader}`
- CLI entry `python -m safemoe pretrain` will instantiate `MultiDataLoader` from YAML config

</code_context>

<specifics>
## Specific Ideas

- The original paper's reference code (`tinystories_tokenize_and_split.py`) is the canonical reference for partitioning logic — researcher should check it
- `data/multilingual-tinystories` already exists with `{en,es}/{train,validation}.parquet`; prep script reads from there, no download logic needed
- Two-parameter split (x, y) enables independent ablation of ES leakage and EN retention in experiments
- Cache directory naming uses integer format `{x}-{y}` (e.g., `0-25`) for human-readable directory names

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 01-data-pipeline*
*Context gathered: 2026-03-14*
*Context updated: 2026-03-15 — two-param split (x/y), Qwen3 tokenizer, revised MultiDataLoader interface (get_loader() instead of next())*
