# Phase 1: Data Pipeline - Context

**Gathered:** 2026-03-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Build a data preparation script that tokenizes the TinyStories bilingual dataset (English + Spanish) and partitions it into three on-disk splits (D_std, D_harmful, D_unlabeled), plus a MultiDataLoader that serves per-split batches to the training loop via dynamic weighted sampling. Val sets for D_std and D_harmful are derived from the parquet validation shards; D_unlabeled has no val set (per paper reference implementation).

</domain>

<decisions>
## Implementation Decisions

### Package structure
- New `safemoe/` top-level package alongside `litgpt/` — all SafeMoE-specific code lives there
- Data code at `safemoe/data/` subdirectory (mirrors `litgpt/data/` layout)
- `safemoe/` imports `litgpt` internals directly (e.g., `from litgpt.tokenizer import Tokenizer`, `from litgpt.data import DataModule`) — no wrappers

### Storage format
- Raw source: `data/multilingual-tinystories/{en,es}/{train,validation}.parquet` — already on disk, no download step
- Read parquet with pandas/pyarrow (both already in litgpt dependency set)
- Tokenized output: LitData streaming format (`litdata.optimize()` + `StreamingDataset`), consistent with existing `litgpt/data/tinystories.py`
- Cache layout: `data/.cache/{tokenizer_name}/{split}/` — e.g., `data/.cache/gpt2/D_std/`, `data/.cache/gpt2/D_harmful/`, `data/.cache/gpt2/D_unlabeled/`
- Separate on-disk directories per split (not combined with tags)

### Validation splits
- Use the parquet validation shards directly:
  - D_std_val = val_EN (from `data/multilingual-tinystories/en/validation.parquet`)
  - D_harmful_val = val_ES (from `data/multilingual-tinystories/es/validation.parquet`)
  - D_unlabeled: **no validation set** (matches paper reference implementation)
- Three separate val loaders exposed by MultiDataLoader (one per split that has a val set)

### Training step sampling (replaces DATA-03 pre-generated list)
- No pre-generated `data_split_order` file — dynamic weighted sampling per step
- At each step, MultiDataLoader samples which split to draw from, weighted by split sizes (derived from x% config)
- Interface: `multi_loader.next()` → `(batch, split_label)` — training loop calls `next()` each step, receives both the data and the split tag (`"D_std"` / `"D_harmful"` / `"D_unlabeled"`)
- Upsample factors (`upsample_std`, `upsample_harmful`, `upsample_unlabeled`) remain configurable per DATA-02

### Claude's Discretion
- Exact LitData chunk size and num_workers for `optimize()`
- How the prep script reports progress (tqdm vs print)
- Seed handling for weighted sampling reproducibility
- Whether MultiDataLoader wraps LitData `StreamingDataLoader` or standard PyTorch `DataLoader`

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `litgpt/data/tinystories.py`: Reference implementation — shows the `optimize()` + `StreamingDataset` + `StreamingDataLoader` pattern for TinyStories tokenization. Reuse the `tokenize()` function pattern and LitData API calls.
- `litgpt/data/base.py`: `DataModule` base class with `connect()`, `train_dataloader()`, `val_dataloader()` — `MultiDataLoader` should follow the same interface.
- `litgpt/tokenizer.py`: `Tokenizer` wrapper — use for tiktoken gpt2 tokenization.

### Established Patterns
- Data prep: `litdata.optimize(fn=tokenize_fn, inputs=files, output_dir=..., chunk_bytes="200MB", item_loader=TokensLoader())`
- DataLoader: `StreamingDataLoader(StreamingDataset(input_dir=..., item_loader=TokensLoader(block_size=...)), batch_size=..., num_workers=...)`
- Config: `@dataclass` extending `DataModule`, with `connect()` for tokenizer/batch_size/max_seq_length

### Integration Points
- Training loop (Phase 3) calls `multi_loader.next()` → `(batch, split_label)` each step
- Val evaluation (Phase 4 / EVAL-01) calls per-split val loaders: `multi_loader.val_dataloaders()` → `{split: DataLoader}`
- CLI entry `python -m safemoe pretrain` will instantiate `MultiDataLoader` from YAML config

</code_context>

<specifics>
## Specific Ideas

- The original paper's reference code (`tinystories_tokenize_and_split.py`) is the canonical reference for partitioning logic — researcher should check it
- `data/multilingual-tinystories` already exists with `{en,es}/{train,validation}.parquet`; prep script reads from there, no download logic needed
- Dynamic sampling means `MultiDataLoader` internally tracks iterators for each split and restarts them when exhausted (infinite cycle), sampling based on configured split weights each call to `next()`

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 01-data-pipeline*
*Context gathered: 2026-03-14*
