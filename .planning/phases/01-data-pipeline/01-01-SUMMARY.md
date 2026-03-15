---
phase: 01-data-pipeline
plan: 01
subsystem: data
tags: [litdata, tokenization, qwen3, parquet, streaming-dataset]

# Dependency graph
requires: []
provides:
  - "safemoe Python package (safemoe/__init__.py, safemoe/data/__init__.py)"
  - "compute_splits(en, es, x, y) — two-param split formula producing D_std/D_harmful/D_unlabeled lists"
  - "prepare() — tokenizes parquet stories and writes LitData chunks to cache_dir/{tokenizer_name}/{x}-{y}/"
  - "tokenize_stories() generator — yields torch.Tensor token sequences for litdata.optimize()"
  - "_maybe_optimize() — idempotent litdata.optimize() wrapper using start_method='fork'"
  - "prepare.py CLI — argparse entry point for --checkpoint_dir/--data_dir/--cache_dir/--x/--y/--num_workers"
affects: [02-data-pipeline, 03-training-loop, 04-evaluation]

# Tech tracking
tech-stack:
  added: [litdata==0.2.59, pandas (already present), pyarrow (already present)]
  patterns:
    - "litdata.optimize() with start_method='fork' to avoid spawn pickle errors for in-memory tokenizers"
    - "inputs=[stories[i::n] for i in range(n)] worker-slicing pattern for parallel optimize()"
    - "integer cache path format: f'{x}-{y}' (e.g. '0-25') — avoids float formatting"
    - "test injection hooks: tokenizer=/en_train=/es_train=/en_val=/es_val= kwargs bypass parquet reads in tests"

key-files:
  created:
    - safemoe/__init__.py
    - safemoe/data/__init__.py
    - safemoe/data/prepare.py
    - tests/safemoe/data/test_prepare.py
  modified: []

key-decisions:
  - "Used start_method='fork' in litdata.optimize() — spawn silently fails with in-memory tokenizer objects; fork preserves process state"
  - "Removed tests/safemoe/__init__.py and tests/safemoe/data/__init__.py — pytest namespace collision: test package 'safemoe' shadowed source package 'safemoe'"
  - "Added test injection kwargs (tokenizer, en_train, es_train, en_val, es_val) to prepare() — avoids loading real Qwen3 tokenizer and parquet files in tests"
  - "TokensLoader(block_size=4) required in StreamingDataset during tests — TokensLoader() without block_size causes TypeError in litdata ROI generation"

patterns-established:
  - "TDD pattern: RED commit (ImportError) then GREEN commit (passing implementation)"
  - "prepare() injection hooks pattern: optional kwargs override defaults for testability"
  - "litdata fork pattern: use start_method='fork' when tokenizer is not pickle-safe from spawn context"

requirements-completed: [DATA-01]

# Metrics
duration: 28min
completed: 2026-03-15
---

# Phase 1 Plan 01: Data Preparation Pipeline Summary

**Two-param split formula (compute_splits) + LitData tokenization pipeline (prepare) with Qwen3 BPE, integer-keyed cache paths, idempotent fork-based optimize()**

## Performance

- **Duration:** 28 min
- **Started:** 2026-03-15T14:37:30Z
- **Completed:** 2026-03-15T15:06:15Z
- **Tasks:** 2 (TDD: RED + GREEN)
- **Files modified:** 4 created, 0 modified from existing

## Accomplishments

- `safemoe/` and `safemoe/data/` packages created and importable alongside `litgpt/`
- `compute_splits(en, es, x, y)` implements the two-param formula exactly: D_std=EN[:y%], D_harmful=ES[:(100-x)%], D_unlabeled=remainder EN+ES
- `prepare()` tokenizes bilingual TinyStories parquet files and writes LitData streaming chunks to `data/.cache/{tokenizer_name}/{x}-{y}/D_{split}/{train|val}/`
- All 7 pytest tests pass: 4 unit tests for compute_splits(), 3 integration tests for prepare() with litdata

## Task Commits

Each task was committed atomically:

1. **Task 1: Package scaffolding + RED test stubs** - `47816ea` (test)
2. **Task 2: Implement compute_splits() and prepare.py CLI** - `01fa01b` (feat)

_Note: TDD tasks have two commits (test RED → feat GREEN)_

## Files Created/Modified

- `safemoe/__init__.py` — Package marker for safemoe top-level package
- `safemoe/data/__init__.py` — Package marker for safemoe.data subpackage
- `safemoe/data/prepare.py` — Core module: compute_splits(), tokenize_stories(), _maybe_optimize(), prepare(), argparse CLI
- `tests/safemoe/data/test_prepare.py` — 7 pytest tests covering split proportions (4) and litdata integration (3)

## Key Function Signatures

```python
def compute_splits(
    en_stories: list[str],
    es_stories: list[str],
    x: int = 0,
    y: int = 25,
) -> dict[str, list[str]]:
    # Returns {"D_std": ..., "D_harmful": ..., "D_unlabeled": ...}
    # D_std      = en_stories[:int(y/100 * n_en)]
    # D_harmful  = es_stories[:int((100-x)/100 * n_es)]
    # D_unlabeled = en_stories[int(y/100*n_en):] + es_stories[int((100-x)/100*n_es):]

def prepare(
    checkpoint_dir: Path = Path("checkpoints/Qwen3-30B-A3B-Base"),
    data_dir: Path = Path("data/multilingual-tinystories"),
    cache_dir: Path = Path("data/.cache"),
    x: int = 0,
    y: int = 25,
    num_workers: int = 4,
    chunk_bytes: str = "200MB",
    # test injection kwargs:
    tokenizer=None, en_train=None, es_train=None, en_val=None, es_val=None,
) -> None: ...
```

**Cache path format:** `cache_dir/{tokenizer.model_name}/{x}-{y}/D_{split}/{train|val}/`
Example: `data/.cache/Qwen3-30B-A3B-Base/0-25/D_std/train/`

**Tokenizer:** `Tokenizer("checkpoints/Qwen3-30B-A3B-Base")` — 151,643-vocab Qwen3 BPE; `model_name = "Qwen3-30B-A3B-Base"`

## Decisions Made

- `start_method='fork'` in `_maybe_optimize()`: litdata's default spawn cannot pickle in-memory tokenizer objects — silent failure creates empty index.json. Fork preserves the process state, making any tokenizer object usable.
- Removed `tests/safemoe/__init__.py` and `tests/safemoe/data/__init__.py`: pytest import mode creates a namespace package named `safemoe` from the test directory, shadowing the source `safemoe` package. Without `__init__.py`, pytest uses rootdir-based import and resolves `safemoe` to the source package correctly.
- Added `tokenizer`, `en_train`, `es_train`, `en_val`, `es_val` kwargs to `prepare()` for test injection: avoids loading the 1.4GB Qwen3 checkpoint and 8M-row parquet files during unit tests.
- `TokensLoader(block_size=4)` in `StreamingDataset` construction: `TokensLoader()` without `block_size` sets `_block_size=None`, which causes `chunk["dim"] // None` TypeError in litdata's ROI generation.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] litdata spawn-based worker silently fails for in-memory tokenizers**
- **Found during:** Task 2 (implement prepare())
- **Issue:** `litdata.optimize()` defaults to `start_method="spawn"`. Spawned workers re-import the main module via `python -c`, which cannot unpickle FakeTokenizer (defined in test scope, not a importable module). Result: workers run and exit with no output, `index.json` has 0 chunks.
- **Fix:** Added `start_method="fork"` parameter to `_maybe_optimize()` and passed it to `optimize()`. Fork inherits the parent process state, making any in-memory tokenizer picklable.
- **Files modified:** `safemoe/data/prepare.py`
- **Verification:** `test_litdata_output_readable` passes — `StreamingDataset` opens output and returns a tensor
- **Committed in:** `01fa01b` (Task 2 commit)

**2. [Rule 3 - Blocking] pytest namespace collision: tests/safemoe/__init__.py shadows source package**
- **Found during:** Task 2 (running tests after implementation)
- **Issue:** pytest adds `tests/` to `sys.path` when `tests/safemoe/__init__.py` exists, making `safemoe` resolve to `tests/safemoe/` (the test package) instead of the source `safemoe/`. `from safemoe.data.prepare import ...` therefore fails with `ModuleNotFoundError`.
- **Fix:** Removed `tests/safemoe/__init__.py` and `tests/safemoe/data/__init__.py`. Pytest uses rootdir-based import when no `__init__.py` is present, correctly resolving `safemoe` to the project root.
- **Files modified:** deleted `tests/safemoe/__init__.py`, `tests/safemoe/data/__init__.py`
- **Verification:** All 7 tests importable and passing
- **Committed in:** `01fa01b` (Task 2 commit)

**3. [Rule 1 - Bug] TokensLoader block_size=None causes TypeError in StreamingDataset**
- **Found during:** Task 2 (test_litdata_output_readable)
- **Issue:** `StreamingDataset(..., item_loader=TokensLoader())` crashes with `TypeError: unsupported operand type(s) for //: 'int' and 'NoneType'` when `_block_size` is None and litdata tries to compute region-of-interest.
- **Fix:** Changed test to use `TokensLoader(block_size=4)` in the StreamingDataset constructor.
- **Files modified:** `tests/safemoe/data/test_prepare.py`
- **Verification:** `test_litdata_output_readable` passes
- **Committed in:** `01fa01b` (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (2 Rule 3 blocking, 1 Rule 1 bug)
**Impact on plan:** All fixes necessary for the tests to discover and run correctly. No scope creep. Production `prepare()` function implementation matches plan spec exactly.

## Issues Encountered

- litdata 0.2.59 was not pre-installed; installed via `uv sync --extra extra`. pytest also not pre-installed in the `extra` group; `uv sync --extra extra --extra test` added.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `compute_splits()` is the canonical split function for Phase 2 (MultiDataLoader) and Phase 3 (training loop)
- `prepare()` CLI is ready to be run once against `data/multilingual-tinystories/` to produce the on-disk LitData cache
- Cache path format `{x}-{y}` is locked and confirmed; downstream phases should use this exact format
- Note: `start_method='fork'` is Linux-only; macOS would need `spawn`. Current environment is Linux only.

---
*Phase: 01-data-pipeline*
*Completed: 2026-03-15*
