---
phase: 01-data-pipeline
plan: 02
subsystem: data
tags: [litdata, streaming-dataloader, datamodule, lightning, multi-split]

# Dependency graph
requires:
  - phase: 01-data-pipeline/01-01
    provides: "LitData cache layout at cache_dir/{tokenizer_name}/{x}-{y}/{split}/{train|val}/"
provides:
  - "MultiDataLoader DataModule subclass with per-split DataLoader registry"
  - "get_loader(split_name) -> DataLoader — primary training loop interface"
  - "val_dataloaders() -> {'D_std': DataLoader, 'D_harmful': DataLoader} — evaluation interface"
  - "train_dataloader() and val_dataloader() — LightningDataModule compatibility methods"
affects: [03-training-loop, 04-evaluation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "DataModule subclass as dataclass: @dataclass + field(default=None, init=False) for non-constructor state"
    - "Deferred StreamingDataLoader construction in setup(): imports inside method to avoid circular deps"
    - "val_dataloaders() creates new loaders each call with shuffle=False — not cached like train loaders"
    - "Module-level named function for litdata.optimize() fn arg (picklable); lambda fails in spawn"
    - "start_method='fork' in test fixture optimize() calls — inherits parent state, avoids pickle errors"

key-files:
  created:
    - safemoe/data/datamodule.py
    - tests/safemoe/data/test_datamodule.py
  modified: []

key-decisions:
  - "get_loader() returns the stored DataLoader directly; training loop manages its own iter() — no next() on MultiDataLoader"
  - "val_dataloaders() is a separate method from val_dataloader(): val_dataloaders() returns a named dict, val_dataloader() returns list for Lightning compat"
  - "Module-level _tokenize_row() function in tests instead of lambda: litdata spawn workers cannot pickle locally-defined closures"

patterns-established:
  - "TDD fork pattern: module-level picklable fn + start_method='fork' in test litdata fixtures"
  - "MultiDataLoader setup/connect pattern: connect() first (sets tokenizer + seq_length), then setup() (builds StreamingDataLoaders)"

requirements-completed: [DATA-02, DATA-03]

# Metrics
duration: 6min
completed: 2026-03-15
---

# Phase 1 Plan 02: MultiDataLoader Summary

**Per-split StreamingDataLoader registry (MultiDataLoader) with get_loader(), val_dataloaders(), and LightningDataModule compatibility for SGTM three-stream training**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-15T15:11:34Z
- **Completed:** 2026-03-15T15:17:00Z
- **Tasks:** 2 (TDD: RED + GREEN)
- **Files modified:** 2 created, 0 modified from existing

## Accomplishments

- `MultiDataLoader` DataModule subclass implements the locked interface from 01-CONTEXT.md exactly
- `get_loader('D_std' | 'D_harmful' | 'D_unlabeled')` returns a `StreamingDataLoader` — training loop manages its own `iter()`
- `val_dataloaders()` returns `{"D_std": DataLoader, "D_harmful": DataLoader}` — no D_unlabeled key (no val set per paper)
- `train_dataloader()` and `val_dataloader()` present for LightningDataModule compatibility
- Cache path uses integer `f"{int(x)}-{int(y)}"` format (e.g. `"0-25"`)
- All 14 tests pass: 7 new datamodule tests + 7 existing prepare tests

## Task Commits

Each task was committed atomically:

1. **Task 1: Write RED test stubs for MultiDataLoader** - `07ab7b8` (test)
2. **Task 2: Implement MultiDataLoader (GREEN)** - `0e4fa3d` (feat)

_Note: TDD tasks have two commits (test RED → feat GREEN)_

## Files Created/Modified

- `safemoe/data/datamodule.py` — MultiDataLoader class: DataModule subclass with get_loader(), val_dataloaders(), train_dataloader(), val_dataloader(), connect(), setup()
- `tests/safemoe/data/test_datamodule.py` — 7 pytest tests for DATA-02 and DATA-03 interface coverage

## Key Interface

```python
@dataclass
class MultiDataLoader(DataModule):
    cache_dir: Path = Path("data/.cache")
    x: int = 0        # ES leak param (integer, matches cache dir name)
    y: int = 25       # EN retention param (integer)
    seed: int = 42
    num_workers: int = 4

    def connect(self, tokenizer=None, batch_size=1, max_seq_length=-1) -> None:
        # Sets tokenizer, batch_size, max_seq_length (+1 for next-token target)

    def setup(self, stage: str = "") -> None:
        # Builds _loaders dict with StreamingDataLoader for each of 3 train splits

    def get_loader(self, split_name: str) -> StreamingDataLoader:
        # split_name: 'D_std' | 'D_harmful' | 'D_unlabeled'
        # Training loop: it = iter(multi_loader.get_loader('D_std'))

    def val_dataloaders(self) -> dict:
        # Returns {"D_std": DataLoader, "D_harmful": DataLoader}
        # NO "D_unlabeled" key

    def train_dataloader(self) -> StreamingDataLoader:  # Lightning compat: D_std loader
    def val_dataloader(self):                           # Lightning compat: list of val loaders
```

**Cache path construction:** `cache_dir / tokenizer.model_name / f"{int(x)}-{int(y)}" / split_name / "train"`
Example: `data/.cache/Qwen3-30B-A3B-Base/0-25/D_std/train/`

## Decisions Made

- `get_loader()` returns the DataLoader directly; no `next()` method on `MultiDataLoader`. Training loop manages its own iterators via `it = iter(multi_loader.get_loader('D_std'))`. This matches the CONTEXT.md locked interface and keeps MultiDataLoader free of state that belongs in the training loop.
- `val_dataloaders()` (plural) creates fresh `StreamingDataLoader` instances each call with `shuffle=False`. It is a separate method from `val_dataloader()` (singular, Lightning compat) which returns the list.
- Module-level `_tokenize_row()` function in test file replaces the plan's suggested `lambda`: litdata spawn workers serialize the `fn` argument via pickle, which fails for locally-defined closures. Named module-level functions are picklable. Also added `start_method='fork'` as a belt-and-suspenders safety measure (same fix as established in plan 01-01).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Lambda in test fixture not picklable by litdata spawn workers**
- **Found during:** Task 2 (running tests after implementation)
- **Issue:** The plan's suggested `fn=lambda data: (torch.tensor(row) for row in data)` in `make_fake_split()` fails with `AttributeError: Can't get local object 'make_fake_split.<locals>.<lambda>'` when litdata's `optimize()` tries to spawn worker processes that serialize the function via pickle.
- **Fix:** Replaced the lambda with a module-level named function `_tokenize_row(data)` (picklable) and added `start_method="fork"` to the `optimize()` call in the test helper.
- **Files modified:** `tests/safemoe/data/test_datamodule.py`
- **Verification:** All 7 datamodule tests pass; `14 passed` in the full suite.
- **Committed in:** `0e4fa3d` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 bug)
**Impact on plan:** Fix required for test fixtures to work. No scope creep. Implementation matches plan spec exactly.

## Issues Encountered

None beyond the lambda pickle issue documented above.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `MultiDataLoader` is the canonical data interface for Phase 3 (SGTM training loop)
- Phase 3 training loop: call `mdl.connect(tokenizer=..., batch_size=..., max_seq_length=...)` then `mdl.setup()`, then use `iter(mdl.get_loader('D_std'))` etc. per step
- `val_dataloaders()` is the evaluation interface for Phase 4
- Integration note: `MultiDataLoader` does NOT hold upsample weights — those belong in Phase 3 training loop per paper design
- Tested with `num_workers=0` in tests (avoids spawning multiple worker processes in test env); production use should set `num_workers=4` or more

---
*Phase: 01-data-pipeline*
*Completed: 2026-03-15*
