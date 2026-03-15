---
phase: 01-data-pipeline
verified: 2026-03-15T00:00:00Z
status: passed
score: 7/7 must-haves verified
re_verification: false
gaps: []
human_verification:
  - test: "Run python -m safemoe.data.prepare --x 0 --y 25 --num_workers 4 against real data/multilingual-tinystories/ parquets"
    expected: "Directories data/.cache/Qwen3-30B-A3B-Base/0-25/D_std/train/, D_harmful/train/, D_std/val/, D_harmful/val/, D_unlabeled/train/ are created and openable by StreamingDataset"
    why_human: "Real parquet files (2.1M row EN/ES) required; Qwen3-30B-A3B-Base checkpoint must be present; cannot run in automated verification without the data on disk"
  - test: "Run prepare twice with same x/y and confirm second run is fast (skips re-tokenization)"
    expected: "Second invocation completes almost instantly with no new writes to the cache dirs"
    why_human: "Idempotency is verified by test_idempotent at unit level with synthetic data; end-to-end behavior with real parquets requires a human smoke-test"
---

# Phase 1: Data Pipeline Verification Report

**Phase Goal:** A tokenized TinyStories bilingual dataset partitioned into D_std/D_harmful/D_unlabeled splits with a two-parameter (x, y) sweep scheme, served through per-split DataLoaders via MultiDataLoader.get_loader()
**Verified:** 2026-03-15
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

From ROADMAP.md Success Criteria and plan must_haves (all three success criteria are directly tested):

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Given x=0, y=25: D_std has 25% EN rows, D_harmful has 100% ES rows, D_unlabeled has 75% EN rows | VERIFIED | `test_split_proportions` passes; formula `int(y / 100.0 * n_en)` confirmed in prepare.py:58 |
| 2 | Given x=50, y=25: D_harmful has 50% ES rows, D_unlabeled has 75% EN + 50% ES rows | VERIFIED | `test_split_proportions_x50` passes; boundary content verified (ES rows do not appear in first 75 of D_unlabeled) |
| 3 | Given x=0, y=50: D_std has 50% EN rows, D_unlabeled has 50% EN rows | VERIFIED | `test_split_y_param` passes |
| 4 | prepare.py writes tokenized LitData chunks to integer-keyed cache dirs | VERIFIED | `test_litdata_output_readable` and `test_cache_path_format` both pass; path uses `f"{x}-{y}"` at prepare.py:231 |
| 5 | prepare.py is idempotent: second run skips existing output dirs | VERIFIED | `test_idempotent` passes; `_maybe_optimize` returns early at prepare.py:126 if dir exists |
| 6 | MultiDataLoader.get_loader(split_name) returns an iterable StreamingDataLoader for all three splits | VERIFIED | `test_get_loader_returns_dataloader` and `test_get_loader_all_splits` pass; setup() builds _loaders dict at datamodule.py:62-66 |
| 7 | val_dataloaders() returns {"D_std": DataLoader, "D_harmful": DataLoader} with no D_unlabeled key | VERIFIED | `test_val_dataloaders_keys` passes; return dict at datamodule.py:95-98 contains exactly two keys |

**Score:** 7/7 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `safemoe/__init__.py` | package marker | VERIFIED | Exists; importable |
| `safemoe/data/__init__.py` | package marker | VERIFIED | Exists; importable |
| `safemoe/data/prepare.py` | tokenization + partitioning CLI; exports compute_splits(), prepare() | VERIFIED | 309 lines; both functions exported and tested; CLI via argparse at line 252 |
| `safemoe/data/datamodule.py` | MultiDataLoader DataModule subclass | VERIFIED | 107 lines; exports MultiDataLoader; contains get_loader, val_dataloaders, train_dataloader, val_dataloader, connect, setup |
| `tests/safemoe/data/test_prepare.py` | 7 pytest tests for DATA-01 | VERIFIED | Contains test_split_proportions, test_split_proportions_x50, test_split_y_param, test_split_boundary_x100, test_litdata_output_readable, test_cache_path_format, test_idempotent; all 7 pass |
| `tests/safemoe/data/test_datamodule.py` | 7 pytest tests for DATA-02 / DATA-03 | VERIFIED | Contains test_get_loader_returns_dataloader, test_val_dataloaders_keys, test_get_loader_all_splits, test_no_next_method, test_no_upsample_fields, test_train_dataloader_compat, test_cache_path_uses_integer_format; all 7 pass |

Note: `tests/safemoe/__init__.py` and `tests/safemoe/data/__init__.py` were intentionally removed (documented in 01-01-SUMMARY.md) to prevent pytest namespace collision that caused `safemoe` source package shadowing. This is the correct behavior.

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `prepare.py::compute_splits` | D_std / D_harmful / D_unlabeled story lists | `int(y / 100.0 * n_en)` formula | WIRED | Pattern confirmed at prepare.py:58-63; formula exactly matches spec |
| `prepare.py::prepare` | `data/.cache/{tokenizer_name}/{x}-{y}/D_std/train/` | `f"{x}-{y}"` integer path format | WIRED | `out_base = Path(cache_dir) / tokenizer.model_name / f"{x}-{y}"` at prepare.py:231; integer cast at line 207 |
| `datamodule.py::get_loader` | StreamingDataLoader wrapping StreamingDataset | `cache_dir / tokenizer.model_name / f'{x}-{y}' / split_name / 'train'` | WIRED | base path at datamodule.py:46; StreamingDataset + StreamingDataLoader constructed at lines 49-60 |
| `datamodule.py::val_dataloaders` | {"D_std": DataLoader, "D_harmful": DataLoader} | StreamingDataset with shuffle=False pointing to /val/ subdirs | WIRED | shuffle=False at datamodule.py:85; returns exactly two-key dict at lines 95-98 |

---

### Requirements Coverage

| Requirement | Source Plans | Description (from REQUIREMENTS.md) | Phase 1 Status | Evidence / Notes |
|-------------|-------------|-------------------------------------|----------------|-----------------|
| DATA-01 | 01-01-PLAN.md | Data preparation script: tokenization + three-split partitioning (D_std, D_harmful, D_unlabeled) with configurable x/y sweep parameters | SATISFIED | compute_splits() implements two-param formula; prepare() writes LitData chunks; 7 tests green. **Documented override:** CONTEXT.md records decision to use Qwen3 BPE tokenizer instead of tiktoken gpt2 specified in REQUIREMENTS.md. Tokenizer choice does not change the DATA-01 interface or behavior contract. |
| DATA-02 | 01-02-PLAN.md | MultiDataLoader with per-split DataLoaders (D_std/D_harmful/D_unlabeled) consistent with LitGPT DataModule | SATISFIED (Phase 1 obligations) | get_loader(), val_dataloaders(), train_dataloader(), val_dataloader() all implemented and tested. **Scoped delegation:** REQUIREMENTS.md text mentions "configurable upsample factors (upsample_std, upsample_harmful, upsample_unlabeled)". CONTEXT.md locked design explicitly assigns upsample weights to the Phase 3 training loop, not MultiDataLoader. This delegation is intentional and correct per paper design; test_no_upsample_fields enforces the boundary. |
| DATA-03 | 01-02-PLAN.md | Pre-generated data_split_order list (shuffled schedule) controls training mix | SATISFIED (Phase 1 obligations) | **Scoped delegation:** REQUIREMENTS.md describes a pre-generated split order list. CONTEXT.md locked design explicitly replaced this with dynamic sampling (`random.choices()`) in the Phase 3 training loop. MultiDataLoader is a loader registry only; it does not generate or store a split schedule. The ROADMAP.md Phase 1 Success Criterion #3 explicitly states "Dynamic split sampling lives in Phase 3 training loop, not in MultiDataLoader." test_no_next_method and test_no_upsample_fields enforce this boundary. |

#### Orphaned Requirements for Phase 1

No orphaned requirements found. REQUIREMENTS.md Traceability table maps DATA-01, DATA-02, DATA-03 to Phase 1 — all three are claimed in plan frontmatter and accounted for above.

#### Requirement Text vs. Implementation Delta (DATA-02 and DATA-03)

The REQUIREMENTS.md text for DATA-02 and DATA-03 was written before the CONTEXT.md design session that locked the interface. The CONTEXT.md (updated 2026-03-15) supersedes the original requirement text for Phase 1 scope:

- Upsample factors: deferred to Phase 3 (TRAIN-01/TRAIN-02 scope)
- Pre-generated split_order: replaced by dynamic `random.choices()` in Phase 3 training loop

The ROADMAP.md Phase 1 Success Criteria (the operative contract for this verification) correctly reflects the post-context design and is fully satisfied.

---

### Anti-Patterns Found

Scan of all phase-1 created files (`safemoe/data/prepare.py`, `safemoe/data/datamodule.py`, `tests/safemoe/data/test_prepare.py`, `tests/safemoe/data/test_datamodule.py`):

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | - | - | - |

No TODO, FIXME, HACK, placeholder, empty implementation, or stub patterns found in any phase-created file.

---

### Human Verification Required

#### 1. End-to-End Tokenization Run on Real Data

**Test:** From the project root, run `python -m safemoe.data.prepare --x 0 --y 25 --num_workers 4` (requires `data/multilingual-tinystories/` parquets and `checkpoints/Qwen3-30B-A3B-Base/` tokenizer on disk).
**Expected:** Five directories created under `data/.cache/Qwen3-30B-A3B-Base/0-25/`: D_std/train/, D_std/val/, D_harmful/train/, D_harmful/val/, D_unlabeled/train/. Each directory contains LitData binary chunks openable by `StreamingDataset`.
**Why human:** Real 2.1M-row parquet files and Qwen3 checkpoint required. The automated tests use synthetic 20-story lists with a FakeTokenizer; real data volume and tokenizer behavior cannot be covered without a human-triggered run.

#### 2. Idempotency Smoke Test on Real Data

**Test:** Run the prepare CLI twice with the same x/y. Observe that the second invocation finishes immediately.
**Expected:** No re-tokenization occurs; the five output directories are untouched.
**Why human:** The unit test (`test_idempotent`) validates the `_maybe_optimize` logic with synthetic data; real-world confirmation with the full parquet corpus confirms there are no race conditions or edge cases in the skip logic.

---

### Gaps Summary

No gaps. All 7 observable truths are verified, all 6 artifacts exist and are substantive, all 4 key links are wired, all 14 tests pass, and no anti-patterns were found.

The DATA-02/DATA-03 requirement text contains features (upsample weights, pre-generated split order) that were deliberately scoped out of Phase 1 by the CONTEXT.md locked design decision. These features are assigned to Phase 3 (TRAIN-01/TRAIN-02). This is not a gap for Phase 1 verification.

---

## Test Execution Record

```
pytest tests/safemoe/data/ -v
14 passed, 51 warnings in 49.83s

tests/safemoe/data/test_datamodule.py::test_get_loader_returns_dataloader   PASSED
tests/safemoe/data/test_datamodule.py::test_get_loader_all_splits            PASSED
tests/safemoe/data/test_datamodule.py::test_val_dataloaders_keys             PASSED
tests/safemoe/data/test_datamodule.py::test_no_next_method                   PASSED
tests/safemoe/data/test_datamodule.py::test_no_upsample_fields               PASSED
tests/safemoe/data/test_datamodule.py::test_train_dataloader_compat          PASSED
tests/safemoe/data/test_datamodule.py::test_cache_path_uses_integer_format   PASSED
tests/safemoe/data/test_prepare.py::test_split_proportions                   PASSED
tests/safemoe/data/test_prepare.py::test_split_proportions_x50               PASSED
tests/safemoe/data/test_prepare.py::test_split_y_param                       PASSED
tests/safemoe/data/test_prepare.py::test_split_boundary_x100                 PASSED
tests/safemoe/data/test_prepare.py::test_litdata_output_readable             PASSED
tests/safemoe/data/test_prepare.py::test_cache_path_format                   PASSED
tests/safemoe/data/test_prepare.py::test_idempotent                          PASSED
```

---

## Commit Record

| Commit | Type | Description |
|--------|------|-------------|
| `47816ea` | test | Package scaffolding + RED test stubs for DATA-01 |
| `01fa01b` | feat | compute_splits() and prepare() — GREEN state |
| `07ab7b8` | test | RED test stubs for MultiDataLoader (DATA-02/03) |
| `0e4fa3d` | feat | MultiDataLoader implementation — GREEN state |

All four commits confirmed present in git history.

---

_Verified: 2026-03-15_
_Verifier: Claude (gsd-verifier)_
