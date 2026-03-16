---
phase: 03-sgtm-training-loop
plan: "04"
subsystem: testing
tags: [pytest, unittest.mock, behavioral-tests, regression, TRAIN-01, TRAIN-02]

# Dependency graph
requires:
  - phase: 03-sgtm-training-loop
    provides: safemoe/pretrain.py fit() with 3-path SGTM loop, SafeCausalSelfAttention, maskers
provides:
  - Behavioral regression tests for all four TRAIN-01/02 loop invariants
  - REQUIREMENTS.md TRAIN-02 description aligned with actual implementation
  - Autograd-safe SafeCausalSelfAttention.forward() head-zeroing (clone before in-place op)
affects: [04-evaluation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Use _setup_fit_test() helper to build fabric+state for fit() direct-call tests"
    - "Module-level _SynthDataset and _MockMultiDataLoader for sharing across all tests"
    - "patch.object(obj, method, wraps=obj.method) pattern for behavioral spying in tests"
    - "Clone before in-place zeroing on SDPA output to preserve autograd version"

key-files:
  created: []
  modified:
    - tests/safemoe/test_pretrain.py
    - safemoe/pretrain.py
    - .planning/REQUIREMENTS.md

key-decisions:
  - "SafeCausalSelfAttention.forward() must clone y before in-place head-zeroing — SDPA output is part of autograd graph and in-place ops invalidate version tracking during backward"
  - "max_tokens=128 (block_size=128) gives exactly 1 optimizer step for global_batch_size=1,micro_batch_size=1 — max_iters=1 matches accum_iters=1"
  - "max_tokens=256 for accum_iters=2 test — max_iters=2 means 1 optimizer step consumes 2 micro-batches then loop exits"

patterns-established:
  - "Pattern: block_size=128 in TINY_CONFIG-derived test configs for fast forward/backward"
  - "Pattern: upsample weights (0,1,0)/(1,0,0)/(0,0,1) force deterministic split selection"

requirements-completed: [TRAIN-01, TRAIN-02]

# Metrics
duration: 25min
completed: 2026-03-16
---

# Phase 3 Plan 04: Gap Closure — Behavioral Tests + REQUIREMENTS.md Update Summary

**Four import-only test stubs replaced with behavioral assertions verifying D_harmful grad isolation, D_std activation masker lifecycle, D_unlabeled no-masking, and mask() call-count invariant; plus autograd-safe head-zeroing fix and TRAIN-02 description updated to match split_label/masker-dispatch implementation.**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-03-16T05:40:00Z
- **Completed:** 2026-03-16T06:05:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Upgraded 4 import-only test stubs to behavioral assertions that verify SGTM loop invariants
- Added module-level `_SynthDataset`, `_MockMultiDataLoader`, and `_setup_fit_test()` shared test infrastructure
- Fixed latent autograd bug in `SafeCausalSelfAttention.forward()`: in-place head-zeroing on SDPA output corrupted gradient version tracking during D_std backward pass
- Updated REQUIREMENTS.md TRAIN-02 description from superseded `sgtm_mode`/`adjust_gradients()` design to actual `split_label` + 3-path masker dispatch implementation

## Task Commits

Each task was committed atomically:

1. **Task 1: Upgrade four import-stub tests to behavioral assertions** - `d3c3d91` (feat)
2. **Task 2: Update REQUIREMENTS.md TRAIN-02 to match implemented interface** - `ad94e6d` (docs)

## Files Created/Modified
- `tests/safemoe/test_pretrain.py` - Replaced 4 import-only stubs with behavioral tests; added module-level shared classes; updated imports
- `safemoe/pretrain.py` - Cloned SDPA output before in-place head-zeroing to fix autograd version conflict
- `.planning/REQUIREMENTS.md` - Updated TRAIN-02 description to reference actual split_label/masker dispatch interface

## Decisions Made
- Used `block_size=128` in test config overrides to keep forward/backward fast (default 4096 would be prohibitively slow for unit tests)
- Computed `max_tokens` as `accum_iters * micro_batch_size * block_size` to get exactly 1 optimizer step: `128` for accum_iters=1, `256` for accum_iters=2
- Used `patch.object(obj, method, wraps=obj.method)` (spy pattern) for D_std test to verify call counts without breaking real enable/disable behavior
- Used `save_interval=None` and `EvalArgs(initial_validation=False, final_validation=False, interval=99999)` to suppress all checkpoint saves and validation overhead

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed in-place head-zeroing autograd version conflict in SafeCausalSelfAttention.forward()**
- **Found during:** Task 1 (test_fit_std_step_enables_activation_masker)
- **Issue:** `y[:, :, head_idx, :] = 0` in-place assignment on SDPA output caused `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation` during D_std backward pass — autograd tracks a version counter on tensors and the in-place op invalidated the saved version
- **Fix:** Added `y = y.clone()` before the loop that zeroes harmful head slices, creating a new tensor node in the graph so in-place ops on it do not invalidate the SDPA backward path
- **Files modified:** `safemoe/pretrain.py` (SafeCausalSelfAttention.forward, lines 162-165)
- **Verification:** All 7 test_pretrain tests pass; all 24 safemoe tests green
- **Committed in:** d3c3d91 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug)
**Impact on plan:** The bug was latent (only triggered during backward pass with activation masking enabled, which wasn't exercised by prior tests). Fix is minimal and correct — clone before in-place op is the standard autograd-safe pattern.

## Issues Encountered
- D_std behavioral test was the first test to exercise `fabric.backward()` with `activation_masker.enable()` active, revealing the SDPA in-place zeroing bug that was present in the production code since Plan 03-01/03-02.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 3 gap closure complete: all 4 TRAIN-01/02 loop invariants have automated behavioral regression tests
- All 24 safemoe tests GREEN with no regressions
- REQUIREMENTS.md TRAIN-02 accurately describes the implemented interface
- Phase 4 (evaluation/ablation) can proceed with full confidence in training loop correctness

---
*Phase: 03-sgtm-training-loop*
*Completed: 2026-03-16*

## Self-Check: PASSED

- FOUND: .planning/phases/03-sgtm-training-loop/03-04-SUMMARY.md
- FOUND: tests/safemoe/test_pretrain.py
- FOUND: safemoe/pretrain.py
- FOUND commit d3c3d91 (Task 1)
- FOUND commit ad94e6d (Task 2)

