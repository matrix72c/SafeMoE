---
phase: 04-ablation-evaluation
plan: "04"
subsystem: training
tags: [pytorch, lightning-fabric, ablation, evaluation, pretrain, masking]

# Dependency graph
requires:
  - phase: 04-ablation-evaluation
    plan: "01"
    provides: "HarmfulParamRegistry.parameters_by_type() and val_dataloaders() API"
  - phase: 03-sgtm-training-loop
    provides: "fit() training loop and validate() function in safemoe/pretrain.py"
provides:
  - evaluate_with_ablation() function in safemoe/pretrain.py
  - fit() extended with registry/val_loaders optional params
  - main() wired to call data.val_dataloaders() and pass to fit()
affects:
  - 04-05
  - any future training run configurations

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "try/finally guard for parameter restore — zero harmful params, run eval, always restore even on exception"
    - "model.train() in finally block to own eval-mode cleanup (validate() sets eval but caller restores)"
    - "fabric.log_dict(metrics, step=iter_num) for per-checkpoint ablation metrics"

key-files:
  created: []
  modified:
    - safemoe/pretrain.py
    - tests/safemoe/test_pretrain.py

key-decisions:
  - "evaluate_with_ablation() placed before validate() in pretrain.py so it can call validate() without forward reference"
  - "val_loaders dict passed to fit() as Optional param — None means ablation eval is skipped (backward compatible)"
  - "_MockMultiDataLoader.val_dataloaders() added to test stub to match real MultiDataLoader API called by main()"
  - "model.train() in finally is redundant with validate()'s own model.train() but serves as explicit ownership declaration"

patterns-established:
  - "Ablation eval pattern: clone params -> zero -> validate per split -> restore in finally -> log"
  - "EVAL-03 metrics: ablated_val_ppl_D_std and ablated_val_ppl_D_harmful only — no D_unlabeled (locked contract)"

requirements-completed: [EVAL-03]

# Metrics
duration: 20min
completed: 2026-03-16
---

# Phase 04 Plan 04: evaluate_with_ablation() Summary

**try/finally ablation eval that zeros theta_harmful at each save_interval checkpoint, logs ablated_val_ppl for D_std and D_harmful splits, then restores weights with guaranteed recovery on exception**

## Performance

- **Duration:** 20 min
- **Started:** 2026-03-16T14:48:01Z
- **Completed:** 2026-03-16T15:07:55Z
- **Tasks:** 1 (TDD: RED + GREEN commits)
- **Files modified:** 2

## Accomplishments

- Implemented `evaluate_with_ablation()` with try/finally guard ensuring theta_harmful weights are always restored, even if validate() raises mid-call
- Extended `fit()` signature with optional `registry` and `val_loaders` params, called at save_interval block after `save_checkpoint()`
- Wired `main()` to extract `val_loaders_for_eval = data.val_dataloaders()` and pass to `fit()`
- All 9 `test_pretrain.py` tests pass GREEN; full 45-test suite passes

## Task Commits

TDD cycle:

1. **RED — failing EVAL-03 tests** - `a86e8ac` (test)
2. **GREEN — evaluate_with_ablation() + fit() extension + main() wiring** - `5bf1d3a` (feat)

**Plan metadata:** (docs commit follows)

_Note: TDD task with RED commit (a86e8ac) then GREEN commit (5bf1d3a)_

## Files Created/Modified

- `/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/safemoe/pretrain.py` - Added `evaluate_with_ablation()` function, extended `fit()` signature, wired `main()` to pass registry/val_loaders
- `/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/tests/safemoe/test_pretrain.py` - Added `test_evaluate_with_ablation_restores_weights`, `test_evaluate_with_ablation_logs_metrics`, `_build_val_loaders()` helper, `val_dataloaders()` to `_MockMultiDataLoader`

## Decisions Made

- `evaluate_with_ablation()` placed before `validate()` in `pretrain.py` so it can call `validate()` without a forward reference
- `val_loaders` passed as `Optional[dict]` to `fit()` — `None` means ablation eval is silently skipped, preserving backward compatibility with callers that don't pass it
- `model.train()` in finally is technically redundant with `validate()`'s own `model.train()` call, but is retained as explicit ownership declaration — this function owns the eval-mode lifecycle
- `_MockMultiDataLoader.val_dataloaders()` added as Rule 1 auto-fix since `main()` now calls it and the test mock was missing the method

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Added val_dataloaders() to _MockMultiDataLoader test stub**
- **Found during:** Task 1 GREEN phase (running full test suite)
- **Issue:** `test_pretrain_produces_checkpoint` failed with `AttributeError: '_MockMultiDataLoader' object has no attribute 'val_dataloaders'` because `main()` now calls `data.val_dataloaders()` but the test mock only had `val_dataloader()`
- **Fix:** Added `val_dataloaders()` method to `_MockMultiDataLoader` returning `{"D_std": DataLoader, "D_harmful": DataLoader}`, matching the real `MultiDataLoader` API
- **Files modified:** `tests/safemoe/test_pretrain.py`
- **Verification:** `test_pretrain_produces_checkpoint` passes GREEN; full 45-test suite passes
- **Committed in:** `5bf1d3a` (GREEN implementation commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug in test mock)
**Impact on plan:** Necessary correctness fix — plan's action section required wiring `main()` to call `data.val_dataloaders()`, which broke the existing mock that didn't have that method.

## Issues Encountered

None — plan executed cleanly after the mock auto-fix.

## Next Phase Readiness

- `evaluate_with_ablation()` is importable from `safemoe.pretrain` and fully tested
- EVAL-03 requirement satisfied — mid-training ablation evaluation tracks isolation progress at each checkpoint
- Ready for Phase 04-05 (integration/end-to-end evaluation plans) if any remain

---
*Phase: 04-ablation-evaluation*
*Completed: 2026-03-16*
