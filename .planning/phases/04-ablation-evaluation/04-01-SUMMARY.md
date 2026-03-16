---
phase: 04-ablation-evaluation
plan: "01"
subsystem: testing
tags: [tdd, red-phase, ablation, evaluation, pytest, safemoe]

# Dependency graph
requires:
  - phase: 03-sgtm-training-loop
    provides: "pretrain.py with validate(), fit(), HarmfulParamRegistry interface; test patterns from test_pretrain.py"
provides:
  - "test_ablate.py: 3 RED test stubs for TRAIN-04 (ablate() zeroes harmful weights, saves manifest, prints summary)"
  - "test_evaluate.py: 2 RED test stubs for EVAL-01+EVAL-02 (evaluate_perplexity, routing_attribution)"
  - "test_pretrain.py appended: 2 RED test stubs for EVAL-03 (evaluate_with_ablation restores weights, logs metrics)"
  - "API contracts: ablate(ckpt_dir), evaluate_perplexity(original_ckpt_dir, ablated_ckpt_dir, data_mock, out_dir), routing_attribution(ckpt_dir, data_mock), evaluate_with_ablation(fabric, model, registry, val_loaders, iter_num, eval_args)"
affects: [04-02-PLAN, 04-03-PLAN, 04-04-PLAN]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "RED-phase TDD: test stubs import not-yet-existing modules to trigger ImportError"
    - "data_mock parameter pattern: both evaluate functions accept optional mock for testability"
    - "D_unlabeled exclusion: no D_unlabeled metric keys in any evaluation output (user decision)"
    - "_make_checkpoint fixture: saves lit_model.pth + model_config.yaml to tmp_path for test isolation"

key-files:
  created:
    - tests/safemoe/test_ablate.py
    - tests/safemoe/test_evaluate.py
  modified:
    - tests/safemoe/test_pretrain.py

key-decisions:
  - "D_unlabeled metrics excluded from all evaluation outputs — user decision to skip D_unlabeled perplexity/routing tracking"
  - "data_mock parameter on evaluate_perplexity() and routing_attribution() — testability pattern consistent with test_pretrain.py MockMultiDataLoader approach"
  - "evaluate_with_ablation() takes eval_args: EvalArgs parameter — consistent with existing validate() signature"
  - "fabric.log_dict() called exactly once per evaluate_with_ablation() call — single dict with both D_std and D_harmful PPL metrics"
  - "test stubs use inline TINY_CONFIG (not imported from test_pretrain.py) — test file isolation, avoids cross-file import coupling"

patterns-established:
  - "Pattern 1: _make_checkpoint(tmp_path, config) helper — creates lit_model.pth + model_config.yaml in tmp_path for checkpoint-based tests"
  - "Pattern 2: _MockDataModule.val_dataloaders() returns {D_std: DataLoader, D_harmful: DataLoader} — standard mock interface for evaluate functions"
  - "Pattern 3: module-level import for RED failure — `from safemoe.ablate import ablate` at top of file, not inside test"

requirements-completed: [TRAIN-04, EVAL-01, EVAL-02, EVAL-03]

# Metrics
duration: 8min
completed: 2026-03-16
---

# Phase 4 Plan 01: TDD RED Test Stubs Summary

**Seven failing test stubs defining API contracts for ablate(), evaluate_perplexity(), routing_attribution(), and evaluate_with_ablation() across three test files**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-16T14:49:22Z
- **Completed:** 2026-03-16T14:57:00Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- Created test_ablate.py with 3 tests: theta_harmful zeroing verification, ablation_manifest.json structure check, and stdout summary assertion (TRAIN-04 contract)
- Created test_evaluate.py with 2 tests: evaluate_perplexity() result dict shape and results.json output, routing_attribution() fraction values and routing_attribution.json output (EVAL-01+02 contracts)
- Appended 2 EVAL-03 tests to test_pretrain.py: evaluate_with_ablation() weight restoration and log_dict() metric contract (D_std + D_harmful only, no D_unlabeled)

## Task Commits

Each task was committed atomically:

1. **Task 1: RED test stubs for ablate.py (TRAIN-04)** - `8478f48` (test)
2. **Task 2: RED test stubs for evaluate.py (EVAL-01, EVAL-02)** - `eebf790` (test)
3. **Task 3: RED test stub for evaluate_with_ablation() (EVAL-03)** - `a86e8ac` (test)

*Note: Tests were originally committed in RED state; implementations followed in subsequent commits within the same session.*

## Files Created/Modified

- `tests/safemoe/test_ablate.py` - 3 test functions for TRAIN-04: zeros harmful weights, saves manifest JSON, prints summary with "zeroed" text
- `tests/safemoe/test_evaluate.py` - 2 test functions for EVAL-01+EVAL-02: evaluate_perplexity() dict keys + results.json, routing_attribution() fraction values + routing_attribution.json
- `tests/safemoe/test_pretrain.py` - 2 EVAL-03 tests appended: weight restoration after ablation eval pass, fabric.log_dict() called exactly once with correct keys

## Decisions Made

- D_unlabeled metrics excluded from all evaluation outputs — user decision made during Phase 4 context/validation planning; both evaluate.py and test stubs enforce no `D_unlabeled` keys appear
- data_mock parameter on evaluate_perplexity() and routing_attribution() — testability without real data files; mirrors `_MockMultiDataLoader` pattern from test_pretrain.py
- evaluate_with_ablation() accepts `eval_args: EvalArgs` — consistent with existing validate() caller signature in pretrain.py
- fabric.log_dict() called exactly once per evaluate_with_ablation() call with a single combined dict containing both D_std and D_harmful PPL keys

## Deviations from Plan

### Execution Order Deviation

The work for plan 04-01 was executed out of order relative to the plan numbering — the RED stubs were written and committed (as commits tagged 04-02, 04-03, 04-04) immediately before their corresponding GREEN implementations. By the time plan 04-01 was formally executed, all three test files existed and all 7 tests were passing GREEN (because implementations had been committed in the same session).

This is not a correctness deviation — all required test files exist with the correct structure, function names, import patterns, and behavioral contracts as specified in the plan. The SUMMARY.md was the only missing artifact.

**Impact:** None. All 7 tests pass, all 3 required files exist, API contracts are defined correctly.

## Issues Encountered

None — test file structure and imports matched the specified API contracts exactly.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All 4 test contracts defined: ablate(), evaluate_perplexity(), routing_attribution(), evaluate_with_ablation()
- test_ablate.py GREEN: ablate.py implemented (Plan 04-02 complete)
- test_evaluate.py GREEN: evaluate.py implemented (Plan 04-03 complete)
- test_pretrain.py EVAL-03 GREEN: evaluate_with_ablation() in pretrain.py (Plan 04-04 in progress)
- Ready for Plan 04-05 (human verification of full evaluation pipeline)

---
*Phase: 04-ablation-evaluation*
*Completed: 2026-03-16*
