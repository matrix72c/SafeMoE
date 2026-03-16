---
phase: 03-sgtm-training-loop
plan: "01"
subsystem: testing, masking
tags: [tdd, pytorch, gradient-masking, activation-masking, attention-heads, qkv, CausalSelfAttention]

# Dependency graph
requires:
  - phase: 02-model-architecture-masking
    provides: HarmfulParamRegistry._qkv_harmful_metadata, GradientMasker, ActivationMasker, SafeMoELayer
provides:
  - "tests/safemoe/test_pretrain.py with 7 test stubs (2 GREEN, 5 RED)"
  - "GradientMasker._qkv_param_ids + two-pass mask() for per-head row zeroing"
  - "ActivationMasker._attn_layers + config kwarg for CausalSelfAttention flag infrastructure"
  - "_wrap_attn_forward() stub for Phase 03-02 head-output zeroing"
affects: [03-02-PLAN, 03-03-PLAN]

# Tech tracking
tech-stack:
  added: [types (stdlib)]
  patterns:
    - "Two-pass gradient masker: Pass 1 null-out non-qkv theta_std; Pass 2 zero harmful-head qkv row slices"
    - "Monkey-patch infrastructure: _wrap_attn_forward installs a pass-through forward wrapper on CausalSelfAttention instances at ActivationMasker construction time"
    - "id()-set for fast qkv param identity checks (_qkv_param_ids)"

key-files:
  created:
    - tests/safemoe/test_pretrain.py
  modified:
    - safemoe/masking.py

key-decisions:
  - "test_attn_head_activation_masking verifies flag-state only (not head output zeroing) — actual zeroing assertion deferred to Plan 03-03 Task 2 to avoid depending on 03-02 implementation"
  - "_wrap_attn_forward installs a pass-through stub in 03-01; Plan 03-02 replaces with real zeroing via SafeCausalSelfAttention or complete monkey-patch"
  - "ActivationMasker(model) still works with no config kwarg — backward-compatible; _attn_layers is empty list when config is None or harmful_attn_heads is []"

patterns-established:
  - "TDD RED/GREEN split across plans: test stubs in plan N, implementation in plan N+1, fuller assertions in plan N+2"
  - "Two-pass GradientMasker.mask(): separate null-out sweep and row-slice zeroing sweep for qkv params"

requirements-completed: [TRAIN-01, TRAIN-02]

# Metrics
duration: 3min
completed: 2026-03-16
---

# Phase 3 Plan 01: Phase 3 RED Test Stubs + Masker Attn-Head Extension Summary

**Two-pass GradientMasker and ActivationMasker extended for attn-head masking with 7 RED/GREEN test stubs in test_pretrain.py**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-16T05:08:36Z
- **Completed:** 2026-03-16T05:11:56Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Created tests/safemoe/test_pretrain.py with 7 test functions covering TRAIN-01/02/03 behaviors; 5 RED stubs (pretrain.py-dependent), 2 proper unit tests
- Extended GradientMasker with `_qkv_param_ids` and two-pass `mask()` that preserves standard-head rows while zeroing harmful-head rows of qkv.weight.grad
- Extended ActivationMasker with `config` kwarg, `_attn_layers` list, CausalSelfAttention flag infrastructure, and `_wrap_attn_forward()` pass-through stub
- test_attn_head_gradient_masking and test_attn_head_activation_masking pass GREEN; all 17 Phase 2 tests still GREEN (no regressions)

## Task Commits

1. **Task 1: Write RED test stubs for test_pretrain.py** - `903f71f` (test)
2. **Task 2: Extend GradientMasker + ActivationMasker for attn head masking** - `c60dbbf` (feat)

## Files Created/Modified

- `tests/safemoe/test_pretrain.py` - 7 test functions: 5 RED stubs (import safemoe.pretrain), 2 proper unit tests for attn-head masking
- `safemoe/masking.py` - GradientMasker two-pass mask(), ActivationMasker config kwarg + _attn_layers, _wrap_attn_forward helper

## Decisions Made

- `test_attn_head_activation_masking` verifies flag-state only (not actual head output zeroing); the stronger assertion requiring forward-pass zeroing is deferred to Plan 03-03 Task 2 since it depends on the SafeCausalSelfAttention implementation in 03-02
- `_wrap_attn_forward` is a transparent pass-through in this plan; Plan 03-02 replaces it with real head-output zeroing logic
- `ActivationMasker(model)` (no config kwarg) remains backward-compatible — `_attn_layers` defaults to empty list, no changes to Phase 2 test behaviour

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 03-02 can immediately consume `_attn_layers`, `_wrap_attn_forward`, and `_qkv_param_ids` from this plan
- The 5 RED test stubs in test_pretrain.py define the exact contract that 03-02 (pretrain.py implementation) must satisfy
- test_attn_head_activation_masking will need a stronger assertion added in 03-03 Task 2 (actual head output zeroing check)

---
*Phase: 03-sgtm-training-loop*
*Completed: 2026-03-16*

## Self-Check: PASSED

- tests/safemoe/test_pretrain.py: FOUND
- safemoe/masking.py: FOUND
- 03-01-SUMMARY.md: FOUND
- Commit 903f71f: FOUND
- Commit c60dbbf: FOUND
