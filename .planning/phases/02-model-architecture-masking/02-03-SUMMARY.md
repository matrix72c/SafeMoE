---
phase: 02-model-architecture-masking
plan: "03"
subsystem: model
tags: [pytorch, litgpt, moe, safemoe, masking, registry]

# Dependency graph
requires:
  - phase: 02-model-architecture-masking
    plan: "01"
    provides: "RED test stubs for HarmfulParamRegistry, GradientMasker, ActivationMasker (test_registry.py, test_masking.py)"
  - phase: 02-model-architecture-masking
    plan: "02"
    provides: "SafeMoEConfig with harmful_expert_indices/harmful_attn_heads, SafeMoELayer with _activation_masking_enabled flag"
provides:
  - "HarmfulParamRegistry class in safemoe/masking.py: scans named_parameters(), classifies theta_harmful (expert params) and theta_std (all others), exhaustive + non-overlapping via id()-based validation"
  - "GradientMasker stub in safemoe/masking.py (NotImplementedError until Plan 04)"
  - "ActivationMasker stub in safemoe/masking.py (NotImplementedError until Plan 04)"
  - "_qkv_harmful_metadata: list of (param, [slice, ...]) tuples for Phase 3 per-head gradient masking"
affects:
  - 02-04-PLAN
  - Phase 3 (dual optimizer param groups use parameters_by_type)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "id()-based set operations for nn.Parameter identity checks (avoids tensor __eq__ pitfall)"
    - "Regex-based named_parameters() scan with class-level compiled patterns for expert and qkv matching"
    - "Dual metadata storage: qkv.weight in theta_std (full parameter) + _qkv_harmful_metadata (row slices) for Phase 3"

key-files:
  created:
    - safemoe/masking.py
  modified: []

key-decisions:
  - "qkv.weight goes into theta_std (full parameter) for Phase 2; _qkv_harmful_metadata stores (param, slices) separately for Phase 3 row-level gradient masking — duality is intentional"
  - "GradientMasker.__init__ takes registry (not model) as its sole argument; ActivationMasker.__init__ takes model (not registry) — matches RED test stub signatures"
  - "Regex patterns compiled at class level (_EXPERT_RE, _QKV_RE) for efficient reuse across many named_parameters() entries"

patterns-established:
  - "Pattern: HarmfulParamRegistry validates non-overlapping and exhaustive coverage at construction time via ValueError — correctness guarantee, not optional"
  - "Pattern: parameters_by_type(split) raises KeyError for unrecognised split names (fail-fast interface)"

requirements-completed: [MOE-02, MASK-03]

# Metrics
duration: 3min
completed: 2026-03-16
---

# Phase 2 Plan 03: HarmfulParamRegistry Implementation Summary

**HarmfulParamRegistry with id()-based exhaustive classification of theta_harmful/theta_std and _qkv_harmful_metadata for Phase 3 per-head gradient masking, plus GradientMasker/ActivationMasker stubs**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-16T02:04:34Z
- **Completed:** 2026-03-16T02:08:00Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Created `safemoe/masking.py` with fully implemented HarmfulParamRegistry that correctly classifies all 70 parameters of a 2-layer SMALL_CONFIG model (24 theta_harmful, 46 theta_std)
- Non-overlapping and exhaustive validation via id()-based set operations; raises ValueError on any classification error
- _qkv_harmful_metadata stores (qkv_param, row_slices) tuples for Phase 3 per-head gradient masking; empty list when harmful_attn_heads=[]
- GradientMasker and ActivationMasker stubs allow test_masking.py to collect 4 tests without ImportError (stubs raise NotImplementedError until Plan 04)
- All 5 tests in test_registry.py pass GREEN

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement HarmfulParamRegistry in safemoe/masking.py** - `55a088f` (feat)

## Files Created/Modified

- `safemoe/masking.py` — HarmfulParamRegistry (fully implemented), GradientMasker (stub), ActivationMasker (stub); 185 lines

## Decisions Made

- **qkv.weight duality**: The qkv.weight parameter is added to theta_std so the Phase 2 maskers never touch it, while (param, slices) metadata is also stored in _qkv_harmful_metadata. This preserves the set-cover invariant while giving Phase 3 the row-slice information it needs for per-head gradient masking. This is intentional and documented in both the module docstring and code comments.
- **GradientMasker takes registry, ActivationMasker takes model**: The RED test stubs establish these constructor signatures (`GradientMasker(registry)`, `ActivationMasker(model)`) — stubs match this exactly so Plan 04 can complete the implementation without interface changes.
- **Class-level compiled regex**: `_EXPERT_RE` and `_QKV_RE` are compiled at class definition time for efficiency across the named_parameters() scan.

## Deviations from Plan

None — plan executed exactly as written. safemoe/config.py and safemoe/model.py were already in place from plan 02-02 (executed in a prior session).

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `safemoe/masking.py` is ready; Plan 04 can implement GradientMasker.mask() and ActivationMasker.enable()/disable() against the existing stubs
- `parameters_by_type('theta_harmful')` and `parameters_by_type('theta_std')` are ready for Phase 3 dual AdamW optimizer construction
- `_qkv_harmful_metadata` is ready for Phase 3 per-head gradient masking on qkv.weight rows
- `test_masking.py` collects 4 tests (no ImportError); Plan 04 will make them GREEN

---
*Phase: 02-model-architecture-masking*
*Completed: 2026-03-16*

## Self-Check: PASSED

- safemoe/masking.py: FOUND
- 02-03-SUMMARY.md: FOUND
- Commit 55a088f (HarmfulParamRegistry): FOUND
