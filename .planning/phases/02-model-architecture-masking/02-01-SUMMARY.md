---
phase: 02-model-architecture-masking
plan: "01"
subsystem: testing
tags: [pytest, tdd, safemoe, masking, moe, litgpt]

# Dependency graph
requires:
  - phase: 02-model-architecture-masking
    provides: "Research, context, and validation strategy for SafeMoEConfig, SafeMoELayer, HarmfulParamRegistry, GradientMasker, ActivationMasker"
provides:
  - "Failing RED test stubs for all Phase 2 requirements (4 test files)"
  - "safemoe/configs/safemoe-tinystories.yaml experiment config"
  - "Complete test contract for plans 02-02 through 02-04 to implement against"
affects:
  - 02-02-PLAN
  - 02-03-PLAN
  - 02-04-PLAN

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "TDD RED phase: import-failing test stubs establish contract before implementation"
    - "id()-based set comparisons for nn.Parameter identity (avoids tensor __eq__ pitfall)"
    - "SMALL_CONFIG dict repeated in each test file (no conftest.py to avoid namespace collision)"

key-files:
  created:
    - tests/safemoe/test_config.py
    - tests/safemoe/test_model.py
    - tests/safemoe/test_registry.py
    - tests/safemoe/test_masking.py
    - safemoe/configs/safemoe-tinystories.yaml
  modified: []

key-decisions:
  - "No tests/safemoe/__init__.py — Phase 1 lesson: pytest namespace collision shadows source package"
  - "SMALL_CONFIG dims inlined in each test file (no shared conftest.py) for test isolation"
  - "id()-based set comparisons for registry exhaustive-coverage and non-overlapping tests"
  - "ActivationMasker test checks _activation_masking_enabled flag, not per-expert output values (flag-based approach)"

patterns-established:
  - "Pattern: RED stubs import from not-yet-existing modules; ImportError is the correct RED state"
  - "Pattern: test_registry.py uses id(p) for set comparisons, not p directly"
  - "Pattern: safemoe/configs/ YAML uses SafeMoEConfig-specific fields as top-level keys"

requirements-completed: [MOE-01, MOE-02, MOE-03, MOE-04, MASK-01, MASK-02, MASK-03, MASK-04]

# Metrics
duration: 3min
completed: 2026-03-16
---

# Phase 2 Plan 01: RED Test Stubs for SafeMoE Model Architecture Summary

**Pytest RED stubs for SafeMoEConfig, SafeMoELayer, HarmfulParamRegistry, GradientMasker, and ActivationMasker — plus safemoe-tinystories.yaml experiment config**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-16T01:57:50Z
- **Completed:** 2026-03-16T02:00:56Z
- **Tasks:** 4
- **Files modified:** 5

## Accomplishments

- Created 4 failing RED test files covering all 8 Phase 2 requirements (MOE-01 through MASK-04)
- Created `safemoe/configs/safemoe-tinystories.yaml` with valid YAML (mlp_class_name: LLaMAMoE, n_expert: 8, harmful_expert_indices: [0, 1])
- All four test files fail with ImportError (expected RED state) — implementation plans 02-02 through 02-04 will make them green
- No `tests/safemoe/__init__.py` created (Phase 1 lesson applied: prevents namespace collision)

## Task Commits

Each task was committed atomically:

1. **Task 1: RED stubs for SafeMoEConfig** - `f302ad3` (test)
2. **Task 2: RED stubs for SafeMoELayer + YAML** - `ce2dbdc` (test)
3. **Task 3: RED stubs for HarmfulParamRegistry** - `6b80a2b` (test)
4. **Task 4: RED stubs for GradientMasker and ActivationMasker** - `4c30582` (test)

## Files Created/Modified

- `tests/safemoe/test_config.py` — 4 failing stubs for MOE-01 (SafeMoEConfig instantiation, defaults, inheritance, mlp_class property)
- `tests/safemoe/test_model.py` — 4 failing stubs for MOE-03/MOE-04 (SafeMoELayer structure, random/copy init, masking flag)
- `tests/safemoe/test_registry.py` — 5 failing stubs for MOE-02/MASK-03 (registry coverage, non-overlapping, expert param classification)
- `tests/safemoe/test_masking.py` — 4 failing stubs for MASK-01/MASK-02/MASK-04 (gradient masking, activation masking, combined invariants, AdamW state)
- `safemoe/configs/safemoe-tinystories.yaml` — Experiment config: n_layer=4, n_embd=128, n_expert=8, harmful_expert_indices=[0,1]

## Decisions Made

- **No conftest.py**: SMALL_CONFIG dims inlined in each test file to avoid the pytest namespace collision that caused issues in Phase 1. Each test file is independently importable.
- **id()-based comparisons**: Registry tests use `{id(p) for p in ...}` for set operations, per RESEARCH.md Pitfall 2 guidance (nn.Parameter inherits tensor __eq__ which is element-wise, not identity).
- **Flag-based activation masker test**: `test_activation_masker_zeroes_harmful_expert_output` checks `_activation_masking_enabled` flag state rather than per-expert output tensor values (consistent with flag-based design decision from CONTEXT.md/RESEARCH.md).

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- All 4 test files exist and fail RED as required — plans 02-02, 02-03, 02-04 can implement against these contracts
- `safemoe/configs/safemoe-tinystories.yaml` ready for use in training experiments
- Wave 1 plans (02-02: SafeMoEConfig + SafeMoELayer, 02-03: HarmfulParamRegistry, 02-04: maskers) will make these tests green

---
*Phase: 02-model-architecture-masking*
*Completed: 2026-03-16*
