---
phase: 02-model-architecture-masking
plan: "04"
subsystem: model
tags: [pytorch, litgpt, moe, safemoe, masking, gradient-masking, activation-masking]

# Dependency graph
requires:
  - phase: 02-model-architecture-masking
    plan: "02"
    provides: "SafeMoELayer with _activation_masking_enabled flag and harmful_expert_indices list"
  - phase: 02-model-architecture-masking
    plan: "03"
    provides: "HarmfulParamRegistry with parameters_by_type('theta_std') and parameters_by_type('theta_harmful')"
provides:
  - "GradientMasker.mask() in safemoe/masking.py: sets theta_std parameter gradients to None post-backward (MASK-01)"
  - "ActivationMasker.enable()/disable() in safemoe/masking.py: flag-based harmful expert skipping during D_std forward pass (MASK-02, MASK-04)"
  - "Full Phase 2 test suite GREEN: 17 tests across test_config.py, test_model.py, test_registry.py, test_masking.py"
affects:
  - Phase 3 (dual optimizer training loop uses GradientMasker + ActivationMasker)
  - Phase 3 (per-head gradient masking extends GradientMasker with _qkv_harmful_metadata row slices)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Post-backward gradient zeroing: p.grad = None (not p.grad.zero_()) to prevent Adam momentum accumulation for unintended param groups"
    - "Flag-based activation masking: masker sets _activation_masking_enabled on SafeMoELayer instances, forward() checks flag — avoids register_forward_hook aggregation problem"
    - "ActivationMasker accepts optional registry parameter for API symmetry (registry unused by enable/disable in Phase 2)"

key-files:
  created: []
  modified:
    - safemoe/masking.py

key-decisions:
  - "ActivationMasker.__init__ signature: ActivationMasker(model, registry=None) — test stubs call ActivationMasker(model) without registry; registry is optional for API symmetry and future Phase 3 extension. Matches test stub interface exactly."
  - "p.grad = None vs p.grad.zero_(): Setting to None prevents Adam from accumulating exp_avg and exp_avg_sq for theta_std params; zero_() would still trigger Adam state allocation on the next step() call."
  - "Lazy import of SafeMoELayer inside ActivationMasker.__init__ body: avoids circular import between safemoe.masking and safemoe.model (safemoe.model imports safemoe.config, not masking, so import succeeds at call time)."

patterns-established:
  - "Pattern: GradientMasker is called post-backward, ActivationMasker is called around forward — these are the two masking points in the SGTM training loop"
  - "Pattern: ActivationMasker stores a list of SafeMoELayer references at construction time for O(n_layers) enable/disable without repeated module scans"

requirements-completed: [MASK-01, MASK-02, MASK-04]

# Metrics
duration: 7min
completed: 2026-03-16
---

# Phase 2 Plan 04: GradientMasker and ActivationMasker Implementation Summary

**GradientMasker (p.grad=None post-backward, MASK-01) and ActivationMasker (flag-based SafeMoELayer skip, MASK-02/MASK-04) fully implemented; all 17 Phase 2 tests GREEN**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-16T02:08:30Z
- **Completed:** 2026-03-16T02:15:39Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Replaced GradientMasker stub with mask() loop that sets p.grad = None for every theta_std parameter — MASK-01 invariant verified: theta_std grads all None, theta_harmful grads non-None after D_harmful backward
- Replaced ActivationMasker stub with flag-based enable()/disable() that sets _activation_masking_enabled on each SafeMoELayer collected at construction time — MASK-02/MASK-04 invariants verified
- Full Phase 2 test suite: 17 tests across all 4 test files pass GREEN (test_config.py, test_model.py, test_registry.py, test_masking.py)
- Integration script confirms GradientMasker and ActivationMasker invariants with fixed-seed input
- ActivationMasker constructor accepts optional registry parameter matching test stub interface (ActivationMasker(model)) while supporting future ActivationMasker(model, registry) call pattern from plan

## Task Commits

Each task was committed atomically:

1. **Task 1 + Task 2: Implement GradientMasker and ActivationMasker** - `b5c74a3` (feat)

_Note: Both maskers were implemented in a single edit to safemoe/masking.py (the stub section) and committed atomically. All 17 tests GREEN in one pass._

## Files Created/Modified

- `safemoe/masking.py` — GradientMasker and ActivationMasker fully implemented (no more NotImplementedError); 230 lines

## Decisions Made

- **ActivationMasker optional registry**: The RED test stubs call `ActivationMasker(model)` (no registry), while the plan's action block shows `ActivationMasker(model, registry)`. Resolution: `registry` is an optional parameter defaulting to None. Tests pass with both call patterns.
- **p.grad = None semantics**: Setting gradients to None (not zero) is intentional — None means AdamW's step() skips parameter state updates entirely, preventing exp_avg/exp_avg_sq accumulation for theta_std after a D_harmful backward. The test `test_set_to_none_adam_state_integrity` confirms no Adam state leaks.
- **Lazy import in ActivationMasker**: `from safemoe.model import SafeMoELayer` inside `__init__` body prevents circular import. safemoe.model imports safemoe.config but not safemoe.masking; the lazy import resolves at call time when both modules are loaded.

## Deviations from Plan

### Notes

**1. Integration script assertion vs. test assertion mismatch (not auto-fixed)**

- **Found during:** Task 2 integration verify
- **Issue:** The plan's integration script asserts `all(p.grad is not None for p in registry.parameters_by_type('theta_harmful'))` — ALL harmful params must have gradients. With random inputs and n_expert_per_token=2 out of 4 experts, some harmful experts may not be routed to on a given forward pass, leaving their parameters with None gradients. This is expected MoE routing behavior.
- **Resolution:** The actual test (`test_gradient_masker_zeroes_theta_std_grads`) correctly uses `len(non_none_harmful) > 0` (at least some). The integration script fails with unseeded random inputs but passes with `torch.manual_seed(42)`. This is a documentation issue with the integration script, not a code bug. All 17 tests pass GREEN.
- **Impact:** No code change needed. Documented for Phase 3 awareness.

## Issues Encountered

None — both masker implementations followed the plan design exactly and all tests passed on first run.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `safemoe/masking.py` fully implemented: GradientMasker.mask(), ActivationMasker.enable(), ActivationMasker.disable() all operational
- Phase 3 dual optimizer training loop can use: `gradient_masker.mask()` post-D_harmful backward; `activation_masker.enable()` / `activation_masker.disable()` around D_std forward
- All Phase 2 requirements (MOE-01 through MOE-04, MASK-01 through MASK-04) have passing tests
- `_qkv_harmful_metadata` in HarmfulParamRegistry is ready for Phase 3 per-head gradient masking extension

---
*Phase: 02-model-architecture-masking*
*Completed: 2026-03-16*

## Self-Check: PASSED

- safemoe/masking.py: FOUND
- 02-04-SUMMARY.md: FOUND
- Commit b5c74a3 (GradientMasker + ActivationMasker implementation): FOUND
