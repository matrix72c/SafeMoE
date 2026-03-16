---
phase: 02-model-architecture-masking
plan: "02"
subsystem: model
tags: [pytorch, litgpt, moe, safemoe, dataclass, masking]

# Dependency graph
requires:
  - phase: 02-model-architecture-masking
    plan: "01"
    provides: "RED test stubs for SafeMoEConfig, SafeMoELayer, HarmfulParamRegistry, GradientMasker, ActivationMasker"
provides:
  - "SafeMoEConfig @dataclass subclass of litgpt.Config with harmful_expert_indices, harmful_attn_heads, num_harmful_experts fields and mlp_class property override"
  - "SafeMoELayer subclass of LLaMAMoE with _activation_masking_enabled flag, init_strategy constructor arg, and forward() that zeroes harmful experts when masked"
affects:
  - 02-03-PLAN
  - 02-04-PLAN

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Lazy import inside @property body to break circular dependency (config.py imports model.py at call time only)"
    - "init_strategy constructor arg (not config field) to control harmful expert weight initialization"
    - "Expert masking via enumerate(zip(masks, experts)) with index-in-set check for O(1) skip logic"

key-files:
  created:
    - safemoe/config.py
    - safemoe/model.py
  modified: []

key-decisions:
  - "init_strategy='random'|'copy' is a SafeMoELayer constructor arg, not a SafeMoEConfig field — tests call SafeMoELayer(config, init_strategy='copy') directly, so config.harmful_expert_init field is not needed"
  - "Lazy import of SafeMoELayer inside SafeMoEConfig.mlp_class property body resolves the circular import (config.py is imported by model.py; model.py cannot be top-level imported from config.py)"
  - "SafeMoELayer.forward() mirrors full LLaMAMoE.forward() logic including routed_scaling_factor and n_shared_expert branches to avoid subtle divergence from parent"
  - "list[int] type annotation written as 'list' (no subscript) for Python 3.9 compat; field(default_factory=list) prevents shared mutable default"

patterns-established:
  - "Pattern: SafeMoEConfig.mlp_class returns SafeMoELayer for LLaMAMoE, delegates to super().mlp_class for all other mlp_class_name values"
  - "Pattern: _harmful_indices stored as plain list[int] on SafeMoELayer (copy of config.harmful_expert_indices) for fast membership checks in forward()"

requirements-completed: [MOE-01, MOE-03, MOE-04]

# Metrics
duration: 2min
completed: 2026-03-16
---

# Phase 2 Plan 02: SafeMoEConfig and SafeMoELayer Implementation Summary

**SafeMoEConfig dataclass wiring SafeMoELayer into every LLaMAMoE block via mlp_class override, with activation masking and copy-weights harmful expert initialization**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-16T02:04:04Z
- **Completed:** 2026-03-16T02:05:56Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Created `safemoe/config.py` with SafeMoEConfig: litgpt.Config subclass with three harmful-expert tracking fields and a lazy-import mlp_class property override
- Created `safemoe/model.py` with SafeMoELayer: LLaMAMoE subclass with `_activation_masking_enabled` flag, `init_strategy` constructor arg, and full forward() override that silently zeros harmful expert contributions when masking is active
- All 8 tests across test_config.py and test_model.py pass GREEN; model instantiation with litgpt.GPT(SafeMoEConfig(...)) confirmed

## Task Commits

Each task was committed atomically:

1. **Task 1: SafeMoEConfig implementation** - `29ec526` (feat)
2. **Task 2: SafeMoELayer implementation** - `c1b5307` (feat)

## Files Created/Modified

- `safemoe/config.py` — SafeMoEConfig dataclass with harmful_expert_indices, harmful_attn_heads, num_harmful_experts, and mlp_class property override
- `safemoe/model.py` — SafeMoELayer with _activation_masking_enabled flag, init_strategy='random'|'copy' constructor arg, forward() with harmful-expert skip logic

## Decisions Made

- **init_strategy is a constructor arg, not a config field**: The RED tests call `SafeMoELayer(config, init_strategy="copy")` directly. The plan's suggested approach (adding `harmful_expert_init` to SafeMoEConfig) would not satisfy this test signature. Implemented as a constructor argument instead.
- **Lazy import in mlp_class property**: `safemoe/config.py` imports `safemoe/model.py` at call time inside the property body, while `safemoe/model.py` imports `safemoe/config.py` at module level for the type annotation (TYPE_CHECKING guard used). This is the established pattern from litgpt's own mlp_class property.
- **Full forward() reimplementation**: SafeMoELayer.forward() reproduces the full LLaMAMoE forward logic (including routed_scaling_factor and n_shared_expert) rather than calling super().forward() with a monkey-patched skip, to keep the harmful-expert masking logic explicit and auditable.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] init_strategy as constructor arg, not config field**
- **Found during:** Task 2 (SafeMoELayer implementation)
- **Issue:** Plan suggested adding `harmful_expert_init: str = "random"` to SafeMoEConfig and using `getattr(config, 'harmful_expert_init', 'random')` in SafeMoELayer.__init__. But the RED test stubs call `SafeMoELayer(config, init_strategy="random")` and `SafeMoELayer(config, init_strategy="copy")` — a positional/keyword constructor argument, not a config field.
- **Fix:** Added `init_strategy: str = "random"` as second constructor arg to SafeMoELayer.__init__(). No config field needed.
- **Files modified:** safemoe/model.py
- **Verification:** `test_harmful_expert_init_random` and `test_harmful_expert_init_copy` both pass GREEN.
- **Committed in:** c1b5307 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — test contract mismatch between plan text and RED stubs)
**Impact on plan:** Fix was necessary for correctness — plan text and test stubs disagreed on the interface. Test stubs are authoritative per TDD contract.

## Issues Encountered

None beyond the init_strategy interface mismatch documented above.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `safemoe/config.py` and `safemoe/model.py` are ready; plans 02-03 (HarmfulParamRegistry) and 02-04 (GradientMasker, ActivationMasker) can now import SafeMoEConfig and SafeMoELayer
- `tests/safemoe/test_registry.py` and `tests/safemoe/test_masking.py` still fail RED (expected — awaiting 02-03 and 02-04)
- The `_activation_masking_enabled` flag interface is in place; ActivationMasker (02-04) just needs to set this flag

---
*Phase: 02-model-architecture-masking*
*Completed: 2026-03-16*

## Self-Check: PASSED

- safemoe/config.py: FOUND
- safemoe/model.py: FOUND
- 02-02-SUMMARY.md: FOUND
- Commit 29ec526 (SafeMoEConfig): FOUND
- Commit c1b5307 (SafeMoELayer): FOUND
