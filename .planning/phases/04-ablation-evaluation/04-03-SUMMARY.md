---
phase: 04-ablation-evaluation
plan: "03"
subsystem: evaluation
tags: [evaluate, perplexity, routing-attribution, moe, safemoe, tensorboard, jsonargparse]

# Dependency graph
requires:
  - phase: 04-01
    provides: ablation checkpoint format (lit_model.pth, model_config.yaml)
  - phase: 02-model-architecture-masking
    provides: SafeMoELayer with _activation_masking_enabled, SafeMoEConfig.harmful_expert_indices
  - phase: 03-sgtm-training-loop
    provides: validate() function for per-split evaluation
provides:
  - safemoe/evaluate.py with evaluate_perplexity() and routing_attribution() (EVAL-01, EVAL-02)
  - safemoe/model.py SafeMoELayer._last_indices attribute for routing hook access
  - python -m safemoe evaluate CLI subcommand
affects: [04-04, 04-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - data_mock parameter for testability without real streaming data
    - Forward hook pattern for expert dispatch collection (SafeMoELayer._last_indices)
    - fabric.setup() called on raw model (no DDP prefix) for standalone eval CLI
    - TensorBoard optional (try/except ImportError) for routing metrics logging

key-files:
  created:
    - safemoe/evaluate.py
    - tests/safemoe/test_evaluate.py
  modified:
    - safemoe/model.py
    - safemoe/__main__.py

key-decisions:
  - "SafeMoELayer._last_indices written in both topk branches (standard and expert groups) — negligible overhead since attribute unused unless hook reads it"
  - "evaluate_perplexity re-fetches val loaders for ablated model to handle exhausted DataLoader iterators"
  - "data_mock duck-typing: any object with val_dataloaders() returning {D_std, D_harmful} works"
  - "routing hook captures _last_indices.flatten().tolist() per layer per forward call — avoids GPU tensor retention"

patterns-established:
  - "Evaluate CLI uses data_mock for testability, real MultiDataLoader only in production path"
  - "results.json and routing_attribution.json always written alongside checkpoint for traceability"
  - "No D_unlabeled keys anywhere in evaluation outputs (user decision locked)"

requirements-completed: [EVAL-01, EVAL-02]

# Metrics
duration: 14min
completed: 2026-03-16
---

# Phase 4 Plan 03: Evaluation CLI Summary

**Perplexity evaluation CLI and routing attribution analysis using SafeMoELayer forward hooks, producing results.json and routing_attribution.json per checkpoint**

## Performance

- **Duration:** 14 min
- **Started:** 2026-03-16T14:48:18Z
- **Completed:** 2026-03-16T14:03:43Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- SafeMoELayer.forward() now stores `_last_indices` in both routing branches, enabling routing attribution via forward hooks
- evaluate_perplexity() loads original/ablated checkpoints, runs validate() per D_std/D_harmful split, prints comparison table, writes results.json
- routing_attribution() installs forward hooks on SafeMoELayer, computes harmful expert dispatch fractions per split, writes routing_attribution.json and TensorBoard scalars
- python -m safemoe evaluate CLI subcommand registered alongside pretrain and ablate

## Task Commits

Each task was committed atomically:

1. **Task 1: Add _last_indices to SafeMoELayer.forward()** - `d1c65d8` (feat)
2. **Task 2 RED: Failing tests for evaluate_perplexity and routing_attribution** - `eebf790` (test)
3. **Task 2 GREEN: Implement safemoe/evaluate.py (EVAL-01 + EVAL-02)** - `052f7e3` (feat)
4. **Task 3: Add evaluate subcommand to __main__.py** - `4ed4e3b` (feat)

_TDD task 2 has RED + GREEN commits as per TDD protocol._

## Files Created/Modified

- `safemoe/model.py` - Added `self._last_indices = indices` in both forward branches of SafeMoELayer
- `safemoe/evaluate.py` - New: evaluate_perplexity(), routing_attribution(), setup() CLI entry point
- `tests/safemoe/test_evaluate.py` - New: TDD tests for both evaluation functions
- `safemoe/__main__.py` - Added evaluate subcommand (evaluate_fn import + PARSER_DATA entry)

## Decisions Made

- **Re-fetch val loaders for ablated model**: DataLoader objects from the first eval pass may be exhausted; calling `_get_val_loaders()` again ensures fresh iterators for ablated model evaluation.
- **Routing hook closure**: Used default argument `_dispatch=dispatch_all` in hook closure to capture the correct list reference per split iteration, avoiding closure-over-loop-variable bug.
- **data_mock duck typing**: Any object with a `val_dataloaders()` method returning `{"D_std": ..., "D_harmful": ...}` works — no base class required, enabling test mocks without importing MultiDataLoader.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Evaluation CLI fully functional — ready for ablation sweep experiments (04-04)
- routing_attribution() can be used to quantify harmful expert routing specialization
- results.json schema locked: original/ablated/delta keys, val_ppl_D_std/val_ppl_D_harmful values

---
*Phase: 04-ablation-evaluation*
*Completed: 2026-03-16*

## Self-Check: PASSED

All files present and all commits verified:
- safemoe/evaluate.py: FOUND
- safemoe/model.py: FOUND
- safemoe/__main__.py: FOUND
- tests/safemoe/test_evaluate.py: FOUND
- 04-03-SUMMARY.md: FOUND
- d1c65d8 (Task 1): FOUND
- eebf790 (Task 2 RED): FOUND
- 052f7e3 (Task 2 GREEN): FOUND
- 4ed4e3b (Task 3): FOUND
