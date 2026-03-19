---
phase: 07-registry-and-routing-observability
plan: 02
subsystem: observability
tags: [routing, observability, parity, pytest, qwen]
requires:
  - phase: 07-01
    provides: registry ownership inventory and summary artifacts for direct-Qwen checkpoints
provides:
  - shared routing observability collector reused by eval and pretrain checkpoint flows
  - routing observability JSON and Markdown artifacts with dispatch counts and harmful fractions
  - parity helper that writes routing_parity.json and hard-fails on mismatches
affects: [warmup, transfer, evaluation, routing telemetry]
tech-stack:
  added: []
  patterns: [shared observability helper module, checkpoint-local routing artifacts, parity-as-artifact]
key-files:
  created: [safemoe/observability.py]
  modified: [tests/safemoe/test_evaluate.py, tests/safemoe/test_pretrain.py, safemoe/evaluate.py, safemoe/pretrain.py]
key-decisions:
  - "Put shared routing collection, artifact writing, and parity checks in safemoe/observability.py so eval and pretrain reuse one codepath."
  - "Keep routing_attribution() backward-compatible by preserving fraction returns and the legacy routing_attribution.json while also writing the Phase 7 routing_observability artifacts."
patterns-established:
  - "Routing observability: collect SafeMoELayer._last_indices through a shared collector and emit flat per-split metrics."
  - "Parity verification: write routing_parity.json before raising so failures remain inspectable."
requirements-completed: [ROUT-02, ROUT-03]
duration: 14min
completed: 2026-03-19
---

# Phase 7 Plan 02: Shared routing observability with dispatch-count artifacts and hard-fail parity reporting

**Shared routing observability now emits dispatch counts and harmful fractions through one collector reused by eval and checkpoint-local pretrain flows, with parity failures persisted as inspectable artifacts.**

## Performance

- **Duration:** 14 min
- **Started:** 2026-03-19T13:56:30Z
- **Completed:** 2026-03-19T14:10:42Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Added RED coverage locking the shared artifact names, dispatch-count contract, omitted-split behavior, and parity hard-failure artifact.
- Created `safemoe/observability.py` with `RoutingObservabilityCollector`, `write_routing_artifacts()`, and `assert_routing_parity()`.
- Refactored eval routing attribution into a compatibility wrapper over the shared collector and added checkpoint-local routing observability output in pretrain.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add RED coverage for the shared routing artifact contract** - `4c29ef0` (test)
2. **Task 2: Implement the shared collector and wire it into eval and pretrain** - `8a69fe3` (feat)

## Files Created/Modified
- `safemoe/observability.py` - Shared routing collector, artifact writer, and parity helper.
- `safemoe/evaluate.py` - Compatibility wrapper over the shared routing collector plus shared artifact writing.
- `safemoe/pretrain.py` - Checkpoint-local routing observability emission hook for training flows.
- `tests/safemoe/test_evaluate.py` - Shared routing artifact coverage for eval.
- `tests/safemoe/test_pretrain.py` - Parity failure artifact coverage.

## Decisions Made
- Centralized train/eval routing observability in `safemoe/observability.py` to prevent schema drift across later warmup, transfer, and evaluation phases.
- Preserved `routing_attribution()` compatibility for existing callers while extending its outputs to the Phase 7 observability contract.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- The plan’s `pytest -k "routing parity"` selector is invalid pytest syntax. Verification used `pytest tests/safemoe/test_pretrain.py -k "routing and parity" -x` to run the intended subset.
- A stale `.git/index.lock` appeared during staging. It was removed with `rm -f .git/index.lock`, then staging continued normally.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Later warmup, transfer, and evaluation flows can reuse the same routing artifact schema without adding another routing collector.
- Parity enforcement is implemented and artifactized, ready for any targeted flow that chooses to pass logged routing metrics into the shared helper.

## Self-Check: PASSED

- Verified `.planning/phases/07-registry-and-routing-observability/07-02-SUMMARY.md` exists.
- Verified task commits `4c29ef0` and `8a69fe3` exist in `git log --oneline --all`.

---
*Phase: 07-registry-and-routing-observability*
*Completed: 2026-03-19*
