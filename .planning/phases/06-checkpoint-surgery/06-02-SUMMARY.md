---
phase: 06-checkpoint-surgery
plan: "02"
subsystem: checkpoint
tags: [litgpt, qwen3, safemoe, checkpoint-surgery, verification, pytest]
requires:
  - phase: 06-01
    provides: deterministic intervention manifest planning and serialization
provides:
  - manifest-backed checkpoint surgery CLI under `python -m safemoe surgery`
  - deterministic expert/router/qkv checkpoint mutation with staging finalization
  - reload-based verification reports that fail closed before output publication
affects: [warmup, transfer, evaluation, registry, observability]
tech-stack:
  added: []
  patterns: [manifest-first checkpoint mutation, deterministic CPU-noise replay, fail-closed staged publication]
key-files:
  created: [safemoe/interventions/surgery.py, safemoe/interventions/verify.py, safemoe/surgery.py]
  modified: [tests/safemoe/test_checkpoint_surgery.py, safemoe/interventions/__init__.py, safemoe/__main__.py]
key-decisions:
  - "Load SafeMoE checkpoint configs by filtering YAML through SafeMoEConfig fields so LitGPT-derived keys and SafeMoE metadata can coexist in model_config.yaml."
  - "Treat reload failures the same as parity mismatches: write FAIL artifacts and re-raise ValueError('Checkpoint surgery verification failed')."
patterns-established:
  - "Checkpoint surgery writes to `<output>.tmp`, verifies in place, then atomically renames only after PASS."
  - "Verifier recomputes expected expert, router, and qkv targets from the untouched base checkpoint using the manifest seed and mutation order."
requirements-completed: [INIT-02, INIT-03]
duration: 15min
completed: 2026-03-19
---

# Phase 6 Plan 02: Checkpoint Surgery Summary

**Manifest-backed Qwen checkpoint surgery with deterministic expert/router/qkv cloning, reload verification, and fail-closed publication artifacts**

## Performance

- **Duration:** 15 min
- **Started:** 2026-03-19T09:15:00Z
- **Completed:** 2026-03-19T09:29:29Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Added a researcher-facing `python -m safemoe surgery` entrypoint that plans one manifest, mutates the LitGPT checkpoint, and publishes a verified output checkpoint directory.
- Implemented deterministic expert-weight, router-column, and packed-QKV slice cloning driven entirely by the manifest seed and source/target layout.
- Added verifier reports that prove reloadability and mapping parity, and block final output publication on any mismatch or reload failure.

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement manifest-backed checkpoint mutation and blessed output writing** - `f321ebf` (test), `7b022f2` (feat)
2. **Task 2: Add deterministic value-parity verification and fail-closed finalization** - `ba8cdb3` (test), `237e865` (fix)

## Files Created/Modified
- `tests/safemoe/test_checkpoint_surgery.py` - Synthetic checkpoint integration coverage for successful surgery, verifier mismatch handling, reload hard-failure wrapping, and staging cleanup.
- `safemoe/interventions/surgery.py` - Deterministic mutation helpers, filtered config loading, sidecar copying, staging writes, and atomic finalize flow.
- `safemoe/interventions/verify.py` - Reload/parity verifier and report writers for PASS/FAIL outcomes.
- `safemoe/surgery.py` - Researcher-facing CLI setup that plans the manifest and delegates execution.
- `safemoe/__main__.py` - CLI wiring for the new `surgery` subcommand.
- `safemoe/interventions/__init__.py` - Public exports for surgery and verification helpers.

## Decisions Made
- Use the manifest as the only mutation contract; surgery derives router column pairs from expert mappings instead of accepting redundant router inputs.
- Persist verification artifacts for reload failures as well as tensor mismatches so every hard failure leaves a readable and machine-readable diagnosis.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added filtered SafeMoE config loading for checkpoint YAML**
- **Found during:** Task 1 (Implement manifest-backed checkpoint mutation and blessed output writing)
- **Issue:** Direct `from_file()` loading failed because LitGPT `model_config.yaml` includes derived keys like `rope_n_elem`, while SafeMoE adds harmful-layout fields that plain LitGPT `Config` rejects.
- **Fix:** Added a filtered YAML loader that reconstructs `SafeMoEConfig` from supported dataclass fields before mutation and verification.
- **Files modified:** `safemoe/interventions/surgery.py`, `safemoe/surgery.py`, `tests/safemoe/test_checkpoint_surgery.py`
- **Verification:** `pytest tests/safemoe/test_checkpoint_surgery.py::test_surgery_writes_loadable_checkpoint_directory -x`
- **Committed in:** `7b022f2`

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Necessary to make real LitGPT checkpoint configs load through the SafeMoE surgery path. No scope creep.

## Issues Encountered
- The first implementation created an import cycle between the surgery and verification modules; this was resolved by making verification imports local to the staging finalize path.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Phase 6 now produces verified surgery artifacts that later warmup and evaluation phases can consume as normal LitGPT checkpoint directories.
- The CLI help surface is available, and the focused regression suite covering surgery, registry, and ablation is green.

## Self-Check: PASSED
- Found `.planning/phases/06-checkpoint-surgery/06-02-SUMMARY.md`
- Verified task commits `f321ebf`, `7b022f2`, `ba8cdb3`, and `237e865` in git history
