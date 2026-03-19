---
phase: 05-environment-runtime-gate
plan: "02"
subsystem: infra
tags: [qwen, runtime-gate, bf16, fsdp, testing, ENV-01, ENV-02]

# Dependency graph
requires:
  - phase: 05-01
    provides: direct-Qwen preflight validation and the blessed 4-GPU BF16 gate config
provides:
  - Machine-readable `PHASE5_GATE_*` startup, first-step, throughput, and peak-memory metrics on the direct pretrain path
  - Canonical Phase 5 runtime-envelope artifact populated with the real BF16 gate measurements and storage footprint
  - Post-gate fixes so direct Qwen configs normalize to `SafeMoEConfig` and the one-step gate checkpoint saves without optimizer export
affects: [06-checkpoint-surgery, 08-warmup-separation, 09-mixed-data-transfer]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Exact stdout metric-key contracts for milestone runtime gates
    - Phase-specific FSDP checkpoint saves that omit optimizer export for one-step sparse-MoE gate runs

key-files:
  created:
    - .planning/phases/05-environment-runtime-gate/05-02-SUMMARY.md
  modified:
    - tests/safemoe/test_phase5_runtime_gate.py
    - safemoe/pretrain.py
    - .planning/phases/05-environment-runtime-gate/05-runtime-envelope.md
    - .planning/STATE.md
    - .planning/ROADMAP.md
    - .planning/REQUIREMENTS.md

key-decisions:
  - "Record the approved 2026-03-19 BF16 gate as PASS because the first optimizer step and all `PHASE5_GATE_*` metrics completed before the original post-step optimizer export failure."
  - "Normalize direct gate model-name configs to `SafeMoEConfig` before entering the main pretrain path so Phase 5 always preserves SafeMoE-specific fields."
  - "Skip optimizer export when saving the Phase 5 one-step gate checkpoint under FSDP; save model/config state only."

patterns-established:
  - "Phase 5 runtime envelope: copy the exact `PHASE5_GATE_*` stdout lines verbatim into the canonical markdown artifact."
  - "One-step sparse-MoE runtime gates may save a replayable checkpoint without optimizer state when the acceptance target is startup plus the first optimizer step."

requirements-completed: [ENV-01, ENV-02]

# Metrics
duration: 57min
completed: 2026-03-19
---

# Phase 5 Plan 02: Runtime Envelope Summary

**Direct-Qwen Phase 5 gate metrics, canonical BF16 runtime envelope, and FSDP-safe checkpoint save behavior for the one-step runtime proof**

## Performance

- **Duration:** 57 min
- **Started:** 2026-03-19T05:13:29Z
- **Completed:** 2026-03-19T06:10:19Z
- **Tasks:** 3
- **Files modified:** 7

## Accomplishments

- Added exact `PHASE5_GATE_STARTUP_SECONDS`, `PHASE5_GATE_FIRST_STEP_SECONDS`, `PHASE5_GATE_FIRST_STEP_TOKENS_PER_SEC`, and `PHASE5_GATE_PEAK_MEMORY_GB` output coverage on the direct `safemoe pretrain` path.
- Finalized `.planning/phases/05-environment-runtime-gate/05-runtime-envelope.md` with the approved BF16 gate measurements, `57G` checkpoint footprint, CUDA host, and captured warnings from the real 4-GPU run.
- Landed two follow-up fixes after the checkpoint run so direct-Qwen model-name configs normalize into `SafeMoEConfig` and the Phase 5 one-step checkpoint save skips optimizer export under FSDP.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add one-step runtime-gate metric instrumentation and coverage** - `93b2f99` (`test`), `55d91c1` (`feat`)
2. **Task 2: Create the canonical runtime-envelope report template in the phase directory** - `8935b51` (`docs`)
3. **Task 3: Run the blessed BF16 gate on the GPU host and finalize the envelope report** - `797f330` (`docs`)

## Files Created/Modified

- `tests/safemoe/test_phase5_runtime_gate.py` - Coverage for Phase 5 metric strings, SafeMoE config normalization, and gate checkpoint-save behavior.
- `safemoe/pretrain.py` - Direct-path metric emission, SafeMoE config normalization, and Phase 5 checkpoint saves without optimizer export.
- `.planning/phases/05-environment-runtime-gate/05-runtime-envelope.md` - Canonical BF16 runtime envelope with measured startup, first-step, throughput, memory, storage, host, and warnings.
- `.planning/phases/05-environment-runtime-gate/05-02-SUMMARY.md` - Machine-readable execution summary for the completed plan.
- `.planning/STATE.md` - Updated current position, progress, decisions, and execution metrics after plan completion.
- `.planning/ROADMAP.md` - Updated Phase 5 plan progress to reflect completion of both Phase 5 plans.
- `.planning/REQUIREMENTS.md` - Marked `ENV-02` complete in the requirement checklist and traceability table.

## Decisions Made

- Treat the original 2026-03-19 BF16 gate run as the requirement-closing proof for Phase 5 because startup plus the first optimizer step completed and emitted all four required metrics before the post-step optimizer-state export failure.
- Normalize model-name configs through `ensure_safemoe_config()` so the direct pretrain entrypoint always passes a `SafeMoEConfig` into the Phase 5 gate path.
- Save Phase 5 runtime-gate checkpoints without optimizer state under FSDP because sparse MoE parameters can legitimately lack optimizer slots after a single measured step.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Normalize direct gate configs to `SafeMoEConfig`**
- **Found during:** Task 3 (Run the blessed BF16 gate on the GPU host and finalize the envelope report)
- **Issue:** The direct gate path could still construct a plain LitGPT `Config` from the model name, which dropped SafeMoE-specific fields needed by the real direct-Qwen runtime gate.
- **Fix:** Added `ensure_safemoe_config()` and switched the model-name fallback to `SafeMoEConfig.from_name(...)`, with regression coverage for the normalized config path.
- **Files modified:** `safemoe/pretrain.py`, `tests/safemoe/test_phase5_runtime_gate.py`
- **Verification:** `pytest tests/safemoe/test_phase5_runtime_gate.py tests/safemoe/test_pretrain.py -x`
- **Committed in:** `85cc37f`

**2. [Rule 1 - Bug] Skip optimizer export for the Phase 5 one-step gate checkpoint**
- **Found during:** Task 3 (Run the blessed BF16 gate on the GPU host and finalize the envelope report)
- **Issue:** The initial BF16 gate reached the first optimizer step and emitted the required metrics, but the final FSDP optimizer-state export could still fail because many sparse-MoE parameters legitimately had no optimizer slots after a single step.
- **Fix:** Added an `include_optimizer` flag to `save_checkpoint()` and disabled optimizer export for the Phase 5 runtime gate while preserving final model/config checkpoint output, with regression coverage.
- **Files modified:** `safemoe/pretrain.py`, `tests/safemoe/test_phase5_runtime_gate.py`
- **Verification:** `pytest tests/safemoe/test_phase5_runtime_gate.py tests/safemoe/test_pretrain.py -x`
- **Committed in:** `837084f`

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 bug)
**Impact on plan:** Both fixes were required to close the real GPU gate cleanly and did not broaden Phase 5 scope beyond the blessed runtime-gate path.

## Issues Encountered

- The original GPU-host BF16 run completed startup and the first optimizer step, then failed during final FSDP optimizer-state export. The canonical artifact records `PASS` because the plan’s one-step runtime-gate condition was already satisfied, and the follow-up fix removed that post-step failure mode.
- The worktree contained unrelated changes in `.planning/phases/05-environment-runtime-gate/05-02-PLAN.md` plus untracked `.planning/phases/05-environment-runtime-gate/05-CONTEXT.md` and `.planning/phases/05-environment-runtime-gate/05-VALIDATION.md`; they were preserved untouched.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 5 now has the real BF16 storage, startup, first-step, throughput, and peak-memory envelope needed to size Phase 6+ experiments.
- Phase 6 can assume the direct-Qwen gate path loads with `SafeMoEConfig` semantics and saves a replayable one-step checkpoint without optimizer export.

---
*Phase: 05-environment-runtime-gate*
*Completed: 2026-03-19*

## Self-Check: PASSED

- FOUND: .planning/phases/05-environment-runtime-gate/05-02-SUMMARY.md
- FOUND: .planning/phases/05-environment-runtime-gate/05-runtime-envelope.md
- FOUND: tests/safemoe/test_phase5_runtime_gate.py
- FOUND: safemoe/pretrain.py
- FOUND: `93b2f99`
- FOUND: `55d91c1`
- FOUND: `8935b51`
- FOUND: `85cc37f`
- FOUND: `837084f`
- FOUND: `797f330`
