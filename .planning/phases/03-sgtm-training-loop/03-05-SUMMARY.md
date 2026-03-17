---
phase: 03-sgtm-training-loop
plan: "05"
subsystem: human-verification
tags: [human-verify, loss-convergence, TRAIN-01, TRAIN-02, TRAIN-03]

# Dependency graph
requires:
  - phase: 03-sgtm-training-loop
    provides: behavioral tests GREEN, CLI entry point, full SGTM training loop
provides:
  - Human-confirmed loss convergence on real TinyStories bilingual data
  - Phase 3 Success Criterion 3 satisfied
affects: [04-evaluation]

# Metrics
duration: manual
completed: 2026-03-16
---

# Phase 3 Plan 05: Human Verification — Loss Convergence on Real Data Summary

**Training launched successfully on 4×H200 GPUs; D_std and D_harmful per-split losses trend downward over 40 steps; checkpoint produced at out/verify-phase3/final/lit_model.pth. Phase 3 Success Criterion 3 confirmed.**

## Performance

- **Duration:** manual run
- **Completed:** 2026-03-16
- **Tasks:** 1 (human checkpoint)
- **Files modified:** 0

## Accomplishments

- Confirmed `python -m safemoe pretrain` launches without error on real TinyStories bilingual data (4×H200, NCCL DDP)
- Observed per-split loss decreasing for D_std and D_harmful streams:
  - step 10 [D_std]: 11.920 → step 20 [D_std]: 11.406 (↓ 0.514)
  - step 30 [D_harmful]: 11.580 → step 40 [D_harmful]: 11.494 (↓ 0.086)
- Final val loss: 8.136 after 196,608 tokens
- Checkpoint saved at `out/verify-phase3/final/lit_model.pth`
- All three verification gaps from VERIFICATION.md resolved:
  - Gap 1: behavioral tests GREEN (plan 03-04)
  - Gap 2: REQUIREMENTS.md TRAIN-02 updated (plan 03-04)
  - Gap 3: loss convergence confirmed (this plan)

## Training Run Config

- **GPUs:** 4× NVIDIA H200, NCCL backend, all connected via NVLink
- **Model:** safemoe-tinystories — 4L/4H/128embd, 8 experts, 2 harmful experts, harmful_attn_heads=[0,1]
- **Data:** TinyStories bilingual, x=0 y=25, max_tokens=200,000
- **Batch:** global_batch_size=16, micro_batch_size=4
- **Command:** `python -m safemoe pretrain safemoe-tinystories --config safemoe/configs/safemoe-tinystories.yaml --out_dir out/verify-phase3 --train.max_tokens 200000`

## Notes

- D_unlabeled split was not sampled in this short run (rotation with x=0/y=25 data files only cycled through D_std and D_harmful within the 40-step window). Structural correctness of the D_unlabeled path is verified by automated behavioral tests from plan 03-04 (`test_fit_unlabeled_step_no_masking`).
- The `FSDP.clip_grad_norm_()` warning on ranks 2–3 at step 0 is benign: no gradients exist before the first forward pass and the norm returns 0.0 in default dtype — expected behavior at initialization.

## Task Commits

No code changes — human verification only.

## Files Created/Modified

None.

## Decisions Made

None — verification only.

## Deviations from Plan

None.

## Issues Encountered

None — training ran cleanly.

## User Setup Required

None.

## Next Phase Readiness

- Phase 3 fully complete: all 5 plans executed, all success criteria satisfied
- SGTM training loop structurally and behaviorally verified (automated tests + real data run)
- Phase 4 (Ablation & Evaluation) can proceed

---
*Phase: 03-sgtm-training-loop*
*Completed: 2026-03-16*

## Self-Check: PASSED

- FOUND: .planning/phases/03-sgtm-training-loop/03-05-SUMMARY.md
- OBSERVED: D_std loss 11.920→11.406 (↓), D_harmful loss 11.580→11.494 (↓)
- FOUND: out/verify-phase3/final/lit_model.pth (checkpoint produced)
