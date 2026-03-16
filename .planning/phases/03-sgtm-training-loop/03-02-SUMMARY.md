---
phase: 03-sgtm-training-loop
plan: "02"
subsystem: training
tags: [pytorch, lightning-fabric, dual-optimizer, sgtm, gradient-masking, activation-masking, attention-heads, pretrain, adamw]

# Dependency graph
requires:
  - phase: 02-model-architecture-masking
    provides: HarmfulParamRegistry, GradientMasker, ActivationMasker, SafeMoELayer
  - phase: 03-sgtm-training-loop
    plan: "01"
    provides: test stubs in test_pretrain.py, GradientMasker two-pass mask(), ActivationMasker._attn_layers
provides:
  - "safemoe/pretrain.py — SGTM fork of litgpt/pretrain.py with dual AdamW, 3-path branching, split sampling"
  - "SafeCausalSelfAttention subclass with head-output zeroing before reshape+proj (TRAIN-02)"
  - "fit() with try/finally activation masker, gradient_masker.mask() once per step, dual LR schedule"
  - "State dict with optimizer_harmful + optimizer_std keys for checkpoint resume"
affects: [03-03-PLAN, 04-eval-ablation]

# Tech tracking
tech-stack:
  added: [random (stdlib)]
  patterns:
    - "3-path if/elif/else branching in fit() for D_std/D_harmful/D_unlabeled split labels"
    - "try/finally around D_std micro-batch window prevents activation masker stuck True on exception"
    - "gradient_masker.mask() called once at accumulation boundary — after all micro-batch backwards on D_harmful step, not per micro-batch"
    - "SafeCausalSelfAttention: replicate parent forward body so y (B,T,n_head,hs) is interceptable before reshape+proj; head zeroing at y[:, :, head_idx, :] = 0"
    - "CausalSelfAttention replacement loop using __new__ + __dict__.update — preserves all module state while swapping class"
    - "random.seed(seed + iter_num) on resume to restore split-sampling reproducibility (Pitfall 3)"

key-files:
  created:
    - safemoe/pretrain.py
  modified: []

key-decisions:
  - "SafeCausalSelfAttention.forward replicates parent forward body (not calls super) so y (B,T,n_head,hs) is accessible before reshape — calling super().forward() returns the final output too late for head zeroing"
  - "y[:, :, head_idx, :] = 0 (not y[:, head_idx, :, :]) — scaled_dot_product_attention returns y.transpose(1,2) giving (B,T,n_head,hs) layout, head axis is dim=2"
  - "torch.compile(model) removed — Python bool flag checks (_activation_masking_enabled) are traced through by torch.compile, making dynamic flag toggles ineffective at runtime"
  - "val_dataloader() returns list; use first entry (D_std val) for scalar val_loss in fit()"
  - "iter_num counts micro-batches; outer while loop = one optimizer step per iteration"

patterns-established:
  - "SafeCausalSelfAttention head zeroing via class replacement before ActivationMasker construction — ensures isinstance(m, CausalSelfAttention) check in masker still passes"

requirements-completed: [TRAIN-01, TRAIN-02]

# Metrics
duration: 6min
completed: 2026-03-16
---

# Phase 3 Plan 02: SGTM Pretrain Fork Summary

**safemoe/pretrain.py forked from litgpt/pretrain.py with dual AdamW optimizers, 3-path SGTM accumulation loop, and SafeCausalSelfAttention head-output zeroing for TRAIN-01/02**

## Performance

- **Duration:** ~6 min
- **Started:** 2026-03-16T05:13:15Z
- **Completed:** 2026-03-16T05:19:03Z
- **Tasks:** 1 (TDD — RED was pre-existing from Plan 03-01, GREEN implemented here)
- **Files modified:** 1

## Accomplishments

- Created `safemoe/pretrain.py` (821 lines) as an auditable fork of `litgpt/pretrain.py` with `# SGTM: <reason>` comment blocks at every divergence point
- `SafeCausalSelfAttention` subclass implements actual head-output zeroing (`y[:, :, head_idx, :] = 0`) before reshape+proj — completing the TRAIN-02 deferred implementation from Plan 03-01
- Dual optimizer setup (`optimizer_harmful` + `optimizer_std`) from `HarmfulParamRegistry.parameters_by_type()` with `fabric.setup_optimizers()` — both persisted in state dict for checkpoint resume
- `fit()` 3-path branching with `try/finally` around D_std activation masking window; `gradient_masker.mask()` called exactly once per optimizer step at the accumulation boundary
- All 4 RED pretrain stubs from Plan 03-01 now GREEN; all 24 safemoe tests pass (no regressions)

## Task Commits

1. **Task 1: Fork litgpt/pretrain.py into safemoe/pretrain.py with SGTM dual optimizer** - `962228b` (feat)

## Files Created/Modified

- `safemoe/pretrain.py` - SGTM fork of litgpt/pretrain.py; SafeCausalSelfAttention, dual optimizer, 3-path fit(), split sampling, state dict with dual optimizer keys

## Decisions Made

- `SafeCausalSelfAttention.forward` replicates the parent forward body rather than calling `super().forward()`, because `scaled_dot_product_attention` returns `y.transpose(1, 2)` giving `(B, T, n_head, hs)` shape — the head zeroing must happen at `y[:, :, head_idx, :] = 0` (dim-2 head axis) before `y.reshape(B, T, head_size * n_head)`
- `torch.compile(model)` removed as specified by RESEARCH.md anti-pattern — Python bool flag checks used by `_activation_masking_enabled` are traced and constant-folded by the compiler
- `val_dataloader()` on `MultiDataLoader` returns a list; the first entry (D_std val) is used for scalar val_loss tracking in `fit()`
- `iter_num` counts individual micro-batches; the outer `while` loop represents one complete optimizer step encompassing `accum_iters` micro-batches

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- `tests/safemoe/data/test_datamodule.py` fails to collect due to `ModuleNotFoundError: No module named 'litdata'` — this is a pre-existing environment issue (litdata not installed), not caused by this plan's changes. Confirmed by checking that `--ignore=tests/safemoe/data/` shows all 24 remaining tests passing.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 03-03 can now test the full `fit()` loop end-to-end using the pretrain module; the `test_pretrain_produces_checkpoint` stub and the stronger head-output zeroing assertion deferred in 03-01 are ready for 03-03 implementation
- `SafeCausalSelfAttention` head zeroing is operational — `ActivationMasker.enable()` toggles `_activation_masking_enabled` on the replaced instances via the existing `_wrap_attn_forward` stub's flag, and `SafeCausalSelfAttention.forward` reads it directly (no monkey-patching needed on the new subclass)

---
*Phase: 03-sgtm-training-loop*
*Completed: 2026-03-16*

## Self-Check: PASSED

- safemoe/pretrain.py: FOUND
- 03-02-SUMMARY.md: FOUND
- Commit 962228b: FOUND
