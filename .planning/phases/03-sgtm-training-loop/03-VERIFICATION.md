---
phase: 03-sgtm-training-loop
verified: 2026-03-16T06:30:00Z
status: gaps_found
score: 7/10 must-haves verified
gaps:
  - truth: "All pretrain-loop unit tests in test_pretrain.py PASS GREEN (substantive behavioral assertions)"
    status: failed
    reason: "4 of 7 test functions (test_fit_harmful_step_masks_theta_std, test_fit_std_step_enables_activation_masker, test_fit_unlabeled_step_no_masking, test_masker_called_once_per_step) only assert that safemoe.pretrain is importable. Their docstrings claim behavioral coverage (grad=None after D_harmful step; masker enabled/disabled around D_std window; no masking on D_unlabeled; mask() called once with accum_iters=2) but the test bodies contain only an import check. These behaviors are implemented in pretrain.py but are UNTESTED by any automated assertion."
    artifacts:
      - path: "tests/safemoe/test_pretrain.py"
        issue: "Lines 77-83, 91-97, 105-111, 119-125: test bodies are import-only stubs. The behavioral assertions described in docstrings (grad=None, masker flag state, call-count with accum) are absent."
    missing:
      - "test_fit_harmful_step_masks_theta_std: add assertion that after one D_harmful step all theta_std params have grad=None"
      - "test_fit_std_step_enables_activation_masker: add assertion that activation_masker.enable() is called before forward and disable() is called after (mock or spy on ActivationMasker)"
      - "test_fit_unlabeled_step_no_masking: add assertion that no masker methods are called and both optimizers step"
      - "test_masker_called_once_per_step: add assertion that gradient_masker.mask() is called exactly once when gradient_accumulation_iters=2"

  - truth: "TRAIN-02 requirement text is satisfied: sgtm_mode scalar in batch dict + adjust_gradients() per backward"
    status: failed
    reason: "REQUIREMENTS.md TRAIN-02 describes 'sgtm_mode scalar passed as part of batch dict to model forward; adjust_gradients(sgtm_mode) called after each backward pass'. The implementation instead uses split_label string + direct masker calls (gradient_masker.mask(), activation_masker.enable/disable). No sgtm_mode field exists in any batch dict and no adjust_gradients() function exists anywhere. This is a deliberate design evolution captured in CONTEXT.md but the REQUIREMENTS.md text was never updated to reflect the changed interface."
    artifacts:
      - path: ".planning/REQUIREMENTS.md"
        issue: "TRAIN-02 description (line 33) still says 'sgtm_mode scalar passed as part of batch dict to model forward; adjust_gradients(sgtm_mode) called after each backward pass' — the actual implementation uses split_label + direct masker dispatch, which is a different interface contract."
    missing:
      - "Update REQUIREMENTS.md TRAIN-02 description to match the implemented interface: split_label string sampling via random.choices; three-path if/elif/else dispatch calling gradient_masker.mask() for D_harmful, activation_masker.enable/disable for D_std, no-op for D_unlabeled"

  - truth: "Success Criterion 3: Training loss decreases over steps for all three split types"
    status: failed
    reason: "No automated test verifies that loss decreases over multiple steps. test_pretrain_produces_checkpoint confirms a checkpoint is produced but does not assert loss convergence trend. This criterion requires a multi-step training run with loss tracking — not achievable with the current single-step import-check or checkpoint-existence test."
    artifacts:
      - path: "tests/safemoe/test_pretrain.py"
        issue: "No test asserts loss_D_std, loss_D_harmful, or loss_D_unlabeled trends downward over several optimizer steps."
    missing:
      - "Add a test that runs pretrain.main() for at least 10 optimizer steps with a tiny model and asserts that mean loss over the last 5 steps is lower than mean loss over the first 5 steps for at least one split, OR mark this criterion as NEEDS HUMAN (real training run required)"
human_verification:
  - test: "Run python -m safemoe pretrain --config safemoe/configs/safemoe-tinystories.yaml --model_name safemoe-tinystories with a real MultiDataLoader pointed at prepared data"
    expected: "Training launches, per-split loss (loss_D_std, loss_D_harmful, loss_D_unlabeled) is logged, loss decreases monotonically across all three streams over time, and a checkpoint file is produced at save intervals"
    why_human: "Success Criterion 3 (loss decreases) requires a real multi-step training run with prepared on-disk data. The checkpoint test uses synthetic data for only a handful of steps and does not assert convergence."
---

# Phase 3: SGTM Training Loop Verification Report

**Phase Goal:** An end-to-end SGTM pretraining script that consumes the three-split data, applies the correct masking path per split label, and produces a trained SafeMoE checkpoint
**Verified:** 2026-03-16T06:30:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `safemoe/pretrain.py` exists as fork of `litgpt/pretrain.py` with dual-optimizer SGTM branching | VERIFIED | File exists at 830 lines; `SPLIT_LABELS`, dual AdamW, `if/elif/else` 3-path present |
| 2 | D_harmful step calls `gradient_masker.mask()` once at accumulation boundary; only `optimizer_harmful` steps | VERIFIED | Lines 605-610: `gradient_masker.mask()` called once after inner loop, inside `elif split_label == "D_harmful"` block, before `optimizer_harmful.step()` |
| 3 | D_std step wraps micro-batch window in `try/finally` with activation masker; only `optimizer_std` steps | VERIFIED | Lines 572-601: `activation_masker.enable()` before try-block, `activation_masker.disable()` in `finally`, `optimizer_std.step()` in D_std branch |
| 4 | D_unlabeled step: no masking, both optimizers step | VERIFIED | Lines 611-618: `else` branch steps both `optimizer_harmful` and `optimizer_std`, no masker calls |
| 5 | Split sampling via `random.choices` with upsample weights | VERIFIED | Line 560: `split_label = random.choices(SPLIT_LABELS, weights=weights, k=1)[0]` |
| 6 | Dual optimizer state dict persisted for resume | VERIFIED | Lines 410-417: `state = {"model": ..., "optimizer_harmful": ..., "optimizer_std": ..., "iter_num": 0, "step_count": 0, "split_label": "D_std"}` |
| 7 | `SafeCausalSelfAttention.forward` zeros `y[:, :, head_idx, :]` for harmful heads when flag is True | VERIFIED | Lines 163-165: `if self._activation_masking_enabled: for head_idx in self._harmful_heads: y[:, :, head_idx, :] = 0` before `y.reshape(...)` |
| 8 | All pretrain-loop unit tests have substantive behavioral assertions (not just import checks) | FAILED | `test_fit_harmful_step_masks_theta_std`, `test_fit_std_step_enables_activation_masker`, `test_fit_unlabeled_step_no_masking`, `test_masker_called_once_per_step` contain only `import safemoe.pretrain` — no behavioral assertion |
| 9 | TRAIN-02 requirement text matches the implemented interface | FAILED | REQUIREMENTS.md TRAIN-02 still describes `sgtm_mode` scalar + `adjust_gradients()` interface; implementation uses `split_label` string + direct masker dispatch; no `sgtm_mode` or `adjust_gradients` exist anywhere in the codebase |
| 10 | Success Criterion 3: training loss decreases over steps for all three split types | FAILED | No test covers multi-step loss convergence; requires human verification with real training run |

**Score:** 7/10 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tests/safemoe/test_pretrain.py` | 7 test functions covering TRAIN-01/02/03 behaviors | PARTIAL | 7 functions exist; 2 are substantive (`test_attn_head_gradient_masking`, `test_attn_head_activation_masking`, `test_pretrain_produces_checkpoint` = 3 real tests); 4 are import-only stubs |
| `safemoe/masking.py` | Extended GradientMasker + ActivationMasker; `_qkv_harmful_metadata`, `_qkv_param_ids`, `_attn_layers` | VERIFIED | `_qkv_harmful_metadata` (line 132), `_qkv_param_ids` (line 201), `_attn_layers` (line 277), two-pass `mask()` (lines 219-226), `enable/disable` on `_attn_layers` (lines 296, 309) |
| `safemoe/pretrain.py` | SGTM training loop; exports `setup`, `main`, `fit`, `validate`; min 400 lines; dual AdamW | VERIFIED | 830 lines; all four functions present; `SafeCausalSelfAttention` subclass; `# SGTM:` comment blocks throughout |
| `safemoe/__main__.py` | CLI entry point; `PARSER_DATA = {"pretrain": pretrain_fn}`; imports `safemoe.pretrain` | VERIFIED | `PARSER_DATA = {"pretrain": pretrain_fn}` (line 11); `from safemoe.pretrain import setup as pretrain_fn` (line 9); `main()` calls `CLI(PARSER_DATA)` |
| `safemoe/configs/safemoe-tinystories.yaml` | Contains `upsample_std`, `upsample_harmful`, `upsample_unlabeled`, `harmful_attn_heads: [0, 1]`, `micro_batch_size`, `gradient_accumulation_iters` | VERIFIED | All six required fields present (lines 23-29); `harmful_attn_heads: [0, 1]` (line 23) |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `fit()` | `GradientMasker.mask()` | Called once at accumulation boundary after D_harmful window | WIRED | `gradient_masker.mask()` at line 606, outside inner loop, inside `elif split_label == "D_harmful"` |
| `fit()` | `ActivationMasker.enable/disable` | `try/finally` bracket around D_std micro-batch window | WIRED | `enable()` at line 573 before try-block; `disable()` at line 595 in `finally` |
| `main()` | `HarmfulParamRegistry` | `registry = HarmfulParamRegistry(model, config)` before `fabric.setup(model)` | WIRED | Line 385: `registry = HarmfulParamRegistry(model, config)` |
| `fit()` | `random.choices` | Split label sampled once per optimizer step from `SPLIT_LABELS` with upsample weights | WIRED | Line 560: `split_label = random.choices(SPLIT_LABELS, weights=weights, k=1)[0]` |
| `SafeCausalSelfAttention._activation_masking_enabled` | `attn_out` zeroing | `SafeCausalSelfAttention.forward` checks flag, zeros `y[:, :, head_idx, :]` before reshape+proj | WIRED | Lines 163-165 |
| `python -m safemoe pretrain` | `safemoe/__main__.py main()` | `PARSER_DATA = {"pretrain": pretrain_fn}`; `CLI(PARSER_DATA)` | WIRED | `PARSER_DATA` at line 11; `CLI(PARSER_DATA)` at line 15 |
| `safemoe/__main__.py` | `safemoe.pretrain.setup` | `from safemoe.pretrain import setup as pretrain_fn` | WIRED | Line 9 |

---

### Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| TRAIN-01 | 03-01, 03-02 | Fork `litgpt/pretrain.py` with SGTM 3-path branching per step label | SATISFIED | `safemoe/pretrain.py` implements `D_harmful → gradient masking`, `D_std → activation masking`, `D_unlabeled → standard`. Behavioral tests for `mask()` call site and optimizer isolation exist in `test_attn_head_gradient_masking` and `test_pretrain_produces_checkpoint`. **Caveat:** 4 loop-level tests are import-only stubs. |
| TRAIN-02 | 03-01, 03-02 | Masking mode per backward pass (original spec: `sgtm_mode` scalar + `adjust_gradients()`) | PARTIAL | Implementation delivers the correct behavioral outcome (masking applied per split) using a different interface (`split_label` + direct masker dispatch). REQUIREMENTS.md text still describes the superseded `sgtm_mode`/`adjust_gradients` interface — creating a textual mismatch. `test_attn_head_activation_masking` with delta assertion confirms `SafeCausalSelfAttention` zeroing works. |
| TRAIN-03 | 03-03 | CLI entry point `python -m safemoe pretrain` with YAML config support | SATISFIED | `safemoe/__main__.py` with `PARSER_DATA = {"pretrain": pretrain_fn}` wired to `jsonargparse.CLI`; YAML contains all required fields |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `tests/safemoe/test_pretrain.py` | 77-125 | 4 test bodies reduced to `import safemoe.pretrain` only — described behaviors never asserted | Warning | Loop-level behavioral invariants (grad=None after D_harmful, masker flag lifecycle, call-count invariant) are undocumented by tests; regressions in pretrain.py would pass undetected |
| `.planning/REQUIREMENTS.md` | 33 | TRAIN-02 description references `sgtm_mode` and `adjust_gradients()` which do not exist in the codebase | Info | Requirements document does not match the actual implementation contract; creates traceability confusion for reviewers |

---

### Human Verification Required

#### 1. Loss Convergence Across All Three Split Types

**Test:** Run `python -m safemoe pretrain` with a real `MultiDataLoader` on the prepared TinyStories bilingual data for at least 100 optimizer steps. Observe per-split loss logged as `loss_D_std`, `loss_D_harmful`, `loss_D_unlabeled`.
**Expected:** All three per-split losses trend downward, confirming the model learns from each data stream without masking errors stalling optimization for any split.
**Why human:** Success Criterion 3 requires real multi-step training on real data. The checkpoint test uses synthetic in-memory data for only a handful of steps and does not assert loss values.

---

### Gaps Summary

**Three gaps block full goal verification:**

**Gap 1 — Behavioral test stubs** (most significant): Four test functions named for critical training-loop invariants (`test_fit_harmful_step_masks_theta_std`, `test_fit_std_step_enables_activation_masker`, `test_fit_unlabeled_step_no_masking`, `test_masker_called_once_per_step`) contain only an import assertion. The Phase 3 plans identified these as TDD RED stubs that would become GREEN in Plan 03-02, but the 03-02 and 03-03 plans pivoted to verifying import-ability rather than upgrading the bodies to real behavioral assertions. The implementations in `pretrain.py` are correct (verified by code reading), but no automated regression protection exists for the loop-level invariants.

**Gap 2 — TRAIN-02 requirements text** (documentation): The REQUIREMENTS.md description of TRAIN-02 refers to a `sgtm_mode` scalar and `adjust_gradients()` function that were deliberately not implemented. The implementation (split_label + direct masker dispatch) achieves the same isolation outcome via a cleaner interface. The REQUIREMENTS.md text was never updated to reflect this design evolution. This is a documentation gap, not a code defect.

**Gap 3 — Loss convergence** (human-only): Success Criterion 3 cannot be verified without a real training run on prepared disk data.

---

_Verified: 2026-03-16T06:30:00Z_
_Verifier: Claude (gsd-verifier)_
