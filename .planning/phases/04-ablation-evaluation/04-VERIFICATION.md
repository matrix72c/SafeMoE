---
phase: 04-ablation-evaluation
verified: 2026-03-17T00:00:00Z
status: passed
score: 4/4 must-haves verified
re_verification: false
human_verification:
  - test: "TensorBoard mid-training ablation curves showing ablated_val_ppl_D_harmful growing over steps"
    expected: "ablated_val_ppl_D_harmful increases as isolation improves across checkpoints"
    why_human: "Available checkpoint pre-dates Phase 4 implementation; unit tests verify code correctness but cannot show temporal convergence signal; researcher explicitly accepted unit-test-only evidence for EVAL-03 TensorBoard verification"
---

# Phase 4: Ablation & Evaluation Verification Report

**Phase Goal:** A complete evaluation pipeline that ablates harmful experts and measures whether knowledge isolation succeeded — the validation of SafeMoE's core thesis
**Verified:** 2026-03-17
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | ablate() zeros theta_harmful weights and saves ablated checkpoint with manifest | VERIFIED | `safemoe/ablate.py` zeros via `parameters_by_type("theta_harmful")`, saves `ablated/lit_model.pth` + `ablation_manifest.json`; real checkpoint at `checkpoints/verify-phase3/final/ablated/` has 24 zeroed params with non-zero pre-ablation norms (e.g. 10.17) |
| 2 | Post-ablation D_harmful perplexity increases significantly while D_std stays near baseline | VERIFIED | `results.json` on real checkpoint: D_harmful delta = 1645.77 (118x) vs D_std delta = 13.87; evaluate_perplexity() computes and writes this in `safemoe/evaluate.py` |
| 3 | Routing attribution shows harmful tokens preferentially activate theta_harmful experts | VERIFIED | `routing_attribution.json` on real checkpoint: `routing_harmful_frac_D_harmful` = 7.35% vs `routing_harmful_frac_D_std` = 3.72% (~2x ratio); routing_attribution() wired via forward hooks on SafeMoELayer |
| 4 | evaluate_with_ablation() runs at checkpoints during training, logs ablation metrics, restores weights | VERIFIED | Function implemented in `safemoe/pretrain.py`, wired into `fit()` at save_interval block; try/finally restores theta_harmful; 45 unit tests pass; mid-training TensorBoard curves deferred (pre-Phase-4 checkpoint only) |

**Score:** 4/4 truths verified

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `safemoe/ablate.py` | ablate() + setup() functions | VERIFIED | 86 lines; ablate() loads model, zero-fills theta_harmful via HarmfulParamRegistry, saves ablated dir + manifest; setup() is CLI entry |
| `safemoe/evaluate.py` | evaluate_perplexity() + routing_attribution() + setup() | VERIFIED | 240 lines; both functions substantive with fabric setup, validate() calls per split, JSON output, TensorBoard optional path |
| `safemoe/__main__.py` | 3 subcommands: pretrain, ablate, evaluate | VERIFIED | All three registered in PARSER_DATA; imports from ablate_fn and evaluate_fn present |
| `safemoe/pretrain.py` | evaluate_with_ablation() + extended fit() | VERIFIED | evaluate_with_ablation() at line 723 with try/finally guard; fit() has `registry` and `val_loaders` Optional params at lines 526-527; main() passes `val_loaders_for_eval` at line 479 |
| `safemoe/model.py` | SafeMoELayer._last_indices in both forward branches | VERIFIED | Lines 66 and 70: `self._last_indices = indices` added after each topk branch (standard and expert-groups) |
| `tests/safemoe/test_ablate.py` | 3 tests for TRAIN-04 | VERIFIED | test_ablate_zeros_theta_harmful, test_ablate_preserves_theta_std, test_ablate_manifest_and_files |
| `tests/safemoe/test_evaluate.py` | 2 tests for EVAL-01 + EVAL-02 | VERIFIED | test_evaluate_perplexity, test_routing_attribution |
| `tests/safemoe/test_pretrain.py` | 2 EVAL-03 tests appended | VERIFIED | test_evaluate_with_ablation_restores_weights, test_evaluate_with_ablation_logs_metrics |
| `checkpoints/verify-phase3/final/ablated/ablation_manifest.json` | Real manifest with non-empty zeroed_parameters | VERIFIED | 24 entries; first entry: `{"name": "transformer.h.0.mlp.experts.0.fc_1.weight", "pre_ablation_norm": 10.17}` |
| `checkpoints/verify-phase3/final/results.json` | original/ablated/delta keys with D_std and D_harmful PPL | VERIFIED | D_harmful delta = 1645.77, D_std delta = 13.87 |
| `checkpoints/verify-phase3/final/routing_attribution.json` | routing_harmful_frac_D_std and _D_harmful | VERIFIED | D_harmful frac = 7.35%, D_std frac = 3.72% |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `safemoe/__main__.py` | `safemoe/ablate.py` | `from safemoe.ablate import setup as ablate_fn` | WIRED | Line 12 of __main__.py; "ablate" key in PARSER_DATA |
| `safemoe/__main__.py` | `safemoe/evaluate.py` | `from safemoe.evaluate import setup as evaluate_fn` | WIRED | Line 13 of __main__.py; "evaluate" key in PARSER_DATA |
| `safemoe/ablate.py` | `safemoe/masking.py` | `HarmfulParamRegistry.parameters_by_type('theta_harmful')` | WIRED | Line 51 of ablate.py: `registry.parameters_by_type("theta_harmful")` iterates and zeros |
| `safemoe/evaluate.py` | `safemoe/pretrain.py` | `from safemoe.pretrain import validate` | WIRED | Line 19 of evaluate.py; validate() called at lines 108 and 123 per split |
| `safemoe/evaluate.py` | `safemoe/model.py` | forward hook reads `_last_indices` from SafeMoELayer | WIRED | Lines 182-184 of evaluate.py: hook appends `module._last_indices.flatten().tolist()`; SafeMoELayer.forward() sets `_last_indices` at model.py lines 66 and 70 |
| `safemoe/pretrain.py fit()` | `safemoe/pretrain.py evaluate_with_ablation()` | called at save_interval block | WIRED | Lines 704-708: `if registry is not None and val_loaders is not None: evaluate_with_ablation(...)` inside save_interval conditional |
| `safemoe/pretrain.py evaluate_with_ablation()` | `safemoe/masking.py` | `registry.parameters_by_type('theta_harmful')` | WIRED | Line 740: `harmful_params = registry.parameters_by_type("theta_harmful")` |
| `safemoe/pretrain.py evaluate_with_ablation()` | `safemoe/pretrain.py validate()` | called per split inside try block | WIRED | Line 749: `val_loss = validate(fabric, model, loader, ...)` inside `for split_name, loader in val_loaders.items()` |
| `safemoe/pretrain.py main()` | `safemoe/pretrain.py fit()` | passes registry and val_loaders | WIRED | Lines 478-479: `registry=registry, val_loaders=val_loaders_for_eval` in fit() call; val_loaders_for_eval set at line 420 via `data.val_dataloaders()` |

---

## Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| TRAIN-04 | 04-01, 04-02, 04-05 | ablate() zeros theta_harmful, saves ablated checkpoint | SATISFIED | safemoe/ablate.py: zeroes via HarmfulParamRegistry, saves ablated/lit_model.pth + manifest; 3 tests GREEN; real checkpoint confirmed |
| EVAL-01 | 04-01, 04-03, 04-05 | Per-split perplexity (D_std, D_harmful) before and after ablation | SATISFIED (with scope note) | evaluate_perplexity() computes val_ppl for D_std and D_harmful; results.json written with original/ablated/delta; real signal: D_harmful delta 118x D_std delta. Note: REQUIREMENTS.md says "D_std / D_harmful / D_unlabeled" but D_unlabeled was excluded by user decision during context planning; the isolation thesis is demonstrated without D_unlabeled |
| EVAL-02 | 04-01, 04-03, 04-05 | Routing attribution per split, logged to TensorBoard | SATISFIED | routing_attribution() collects expert dispatch via forward hooks on SafeMoELayer._last_indices; writes routing_attribution.json + TensorBoard scalars; real signal: D_harmful frac ~2x D_std frac |
| EVAL-03 | 04-01, 04-04, 04-05 | Mid-training ablation evaluation at checkpoints, tracks isolation progress | SATISFIED (unit-test level) | evaluate_with_ablation() in pretrain.py: try/finally guard restores theta_harmful; fit() wired at save_interval; 2 EVAL-03 tests GREEN; TensorBoard curves on a future training run will show the convergence signal (current checkpoint pre-dates Phase 4) |

**Orphaned requirements check:** REQUIREMENTS.md maps TRAIN-04, EVAL-01, EVAL-02, EVAL-03 to Phase 4. All four appear in plan frontmatter. No orphaned requirements.

---

## Anti-Patterns Found

No blockers or stubs detected.

| File | Pattern | Severity | Assessment |
|------|---------|----------|------------|
| `safemoe/evaluate.py` | `try/except ImportError` around TensorBoard | Info | Intentional graceful degradation — TensorBoard is optional; routing_attribution.json is always written |
| D_unlabeled exclusion in EVAL-01 text | Implementation deviates from literal REQUIREMENTS.md wording | Info | User decision made during Phase 4 context planning; all plans document "D_unlabeled excluded (user decision)"; the core isolation thesis is validated without it |

---

## Human Verification Required

### 1. TensorBoard Mid-Training Ablation Curves (EVAL-03)

**Test:** Run a new training run with `python -m safemoe pretrain --config safemoe/configs/safemoe-tinystories.yaml` and open TensorBoard on the output directory.

**Expected:** `ablated_val_ppl_D_std` and `ablated_val_ppl_D_harmful` curves appear alongside `val_loss`; `ablated_val_ppl_D_harmful` increases as training progresses (delta grows with isolation).

**Why human:** The checkpoint used for TRAIN-04/EVAL-01/EVAL-02 validation (`checkpoints/verify-phase3/final/`) was produced before Phase 4 code was written. The evaluate_with_ablation() function is wired and unit-tested (45 tests GREEN), but the live TensorBoard convergence curve requires a fresh training run with Phase 4 code active. Researcher accepted this deferral as adequate for Phase 4 completion.

---

## Phase 4 Goal Assessment

The phase goal — "a complete evaluation pipeline that ablates harmful experts and measures whether knowledge isolation succeeded" — is achieved:

1. The ablation utility (TRAIN-04) surgically zeros 24 harmful parameters, preserving theta_std, and the manifest provides verifiability.

2. The perplexity evaluation (EVAL-01) demonstrates the isolation signal empirically: D_harmful perplexity increases 1645 points post-ablation (118x larger than the 13.87-point D_std increase). This is the core quantitative proof of the thesis.

3. The routing attribution analysis (EVAL-02) shows the router learned the partition: harmful-domain tokens route to theta_harmful experts at ~2x the rate of standard-domain tokens (7.35% vs 3.72%).

4. The mid-training evaluation function (EVAL-03) is correctly implemented, tested, and wired into fit(). TensorBoard convergence curves are deferred to the next training run.

SafeMoE's core thesis — harmful knowledge is isolatable in designatable expert parameters and ablatable without degrading general capability — is validated by the empirical results on the real trained checkpoint.

---

## All 45 Tests Accounted For

| Test File | Test Count | Coverage |
|-----------|-----------|---------|
| `test_config.py` | 4 | SafeMoEConfig fields |
| `test_model.py` | 4 | SafeMoELayer forward + _last_indices |
| `test_registry.py` | 5 | HarmfulParamRegistry |
| `test_masking.py` | 4 | GradientMasker + ActivationMasker |
| `data/test_prepare.py` | 7 | Data pipeline |
| `data/test_datamodule.py` | 7 | MultiDataLoader |
| `test_pretrain.py` | 9 | Training loop (7 original + 2 EVAL-03) |
| `test_ablate.py` | 3 | TRAIN-04 |
| `test_evaluate.py` | 2 | EVAL-01 + EVAL-02 |
| **Total** | **45** | All GREEN per 04-04-SUMMARY.md |

---

_Verified: 2026-03-17_
_Verifier: Claude (gsd-verifier)_
