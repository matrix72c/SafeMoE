---
phase: 03-sgtm-training-loop
plan: "03"
subsystem: training
tags: [cli, jsonargparse, pretrain, checkpoint, activation-masking, SafeCausalSelfAttention]

requires:
  - phase: 03-sgtm-training-loop/03-02
    provides: SafeCausalSelfAttention with head-output zeroing, safemoe/pretrain.py setup()

provides:
  - safemoe/__main__.py: python -m safemoe pretrain CLI entry point via PARSER_DATA pattern
  - Updated safemoe/configs/safemoe-tinystories.yaml with all Phase 3 SGTM fields
  - test_pretrain_produces_checkpoint GREEN: end-to-end pretrain.main() produces lit_model.pth
  - test_attn_head_activation_masking GREEN with delta forward-pass assertion
  - All 24 Phase 2+3 tests GREEN

affects:
  - 04-evaluation: CLI entry point and checkpoint format are the inference interface
  - Any future phase that runs safemoe pretrain via CLI

tech-stack:
  added: [jsonargparse>=4.47.0 (CLI entry point), docstring-parser>=0.17.0]
  patterns:
    - PARSER_DATA dict pattern for jsonargparse multi-subcommand CLI (mirrors litgpt/__main__.py)
    - HarmfulParamRegistry must be constructed BEFORE fabric.setup(model) to avoid _forward_module. prefix breaking expert regex
    - measure_flops try/except NotImplementedError for MoE models (torch.where not supported on meta device)
    - MockMultiDataLoader pattern for in-memory unit tests of pretrain.main()
    - Delta approach for activation masking assertion (compare masked vs unmasked output)

key-files:
  created:
    - safemoe/__main__.py
  modified:
    - safemoe/configs/safemoe-tinystories.yaml
    - safemoe/pretrain.py
    - tests/safemoe/test_pretrain.py

key-decisions:
  - "HarmfulParamRegistry must be constructed BEFORE fabric.setup(model) — Lightning wraps model and prefixes all param names with _forward_module., breaking HarmfulParamRegistry regex patterns"
  - "measure_flops wrapped in try/except for MoE models — torch.where() in LLaMAMoE forward is not supported on meta device used by measure_flops"
  - "test_pretrain_produces_checkpoint catches SystemExit from save_hyperparameters CLI parse — fabric.save() runs before save_hyperparameters so checkpoint IS written even when SystemExit is raised"
  - "Delta approach for activation masking test — compare output_unmasked vs output_masked; simpler and more robust than forward hook approach"

patterns-established:
  - "Pattern: PARSER_DATA = {'pretrain': pretrain_fn} for jsonargparse CLI subcommand routing"
  - "Pattern: Registry-before-setup — always build HarmfulParamRegistry before fabric.setup()"
  - "Pattern: MockMultiDataLoader for CPU-only pretrain unit tests"

requirements-completed: [TRAIN-03]

duration: 11min
completed: 2026-03-16
---

# Phase 3 Plan 03: CLI Entry Point and Test Completion Summary

**`python -m safemoe pretrain` CLI via jsonargparse PARSER_DATA pattern, complete YAML config, and full Phase 3 test suite GREEN with end-to-end checkpoint production test**

## Performance

- **Duration:** 11 min
- **Started:** 2026-03-16T05:22:32Z
- **Completed:** 2026-03-16T05:33:48Z
- **Tasks:** 2
- **Files modified:** 4 (safemoe/__main__.py created; safemoe/configs/safemoe-tinystories.yaml, safemoe/pretrain.py, tests/safemoe/test_pretrain.py modified)

## Accomplishments

- `python -m safemoe pretrain --help` exits 0, shows all setup() arguments including upsample_std/harmful/unlabeled
- `safemoe-tinystories.yaml` updated with harmful_attn_heads: [0,1] and all SGTM Phase 3 fields
- `test_pretrain_produces_checkpoint` GREEN: calls `pretrain.main()` with MockMultiDataLoader, asserts `out_dir/final/lit_model.pth` exists
- `test_attn_head_activation_masking` strengthened with delta forward-pass assertion confirming SafeCausalSelfAttention zeroes harmful head outputs
- All 24 Phase 2+3 tests GREEN (4 config + 4 masking + 4 model + 7 pretrain + 5 registry)

## Task Commits

1. **Task 1: Create safemoe/__main__.py CLI entry point + update YAML config** - `a4f5996` (feat)
2. **Task 2: Make test_pretrain_produces_checkpoint GREEN + strengthen test_attn_head_activation_masking** - `73df074` (feat)

**Plan metadata:** (see final docs commit below)

## Files Created/Modified

- `safemoe/__main__.py` - CLI entry point; PARSER_DATA = {"pretrain": pretrain_fn}; `main()` calls `CLI(PARSER_DATA)`
- `safemoe/configs/safemoe-tinystories.yaml` - Updated harmful_attn_heads: [0,1]; added upsample_std/harmful/unlabeled: 1, micro_batch_size: 4, gradient_accumulation_iters: 4
- `safemoe/pretrain.py` - Bug fix: registry before fabric.setup(); bug fix: measure_flops try/except for meta device
- `tests/safemoe/test_pretrain.py` - Full checkpoint test with MockMultiDataLoader; delta assertion in test_attn_head_activation_masking

## Decisions Made

- `HarmfulParamRegistry` constructed BEFORE `fabric.setup(model)`: Lightning wraps the model with a `_forward_module.` prefix on all parameter names, which breaks the `transformer.h.\d+.mlp.experts.\d+.` regex. Moving registry construction before `fabric.setup()` fixes this (Rule 1 auto-fix).
- `measure_flops` wrapped in try/except for MoE models: `LLaMAMoE.forward()` calls `torch.where()` which is not supported on the meta device used by `measure_flops`. Fallback to `measured_flops = 0` so `ThroughputMonitor` still works (Rule 1 auto-fix).
- Delta approach for activation masking test: comparing `output_unmasked` vs `output_masked` is simpler and more robust than attaching forward hooks to capture intermediate tensors.
- `save_hyperparameters` raises `SystemExit(2)` in tests because it calls `CLI(setup)` which requires `model_name` positional arg. Since `fabric.save()` runs before `save_hyperparameters`, the checkpoint IS written. Test catches `SystemExit` and verifies checkpoint path.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] HarmfulParamRegistry construction order in pretrain.main()**
- **Found during:** Task 2 (test_pretrain_produces_checkpoint implementation)
- **Issue:** Registry was built AFTER `fabric.setup(model)`, which wraps the model and prepends `_forward_module.` to all parameter names. This caused `HarmfulParamRegistry._EXPERT_RE` regex to match nothing, leaving `theta_harmful` empty and causing `ValueError: optimizer got an empty parameter list`.
- **Fix:** Moved `registry = HarmfulParamRegistry(model, config)` to before `model = fabric.setup(model)` in `safemoe/pretrain.py main()`.
- **Files modified:** `safemoe/pretrain.py`
- **Verification:** `len(registry.parameters_by_type('theta_harmful'))` = 12 (expected); `main()` completes without optimizer error.
- **Committed in:** `73df074` (Task 2 commit)

**2. [Rule 1 - Bug] measure_flops() incompatible with MoE model on meta device**
- **Found during:** Task 2 (test_pretrain_produces_checkpoint implementation)
- **Issue:** `fit()` calls `measure_flops()` using `torch.device("meta")`, but `LLaMAMoE.forward()` calls `torch.where(mask)` which raises `NotImplementedError` on meta device.
- **Fix:** Wrapped `with torch.device("meta"):` block and `measure_flops()` call in `try/except (NotImplementedError, RuntimeError)`, falling back to `measured_flops = 0`.
- **Files modified:** `safemoe/pretrain.py`
- **Verification:** `fit()` completes; training loop runs; final checkpoint saved.
- **Committed in:** `73df074` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 Rule 1 bugs)
**Impact on plan:** Both auto-fixes necessary for pretrain.main() to run correctly. No scope creep.

## Issues Encountered

- `save_hyperparameters(setup, checkpoint_file.parent)` in `save_checkpoint()` calls `CLI(setup)` internally, which expects `model_name` as a positional CLI argument. When called programmatically from tests (no `sys.argv`), this raises `SystemExit(2)`. Since `fabric.save()` runs before `save_hyperparameters()`, the checkpoint IS written; the test catches `SystemExit` and verifies the file path.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 3 complete: TRAIN-01 (GradientMasker), TRAIN-02 (ActivationMasker + SafeCausalSelfAttention), TRAIN-03 (CLI + checkpoint) all verified GREEN
- `python -m safemoe pretrain --config safemoe/configs/safemoe-tinystories.yaml` is the primary training entry point (requires `model_name` arg and real MultiDataLoader with disk data)
- Phase 4 (evaluation/ablation) can now load checkpoints produced by Phase 3

---
*Phase: 03-sgtm-training-loop*
*Completed: 2026-03-16*

## Self-Check: PASSED

- safemoe/__main__.py: FOUND
- 03-03-SUMMARY.md: FOUND
- safemoe/configs/safemoe-tinystories.yaml: FOUND
- Commit a4f5996: FOUND
- Commit 73df074: FOUND
- upsample_std in YAML: FOUND
- harmful_attn_heads: [0, 1] in YAML: FOUND
- PARSER_DATA in __main__.py: FOUND
