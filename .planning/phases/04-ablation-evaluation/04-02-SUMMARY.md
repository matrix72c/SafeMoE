---
phase: 04-ablation-evaluation
plan: "02"
subsystem: ablation
tags: [pytorch, safemoe, checkpoint, ablation, masking, HarmfulParamRegistry]

# Dependency graph
requires:
  - phase: 04-01
    provides: "HarmfulParamRegistry.parameters_by_type('theta_harmful') tested and proven"
  - phase: 03-sgtm-training-loop
    provides: "lit_model.pth checkpoint format + model_config.yaml convention"
provides:
  - "safemoe/ablate.py: ablate() + setup() for surgical theta_harmful weight removal"
  - "ablated/ checkpoint directory with lit_model.pth + model_config.yaml + ablation_manifest.json"
  - "'python -m safemoe ablate <ckpt_dir>' CLI subcommand"
affects:
  - "04-03"
  - "04-05"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Standalone checkpoint manipulation via torch.load() — no fabric.launch() or DDP wrapping"
    - "id()-based param-to-name mapping before in-place zeroing"
    - "Ablation manifest JSON for verifiability of which weights were zeroed"

key-files:
  created:
    - safemoe/ablate.py
    - tests/safemoe/test_ablate.py
  modified:
    - safemoe/__main__.py

key-decisions:
  - "Use torch.load() directly (not fabric.load()) for ablation — no DDP prefix stripping needed on standalone checkpoint"
  - "Build id_to_name map before zeroing to capture parameter names without iterating model twice"
  - "Task 2 (__main__.py wiring) already landed via parallel plan 04-03 — no re-commit needed"

patterns-established:
  - "Ablation pattern: load → registry → id_to_name map → zero in-place → save → manifest"
  - "Model config loading: strip nested dict values from YAML before SafeMoEConfig(**raw)"

requirements-completed: [TRAIN-04]

# Metrics
duration: 15min
completed: 2026-03-16
---

# Phase 4 Plan 02: Ablation Utility (TRAIN-04) Summary

**Standalone ablation CLI that loads a SafeMoE checkpoint, zeros all theta_harmful weights via HarmfulParamRegistry, and saves an ablated copy with a JSON manifest listing every zeroed parameter's pre-ablation norm**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-03-16T14:48:09Z
- **Completed:** 2026-03-16T15:03:05Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- `safemoe/ablate.py` implements `ablate()` + `setup()` with full TDD (RED → GREEN)
- All 3 test assertions pass: theta_harmful norms = 0.0, theta_std unchanged, manifest + files correct
- `python -m safemoe ablate <ckpt_dir>` subcommand available via PARSER_DATA dispatch

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests for ablation utility** - `8478f48` (test)
2. **Task 1 GREEN: Implement safemoe/ablate.py** - `edd88d2` (feat)
3. **Task 2: Wire ablate subcommand into __main__.py** - already committed by parallel plan `4ed4e3b` (feat(04-03))

_Note: Task 2 __main__.py update was already applied by parallel plan 04-03 before this plan ran. No duplicate commit needed._

## Files Created/Modified
- `safemoe/ablate.py` — ablate() function: load config/model/weights, zero theta_harmful via HarmfulParamRegistry, save ablated/ dir + manifest; setup() CLI entry point
- `tests/safemoe/test_ablate.py` — 3 TDD tests: zeros_theta_harmful, preserves_theta_std, manifest_and_files
- `safemoe/__main__.py` — already updated by 04-03 to include ablate_fn in PARSER_DATA

## Decisions Made
- `torch.load()` used directly instead of `fabric.load()` — standalone checkpoint manipulation (no DDP wrapping, no prefix stripping needed per plan's Pitfall 2 note)
- `id_to_name = {id(p): n for n, p in model.named_parameters()}` built before zeroing — captures parameter identity-to-name mapping before in-place mutation
- `SafeMoEConfig(**{k: v for k, v in raw.items() if not isinstance(v, dict)})` — strips nested litgpt sub-keys that SafeMoEConfig doesn't accept

## Deviations from Plan

None - plan executed exactly as written.

The only notable observation: `safemoe/__main__.py` was already updated by parallel plan 04-03 to include both `ablate_fn` and `evaluate_fn`. My task 2 content was already applied; no re-commit was needed.

## Issues Encountered

**Pre-existing regression in test_pretrain_produces_checkpoint (out of scope):** Uncommitted changes to `safemoe/pretrain.py` (from parallel plan work) added `data.val_dataloaders()` call, but the `_MockMultiDataLoader` in `test_pretrain.py` only has `val_dataloader()`. This failure is caused by uncommitted changes in the working tree from a parallel plan — not caused by any changes in this plan. Logged to deferred items.

## Next Phase Readiness
- `safemoe/ablate.py` is complete and tested — ready for use by EVAL-01/EVAL-02 evaluate CLI
- `python -m safemoe ablate <ckpt_dir>` works end-to-end
- Ablated checkpoints are compatible with the same loading pattern used in `safemoe/evaluate.py`

---
*Phase: 04-ablation-evaluation*
*Completed: 2026-03-16*
