---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: verifying
stopped_at: Completed 04-05-PLAN.md — Phase 4 complete, human verification of all four requirements approved
last_updated: "2026-03-16T16:27:48.815Z"
last_activity: "2026-03-16 -- Verified 03-05: loss convergence confirmed on real TinyStories data"
progress:
  total_phases: 4
  completed_phases: 4
  total_plans: 16
  completed_plans: 16
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-13)

**Core value:** Harmful knowledge must be fully containable in a designatable set of MoE parameters that can be zeroed out at inference time without degrading general model capability.
**Current focus:** Phase 4 fully complete — SafeMoE v1.0 milestone achieved; isolation thesis verified on real checkpoint

## Current Position

Phase: 4 of 4 (Ablation & Evaluation) — COMPLETE
Status: All 5 plans complete including human verification of isolation signal on real checkpoint
Last activity: 2026-03-17 -- Verified 04-05: isolation signal confirmed — D_harmful delta 118x D_std delta, routing_harmful_frac ~2x

Progress: [██████████] 100%

## Performance Metrics

**Velocity:**
- Total plans completed: 1
- Average duration: 28 min
- Total execution time: 0.47 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-data-pipeline | 1/2 completed | 28 min | 28 min |

**Recent Trend:**
- Last 5 plans: 28 min
- Trend: first data point

*Updated after each plan completion*
| Phase 01-data-pipeline P02 | 6 | 2 tasks | 2 files |
| Phase 02-model-architecture-masking P01 | 3 | 4 tasks | 5 files |
| Phase 02-model-architecture-masking P02 | 2 | 2 tasks | 2 files |
| Phase 02-model-architecture-masking P03 | 3 | 1 task | 1 file |
| Phase 02-model-architecture-masking P04 | 525668 | 2 tasks | 1 files |
| Phase 02-model-architecture-masking P04 | 8 | 2 tasks | 1 files |
| Phase 03-sgtm-training-loop P01 | 3 | 2 tasks | 2 files |
| Phase 03-sgtm-training-loop P02 | 6 | 1 tasks | 1 files |
| Phase 03-sgtm-training-loop P03 | 11 | 2 tasks | 4 files |
| Phase 03-sgtm-training-loop P04 | 25 | 2 tasks | 3 files |
| Phase 04-ablation-evaluation P03 | 14 | 3 tasks | 4 files |
| Phase 04-ablation-evaluation P02 | 15 | 2 tasks | 3 files |
| Phase 04-ablation-evaluation P01 | 8 | 3 tasks | 3 files |
| Phase 04-ablation-evaluation P04 | 20 | 1 tasks | 2 files |
| Phase 04-ablation-evaluation P05 | 30 | 2 tasks | 0 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: Phases 1 and 2 have no mutual dependency -- can be parallelized
- [Roadmap]: TRAIN-04 (ablation) grouped with EVAL phase, not training phase, because ablation is an evaluation prerequisite
- [Roadmap]: Masking grouped with architecture (not separate phase) because masking primitives are tightly coupled to HarmfulParamRegistry and SafeMoELayer
- [Revision]: Swapped Phase 1/2 ordering -- Data Pipeline is now Phase 1, Model Architecture & Masking is Phase 2
- [Revision]: Unified terminology across all planning files -- D_harmful (not "forget"), D_std (not "retain"), D_unlabeled (not "adjacent")
- [01-01]: Used start_method='fork' in litdata.optimize() — spawn silently fails with in-memory tokenizer objects
- [01-01]: Removed tests/safemoe/__init__.py — pytest namespace collision shadowed source safemoe package
- [01-01]: Added test injection kwargs to prepare() (tokenizer, en_train, es_train, en_val, es_val) for testability without Qwen3 checkpoint
- [01-01]: TokensLoader(block_size=N) required in StreamingDataset — TokensLoader() without block_size causes TypeError in ROI generation
- [Phase 01-02]: get_loader() returns DataLoader directly; training loop manages its own iter() — no next() on MultiDataLoader
- [Phase 01-02]: Module-level _tokenize_row() in tests replaces lambda — litdata spawn workers cannot pickle local closures; start_method='fork' added as safety
- [Phase 02-01]: No tests/safemoe/__init__.py to avoid pytest namespace collision (Phase 1 lesson applied)
- [Phase 02-01]: id()-based set comparisons for nn.Parameter identity in registry tests (avoids tensor __eq__ pitfall)
- [Phase 02-01]: ActivationMasker test uses _activation_masking_enabled flag checks, not per-expert output tensors
- [Phase 02-model-architecture-masking]: [02-02]: init_strategy='random'|'copy' is SafeMoELayer constructor arg — test stubs call SafeMoELayer(config, init_strategy='copy'), not a config field
- [Phase 02-model-architecture-masking]: [02-02]: Lazy import of SafeMoELayer inside SafeMoEConfig.mlp_class property body breaks circular import
- [Phase 02-03]: qkv.weight in theta_std (full parameter) + _qkv_harmful_metadata (row slices) — duality intentional for Phase 2/3 masker scope split
- [Phase 02-03]: GradientMasker(registry), ActivationMasker(model) constructor signatures match RED test stubs exactly
- [Phase 02-model-architecture-masking]: [Phase 02-04]: ActivationMasker.__init__ takes model + optional registry=None — test stubs call ActivationMasker(model) but plan shows (model, registry); registry optional for symmetry and Phase 3 extension
- [Phase 02-model-architecture-masking]: [Phase 02-04]: p.grad = None (not zero_()) in GradientMasker.mask() — None prevents AdamW exp_avg/exp_avg_sq accumulation for theta_std entirely; zero_() would still trigger Adam state creation on next step()
- [Phase 03-01]: test_attn_head_activation_masking verifies flag-state only — actual head output zeroing assertion deferred to Plan 03-03 Task 2
- [Phase 03-01]: _wrap_attn_forward installs a pass-through stub in 03-01; Plan 03-02 replaces with real head-output zeroing via SafeCausalSelfAttention
- [Phase 03-01]: ActivationMasker(model) backward-compatible; _attn_layers empty list when config is None or harmful_attn_heads is []
- [Phase 03-sgtm-training-loop]: SafeCausalSelfAttention.forward replicates parent body not super() — y tensor is (B,T,n_head,hs) post-SDPA; head zeroing at y[:, :, head_idx, :] before reshape
- [Phase 03-sgtm-training-loop]: torch.compile removed from safemoe/pretrain.py — Python bool flag _activation_masking_enabled traced and constant-folded by compiler
- [Phase 03-sgtm-training-loop]: HarmfulParamRegistry must be constructed BEFORE fabric.setup(model) — Lightning wraps model and prefixes param names with _forward_module., breaking expert regex
- [Phase 03-sgtm-training-loop]: measure_flops wrapped in try/except for MoE models — torch.where() not supported on meta device; fall back to measured_flops=0
- [Phase 03-sgtm-training-loop]: test_pretrain_produces_checkpoint catches SystemExit(2) from save_hyperparameters CLI parse — fabric.save() runs before save_hyperparameters so checkpoint IS written
- [Phase 03-sgtm-training-loop]: SafeCausalSelfAttention.forward() must clone y before in-place head-zeroing — SDPA output is part of autograd graph
- [Phase 03-sgtm-training-loop]: [03-04]: block_size=128 in test configs + max_tokens=accum_iters*micro_batch_size*block_size for exactly 1 optimizer step in behavioral tests
- [Phase 04-ablation-evaluation]: SafeMoELayer._last_indices written in both topk branches — negligible overhead, attribute unused unless hook reads it
- [Phase 04-ablation-evaluation]: evaluate_perplexity re-fetches val loaders for ablated model to handle exhausted DataLoader iterators
- [Phase 04-ablation-evaluation]: data_mock duck-typing in evaluate.py: any object with val_dataloaders() method works
- [Phase 04-ablation-evaluation]: Use torch.load() directly (not fabric.load()) for ablation — standalone checkpoint manipulation, no DDP prefix stripping needed
- [Phase 04-ablation-evaluation]: Build id_to_name map before zeroing to capture parameter names without iterating model twice during in-place ablation
- [Phase 04-01]: D_unlabeled metrics excluded from all evaluation outputs — user decision to skip D_unlabeled perplexity/routing tracking
- [Phase 04-01]: data_mock parameter on evaluate_perplexity() and routing_attribution() — testability without real data files; mirrors _MockMultiDataLoader pattern from test_pretrain.py
- [Phase 04-01]: evaluate_with_ablation() takes eval_args: EvalArgs and calls fabric.log_dict() exactly once with combined D_std and D_harmful PPL metrics
- [Phase 04-04]: evaluate_with_ablation() uses try/finally to guarantee weight restore even on exception; model.train() in finally owns eval-mode cleanup
- [Phase 04-04]: val_loaders passed as Optional dict to fit() — None skips ablation eval, preserving backward compatibility
- [Phase 04-ablation-evaluation]: EVAL-03 TensorBoard curves deferred — checkpoint pre-dates Phase 4 implementation; unit tests and wiring sufficient for requirement verification
- [Phase 04-ablation-evaluation]: Human verification confirmed isolation signal: D_harmful ppl delta (1645) is 118x D_std delta (13.87); routing_harmful_frac_D_harmful (7.35%) approx 2x D_std fraction (3.72%)

### Pending Todos

None yet.

### Blockers/Concerns

- [Research]: Load-balancing loss design for SGTM remains an open question -- standard MoE balance loss fights harmful-expert concentration
- [Research]: Spanish TinyStories data source availability needs confirmation before Phase 1 execution (NOTE: parquet files confirmed to exist on disk — blocker resolved)

## Session Continuity

Last session: 2026-03-16T16:27:48.808Z
Stopped at: Completed 04-05-PLAN.md — Phase 4 complete, human verification of all four requirements approved
Resume file: None
