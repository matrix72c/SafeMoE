# SafeMoE

## What This Is

A research framework built on LitGPT that implements SGTM (Selective Gradient/Token Masking) for Mixture-of-Experts models. `v1.0` shipped the full PT-phase validation stack: bilingual proxy data preparation, SafeMoE architecture and masking, SGTM pretraining, checkpoint ablation, and evaluation that verifies harmful knowledge can be isolated into designated experts and removed with limited standard-domain degradation.

The project continues in milestone-validated steps: CPT-phase routing analysis on a pretrained model, then CPT-phase knowledge transfer using unlabeled data.

## Core Value

Harmful knowledge must be fully containable in a designatable set of MoE experts that can be zeroed out at inference time without degrading general model capability.

## Current State

- **Shipped:** `v1.0 PT Phase Validation` on `2026-03-17`
- **Delivered:** 4 phases, 16 plans, 32 tracked tasks
- **Validated result:** real-checkpoint evaluation showed `D_harmful` perplexity delta `1645.77` versus `D_std` delta `13.87` after ablation, with harmful routing fraction `7.35%` versus `3.72%` on `D_std`
- **Audit status:** `tech_debt` in `.planning/milestones/v1.0-MILESTONE-AUDIT.md`
- **Current focus:** define the next milestone around CPT routing validation

## Requirements

### Validated

- ✓ TinyStories bilingual data pipeline with D_std / D_harmful / D_unlabeled partitioning and per-split loaders — `v1.0`
- ✓ SafeMoEConfig, SafeMoELayer, HarmfulParamRegistry, and masking primitives — `v1.0`
- ✓ SGTM training loop with split-aware branching and CLI entry point — `v1.0`
- ✓ Expert ablation utility for `theta_harmful` checkpoint removal — `v1.0`
- ✓ Per-split perplexity and routing attribution evaluation — `v1.0`
- ✓ Real-checkpoint isolation validation for the PT-phase thesis — `v1.0`

### Active

- [ ] Inject `theta_harmful` experts into a pretrained LitGPT checkpoint
- [ ] Run CPT training with router logging on a real harmful-domain dataset
- [ ] Verify harmful-domain tokens preferentially route to `theta_harmful`
- [ ] Measure whether harmful knowledge transfers from `theta_std` into `theta_harmful` during CPT
- [ ] Define residual-harm metrics before and after ablation in the CPT setting

### Out of Scope

- Real-time serving / deployment of SafeMoE models
- Multi-node distributed training beyond research-scale experiments
- Novel tokenizer architecture or unrelated transformer redesign

## Context

**Foundation:** SafeMoE extends LitGPT internals directly rather than introducing a separate training framework.

**Shipped v1.0 path:** TinyStories bilingual proxy data -> SafeMoE architecture and masking -> SGTM training loop -> ablation and evaluation pipeline -> real-checkpoint isolation verification.

**Codebase snapshot:** roughly `15.8k` lines of Python across `safemoe/` and `tests/` at milestone completion.

**Known debt:** Phase validation files remain draft, Phase 3 verification text is stale relative to later fixes, and live EVAL-03 TensorBoard convergence curves were deferred to a future fresh run.

## Next Milestone Goals

- Define the CPT routing-validation milestone scope and requirements
- Select the pretrained checkpoint and harmful-domain dataset for CPT experiments
- Design the expert-injection path and router instrumentation
- Preserve the v1.0 isolation measurements as the baseline for later transfer studies

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Build MoE from LitGPT components instead of adopting an external MoE stack | Full control over expert designation, routing hooks, and masking behavior | ✓ Good |
| Use TinyStories bilingual data as the v1.0 harmful proxy | Clean ground truth for isolation without real harmful content | ✓ Good |
| Validate in milestones: PT-phase first, then CPT routing, then transfer | Each stage has a separable thesis and verification burden | ✓ Good |
| Exclude `D_unlabeled` from v1.0 evaluation outputs | Keep the isolation readout focused on the core D_std vs D_harmful thesis | ✓ Good |
| Accept deferred live EVAL-03 curves once unit tests and real checkpoint evidence passed | Preserved milestone momentum without blocking on a fresh retraining run | ⚠ Revisit |

---
*Last updated: 2026-03-17 after v1.0 milestone completion*
