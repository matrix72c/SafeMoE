# SafeMoE

## What This Is

A research framework for testing whether harmful knowledge in Mixture-of-Experts models can be isolated into designated experts and removed with limited collateral damage. `v1.0` validated the PT-phase thesis in the LitGPT-based SafeMoE stack, and the current milestone moves to direct `Qwen3-30B-A3B-Base` intervention with expert/head cloning, routing-supervised warmup, and CPT-style knowledge transfer.

The project continues in milestone-validated steps: CPT-phase routing analysis on a pretrained model, then CPT-phase knowledge transfer using unlabeled data.

## Core Value

Harmful knowledge must be fully containable in a designatable set of MoE experts that can be zeroed out at inference time without degrading general model capability.

## Current Milestone: v1.1 Qwen Harmful Transfer

**Goal:** Validate direct harmful-expert initialization, routing-supervised warmup, and SGTM knowledge transfer on `Qwen3-30B-A3B-Base` using bilingual TinyStories splits.

**Target features:**
- Clone `k` experts and `n` attention heads with small noise to initialize `theta_harmful` while preserving the original parameters as `theta_std`
- Add a routing loss that pushes `D_harmful` tokens toward `theta_harmful` and `D_std` tokens away during warmup
- Run mixed-data knowledge transfer with `D_unlabeled`, `D_harmful`, and `D_std` and measure whether harmful capability concentrates into `theta_harmful`

## Requirements

### Validated

- ✓ TinyStories bilingual data pipeline with D_std / D_harmful / D_unlabeled partitioning and per-split loaders — `v1.0`
- ✓ SafeMoEConfig, SafeMoELayer, HarmfulParamRegistry, and masking primitives — `v1.0`
- ✓ SGTM training loop with split-aware branching and CLI entry point — `v1.0`
- ✓ Expert ablation utility for `theta_harmful` checkpoint removal — `v1.0`
- ✓ Per-split perplexity and routing attribution evaluation — `v1.0`
- ✓ Real-checkpoint isolation validation for the PT-phase thesis — `v1.0`

### Active

- [ ] Initialize `theta_harmful` in `Qwen3-30B-A3B-Base` by cloning selected experts and attention heads, copying routing columns, and adding controlled noise
- [ ] Train a warmup stage on mixed `D_harmful` and `D_std` with an explicit routing loss that separates harmful and standard token flow
- [ ] Verify unlabeled harmful data naturally routes to harmful experts while `D_std` continues routing to standard experts
- [ ] Run SGTM-style knowledge transfer on mixed `D_unlabeled`, `D_harmful`, and `D_std` to concentrate harmful knowledge into `theta_harmful`
- [ ] Evaluate whether adversarial finetuning cost rises materially after harmful knowledge is isolated and ablated

### Out of Scope

- Real-time serving / deployment of SafeMoE models
- Multi-node distributed training beyond research-scale experiments
- Novel tokenizer architecture or unrelated transformer redesign
- Porting the milestone back to a smaller LitGPT prototype before validating the direct `Qwen3-30B-A3B-Base` path

## Context

**Foundation:** `v1.0` extended LitGPT internals directly to validate the PT-phase isolation thesis. `v1.1` now needs a direct intervention path into `Qwen3-30B-A3B-Base`, including expert/head cloning, router-column duplication, and Qwen-specific training instrumentation.

**Shipped v1.0 path:** TinyStories bilingual proxy data -> SafeMoE architecture and masking -> SGTM training loop -> ablation and evaluation pipeline -> real-checkpoint isolation verification.

**Milestone v1.1 experiment outline:** initialize `theta_harmful` from randomly selected experts and attention heads with small perturbations, warm up using a supervised routing margin loss on mixed `D_harmful`/`D_std`, then perform SGTM knowledge transfer with `D_unlabeled` plus partial labeled data.

**Target stack shift:** this milestone targets `Qwen3-30B-A3B-Base` directly rather than a smaller prototype, so compatibility, memory budget, checkpoint surgery, and routing observability matter more than in `v1.0`.

**Known debt:** Phase validation files remain draft, Phase 3 verification text is stale relative to later fixes, and live EVAL-03 TensorBoard convergence curves were deferred to a future fresh run.

## Constraints

- **Model**: `Qwen3-30B-A3B-Base` — the milestone must validate the direct large-model path rather than a proxy architecture
- **Data**: bilingual TinyStories splits remain the controlled benchmark — preserves comparability with `v1.0` while testing CPT-style transfer behavior
- **Verification**: routing separation and transfer claims must be measured empirically — the NTP loss alone is insufficient because `theta_std` already contains harmful knowledge
- **Baseline continuity**: `v1.0` ablation and routing metrics remain the comparison baseline — later claims must quantify improvement or regression against them

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Build MoE from LitGPT components instead of adopting an external MoE stack | Full control over expert designation, routing hooks, and masking behavior | ✓ Good |
| Use TinyStories bilingual data as the v1.0 harmful proxy | Clean ground truth for isolation without real harmful content | ✓ Good |
| Validate in milestones: PT-phase first, then CPT routing, then transfer | Each stage has a separable thesis and verification burden | ✓ Good |
| Exclude `D_unlabeled` from v1.0 evaluation outputs | Keep the isolation readout focused on the core D_std vs D_harmful thesis | ✓ Good |
| Accept deferred live EVAL-03 curves once unit tests and real checkpoint evidence passed | Preserved milestone momentum without blocking on a fresh retraining run | ⚠ Revisit |
| Target `Qwen3-30B-A3B-Base` directly for v1.1 | The next thesis depends on direct large-model intervention rather than another proxy-stage proof | — Pending |
| Add an explicit routing margin loss in warmup | Router supervision is needed because harmful knowledge already exists in standard experts | — Pending |

---
*Last updated: 2026-03-19 after starting v1.1 milestone definition*
