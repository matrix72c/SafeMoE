# SafeMoE

## What This Is

A research framework built on LitGPT that implements SGTM (Selective Gradient/Token Masking) for Mixture-of-Experts models. The goal is to isolate harmful knowledge into dedicated "Harmful Experts" via controlled (pre)training, then permanently ablate those experts at inference time — surgically removing harmful capabilities while leaving general capabilities intact.

The project proceeds in three milestone-validated steps: PT-phase validation with a bilingual proxy dataset, CPT-phase routing analysis on a pretrained model, and CPT-phase knowledge transfer using unlabeled data.

## Core Value

Harmful knowledge must be fully containable in a designatable set of MoE experts that can be zeroed out at inference time without degrading general model capability.

## Requirements

### Validated

<!-- What LitGPT already provides — the foundation we build on. -->

- ✓ Decoder-only transformer (GPT) with configurable Block/Attention/MLP — `litgpt/model.py`
- ✓ Pretraining infrastructure with Lightning Fabric (gradient accumulation, checkpointing, distributed) — `litgpt/pretrain.py`
- ✓ DataModule abstraction for pluggable datasets — `litgpt/data/`
- ✓ Checkpoint save/load utilities — `litgpt/scripts/`, `litgpt/utils.py`
- ✓ CLI entry point with jsonargparse — `litgpt/__main__.py`
- ✓ LM evaluation harness integration — `litgpt/eval/`
- ✓ LoRA / Adapter parameter-efficient training patterns (reference for selective param masking)

### Active

<!-- Milestone 1: PT Phase Validation -->

- [ ] MoE model architecture: Top-K router + expert dispatch layer built from LitGPT components
- [ ] Harmful Expert designation: mechanism to mark specific experts and attention heads as θ_harmful
- [ ] SGTM training loop: selective gradient masking (D_harmful → only θ_harmful updates) + selective parameter masking (D_std → θ_harmful zeroed in forward pass) + normal pass (D_unlabeled)
- [ ] TinyStories bilingual data pipeline: partition English/Spanish into D_std / D_harmful / D_unlabeled with configurable x%
- [ ] Expert ablation utility: zero out θ_harmful weights post-training
- [ ] Evaluation suite: per-language perplexity (pre/post ablation) + routing attribution analysis (token → expert assignment statistics)

<!-- Milestone 2: CPT Routing -->

- [ ] Harmful Expert injection into a pretrained LitGPT model (add θ_harmful experts to existing checkpoint)
- [ ] CPT training loop with router logging (track which tokens route to which experts)
- [ ] Real harmful dataset integration (e.g. WildGuard, HarmBench)
- [ ] Routing analysis: confirm harmful tokens preferentially activate Harmful Experts

<!-- Milestone 3: CPT Transfer -->

- [ ] CPT with large unlabeled corpus: demonstrate harmful knowledge migrates from Std Experts → Harmful Experts over training
- [ ] Knowledge transfer metrics: measure residual harmful capability in Std Experts before and after CPT
- [ ] Final ablation evaluation: perplexity + harm benchmarks pre/post Harmful Expert removal

### Out of Scope

- Real-time serving / deployment of SafeMoE models — research focus only
- Multi-node distributed training — single-machine GPU experiments sufficient for validation
- Novel tokenizer or model architecture changes beyond MoE expert layer additions

## Context

**Foundation:** Official LitGPT repository (Python/PyTorch/Lightning). All SafeMoE components extend LitGPT's existing `model.py`, `pretrain.py`, and data pipeline patterns. The codebase has no MoE implementation yet — it must be added as a new module.

**Proxy Setup (Milestone 1):** TinyStories bilingual dataset (English + Spanish) is used as a controlled proxy where Spanish represents "harmful knowledge." This allows clean measurement of knowledge isolation without ethical concerns around real harmful content. x% of Spanish goes into D_unlabeled (sweep parameter).

**SGTM Algorithm:**
- D_harmful samples: after backward pass, zero-out ∇θ_std; only θ_harmful parameters update
- D_std samples: during forward pass, zero θ_harmful activations; model must predict using θ_std only
- D_unlabeled samples: standard forward + backward, no masking

**Evaluation Reference Points:**
- Post-ablation Spanish perplexity should increase significantly (harmful knowledge removed)
- Post-ablation English perplexity should remain near pre-training baseline (general knowledge intact)
- Routing histograms should show Spanish tokens concentrating on θ_harmful experts during training

## Constraints

- **Tech Stack**: Must extend LitGPT internals directly — no separate training framework
- **Scale**: PT experiments target single-GPU (Tiny Stories + small custom MoE); CPT experiments TBD based on pretrained model choice
- **Reproducibility**: All experiments must be configurable via CLI args consistent with LitGPT's jsonargparse convention

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Build MoE from LitGPT components (not Mixtral) for PT phase | Full control over expert assignment and masking hooks; Mixtral's MoE would require non-trivial surgery | — Pending |
| TinyStories bilingual as harmful proxy for Milestone 1 | Clean ground truth for isolation; avoids real harmful content during algorithm validation | — Pending |
| Three separate milestones (PT validate → CPT routing → CPT transfer) | Each step has independent success criteria; avoids building on unvalidated assumptions | — Pending |

---
*Last updated: 2026-03-13 after initialization*
