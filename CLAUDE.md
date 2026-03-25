# CLAUDE.md

## Project Overview

SafeMoE explores selective harmful-knowledge isolation in Mixture-of-Experts language models.

- Start from a pretrained MoE model that already contains both safe/general knowledge and harmful knowledge.
- Introduce dedicated **Harmful Experts** and optionally harmful attention heads.
- Use routing constraints and selective gradient/parameter masking so that:
  - harmful knowledge is enriched into harmful-specific parameters;
  - standard experts retain general knowledge;
  - harmful experts can later be ablated at inference time with minimal loss of general capability.

## Core Research Objective

The project has two main targets:
1. **Enrichment**: move harmful knowledge out of original/shared experts and into newly introduced Harmful Experts.
2. **Ablation**: remove harmful capability at inference by zeroing or disabling Harmful Experts while preserving general utility.

## Experiment Steps

### Step 1: PT (Complete)

Purpose:
- Validate the SGTM-style setup on MoE models.
- Ensure Std Experts primarily preserve general knowledge.
- Ensure Harmful Experts primarily preserve harmful knowledge.

Parameter partitioning:
- `theta_harmful`: Harmful Attention Heads + Harmful Experts
- `theta_std`: remaining attention heads + Std Experts
- `theta_shared`: shared/router/embedding and other remaining parameters

Data behaviors:
- On `D_harmful`: apply selective gradient masking so only `theta_harmful` and `theta_shared` update.
- On `D_std`: apply selective parameter masking so `theta_harmful` is masked during forward pass.
- On `D_unlabeled`: standard training without masks.
- `theta_shared` updates on all three data types.

Expected PT result:
- After training, ablate `theta_harmful` and verify harmful capability is removed while general capability remains mostly intact.

### Step 2: CPT (Current)

#### Model initialization
- Randomly select `k` experts and `n` attention heads.
- Duplicate them with small noise to initialize new harmful-specific parameters.
- Duplicate corresponding router columns for the new experts.
- Treat original parameters as standard parameters and duplicated ones as `theta_harmful`.

#### Warm-up
Purpose:
- Force router preference so harmful tokens are sent to Harmful Experts.
- NTP loss alone is not enough because original Std Experts already contain harmful knowledge.

Routing objective from design:
- Let `z_t = sum_{i in H} P_i(x_t)` be total routing probability into harmful experts.
- `y_t = 1` for harmful tokens, `y_t = 0` for standard tokens.
- Encourage harmful tokens to route into Harmful Experts with threshold `tau_h`.
- Encourage standard tokens to avoid Harmful Experts with threshold `tau_s`.

Warm-up validation:
- unlabeled harmful data should naturally route to Harmful Experts;
- unlabeled standard data should route to Std Experts.

#### Knowledge transfer
Purpose:
- Use the SGTM training paradigm with large unlabeled general data.
- Gradually squeeze harmful knowledge out of Std Experts and into Harmful Experts.

Validation target:
- adversarial finetuning cost should increase significantly after transfer.

## Guidance for Claude Code

When modifying this repository:
- Preserve the separation between harmful-specific, standard, and shared parameters.
- Keep routing logic, masking logic, and ablation logic explicit and easy to inspect.
- Prefer minimal changes aligned with the current research design.
- Avoid introducing abstractions unless they clearly simplify repeated logic.
- When implementing training changes, verify whether they affect:
  - harmful/std/shared parameter partitioning,
  - router behavior,
  - masking behavior on different dataset splits,
  - ablation-time disabling of harmful experts.
