# Roadmap: SafeMoE

## Overview

Milestone 1 (PT Phase Validation) delivers a working SGTM pretraining pipeline on TinyStories bilingual data, proving that harmful knowledge can be isolated into designated MoE experts and ablated without degrading general capability. The build progresses from data pipeline, through model architecture and masking primitives, into an integrated training loop, and finally ablation and evaluation that validate the core thesis.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Data Pipeline** - TinyStories bilingual tokenization, three-split partitioning, and multi-stream DataLoader
- [ ] **Phase 2: Model Architecture & Masking** - SafeMoELayer, HarmfulParamRegistry, gradient/activation maskers with unit tests
- [ ] **Phase 3: SGTM Training Loop** - Forked pretrain.py with 3-path SGTM branching, CLI entry point, and dual optimizer
- [ ] **Phase 4: Ablation & Evaluation** - Expert ablation utility, per-split perplexity, routing attribution, mid-training eval

## Phase Details

### Phase 1: Data Pipeline
**Goal**: A tokenized TinyStories bilingual dataset partitioned into D_std/D_harmful/D_unlabeled splits with a configurable x% sweep parameter, served through per-split DataLoaders
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02, DATA-03
**Success Criteria** (what must be TRUE):
  1. Running the data preparation script produces three on-disk splits (D_std, D_harmful, D_unlabeled) from TinyStories English+Spanish with the correct proportions (25% EN D_std, (100-x)% ES D_harmful, 75% EN + x% ES D_unlabeled) for a given x
  2. MultiDataLoader yields batches from each split independently, with configurable upsample factors, and each batch is tagged with its split label (D_std/D_harmful/D_unlabeled)
  3. A pre-generated data_split_order schedule controls the per-step training mix, and iterating through it draws from the correct split at each step
**Plans**: 2 plans

Plans:
- [ ] 01-01-PLAN.md — Setup + prepare.py: env setup, test stubs (RED), implement compute_splits/prepare (GREEN) [DATA-01]
- [ ] 01-02-PLAN.md — MultiDataLoader: test stubs (RED), implement datamodule.py with dynamic weighted sampling (GREEN) [DATA-02, DATA-03]

### Phase 2: Model Architecture & Masking
**Goal**: A SafeMoE model with designatable harmful experts and verified masking primitives that correctly isolate gradient and activation flow
**Depends on**: Nothing (independent of Phase 1; can be parallelized)
**Requirements**: MOE-01, MOE-02, MOE-03, MOE-04, MASK-01, MASK-02, MASK-03, MASK-04
**Success Criteria** (what must be TRUE):
  1. A SafeMoEConfig can be instantiated with harmful_expert_indices and harmful_attn_heads, and a model built from it contains SafeMoELayer instances with the correct number of experts
  2. HarmfulParamRegistry correctly classifies every model parameter as theta_harmful or theta_std, and parameters_by_type returns non-overlapping, exhaustive parameter sets
  3. After a backward pass on a test batch with GradientMasker active, theta_std parameters have grad=None and theta_harmful parameters have non-zero gradients
  4. During a forward pass with ActivationMasker active, harmful expert outputs are exactly zero while standard expert outputs are non-zero
  5. Unit tests for all four masking invariants (MASK-04) pass: grad isolation, activation zeroing, and dual optimizer param groups with set_to_none=True do not corrupt Adam state
**Plans**: TBD

### Phase 3: SGTM Training Loop
**Goal**: An end-to-end SGTM pretraining script that consumes the three-split data, applies the correct masking path per split label, and produces a trained SafeMoE checkpoint
**Depends on**: Phase 1, Phase 2
**Requirements**: TRAIN-01, TRAIN-02, TRAIN-03
**Success Criteria** (what must be TRUE):
  1. Running `python -m safemoe pretrain --config <yaml>` launches training on TinyStories bilingual data and produces a checkpoint file, consistent with LitGPT's jsonargparse conventions
  2. During training, D_harmful-labeled steps apply GradientMasker (only theta_harmful updates), D_std-labeled steps apply ActivationMasker (theta_harmful zeroed in forward), and D_unlabeled-labeled steps perform standard forward+backward with no masking
  3. Training loss decreases over steps for all three split types, confirming the model learns from all data streams without masking errors stalling optimization
**Plans**: TBD

### Phase 4: Ablation & Evaluation
**Goal**: A complete evaluation pipeline that ablates harmful experts and measures whether knowledge isolation succeeded -- the validation of SafeMoE's core thesis
**Depends on**: Phase 3
**Requirements**: TRAIN-04, EVAL-01, EVAL-02, EVAL-03
**Success Criteria** (what must be TRUE):
  1. The ablate() utility zeros theta_harmful weights in a checkpoint and saves a separate ablated checkpoint file; loading the ablated checkpoint produces a model with exactly zero weights in all harmful expert parameters
  2. Per-split perplexity evaluation shows that post-ablation D_harmful-split (Spanish) perplexity increases significantly while D_std-split (English) perplexity remains near the pre-ablation baseline
  3. Routing attribution histograms show that D_harmful-split tokens preferentially activate theta_harmful experts during training, confirming the router learned to route harmful-domain tokens to designated experts
  4. Mid-training ablation evaluation at periodic checkpoints shows isolation improving over training (D_harmful-split perplexity delta between ablated/non-ablated models grows as training progresses)
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4
(Phases 1 and 2 have no mutual dependency and may be parallelized.)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Data Pipeline | 0/2 | Not started | - |
| 2. Model Architecture & Masking | 0/TBD | Not started | - |
| 3. SGTM Training Loop | 0/TBD | Not started | - |
| 4. Ablation & Evaluation | 0/TBD | Not started | - |
