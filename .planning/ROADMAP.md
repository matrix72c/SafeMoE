# Roadmap: SafeMoE

## Overview

Milestone 1 (PT Phase Validation) delivers a working SGTM pretraining pipeline on TinyStories bilingual data, proving that harmful knowledge can be isolated into designated MoE experts and ablated without degrading general capability. The build progresses from data pipeline, through model architecture and masking primitives, into an integrated training loop, and finally ablation and evaluation that validate the core thesis.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Data Pipeline** - TinyStories bilingual tokenization, three-split partitioning, and multi-stream DataLoader (completed 2026-03-15)
- [x] **Phase 2: Model Architecture & Masking** - SafeMoELayer, HarmfulParamRegistry, gradient/activation maskers with unit tests (completed 2026-03-16)
- [ ] **Phase 3: SGTM Training Loop** - Forked pretrain.py with 3-path SGTM branching, CLI entry point, and dual optimizer
- [ ] **Phase 4: Ablation & Evaluation** - Expert ablation utility, per-split perplexity, routing attribution, mid-training eval

## Phase Details

### Phase 1: Data Pipeline
**Goal**: A tokenized TinyStories bilingual dataset partitioned into D_std/D_harmful/D_unlabeled splits with a two-parameter (x, y) sweep scheme, served through per-split DataLoaders via MultiDataLoader.get_loader()
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02, DATA-03
**Success Criteria** (what must be TRUE):
  1. Running the data preparation script produces three on-disk splits (D_std, D_harmful, D_unlabeled) from TinyStories English+Spanish with the correct proportions for given x, y: D_std=y% EN, D_harmful=(100-x)% ES, D_unlabeled=(100-y)% EN + x% ES; cached at data/.cache/Qwen3-30B-A3B-Base/{x}-{y}/
  2. MultiDataLoader.get_loader(split_name) returns a StreamingDataLoader for the named split; val_dataloaders() returns {"D_std": DataLoader, "D_harmful": DataLoader}; training loop manages its own iterators
  3. Dynamic split sampling (random.choices with weights) lives in the Phase 3 training loop, not in MultiDataLoader; MultiDataLoader is a loader registry only
**Plans**: 2 plans

Plans:
- [x] 01-01-PLAN.md — prepare.py: litdata install, test stubs (RED), implement compute_splits(x,y) + tokenization to integer-keyed cache dirs (GREEN) [DATA-01]
- [ ] 01-02-PLAN.md — MultiDataLoader: test stubs (RED), implement datamodule.py with get_loader() registry interface and val_dataloaders() (GREEN) [DATA-02, DATA-03]

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
**Plans**: 4 plans

Plans:
- [x] 02-01-PLAN.md — TDD RED: test stubs for all 4 test files + safemoe-tinystories.yaml experiment config [MOE-01, MOE-02, MOE-03, MOE-04, MASK-01, MASK-02, MASK-03, MASK-04]
- [x] 02-02-PLAN.md — TDD GREEN: SafeMoEConfig (config.py) + SafeMoELayer (model.py) [MOE-01, MOE-03, MOE-04]
- [x] 02-03-PLAN.md — TDD GREEN: HarmfulParamRegistry in masking.py with GradientMasker/ActivationMasker stubs [MOE-02, MASK-03]
- [ ] 02-04-PLAN.md — TDD GREEN: GradientMasker + ActivationMasker complete implementation, full suite GREEN [MASK-01, MASK-02, MASK-04]

### Phase 3: SGTM Training Loop
**Goal**: An end-to-end SGTM pretraining script that consumes the three-split data, applies the correct masking path per split label, and produces a trained SafeMoE checkpoint
**Depends on**: Phase 1, Phase 2
**Requirements**: TRAIN-01, TRAIN-02, TRAIN-03
**Success Criteria** (what must be TRUE):
  1. Running `python -m safemoe pretrain --config <yaml>` launches training on TinyStories bilingual data and produces a checkpoint file, consistent with LitGPT's jsonargparse conventions
  2. During training, D_harmful-labeled steps apply GradientMasker (only theta_harmful updates), D_std-labeled steps apply ActivationMasker (theta_harmful zeroed in forward), and D_unlabeled-labeled steps perform standard forward+backward with no masking
  3. Training loss decreases over steps for all three split types, confirming the model learns from all data streams without masking errors stalling optimization
**Plans**: 3 plans

Plans:
- [ ] 03-01-PLAN.md — TDD RED: test stubs for TRAIN-01/02/03 + extend GradientMasker/ActivationMasker for attn head masking (GREEN for masker tests) [TRAIN-01, TRAIN-02]
- [ ] 03-02-PLAN.md — Fork litgpt/pretrain.py: dual optimizer setup, 3-path SGTM branching, split sampling, pretrain loop unit tests GREEN [TRAIN-01, TRAIN-02]
- [ ] 03-03-PLAN.md — CLI entry point safemoe/__main__.py + YAML config update + test_pretrain_produces_checkpoint GREEN [TRAIN-03]

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
| 1. Data Pipeline | 2/2 | Complete   | 2026-03-15 |
| 2. Model Architecture & Masking | 4/4 | Complete   | 2026-03-16 |
| 3. SGTM Training Loop | 2/3 | In Progress|  |
| 4. Ablation & Evaluation | 0/TBD | Not started | - |
