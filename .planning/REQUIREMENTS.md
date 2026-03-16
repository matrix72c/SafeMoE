# Requirements: SafeMoE

**Defined:** 2026-03-14
**Core Value:** Harmful knowledge must be fully containable in a designatable set of MoE parameters that can be zeroed out at inference time without degrading general model capability.

## v1 Requirements

Requirements for Milestone 1 (PT Phase Validation). Maps to roadmap phases.

### MoE Architecture

- [x] **MOE-01**: SafeMoEConfig extends LitGPT `Config` with `harmful_expert_indices`, `num_harmful_experts`, `harmful_attn_heads` fields
- [x] **MOE-02**: HarmfulParamRegistry registers and manages full-model parameter -> theta_harmful/theta_std mapping (covers MoE experts, attention heads, and all other layers); exposes `parameters_by_type(sgtm_split)` interface for optimizer and masker use
- [x] **MOE-03**: SafeMoELayer subclasses `LLaMAMoE` from `litgpt/model.py`, integrates HarmfulParamRegistry for per-expert routing and designation
- [x] **MOE-04**: Harmful expert initialization strategy is configurable (random init vs. copy weights from existing std experts)

### Masking Infrastructure

- [x] **MASK-01**: GradientMasker -- after backward pass on D_harmful batch, sets theta_std parameter gradients to `None` (post-backward zeroing, not detach-in-forward), ensuring only theta_harmful parameters update
- [x] **MASK-02**: ActivationMasker -- zeros theta_harmful expert outputs during D_std forward pass (skips harmful expert dispatch), forcing model to rely solely on theta_std
- [x] **MASK-03**: Dual optimizer param groups (separate AdamW for theta_harmful and theta_std) with `zero_grad(set_to_none=True)` to prevent Adam momentum corruption from zero gradients
- [x] **MASK-04**: Unit tests confirming: grad norm = 0 (or grad is None) for masked params after D_harmful backward; grad norm > 0 for unmasked params; theta_harmful output = 0 during D_std forward

### Data Pipeline

- [x] **DATA-01**: Data preparation script tokenizes `ffuuugor/tinystories-spanish` (tiktoken gpt2) and partitions into three splits: D_std (25% EN), D_harmful ((100-x)% ES), D_unlabeled (75% EN + x% ES), where x is a configurable sweep parameter; aligned with paper's `tinystories_tokenize_and_split.py`
- [x] **DATA-02**: `MultiDataLoader` wrapper provides per-split DataLoaders (D_std/D_harmful/D_unlabeled) with configurable upsample factors (`upsample_std`, `upsample_harmful`, `upsample_unlabeled`); consistent with LitGPT `DataModule` abstraction
- [x] **DATA-03**: Pre-generated `data_split_order` list (shuffled schedule of "D_std"/"D_harmful"/"D_unlabeled" step labels) controls training mix; each step draws from the corresponding split

### Training Loop

- [x] **TRAIN-01**: Fork `litgpt/pretrain.py` -> `safemoe/pretrain.py` implementing SGTM 3-path branching per step label: D_harmful -> gradient masking (MASK-01 post-backward), D_std -> activation masking (MASK-02 in forward), D_unlabeled -> standard forward+backward
- [x] **TRAIN-02**: Per-step split label sampled via `random.choices(SPLIT_LABELS, weights=[upsample_std, upsample_harmful, upsample_unlabeled])` once per optimizer step; 3-path `if/elif/else` dispatch in `fit()` calls `gradient_masker.mask()` post-backward for D_harmful, brackets the micro-batch window with `activation_masker.enable()/disable()` (try/finally) for D_std, and runs standard forward+backward with both optimizers stepping for D_unlabeled
- [x] **TRAIN-03**: CLI entry point `python -m safemoe pretrain` with YAML config support, consistent with LitGPT's jsonargparse conventions and config pattern from `configs/tinystories/`
- [ ] **TRAIN-04**: `ablate()` utility zeros theta_harmful weights in-place (`set grad/weights = 0`) and saves the ablated checkpoint as a separate file for inference evaluation

### Evaluation

- [ ] **EVAL-01**: Per-split perplexity evaluation (D_std / D_harmful / D_unlabeled) on validation sets before and after ablation, mirroring paper's `evaluate_all_datasets()` structure
- [ ] **EVAL-02**: Routing attribution analysis -- per-token histogram of which expert type (theta_harmful vs theta_std) each data split preferentially activates; logged to W&B or TensorBoard
- [ ] **EVAL-03**: Mid-training ablation evaluation -- at each eval checkpoint, temporarily ablate theta_harmful and evaluate the ablated model, then restore; tracks isolation progress over training

## v2 Requirements

Deferred to future milestones.

### Milestone 2 -- CPT Routing Validation

- **CPT-01**: Inject theta_harmful experts into a pretrained LitGPT checkpoint (router extension + new expert weight init)
- **CPT-02**: CPT training loop with router logging; confirm harmful-domain tokens preferentially route to theta_harmful experts
- **CPT-03**: Real harmful dataset integration (e.g. WildGuard, HarmBench) replacing TinyStories proxy

### Milestone 3 -- CPT Knowledge Transfer

- **XFER-01**: CPT with large unlabeled corpus demonstrates harmful knowledge migrates from theta_std -> theta_harmful over training
- **XFER-02**: Knowledge transfer metrics: residual harmful capability in theta_std before and after CPT
- **XFER-03**: Final ablation evaluation: perplexity + harm benchmarks pre/post theta_harmful removal

## Out of Scope

| Feature | Reason |
|---------|--------|
| Real-time serving / model deployment | Research focus only; not needed for validation |
| Multi-node distributed training | Single-machine GPU sufficient for research-scale experiments |
| Novel tokenizer architecture | Use paper's tiktoken gpt2 tokenizer; no tokenizer changes needed |
| Load-balancing loss (standard MoE) | Intentionally omitted -- standard balance loss fights harmful-expert concentration |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Complete |
| DATA-02 | Phase 1 | Complete |
| DATA-03 | Phase 1 | Complete |
| MOE-01 | Phase 2 | Complete |
| MOE-02 | Phase 2 | Complete |
| MOE-03 | Phase 2 | Complete |
| MOE-04 | Phase 2 | Complete |
| MASK-01 | Phase 2 | Complete |
| MASK-02 | Phase 2 | Complete |
| MASK-03 | Phase 2 | Complete |
| MASK-04 | Phase 2 | Complete |
| TRAIN-01 | Phase 3 | Complete |
| TRAIN-02 | Phase 3 | Complete |
| TRAIN-03 | Phase 3 | Complete |
| TRAIN-04 | Phase 4 | Pending |
| EVAL-01 | Phase 4 | Pending |
| EVAL-02 | Phase 4 | Pending |
| EVAL-03 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 18 total
- Mapped to phases: 18
- Unmapped: 0

---
*Requirements defined: 2026-03-14*
*Last updated: 2026-03-14 after roadmap revision (phase swap + terminology unification)*
