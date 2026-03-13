# Requirements: SafeMoE

**Defined:** 2026-03-14
**Core Value:** Harmful knowledge must be fully containable in a designatable set of MoE parameters that can be zeroed out at inference time without degrading general model capability.

## v1 Requirements

Requirements for Milestone 1 (PT Phase Validation). Maps to roadmap phases.

### MoE Architecture

- [ ] **MOE-01**: SafeMoEConfig extends LitGPT `Config` with `harmful_expert_indices`, `num_harmful_experts`, `harmful_attn_heads` fields
- [ ] **MOE-02**: HarmfulParamRegistry registers and manages full-model parameter → θ_harmful/θ_std mapping (covers MoE experts, attention heads, and all other layers); exposes `parameters_by_type(sgtm_split)` interface for optimizer and masker use
- [ ] **MOE-03**: SafeMoELayer subclasses `LLaMAMoE` from `litgpt/model.py`, integrates HarmfulParamRegistry for per-expert routing and designation
- [ ] **MOE-04**: Harmful expert initialization strategy is configurable (random init vs. copy weights from existing std experts)

### Masking Infrastructure

- [ ] **MASK-01**: GradientMasker — after backward pass on D_harmful batch, sets θ_std parameter gradients to `None` (post-backward zeroing, not detach-in-forward), ensuring only θ_harmful parameters update
- [ ] **MASK-02**: ActivationMasker — zeros θ_harmful expert outputs during D_std forward pass (skips harmful expert dispatch), forcing model to rely solely on θ_std
- [ ] **MASK-03**: Dual optimizer param groups (separate AdamW for θ_harmful and θ_std) with `zero_grad(set_to_none=True)` to prevent Adam momentum corruption from zero gradients
- [ ] **MASK-04**: Unit tests confirming: grad norm = 0 (or grad is None) for masked params after D_harmful backward; grad norm > 0 for unmasked params; θ_harmful output = 0 during D_std forward

### Data Pipeline

- [ ] **DATA-01**: Data preparation script tokenizes `ffuuugor/tinystories-spanish` (tiktoken gpt2) and partitions into three splits: retain (25% EN), forget ((100-x)% ES), adjacent (75% EN + x% ES), where x is a configurable sweep parameter; aligned with paper's `tinystories_tokenize_and_split.py`
- [ ] **DATA-02**: `MultiDataLoader` wrapper provides per-split DataLoaders (retain/forget/adjacent) with configurable upsample factors (`upsample_retain`, `upsample_forget`, `upsample_adjacent`); consistent with LitGPT `DataModule` abstraction
- [ ] **DATA-03**: Pre-generated `data_split_order` list (shuffled schedule of "retain"/"forget"/"adjacent" step labels) controls training mix; each step draws from the corresponding split

### Training Loop

- [ ] **TRAIN-01**: Fork `litgpt/pretrain.py` → `safemoe/pretrain.py` implementing SGTM 3-path branching per step label: forget → gradient masking (MASK-01 post-backward), retain → activation masking (MASK-02 in forward), adjacent → standard forward+backward
- [ ] **TRAIN-02**: `sgtm_mode` scalar passed as part of batch dict to model forward; `adjust_gradients(sgtm_mode)` called after each backward pass before optimizer step
- [ ] **TRAIN-03**: CLI entry point `python -m safemoe pretrain` with YAML config support, consistent with LitGPT's jsonargparse conventions and config pattern from `configs/tinystories/`
- [ ] **TRAIN-04**: `ablate()` utility zeros θ_harmful weights in-place (`set grad/weights = 0`) and saves the ablated checkpoint as a separate file for inference evaluation

### Evaluation

- [ ] **EVAL-01**: Per-split perplexity evaluation (retain / forget / adjacent) on validation sets before and after ablation, mirroring paper's `evaluate_all_datasets()` structure
- [ ] **EVAL-02**: Routing attribution analysis — per-token histogram of which expert type (θ_harmful vs θ_std) each data split preferentially activates; logged to W&B or TensorBoard
- [ ] **EVAL-03**: Mid-training ablation evaluation — at each eval checkpoint, temporarily ablate θ_harmful and evaluate the ablated model, then restore; tracks isolation progress over training

## v2 Requirements

Deferred to future milestones.

### Milestone 2 — CPT Routing Validation

- **CPT-01**: Inject θ_harmful experts into a pretrained LitGPT checkpoint (router extension + new expert weight init)
- **CPT-02**: CPT training loop with router logging; confirm harmful-domain tokens preferentially route to θ_harmful experts
- **CPT-03**: Real harmful dataset integration (e.g. WildGuard, HarmBench) replacing TinyStories proxy

### Milestone 3 — CPT Knowledge Transfer

- **XFER-01**: CPT with large unlabeled corpus demonstrates harmful knowledge migrates from θ_std → θ_harmful over training
- **XFER-02**: Knowledge transfer metrics: residual harmful capability in θ_std before and after CPT
- **XFER-03**: Final ablation evaluation: perplexity + harm benchmarks pre/post θ_harmful removal

## Out of Scope

| Feature | Reason |
|---------|--------|
| Real-time serving / model deployment | Research focus only; not needed for validation |
| Multi-node distributed training | Single-machine GPU sufficient for research-scale experiments |
| Novel tokenizer architecture | Use paper's tiktoken gpt2 tokenizer; no tokenizer changes needed |
| Load-balancing loss (standard MoE) | Intentionally omitted — standard balance loss fights harmful-expert concentration |

## Traceability

Populated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| MOE-01 | — | Pending |
| MOE-02 | — | Pending |
| MOE-03 | — | Pending |
| MOE-04 | — | Pending |
| MASK-01 | — | Pending |
| MASK-02 | — | Pending |
| MASK-03 | — | Pending |
| MASK-04 | — | Pending |
| DATA-01 | — | Pending |
| DATA-02 | — | Pending |
| DATA-03 | — | Pending |
| TRAIN-01 | — | Pending |
| TRAIN-02 | — | Pending |
| TRAIN-03 | — | Pending |
| TRAIN-04 | — | Pending |
| EVAL-01 | — | Pending |
| EVAL-02 | — | Pending |
| EVAL-03 | — | Pending |

**Coverage:**
- v1 requirements: 18 total
- Mapped to phases: 0
- Unmapped: 18 ⚠️

---
*Requirements defined: 2026-03-14*
*Last updated: 2026-03-14 after initial definition*
