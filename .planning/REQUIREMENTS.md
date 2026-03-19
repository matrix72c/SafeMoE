# Requirements: SafeMoE

**Defined:** 2026-03-19
**Core Value:** Harmful knowledge must be fully containable in a designatable set of MoE experts that can be zeroed out at inference time without degrading general model capability.

## v1 Requirements

### Environment Readiness

- [x] **ENV-01**: Researcher can load the existing LitGPT-converted `Qwen3-30B-A3B-Base` checkpoint from `checkpoints/Qwen3-30B-A3B-Base` without missing-file, schema, or checkpoint-compatibility errors.
- [x] **ENV-02**: Researcher can run a dry-start BF16 training or evaluation job on the direct-Qwen stack and record the storage, memory, and runtime envelope needed for milestone runs.

### Intervention Initialization

- [x] **INIT-01**: Researcher can define a deterministic intervention manifest that records the selected source experts, target harmful experts, source attention heads, target harmful heads, cloned router columns, random seed, and noise scale for a run.
- [x] **INIT-02**: Researcher can initialize `theta_harmful` in `Qwen3-30B-A3B-Base` by cloning selected experts and attention heads from `theta_std`, copying the corresponding router columns, and adding controlled noise while preserving a loadable checkpoint.
- [x] **INIT-03**: Researcher can verify that post-surgery tensors match the manifest semantics through parity checks on tensor shapes, source-to-target mappings, and checkpoint reload behavior.

### Registry and Routing Observability

- [ ] **ROUT-01**: Researcher can classify direct-Qwen parameters into `theta_harmful`, `theta_std`, and `theta_shared` with exhaustive, non-overlapping registry coverage after the intervention is applied.
- [ ] **ROUT-02**: Researcher can capture per-split routing metrics for designated harmful experts during baseline, warmup, transfer, and ablation evaluation using one shared observability path.
- [ ] **ROUT-03**: Researcher can verify that the routing signal used for supervision matches the real dispatched expert behavior for the pinned Qwen stack.

### Warmup Separation

- [ ] **WARM-01**: Researcher can run a warmup stage on mixed `D_harmful` and `D_std` while keeping next-token loss active and logging routing loss separately from LM loss.
- [ ] **WARM-02**: Researcher can apply a supervised routing objective that increases harmful-routing mass for `D_harmful` tokens and suppresses harmful-routing mass for `D_std` tokens.
- [ ] **WARM-03**: Researcher can demonstrate, with a confound-controlled evaluation, that post-warmup `D_harmful` routes more strongly to `theta_harmful` than `D_std`.

### Mixed-Data Transfer

- [ ] **XFER-01**: Researcher can resume from a saved warmup checkpoint into SGTM-style mixed-data transfer on `D_unlabeled`, `D_harmful`, and `D_std` without changing the existing split semantics.
- [ ] **XFER-02**: Researcher can track whether harmful capability diffuses back into `theta_std` or `theta_shared` during transfer using periodic routing and ablation checkpoints.
- [ ] **XFER-03**: Researcher can measure whether unlabeled harmful examples naturally route toward `theta_harmful` after transfer.

### Evaluation and Safety Gate

- [ ] **EVAL-01**: Researcher can run manifest-aware pre- and post-ablation evaluation on `D_harmful`, `D_std`, and selected unlabeled probes using metrics comparable to the shipped `v1.0` baseline.
- [ ] **EVAL-02**: Researcher can show that ablating `theta_harmful` causes a material harmful-capability drop while keeping standard-domain regression bounded.
- [ ] **EVAL-03**: Researcher can run one fixed adversarial recovery protocol after ablation with a pinned budget and report whether recovery cost rises relative to the unablated or pre-intervention baseline.

## v2 Requirements

### Experiment Expansion

- **EXP-01**: Researcher can run a bounded warmup sensitivity sweep over routing-loss weight or warmup duration after the primary configuration passes.
- **EXP-02**: Researcher can add finer-grained routing diagnostics such as per-layer or per-head separation summaries.
- **EXP-03**: Researcher can compare multiple adversarial recovery budgets or attack recipes after the fixed gate protocol is stable.

### Broader Generalization

- **GEN-01**: Researcher can replace the bilingual TinyStories proxy with a more realistic harmful dataset once the direct-Qwen intervention path is validated.
- **GEN-02**: Researcher can replicate the milestone on additional MoE model families beyond `Qwen3-30B-A3B-Base`.

## Out of Scope

| Feature | Reason |
|---------|--------|
| DeepSpeed, HF `Trainer`, PEFT, QLoRA, or a second primary training stack | Increases integration risk without helping validate the direct-Qwen thesis |
| Quantized 8-bit or 4-bit mainline training | Not the intended path for full-weight intervention and would distort milestone conclusions |
| Broad redesign of the existing TinyStories data pipeline | Existing split semantics are already validated and should remain the control baseline |
| Production serving, chat interfaces, or demo infrastructure | Does not advance routing separation, transfer, or ablation validation |
| Open-ended hyperparameter sweeps or adversarial benchmark matrices | Turns a milestone feasibility test into an unbounded optimization project |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ENV-01 | Phase 5 | Complete |
| ENV-02 | Phase 5 | Complete |
| INIT-01 | Phase 6 | Complete |
| INIT-02 | Phase 6 | Complete |
| INIT-03 | Phase 6 | Complete |
| ROUT-01 | Phase 7 | Pending |
| ROUT-02 | Phase 7 | Pending |
| ROUT-03 | Phase 7 | Pending |
| WARM-01 | Phase 8 | Pending |
| WARM-02 | Phase 8 | Pending |
| WARM-03 | Phase 8 | Pending |
| XFER-01 | Phase 9 | Pending |
| XFER-02 | Phase 9 | Pending |
| XFER-03 | Phase 9 | Pending |
| EVAL-01 | Phase 10 | Pending |
| EVAL-02 | Phase 10 | Pending |
| EVAL-03 | Phase 10 | Pending |

**Coverage:**
- v1 requirements: 17 total
- Mapped to phases: 17
- Unmapped: 0

---
*Requirements defined: 2026-03-19*
*Last updated: 2026-03-19 after completing Phase 06 Plan 01*
