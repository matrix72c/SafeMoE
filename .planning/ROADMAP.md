# Roadmap: SafeMoE

## Milestones

- ✅ **v1.0 PT Phase Validation** - Phases 1-4 shipped `2026-03-17`
  Archive: `.planning/milestones/v1.0-ROADMAP.md`
- 🚧 **v1.1 Qwen Harmful Transfer** - Phases 5-10 planned

## Overview

`v1.1` validates whether harmful capability in `Qwen3-30B-A3B-Base` can be initialized into designated experts, separated through routing-supervised warmup, concentrated through mixed-data transfer, and then cleanly removed with bounded standard-domain damage and higher adversarial recovery cost. The milestone starts from the existing LitGPT-converted base checkpoint and follows the tight research chain: environment/runtime gate -> checkpoint surgery -> registry/observability -> warmup -> transfer -> evaluation.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [x] **Phase 5: Environment Runtime Gate** - Verify the direct-Qwen checkpoint loads cleanly and establish the BF16 runtime envelope for milestone runs.
- [ ] **Phase 6: Checkpoint Surgery** - Create deterministic manifest-driven harmful expert and head cloning with loadable post-surgery checkpoints.
- [ ] **Phase 7: Registry and Routing Observability** - Prove harmful/std/shared classification and routing telemetry are correct on the direct-Qwen stack.
- [ ] **Phase 8: Warmup Separation** - Run routing-supervised warmup and verify harmful tokens separate from standard tokens under controlled evaluation.
- [ ] **Phase 9: Mixed-Data Transfer** - Resume from warmup into SGTM-style transfer and track whether harmful capability stays concentrated in `theta_harmful`.
- [ ] **Phase 10: Evaluation and Adversarial-Cost Gate** - Measure pre/post-ablation isolation quality and test whether adversarial recovery becomes materially harder.

## Phase Details

### Phase 5: Environment Runtime Gate
**Goal**: Researcher can start direct `Qwen3-30B-A3B-Base` milestone runs from the existing checkpoint with a known storage, memory, and runtime envelope.
**Depends on**: Phase 4
**Requirements**: ENV-01, ENV-02
**Success Criteria** (what must be TRUE):
  1. Researcher can load `checkpoints/Qwen3-30B-A3B-Base` through the direct-Qwen path without missing-file, schema, or compatibility failures.
  2. Researcher can run a BF16 dry-start train or eval job on the direct-Qwen stack and complete startup plus one measured step.
  3. Researcher can inspect a recorded envelope for storage footprint, peak memory, and runtime throughput before planning warmup or transfer runs.
**Plans**: 2 plans

Plans:
- [x] 05-01-PLAN.md — Build direct-Qwen checkpoint/data preflight coverage and bless the one-step BF16 gate config
- [x] 05-02-PLAN.md — Add runtime metric output and capture the canonical BF16 envelope report

### Phase 6: Checkpoint Surgery
**Goal**: Researcher can reproducibly create `theta_harmful` from the base Qwen checkpoint with manifest-backed expert/head/router cloning that survives reload.
**Depends on**: Phase 5
**Requirements**: INIT-01, INIT-02, INIT-03
**Success Criteria** (what must be TRUE):
  1. Researcher can generate a deterministic intervention manifest for a run, including expert/head selections, router-column mappings, seed, and noise scale.
  2. Researcher can apply the manifest to clone designated experts, attention heads, and router columns into `theta_harmful` while preserving a loadable checkpoint.
  3. Researcher can verify that post-surgery tensors and checkpoint metadata match the manifest’s source-to-target mapping and shape semantics.
**Plans**: 2 plans

Plans:
- [x] 06-01-PLAN.md — Define the manifest schema, persistence helpers, and deterministic intervention planner
- [ ] 06-02-PLAN.md — Implement manifest-backed checkpoint surgery, reload verification, and the researcher CLI

### Phase 7: Registry and Routing Observability
**Goal**: Researcher can trust the direct-Qwen harmful/std/shared parameter registry and the routing signal used for supervision and evaluation.
**Depends on**: Phase 6
**Requirements**: ROUT-01, ROUT-02, ROUT-03
**Success Criteria** (what must be TRUE):
  1. Researcher can inspect exhaustive, non-overlapping assignment of post-intervention parameters into `theta_harmful`, `theta_std`, and `theta_shared`.
  2. Researcher can collect comparable routing metrics for designated harmful experts during baseline, warmup, transfer, and ablation evaluation through one observability path.
  3. Researcher can confirm that the logged routing signal matches the actual dispatched expert behavior on the pinned Qwen stack.
**Plans**: 2 plans

Plans:
- [x] 07-01: Extend registry coverage and classification for direct Qwen interventions
- [ ] 07-02: Add shared routing telemetry and parity checks for train/eval

### Phase 8: Warmup Separation
**Goal**: Researcher can use routing-supervised warmup to separate harmful and standard token flow before mixed-data transfer begins.
**Depends on**: Phase 7
**Requirements**: WARM-01, WARM-02, WARM-03
**Success Criteria** (what must be TRUE):
  1. Researcher can run warmup on mixed `D_harmful` and `D_std` with next-token loss active and routing loss reported separately.
  2. Researcher can observe the supervised routing objective increase harmful-routing mass for `D_harmful` while suppressing it for `D_std`.
  3. Researcher can review a confound-controlled evaluation showing stronger post-warmup routing concentration on `theta_harmful` for `D_harmful` than for `D_std`.
**Plans**: TBD

Plans:
- [ ] 08-01: Integrate stage-aware warmup loss and logging
- [ ] 08-02: Add confound-controlled warmup evaluation and acceptance checks

### Phase 9: Mixed-Data Transfer
**Goal**: Researcher can continue from warmup into SGTM-style transfer and monitor whether harmful capability remains concentrated instead of diffusing back into standard/shared parameters.
**Depends on**: Phase 8
**Requirements**: XFER-01, XFER-02, XFER-03
**Success Criteria** (what must be TRUE):
  1. Researcher can resume from a saved warmup checkpoint into mixed-data transfer on `D_unlabeled`, `D_harmful`, and `D_std` without changing split semantics.
  2. Researcher can inspect periodic routing and ablation checkpoints that show whether harmful capability is diffusing into `theta_std` or `theta_shared` during transfer.
  3. Researcher can measure whether unlabeled harmful examples route toward `theta_harmful` after transfer.
**Plans**: TBD

Plans:
- [ ] 09-01: Resume warmup checkpoints into mixed-data SGTM transfer
- [ ] 09-02: Add diffusion and unlabeled-routing monitoring during transfer

### Phase 10: Evaluation and Adversarial-Cost Gate
**Goal**: Researcher can decide whether the milestone thesis passed by comparing pre/post-ablation behavior and a fixed adversarial recovery cost against the v1.0 baseline.
**Depends on**: Phase 9
**Requirements**: EVAL-01, EVAL-02, EVAL-03
**Success Criteria** (what must be TRUE):
  1. Researcher can run manifest-aware pre- and post-ablation evaluation on `D_harmful`, `D_std`, and selected unlabeled probes with metrics comparable to `v1.0`.
  2. Researcher can verify that ablating `theta_harmful` materially reduces harmful capability while keeping standard-domain regression within the milestone’s bounded tolerance.
  3. Researcher can run one pinned adversarial recovery protocol after ablation and compare recovery cost against the unablated or pre-intervention baseline.
**Plans**: TBD

Plans:
- [ ] 10-01: Build manifest-aware evaluation and ablation comparison reports
- [ ] 10-02: Run the fixed adversarial recovery gate and baseline comparison

## Progress

**Execution Order:**
Phases execute in numeric order: 5 -> 6 -> 7 -> 8 -> 9 -> 10

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 5. Environment Runtime Gate | 2/2 | Complete | 2026-03-19 |
| 6. Checkpoint Surgery | 1/2 | In Progress | - |
| 7. Registry and Routing Observability | 1/2 | In Progress | - |
| 8. Warmup Separation | 0/2 | Not started | - |
| 9. Mixed-Data Transfer | 0/2 | Not started | - |
| 10. Evaluation and Adversarial-Cost Gate | 0/2 | Not started | - |
