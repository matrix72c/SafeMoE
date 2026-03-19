# Project Research Summary

**Project:** SafeMoE v1.1 Qwen Harmful Transfer
**Domain:** Direct harmful-transfer research on `Qwen3-30B-A3B-Base`
**Researched:** 2026-03-19
**Confidence:** HIGH

## Executive Summary

This milestone is not a new product surface or a new training framework. It is a direct-intervention research milestone on a real pretrained MoE model: clone designated experts and attention heads from `Qwen3-30B-A3B-Base`, supervise routing during warmup, then run mixed-data SGTM transfer to test whether harmful capability can be concentrated into `theta_harmful` and cleanly removed by ablation. The correct expert approach is to keep SafeMoE’s existing LitGPT/Fabric training path, insert a thin Qwen-specific checkpoint-surgery layer, and extend the current trainer, evaluation, and ablation interfaces rather than forking the system.

The stack decision is therefore conservative: stay on `torch>=2.7`, Lightning Fabric, the existing LitGPT conversion scripts, and HF `transformers` only for model access and parity validation. Add only the pieces the milestone truly needs: direct checkpoint import/surgery utilities, stage-aware routing supervision, richer routing observability, and manifest-based provenance for cloned experts, heads, and router columns. Do not add DeepSpeed, HF `Trainer`, PEFT, quantized mainline training, or serving infrastructure; each of those would increase integration risk without helping validate the milestone thesis.

The main risks should define the roadmap. Phase boundaries must protect against semantically wrong checkpoint surgery, incorrect harmful/std/shared parameter classification after wrapping, router-loss wiring bugs, dataset confounds that masquerade as harmful routing, and SGTM diffusion that re-entangles harmful capability into standard/shared parameters. The roadmap should front-load parity checks, registry invariants, and observability before warmup and transfer, and it should treat ablation-aware evaluation plus bounded adversarial-cost validation as milestone gates rather than cleanup work.

## Key Findings

### Recommended Stack

The stack addition story is narrow and explicit. SafeMoE already has the right core runtime for this milestone; the work is to extend it for direct Qwen surgery and stage-aware measurement, not to replace it. `Qwen3-30B-A3B-Base` should be pulled through Hugging Face and converted into the existing LitGPT naming/layout so all downstream masking, training, and ablation code stays coherent.

**Core technologies:**
- `PyTorch >=2.7`: full-weight training, checkpoint surgery, and FSDP state handling; BF16 is the intended operating mode.
- `Lightning Fabric >=2.6.1`: keep the current distributed/runtime orchestration in [`safemoe/pretrain.py`](/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/safemoe/pretrain.py).
- `transformers >=4.51.3,<4.57`: authoritative loader and parity layer for `Qwen3MoeForCausalLM`, but not a second training loop.
- `huggingface-hub` + `safetensors`: required to fetch and read the 16-shard `Qwen3-30B-A3B-Base` checkpoint.
- `tensorboard` + `torchmetrics`: needed for per-split routing telemetry and consistent aggregation across `D_std`, `D_harmful`, and `D_unlabeled`.

**Explicit additions:**
- Qwen-specific intervention planner/applier and manifest I/O under `safemoe/interventions/`.
- Routing-supervision loss under `safemoe/losses/`.
- Shared routing observability helpers for train/eval under `safemoe/observability/`.

**Explicit non-additions:**
- No DeepSpeed.
- No HF `Trainer` / `Accelerate` as a new primary loop.
- No PEFT / LoRA / QLoRA as the intervention mechanism.
- No 8-bit/4-bit mainline training.
- No serving stack (`vLLM`, TGI) or new MoE frameworks.

### Expected Features

The milestone has a tight six-link chain. If any link is missing, the direct-Qwen thesis is not actually validated. Table stakes are therefore mostly experiment integrity features, not UI or platform features.

**Must have (table stakes):**
- Deterministic harmful expert/head initialization from the pretrained Qwen checkpoint, including router-column duplication and reproducible source-to-target mapping.
- Routing-supervised warmup on labeled `D_harmful` and `D_std`, with routing loss logged separately from LM loss.
- Mixed-data SGTM transfer on `D_unlabeled` + `D_harmful` + `D_std`, starting from a saved warmup state.
- Routing instrumentation that proves separation across baseline, warmup, transfer, and ablation checkpoints.
- Pre/post-ablation isolation evaluation showing harmful capability drops with bounded standard regression.
- One fixed adversarial-cost validation protocol after ablation.

**Should have (differentiators):**
- Direct large-model checkpoint intervention rather than another proxy-model proof.
- Routing-supervised warmup as a distinct bridge between initialization and SGTM transfer.
- Evaluation of unlabeled harmful routing drift, not only labeled harmful routing.
- Adversarial-cost framing tied to a fixed recovery budget.
- Metric continuity against the shipped v1.0 baseline.

**Defer (v1.x / v2+):**
- Broad warmup or transfer hyperparameter sweeps.
- Richer routing diagnostics beyond the core separation readouts.
- Expanded adversarial benchmark matrices.
- New harmful-data labeling systems, data-pipeline redesign, or broader architecture experimentation.
- Broader model-family replication beyond `Qwen3-30B-A3B-Base`.

### Architecture Approach

The architecture recommendation is to preserve a single SafeMoE training stack with five layers: CLI/experiment entrypoints, a thin Qwen intervention layer, the stable SafeMoE model layer, a stage-aware training layer, and evaluation/ablation extensions. `SafeMoEConfig` remains the canonical config object; `HarmfulParamRegistry` remains the single source of truth for `theta_harmful`, `theta_std`, and `theta_shared`; `safemoe/pretrain.py` remains the only trainer.

**Major components:**
1. `safemoe/interventions/*` — plan and apply expert/head/router cloning, noise injection, and manifest persistence.
2. `safemoe/config.py` + `safemoe/masking.py` — extend config and registry to classify cloned router columns and any head slices without breaking existing contracts.
3. [`safemoe/pretrain.py`](/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/safemoe/pretrain.py) — orchestrate surgery, warmup, and transfer in one run with stage-aware losses and checkpoints.
4. `safemoe/losses/routing_supervision.py` — compute warmup routing objectives on the correct router signal.
5. `safemoe/observability/routing.py`, [`safemoe/evaluate.py`](/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/safemoe/evaluate.py), and [`safemoe/ablate.py`](/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/safemoe/ablate.py) — collect stage-consistent routing/concentration metrics and manifest-aware ablation reports.

### Critical Pitfalls

1. **Semantically wrong checkpoint surgery** — require manifest logging, tensor-level checksum/cosine checks, and a parity harness before any training.
2. **Broken harmful/std/shared classification after wrapping** — test exhaustive, non-overlapping registry coverage on the real Qwen config and the Fabric-wrapped model.
3. **Router supervision on the wrong tensor or double-counted loss** — pin versions, assert router tensor shape explicitly, and log LM loss, custom routing loss, built-in aux loss, and total loss separately.
4. **Dataset confounds mistaken for harmful routing** — add pre-warmup audits and randomized-label / matched-control evals before trusting warmup separation.
5. **SGTM diffusion back into standard/shared params** — use short transfer intervals, periodic ablation checkpoints, and diffusion-budget metrics rather than long blind runs.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 0: Resource Envelope and Qwen Access
**Rationale:** The milestone depends on a 61.1 GB public checkpoint and BF16-class training resources; failing to validate disk, download, and runtime envelope early wastes later work.
**Delivers:** Verified model access, conversion dry run, storage budgeting, and a peak RAM/VRAM/tokens-per-second report.
**Addresses:** Stack additions, explicit non-additions, and operational readiness for direct-Qwen work.
**Avoids:** Resource-envelope failure and late-stage environment churn.

### Phase 1: Config, Manifest, and Checkpoint Surgery Parity
**Rationale:** Every later claim depends on semantically correct initialization. This is the first hard gate.
**Delivers:** Extended `SafeMoEConfig`, intervention manifest schema, Qwen surgery planner/applier, and parity tests for expert/head/router cloning.
**Addresses:** Harmful expert/head initialization and reproducible source mapping.
**Avoids:** Semantically wrong checkpoint surgery and hidden provenance loss.

### Phase 2: Registry Invariants and Routing Observability
**Rationale:** Warmup cannot be trusted until the model knows what counts as harmful and the team can observe true router behavior.
**Delivers:** Registry coverage tests on wrapped models, router-column classification, shared train/eval routing collectors, and hook-vs-output parity validation.
**Addresses:** Routing instrumentation and the harmful/std/shared parameter contract.
**Avoids:** Misclassification bugs and measuring the wrong dispatch signal.

### Phase 3: Warmup Stage Integration and Confound Control
**Rationale:** Routing-supervised warmup is the key new mechanism. It needs its own verification gate before SGTM transfer starts.
**Delivers:** Stage-aware warmup in `pretrain.py`, separated routing-loss logging, labeled `D_harmful`/`D_std` training path, dataset audits, and confound-controlled warmup eval.
**Addresses:** Warmup separation and milestone comparability against v1.0 metrics.
**Avoids:** Router-loss wiring errors and shortcut learning on non-harmful confounds.

### Phase 4: Mixed-Data SGTM Transfer with Diffusion Controls
**Rationale:** Transfer should only start from a known-good warmup checkpoint and must be instrumented to detect harmful-capability diffusion back into standard/shared parameters.
**Delivers:** Resumable mixed-data transfer, `D_unlabeled` routing-drift metrics, periodic ablation checkpoints, and conservative initial transfer schedules.
**Uses:** Existing SafeMoE SGTM loop, Fabric/FSDP, and the new observability layer.
**Implements:** Stage-aware training and checkpointing without a second trainer.

### Phase 5: Evaluation, Ablation, and Adversarial-Cost Gate
**Rationale:** The milestone is only complete if isolation survives ablation-based evaluation and a bounded retuning-cost check.
**Delivers:** Manifest-aware evaluation, sham/shared-only controls, pre/post-ablation reporting on all splits, and one fixed adversarial recovery protocol with a pinned budget.
**Addresses:** Isolation quality and the safety claim that ablation raises adversarial recovery cost.
**Avoids:** “Looks done but isn’t” conclusions based only on perplexity or unconstrained attack budgets.

### Phase Ordering Rationale

- The order follows the hard dependency chain from the research: config/manifest -> checkpoint surgery -> registry/observability -> warmup -> transfer -> evaluation.
- Checkpoint surgery and registry validation are intentionally separated from warmup so semantic errors are caught before optimization confounds the diagnosis.
- Warmup and transfer are split into different phases because warmup is a mechanism-validation gate, while transfer is a contamination-risk phase.
- Evaluation and adversarial-cost validation stay at the end, but ablation-aware checkpoints must exist during transfer so contamination can be localized.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 1:** Qwen head-slice cloning details and LitGPT/HF parity need careful implementation-level validation.
- **Phase 2:** Router signal choice is version-sensitive because of known Qwen3MoE router-logit issues and aux-loss ambiguity.
- **Phase 3:** Confound-controlled warmup evaluation needs explicit dataset-audit criteria, not just training wiring.
- **Phase 5:** Adversarial-cost protocol design needs a tightly defined, comparable recovery budget and controls.

Phases with standard patterns (skip research-phase):
- **Phase 0:** Storage/download/runtime envelope validation is operational and well-bounded.
- **Phase 4:** Once warmup, registry, and observability are correct, the transfer phase mostly reuses existing SGTM patterns with tighter checkpoint cadence.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Grounded in local repo pins and official Qwen/HF/PyTorch requirements; the non-addition list is unusually clear. |
| Features | HIGH | Directly aligned with milestone goals and tightly convergent across project context and feature research. |
| Architecture | HIGH | Strong fit with the existing SafeMoE/LitGPT codebase; recommended build order is explicit and dependency-driven. |
| Pitfalls | MEDIUM | Risks are credible and specific, but several depend on community issue reports and implementation inference rather than local reproduction. |

**Overall confidence:** HIGH

### Gaps to Address

- Exact head-cloning necessity and slice mapping: treat as a planning-time validation item; if head cloning proves unnecessary for the thesis, cut it early rather than carrying speculative complexity.
- Exact router tensor contract in the pinned library stack: verify during Phase 2 with an explicit parity harness before finalizing warmup loss code.
- Adversarial-cost budget design: define the fixed recovery ladder during planning so the milestone gate is stable before training begins.
- Storage and runtime headroom under real checkpoint conversion/resume behavior: validate with a dry run in Phase 0 rather than relying only on size estimates.
- Reproducibility floor: require at least two seeds and resume-vs-fresh consistency in the later evaluation phase, because the current pitfall analysis flags this as unresolved.

## Sources

### Primary (HIGH confidence)
- `.planning/PROJECT.md` — milestone scope, active requirements, and v1.0 continuity constraints
- `.planning/research/STACK.md` — stack decisions, explicit non-additions, and model-access requirements
- `.planning/research/FEATURES.md` — table stakes, differentiators, dependencies, and scope boundaries
- `.planning/research/ARCHITECTURE.md` — build order, component boundaries, and stable interface constraints

### Secondary (MEDIUM confidence)
- `.planning/research/PITFALLS.md` — milestone failure modes, controls, and phase-to-risk mapping
- Qwen quickstart and HF model/config docs referenced in the research files — version and model-format validation

### Tertiary (LOW confidence)
- Community issue reports referenced in `PITFALLS.md` about Qwen3MoE router-logit and aux-loss behavior — useful warnings, but must be locally validated in implementation

---
*Research completed: 2026-03-19*
*Ready for roadmap: yes*
