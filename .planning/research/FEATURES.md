# Feature Research

**Domain:** Direct harmful-transfer experiments on `Qwen3-30B-A3B-Base`
**Researched:** 2026-03-19
**Confidence:** HIGH

## Feature Landscape

### Table Stakes (Users Expect These)

Features the milestone must ship or the direct-`Qwen3-30B-A3B-Base` thesis is not actually tested.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Harmful-expert and harmful-head initialization from the pretrained checkpoint | The milestone exists to test direct intervention on `Qwen3-30B-A3B-Base`, not another proxy architecture. If experts, heads, and router columns cannot be cloned into a designated harmful subspace, the milestone goal is unmet. | HIGH | Must clone selected experts and attention heads from `theta_std`, duplicate the matching router pathways, add controlled noise, and preserve a reproducible mapping of source -> harmful components. This should be requirementized as deterministic checkpoint surgery plus verification that tensor shapes, routing indices, and load paths remain valid. |
| Warmup stage with explicit routing supervision on `D_harmful` and `D_std` | The project already knows harmful knowledge exists in standard parameters. A supervised warmup is the minimum mechanism needed to begin separating traffic before transfer. | HIGH | Must support a dedicated pre-transfer stage that mixes only labeled `D_harmful` and `D_std`, applies a routing loss that attracts harmful tokens to harmful experts and repels standard tokens, and logs the routing-separation signal independently from next-token loss. |
| Mixed-data SGTM transfer on `D_unlabeled` + `D_harmful` + `D_std` | Warmup alone is not the milestone claim. The claim is that transfer can concentrate harmful capability into `theta_harmful` under mixed-data training. | HIGH | Must run after warmup using the existing SGTM split-aware training loop, but now on direct-Qwen harmful components. Requirement should be framed as a resumable transfer phase with explicit batch accounting for all three streams and checkpointed before/after ablation evaluation. |
| Routing instrumentation that can prove separation happened | This milestone cannot rely on loss curves alone. It needs evidence that harmful and standard tokens route differently after warmup and after transfer. | MEDIUM | Must emit per-split router statistics for designated harmful experts versus standard experts, including token counts or routing mass, and support comparison across baseline, post-warmup, post-transfer, and post-ablation checkpoints. |
| Evaluation bundle for isolation quality | A direct-intervention milestone is incomplete without showing whether isolation worked behaviorally. | MEDIUM | Must report pre/post-ablation metrics for `D_harmful`, `D_std`, and relevant unlabeled probes, plus a direct comparison against the shipped v1.0 baseline. The minimum success shape is: harmful capability drops materially after ablating `theta_harmful`, while standard capability remains near intact. |
| Adversarial-cost validation after ablation | The milestone goal explicitly includes raising adversarial retuning cost. If this is not measured, the safety claim remains incomplete. | HIGH | Must include a reproducible post-ablation adversarial recovery experiment or proxy cost measure with a fixed budget. Requirement should pin down the budget, dataset slice, number of update steps, and success criterion so the outcome is comparable across runs. |

### Differentiators (Competitive Advantage)

Features that make this milestone more than a straightforward port of the existing SafeMoE pipeline.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Direct large-model checkpoint intervention instead of re-validating on a smaller proxy | Demonstrates that the thesis survives contact with a real pretrained MoE model and avoids another round of architecture-only proof. | HIGH | The milestone should treat direct Qwen surgery as a first-class experiment, not as a preparatory artifact for later work. This is the main differentiator relative to v1.0. |
| Routing-supervised warmup before SGTM transfer | Creates a controlled bridge between pretrained harmful knowledge already present in `theta_std` and the later transfer stage. This is the key new mechanism, not just a rerun of existing SGTM code. | HIGH | The differentiator is the explicit staged behavior: initialize harmful capacity, force early routing separation, then hand off to mixed-data transfer. |
| Separation metrics on unlabeled harmful content | The important research question is not only whether labeled harmful data can be isolated, but whether the routing bias persists on unlabeled harmful examples. | MEDIUM | This should be framed as an evaluation differentiator, not a training prerequisite. The requirement is to measure natural routing of unlabeled harmful data after warmup/transfer. |
| Adversarial-cost framing rather than only perplexity framing | Makes the result more meaningful for safety claims. A model that forgets under ablation but is trivially re-poisoned has limited value. | HIGH | This differentiator matters only if the experiment budget is fixed and comparable. It should remain narrow: retuning cost, not a full red-team platform. |
| End-to-end milestone comparability against the shipped v1.0 baseline | Keeps the new Qwen path scientifically interpretable instead of becoming a disconnected experiment. | LOW | Every major readout should be comparable to the existing routing and ablation evaluations where possible, even if absolute numbers differ. |

### Anti-Features (Commonly Requested, Often Problematic)

Features that would expand the milestone without increasing confidence in the core thesis.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| New general-purpose data pipeline redesign | Direct-Qwen work often tempts a broader refactor once new checkpoint and routing needs appear. | This milestone already has validated proxy data plumbing. Reworking loaders, dataset abstractions, or tokenization beyond what Qwen compatibility strictly needs adds risk without testing the new thesis. | Limit changes to the minimal Qwen-compatible path and keep the existing split semantics. |
| Broad architecture experimentation beyond harmful expert/head injection | It is tempting to test new router types, different MoE layouts, or unrelated transformer changes while touching Qwen internals. | That would make negative results uninterpretable because the intervention path and the base model would both be changing. | Hold architecture fixed and isolate only the harmful expert/head initialization plus routing supervision. |
| Full hyperparameter sweep of warmup and transfer settings | Research work often drifts into large search grids once a new training stage exists. | A broad sweep turns one milestone into an open-ended optimization project and delays the basic yes/no answer on feasibility. | Define one primary configuration and, at most, a small sanity sweep around routing-loss weight or warmup length. |
| Production-grade serving or chat evaluation harness | Direct work on a large model makes deployment and demo tooling look attractive. | Serving infrastructure does not establish isolation quality, routing separation, or adversarial retuning cost. It is pure scope creep here. | Reuse offline evaluation scripts and batch metrics. |
| General harmfulness detection or classifier training | Once harmful/standard routing is in scope, adding automatic harmful-content labeling can look adjacent. | Detection is a separate problem and would contaminate the interpretation of transfer results by mixing labeling error into the intervention study. | Keep using the controlled bilingual proxy and explicit split labels. |
| Open-ended adversarial training benchmark suite | The milestone needs a cost validation, but a large benchmark matrix would dominate the schedule. | Too many attack recipes or recovery protocols would blur the requirement and make results hard to compare. | Pick one fixed adversarial recovery protocol with a defined budget and use it as the milestone gate. |

## Feature Dependencies

```text
Harmful component initialization
    -> Routing-supervised warmup
        -> Routing separation instrumentation
        -> Warmup checkpoint selection
            -> Mixed-data SGTM transfer
                -> Post-transfer routing evaluation
                -> Harmful ablation evaluation
                    -> Adversarial-cost validation

Existing validated proxy data pipeline
    -> Warmup data assembly (`D_harmful` + `D_std`)
    -> Transfer data assembly (`D_unlabeled` + `D_harmful` + `D_std`)

Baseline v1.0 routing + ablation evaluation
    -> Cross-milestone comparability for Qwen results
```

### Dependency Notes

- **Harmful component initialization requires no prior warmup logic:** the model must first contain designated harmful experts, heads, and router columns before any routing supervision can target them.
- **Routing-supervised warmup requires labeled `D_harmful` and `D_std`:** without clean labeled streams, the routing loss has no trustworthy attract/repel signal.
- **Mixed-data SGTM transfer requires a saved warmup state:** transfer should start only after there is measurable routing separation, otherwise the transfer stage is confounded with failed initialization.
- **Instrumentation is coupled to both warmup and transfer:** the same routing readouts should be available across stages so phase-to-phase changes are comparable instead of being inferred from different metrics.
- **Adversarial-cost validation requires ablation first:** the question is whether the ablated model is harder to re-poison, not whether the unablated model can be tuned.
- **Cross-milestone comparability depends on reusing existing evaluation semantics:** if Qwen metrics are defined differently from v1.0, the milestone loses its baseline.

## MVP Definition

### Launch With (v1)

Minimum milestone scope needed to validate the direct-Qwen thesis.

- [ ] Deterministic harmful-expert and harmful-head initialization on `Qwen3-30B-A3B-Base` with reproducible source mapping and loadable checkpoints.
- [ ] Routing-supervised warmup on mixed `D_harmful` and `D_std` with a measurable separation metric.
- [ ] Mixed-data SGTM transfer on `D_unlabeled`, `D_harmful`, and `D_std` starting from the warmup checkpoint.
- [ ] Routing instrumentation that reports harmful-versus-standard expert usage across baseline, warmup, transfer, and ablation.
- [ ] Pre/post-ablation evaluation proving the harmful capability drop and bounded standard-capability regression.
- [ ] One fixed adversarial-cost validation protocol after ablation.

### Add After Validation (v1.x)

- [ ] Small warmup sensitivity sweep such as routing-loss weight or warmup duration, only after the primary configuration works.
- [ ] More granular routing diagnostics such as per-layer or per-head separation summaries, if the base instrumentation indicates ambiguous behavior.
- [ ] Additional adversarial recovery budgets or attack recipes, once the single gate protocol is stable.

### Future Consideration (v2+)

- [ ] Replace the bilingual proxy with a more realistic harmful corpus once the direct intervention path is validated.
- [ ] Broader model-family replication beyond `Qwen3-30B-A3B-Base`.
- [ ] Automation for multi-run statistical confidence and broader hyperparameter exploration.

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Harmful-expert and harmful-head initialization | HIGH | HIGH | P1 |
| Routing-supervised warmup | HIGH | HIGH | P1 |
| Mixed-data SGTM transfer | HIGH | HIGH | P1 |
| Routing instrumentation | HIGH | MEDIUM | P1 |
| Pre/post-ablation isolation evaluation | HIGH | MEDIUM | P1 |
| Adversarial-cost validation | HIGH | HIGH | P1 |
| Warmup sensitivity sweep | MEDIUM | MEDIUM | P2 |
| Fine-grained routing diagnostics | MEDIUM | MEDIUM | P2 |
| Multi-protocol adversarial benchmark expansion | LOW | HIGH | P3 |

**Priority key:**
- P1: Must have for milestone validation
- P2: Add only after the primary experiment works
- P3: Explicitly defer

## Scope Boundary Summary

This milestone should be scoped as a chained experiment with six must-have capabilities:

1. checkpoint intervention,
2. warmup separation,
3. mixed-data transfer,
4. routing observability,
5. ablation-based isolation evaluation,
6. adversarial-cost validation.

Anything outside that chain is probably scope creep unless it is strictly required for Qwen compatibility.

## Sources

- [PROJECT.md](/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/.planning/PROJECT.md) - milestone definition and active requirements
- [SUMMARY.md](/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/.planning/research/SUMMARY.md) - prior research constraints, baseline architecture, and evaluation context

---
*Feature research for: direct harmful-transfer on `Qwen3-30B-A3B-Base`*
*Researched: 2026-03-19*
