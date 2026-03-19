# Phase 7: Registry and Routing Observability - Context

**Gathered:** 2026-03-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver trustworthy researcher-facing observability for the direct-Qwen intervention state after Phase 6: exhaustive, non-overlapping registry coverage across `theta_harmful`, `theta_std`, and `theta_shared`, plus one shared routing observability path that reports comparable harmful-expert routing metrics and verifies that the logged routing signal matches real dispatched behavior on the pinned Qwen stack. This phase does not add new training behavior, new routing objectives, or broader platform infrastructure.

</domain>

<decisions>
## Implementation Decisions

### Registry ownership semantics
- All router/gate weights remain classified as `theta_shared`, including router columns cloned during Phase 6. They may be reported separately for provenance, but they do not become `theta_harmful`.
- Fused attention `qkv.weight` ownership must be exposed at harmful/std slice granularity as first-class registry entries for researcher inspection, not only as hidden masking metadata.
- `theta_shared` continues to mean the leftover bucket for every parameter that is neither harmful nor standard expert/head-specific.
- Registry outputs must provide both:
  - an authoritative parameter inventory
  - a grouped summary by ownership/category for quick inspection

### Routing metric contract
- The canonical routing metric bundle is:
  - harmful-routing fraction
  - raw dispatch counts
- Phase 7 should establish one shared observability path and one shared artifact layout that can be reused across baseline, warmup, transfer, and ablation stages.
- Metrics should remain expert-level only in Phase 7; attention-head routing-like observability is out of scope here.
- If a given stage or flow does not naturally produce some routing metrics, those metrics should be omitted rather than emitted as placeholder or `not_applicable` values.

### Signal-vs-dispatch parity bar
- Parity means exact equality between the logged routing signal and the actual dispatched harmful-expert behavior.
- Parity checks are required in targeted verification/evaluation flows, not necessarily on every telemetry-emitting run path.
- Any parity mismatch is a hard failure for the parity-checking flow.
- Aggregate whole-run evidence is sufficient; per-split parity breakdowns are not required for Phase 7.

### Researcher-facing output format
- Registry coverage must produce both a JSON artifact and a Markdown summary.
- Routing observability artifacts should live alongside the checkpoint/run that produced them, with stage-specific outputs rather than one consolidated analysis directory.
- Registry and routing reports should be saved as separate artifacts, not merged into one package.
- The Markdown summaries should optimize for quick sign-off rather than audit-style exhaustive narration.

### Claude's Discretion
- Exact artifact filenames and naming conventions for the registry inventory/summary and routing telemetry/parity outputs.
- Exact JSON schema field names, as long as slice-level registry ownership, raw dispatch counts, harmful-routing fractions, and pass/fail parity outcomes are unambiguous.
- Exact Markdown layout, provided it stays short and sign-off oriented.

</decisions>

<specifics>
## Specific Ideas

- The researcher should be able to inspect post-intervention ownership without rereading masking internals; slice-level attention ownership needs to be visible as first-class registry data.
- Phase 7 should repair the current semantic gap where router-column provenance exists in the Phase 6 manifest, but the runtime registry still treats all router weights as undifferentiated shared parameters.
- The routing path should unify training/eval stage reporting around one comparable contract rather than leaving Phase 4's eval-only routing attribution as the de facto interface.

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase definition
- `.planning/ROADMAP.md` — Defines the fixed Phase 7 scope, `ROUT-01/02/03`, and the plan slots `07-01` and `07-02`.
- `.planning/REQUIREMENTS.md` — Defines the registry coverage, shared routing-path, and parity requirements this phase must satisfy.
- `.planning/PROJECT.md` — Confirms the milestone stays on the direct-Qwen research path without new platform scope.
- `.planning/STATE.md` — Carries forward the milestone dependency chain `environment/runtime -> checkpoint surgery -> registry/observability -> warmup -> transfer -> evaluation`.

### Upstream phase context
- `.planning/phases/02-model-architecture-masking/02-CONTEXT.md` — Locks the original harmful/std/shared registry semantics and the special-case treatment of fused `qkv.weight`.
- `.planning/phases/04-ablation-evaluation/04-CONTEXT.md` — Defines the current routing attribution expectations and metric naming patterns that Phase 7 should tighten into a shared observability path.
- `.planning/phases/06-checkpoint-surgery/06-CONTEXT.md` — Locks manifest-driven source/target layout, derived router-column mappings, and the post-surgery checkpoint as the downstream input artifact for this phase.

### Current code surfaces
- `safemoe/masking.py` — Current `theta_harmful`/`theta_std`/`theta_shared` classification and qkv slice metadata behavior that Phase 7 will expand for researcher-visible coverage.
- `safemoe/evaluate.py` — Current eval-only routing attribution path and artifact shape that needs to become part of the shared Phase 7 observability contract.
- `safemoe/pretrain.py` — Existing training/eval logging path, metric names, and checkpoint/run layout where shared routing observability will need to integrate cleanly.
- `safemoe/interventions/manifest.py` — Canonical provenance source for Phase 6 source/target mappings, including the derived router-column relationship.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `HarmfulParamRegistry` in `safemoe/masking.py` already scans named parameters and validates exhaustive, non-overlapping grouping by `id()`, which is the right foundation for Phase 7 registry reporting.
- The qkv metadata already computed in `safemoe/masking.py` provides a concrete starting point for turning harmful/std attention ownership into first-class registry artifacts.
- `routing_attribution()` in `safemoe/evaluate.py` already hooks `SafeMoELayer` dispatch state and computes harmful-expert fractions; Phase 7 can evolve this path instead of inventing a second routing collector.
- Phase 6 manifests in `safemoe/interventions/manifest.py` already encode explicit source/target expert/head mappings and derived router-column relationships that can anchor registry provenance reporting.

### Established Patterns
- The codebase already favors paired machine-readable plus human-readable artifacts for checkpoint mutation and verification work; Phase 7 should follow that pattern for registry coverage.
- Checkpoint/run-local artifacts are already the norm (`results.json`, surgery manifests, verification reports), so routing observability should attach to the producing stage/checkpoint rather than a new global store.
- Metric naming is currently flat and explicit (`loss_D_std`, `routing_harmful_frac_D_std`, `ablated_val_ppl_D_std`); Phase 7 should preserve that readability when extending the routing contract.

### Integration Points
- Plan `07-01` should attach to `safemoe/masking.py` plus manifest-aware provenance so the registry can expose exhaustive harmful/std/shared ownership, including first-class attention slice entries.
- Plan `07-02` should attach to the existing routing collection path in `safemoe/evaluate.py` and the training/eval logging seams in `safemoe/pretrain.py` to create one shared routing observability contract.
- The parity-checking flow should consume the same routing collector used for observability and compare aggregate logged values against aggregate dispatched counts, failing hard on mismatch.

</code_context>

<deferred>
## Deferred Ideas

- Attention-head routing-like observability beyond expert-level dispatch metrics.
- Per-split parity reports as a formal Phase 7 requirement.
- Any new training objective, supervision path, or warmup/transfer behavior changes.
- Consolidating stage artifacts into one global observability dashboard or analysis directory.

</deferred>

---

*Phase: 07-registry-and-routing-observability*
*Context gathered: 2026-03-19*
