# Phase 7: Registry and Routing Observability - Research

**Researched:** 2026-03-19
**Domain:** Manifest-aware parameter ownership reporting plus shared routing telemetry on the direct-Qwen SafeMoE stack
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
### Registry ownership semantics
- Router and gate weights stay in `theta_shared`, even when Phase 6 cloned router columns from standard to harmful slots.
- Fused `attn.qkv.weight` ownership must become researcher-visible slice-level data rather than remaining implicit internal metadata.
- Registry output must be both exhaustive and non-overlapping across `theta_harmful`, `theta_std`, and `theta_shared`.
- Registry output must produce both a machine-readable inventory and a short grouped Markdown summary.

### Routing metric contract
- The canonical routing bundle is raw dispatch counts plus harmful-routing fraction.
- Phase 7 must establish one shared observability path reused by baseline, warmup, transfer, and ablation evaluation.
- Phase 7 stays at expert-level routing only; per-head or per-layer routing diagnostics are deferred.
- If a stage cannot naturally emit some routing metrics, omit them rather than writing placeholder values.

### Signal-vs-dispatch parity bar
- Parity means exact equality between the logged routing signal and the actual dispatched harmful-expert behavior.
- Parity is required in targeted verification/evaluation flows, not necessarily every telemetry-emitting run.
- Any parity mismatch is a hard failure in the parity-checking flow.

### Output constraints
- Registry reports and routing reports stay separate artifacts.
- Routing artifacts live next to the checkpoint or run that produced them.
- Markdown output should be short and researcher-signoff oriented.

### Claude's Discretion
- Exact artifact filenames, JSON field names, and helper/module split.
- Whether the shared routing path lives in a new `safemoe/observability.py` module or is embedded into existing files, as long as train/eval share the same collector and artifact schema.

### Deferred Ideas (OUT OF SCOPE)
- Attention-head routing observability.
- Per-layer or per-split parity drilldowns.
- New routing objectives or training behavior changes.
- Global dashboards or cross-run aggregation stores.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ROUT-01 | Researcher can classify direct-Qwen parameters into `theta_harmful`, `theta_std`, and `theta_shared` with exhaustive, non-overlapping registry coverage after the intervention is applied. | Extend `HarmfulParamRegistry` to emit an inspectable inventory keyed by parameter name plus slice-level `qkv` records, and write `registry_inventory.json` + `registry_summary.md` next to the inspected checkpoint/run. |
| ROUT-02 | Researcher can capture per-split routing metrics for designated harmful experts during baseline, warmup, transfer, and ablation evaluation using one shared observability path. | Extract the current eval-only dispatch hook into one shared collector/writer used by both `safemoe/evaluate.py` and `safemoe/pretrain.py`, with one artifact schema containing raw dispatch counts and harmful-routing fractions. |
| ROUT-03 | Researcher can verify that the routing signal used for supervision matches the real dispatched expert behavior for the pinned Qwen stack. | Add a parity flow that compares the logged routing metrics against aggregate dispatch counts produced by the same collector and raises a hard failure on any mismatch. |
</phase_requirements>

## Summary

The codebase already contains the two primitives Phase 7 needs. First, `safemoe/masking.py` already computes exhaustive, non-overlapping parameter grouping by `id()` and already knows the exact harmful/std row slices inside fused `attn.qkv.weight`. Second, `safemoe/evaluate.py` already captures dispatch behavior from `SafeMoELayer._last_indices` and turns it into harmful-routing fractions. Phase 7 should therefore standardize and expose existing semantics instead of inventing new registry or routing abstractions.

The right split is the roadmap split: Plan `07-01` should make registry ownership inspectable for researchers, including router-column provenance from the Phase 6 manifest and first-class `qkv` slice visibility. Plan `07-02` should extract one shared routing observability collector used by both evaluation and training/checkpoint flows so warmup, transfer, and ablation all produce the same raw counts and harmful-routing fractions. The parity check should be implemented on top of that same collector, not as a second independently computed signal.

**Primary recommendation:** plan Phase 7 as two plans. First, extend `HarmfulParamRegistry` with artifact-grade inventory and summary writers that preserve `theta_shared` router semantics while surfacing manifest-derived provenance and `qkv` slices. Second, introduce one shared routing observability module and wire it into both `safemoe/evaluate.py` and `safemoe/pretrain.py`, including an explicit parity-checking path that fails hard on mismatch.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| local `litgpt` fork | `0.5.12` | Canonical Qwen3/SafeMoE model structure and checkpoint layout | `HarmfulParamRegistry`, `SafeMoELayer`, and checkpoint loading already depend on it. |
| `torch` | `2.10.0+cu128` | Parameter inspection, tensor counting, and deterministic metric math | Dispatch counting and parity checks stay in PyTorch tensors/lists already used by the repo. |
| `lightning` | `2.6.1` | Existing Fabric logging and checkpoint flow | Shared routing observability must plug into current training/eval execution instead of a second runner. |
| `pytest` | `9.0.2` | Nyquist test coverage for registry artifacts and routing parity | Existing registry/evaluate/pretrain tests already cover the same surfaces Phase 7 extends. |

### Existing Code Surfaces
| File | What already exists | Phase 7 implication |
|------|---------------------|---------------------|
| `safemoe/masking.py` | Exhaustive `theta_harmful` / `theta_std` / `theta_shared` grouping, plus hidden `qkv` slice metadata | Promote those internals into explicit inventory/report outputs rather than redoing classification. |
| `safemoe/model.py` | `SafeMoELayer.forward()` stores `self._last_indices` on every call | Shared routing observability can keep using forward hooks; no router rewrite is needed. |
| `safemoe/evaluate.py` | `routing_attribution()` already collects dispatch indices and writes JSON | Refactor this into a reusable collector/schema rather than leaving eval as the only routing path. |
| `safemoe/pretrain.py` | Current split-aware metric logging and checkpoint save intervals | Warmup/transfer observability should attach here and write stage-local routing artifacts beside checkpoints. |
| `safemoe/interventions/manifest.py` | Explicit expert/head source-target mappings and derived router-column pairs | Registry summaries can annotate router provenance without reclassifying router weights as harmful. |

## Architecture Patterns

### Pattern 1: Registry inventory on top of `HarmfulParamRegistry`
**What:** Keep `HarmfulParamRegistry` as the source of truth, but add artifact-facing methods that return named inventory records and grouped summaries.
**When to use:** `ROUT-01`.
**Example target shape:**
```python
{
    "parameter_name": "transformer.h.0.mlp.gate.weight",
    "ownership": "theta_shared",
    "category": "router_gate",
    "shape": [128, 4096],
    "manifest_provenance": {
        "derived_router_column_pairs": [[12, 0], [47, 1]]
    },
}
```

### Pattern 2: First-class fused-`qkv` slice records
**What:** Emit slice records for harmful/std attention ownership while still keeping the full `qkv.weight` parameter in exhaustive registry coverage.
**When to use:** `ROUT-01`.
**Why:** The current hidden metadata is correct for masking but insufficient for researcher sign-off.
**Example target shape:**
```python
{
    "parameter_name": "transformer.h.0.attn.qkv.weight",
    "ownership": "theta_harmful",
    "category": "attn_qkv_slice",
    "slice_role": "harmful",
    "slice_rows": [[0, 128], [4096, 4224], [4608, 4736]],
}
```

### Pattern 3: One shared routing collector
**What:** Move dispatch accumulation, harmful-count aggregation, artifact writing, and Markdown summary generation behind one shared collector used by train and eval.
**When to use:** `ROUT-02`.
**Why:** `safemoe/evaluate.py` already proves the collection mechanism works; the missing part is shared reuse and a common artifact schema.
**Recommended module split:** add `safemoe/observability.py` containing `RoutingObservabilityCollector`, `write_routing_artifacts()`, and a parity helper; keep `routing_attribution()` as a thin compatibility wrapper around the shared collector.

### Pattern 4: Parity on the same collector output
**What:** Compare logged routing metrics against aggregate counts from the collector and fail if they differ.
**When to use:** `ROUT-03`.
**Why:** The phase requirement is parity between the supervised/logged signal and real dispatch behavior, not parity between two unrelated implementations.
**Exact contract:** parity compares aggregate harmful-routing fraction and aggregate harmful/raw dispatch counts for each available split; any mismatch raises a `ValueError` and writes a FAIL artifact.

### Pattern 5: Stage-local artifact layout
**What:** Save observability files beside the checkpoint or run that produced them.
**When to use:** always.
**Recommended layout:**
```text
checkpoints/<artifact>/
├── registry_inventory.json
├── registry_summary.md
├── routing_observability.json
├── routing_observability.md
└── routing_parity.json   # only for parity-checking flows
```

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Parameter ownership classification | A second registry implementation in eval/train code | `HarmfulParamRegistry` plus artifact helpers | Avoids semantic drift between masking and observability. |
| Routing signal source | New router instrumentation or model surgery | Existing `SafeMoELayer._last_indices` + forward hooks | The model already exposes the dispatched expert indices needed for observability. |
| Router provenance | Reclassifying router columns as harmful | Manifest-derived annotations on shared router weights | Locked decisions require router weights to remain `theta_shared`. |
| Parity verification | Separate “expected routing” simulation | Same collector output compared against logged metrics | Prevents false agreement between two independent approximations. |

## Common Pitfalls

### Pitfall 1: Making `qkv` slices visible by double-classifying the full parameter
**What goes wrong:** The full `qkv.weight` lands in both `theta_harmful` and `theta_std`, breaking registry disjointness.
**How to avoid:** Keep exhaustive parameter grouping exactly as today; add separate slice records for visibility.

### Pitfall 2: Reclassifying router columns as `theta_harmful`
**What goes wrong:** Researcher-visible provenance changes the semantic contract of `theta_shared`.
**How to avoid:** Report router provenance as annotations in the artifact while keeping the actual parameter ownership unchanged.

### Pitfall 3: Leaving eval and train on different routing schemas
**What goes wrong:** Warmup/transfer metrics cannot be compared to eval metrics, defeating `ROUT-02`.
**How to avoid:** Put artifact generation and metric field names behind one collector/writer used in both codepaths.

### Pitfall 4: Checking parity against rounded/log-friendly values only
**What goes wrong:** Formatting or omission hides a real mismatch.
**How to avoid:** Compare raw counts and exact unrounded fractions before writing PASS.

### Pitfall 5: Writing placeholder keys for missing splits
**What goes wrong:** Downstream consumers cannot tell whether a metric is truly absent or just unsupported.
**How to avoid:** Omit unavailable split keys entirely, per locked decision.

## Open Questions

1. **Where should the shared routing collector live?**
   - Best answer: a small `safemoe/observability.py` module reused by `evaluate.py` and `pretrain.py`.
   - Why: avoids coupling artifact/report logic to either CLI surface.

2. **How much manifest provenance belongs in registry artifacts?**
   - Best answer: include derived router-column pairs and enough source-target context to explain why router weights remain shared.
   - Why: this closes the semantic gap left by Phase 6 without changing the grouping rules.

3. **Which training points should emit routing artifacts?**
   - Best answer: checkpoint/eval boundaries only, especially at save intervals and parity-checking validation flows.
   - Why: satisfies comparability while avoiding per-step overhead.

## Validation Architecture

### Test Framework
- `pytest 9.0.2`
- Existing relevant files: `tests/safemoe/test_registry.py`, `tests/safemoe/test_evaluate.py`, `tests/safemoe/test_pretrain.py`

### Phase Requirements -> Test Map
| Requirement | Coverage target | Automated command |
|-------------|-----------------|-------------------|
| ROUT-01 | Registry inventory is exhaustive, non-overlapping, exposes first-class `qkv` slice rows, and writes both JSON + Markdown summaries. | `pytest tests/safemoe/test_registry.py -k "inventory or qkv or summary" -x` |
| ROUT-02 | Shared routing collector writes one artifact schema with raw dispatch counts and harmful fractions from eval and pretrain/checkpoint flows. | `pytest tests/safemoe/test_evaluate.py -k "routing" -x` |
| ROUT-03 | Parity flow fails hard when logged routing metrics diverge from dispatched counts. | `pytest tests/safemoe/test_pretrain.py -k "routing parity" -x` |

### Sampling Rate
- After every registry-task commit: run the narrowest registry test first.
- After every routing-task commit: run the narrowest routing/parity test first.
- After the phase wave completes: run `pytest tests/safemoe/test_registry.py tests/safemoe/test_evaluate.py tests/safemoe/test_pretrain.py -x`.
- Keep feedback latency under ~25 seconds by preferring targeted tests during tasks.

### Wave 0 Gaps
- `tests/safemoe/test_registry.py` does not yet cover JSON/Markdown artifact writing or first-class `qkv` slice inventory records.
- `tests/safemoe/test_evaluate.py` currently validates eval-only routing JSON, not a shared artifact schema with raw dispatch counts.
- `tests/safemoe/test_pretrain.py` does not yet validate a routing parity hard-failure path.

## Sources

### Primary (HIGH confidence)
- Local source inspection on 2026-03-19:
  - `safemoe/masking.py`
  - `safemoe/model.py`
  - `safemoe/evaluate.py`
  - `safemoe/pretrain.py`
  - `safemoe/interventions/manifest.py`
  - `tests/safemoe/test_registry.py`
  - `tests/safemoe/test_evaluate.py`

## Metadata

- Research mode: local codebase inspection
- External browsing: none
- Downstream plans recommended: `07-01`, `07-02`
