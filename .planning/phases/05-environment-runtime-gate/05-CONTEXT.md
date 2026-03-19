# Phase 5: Environment Runtime Gate - Context

**Gathered:** 2026-03-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Verify that the existing LitGPT-converted `Qwen3-30B-A3B-Base` checkpoint can be loaded through the direct-Qwen path and that the milestone baseline can complete BF16 startup plus one measured optimizer step with a recorded runtime envelope. This phase establishes environment/runtime readiness only; checkpoint surgery, routing observability, and later training behavior belong to Phases 6-10.

</domain>

<decisions>
## Implementation Decisions

### Gate path strictness
- Primary pass condition is a real BF16 `safemoe pretrain` dry-start from `checkpoints/Qwen3-30B-A3B-Base` that completes startup plus one optimizer step.
- Eval-only evidence does not satisfy Phase 5; if training-path dry-start fails, Phase 5 fails even if eval can load and run.
- Checkpoint validation stays lightweight before runtime execution: confirm required files exist and that the direct-Qwen model can load `lit_model.pth` cleanly.
- The phase should optimize for the earliest trustworthy proof that the milestone training path is viable, not for broad compatibility coverage.

### Execution topology to bless
- Phase 5 certifies one concrete execution topology only: the actual first milestone run shape expected for v1.1.
- The required precision baseline is `bf16-true`; mixed-precision variants are optional follow-up checks and do not define readiness.
- If the blessed topology works, non-primary modes such as alternative device counts or strategies can be deferred without blocking Phase 5.
- CPU-only or non-BF16 fallback paths are out of scope for the gate. They may be noted if discovered incidentally, but they do not satisfy the phase.

### Runtime envelope artifact
- Record one committed markdown report in the Phase 5 planning directory as the canonical runtime envelope artifact.
- The artifact must capture storage footprint, peak GPU memory, startup time, first-step time, and tokens/sec after the first measured step.
- One representative measured run is sufficient for Phase 5, provided the exact command and run shape are captured.
- The artifact must include enough replay context to reproduce the measurement: command, checkpoint path, topology, precision, seed, and notable environment assumptions.

### Failure boundary
- Phase 5 may pass with a narrow runtime envelope only if that narrow topology is the explicitly blessed milestone baseline and its limits are documented.
- Hard failures for Phase 5 are: missing required files, config/tokenizer incompatibility, model load failure, BF16 startup failure, or inability to complete one optimizer step on the blessed topology.
- Incidental warnings do not block the phase if the blessed path succeeds; they should be recorded for downstream planning.
- Likely future blockers discovered during the gate should be captured as downstream risks or deferred notes rather than expanding Phase 5 scope.

### Claude's Discretion
- Exact command-line shape used to exercise the dry-start, as long as it stays on the real `safemoe pretrain` path and completes one optimizer step.
- Exact markdown structure of the runtime-envelope report.
- Any minimal instrumentation or logging additions needed to expose the required measurements without broadening phase scope.

</decisions>

<specifics>
## Specific Ideas

- The existing checkpoint directory `checkpoints/Qwen3-30B-A3B-Base` is already the phase anchor and currently contains a `57G` `lit_model.pth` plus tokenizer/config files, so this phase should validate direct use rather than revisit conversion or acquisition.
- `safemoe/pretrain.py` already exposes the direct checkpoint initialization path via `initial_checkpoint_dir`, uses Lightning Fabric, and prints end-of-run memory/performance information that can anchor the Phase 5 artifact.
- The point of the gate is not “can Qwen run somewhere”; it is “can the actual v1.1 training path run in the blessed BF16 shape with one measured step and a usable budget.”

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase definition
- `.planning/ROADMAP.md` — Defines the fixed scope, requirements mapping, success criteria, and Phase 5 plan slots `05-01` and `05-02`.
- `.planning/REQUIREMENTS.md` — Defines `ENV-01` and `ENV-02`, which this phase must satisfy.
- `.planning/STATE.md` — Confirms v1.1 is at roadmap-created state and that Phase 5 is the current focus.
- `.planning/PROJECT.md` — Captures the direct-`Qwen3-30B-A3B-Base` milestone intent and the decision to avoid new platform/infra scope.

### Runtime entry points and checkpoint assets
- `safemoe/pretrain.py` — Direct training-path entry point, checkpoint init path, Fabric strategy selection, and existing runtime logging behavior.
- `checkpoints/Qwen3-30B-A3B-Base/model_config.yaml` — Pinned model configuration for the direct-Qwen checkpoint.
- `checkpoints/Qwen3-30B-A3B-Base/` — Canonical checkpoint directory to validate and measure in this phase.

### Codebase conventions
- `.planning/codebase/STACK.md` — Confirms the runtime stack and dependency expectations around PyTorch, Lightning, and BF16-capable training environments.
- `.planning/codebase/INTEGRATIONS.md` — Captures logging/integration expectations and the local-filesystem checkpoint model relevant to this phase.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `safemoe/pretrain.py setup()` accepts `initial_checkpoint_dir`, `precision`, `devices`, and `num_nodes`, which is the exact path Phase 5 needs to validate rather than creating a separate loader.
- `safemoe/pretrain.py main()` already constructs the model, calls `fabric.load_raw(initial_checkpoint_dir / "lit_model.pth", model)`, and prints total parameters plus end-of-run memory/performance summaries.
- `litgpt.utils.extend_checkpoint_dir` is already used in `safemoe/pretrain.py`, so checkpoint-path resolution should follow existing conventions.
- The checkpoint directory already includes `lit_model.pth`, `model_config.yaml`, and tokenizer/config files, reducing Phase 5 work to validation and measurement.

### Established Patterns
- Single-device runs use Fabric strategy `"auto"` while multi-device runs switch to FSDP `HYBRID_SHARD`; Phase 5 should bless one of these concretely rather than treating both as mandatory.
- Precision is already configurable in the training entry point, so `bf16-true` can be pinned without adding a second stack.
- Existing training code logs throughput and reports `torch.cuda.max_memory_allocated()` at the end of execution, which can be reused or minimally adapted for the envelope artifact.

### Integration Points
- Plan `05-01` should attach to checkpoint presence/loadability checks around `checkpoints/Qwen3-30B-A3B-Base` and the `fabric.load_raw(...)` path in `safemoe/pretrain.py`.
- Plan `05-02` should attach to the actual dry-start invocation path in `safemoe/pretrain.py`, using the blessed topology and producing the committed runtime report in this phase directory.
- Any instrumentation added in Phase 5 should be narrow and support later planning for Phases 6-10 by exposing a trustworthy memory and throughput envelope.

</code_context>

<deferred>
## Deferred Ideas

- Broader topology certification across multiple device counts or strategies.
- CPU-only, non-BF16, or mixed-precision fallback qualification.
- Deeper checkpoint introspection or schema auditing beyond what is needed to prove loadability.
- Any checkpoint surgery, registry validation, routing observability, warmup, transfer, or evaluation behavior beyond the one-step environment/runtime gate.

</deferred>

---

*Phase: 05-environment-runtime-gate*
*Context gathered: 2026-03-19*
