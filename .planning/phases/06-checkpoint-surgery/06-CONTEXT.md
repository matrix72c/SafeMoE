# Phase 6: Checkpoint Surgery - Context

**Gathered:** 2026-03-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Create a deterministic, manifest-driven checkpoint-surgery flow for `Qwen3-30B-A3B-Base` that clones designated experts, attention heads, and matching router columns into `theta_harmful`, adds controlled noise, and emits a loadable post-surgery checkpoint plus verification artifacts. This phase covers initialization correctness only; registry expansion, routing observability, warmup, transfer, and evaluation belong to later phases.

</domain>

<decisions>
## Implementation Decisions

### Harmful slot layout
- Harmful expert target slots are reused across all layers rather than chosen per layer.
- Harmful attention head targets use one global head set across all layers rather than per-layer head targets.
- Different manifests may choose different harmful target layouts across runs; the layout is not globally fixed for the milestone.
- Each manifest must record the exact chosen target experts and target heads explicitly rather than relying only on a selection policy or seed.

### Clone perturbation policy
- Phase 6 should produce lightly perturbed clones rather than exact copies.
- One shared noise scale applies across cloned experts, cloned attention-head slices, and cloned router columns.
- Noise application must be deterministic from the manifest seed so rerunning the same manifest follows the same perturbation recipe.
- Zero-noise manifests are out of scope for this phase; every valid Phase 6 surgery artifact must use nonzero perturbation.

### Router inheritance
- Matching router columns are cloned immediately during Phase 6 rather than deferred to warmup.
- Router-column copies receive the same shared deterministic noise as the cloned experts and cloned head slices.
- The manifest does not need redundant explicit router-column mappings when they are implied by the chosen source and target layout.
- A harmful layout must come from one coherent source bundle; head clones and expert clones should not mix unrelated source layouts within one manifest.

### Verification bar
- Phase 6 verification only needs to prove that the post-surgery checkpoint reloads successfully and that tensor shapes plus manifest-declared mappings match.
- Deterministic replay is useful for implementation discipline but is not itself an acceptance criterion for this phase.
- Verification output should include both a machine-readable report and a readable researcher summary.
- Any verification mismatch is a hard failure; the surgery flow should not write or bless a suspect output artifact.

### Output checkpoint lifecycle
- Phase 6 creates one canonical post-surgery checkpoint directory per manifest/run.
- The post-surgery checkpoint is a real downstream input artifact for later phases such as warmup and evaluation, not just a Phase 6 proof artifact.
- Surgery outputs should live under `checkpoints/`, alongside the base checkpoint rather than only under `out/`.
- Outputs for different manifests should coexist as separate named artifacts rather than overwriting a single current surgery checkpoint.

### Claude's Discretion
- Exact manifest JSON/YAML field names and file naming conventions.
- Exact naming scheme for per-manifest checkpoint directories under `checkpoints/`, as long as multiple outputs can coexist and remain easy to trace back to a manifest.
- Exact formatting of the human-readable verification summary.
- Exact implementation split across `safemoe/interventions/` modules, so long as the thin intervention-layer architecture stays intact.

</decisions>

<specifics>
## Specific Ideas

- The existing base checkpoint at `checkpoints/Qwen3-30B-A3B-Base` is the surgery source of truth and already includes `lit_model.pth`, `model_config.yaml`, tokenizer files, and HF config files.
- The current LitGPT parameter layout already exposes the surgery surfaces this phase needs:
  - Experts: `transformer.h.{layer}.mlp.experts.{idx}.{fc_1|fc_2|proj}.weight`
  - Router/gate: `transformer.h.{layer}.mlp.gate.weight`
  - Attention QKV: `transformer.h.{layer}.attn.qkv.weight`
- `SafeMoEConfig` should remain the canonical config object after surgery, since later phases already depend on `harmful_expert_indices`, `harmful_attn_heads`, and the LitGPT-compatible `model_config.yaml` contract.
- The manifest should be the provenance anchor for later registry, evaluation, and ablation work, but it only needs to record explicit target layouts and the coherent source bundle, not redundant router mappings when those are implied.

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase definition
- `.planning/ROADMAP.md` — Defines the fixed Phase 6 scope, `INIT-01/02/03` success criteria, and plan slots `06-01` and `06-02`.
- `.planning/REQUIREMENTS.md` — Defines the intervention-manifest, cloning, and post-surgery verification requirements this phase must satisfy.
- `.planning/PROJECT.md` — Captures the v1.1 direct-Qwen milestone intent and the requirement to initialize `theta_harmful` by cloning experts, heads, and router columns.
- `.planning/STATE.md` — Confirms Phase 6 is the active milestone phase after the Phase 5 runtime gate.

### Architecture guidance
- `.planning/research/ARCHITECTURE.md` — Locks the thin `safemoe/interventions/` layer approach, stable CLI seams, and LitGPT-compatible checkpoint layout.
- `.planning/research/SUMMARY.md` — Captures the milestone-wide recommendation to keep one SafeMoE training stack and add manifest-based provenance for cloned experts, heads, and router columns.
- `.planning/phases/05-environment-runtime-gate/05-CONTEXT.md` — Carries forward the decision to build on the existing direct-Qwen checkpoint and runtime path rather than introducing a second stack.

### Current code surfaces
- `safemoe/pretrain.py` — Current direct-Qwen checkpoint loading path, `SafeMoEConfig` normalization, and checkpoint save contract that later phases will consume.
- `safemoe/config.py` — Canonical SafeMoE config dataclass that must remain loadable from `model_config.yaml`.
- `safemoe/masking.py` — Existing harmful/std/shared registry semantics and slice-metadata pattern that later phases will extend from the Phase 6 manifest.
- `safemoe/ablate.py` — Existing manifest-producing checkpoint mutation pattern and checkpoint-directory expectations relevant to Phase 6 artifact design.
- `checkpoints/Qwen3-30B-A3B-Base/model_config.yaml` — Pinned base-model architecture and MoE dimensions for surgery planning.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `safemoe/pretrain.py` already loads the base checkpoint through `fabric.load_raw(initial_checkpoint_dir / "lit_model.pth", model)` and saves LitGPT-compatible checkpoint directories via `save_checkpoint(...)`.
- `safemoe/config.py` already extends `litgpt.Config` with `harmful_expert_indices`, `harmful_attn_heads`, and `num_harmful_experts`, which later phases already expect.
- `safemoe/masking.py` already uses full expert-parameter grouping plus slice metadata for fused QKV weights, which is the right precedent for Phase 6 manifest-aware surgery metadata.
- `safemoe/ablate.py` already writes a sibling artifact plus JSON manifest, which is a useful local pattern for Phase 6 provenance and report outputs.

### Established Patterns
- The codebase expects checkpoint directories with `lit_model.pth` and `model_config.yaml`, optionally alongside tokenizer/config files copied into the output directory.
- The direct-Qwen stack already normalizes to `SafeMoEConfig` before entering the main training path, so post-surgery checkpoints should preserve that contract instead of inventing a second config format.
- LitGPT MoE parameter naming is stable and surgery-friendly:
  - `transformer.h.{layer}.mlp.experts.{idx}.*` for expert weights
  - `transformer.h.{layer}.mlp.gate.weight` for router columns
  - `transformer.h.{layer}.attn.qkv.weight` for fused attention-head slices
- Existing milestone research already recommends keeping Qwen-specific cloning logic in `safemoe/interventions/` rather than embedding surgery behavior inside training or model-forward code.

### Integration Points
- Phase 6 should produce a post-surgery checkpoint artifact that later phases can pass back through `safemoe/pretrain.py` as the new initialization source.
- The manifest produced here becomes the provenance source for Phase 7 registry coverage, Phase 8 warmup routing supervision, and Phase 10 manifest-aware evaluation.
- The target harmful expert indices and global harmful head set written into the manifest must stay consistent with `SafeMoEConfig` so downstream registry and masking code can consume them without reinterpretation.
- Verification should attach directly to the saved checkpoint artifact and fail before downstream phases ever see an invalid surgery output.

</code_context>

<deferred>
## Deferred Ideas

- Stronger tensor-similarity or cosine-parity reporting beyond reload and mapping/shape correctness.
- Making deterministic replay a formal acceptance criterion for this phase.
- Per-layer harmful slot selection or per-layer harmful head layouts.
- Zero-noise baseline surgery artifacts.
- Ephemeral or overwrite-in-place surgery outputs instead of coexistable checkpoint artifacts.

</deferred>

---

*Phase: 06-checkpoint-surgery*
*Context gathered: 2026-03-19*
