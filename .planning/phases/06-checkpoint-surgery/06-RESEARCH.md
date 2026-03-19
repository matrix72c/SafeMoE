# Phase 6: Checkpoint Surgery - Research

**Researched:** 2026-03-19
**Domain:** Deterministic manifest-driven Qwen3 MoE checkpoint surgery on the existing SafeMoE/LitGPT stack
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
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

### Deferred Ideas (OUT OF SCOPE)
- Stronger tensor-similarity or cosine-parity reporting beyond reload and mapping/shape correctness.
- Making deterministic replay a formal acceptance criterion for this phase.
- Per-layer harmful slot selection or per-layer harmful head layouts.
- Zero-noise baseline surgery artifacts.
- Ephemeral or overwrite-in-place surgery outputs instead of coexistable checkpoint artifacts.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| INIT-01 | Researcher can define a deterministic intervention manifest that records the selected source experts, target harmful experts, source attention heads, target harmful heads, cloned router columns, random seed, and noise scale for a run. | Use a pure manifest planner under `safemoe/interventions/` that emits explicit target experts/heads, a coherent source bundle, seed, noise scale, base checkpoint ID, and derived router-column mapping. Persist the manifest next to the output checkpoint before any tensor mutation. |
| INIT-02 | Researcher can initialize `theta_harmful` in `Qwen3-30B-A3B-Base` by cloning selected experts and attention heads from `theta_std`, copying the corresponding router columns, and adding controlled noise while preserving a loadable checkpoint. | Apply surgery on LitGPT tensor names only: clone `mlp.experts.*`, copy `mlp.gate.weight` columns, rewrite the appropriate `attn.qkv.weight` row slices, update `SafeMoEConfig.harmful_expert_indices` and `harmful_attn_heads`, then save via the existing LitGPT-compatible checkpoint contract. |
| INIT-03 | Researcher can verify that post-surgery tensors match the manifest semantics through parity checks on tensor shapes, source-to-target mappings, and checkpoint reload behavior. | Build a verification pass that reloads the saved checkpoint, recomputes expected source/target slices from the manifest, checks shape equality and mapping parity, and writes both `verification_report.json` and `verification_summary.md`. Treat any mismatch as a hard failure. |
</phase_requirements>

## Summary

Phase 6 should stay entirely inside the existing SafeMoE/LitGPT checkpoint contract. The repo already has the two critical ingredients: a real converted base checkpoint at `checkpoints/Qwen3-30B-A3B-Base`, and local Qwen3 conversion code that fixes the exact parameter names and QKV layout that later training code expects. The correct plan is therefore not “invent checkpoint surgery.” It is “plan a thin intervention layer that mutates the already-converted LitGPT state dict deterministically, then saves a normal LitGPT-compatible checkpoint directory plus provenance artifacts.”

The most important planning choice is to make the manifest the source of truth and the tensor edits a pure application of that manifest. The manifest must explicitly record the chosen harmful expert slots and harmful head slots, the coherent source bundle, the random seed, the shared nonzero noise scale, and the base checkpoint identity. Router-column mappings can stay derived rather than duplicated, because the locked decisions already say they are implied by the source/target layout. If the manifest is complete enough, the surgery code becomes auditable, replayable, and easy to test on tiny CPU checkpoints before touching the real 61 GB artifact.

**Primary recommendation:** Plan Phase 6 as two tasks: first, define and test a deterministic manifest planner plus file layout; second, implement a state-dict surgery + verification pipeline that mutates LitGPT expert/router/QKV tensors, saves a standard checkpoint directory under `checkpoints/`, and fails closed on any reload or parity mismatch.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| local `litgpt` fork | `0.5.12` (repo `pyproject.toml`) | Canonical Qwen3 MoE config/model/checkpoint layout | The converted checkpoint, trainer, and conversion tests already depend on it. |
| `torch` | `2.10.0+cu128` (installed) | Raw tensor cloning, deterministic noise, checkpoint IO | Phase 6 is fundamentally tensor surgery plus save/load verification. |
| `lightning` | `2.6.1` (installed) | Existing checkpoint save path via Fabric in later flows | Keep output checkpoints compatible with the current pretrain path instead of inventing a second serializer. |
| `PyYAML` | local runtime dependency | Read/write `model_config.yaml` and manifest/report metadata | The repo already persists config through YAML and later tools expect it. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `safetensors` | `0.7.0` (installed) | Read surrounding HF shard metadata if needed for provenance only | Optional; surgery itself should target `lit_model.pth`, not the 16 HF shards. |
| `huggingface_hub` | `1.5.0` (installed) | Trace back to upstream model metadata if planner wants model-card provenance | Optional; not required for the actual checkpoint mutation path. |
| `pytest` | `9.0.2` (installed) | Unit/integration verification of manifest planning and checkpoint surgery | Required for Phase 6 Nyquist coverage. |
| `transformers` | repo optional extra `>=4.51.3,<4.57`; not installed here | HF parity/debug path for Qwen3MoE if needed | Useful only as a secondary validation tool, not a Phase 6 runtime dependency. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Mutating the LitGPT checkpoint directly | Rebuild from HF `model-*.safetensors` shards | Worse: duplicates conversion logic the repo already solved and increases failure surface. |
| Adjacent manifest + report files | Hide all provenance inside `lit_model.pth` only | Worse: planner, verifier, ablation, and later phases need inspectable metadata without loading tensors. |
| Pure manifest planner + pure applier | One monolithic “do surgery” function | Harder to test determinism and easier to couple layout selection to mutation bugs. |
| Local conversion helpers for QKV slice math | Hand-derived QKV indexing from HF docs alone | Risky: the repo already codifies the exact LitGPT ordering and tests it. |

**Installation:**
```bash
python -m pip install -e ".[test]"
```

**Version verification:** Local runtime versions were verified in this workspace on 2026-03-19.
```bash
python - <<'PY'
import importlib
for name in ["torch", "lightning", "tokenizers", "torchmetrics", "safetensors", "huggingface_hub", "pytest"]:
    mod = importlib.import_module(name)
    print(name, getattr(mod, "__version__", "unknown"))
PY
```
Observed versions:
- `torch` `2.10.0+cu128`
- `lightning` `2.6.1`
- `tokenizers` `0.22.2`
- `torchmetrics` `1.8.2`
- `safetensors` `0.7.0`
- `huggingface_hub` `1.5.0`
- `pytest` `9.0.2`

## Architecture Patterns

### Recommended Project Structure
```text
safemoe/
├── interventions/
│   ├── __init__.py
│   ├── manifest.py        # schema/dataclasses + load/save helpers
│   ├── planner.py         # deterministic source/target selection
│   ├── surgery.py         # state-dict mutation for experts, router, qkv slices
│   └── verify.py          # reload + parity report generation
├── pretrain.py            # unchanged training entry point that consumes saved checkpoints
├── config.py              # SafeMoEConfig remains canonical
└── ablate.py              # later phases consume manifest-aware outputs

checkpoints/
├── Qwen3-30B-A3B-Base/
└── Qwen3-30B-A3B-Base-surgery-<manifest_id>/
    ├── lit_model.pth
    ├── model_config.yaml
    ├── intervention_manifest.json
    ├── verification_report.json
    ├── verification_summary.md
    └── tokenizer/config sidecar files copied from base checkpoint
```

### Pattern 1: Manifest-first planning
**What:** Compute all source and target selections before touching any tensor.
**When to use:** Always for `INIT-01`.
**Example:**
```python
# Source: recommended Phase 6 pattern based on local SafeMoE config + context decisions
manifest = InterventionManifest(
    manifest_version=1,
    base_checkpoint="checkpoints/Qwen3-30B-A3B-Base",
    source_bundle=SourceBundle(
        source_experts=[12, 47],
        source_heads=[3, 11],
    ),
    target_layout=TargetLayout(
        harmful_expert_indices=[0, 1],
        harmful_attn_heads=[0, 1],
    ),
    seed=1234,
    noise_scale=1e-3,
)
manifest.validate()
```

### Pattern 2: State-dict surgery on LitGPT tensor names
**What:** Load the base LitGPT checkpoint, mutate only the relevant tensors, and save back out as a normal LitGPT checkpoint directory.
**When to use:** For `INIT-02`.
**Example:**
```python
# Source: local tensor names verified in checkpoints/Qwen3-30B-A3B-Base/model_config.yaml
expert_prefix = f"transformer.h.{layer}.mlp.experts.{target_expert}"
gate_name = f"transformer.h.{layer}.mlp.gate.weight"
qkv_name = f"transformer.h.{layer}.attn.qkv.weight"

state[f"{expert_prefix}.fc_1.weight"] = clone_with_noise(
    state[f"transformer.h.{layer}.mlp.experts.{source_expert}.fc_1.weight"],
    generator,
    noise_scale,
)
state[gate_name][:, target_expert] = clone_with_noise(
    state[gate_name][:, source_expert],
    generator,
    noise_scale,
)
copy_qkv_head_rows_(state[qkv_name], source_head, target_head, config, generator, noise_scale)
```

### Pattern 3: Save config truth into `model_config.yaml`
**What:** The saved checkpoint config must become the downstream truth for harmful slots.
**When to use:** Before writing the output checkpoint.
**Example:**
```python
# Source: local save path in safemoe/pretrain.py and SafeMoEConfig fields in safemoe/config.py
config = SafeMoEConfig(**base_config_dict)
config.harmful_expert_indices = manifest.target_layout.harmful_expert_indices
config.harmful_attn_heads = manifest.target_layout.harmful_attn_heads
config.num_harmful_experts = len(config.harmful_expert_indices)
save_config(config, output_dir)
```

### Pattern 4: Verify by reload, then bless
**What:** The verifier must reopen the saved checkpoint and compare manifest-declared mappings against the actual tensors in the saved artifact.
**When to use:** Always for `INIT-03`.
**Example:**
```python
# Source: local load pattern in safemoe/evaluate.py and safemoe/ablate.py
saved = torch.load(output_dir / "lit_model.pth", map_location="cpu", weights_only=False)
model = GPT(saved_config)
model.load_state_dict(saved["model"])
report = verify_manifest_parity(model.state_dict(), base_state, manifest, saved_config)
if not report["ok"]:
    raise ValueError("Checkpoint surgery verification failed")
```

### Anti-Patterns to Avoid
- **Mutating HF shard files directly:** the real downstream consumer is `lit_model.pth` plus `model_config.yaml`.
- **Selecting harmful slots implicitly from a seed only:** locked decisions require explicit target expert/head lists in the manifest.
- **Writing outputs before verification:** verification mismatch is a hard failure, so use a temp/staging directory and only finalize on success.
- **Treating router columns as untracked incidental edits:** router inheritance is a required part of the source bundle semantics.
- **Hand-coding QKV order from memory:** reuse the local `qkv_reassemble` conventions and the registry slice logic.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Qwen3 MoE tensor naming | A new naming convention map | Existing LitGPT conversion names in `litgpt/scripts/convert_hf_checkpoint.py` and `convert_lit_checkpoint.py` | Those helpers already encode Qwen3 expert, gate, and QKV mappings. |
| QKV row ordering | Custom head-index math invented from scratch | Local `qkv_reassemble()` semantics plus `HarmfulParamRegistry.build_qkv_slices()` pattern | This is the highest-risk silent bug area; local code already nails the layout. |
| Checkpoint output layout | Custom artifact folder format under `out/` only | Existing LitGPT checkpoint directory contract with copied config/tokenizer files | Later phases already expect loadable checkpoint directories. |
| Config serialization | Separate surgery-only config file | `SafeMoEConfig` written to `model_config.yaml` | Downstream training, eval, and ablation already load this path. |
| Verification report format | Console-only prints | `verification_report.json` plus `verification_summary.md` | Locked decisions require both machine-readable and human-readable output. |

**Key insight:** The difficult parts of Phase 6 are already solved locally at the representation layer. The planner should spend effort on determinism, provenance, and fail-closed verification, not on inventing a new Qwen checkpoint abstraction.

## Common Pitfalls

### Pitfall 1: Confusing LitGPT QKV order with HF Q/K/V module boundaries
**What goes wrong:** Head cloning copies the wrong row ranges inside `attn.qkv.weight`.
**Why it happens:** HF exposes separate `q_proj`, `k_proj`, and `v_proj`, but the converted checkpoint stores one packed tensor in LitGPT order.
**How to avoid:** Base row slicing on the local conversion helpers and `HarmfulParamRegistry` slice logic, not on ad hoc formulas.
**Warning signs:** Reload succeeds, but verification reports shape matches with wrong head semantics or unexpected overlap in copied rows.

### Pitfall 2: Updating weights without updating `SafeMoEConfig`
**What goes wrong:** The checkpoint reloads, but downstream registry/eval code classifies the wrong harmful experts or heads.
**Why it happens:** `model_config.yaml` still reflects the base checkpoint rather than the post-surgery harmful layout.
**How to avoid:** Treat `model_config.yaml` as part of the surgery output, not as a copied sidecar.
**Warning signs:** `HarmfulParamRegistry` on the saved checkpoint disagrees with the manifest target layout.

### Pitfall 3: Applying nondeterministic noise
**What goes wrong:** The same manifest produces different weights across reruns, making debugging and provenance brittle.
**Why it happens:** Global RNG state leaks in, or CPU/GPU generator usage is inconsistent.
**How to avoid:** Use a dedicated `torch.Generator(device="cpu")` seeded from the manifest and generate all noise on CPU before final tensor placement.
**Warning signs:** Running the same manifest twice produces different verification hashes or tensor deltas.

### Pitfall 4: Verifying only shapes, not source-to-target parity
**What goes wrong:** Tensors have compatible shapes but were copied from the wrong source expert/head.
**Why it happens:** Shape checks are easy; semantic checks require explicit source/target accounting.
**How to avoid:** Verification must compare each manifest mapping against base and output tensors after subtracting the recorded noise recipe or by recomputing expected mutated tensors from the manifest seed.
**Warning signs:** Verification passes even when source selections are intentionally perturbed in a test.

### Pitfall 5: Writing partial artifacts on failure
**What goes wrong:** Later phases accidentally consume a half-written surgery checkpoint.
**Why it happens:** Output files are written directly into the final directory before verification completes.
**How to avoid:** Write into a temp directory, verify, then atomically rename or only mark the artifact complete after report success.
**Warning signs:** A checkpoint directory exists without both verification files or with mismatched manifest/config content.

### Pitfall 6: Forgetting that router columns remain slices of a shared tensor
**What goes wrong:** Planner assumes router columns become standalone harmful parameters.
**Why it happens:** Expert cloning uses whole parameters, but router cloning is column-slice mutation within `mlp.gate.weight`.
**How to avoid:** Keep router mapping in the manifest/report and leave full registry semantics for Phase 7, which will extend slice metadata coverage.
**Warning signs:** Design starts introducing fake per-column `nn.Parameter`s or a new gate module just for Phase 6.

## Code Examples

Verified patterns from local code and official docs:

### Qwen3 MoE expert/router mapping
```python
# Source: local litgpt/scripts/convert_hf_checkpoint.py
weight_map = {
    "model.layers.{}.mlp.experts.{}.gate_proj.weight": "transformer.h.{}.mlp.experts.{}.fc_1.weight",
    "model.layers.{}.mlp.experts.{}.up_proj.weight": "transformer.h.{}.mlp.experts.{}.fc_2.weight",
    "model.layers.{}.mlp.experts.{}.down_proj.weight": "transformer.h.{}.mlp.experts.{}.proj.weight",
    "model.layers.{}.mlp.gate.weight": "transformer.h.{}.mlp.gate.weight",
}
```

### LitGPT packed-QKV layout
```python
# Source: local litgpt/scripts/convert_lit_checkpoint.py
q, k, v = param.split(
    (
        config.n_head * config.head_size,
        config.n_query_groups * config.head_size,
        config.n_query_groups * config.head_size,
    )
)
```

### Existing checkpoint reload contract
```python
# Source: local safemoe/evaluate.py
raw = yaml.safe_load((ckpt_dir / "model_config.yaml").read_text())
config = SafeMoEConfig(**{k: v for k, v in raw.items() if not isinstance(v, dict)})

model = GPT(config)
state = torch.load(ckpt_dir / "lit_model.pth", map_location="cpu", weights_only=False)
model.load_state_dict(state["model"])
```

### Existing checkpoint save contract
```python
# Source: local safemoe/pretrain.py
fabric.save(checkpoint_file, save_state)
if fabric.global_rank == 0:
    if tokenizer_dir is not None:
        copy_config_files(tokenizer_dir, checkpoint_file.parent)
    save_config(model.config, checkpoint_file.parent)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Clone from HF modules directly | Operate on the already-converted LitGPT checkpoint | Current v1.1 milestone | Keeps training/eval/ablation on one representation. |
| Treat harmful layout as an implementation detail | Record explicit harmful target layout in the manifest | Locked in Phase 6 context on 2026-03-19 | Makes later registry/eval work traceable. |
| Save only tensors | Save tensors plus manifest and verification artifacts | Current milestone requirement | Enables reproducible downstream planning and auditing. |
| Use generic MoE assumptions | Use Qwen3-specific LitGPT mapping already validated in local conversion tests | Current repo state on 2026-03-19 | Reduces silent tensor-layout bugs. |

**Deprecated/outdated:**
- Building Phase 6 around a second training stack or HF-only loader path: explicitly out of scope for this milestone.
- Assuming `transformers` is a hard dependency for surgery: current workspace proves the core surgery path can stay on LitGPT + PyTorch only.

## Open Questions

1. **How much of deterministic replay should be encoded in the verification report?**
   - What we know: replay is useful but not a formal acceptance criterion.
   - What's unclear: whether the report should include full tensor hashes or only manifest/config identity plus pass/fail checks.
   - Recommendation: Record manifest hash, base checkpoint identifier, and per-mapping pass/fail details now; leave richer hashing as optional follow-up.

2. **Should the surgery pipeline operate on a raw state dict or an instantiated `GPT` model?**
   - What we know: both are possible in local code, but surgery itself only needs tensors and config metadata.
   - What's unclear: whether planner prefers a pure state-dict path for simpler CPU tests or a model-instantiated path for easier reuse with existing loaders.
   - Recommendation: Plan the applier around a raw state dict plus config. Instantiate `GPT` only in verification reload, where loadability actually matters.

3. **How should checkpoint directory names be generated?**
   - What we know: outputs must coexist under `checkpoints/` and remain easy to trace back to a manifest.
   - What's unclear: whether to key by timestamp, user label, manifest hash, or both.
   - Recommendation: Use a stable manifest ID or short hash in the directory name and store the full manifest inside the directory; avoid timestamp-only naming because it is not reproducible.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | `pytest 9.0.2` |
| Config file | `pyproject.toml` |
| Quick run command | `pytest tests/safemoe/test_checkpoint_surgery.py -x` |
| Full suite command | `pytest tests/safemoe -x` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| INIT-01 | Manifest planner emits explicit target experts/heads, coherent source bundle, nonzero noise scale, and stable output for the same seed | unit | `pytest tests/safemoe/test_checkpoint_surgery.py::test_manifest_planner_is_deterministic -x` | ❌ Wave 0 |
| INIT-02 | Surgery clones expert tensors, router columns, and QKV head rows into the target harmful layout and saves a normal checkpoint directory | integration | `pytest tests/safemoe/test_checkpoint_surgery.py::test_surgery_writes_loadable_checkpoint_directory -x` | ❌ Wave 0 |
| INIT-03 | Verifier reloads the saved checkpoint and catches any mapping/shape mismatch against the manifest | integration | `pytest tests/safemoe/test_checkpoint_surgery.py::test_verifier_fails_on_manifest_mismatch -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/safemoe/test_checkpoint_surgery.py -x`
- **Per wave merge:** `pytest tests/safemoe/test_checkpoint_surgery.py tests/safemoe/test_registry.py tests/safemoe/test_ablate.py -x`
- **Phase gate:** `pytest tests/safemoe -x` plus one manual reload smoke on a real surgery output under `checkpoints/`

### Wave 0 Gaps
- [ ] `tests/safemoe/test_checkpoint_surgery.py` — covers manifest planning, state-dict mutation, reload parity, and failure cases
- [ ] `safemoe/interventions/manifest.py` — schema + validation helpers
- [ ] `safemoe/interventions/planner.py` — deterministic source/target planner
- [ ] `safemoe/interventions/surgery.py` — expert/router/QKV mutation logic
- [ ] `safemoe/interventions/verify.py` — report generation and hard-fail verification
- [ ] Temp-output/finalize helper — ensures invalid surgery runs never leave blessed artifacts in `checkpoints/`

## Sources

### Primary (HIGH confidence)
- Local repo: `safemoe/pretrain.py` — verified the checkpoint save contract and downstream consumer expectations
- Local repo: `safemoe/config.py` — verified `SafeMoEConfig` is the canonical harmful-layout carrier
- Local repo: `safemoe/masking.py` — verified current harmful expert/QKV slice semantics and reusable slice logic
- Local repo: `safemoe/ablate.py` — verified current standalone checkpoint mutation + adjacent manifest pattern
- Local repo: `safemoe/evaluate.py` — verified the checkpoint reload contract used by later phases
- Local repo: `litgpt/scripts/convert_hf_checkpoint.py` — verified Qwen3 expert/router/QKV mapping into LitGPT names
- Local repo: `litgpt/scripts/convert_lit_checkpoint.py` — verified packed-QKV split order for LitGPT checkpoints
- Local repo: `tests/test_model.py` and `tests/convert/test_hf_checkpoint.py` — verified local parity coverage for Qwen3/QKV conversion semantics
- Local repo: `checkpoints/Qwen3-30B-A3B-Base/` and `model_config.yaml` — verified the actual local checkpoint surface and pinned architecture
- Official Hugging Face Qwen3MoE docs: https://huggingface.co/docs/transformers/en/model_doc/qwen3_moe
- Official Qwen config reference: https://huggingface.co/Qwen/Qwen3-30B-A3B-Base/blob/main/config.json
- Official PyTorch docs: https://docs.pytorch.org/docs/stable/generated/torch.load.html
- Official PyTorch docs: https://docs.pytorch.org/docs/stable/generated/torch.save

### Secondary (MEDIUM confidence)
- Official Lightning Fabric FSDP docs: https://lightning.ai/docs/fabric/2.4.0/api/generated/lightning.fabric.strategies.FSDPStrategy.html
- Local repo: `pyproject.toml` — verified dependency floors and test framework configuration

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - grounded in the local checkpoint, local runtime, and installed package versions
- Architecture: HIGH - derived directly from existing SafeMoE/LitGPT boundaries and local conversion helpers
- Pitfalls: MEDIUM - strongly evidenced by local representation details, but a few failure modes still await implementation-time confirmation on the full checkpoint

**Research date:** 2026-03-19
**Valid until:** 2026-04-18
