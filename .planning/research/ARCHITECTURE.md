# Architecture Research

**Domain:** SafeMoE v1.1 integration architecture for direct harmful-transfer on `Qwen3-30B-A3B-Base`
**Researched:** 2026-03-19
**Confidence:** HIGH

## Standard Architecture

### System Overview

The existing `safemoe/` package already provides the right top-level shape: config-driven model construction, a registry that classifies harmful versus standard parameters, a forked training loop, checkpoint ablation, and evaluation helpers. For this milestone, do not create a second training stack or a Qwen-only model fork. Add a thin Qwen intervention layer between checkpoint loading and the existing SGTM loop, then extend the loop with stage-aware losses and routing metrics.

The architecture should split into five layers:

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Experiment / CLI Layer                       │
├──────────────────────────────────────────────────────────────────────┤
│  safemoe/__main__.py   safemoe/pretrain.py setup()   eval entrypts  │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
┌──────────────────────────────────────────────────────────────────────┐
│                      Qwen Intervention Layer                         │
├──────────────────────────────────────────────────────────────────────┤
│  checkpoint loader  →  clone planner  →  model surgery/applier      │
│                    →  intervention manifest writer                   │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
┌──────────────────────────────────────────────────────────────────────┐
│                    Stable SafeMoE Model Layer                        │
├──────────────────────────────────────────────────────────────────────┤
│  SafeMoEConfig  HarmfulParamRegistry  SafeMoELayer  attn masking    │
│  router capture / routing stats hooks                               │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
┌──────────────────────────────────────────────────────────────────────┐
│                    Stage-Aware Training Layer                        │
├──────────────────────────────────────────────────────────────────────┤
│  warmup stage: D_std + D_harmful + routing loss                     │
│  transfer stage: D_std + D_harmful + D_unlabeled + SGTM losses      │
│  shared optimizer / scheduler / checkpointing                       │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
┌──────────────────────────────────────────────────────────────────────┐
│                    Evaluation / Ablation Layer                       │
├──────────────────────────────────────────────────────────────────────┤
│  perplexity eval   routing attribution   warmup separation metrics   │
│  ablation utility  transfer concentration metrics                    │
└──────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| `SafeMoEConfig` | Remains the canonical experiment config and harmful index carrier | Extend existing dataclass with intervention and warmup settings, not a new config type |
| `HarmfulParamRegistry` | Remains the single source of truth for harmful/std/shared parameter grouping | Adapt to classify cloned router columns and optional cloned attention head slices |
| `Qwen intervention planner` | Chooses source experts/heads, target harmful experts/heads, router columns, noise scale | New module under `safemoe/interventions/` returning a pure manifest |
| `Qwen intervention applier` | Applies cloning, noise, and router-column duplication to a loaded `GPT(config)` model | New module under `safemoe/interventions/` working on LitGPT parameter names |
| `safemoe.pretrain` stage controller | Orchestrates warmup then transfer without changing the outer CLI contract | Modify existing `setup()`, `main()`, and `fit()` rather than adding a second trainer |
| `routing supervision loss` | Computes warmup loss from router assignments/logits and split label | New small module under `safemoe/losses/` |
| `routing observability hooks` | Captures per-layer harmful routing mass, cloned-column usage, and separation metrics | New helper module used by train and eval |
| `ablate.py` / `evaluate.py` | Keep post-training evaluation entry points stable | Extend to read intervention manifest and report new metrics |

## Recommended Project Structure

```
safemoe/
├── config.py                    # Extend existing SafeMoEConfig
├── masking.py                   # Extend registry for cloned experts/heads/router cols
├── pretrain.py                  # Add stage orchestration and warmup/transfer loss wiring
├── ablate.py                    # Keep entry point, add manifest-aware reporting
├── evaluate.py                  # Keep entry point, add warmup/transfer routing metrics
├── interventions/
│   ├── __init__.py
│   ├── plan.py                  # CloneSpec / InterventionManifest
│   ├── apply.py                 # Expert/head/router cloning + noise
│   └── io.py                    # Save/load intervention manifest
├── losses/
│   └── routing_supervision.py   # Warmup routing-margin or target-mass loss
└── observability/
    └── routing.py               # Shared train/eval routing metric collectors
```

### Structure Rationale

- **Keep `safemoe/pretrain.py` as the only trainer:** the current code already owns optimizer setup, split sampling, checkpointing, and mid-training ablation. Replacing it would duplicate the v1.0 milestone.
- **Put Qwen-specific logic in `safemoe/interventions/`:** cloning and checkpoint surgery are a distinct concern from SGTM masking. They should run once before training, not leak into `SafeMoELayer.forward()`.
- **Keep `config.py`, `masking.py`, `evaluate.py`, and `ablate.py` as stable public seams:** downstream roadmap work can build incrementally without rethreading the CLI or checkpoint format.

## Architectural Patterns

### Pattern 1: One-Time Checkpoint Surgery Before Any Training

**What:** Load the LitGPT `Qwen3-30B-A3B-Base` model, apply expert/head/router cloning exactly once, then hand the mutated model to the existing SafeMoE training stack.

**When to use:** Always for this milestone. Cloning is initialization, not a runtime behavior.

**Trade-offs:** Clear separation and easy reproducibility; slightly more metadata to persist.

**Example:**
```python
manifest = plan_qwen_intervention(config, seed=seed)
model = GPT(config)
fabric.load_raw(initial_checkpoint_dir / "lit_model.pth", model)
apply_qwen_intervention(model, config, manifest)
save_intervention_manifest(out_dir, manifest)
```

### Pattern 2: Manifest-Driven Harmful Parameter Semantics

**What:** The intervention manifest should define the cloned source-to-target mapping, and the rest of the stack should consume only the manifest plus `SafeMoEConfig`.

**When to use:** For registry classification, evaluation reporting, and ablation manifests.

**Trade-offs:** Adds a metadata artifact, but avoids hidden assumptions about which harmful expert came from which base expert.

**Example:**
```python
@dataclass
class CloneSpec:
    layer: int
    source_expert: int
    target_expert: int
    source_heads: list[int]
    target_heads: list[int]
    cloned_router_columns: list[int]
    noise_std: float
```

### Pattern 3: Stage-Aware Loss Composition in the Existing Loop

**What:** Keep the current split-based SGTM loop, but add a stage enum and a loss composer:

- `warmup`: `lm_loss + lambda_router * routing_loss` on `D_std` and `D_harmful`
- `transfer`: existing SGTM next-token loss on all three splits, with optional low-weight routing regularizer

**When to use:** Warmup and transfer are different optimization regimes, but they should share data loading, masking, scheduler, logging, and checkpointing.

**Trade-offs:** Slightly more branching inside `fit()`, but far less code duplication than separate scripts.

## Data Flow

### Request Flow

```
CLI / config
    ↓
pretrain.setup()
    ↓
build SafeMoEConfig + load tokenizer/data
    ↓
instantiate GPT(config)
    ↓
load Qwen checkpoint
    ↓
plan + apply intervention
    ↓
build HarmfulParamRegistry / maskers / routing collectors
    ↓
run warmup stage
    ↓
run transfer stage
    ↓
save checkpoint + intervention manifest + routing metrics
```

### State Management

The current state dict in `safemoe/pretrain.py` should remain the primary checkpoint container. Add stage metadata rather than inventing a second checkpoint format:

```
state = {
    "model": model,
    "optimizer": optimizer,
    "iter_num": ...,
    "step_count": ...,
    "split_label": ...,
    "stage": "warmup" | "transfer",
    "intervention_manifest": {...},
}
```

### Key Data Flows

1. **Initialization flow:** `initial_checkpoint_dir` -> LitGPT `GPT(config)` -> Qwen intervention planner/applier -> registry rebuild -> trainable model.
2. **Warmup flow:** `D_std` and `D_harmful` batches -> forward with router capture -> LM loss + routing supervision -> existing activation/gradient masking -> optimizer step.
3. **Transfer flow:** `D_std`, `D_harmful`, `D_unlabeled` -> existing SGTM branching -> optional routing metric logging -> checkpoint + ablation eval.
4. **Evaluation flow:** trained checkpoint + manifest -> routing attribution + pre/post ablation perplexity + harmful-routing concentration summaries.

## Integration Points

### New Modules

| Module | Why New | Notes |
|--------|---------|-------|
| `safemoe/interventions/plan.py` | Source/target clone selection is a new concern not covered by v1.0 | Keep pure and deterministic from config + seed |
| `safemoe/interventions/apply.py` | Expert/head/router-column cloning should not live in `pretrain.py` | Operate on actual parameter names exposed by LitGPT |
| `safemoe/interventions/io.py` | Manifest persistence needs a single implementation | Reused by train, eval, and ablate |
| `safemoe/losses/routing_supervision.py` | Warmup routing objective is new | Should accept captured routing tensors and split label |
| `safemoe/observability/routing.py` | Current eval only logs final harmful fraction | Needs reusable per-step and per-checkpoint collectors |

### Existing Modules to Modify

| Module | Required Change | Stability Requirement |
|--------|-----------------|-----------------------|
| `safemoe/config.py` | Add intervention and stage hyperparameters: source selection, head clone count, router clone flag, noise std, warmup steps, routing loss weight | Keep `SafeMoEConfig` as a `litgpt.Config` subclass and preserve current harmful index fields |
| `safemoe/masking.py` | Extend `HarmfulParamRegistry` to classify cloned router columns and optional attention head slices explicitly | Preserve `parameters_by_type()` and current split labels |
| `safemoe/pretrain.py` | Add pre-training surgery, stage transitions, routing loss composition, and richer checkpoint state | Keep `setup()`, `main()`, `fit()`, `validate()`, and CLI compatibility stable |
| `safemoe/evaluate.py` | Add manifest loading, cloned-router metrics, warmup separation metrics, and transfer concentration metrics | Keep `evaluate_perplexity()` and `routing_attribution()` callable entry points |
| `safemoe/ablate.py` | Include manifest-aware reporting so ablation knows what harmful experts/heads were initialized from | Keep checkpoint output layout unchanged |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `interventions/* ↔ pretrain.py` | Direct function call with typed manifest | No training logic inside interventions |
| `masking.py ↔ pretrain.py` | Existing registry + masker API | Extend rather than replace |
| `observability/routing.py ↔ pretrain.py/evaluate.py` | Shared collector interface | Avoid separate train-only and eval-only hook implementations |
| `evaluate.py ↔ ablate.py` | Filesystem checkpoint contract | `lit_model.pth`, `model_config.yaml`, and manifest file must remain co-located |

## Stable Interfaces That Must Not Drift

1. `python -m safemoe pretrain`, `python -m safemoe evaluate`, and `python -m safemoe ablate` should remain the top-level entry points.
2. `SafeMoEConfig` must stay loadable from `model_config.yaml` without a custom deserializer.
3. `HarmfulParamRegistry.parameters_by_type("theta_harmful" | "theta_std" | "theta_shared")` must remain the training and ablation contract.
4. `MultiDataLoader` should remain the source of `D_std`, `D_harmful`, and `D_unlabeled`; warmup should filter or weight splits, not invent a second datamodule.
5. Checkpoint layout should stay LitGPT-compatible: `lit_model.pth`, `model_config.yaml`, `hyperparameters.yaml`, plus one additional intervention manifest file.

## Qwen-Specific Integration Decisions

### Expert / Head Cloning

- Clone at the LitGPT parameter-name level, not via Hugging Face model objects. The repo already converts Qwen3 MoE weights into LitGPT layout, and the rest of SafeMoE expects LitGPT names.
- Expert cloning should copy `transformer.h.{layer}.mlp.experts.{src}.*` to designated harmful expert slots, then add controlled noise.
- Attention-head cloning should be implemented as row-slice copying inside `transformer.h.{layer}.attn.qkv.weight` and corresponding projection handling only if the milestone truly needs head cloning for harmful initialization. Do not create synthetic per-head `nn.Parameter`s.

### Router-Column Duplication

- Treat router duplication as gate-weight column cloning inside `transformer.h.{layer}.mlp.gate.weight`.
- The cloned router columns are part of harmful initialization, but the gate tensor itself remains a shared parameter tensor. The registry therefore needs metadata for harmful column slices, similar to current qkv slice metadata.
- Warmup routing loss should supervise router behavior through these columns, not by hard-masking logits to force expert selection.

### Warmup Loss Integration

- Add a routing-loss composer that consumes router outputs captured during forward.
- For `D_harmful`, encourage mass on harmful experts / cloned router columns.
- For `D_std`, penalize harmful routing mass.
- Keep the next-token LM loss active during warmup. Pure routing-only warmup would overfit dispatch and destabilize language behavior.

### Mixed-Data Transfer Training

- Reuse the existing SGTM split sampler and maskers.
- Transfer should start from the post-warmup state in the same run by default; separate scripts/checkpoints only add operational friction.
- `D_unlabeled` remains the unmasked path, but evaluation must explicitly test whether its routing moves toward harmful experts after warmup.

### Evaluation Hooks

- Extend routing attribution to report more than final harmful fraction:
  - harmful routing fraction on `D_std`, `D_harmful`, and optionally `D_unlabeled`
  - harmful-router-column mass versus standard-column mass
  - per-layer concentration, not just global aggregate
- Mid-training ablation in `safemoe/pretrain.py` should remain, but the logged metrics should include the current stage and routing separation summary.

## Anti-Patterns

### Anti-Pattern 1: Qwen-Specific Training Fork

**What people do:** Add a new `qwen_pretrain.py` or a parallel trainer because initialization is Qwen-specific.

**Why it's wrong:** The milestone adds one initialization path and a new warmup loss, not a new training architecture. A second trainer would duplicate checkpointing, resume, logging, masking, and evaluation glue.

**Do this instead:** Keep one trainer and insert Qwen-specific logic only at model initialization plus loss composition.

### Anti-Pattern 2: Force Routing by Logit Masking

**What people do:** Set standard-expert logits to `-inf` for `D_harmful` or harmful logits to `-inf` for `D_std`.

**Why it's wrong:** It destroys the experiment. The milestone is trying to learn whether cloned harmful experts attract harmful traffic after supervised warmup, not hard-code the answer.

**Do this instead:** Supervise routing softly with a loss and leave the actual router free to learn.

### Anti-Pattern 3: Encode Clone Semantics Implicitly in `harmful_expert_indices`

**What people do:** Assume harmful indices alone are enough to recover which source expert/head/router column was cloned.

**Why it's wrong:** Evaluation, reproducibility, and later ablation analysis need source-to-target provenance.

**Do this instead:** Persist an explicit intervention manifest beside the checkpoint.

## Build Order

Dependency-respecting roadmap decomposition should be:

1. **Config and manifest schema**
   - Extend `SafeMoEConfig` with intervention and warmup fields.
   - Define `CloneSpec` / intervention manifest dataclasses and file format.
   - This unlocks every later phase.

2. **Checkpoint surgery layer**
   - Implement planner + applier for expert cloning, optional head cloning, router-column duplication, and noise injection.
   - Add focused tests that verify exact parameter copies in LitGPT naming space.
   - This must land before any warmup or transfer work.

3. **Registry and observability extensions**
   - Teach `HarmfulParamRegistry` about cloned router-column slices and any new head-slice metadata.
   - Add reusable routing collectors that expose per-layer harmful mass.
   - Warmup loss depends on this instrumentation.

4. **Warmup-stage integration**
   - Add stage support to `safemoe/pretrain.py`.
   - Integrate routing supervision loss on `D_std` and `D_harmful`.
   - Save stage metadata and intervention manifest into checkpoints.

5. **Transfer-stage integration**
   - Reuse the current SGTM paths, now starting from warmup-initialized state.
   - Add mixed-data metrics for `D_unlabeled` routing drift and concentration.

6. **Evaluation and ablation hooks**
   - Extend `evaluate.py` and `ablate.py` to consume the manifest and log new routing/concentration outputs.
   - Only after this step can the milestone claim end-to-end verification.

### Dependency Graph

```
config/manifest
    ↓
checkpoint surgery
    ↓
registry + routing observability
    ↓
warmup-stage trainer changes
    ↓
transfer-stage trainer changes
    ↓
evaluation + ablation extensions
```

## Sources

- `.planning/PROJECT.md`
- `.planning/codebase/ARCHITECTURE.md`
- `.planning/codebase/INTEGRATIONS.md`
- `safemoe/config.py`
- `safemoe/model.py`
- `safemoe/masking.py`
- `safemoe/pretrain.py`
- `safemoe/evaluate.py`
- `safemoe/ablate.py`
- `safemoe/data/datamodule.py`
- `litgpt/config.py`
- `litgpt/scripts/convert_hf_checkpoint.py`
- `tests/test_model.py`

---
*Architecture research for: SafeMoE v1.1 direct Qwen harmful-transfer integration*
