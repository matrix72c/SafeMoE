# Phase 3: SGTM Training Loop - Research

**Researched:** 2026-03-16
**Domain:** PyTorch training loop engineering — dual-optimizer fork of litgpt/pretrain.py, SGTM 3-path masking integration, Lightning Fabric gradient accumulation, jsonargparse CLI
**Confidence:** HIGH (all findings grounded in source code read directly from the repo)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Attention head masking scope**
- Phase 3 extends BOTH GradientMasker and ActivationMasker to cover `harmful_attn_heads` (not just MoE experts)
- GradientMasker: post-backward zeroing of QKV-row gradients in `attn.attn.weight` — same explicit post-backward approach as expert masking; no hooks
- ActivationMasker: zeros harmful head output contributions before `attn.proj` during D_std forward (zero the head-size rows in the attention output tensor for each designated harmful head)
- `harmful_attn_heads: [0, 1]` set in `safemoe-tinystories.yaml` experiment config; Phase 3 unit tests exercise this masking path explicitly

**Default sampling weights**
- `upsample_std`, `upsample_harmful`, `upsample_unlabeled` are required YAML fields — no opinionated defaults
- Training fails loudly with a clear error if any are missing (no silent fallback to 1:1:1)
- Fields live flat at the top-level config (not nested under a sub-section)
- `safemoe-tinystories.yaml` uses `upsample_std: 1`, `upsample_harmful: 1`, `upsample_unlabeled: 1`

**Gradient accumulation + SGTM interaction**
- One split label per optimizer step: all micro-batches in an accumulation window use the same split
- Masker called once per optimizer step (not per micro-batch):
  - D_std: `activation_masker.enable()` before first micro-batch forward, `activation_masker.disable()` after last forward
  - D_harmful: accumulate all micro-batch backwards, `gradient_masker.mask()` once after final backward, before `optimizer.step()`
  - D_unlabeled: no masking
- DDP sync: `fabric.no_backward_sync(model, enabled=is_accumulating)` — identical to litgpt pattern
- `safemoe-tinystories.yaml`: `micro_batch_size: 4`, `gradient_accumulation_iters: 4` (effective batch size = 16)

**Dual optimizer LR schedule**
- Shared LR schedule: both theta_harmful and theta_std AdamW use same LR, warmup, min_lr, weight_decay from TrainArgs
- Per-split selective stepping:
  - D_harmful step: only theta_harmful optimizer steps; theta_std calls `zero_grad(set_to_none=True)` only
  - D_std step: only theta_std optimizer steps; theta_harmful calls `zero_grad(set_to_none=True)` only
  - D_unlabeled step: BOTH optimizers step
- Gradient clipping: two separate `fabric.clip_gradients()` calls — one per optimizer, applied only when that optimizer is active
- LR counter: advances once per optimizer step regardless of which optimizer(s) stepped

### Claude's Discretion
- Internal loop structure for the 3-path branching (if/elif/else vs. dispatch dict)
- How the split label is stored in the training `state` dict for checkpoint resumability
- Logging format for per-split loss tracking during training
- Whether two separate `fabric.setup_optimizers()` calls are needed or if both can be passed together

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TRAIN-01 | Fork `litgpt/pretrain.py` -> `safemoe/pretrain.py` implementing SGTM 3-path branching per step label | litgpt/pretrain.py read in full; exact fork target identified; gradient accumulation, CycleIterator, save_checkpoint, validate patterns all extracted |
| TRAIN-02 | `sgtm_mode` scalar passed as part of batch dict to model forward; `adjust_gradients(sgtm_mode)` called after each backward before optimizer step | GradientMasker.mask() and ActivationMasker.enable()/disable() interfaces confirmed from safemoe/masking.py; Phase 3 extends both to cover attn heads |
| TRAIN-03 | CLI entry point `python -m safemoe pretrain` with YAML config support, consistent with LitGPT's jsonargparse conventions | litgpt/__main__.py CLI pattern confirmed (jsonargparse CLI over PARSER_DATA dict); safemoe/__main__.py does not yet exist — must be created |
</phase_requirements>

---

## Summary

Phase 3 forks `litgpt/pretrain.py` into `safemoe/pretrain.py`, the primary change being the replacement of a single training stream with three parallel streams (D_std, D_harmful, D_unlabeled) that each use a different masking path. The litgpt fork target is well-understood: the key patterns are `fabric.launch()` -> `setup()` -> `main()` -> `fit()`, `CycleIterator` for infinite data iteration, `get_lr()` cosine schedule with warmup, `save_checkpoint()` with hyperparameter saving, and `fabric.no_backward_sync()` for gradient accumulation. All of these are reused without modification.

The principal novelty is the dual-optimizer architecture. Two separate AdamW instances (`optimizer_harmful` over `theta_harmful` params, `optimizer_std` over `theta_std` params) replace the single optimizer from litgpt. At each optimizer step, a split label is sampled via `random.choices()` from the three labels weighted by `upsample_*` factors (these weights live in the training loop, not in MultiDataLoader). The split label governs which masker runs and which optimizer(s) step. GradientMasker and ActivationMasker from Phase 2 are extended in this phase to also cover attention heads (`harmful_attn_heads`), but their call sites in the training loop are straightforward: `enable()/disable()` brackets the D_std micro-batch window; `mask()` is called once after the accumulation window closes for D_harmful; nothing special for D_unlabeled.

The CLI entry point `python -m safemoe pretrain` requires creating `safemoe/__main__.py` that mirrors `litgpt/__main__.py` — a `PARSER_DATA` dict mapping `"pretrain"` to `safemoe.pretrain.setup`, then `jsonargparse.CLI(PARSER_DATA)`. The YAML config follows the same flat-field pattern as litgpt's `configs/tinystories/*.yaml`.

**Primary recommendation:** Fork litgpt/pretrain.py with surgical changes — keep 90% of the code untouched, add the dual-optimizer setup, replace the single `CycleIterator` with three iterators + split sampling, and wrap the inner loop body in an if/elif/else on the current split label.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| lightning (fabric) | >=2.6.1 | Device setup, backward sync, gradient clipping, checkpointing | Already used in litgpt; fabric is the single-device + multi-GPU abstraction |
| torch | >=2.7 | AdamW optimizer, tensor ops, gradient manipulation | Core training engine |
| jsonargparse | 4.37-4.41 | YAML + CLI arg parsing for setup() | Already used by all litgpt pretrain commands |
| litdata | 0.2.59 | StreamingDataLoader for each split | Already used in Phase 1 MultiDataLoader |
| torchmetrics | >=1.3.1 | RunningMean for smoothed loss logging | Used in litgpt/pretrain.py |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tensorboard | >=2.14 | Loss/LR metric logging | Default logger; wandb optional |
| lightning.fabric.utilities.throughput | (same as lightning) | ThroughputMonitor | Tokens/sec logging — reuse from litgpt |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| if/elif/else branch | dispatch dict `{label: handler_fn}` | Dict is cleaner for 3 paths but adds indirection; if/elif/else is more readable for a research codebase |
| Two separate `fabric.setup_optimizers()` | Single call with a list | Lightning Fabric accepts a list of optimizers in one call — recommended for cleaner state management |

**Installation:** No new packages needed — all dependencies already in `pyproject.toml` extras.

---

## Architecture Patterns

### Recommended Project Structure
```
safemoe/
├── pretrain.py          # Fork of litgpt/pretrain.py — SGTM training loop
├── __main__.py          # CLI entry: python -m safemoe pretrain
├── masking.py           # Extended GradientMasker + ActivationMasker (attn heads)
├── config.py            # SafeMoEConfig (unchanged)
├── model.py             # SafeMoELayer (unchanged)
├── configs/
│   └── safemoe-tinystories.yaml  # Updated with upsample_* + micro_batch_size + harmful_attn_heads
└── data/
    ├── datamodule.py    # MultiDataLoader (unchanged)
    └── prepare.py       # (unchanged)
tests/safemoe/
└── test_pretrain.py     # New — TRAIN-01/02/03 unit tests (CPU-only small model)
```

### Pattern 1: Split Sampling (weighted random.choices)
**What:** Each optimizer step samples one split label from `['D_std', 'D_harmful', 'D_unlabeled']` weighted by upsample factors. This replaces litgpt's single `CycleIterator`.
**When to use:** At the start of each accumulation window (once per optimizer step, before any micro-batch).

```python
# Source: CONTEXT.md decision + DATA-02/DATA-03 spec
import random

SPLIT_LABELS = ["D_std", "D_harmful", "D_unlabeled"]

# At training loop setup:
weights = [train.upsample_std, train.upsample_harmful, train.upsample_unlabeled]

# At start of each optimizer step:
split_label = random.choices(SPLIT_LABELS, weights=weights, k=1)[0]

# Then iterate micro-batches from the sampled split:
batch = next(split_iters[split_label])
```

### Pattern 2: Three-Path Masking (if/elif/else on split label)
**What:** Inner accumulation loop checks `split_label` once at the optimizer step boundary.
**When to use:** Inside `fit()`, wrapping the `is_accumulating` boundary actions.

```python
# Source: CONTEXT.md gradient accumulation + SGTM interaction decisions
if split_label == "D_std":
    activation_masker.enable()

for micro_batch_idx in range(accum_iters):
    is_accumulating = micro_batch_idx < accum_iters - 1
    with fabric.no_backward_sync(model, enabled=is_accumulating):
        logits = model(input_ids)
        loss = chunked_cross_entropy(logits, targets)
        fabric.backward(loss / accum_iters)

if split_label == "D_std":
    activation_masker.disable()
    fabric.clip_gradients(model, optimizer_std, max_norm=train.max_norm)
    optimizer_std.step()
    optimizer_std.zero_grad(set_to_none=True)
    optimizer_harmful.zero_grad(set_to_none=True)

elif split_label == "D_harmful":
    gradient_masker.mask()
    fabric.clip_gradients(model, optimizer_harmful, max_norm=train.max_norm)
    optimizer_harmful.step()
    optimizer_harmful.zero_grad(set_to_none=True)
    optimizer_std.zero_grad(set_to_none=True)

else:  # D_unlabeled
    fabric.clip_gradients(model, optimizer_harmful, max_norm=train.max_norm)
    fabric.clip_gradients(model, optimizer_std, max_norm=train.max_norm)
    optimizer_harmful.step()
    optimizer_std.step()
    optimizer_harmful.zero_grad(set_to_none=True)
    optimizer_std.zero_grad(set_to_none=True)

state["step_count"] += 1
```

### Pattern 3: Dual Optimizer Setup
**What:** Two AdamW instances from `instantiate_torch_optimizer`, wrapped by Fabric together.
**When to use:** In `main()` after model setup, before training starts.

```python
# Source: litgpt/utils.py instantiate_torch_optimizer + CONTEXT.md dual optimizer decision
from litgpt.utils import instantiate_torch_optimizer
from safemoe.masking import HarmfulParamRegistry

registry = HarmfulParamRegistry(model, config)
extra_kwargs = {"fused": fabric.device.type == "cuda"}

optimizer_harmful = instantiate_torch_optimizer(
    optimizer, registry.parameters_by_type("theta_harmful"), **extra_kwargs
)
optimizer_std = instantiate_torch_optimizer(
    optimizer, registry.parameters_by_type("theta_std"), **extra_kwargs
)
# Single fabric.setup_optimizers() call with both:
optimizer_harmful, optimizer_std = fabric.setup_optimizers(optimizer_harmful, optimizer_std)
```

### Pattern 4: Dual LR Update
**What:** Both optimizers receive the same LR from `get_lr()` keyed on the same `iter_num`.
**When to use:** At the start of each micro-batch iteration (same as litgpt's single-optimizer LR update).

```python
# Source: litgpt/pretrain.py get_lr() + CONTEXT.md LR counter decision
lr = get_lr(base_lr, state["iter_num"], warmup_iters, max_iters, train.min_lr)
for opt in (optimizer_harmful, optimizer_std):
    for pg in opt.param_groups:
        pg["lr"] = lr
```

### Pattern 5: Checkpoint State Dict with Dual Optimizers
**What:** `state` dict carries both optimizers for resume support.
**When to use:** In `main()` when constructing state and when calling `save_checkpoint`.

```python
# Source: litgpt/pretrain.py state dict pattern + CONTEXT.md checkpoint decision
state = {
    "model": model,
    "optimizer_harmful": optimizer_harmful,
    "optimizer_std": optimizer_std,
    "iter_num": 0,
    "step_count": 0,
    "split_label": "D_std",    # Last sampled label — stored for resumability (Claude's discretion)
}
fabric.load(resume, state)  # Restores all keys including both optimizers
```

### Pattern 6: CLI Entry Point (safemoe/__main__.py)
**What:** Mirror of `litgpt/__main__.py` registering only the `pretrain` subcommand.
**When to use:** Enables `python -m safemoe pretrain --config safemoe-tinystories.yaml`.

```python
# Source: litgpt/__main__.py pattern
from jsonargparse import CLI
from safemoe.pretrain import setup as pretrain_fn

PARSER_DATA = {"pretrain": pretrain_fn}

def main():
    CLI(PARSER_DATA)

if __name__ == "__main__":
    main()
```

### Pattern 7: Attention Head Activation Masking in CausalSelfAttention
**What:** ActivationMasker needs to zero head output contributions before `attn.proj`. The attention output tensor after `scaled_dot_product_attention` has shape `(B, nh_q, T, hs)`. Zeroing head `i` means setting `attn_out[:, i, :, :] = 0` before the reshape and proj.
**When to use:** When `harmful_attn_heads` is non-empty and ActivationMasker is enabled.
**Implementation approach:** Subclass `CausalSelfAttention` as `SafeCausalSelfAttention` with a `_harmful_heads` list and `_activation_masking_enabled` flag (analogous to SafeMoELayer), OR inject the flag into existing CausalSelfAttention instances at setup time. The SafeMoEConfig drives which approach to use. Given the SafeMoELayer precedent, subclassing is the cleaner approach and consistent with Phase 2 patterns.

### Anti-Patterns to Avoid
- **Calling masker per micro-batch:** The masker must be called once per optimizer step at the accumulation boundary. Calling `gradient_masker.mask()` after every micro-batch backward would corrupt gradients across the accumulation window.
- **Mixed split labels within an accumulation window:** Sampling a new split label per micro-batch would break gradient isolation — the CONTEXT.md is explicit that one label covers the entire accumulation window.
- **Stepping the idle optimizer on D_harmful/D_std steps:** Only `zero_grad(set_to_none=True)` on the inactive optimizer. Calling `.step()` would accumulate Adam momentum state even with zero (None) gradients.
- **Using `p.grad.zero_()` instead of `p.grad = None`:** `zero_()` leaves the grad tensor allocated and triggers Adam state accumulation on the next step. `= None` is the established pattern from Phase 2.
- **Putting upsample weights in MultiDataLoader:** Confirmed in Phase 1 SUMMARY: MultiDataLoader deliberately has no upsample fields. The training loop owns weighted sampling.
- **torch.compile on SafeMoE model:** The `_activation_masking_enabled` flag is a Python bool — `torch.compile` may trace the control flow once and bake in the value. Test with `torch.compile` disabled first; enable only if confirmed working.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Cosine LR schedule with warmup | Custom scheduler | `get_lr()` from litgpt/pretrain.py | Already correct, handles warmup + cosine + min_lr edge cases |
| Infinite data cycling | Custom cycle logic | `CycleIterator` from litgpt/utils.py | Handles StopIteration cleanly, exposes epoch counter |
| Optimizer instantiation from name/dict | Custom factory | `instantiate_torch_optimizer()` from litgpt/utils.py | Handles string name, dict with class_path, fused kwarg filtering |
| Checkpoint save/load | Custom pickle/torch.save | `fabric.save()` / `fabric.load()` + `save_checkpoint()` from litgpt | Handles FSDP state_dict, rank-zero guarantees, hyperparameter YAML |
| CLI YAML parsing | argparse + custom YAML load | `jsonargparse.CLI()` | Handles nested dataclasses, config file merging, type coercion automatically |
| GPU memory measurement | psutil / custom | `torch.cuda.max_memory_allocated()` | Already in litgpt pretrain final summary block |
| Throughput tracking | Manual time arithmetic | `ThroughputMonitor` from lightning.fabric | Windowed FLOP/s, tok/s, already wired in litgpt |

**Key insight:** The litgpt codebase solves nearly every infrastructure problem already. Phase 3's job is surgical modification of the training logic, not rebuilding infrastructure.

---

## Common Pitfalls

### Pitfall 1: Activation Masking Left Enabled After Forward Error
**What goes wrong:** If a D_std forward pass raises an exception (OOM, assertion), `activation_masker.disable()` in the except path is skipped. Subsequent D_unlabeled and D_harmful steps run with masking still enabled, silently suppressing harmful expert contributions.
**Why it happens:** `enable()/disable()` are stateful; exceptions skip the disable call.
**How to avoid:** Use a context manager or try/finally:
```python
activation_masker.enable()
try:
    # micro-batch forward passes
finally:
    activation_masker.disable()
```
**Warning signs:** D_unlabeled loss unexpectedly lower than D_harmful loss; harmful expert parameters have zero gradients on D_unlabeled steps.

### Pitfall 2: fabric.setup_optimizers() Return Order
**What goes wrong:** `fabric.setup_optimizers(opt_a, opt_b)` returns wrapped optimizers in the same order as arguments. Swapping the return assignment silently makes `optimizer_harmful` point to the std param group and vice versa.
**Why it happens:** Both optimizers are `AdamW` instances — no type difference to catch the swap.
**How to avoid:** Name explicitly at the call site:
```python
optimizer_harmful, optimizer_std = fabric.setup_optimizers(optimizer_harmful, optimizer_std)
```
**Warning signs:** D_harmful backward updates theta_std parameters; loss fails to decrease for harmful splits.

### Pitfall 3: `random.choices()` Seed Reproducibility
**What goes wrong:** `random.choices()` uses Python's global random state. Without seeding before training, runs are not reproducible. Resuming from checkpoint without restoring the random state causes a different split sequence from the resume point.
**Why it happens:** `fabric.seed_everything(seed)` seeds numpy, torch, and random simultaneously, but only at training start — not at resume time.
**How to avoid:** Store and restore `random.getstate()` in the checkpoint `state` dict, or seed deterministically from `iter_num` at resume time. Simplest: `random.seed(seed + state["iter_num"])` after loading checkpoint.
**Warning signs:** Identical configs produce different final checkpoints across restarts.

### Pitfall 4: Attention Head Masking Dimension Mismatch in ActivationMasker
**What goes wrong:** CausalSelfAttention attention output has shape `(B, n_head, T, head_size)` after transpose but before proj. Zeroing the wrong dimension axis silently produces incorrect masking (e.g. zeroing a time step instead of a head).
**Why it happens:** The tensor is transposed twice during forward — easy to target the wrong axis index.
**How to avoid:** Zero `attn_out[:, harmful_head_idx, :, :]` — axis 1 is the head dimension post-transpose. Confirm with `assert attn_out.shape[1] == config.n_head` in a debug assertion.
**Warning signs:** Validation loss worse than expected on D_std; harmful experts still have non-zero contributions on D_std steps.

### Pitfall 5: `gradient_accumulation_iters` Calculation vs. Accumulation Window
**What goes wrong:** litgpt's `TrainArgs.gradient_accumulation_iters(devices, num_nodes)` computes accumulation from `global_batch_size // (micro_batch_size * devices * num_nodes)`. If `devices > 1`, the accum window is smaller per device. The SGTM loop must use the same calculation or it will run the wrong number of micro-batches before the masker fires.
**Why it happens:** Forgetting to pass `devices` and `num_nodes` to `gradient_accumulation_iters()`.
**How to avoid:** Compute `accum_iters = train.gradient_accumulation_iters(devices, num_nodes)` once in `fit()` and use consistently.
**Warning signs:** AssertionError in `gradient_accumulation_iters()` or loss NaN due to misaligned accumulation window.

### Pitfall 6: GradientMasker Extension for Attention Heads
**What goes wrong:** `GradientMasker.mask()` currently sets all `theta_std` gradients to `None`. After Phase 3 extends it to also handle qkv row-slice masking (from `_qkv_harmful_metadata`), the masker needs to set only the harmful-head rows of `qkv.weight.grad` to zero, NOT the full grad tensor (since qkv.weight is in theta_std). Incorrect implementation would either set the entire qkv grad to None (losing std gradient signal) or fail to zero the harmful head rows.
**Why it happens:** The `_qkv_harmful_metadata` duality from Phase 2: qkv.weight is in theta_std, but specific row slices belong semantically to theta_harmful.
**How to avoid:** In the extended `mask()`: iterate `_registry._qkv_harmful_metadata`, and for each `(param, slices)` where `param.grad is not None`, set `param.grad[s] = 0` for each slice `s`.
**Warning signs:** QKV weight gradient norm on D_harmful steps is larger than expected (harmful head rows not being zeroed), or qkv gradient is None entirely (whole grad was wrongly cleared).

---

## Code Examples

### Split-Weighted Sampling at Optimizer Step Boundary
```python
# Source: CONTEXT.md + Phase 1 plan 01-02-SUMMARY.md upsample design
SPLIT_LABELS = ["D_std", "D_harmful", "D_unlabeled"]

# Before fit() loop:
weights = [train.upsample_std, train.upsample_harmful, train.upsample_unlabeled]
split_iters = {
    label: CycleIterator(data.get_loader(label))
    for label in SPLIT_LABELS
}

# At start of each optimizer step:
split_label = random.choices(SPLIT_LABELS, weights=weights, k=1)[0]
state["split_label"] = split_label  # for checkpoint resumability
```

### iter_num vs step_count Tracking
```python
# Source: litgpt/pretrain.py — iter_num counts micro-batches; step_count counts optimizer steps
# In the SGTM loop, iter_num still increments every micro-batch.
# step_count increments only at the optimizer step boundary (same as litgpt).
state["iter_num"] += 1  # every micro-batch
if not is_accumulating:
    # ... masking + optimizer steps ...
    state["step_count"] += 1

# For per-split loss logging, use step_count as the x-axis:
# e.g. metrics["loss_D_std"] logged only when split_label == "D_std"
```

### Masker Extension for Attention Heads in GradientMasker
```python
# Source: safemoe/masking.py _qkv_harmful_metadata design (Phase 2)
def mask(self) -> None:
    # Step 1: zero all theta_std gradients (existing behavior)
    for p in self._registry.parameters_by_type("theta_std"):
        p.grad = None

    # Step 2: for qkv.weight, re-zero only the harmful-head rows
    # (qkv.weight is in theta_std, so its grad was set to None above;
    # but we need the std-head rows to have gradients — restore them)
    # NOTE: This requires a forward hook or a pre-save of the full grad.
    # Simpler approach: set only harmful-head rows to zero, leave rest.
    # Implementation: do NOT set qkv.weight.grad = None in step 1;
    # handle it separately.
```

**Implementation note:** The cleanest approach is to exclude qkv.weight params from the `p.grad = None` sweep in step 1, then handle them separately in step 2: set `p.grad[s] = 0` for each harmful-head slice `s`. This requires a small refactor of `GradientMasker.mask()` to split its iteration into two passes.

### YAML Config Fields to Add (safemoe-tinystories.yaml)
```yaml
# Source: CONTEXT.md decisions — flat top-level, required fields, no defaults
# Training loop fields (add to existing safemoe-tinystories.yaml):
harmful_attn_heads: [0, 1]      # was [] in Phase 2 config — updated to [0, 1]
upsample_std: 1
upsample_harmful: 1
upsample_unlabeled: 1
micro_batch_size: 4
gradient_accumulation_iters: 4  # effective batch = 16
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single optimizer in litgpt pretrain | Dual optimizer (theta_harmful + theta_std) | Phase 3 (SGTM) | Adam momentum isolated per parameter set |
| CycleIterator over single train_dataloader | Three CycleIterators + weighted random.choices | Phase 3 | Each step draws from one split |
| No masking in forward/backward | GradientMasker (post-backward) / ActivationMasker (in-forward) | Phase 2/3 | Gradient isolation between theta sets |
| Pre-generated data_split_order list (REQUIREMENTS.md DATA-03 original spec) | Dynamic weighted sampling via random.choices() | Phase 1 CONTEXT.md override | Simpler, no pre-generated schedule file needed |

**Superseded:**
- Pre-generated `data_split_order` file: overridden by Phase 1 CONTEXT.md decision. The training loop samples split labels dynamically via `random.choices()` with upsample weights.

---

## Open Questions

1. **torch.compile compatibility with _activation_masking_enabled flag**
   - What we know: SafeMoELayer forward reads `self._activation_masking_enabled` as a Python bool; torch.compile may trace this once
   - What's unclear: Whether `torch.compile` graph breaks cleanly on the flag change between steps or silently caches the traced path
   - Recommendation: Keep `torch.compile` from litgpt's `main()` disabled for safemoe initially (delete the `model = torch.compile(model)` line), add it back only after confirming correctness without it

2. **Attention head ActivationMasker: subclass vs. flag injection**
   - What we know: CausalSelfAttention is constructed inside litgpt Block; SafeMoEConfig.mlp_class redirects MLP construction to SafeMoELayer but there is no equivalent hook for attention
   - What's unclear: The cleanest way to inject `_activation_masking_enabled` + `_harmful_heads` into existing `CausalSelfAttention` instances without subclassing Block
   - Recommendation: At model construction time in `main()`, iterate `model.modules()` to find `CausalSelfAttention` instances and monkey-patch `_activation_masking_enabled = False` and `_harmful_heads = config.harmful_attn_heads`. ActivationMasker.enable()/disable() then sets the flag on these instances (same pattern as SafeMoELayer). This avoids subclassing Block entirely.

3. **GradientMasker QKV row-slice implementation: two-pass vs. exclusion list**
   - What we know: qkv.weight is in theta_std; `_qkv_harmful_metadata` stores `(param, [slice, ...])` pairs
   - What's unclear: Whether to exclude qkv.weight from the `p.grad = None` sweep and handle separately, or to save and restore qkv grad rows
   - Recommendation: Exclusion list approach — build a set of qkv.weight param ids in `__init__`, skip them in the `p.grad = None` loop, then zero only the harmful-head rows. This is O(1) overhead and conceptually clean.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >= 8.1.1 |
| Config file | pyproject.toml `[tool.pytest.ini_options]` (--strict-markers, --color=yes) |
| Quick run command | `pytest tests/safemoe/test_pretrain.py -x` |
| Full suite command | `pytest tests/safemoe/ -x` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TRAIN-01 | fit() runs one D_harmful step: GradientMasker.mask() called, theta_std.grad is None after step | unit | `pytest tests/safemoe/test_pretrain.py::test_fit_harmful_step_masks_theta_std -x` | ❌ Wave 0 |
| TRAIN-01 | fit() runs one D_std step: ActivationMasker enabled during forward, disabled after | unit | `pytest tests/safemoe/test_pretrain.py::test_fit_std_step_enables_activation_masker -x` | ❌ Wave 0 |
| TRAIN-01 | fit() runs one D_unlabeled step: no masking, both optimizers step | unit | `pytest tests/safemoe/test_pretrain.py::test_fit_unlabeled_step_no_masking -x` | ❌ Wave 0 |
| TRAIN-01 | Gradient accumulation: masker called once per optimizer step (not per micro-batch) | unit | `pytest tests/safemoe/test_pretrain.py::test_masker_called_once_per_step -x` | ❌ Wave 0 |
| TRAIN-01 | Attn head gradient masking: qkv harmful-head rows zeroed on D_harmful backward | unit | `pytest tests/safemoe/test_pretrain.py::test_attn_head_gradient_masking -x` | ❌ Wave 0 |
| TRAIN-02 | D_std forward with harmful_attn_heads=[0,1]: harmful head output contributions are zero | unit | `pytest tests/safemoe/test_pretrain.py::test_attn_head_activation_masking -x` | ❌ Wave 0 |
| TRAIN-03 | `python -m safemoe pretrain --help` exits 0 and shows pretrain args | smoke | `python -m safemoe pretrain --help` (manual check) | ❌ Wave 0 |
| TRAIN-03 | safemoe pretrain with tiny config produces lit_model.pth checkpoint | integration | `pytest tests/safemoe/test_pretrain.py::test_pretrain_produces_checkpoint -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/safemoe/test_pretrain.py -x`
- **Per wave merge:** `pytest tests/safemoe/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/safemoe/test_pretrain.py` — covers TRAIN-01, TRAIN-02, TRAIN-03
- [ ] `safemoe/__main__.py` — required for `python -m safemoe pretrain` CLI entry point
- [ ] `safemoe/pretrain.py` — the main deliverable (does not exist yet)

*(No new test infrastructure needed — pytest already configured in pyproject.toml; tests/safemoe/ directory already exists)*

---

## Sources

### Primary (HIGH confidence)
- `/mnt/.../safemoe/litgpt/pretrain.py` — full fork target read; all patterns extracted directly from source
- `/mnt/.../safemoe/safemoe/masking.py` — GradientMasker, ActivationMasker, HarmfulParamRegistry interfaces confirmed
- `/mnt/.../safemoe/safemoe/data/datamodule.py` — MultiDataLoader.get_loader() interface confirmed; upsample weights confirmed absent
- `/mnt/.../safemoe/litgpt/utils.py` — CycleIterator, instantiate_torch_optimizer, capture_hparams confirmed
- `/mnt/.../safemoe/litgpt/args.py` — TrainArgs.gradient_accumulation_iters() signature confirmed
- `/mnt/.../safemoe/litgpt/parser_config.py` — save_hyperparameters() pattern confirmed
- `/mnt/.../safemoe/litgpt/__main__.py` — PARSER_DATA + jsonargparse.CLI() pattern confirmed
- `/mnt/.../safemoe/litgpt/model.py` — CausalSelfAttention forward: attention output shape `(B, n_head, T, head_size)` confirmed
- `/mnt/.../safemoe/pyproject.toml` — pytest config, dependency versions confirmed
- `.planning/phases/03-sgtm-training-loop/03-CONTEXT.md` — all locked decisions
- `.planning/phases/01-data-pipeline/01-02-SUMMARY.md` — confirmed upsample weights are Phase 3 responsibility, not MultiDataLoader

### Secondary (MEDIUM confidence)
- `.planning/phases/01-data-pipeline/01-RESEARCH.md` — weighted random.choices sampling design (verified as the implemented approach from SUMMARY)

### Tertiary (LOW confidence)
- torch.compile + Python bool flag interaction: asserted from general knowledge; not verified against a specific PyTorch 2.7 release note

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries confirmed present in pyproject.toml and used in litgpt codebase
- Architecture: HIGH — fork target read in full; all patterns extracted from source code
- Pitfalls: HIGH — most derived directly from code inspection (e.g. qkv duality from masking.py comments, accumulation from pretrain.py loop structure); one (torch.compile) is MEDIUM
- Attention head masking design: MEDIUM — approach is sound but exact injection pattern (monkey-patch vs. SafeBlock subclass) is Claude's discretion; both options are technically correct

**Research date:** 2026-03-16
**Valid until:** 2026-04-16 (stable codebase; litgpt and torch versions pinned in pyproject.toml)
