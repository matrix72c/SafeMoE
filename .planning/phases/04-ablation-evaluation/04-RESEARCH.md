# Phase 4: Ablation & Evaluation - Research

**Researched:** 2026-03-16
**Domain:** Checkpoint manipulation, perplexity evaluation, forward-hook routing attribution, in-training ablation — all within the litgpt / Lightning Fabric stack
**Confidence:** HIGH (all findings grounded in source code read directly from the repo)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Ablation utility (TRAIN-04)**
- Standalone CLI: `python -m safemoe ablate <ckpt_dir>` — consistent with `python -m safemoe pretrain` pattern
- Saves ablated checkpoint to `<ckpt_dir>/ablated/lit_model.pth` — sibling directory alongside the original
- Zeroes all theta_harmful weights to 0.0 in-place using HarmfulParamRegistry to identify them (exactly as the paper describes)
- Operates on full-precision consolidated `lit_model.pth` only — no FSDP sharded checkpoint handling
- Reports verbose output: prints summary table (parameters zeroed, expert indices, total norm before/after) AND saves `<ckpt_dir>/ablated/ablation_manifest.json` listing every zeroed parameter name with its pre-ablation norm

**Perplexity evaluation entry point (EVAL-01)**
- Standalone CLI: `python -m safemoe evaluate --original <ckpt_dir> --ablated <ckpt_dir>/ablated`
- Accepts two checkpoint paths for pre/post comparison (both paths required for comparison table; single path works for non-comparison eval)
- Loads model config and tokenizer from checkpoint directory (no separate --config needed — matches LitGPT checkpoint convention)
- Requires data to be pre-prepared on disk (data/.cache/ must exist — consistent with pretrain.py)
- Runs on all available validation tokens (no max_iters cap for the evaluate CLI — exact perplexity)
- Reports: prints comparison table (pre/post ppl per split) AND writes `results.json` to output directory
- Data source: same `data/.cache/` path as training; x/y params read from checkpoint's saved config

**Routing attribution (EVAL-02)**
- Collected during a post-training analysis pass — no overhead during training
- Triggered via `--routing` flag on the evaluate CLI: `python -m safemoe evaluate --original <ckpt_dir> --routing`
- Captures theta_harmful activation fraction per split (fraction of tokens routing to theta_harmful experts, per D_std / D_harmful / D_unlabeled split) — sufficient to validate thesis
- Implementation: forward hooks on SafeMoELayer during eval inference, accumulate expert dispatch counts
- Output: TensorBoard histograms (logged to checkpoint's existing runs/ directory) AND `routing_attribution.json` with raw per-split fractions

**Mid-training ablation evaluation (EVAL-03)**
- Runs only at `save_interval` checkpoints (not every eval.interval) — minimal training overhead
- Implementation: in-place zero + restore — clone theta_harmful weights before zeroing, run validation pass on ablated model, then restore originals from clone; no separate checkpoint file, no model copy
- Logged to TensorBoard alongside regular val loss curves in the same run — metric names: `ablated_val_ppl_D_std`, `ablated_val_ppl_D_harmful`, `ablated_val_ppl_D_unlabeled`
- Implemented inside `pretrain.py` at the checkpoint-save code path (alongside `save_checkpoint()`)

**Metric naming convention**
- Flat prefix scheme throughout TensorBoard and JSON:
  - Regular: `val_ppl_D_std`, `val_ppl_D_harmful`, `val_ppl_D_unlabeled`
  - Ablated: `ablated_val_ppl_D_std`, `ablated_val_ppl_D_harmful`, `ablated_val_ppl_D_unlabeled`
  - Routing: `routing_harmful_frac_D_std`, `routing_harmful_frac_D_harmful`, `routing_harmful_frac_D_unlabeled`
  - Consistent with existing `loss_D_std` / `loss_D_harmful` naming in pretrain.py

**Eval portability**
- Same machine as training only — data must be on disk at data/.cache/
- No portability shims for Milestone 1 (research scale, single machine)

### Claude's Discretion

None specified — all decisions are locked.

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TRAIN-04 | `ablate()` utility zeros theta_harmful weights in-place and saves the ablated checkpoint as a separate file for inference evaluation | HarmfulParamRegistry.parameters_by_type('theta_harmful') confirmed working; fabric.load/save checkpoint pattern confirmed; ablated/ sibling dir pattern decided |
| EVAL-01 | Per-split perplexity evaluation (D_std / D_harmful / D_unlabeled) on validation sets before and after ablation | pretrain.validate() confirmed reusable; MultiDataLoader.val_dataloaders() returns {D_std, D_harmful} dict; perplexity = exp(loss) trivially derived from val_loss |
| EVAL-02 | Routing attribution analysis — per-token histogram of which expert type each data split preferentially activates | SafeMoELayer.forward() iterates experts with explicit expert_idx — forward hook on SafeMoELayer can capture dispatch counts; TensorBoard logging via fabric.log_dict confirmed |
| EVAL-03 | Mid-training ablation evaluation — at each eval checkpoint, temporarily ablate theta_harmful and evaluate the ablated model, then restore; tracks isolation progress | In-place zero+restore pattern on theta_harmful params is safe and cheap (small subset); clone-before-zero pattern confirmed; validate() is @torch.no_grad and reusable |
</phase_requirements>

---

## Summary

Phase 4 builds three interconnected evaluation capabilities on top of the fully-working Phase 3 training infrastructure. All the hard infrastructure (HarmfulParamRegistry, validate(), MultiDataLoader val splits, TensorBoard logging via fabric.log_dict, checkpoint save/load patterns) already exists and is proven. Phase 4 is predominantly glue work — wiring existing components into new entry points with well-defined inputs and outputs.

The four requirements decompose naturally into three new files and one modification: `safemoe/ablate.py` (TRAIN-04 ablation utility), `safemoe/evaluate.py` (EVAL-01 + EVAL-02 evaluate CLI), a modification to `safemoe/pretrain.py` to add `evaluate_with_ablation()` called at save_interval (EVAL-03), and an update to `safemoe/__main__.py` to dispatch `ablate` and `evaluate` subcommands. The most novel component is the routing attribution forward hook (EVAL-02), which attaches to SafeMoELayer.forward() to count expert dispatch events per split — this is stateless and easily cleaned up after the analysis pass.

The key risk area is EVAL-03 (in-place zero + restore in pretrain.py): cloning only theta_harmful params (a small subset) avoids the 2x memory cost of copying the full model, but the clone-zero-validate-restore sequence must be carefully ordered to avoid leaving the model in an ablated state if an exception occurs during validation. A try/finally guard analogous to the ActivationMasker pattern used in Phase 3 is the correct approach.

**Primary recommendation:** Build `safemoe/ablate.py` first (standalone, testable independently), then `safemoe/evaluate.py` (builds on ablate + validate), then EVAL-03 in pretrain.py (builds on both), then wire everything into `__main__.py`.

---

## Standard Stack

### Core (all already in the project — no new dependencies needed)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Lightning Fabric | (existing) | Model setup, fabric.load/save, fabric.log_dict | All training infrastructure already uses Fabric |
| PyTorch | (existing) | Tensor operations, no_grad, torch.clone | Foundation; used by all existing code |
| jsonargparse | (existing) | CLI argument parsing for ablate/evaluate subcommands | Pattern established in pretrain/__main__.py |
| TensorBoard | (existing) | Routing histograms and mid-training ablation metrics | Already the logger in training runs |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| json (stdlib) | stdlib | Write ablation_manifest.json, results.json, routing_attribution.json | All JSON output files |
| math (stdlib) | stdlib | `math.exp(loss)` for perplexity conversion | Perplexity = exp(cross-entropy loss) |
| pathlib.Path | stdlib | Checkpoint path manipulation | Consistent with all existing code |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| fabric.load() for checkpoint loading | torch.load() directly | fabric.load handles precision/device placement automatically; torch.load requires manual map_location |
| forward hook for routing attribution | Modify SafeMoELayer.forward() to optionally log | Hook is cleaner — zero-overhead when not active, no persistent state in the layer |
| clone + restore for EVAL-03 | Full model deep copy | Deep copy doubles GPU memory; clone of theta_harmful (2 experts) is negligible |

**Installation:** No new packages needed.

---

## Architecture Patterns

### Recommended New File Structure
```
safemoe/
├── ablate.py          # TRAIN-04: ablate() function + CLI setup()
├── evaluate.py        # EVAL-01 + EVAL-02: evaluate() function + CLI setup()
├── pretrain.py        # EVAL-03: add evaluate_with_ablation() function
└── __main__.py        # Update: add 'ablate' and 'evaluate' to PARSER_DATA
```

### Pattern 1: Ablation Checkpoint Workflow (TRAIN-04)

**What:** Load checkpoint into GPT model, build HarmfulParamRegistry, call `parameters_by_type('theta_harmful')` to get the list, zero each tensor in-place with `param.data.zero_()`, save to ablated/ sibling directory.

**Key insight:** The registry must be built from the raw (non-Fabric-wrapped) model, exactly as in pretrain.py. Since ablate.py operates on a saved `lit_model.pth` (not a live training model), there is no DDP/FSDP wrapping — build the model directly with `GPT(config)`, then `fabric.load(ckpt_path, {"model": model})`, then build registry from the raw model (no unwrapping needed).

**Checkpoint loading pattern (from pretrain.py):**
```python
# Source: safemoe/pretrain.py line 419
fabric.load_raw(initial_checkpoint_dir / "lit_model.pth", model)
# For inference-only loads (no optimizer state):
fabric.load(checkpoint_path, {"model": model})
```

**Registry construction for non-wrapped model:**
```python
# Source: safemoe/masking.py HarmfulParamRegistry.__init__
# For ablate.py: model is NOT wrapped by fabric, so no DDP unwrapping needed.
# Just build directly:
registry = HarmfulParamRegistry(model, config)
harmful_params = registry.parameters_by_type('theta_harmful')
```

**In-place zeroing:**
```python
# param.data.zero_() zeroes the tensor in-place without touching grad
for param in harmful_params:
    pre_norm = param.data.norm().item()  # for manifest
    param.data.zero_()
```

**Ablation manifest (JSON):**
```python
import json
manifest = {"zeroed_parameters": [{"name": name, "pre_ablation_norm": norm} for ...]}
(ablated_dir / "ablation_manifest.json").write_text(json.dumps(manifest, indent=2))
```

### Pattern 2: Perplexity Evaluation CLI (EVAL-01)

**What:** Load original checkpoint, build MultiDataLoader, call `validate()` per split, convert loss to perplexity. If `--ablated` also provided, repeat for ablated model and print comparison table.

**validate() reuse (from pretrain.py lines 707-729):**
```python
# Source: safemoe/pretrain.py validate()
# max_iters parameter: pass a very large int (e.g. sys.maxsize) for "all tokens"
@torch.no_grad()
def validate(fabric, model, val_dataloader, max_iters, verbose=True) -> torch.Tensor:
    ...
    return val_loss  # scalar tensor; perplexity = math.exp(val_loss.item())
```

**Val dataloader access (from datamodule.py lines 74-98):**
```python
# Source: safemoe/data/datamodule.py val_dataloaders()
# Returns {"D_std": DataLoader, "D_harmful": DataLoader}
# No D_unlabeled val set exists — by design
val_loaders = data.val_dataloaders()
```

**Config loading from checkpoint dir (LitGPT convention):**
```python
# model_config.yaml is saved by save_config() in save_checkpoint()
from litgpt.utils import save_config  # already saves; load via:
from litgpt.config import Config
config = SafeMoEConfig.from_file(ckpt_dir / "model_config.yaml")
# or: model_config = yaml.safe_load((ckpt_dir / "model_config.yaml").read_text())
```

**results.json output:**
```python
results = {
    "original": {"val_ppl_D_std": ..., "val_ppl_D_harmful": ...},
    "ablated":  {"val_ppl_D_std": ..., "val_ppl_D_harmful": ...},
    "delta":    {"val_ppl_D_std": ..., "val_ppl_D_harmful": ...},
}
(out_dir / "results.json").write_text(json.dumps(results, indent=2))
```

### Pattern 3: Routing Attribution Hooks (EVAL-02)

**What:** Attach a `register_forward_hook` to each `SafeMoELayer` in the model. The hook captures the `indices` tensor (router top-k selections) from the forward pass and accumulates counts per expert per split.

**Critical detail on where to hook:** `SafeMoELayer.forward()` does NOT expose the `indices` tensor through the standard module output. The hook approach requires accessing local variables from the forward pass, which `register_forward_hook` cannot do — it only sees the module's inputs and outputs. The correct approach is to attach the hook to intercept at the module level and use a mechanism to expose routing data.

**Two viable implementations:**

Option A — attribute side-channel (simpler, no forward modification needed):
```python
# In the analysis pass, install a hook that records routing decisions
# by reading SafeMoELayer._last_routing_indices set during forward
# But SafeMoELayer.forward() doesn't currently set this attribute.
# Would require a small extension to SafeMoELayer (or a subclass/wrapper).
```

Option B — register_forward_hook on the SafeMoELayer with a modified forward to expose dispatch:
Since the hook sees `(module, input, output)` but `indices` is a local variable inside `forward()`, a clean approach without modifying SafeMoELayer is to **replace** SafeMoELayer.forward with a thin wrapper that stores the routing indices as a module attribute during the analysis pass, then remove it after.

**Recommended: attribute side-channel via a context manager hook:**
```python
dispatch_counts: dict[str, list[int]] = {"D_std": [], "D_harmful": [], "D_unlabeled": []}

def make_routing_hook(split_name: str):
    def hook(module, inp, out):
        # module._routing_indices must be set inside forward() during hook pass
        if hasattr(module, '_routing_indices'):
            expert_ids = module._routing_indices.flatten().tolist()
            dispatch_counts[split_name].extend(expert_ids)
    return hook

handles = [layer.register_forward_hook(make_routing_hook(split)) for layer in safemoe_layers]
# ... run forward passes per split ...
for h in handles: h.remove()  # clean up
```

**Simpler alternative — monkey-patch during analysis pass only:** Temporarily override `SafeMoELayer.forward` to append routing indices to a list, run the eval pass, then restore. This avoids touching SafeMoELayer permanently.

**Most robust approach (decided by CONTEXT.md):** The CONTEXT.md specifies "forward hooks on SafeMoELayer during eval inference, accumulate expert dispatch counts." This requires SafeMoELayer to expose routing indices via a hook-accessible mechanism. The cleanest implementation:

1. During the analysis pass only, install a `register_forward_pre_hook` that sets a `_collect_routing` flag on each SafeMoELayer.
2. Modify SafeMoELayer.forward to store `self._last_indices = indices` when `_collect_routing` is True (one line added to the existing forward body after the topk call).
3. The `register_forward_hook` then reads `module._last_indices`.

**Since SafeMoELayer is project code**, option 2 (store `_last_indices` when a flag is set) is cleanest and avoids any persistent overhead. Alternatively: store always (one attribute write per forward, negligible cost), remove hook after analysis pass.

**TensorBoard logging for routing histograms:**
```python
# Source: safemoe/pretrain.py line 679
fabric.log_dict(metrics, step=iter_num)
# For histograms, use the underlying TensorBoard writer directly:
tb_logger = fabric.loggers[0]  # TensorBoardLogger
tb_logger.experiment.add_histogram("routing/D_std", harmful_fracs_D_std, global_step=0)
```

### Pattern 4: In-Training Ablation Evaluation (EVAL-03)

**What:** At each `save_interval` step in `pretrain.py fit()`, temporarily zero theta_harmful params, run `validate()` on each split, restore original weights, log ablated metrics.

**Clone + zero + restore pattern:**
```python
def evaluate_with_ablation(fabric, model, registry, val_loaders, iter_num):
    """Clone theta_harmful params, zero in-place, validate all splits, restore."""
    harmful_params = registry.parameters_by_type('theta_harmful')
    # Clone only the harmful params — cheap (small subset)
    saved = [p.data.clone() for p in harmful_params]
    try:
        for p in harmful_params:
            p.data.zero_()
        model.eval()
        metrics = {}
        for split_name, loader in val_loaders.items():
            val_loss = validate(fabric, model, loader, max_iters=sys.maxsize, verbose=False)
            ppl = math.exp(val_loss.item())
            metrics[f"ablated_val_ppl_{split_name}"] = ppl
    finally:
        # Always restore — even if validate() raises
        for p, saved_data in zip(harmful_params, saved):
            p.data.copy_(saved_data)
        model.train()
    fabric.log_dict(metrics, step=iter_num)
```

**Call site in fit() (alongside save_checkpoint):**
```python
# Source: safemoe/pretrain.py line 692-693 (save_interval block)
if train.save_interval is not None and state["step_count"] % train.save_interval == 0:
    save_checkpoint(...)
    evaluate_with_ablation(fabric, model, registry, val_loaders, state["iter_num"])
```

**Critical: registry must be passed to fit()** — registry is currently constructed in `main()` and not passed to `fit()`. EVAL-03 requires adding `registry` as a parameter to `fit()`.

### Anti-Patterns to Avoid

- **Deep-copying the full model for EVAL-03:** Doubles GPU memory for the sake of ablation eval. Clone only `theta_harmful` params instead (a small fraction of total params).
- **Calling `parameters_by_type()` on the Fabric-wrapped model in ablate.py:** The ablate CLI loads a saved checkpoint into a fresh, non-wrapped model. There is no DDP/FSDP wrapping. Do not add the DDP-unwrapping logic from `main()` — it is not needed.
- **Running all validation tokens during EVAL-03:** This would add unbounded latency to the training loop at each save_interval. For EVAL-03 inside the training loop, pass `eval.max_iters` (not `sys.maxsize`) to `validate()`. The EVAL-01 standalone CLI uses all tokens.
- **Leaving the model in eval mode after validate():** `validate()` calls `model.eval()` but returns without calling `model.train()`. Callers must restore training mode. The `evaluate_with_ablation()` function must call `model.train()` in its finally block.
- **Storing routing fraction state in SafeMoELayer permanently:** The `_last_indices` attribute (or whatever side-channel is chosen) should only be populated during the analysis pass, not during training, to avoid unnecessary memory writes.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Perplexity computation | Custom eval loop | `pretrain.validate()` + `math.exp()` | validate() handles model.eval(), barrier(), no_grad, loss accumulation; all tested and working |
| Parameter enumeration for ablation | Custom regex scan | `HarmfulParamRegistry.parameters_by_type('theta_harmful')` | Registry already handles DDP/FSDP prefix stripping; tested in Phase 2/3 |
| Checkpoint loading | `torch.load()` with manual map_location | `fabric.load(ckpt_path, {"model": model})` | Fabric handles device/precision placement; consistent with litgpt convention |
| CLI argument parsing | argparse | `jsonargparse.CLI(PARSER_DATA)` | Established pattern in `__main__.py`; supports YAML config files automatically |
| Val data loading | Custom StreamingDataset construction | `MultiDataLoader.val_dataloaders()` | Already returns `{D_std: DataLoader, D_harmful: DataLoader}`; handles block_size and num_workers |
| TensorBoard histogram writing | Custom logger | `tb_logger.experiment.add_histogram()` | TensorBoard SummaryWriter API; available via `fabric.loggers[0].experiment` |

**Key insight:** Every non-trivial piece of this phase exists in the codebase already. Phase 4 is integration work, not construction work.

---

## Common Pitfalls

### Pitfall 1: validate() leaves model in eval mode

**What goes wrong:** `validate()` calls `model.eval()` but does not call `model.train()` afterward. If EVAL-03's `evaluate_with_ablation()` calls validate() and then exits without calling `model.train()`, the training loop will continue running with the model in eval mode — this silently disables Dropout and affects BatchNorm, producing incorrect training behavior.

**Why it happens:** `validate()` was designed for use after the training loop has finished or as a one-shot call; it leaves mode restoration to the caller.

**How to avoid:** Always call `model.train()` in the `finally` block of `evaluate_with_ablation()`. Never rely on validate() to restore training mode.

**Warning signs:** Training loss becomes unusually stable or low after first ablation evaluation; Dropout behavior changes.

### Pitfall 2: HarmfulParamRegistry on the wrong model reference in ablate.py

**What goes wrong:** In pretrain.py, registry is built from `model._forward_module` (or `model._forward_module.module` for DDP) because fabric.setup() wraps the model. In ablate.py, the model is loaded fresh without fabric.setup(), so named_parameters() returns clean names without any `_forward_module.` or `module.` prefix. Building the registry on `fabric_model` (a _FabricModule) instead of the raw model will cause parameter name mismatches.

**Why it happens:** Muscle memory from pretrain.py; the DDP-unwrapping pattern looks like it should always be applied.

**How to avoid:** In ablate.py, construct the model with `GPT(config)`, load weights with `fabric.load()` or `torch.load()`, then pass the raw model (not a Fabric-wrapped model) to `HarmfulParamRegistry`. For a standalone CLI that doesn't use fabric.setup(model), no unwrapping is needed.

**Warning signs:** `HarmfulParamRegistry.__init__` raises `ValueError: not all parameters classified` or `theta_harmful and theta_std overlap`.

### Pitfall 3: val_dataloaders() vs val_dataloader() API confusion

**What goes wrong:** `MultiDataLoader.val_dataloader()` (LightningDataModule compat) returns a list of DataLoaders. `MultiDataLoader.val_dataloaders()` returns a dict `{"D_std": DataLoader, "D_harmful": DataLoader}`. The evaluate CLI needs per-split access, so it must use `val_dataloaders()` (the dict-returning method), not `val_dataloader()` (the list).

**Why it happens:** Both methods exist; their names differ by only an "s".

**How to avoid:** Always use `data.val_dataloaders()` (plural, returns dict) in evaluate.py. The dict keys are `"D_std"` and `"D_harmful"` — there is no D_unlabeled val set.

**Warning signs:** KeyError on dict access, or treating a list as a dict.

### Pitfall 4: Registry must be passed to fit() for EVAL-03

**What goes wrong:** Currently `fit()` in pretrain.py does not accept a `registry` parameter. EVAL-03 requires calling `evaluate_with_ablation(registry=registry, ...)` inside `fit()`. If registry is not added to `fit()`'s signature and passed through from `main()`, it will not be accessible at the call site.

**Why it happens:** Registry was constructed in `main()` and only used to set up the dual optimizers; it was not needed by `fit()` before Phase 4.

**How to avoid:** Add `registry: HarmfulParamRegistry` as a parameter to `fit()`. Pass it from `main()`. Also need to pass `val_loaders` (the per-split val DataLoaders from `data.val_dataloaders()`) — currently `fit()` only receives the single `val_dataloader` (D_std).

**Warning signs:** NameError for `registry` inside fit(); AttributeError accessing registry from state dict.

### Pitfall 5: In-place zero affects autograd graph if called during training forward

**What goes wrong:** `param.data.zero_()` modifies the tensor storage in-place. If this is called while a gradient tape is active (during a micro-batch forward), it will corrupt the autograd graph and raise `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`.

**Why it happens:** EVAL-03 runs between optimizer steps, but if the timing is wrong (e.g., inside the accumulation loop), the forward pass is still active.

**How to avoid:** Call `evaluate_with_ablation()` only in the save_interval block, outside the micro-batch accumulation loop, after `optimizer.step()` and `zero_grad()` have completed. This is the natural placement — alongside `save_checkpoint()`.

**Warning signs:** RuntimeError about inplace operations; NaN loss after ablation eval step.

### Pitfall 6: routing hook captures wrong expert dispatch (EVAL-02)

**What goes wrong:** SafeMoELayer.forward() skips harmful experts when `_activation_masking_enabled=True` (the ActivationMasker path from training). If routing attribution is run with the ActivationMasker accidentally enabled, harmful experts will never appear in dispatch counts, making D_harmful look identical to D_std.

**Why it happens:** ActivationMasker state might not be explicitly disabled for the analysis pass.

**How to avoid:** Ensure `activation_masker.disable()` (or equivalent) is called before the routing analysis pass, and that the model is in eval mode (not training mode) with no masking active. The evaluate CLI constructs a fresh model without any ActivationMasker, so this is naturally safe — the risk is only if routing attribution is added to a code path that shares state with the maskers.

---

## Code Examples

Verified patterns from existing source code:

### Checkpoint load into fresh model (for ablate.py and evaluate.py)
```python
# Source: safemoe/pretrain.py line 419 and established litgpt convention
# fabric.load_raw for model-only loads; fabric.load for state dict loads
fabric = L.Fabric(devices=1, accelerator="cpu")
fabric.launch()
with fabric.init_module(empty_init=True):
    model = GPT(config)
fabric.load(ckpt_dir / "lit_model.pth", {"model": model})
```

### Building registry on a non-DDP model (ablate.py context)
```python
# Source: safemoe/masking.py HarmfulParamRegistry.__init__
# Model is a plain litgpt.GPT — no wrapping, no prefix stripping needed
registry = HarmfulParamRegistry(model, config)
harmful_params = registry.parameters_by_type('theta_harmful')
# harmful_params is a list of nn.Parameter — zero each with .data.zero_()
```

### Per-split val loader iteration (evaluate.py context)
```python
# Source: safemoe/data/datamodule.py val_dataloaders()
data = MultiDataLoader(cache_dir=Path("data/.cache"), x=config_x, y=config_y)
data.connect(tokenizer=tokenizer, batch_size=micro_batch_size, max_seq_length=block_size)
data.setup()
val_loaders = data.val_dataloaders()  # {"D_std": DataLoader, "D_harmful": DataLoader}
for split_name, loader in val_loaders.items():
    val_loss = validate(fabric, model, fabric.setup_dataloaders(loader), max_iters=sys.maxsize)
    ppl = math.exp(val_loss.item())
```

### Adding subcommand to __main__.py (established pattern)
```python
# Source: safemoe/__main__.py — current state
from jsonargparse import CLI
from safemoe.pretrain import setup as pretrain_fn
PARSER_DATA = {"pretrain": pretrain_fn}

# Phase 4 extension:
from safemoe.ablate import setup as ablate_fn
from safemoe.evaluate import setup as evaluate_fn
PARSER_DATA = {
    "pretrain": pretrain_fn,
    "ablate": ablate_fn,
    "evaluate": evaluate_fn,
}
```

### TensorBoard metric logging (existing pattern)
```python
# Source: safemoe/pretrain.py line 679
fabric.log_dict(metrics, step=state["iter_num"] - 1)
# metrics is a flat dict: {"ablated_val_ppl_D_std": 45.2, "ablated_val_ppl_D_harmful": 312.7, ...}
```

### In-place zero + restore with try/finally (EVAL-03)
```python
# Pattern derived from ActivationMasker try/finally in pretrain.py lines 587-611
harmful_params = registry.parameters_by_type('theta_harmful')
saved = [p.data.clone() for p in harmful_params]
try:
    for p in harmful_params:
        p.data.zero_()
    # ... run validation ...
finally:
    for p, saved_data in zip(harmful_params, saved):
        p.data.copy_(saved_data)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| validate() with max_iters cap (training-time eval) | No max_iters cap for standalone evaluate CLI | Phase 4 decision | Exact perplexity on full val set; training-time eval still uses max_iters cap |
| D_std val only during training loop | Per-split val (D_std + D_harmful) for evaluation | Phase 4 EVAL-01 | Enables pre/post ablation comparison per split |

---

## Open Questions

1. **How to expose router dispatch indices to forward hooks without modifying SafeMoELayer permanently**
   - What we know: `SafeMoELayer.forward()` has `indices` as a local variable (computed from `torch.topk` on gate output); `register_forward_hook` sees `(module, input, output)` but not local variables.
   - What's unclear: Whether to add `self._last_indices = indices` to SafeMoELayer.forward permanently (one attribute write per forward, always present), or only when a flag is set (more complex but zero-overhead).
   - Recommendation: Add `self._last_indices = indices` unconditionally to SafeMoELayer.forward — one attribute write per forward is negligible. The routing hook reads it; without a hook, the attribute is simply unused. This is the simplest correct solution.

2. **Fabric setup for evaluate CLI — single CPU device vs GPU**
   - What we know: evaluate.py will run post-training on the same machine that produced the checkpoint; GPU is available (4×H200).
   - What's unclear: Whether to use `devices="auto"` (may try multi-GPU) or `devices=1` (deterministic, simpler for an eval CLI).
   - Recommendation: Use `devices=1, accelerator="auto"` for the evaluate CLI — evaluation does not need multi-GPU, and single-device eval is simpler and reproducible.

3. **val_dataloaders() does not include D_unlabeled val split**
   - What we know: `MultiDataLoader.val_dataloaders()` returns `{"D_std": DataLoader, "D_harmful": DataLoader}` — D_unlabeled has no dedicated val split.
   - What's unclear: Whether EVAL-01/EVAL-03 should report D_unlabeled metrics.
   - Recommendation: The metric naming convention in CONTEXT.md specifies `val_ppl_D_unlabeled` and `ablated_val_ppl_D_unlabeled`. Since no D_unlabeled val DataLoader exists, these metrics simply cannot be computed from MultiDataLoader.val_dataloaders(). The planner should either (a) skip D_unlabeled metrics for EVAL-01/EVAL-03, or (b) add a D_unlabeled val loader to MultiDataLoader. This is a question the user should resolve before planning.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (pyproject.toml `[tool.pytest.ini_options]`) |
| Config file | pyproject.toml |
| Quick run command | `pytest tests/safemoe/ -x -q` |
| Full suite command | `pytest tests/safemoe/ -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TRAIN-04 | ablate() zeroes theta_harmful weights and saves checkpoint; loading ablated checkpoint has exactly zero harmful-param weights | unit | `pytest tests/safemoe/test_ablate.py -x -q` | Wave 0 |
| EVAL-01 | evaluate CLI loads checkpoint, runs validate() per split, returns perplexity; ablated model shows higher D_harmful ppl | integration | `pytest tests/safemoe/test_evaluate.py::test_evaluate_perplexity -x -q` | Wave 0 |
| EVAL-02 | routing hooks accumulate dispatch counts; D_harmful split activates theta_harmful experts at higher fraction than D_std | unit | `pytest tests/safemoe/test_evaluate.py::test_routing_attribution -x -q` | Wave 0 |
| EVAL-03 | evaluate_with_ablation() in fit(): model restored to original weights after ablation eval; ablated metrics logged to fabric | unit | `pytest tests/safemoe/test_pretrain.py::test_evaluate_with_ablation -x -q` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/safemoe/ -x -q`
- **Per wave merge:** `pytest tests/safemoe/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/safemoe/test_ablate.py` — covers TRAIN-04
- [ ] `tests/safemoe/test_evaluate.py` — covers EVAL-01, EVAL-02
- [ ] `tests/safemoe/test_pretrain.py::test_evaluate_with_ablation` — covers EVAL-03 (new test in existing file)

*(No new conftest.py or framework install needed — existing infrastructure covers all needs)*

---

## Sources

### Primary (HIGH confidence)
- `safemoe/pretrain.py` — validate(), fit(), save_checkpoint(), SafeCausalSelfAttention, full training loop
- `safemoe/masking.py` — HarmfulParamRegistry, parameters_by_type(), ActivationMasker, GradientMasker
- `safemoe/model.py` — SafeMoELayer.forward(), expert dispatch loop with indices tensor
- `safemoe/__main__.py` — PARSER_DATA CLI dispatch pattern
- `safemoe/data/datamodule.py` — MultiDataLoader.val_dataloaders(), get_loader()
- `safemoe/config.py` — SafeMoEConfig fields
- `.planning/phases/04-ablation-evaluation/04-CONTEXT.md` — All implementation decisions

### Secondary (MEDIUM confidence)
- `tests/safemoe/test_pretrain.py` — _MockMultiDataLoader, _SynthDataset, _setup_fit_test patterns for test scaffolding in Phase 4 tests
- `.planning/codebase/TESTING.md` — Test conventions (pytest, tmp_path, mock patterns)
- `.planning/codebase/CONVENTIONS.md` — Naming, docstring, import ordering

### Tertiary (LOW confidence)
- None

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in the project; no new dependencies
- Architecture: HIGH — all patterns read directly from existing source files; no speculation
- Pitfalls: HIGH — derived from concrete code paths and established Phase 2/3 lessons in STATE.md
- Open Questions: MEDIUM — Question 3 (D_unlabeled val) is a genuine gap requiring user input

**Research date:** 2026-03-16
**Valid until:** 2026-04-16 (stable internal codebase; no fast-moving external dependencies)
