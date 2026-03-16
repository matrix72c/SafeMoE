# Phase 3: SGTM Training Loop - Context

**Gathered:** 2026-03-16
**Status:** Ready for planning

<domain>
## Phase Boundary

Fork `litgpt/pretrain.py` into `safemoe/pretrain.py` implementing the SGTM 3-path branching per step label: D_harmful → GradientMasker (post-backward), D_std → ActivationMasker (forward), D_unlabeled → standard forward+backward. Includes dual AdamW optimizers (θ_harmful and θ_std), CLI entry point `python -m safemoe pretrain`, and YAML config. Ablation utility and evaluation (TRAIN-04, EVAL-*) are Phase 4.

</domain>

<decisions>
## Implementation Decisions

### Attention head masking scope
- Phase 3 extends BOTH GradientMasker and ActivationMasker to cover `harmful_attn_heads` (not just MoE experts)
- **GradientMasker**: post-backward zeroing of QKV-row gradients in `attn.attn.weight` — same explicit post-backward approach as expert masking; no hooks
- **ActivationMasker**: zeros harmful head output contributions before `attn.proj` during D_std forward (zero the head-size rows in the attention output tensor for each designated harmful head)
- `harmful_attn_heads: [0, 1]` set in `safemoe-tinystories.yaml` experiment config; Phase 3 unit tests exercise this masking path explicitly

### Default sampling weights
- `upsample_std`, `upsample_harmful`, `upsample_unlabeled` are **required** YAML fields — no opinionated defaults
- Training fails loudly with a clear error if any are missing (no silent fallback to 1:1:1)
- Fields live **flat at the top-level config** (not nested under a sub-section)
- `safemoe-tinystories.yaml` uses `upsample_std: 1`, `upsample_harmful: 1`, `upsample_unlabeled: 1` (uniform baseline for initial experiments)

### Gradient accumulation + SGTM interaction
- **One split label per optimizer step**: all micro-batches in an accumulation window use the same split — determined at the start of the optimizer step, before any micro-batch forwards
- **Masker called once per optimizer step** (not per micro-batch):
  - D_std: `activation_masker.enable()` before first micro-batch forward → accumulate → `activation_masker.disable()` after last micro-batch forward
  - D_harmful: accumulate all micro-batch backwards → `gradient_masker.mask()` once after final backward, before `optimizer.step()`
  - D_unlabeled: no masking — standard accumulation
- DDP sync: `fabric.no_backward_sync(model, enabled=is_accumulating)` — identical to litgpt's pattern
- `safemoe-tinystories.yaml`: `micro_batch_size: 4`, `gradient_accumulation_iters: 4` (effective batch size = 16)

### Dual optimizer LR schedule
- **Shared LR schedule**: both θ_harmful and θ_std AdamW instances use the same LR, warmup steps, min_lr, and weight decay from `TrainArgs`
- **Per-split selective stepping**:
  - D_harmful step: only θ_harmful optimizer steps; θ_std calls `zero_grad(set_to_none=True)` only
  - D_std step: only θ_std optimizer steps; θ_harmful calls `zero_grad(set_to_none=True)` only
  - D_unlabeled step: BOTH optimizers step
- **Gradient clipping**: two separate `fabric.clip_gradients()` calls — one per optimizer, applied only before that optimizer's `step()` when it is active for the current split
- **LR counter**: advances once per optimizer step regardless of which optimizer(s) stepped; both optimizers see the same LR curve over time

### Claude's Discretion
- Internal loop structure for the 3-path branching (if/elif/else vs. dispatch dict)
- How the split label is stored in the training `state` dict for checkpoint resumability
- Logging format for per-split loss tracking during training
- Whether two separate `fabric.setup_optimizers()` calls are needed or if both can be passed together

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `litgpt/pretrain.py`: Direct fork target. Key patterns to preserve: `fabric.launch()` + `setup()` → `main()` → `fit()` + `validate()` structure; `CycleIterator` for infinite data iteration; `get_lr()` cosine schedule with warmup; `save_checkpoint()` and resume logic; `ThroughputMonitor` for performance metrics.
- `litgpt/args.py TrainArgs`: Reuse `TrainArgs` (save_interval, log_interval, global_batch_size, micro_batch_size, max_tokens, max_norm, min_lr, lr_warmup_steps, weight_decay). Add `upsample_std`, `upsample_harmful`, `upsample_unlabeled` as required fields via a new `SGTMArgs` dataclass or extend `TrainArgs`.
- `litgpt/args.py EvalArgs`: Reuse as-is for eval interval and max_iters.
- `litgpt/utils.py instantiate_torch_optimizer`: Creates AdamW from name+config dict. Use twice — once per optimizer.
- `safemoe/masking.py GradientMasker + ActivationMasker`: Phase 2 primitives. Phase 3 extends these (attn head masking) and calls them from the training loop.
- `safemoe/model.py SafeMoELayer`: Has `_activation_masking_enabled` flag checked in forward. Phase 3 also needs `CausalSelfAttention` to participate in activation masking.
- `safemoe/configs/safemoe-tinystories.yaml`: Already exists from Phase 2. Phase 3 adds `upsample_std/harmful/unlabeled`, `micro_batch_size`, `gradient_accumulation_iters`, and updates `harmful_attn_heads: [0, 1]`.

### Established Patterns
- Training loop: `for train_data in train_iterator` → `is_accumulating = iter_num % accum_iters != 0` → `fabric.no_backward_sync(model, enabled=is_accumulating)` → `fabric.backward(loss / accum_iters)` → clip + step at accumulation boundary
- Checkpointing: `state = {"model": model, "optimizer": optimizer, "iter_num": 0, "step_count": 0}`; `fabric.load(resume, state)` for resuming; `save_checkpoint(fabric, state, ...)` at intervals. With dual optimizers, state becomes `{"model": ..., "optimizer_harmful": ..., "optimizer_std": ..., "iter_num": 0, "step_count": 0}`.
- LR schedule: `get_lr(base_lr, iter_num, warmup_iters, max_iters, min_lr)` cosine decay — reuse as-is, apply to both optimizers with same iter_num.
- Config CLI: `litgpt/parser_config.py save_hyperparameters` + jsonargparse — fork and adapt for `safemoe pretrain`.

### Integration Points
- `MultiDataLoader.get_loader(split_label)` (Phase 1) provides the DataLoader for each step's split
- `HarmfulParamRegistry(model, config)` (Phase 2) exposes `parameters_by_type('theta_harmful'/'theta_std')` for building the two AdamW param groups
- `GradientMasker(registry)` and `ActivationMasker(model)` (Phase 2) are instantiated at setup, called per optimizer step in the training loop
- `SafeMoEConfig` (Phase 2) is the model config class — `safemoe pretrain` uses this instead of `litgpt.Config`
- Phase 4 evaluation reads checkpoints produced by `safemoe/pretrain.py` — checkpoint format must match `litgpt.utils.save_checkpoint` conventions

</code_context>

<specifics>
## Specific Ideas

- No specific references beyond the paper's SGTM algorithm and litgpt's pretrain.py as the fork baseline
- The "one split label per optimizer step" rule is a deliberate semantic correctness choice — not just a simplification. Mixing split labels within an accumulation window would corrupt gradient isolation guarantees.
- Per-split selective stepping (only relevant optimizer steps for D_harmful / D_std) prevents Adam momentum from accumulating on zero gradients even with `set_to_none=True` as a belt-and-suspenders approach

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 03-sgtm-training-loop*
*Context gathered: 2026-03-16*
