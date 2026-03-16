# Phase 4: Ablation & Evaluation - Context

**Gathered:** 2026-03-16
**Status:** Ready for planning

<domain>
## Phase Boundary

A complete evaluation pipeline that ablates θ_harmful experts and measures whether knowledge isolation succeeded — per-split perplexity (pre/post ablation), routing attribution histograms, and mid-training isolation curve. This is the validation of SafeMoE's core thesis: harmful knowledge isolated into θ_harmful experts can be surgically removed without degrading general capability.

Requirements: TRAIN-04, EVAL-01, EVAL-02, EVAL-03

</domain>

<decisions>
## Implementation Decisions

### Ablation utility (TRAIN-04)
- Standalone CLI: `python -m safemoe ablate <ckpt_dir>` — consistent with `python -m safemoe pretrain` pattern
- Saves ablated checkpoint to `<ckpt_dir>/ablated/lit_model.pth` — sibling directory alongside the original
- Zeroes all θ_harmful weights to 0.0 in-place using HarmfulParamRegistry to identify them (exactly as the paper describes)
- Operates on full-precision consolidated `lit_model.pth` only — no FSDP sharded checkpoint handling
- Reports verbose output: prints summary table (parameters zeroed, expert indices, total norm before/after) AND saves `<ckpt_dir>/ablated/ablation_manifest.json` listing every zeroed parameter name with its pre-ablation norm

### Perplexity evaluation entry point (EVAL-01)
- Standalone CLI: `python -m safemoe evaluate --original <ckpt_dir> --ablated <ckpt_dir>/ablated`
- Accepts two checkpoint paths for pre/post comparison (both paths required for comparison table; single path works for non-comparison eval)
- Loads model config and tokenizer from checkpoint directory (no separate --config needed — matches LitGPT checkpoint convention)
- Requires data to be pre-prepared on disk (data/.cache/ must exist — consistent with pretrain.py)
- Runs on **all available validation tokens** (no max_iters cap for the evaluate CLI — exact perplexity)
- Reports: prints comparison table (pre/post ppl per split) AND writes `results.json` to output directory
- Data source: same `data/.cache/` path as training; x/y params read from checkpoint's saved config

### Routing attribution (EVAL-02)
- Collected during a **post-training analysis pass** — no overhead during training
- Triggered via `--routing` flag on the evaluate CLI: `python -m safemoe evaluate --original <ckpt_dir> --routing`
- Captures θ_harmful activation **fraction per split** (fraction of tokens routing to θ_harmful experts, per D_std / D_harmful / D_unlabeled split) — sufficient to validate thesis
- Implementation: forward hooks on SafeMoELayer during eval inference, accumulate expert dispatch counts
- Output: TensorBoard histograms (logged to checkpoint's existing runs/ directory) AND `routing_attribution.json` with raw per-split fractions

### Mid-training ablation evaluation (EVAL-03)
- Runs only at `save_interval` checkpoints (not every eval.interval) — minimal training overhead
- Implementation: in-place zero + restore — clone θ_harmful weights before zeroing, run validation pass on ablated model, then restore originals from clone; no separate checkpoint file, no model copy
- Logged to TensorBoard alongside regular val loss curves in the same run — metric names: `ablated_val_ppl_D_std`, `ablated_val_ppl_D_harmful`, `ablated_val_ppl_D_unlabeled`
- Implemented inside `pretrain.py` at the checkpoint-save code path (alongside `save_checkpoint()`)

### Metric naming convention
- Flat prefix scheme throughout TensorBoard and JSON:
  - Regular: `val_ppl_D_std`, `val_ppl_D_harmful`, `val_ppl_D_unlabeled`
  - Ablated: `ablated_val_ppl_D_std`, `ablated_val_ppl_D_harmful`, `ablated_val_ppl_D_unlabeled`
  - Routing: `routing_harmful_frac_D_std`, `routing_harmful_frac_D_harmful`, `routing_harmful_frac_D_unlabeled`
  - Consistent with existing `loss_D_std` / `loss_D_harmful` naming in pretrain.py

### Eval portability
- Same machine as training only — data must be on disk at data/.cache/
- No portability shims for Milestone 1 (research scale, single machine)

</decisions>

<specifics>
## Specific Ideas

- Ablation manifest JSON is for verifiability — researcher should be able to confirm exactly which weights were zeroed and what their pre-ablation magnitudes were
- The in-place zero + restore pattern for mid-training eval avoids 2× memory cost of deep-copy; cloning only θ_harmful weights (a small subset of total params) is cheap
- The evaluate CLI's comparison table output should mirror how the paper presents results: split rows, pre/post columns, with a "delta" column showing ppl increase for D_harmful and stability for D_std

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `safemoe/pretrain.py validate()`: Direct reuse for perplexity evaluation — takes fabric + model + val_dataloader + eval_args, returns val_loss. evaluate CLI wraps this per split, converting loss to perplexity (exp(loss)).
- `safemoe/masking.py HarmfulParamRegistry`: Used by ablate CLI to enumerate θ_harmful parameters via `parameters_by_type('theta_harmful')`. Already tested and proven in Phase 2/3.
- `MultiDataLoader.get_loader(split_name)` (Phase 1): Provides per-split DataLoaders for evaluation. evaluate CLI calls `get_loader('D_std')`, `get_loader('D_harmful')`, `get_loader('D_unlabeled')` for val splits via `val_dataloaders()`.
- `litgpt/utils.py save_config`, `CycleIterator`, `chunked_cross_entropy`: All reusable in evaluate CLI.
- `litgpt/utils.py extend_checkpoint_dir`: Handles checkpoint path resolution — reuse in ablate and evaluate CLIs.
- `safemoe/configs/safemoe-tinystories.yaml`: Config template with all required fields — evaluate CLI reads this from checkpoint's saved copy.

### Established Patterns
- CLI entry point: Add `ablate` and `evaluate` subcommands to `safemoe/__main__.py` (same as `pretrain`) — `python -m safemoe ablate ...` / `python -m safemoe evaluate ...`
- Checkpoint loading: `fabric.load(checkpoint_path, {"model": model})` — load into pre-constructed model, consistent with litgpt resume pattern
- Logging: `fabric.log_dict(metrics, step=iter_num)` — reuse for mid-training ablation metrics
- Model config: `SafeMoEConfig.from_name(model_name)` or load from `model_config.yaml` in checkpoint dir

### Integration Points
- `safemoe/__main__.py`: Add `ablate` and `evaluate` to the subcommand dispatch
- `safemoe/pretrain.py`: Add `evaluate_with_ablation()` function called at save_interval in the training loop
- `safemoe/masking.py HarmfulParamRegistry`: Ablate CLI uses `parameters_by_type('theta_harmful')` to zero weights
- `safemoe/model.py SafeMoELayer`: Routing attribution hooks attach to `forward()` here
- `data/.cache/`: evaluate CLI reads pre-prepared splits (same path as training)
- TensorBoard logs at `out/<run>/logs/`: ablated metrics appear alongside regular val curves

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 04-ablation-evaluation*
*Context gathered: 2026-03-16*
