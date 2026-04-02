# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SafeMoE is a research fork of LitGPT focused on isolating harmful knowledge inside designated MoE experts and optional attention heads so those parameters can later be ablated with minimal loss of general capability.

The repository has two layers:
- `litgpt/`: upstream-style LitGPT runtime for model definitions, CLI workflows, finetuning, generation, evaluation, and deployment.
- `litgpt/safemoe/`: SafeMoE-specific training, surgery, masking, ablation, evaluation, observability, and dataset preparation built on top of LitGPT.

Current research flow:
1. Start from a pretrained MoE checkpoint.
2. Add harmful-specific experts/heads via surgery.
3. Run either `warmup` routing training or `transfer` SGTM training.
4. Ablate `theta_harmful` and compare original vs ablated checkpoints.

## Common Commands

### Environment and install
- Before running project commands, activate the repository virtual environment:
  - `source .venv/bin/activate`
- Install from source with all common extras:
  - `uv sync --all-extras`
  - or `pip install -e ".[extra,compiler,test]"`
- The main CLI entrypoint is:
  - `litgpt ...`
- SafeMoE-specific CLI entrypoint is:
  - `litgpt safemoe_* ...`

### Tests
- Run the full test suite:
  - `pytest tests`
- Run a single test file:
  - `pytest tests/test_pretrain.py`
- Run a single test:
  - `pytest tests/test_pretrain.py -k test_name`
- Pytest is configured with `--strict-markers` and disables pytest warnings by default.
- `tests/conftest.py` filters certain tests via env vars such as `PL_RUN_STANDALONE_TESTS=1` and `RUN_ONLY_CUDA_TESTS=1`.

### Lint / formatting
- Run Ruff over the repository:
  - `ruff check .`
- Optional auto-fix:
  - `ruff check . --fix`

### Core LitGPT workflows
- List supported downloadable models:
  - `litgpt download list`
- Download a model or tokenizer:
  - `litgpt download <model>`
- Pretrain:
  - `litgpt pretrain <model> ...`
- Finetune:
  - `litgpt finetune <model> ...`
- Chat / generate / evaluate / serve:
  - `litgpt chat <checkpoint-or-model>`
  - `litgpt evaluate <checkpoint-or-model> --tasks 'truthfulqa_mc2,mmlu'`
  - `litgpt serve <checkpoint-or-model>`

### SafeMoE workflows
- SafeMoE pretraining / transfer:
  - `litgpt pretrain_safemoe --config safemoe/configs/safemoe-tinystories.yaml`
- Warmup with auto-surgery from a base checkpoint:
  - `litgpt pretrain_safemoe --config safemoe/configs/safemoe-qwen-warmup-tinystories.yaml`
- Manual surgery:
  - `litgpt safemoe_surgery --help`
- Ablate harmful parameters in a checkpoint:
  - `litgpt safemoe_ablate <ckpt_dir>`
- Evaluate a checkpoint, optionally against an ablated copy:
  - `litgpt safemoe_evaluate <ckpt_dir>`
  - `litgpt safemoe_evaluate <ckpt_dir> --ablated <ckpt_dir>/ablated`

### GPU cluster runtime
- This repository is developed on a private cluster dev machine that shares storage with the GPU cluster.
- To run commands that require GPUs, start a container with `rlaunch` and run the command inside the container. For example:
  - `rlaunch --charged-group=sfaethm_gpu --private-machine=group --gpu 4 --memory 1048576 --cpu 64 --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public --mount=gpfs://gpfs2/wenxiaoyu-gpfs02:/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02 --custom-resources brainpp.cn/fuse=1 -- bash -c "cd /mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe && source .venv/bin/activate && litgpt pretrain_safemoe --config safemoe/configs/safemoe-tinystories.yaml"`
- Adding `-d` runs the container in the background. Avoid `-d` when debugging or running tests through Claude Code: foreground execution is preferred because it keeps the shell interactive.
- Background `rlaunch -d` runs are harder to use for debugging because the container name is randomly assigned and the concrete SSH/attach instructions may not be available afterward.
- Prefer interactive foreground sessions for test runs, debugging, and iterative development; use `-d` only when a detached background container is explicitly desired.

## High-Level Architecture

### 1. LitGPT base runtime
- `litgpt/__main__.py` defines the main `litgpt` CLI using `jsonargparse`, wiring commands like `download`, `pretrain`, `finetune`, `generate`, `evaluate`, and `serve`.
- `litgpt/config.py` is the central model configuration registry. `Config.from_name()` maps model names to architecture definitions, and the config object decides attention, MLP, RoPE, MoE, and normalization behavior.
- `litgpt/model.py` contains the full decoder-only transformer implementation in one file. `GPT` builds blocks from `Config`, and MoE behavior is controlled through config fields such as `mlp_class_name`, `n_expert`, and `n_expert_per_token`.
- `litgpt/pretrain.py` is the baseline Fabric/FSDP pretraining loop that SafeMoE forks and extends.
- `litgpt/data/` provides reusable `DataModule` implementations plus dataset preparation scripts. The base `DataModule` contract is `connect() -> prepare_data() -> setup() -> dataloaders`.
- `litgpt/api.py` exposes the Python `LLM` API for loading checkpoints, generating text, and integrating with non-CLI workflows.

### 2. SafeMoE model integration
- `litgpt/safemoe/config.py` defines `SafeMoEConfig`, a `litgpt.Config` subclass that adds `harmful_expert_indices`, `harmful_attn_heads`, and `num_harmful_experts`.
- The key hook is `SafeMoEConfig.mlp_class`: when `mlp_class_name == "LLaMAMoE"`, LitGPT blocks instantiate `litgpt/safemoe/model.py`'s `SafeMoELayer` instead of plain `LLaMAMoE`.
- `litgpt/safemoe/model.py` implements `SafeMoELayer`, which records routing statistics and can zero harmful expert contributions during forward passes when activation masking is enabled.

### 3. Parameter partitioning and masking
SafeMoE’s core training logic depends on explicit parameter ownership:
- `theta_harmful`: harmful experts and harmful attention head slices.
- `theta_std`: standard experts and standard attention head slices.
- `theta_shared`: embeddings, router/gate weights, norms, lm head, and other shared parameters.

This logic lives in `litgpt/safemoe/masking.py`:
- `HarmfulParamRegistry` scans named parameters and classifies them into `theta_harmful`, `theta_std`, and `theta_shared`.
- Fused `attn.qkv.weight` tensors cannot be split into separate parameters, so the registry stores row-slice metadata for harmful vs standard attention heads while keeping the full parameter classified as standard/shared-compatible for bookkeeping.
- `GradientMasker` clears the disallowed gradients after backward depending on the active split.
- `ActivationMasker` toggles `_activation_masking_enabled` on `SafeMoELayer` instances so harmful experts can be skipped during forward passes.

When modifying training code, keep this partitioning explicit and inspectable. Router/gate weights are intentionally treated as shared parameters.

### 4. SafeMoE training stages
- `litgpt/safemoe/pretrain.py` is the main SafeMoE training entrypoint. It is a fork of LitGPT pretraining with a single-optimizer, split-aware SGTM loop.
- Two stages are supported:
  - `stage: warmup`: only `D_std` and `D_harmful` are active, plus an auxiliary routing loss that pushes harmful tokens toward harmful experts and standard tokens away from them.
  - `stage: transfer`: full SGTM training over `D_std`, `D_harmful`, and `D_unlabeled`.
- Warmup can derive a checkpoint on the fly from a base model via `base_checkpoint`, `num_harmful_experts`, `num_harmful_attn_heads`, and `epsilon`; this flows through `maybe_prepare_warmup_checkpoint()` into `litgpt.safemoe.surgery.setup()`.
- The training loop samples among dataset splits, applies activation/gradient masking as needed, logs routing metrics, and can run ablated validation by temporarily zeroing `theta_harmful`.

### 5. SafeMoE data pipeline
- `litgpt/data/safedata.py` defines `SafeData`, which organizes multiple datasets into three training streams:
  - `D_std`
  - `D_harmful`
  - `D_unlabeled`
- Each configured dataset has a `role` (`std` or `harmful`) and `label_ratio`; the labeled portion is routed into either `D_std` or `D_harmful`, while the remainder becomes `D_unlabeled`.
- `prepare_data()` delegates to `litgpt.data.safe_prepare.prepare_dataset` to tokenize and cache streaming-ready splits under `data/.cache/...`.
- Training uses LitData streaming datasets/loaders rather than loading everything into memory.

### 6. Surgery, ablation, and evaluation
- `litgpt/safemoe/surgery.py` is responsible for duplicating experts / attention heads and initializing harmful-specific parameters from a base checkpoint.
- `litgpt/safemoe/ablate.py` builds a `HarmfulParamRegistry`, zeros all `theta_harmful` weights in a checkpoint, and writes an `ablated/` copy plus an `ablation_manifest.json`.
- `litgpt/safemoe/evaluate.py` evaluates original and optionally ablated checkpoints; if needed, it can run surgery first when the checkpoint has no harmful experts but the hyperparameters describe how to create them.
- `litgpt/safemoe/observability.py` contains routing observability utilities used during training/validation artifact generation.

## Important Config and Entry Files
- `pyproject.toml`: package metadata, extras, Ruff config, and pytest defaults.
- `config_hub/`: upstream LitGPT example recipes for pretraining and finetuning.
- `safemoe/configs/`: SafeMoE experiment configs, including TinyStories transfer and Qwen warmup examples.
- `tests/`: upstream LitGPT tests; SafeMoE changes should preserve existing LitGPT behavior unless the change is intentionally SafeMoE-specific.

## Repository-Specific Guidance
- Preserve the distinction between harmful-specific, standard, and shared parameters.
- Keep routing, masking, surgery, and ablation logic explicit rather than abstracting it away.
- For changes in `litgpt/safemoe/pretrain.py`, verify whether they affect:
  - parameter ownership (`theta_harmful` / `theta_std` / `theta_shared`)
  - router behavior during warmup and transfer
  - split-specific masking on `D_std`, `D_harmful`, and `D_unlabeled`
  - ablation-time behavior and evaluation outputs
- Prefer updating existing SafeMoE hooks over creating parallel training paths.
- SafeMoE builds on LitGPT conventions, so many CLI and config patterns come from LitGPT even when the research logic lives in `litgpt/safemoe/`.
