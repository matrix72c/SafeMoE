# Stack Research

**Domain:** Direct harmful-transfer on `Qwen3-30B-A3B-Base`
**Researched:** 2026-03-19
**Confidence:** HIGH

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| PyTorch | `>=2.7` | Full-weight training, checkpoint surgery, FSDP sharding | The repo already targets `torch>=2.7`, and current PyTorch FSDP remains the right fit for full-parameter intervention. This milestone needs real parameter cloning and mutation, not adapter-only training. |
| Lightning Fabric | `>=2.6.1` | Existing distributed/runtime orchestration | Keep the current Fabric/FSDP path in `safemoe/pretrain.py`. It already handles the repo’s training lifecycle and avoids a parallel training stack. |
| Transformers | `>=4.51.3,<4.57` | Canonical HF loading for `Qwen3MoeForCausalLM` and tokenizer/config validation | Qwen’s official docs require `transformers>=4.51.0` for Qwen3. The repo already pins a safer floor. This is the authoritative model-access layer for direct Qwen3 MoE import. |
| huggingface-hub + safetensors | `huggingface-hub>=0.30,<1.4`, `safetensors>=0.4.3` | Download and read the 16-shard base checkpoint | `Qwen3-30B-A3B-Base` is published as `safetensors` shards totaling about `61.1 GB`. You need the Hub client plus safetensors for raw access and conversion. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `hf_transfer` via `huggingface-hub[hf-transfer]` | same hub major line | Faster checkpoint download | Recommended for the first model pull only. The checkpoint is large enough that slow download tooling becomes operational drag. |
| `tensorboard` | `>=2.14` | Router and warmup observability | Required for this milestone’s new telemetry: per-layer routing concentration, warmup routing loss, and harmful-vs-standard dispatch separation. |
| `torchmetrics` | `>=1.3.1` | Split-aware aggregation | Use for consistent aggregation of routing and transfer metrics across `D_std`, `D_harmful`, and `D_unlabeled`. |
| `transformers` model outputs only | same as above | Router-logit extraction and HF-side validation | Use during import validation and for parity tests against HF `Qwen3MoeForCausalLM`. Do not build a second training loop around `Trainer`. |
| `bitsandbytes` | existing optional dep only | Emergency memory triage for inspection/inference, not milestone training | Keep available, but not on the critical path. HF docs state 8-bit and 4-bit training only support training extra parameters, which conflicts with expert/head cloning and full-weight surgery. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| Existing LitGPT conversion scripts | HF checkpoint import/export | Reuse `litgpt/scripts/convert_hf_checkpoint.py` and `litgpt/scripts/convert_lit_checkpoint.py`. They already map Qwen3 MoE expert and gate weights. |
| Existing SafeMoE CLI | Training/eval entry points | Extend `python -m safemoe pretrain` and `python -m safemoe evaluate`; do not add a second orchestration surface for this milestone. |

## Required Additions

### 1. Direct Qwen3 MoE import path

Add a Qwen-specific initialization/surgery layer on top of the existing LitGPT conversion path.

**Why it matters**
- Harmful-expert initialization requires copying specific expert MLP weights and selected attention-head slices from the pretrained base model, not from a randomly initialized proxy.
- Router-supervised warmup requires direct access to Qwen’s router weights so copied experts can receive matching gate columns or explicitly initialized alternatives.
- SGTM transfer needs a checkpoint path that round-trips between raw HF weights and the modified LitGPT/SafeMoE checkpoint without losing MoE structure.

**Concrete integration points**
- `litgpt/config.py`: `Qwen3-30B-A3B-Base` is already registered.
- `litgpt/scripts/convert_hf_checkpoint.py`: already maps `model.layers.{i}.mlp.experts.{j}.*` and `model.layers.{i}.mlp.gate.weight`.
- `litgpt/scripts/convert_lit_checkpoint.py`: already maps the reverse direction.
- New addition should be a small Qwen surgery utility in SafeMoE, not a replacement conversion stack.

**Recommendation**
- Add a dedicated utility that:
- loads the converted LitGPT checkpoint,
- clones `k` experts and `n` attention heads from designated source indices,
- applies controlled noise,
- duplicates or reinitializes the corresponding router columns,
- writes a surgery manifest recording source expert/head indices and noise scale.

### 2. Router-aware observability expansion

The current routing evaluation is too coarse for this milestone.

**Why it matters**
- Warmup success is about **where tokens route**, not just NTP loss.
- SGTM transfer success requires showing that `D_unlabeled` migrates toward `theta_harmful` while `D_std` stays away.
- Expert/head cloning needs evidence that copied components become active rather than remaining dead or collapsing.

**Required new logging**
- Per-layer harmful routing fraction for `D_std`, `D_harmful`, and `D_unlabeled`.
- Per-expert token counts and top-k occupancy, not just a global harmful fraction.
- Warmup routing loss and any margin/separation terms, split by dataset.
- Head-level activation summaries for cloned harmful heads during warmup and transfer.
- Checkpoint-surgery metadata: source expert IDs, source head IDs, duplicated router columns, noise std.

**Concrete local targets**
- Extend `safemoe/evaluate.py` beyond `routing_harmful_frac_*`.
- Add training-time logging in `safemoe/pretrain.py` near the existing logger usage.
- Preserve JSON artifacts alongside TensorBoard so runs remain scriptable.

### 3. Runtime constraints for 30B-A3B training

Treat this milestone as a large-model FSDP job, not as the old TinyStories-scale run.

**Why it matters**
- The public HF checkpoint is about `61.1 GB` across 16 safetensor shards. Direct download plus converted checkpoint plus optimizer/checkpoints means disk headroom becomes a real requirement.
- The published config sets `torch_dtype` to `bfloat16`; BF16-capable GPUs are the intended path.
- Full-weight cloning and warmup are incompatible with the repo’s quantized extra-parameter training patterns.

**Recommendation**
- Use FSDP with BF16 as the default training path.
- Keep optimizer/checkpoint state on the existing Fabric path.
- Budget storage for:
- raw HF checkpoint,
- converted LitGPT checkpoint,
- modified SafeMoE checkpoint(s),
- optimizer and eval artifacts.

**Practical requirement**
- Plan for roughly `130-180 GB` of working disk, not just the raw `61.1 GB` download.

This storage estimate is an inference from the public model size plus local conversion/checkpoint duplication.

### 4. Model-access requirements

| Requirement | Status | Why it matters |
|-------------|--------|----------------|
| Access to `Qwen/Qwen3-30B-A3B-Base` on Hugging Face | Required | This is the direct base checkpoint for the milestone. |
| HF authentication token | Not required for the public model, but still useful operationally | The repo supports `HF_TOKEN`; the model page is public, so gating is not the blocker here. |
| Python `>=3.10` | Required | Matches both repo and Qwen quickstart requirements. |
| `transformers>=4.51.0` | Required | Qwen official docs call this out for Qwen3 support. |
| BF16-capable CUDA GPUs | Required in practice | The model config is published with `torch_dtype: bfloat16`; training this model meaningfully without BF16-class hardware is the wrong operating mode. |

## Nice-to-Haves, Not Requirements

| Addition | Why it can help | Why it is optional |
|----------|-----------------|--------------------|
| `hf_transfer` | Makes the first checkpoint pull less painful | Does not affect correctness or integration. |
| W&B or MLflow logging | Better run comparison across many warmup/transfer sweeps | TensorBoard plus JSON artifacts is enough for this milestone. |
| FlashAttention-specific optimization | May improve throughput on supported GPUs | Not required to prove the thesis. Add only if the existing kernels are the bottleneck after the first end-to-end run. |

## What NOT to Add

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| DeepSpeed | Adds a second distributed-training stack when Fabric/FSDP already exists locally | Keep the current Fabric + FSDP path and make the Qwen work fit it. |
| HF `Trainer` / `Accelerate` as a new primary loop | Duplicates the repo’s training system and fractures instrumentation | Use HF only for model-format validation and raw checkpoint access. |
| PEFT / LoRA / QLoRA as the main intervention mechanism | This milestone needs direct expert/head cloning and full-weight router surgery, not extra-parameter adaptation | Train the actual cloned experts/heads in the existing SafeMoE loop. |
| Quantized 8-bit or 4-bit training for the main run | HF docs explicitly note 8/4-bit training is only for training extra parameters | Use BF16 full-weight training. |
| vLLM / TGI / serving infrastructure | Serving is out of scope and does not help expert cloning or warmup routing supervision | Keep this milestone offline and research-oriented. |
| New MoE frameworks such as MegaBlocks or Tutel | They solve scaling/dispatch problems the repo is not currently blocked by, while making expert surgery and observability harder | Stay on the existing LitGPT/Qwen conversion path. |

## Stack Patterns by Variant

**If the goal is checkpoint import, cloning, and warmup research:**
- Use HF Hub + `transformers` + LitGPT conversion + SafeMoE surgery utilities.
- Because this preserves direct access to Qwen3 MoE weights while keeping the current training stack intact.

**If the goal is only quick inspection or one-off inference:**
- Optional quantized loading is acceptable.
- Because inference memory pressure is different from full-weight training, but this should not become the main milestone path.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `transformers>=4.51.3,<4.57` | `Qwen3MoeForCausalLM` | Repo pin already covers Qwen’s published `>=4.51.0` requirement. |
| `torch>=2.7` | `lightning>=2.6.1` | Matches the repo’s active Fabric/FSDP stack. |
| `huggingface-hub>=0.30,<1.4` | `safetensors>=0.4.3` | Required for large safetensors snapshot download and reading. |
| `bitsandbytes` | inference or extra-parameter tuning only | Do not rely on it for the main direct-intervention training path. |

## Sources

- Local codebase: `pyproject.toml`, `safemoe/pretrain.py`, `safemoe/evaluate.py`, `litgpt/config.py`, `litgpt/scripts/convert_hf_checkpoint.py`, `litgpt/scripts/convert_lit_checkpoint.py` — verified directly in repo.
- Qwen official quickstart: https://qwen.readthedocs.io/en/stable/getting_started/quickstart.html
  - Verified `transformers>=4.51.0`, Python `>=3.10`, PyTorch `>=2.6`.
- Hugging Face model tree for `Qwen/Qwen3-30B-A3B-Base`: https://huggingface.co/Qwen/Qwen3-30B-A3B-Base/tree/main
  - Verified public model availability, safetensors format, 16 shards, and total repository size about `61.1 GB`.
- Hugging Face model config for `Qwen/Qwen3-30B-A3B-Base`: https://huggingface.co/Qwen/Qwen3-30B-A3B-Base/blob/main/config.json
  - Verified `Qwen3MoeForCausalLM`, `model_type=qwen3_moe`, `num_experts=128`, `num_experts_per_tok=8`, `torch_dtype=bfloat16`, and published `transformers_version=4.51.0`.
- PyTorch FSDP docs: https://docs.pytorch.org/docs/stable/fsdp.html
  - Verified FSDP remains the standard full-shard training API and state-dict handling guidance.
- Hugging Face bitsandbytes docs: https://huggingface.co/docs/transformers/main/quantization/bitsandbytes
  - Verified that 8-bit and 4-bit training only support training extra parameters, which is incompatible with this milestone’s full-weight intervention.

---
*Stack research for: direct `Qwen3-30B-A3B-Base` intervention*
*Researched: 2026-03-19*
