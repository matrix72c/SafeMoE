# Phase 5: Environment Runtime Gate - Research

**Researched:** 2026-03-19
**Domain:** Direct-Qwen checkpoint loading and BF16 runtime gating on the existing SafeMoE/LitGPT stack
**Confidence:** MEDIUM

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
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

### Deferred Ideas (OUT OF SCOPE)
- Broader topology certification across multiple device counts or strategies.
- CPU-only, non-BF16, or mixed-precision fallback qualification.
- Deeper checkpoint introspection or schema auditing beyond what is needed to prove loadability.
- Any checkpoint surgery, registry validation, routing observability, warmup, transfer, or evaluation behavior beyond the one-step environment/runtime gate.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ENV-01 | Researcher can load the existing LitGPT-converted `Qwen3-30B-A3B-Base` checkpoint from `checkpoints/Qwen3-30B-A3B-Base` without missing-file, schema, or checkpoint-compatibility errors. | Use the existing `safemoe.pretrain.setup()` -> `main()` -> `fabric.load_raw(initial_checkpoint_dir / "lit_model.pth", model)` path, validate the required checkpoint/tokenizer files up front, and keep loader validation on the real direct-Qwen model config from `model_config.yaml`. |
| ENV-02 | Researcher can run a dry-start BF16 training or evaluation job on the direct-Qwen stack and record the storage, memory, and runtime envelope needed for milestone runs. | Use one real `python -m safemoe pretrain` BF16 dry-start with the blessed topology, pin `bf16-true`, measure startup + first optimizer step, and write a committed markdown envelope artifact with command, topology, storage, peak memory, and throughput. |
</phase_requirements>

## Summary

Phase 5 should plan around the stack that already exists in this repository, not around a new Qwen integration path. The direct milestone entry point is `python -m safemoe pretrain`, which resolves `initial_checkpoint_dir`, instantiates the direct model from `model_config.yaml`, and loads `lit_model.pth` through `Fabric.load_raw(...)`. The checkpoint directory is already present locally and includes the converted `lit_model.pth`, model config, tokenizer files, and Hugging Face symlinks, so the gate is fundamentally a runtime validation and measurement phase.

The key planning decision is to bless exactly one BF16 training topology and prove it with a real first optimizer step. The existing code already switches from single-device `"auto"` to Fabric FSDP with `HYBRID_SHARD` when `devices * num_nodes > 1`, and it already prints total parameters, total training time, token throughput, and peak allocated CUDA memory. That means Phase 5 should avoid a broad “benchmark matrix” and instead add, at most, narrow measurement instrumentation to isolate startup time and first-step time cleanly.

**Primary recommendation:** Plan Phase 5 as two tightly scoped tasks: first, validate `checkpoints/Qwen3-30B-A3B-Base` on the exact `safemoe.pretrain` load path; second, run one real `bf16-true` dry-start on the blessed topology and commit a markdown runtime-envelope report from that run.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| local `litgpt` fork | `0.5.12` (repo `pyproject.toml`) | Base training/runtime stack and Qwen-compatible model/config classes | Already wired through `safemoe/pretrain.py`; replacing it would invalidate prior phases. |
| `lightning` | `2.6.1` (installed; PyPI release 2026-01-30) | Fabric launch, device strategy, checkpoint IO, logging | Current repo runtime already depends on Fabric semantics, including `FSDPStrategy` and `load_raw`. |
| `torch` | `2.10.0` (installed; PyPI release 2026-01-21) | CUDA/BF16 runtime, optimizer execution, peak-memory metrics | Current environment is ahead of the repo minimum and exposes the BF16 and CUDA memory APIs Phase 5 needs. |
| `jsonargparse` | `4.37-4.41` (repo constraint) | CLI/config loading for `python -m safemoe pretrain` | The SafeMoE CLI already depends on it; Phase 5 should use the existing config surface, not a custom launcher. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `litdata` | `0.2.59` (repo pin) | Streaming train/val loaders behind `MultiDataLoader` | Required for the real pretrain path because `safemoe.pretrain.setup()` requires `data`. |
| `tokenizers` | `0.22.2` installed; repo minimum `0.21` | Qwen tokenizer loading from checkpoint assets | Use via `Tokenizer(tokenizer_dir)` with the existing checkpoint directory. |
| `torchmetrics` | `1.8.2` installed; PyPI current `1.9.0` on 2026-03-09 | Running mean aggregation inside training loop | Reuse as-is; not a planning focus, but part of the working runtime path. |
| `tensorboard` logger | repo optional extra | Runtime logging output under `out_dir/logs/tensorboard/` | Use when keeping the default logger; phase artifact should still be a committed markdown file. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `safemoe.pretrain` direct path | Standalone loader script | Worse: duplicates model/config resolution and can pass while the real training path still fails. |
| One blessed topology | Multi-topology sweep | Out of scope and expands Phase 5 into infrastructure benchmarking. |
| Existing Fabric/FSDP behavior | New DeepSpeed or HF Trainer path | Explicitly out of scope in requirements and adds integration risk. |
| Existing end-of-run metrics plus narrow timing hooks | Full profiler stack | Useful later, but too heavy for a gate whose goal is “startup + one measured step.” |

**Installation:**
```bash
python -m pip install -e ".[extra,test]"
```

**Version verification:** Current package versions were verified against PyPI JSON on 2026-03-19.
```bash
python - <<'PY'
import json, urllib.request
for pkg in ["torch", "lightning", "transformers", "tokenizers", "torchmetrics", "safetensors", "huggingface_hub"]:
    data = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/json"))
    print(pkg, data["info"]["version"])
PY
```
Verified current releases:
- `torch` `2.10.0` — published 2026-01-21
- `lightning` `2.6.1` — published 2026-01-30
- `transformers` `5.3.0` — published 2026-03-04
- `tokenizers` `0.22.2` — published 2026-01-05
- `torchmetrics` `1.9.0` — published 2026-03-09
- `safetensors` `0.7.0` — published 2025-11-19
- `huggingface_hub` `1.7.1` — published 2026-03-13

## Architecture Patterns

### Recommended Project Structure
```text
.planning/phases/05-environment-runtime-gate/
├── 05-RESEARCH.md              # This research file
├── 05-PLAN.md                  # Planner output
└── 05-runtime-envelope.md      # Canonical artifact from the blessed run

safemoe/
├── pretrain.py                 # Real runtime gate path and any narrow measurement hooks
└── configs/
   └── safemoe-*.yaml           # Reusable CLI config, if a Phase 5-specific config is needed

tests/safemoe/
└── test_phase5_runtime_gate.py # New gate-specific tests if planner adds them
```

### Pattern 1: Validate on the real training-path loader
**What:** Perform file and compatibility checks on the exact path used by milestone training: `initial_checkpoint_dir` -> `extend_checkpoint_dir(...)` -> `GPT(config)` -> `fabric.load_raw(...)`.
**When to use:** Always for ENV-01. Do not create a parallel loader.
**Example:**
```python
# Source: local code in safemoe/pretrain.py
if initial_checkpoint_dir is not None:
    initial_checkpoint_dir = extend_checkpoint_dir(initial_checkpoint_dir)

with fabric.init_module(empty_init=True):
    model = GPT(config)

model = fabric.setup(model)

if initial_checkpoint_dir:
    fabric.load_raw(initial_checkpoint_dir / "lit_model.pth", model)
```

### Pattern 2: Let topology selection flow from existing Fabric logic
**What:** Use the current strategy rule: single device uses `"auto"`, multi-device uses `FSDPStrategy(auto_wrap_policy={Block}, state_dict_type="full", sharding_strategy="HYBRID_SHARD")`.
**When to use:** For the blessed runtime topology and any preliminary dry-run probes.
**Example:**
```python
# Source: local code in safemoe/pretrain.py
if devices * num_nodes > 1:
    strategy = FSDPStrategy(
        auto_wrap_policy={Block},
        state_dict_type="full",
        sharding_strategy="HYBRID_SHARD",
    )
else:
    strategy = "auto"

fabric = L.Fabric(
    devices=devices,
    num_nodes=num_nodes,
    strategy=strategy,
    precision=precision,
    loggers=[logger],
)
```

### Pattern 3: Measure the gate on one real optimizer step
**What:** The runtime artifact should come from a true `pretrain` launch with `bf16-true`, `train.max_steps=1`, and the real checkpoint/data stack.
**When to use:** For ENV-02 pass/fail and the committed envelope artifact.
**Example:**
```bash
# Source: local CLI contract in python -m safemoe pretrain --help
python -m safemoe pretrain \
  --config safemoe/configs/<phase5-config>.yaml \
  --precision bf16-true \
  --initial_checkpoint_dir checkpoints/Qwen3-30B-A3B-Base \
  --tokenizer_dir checkpoints/Qwen3-30B-A3B-Base \
  --train.max_steps 1 \
  --eval.initial_validation false \
  --eval.final_validation false \
  Qwen3-30B-A3B-Base
```

### Anti-Patterns to Avoid
- **Eval-only gate:** The context explicitly rejects it. Even a successful eval load does not satisfy Phase 5.
- **Custom checkpoint introspection framework:** The phase only needs required-file checks and a real `load_raw(...)` proof.
- **Topology matrix benchmarking:** Certify one real baseline first; record other failures as notes, not as scope.
- **Broad profiler integration:** Add only the timing/memory hooks needed to isolate startup and first-step metrics.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Checkpoint path resolution | Custom symlink/path resolver | `litgpt.utils.extend_checkpoint_dir()` | The repo already uses it for checkpoint and tokenizer dirs. |
| Checkpoint compatibility validation | Manual schema parser for every tensor | `GPT(config)` + `fabric.load_raw(...)` | The real compatibility question is whether the direct training model loads the converted weights. |
| Multi-GPU sharding setup | Custom distributed bootstrap | `Lightning Fabric` + `FSDPStrategy` | Current code already encodes the supported topology behavior. |
| Peak-memory accounting | Ad hoc `nvidia-smi` parsing only | `torch.cuda.reset_peak_memory_stats()` + `torch.cuda.max_memory_allocated()` | PyTorch tracks peak allocated tensor memory directly; use external tools only as supplemental notes. |
| Throughput smoothing | Custom moving-average tracker | `ThroughputMonitor` and existing end-of-run logging | Already in the training loop and sufficient for the gate. |

**Key insight:** Phase 5 is a trust-building gate, not a new runtime subsystem. The safest plan is to reuse the exact runtime, checkpoint, and logger path that later phases will depend on.

## Common Pitfalls

### Pitfall 1: Passing a synthetic loader while the real path still fails
**What goes wrong:** A standalone script can prove files exist or that a partial state dict loads, but later `python -m safemoe pretrain` still fails on config resolution, wrapper setup, or optimizer startup.
**Why it happens:** The validation path diverges from the milestone training path.
**How to avoid:** Run ENV-01 through `safemoe.pretrain` internals or the CLI itself.
**Warning signs:** File checks pass, but the first real training launch fails before or during `load_raw(...)`.

### Pitfall 2: Misreading FSDP-wrapped parameter behavior
**What goes wrong:** Post-setup inspection unwraps the wrong module and sees meta tensors or broken names.
**Why it happens:** Fabric/FSDP wrapping changes where real parameters live.
**How to avoid:** Follow the existing `safemoe/pretrain.py` pattern: only unwrap DDP-like wrappers, not FSDP.
**Warning signs:** Registry code or diagnostics touch `.module` under FSDP and encounter meta-device parameters.

### Pitfall 3: Reporting misleading peak-memory numbers
**What goes wrong:** The artifact records memory accumulated across warmup/setup noise instead of the intended gate window.
**Why it happens:** Peak-memory stats were never reset around the measured run boundary.
**How to avoid:** Reset peak CUDA stats immediately before the timed startup/step measurement and synchronize before sampling.
**Warning signs:** Reported peak memory changes substantially between identical one-step runs without a code change.

### Pitfall 4: Treating BF16 as “configured” rather than “supported”
**What goes wrong:** The command uses `--precision bf16-true`, but the hardware/runtime cannot actually complete the first step.
**Why it happens:** CLI precision selection is not the same as a successful BF16 execution path.
**How to avoid:** Gate BF16 with both `torch.cuda.is_bf16_supported()` and a real one-step launch on the blessed topology.
**Warning signs:** Startup completes but the run fails on the first forward/backward/optimizer step.

### Pitfall 5: Forgetting that data is required even for the runtime gate
**What goes wrong:** The planner assumes Phase 5 can run without data because it only needs one step.
**Why it happens:** `safemoe.pretrain.setup()` requires a real `MultiDataLoader`; the training path is not load-only.
**How to avoid:** Reuse the existing TinyStories cache and configure `MultiDataLoader` explicitly in the gate command/config.
**Warning signs:** The run aborts with `ValueError("data (MultiDataLoader) is required for SGTM training")`.

## Code Examples

Verified patterns from local code and official docs:

### Required checkpoint surface
```text
checkpoints/Qwen3-30B-A3B-Base/
├── lit_model.pth
├── model_config.yaml
├── tokenizer.json
├── tokenizer_config.json
├── config.json
└── generation_config.json
```
Source: local filesystem inspection on 2026-03-19.

### Minimal measurement hook for Phase 5
```python
# Source: recommended Phase 5 instrumentation based on local pretrain path and
# PyTorch CUDA memory APIs
startup_t0 = time.perf_counter()
fabric.launch()

with fabric.init_module(empty_init=True):
    model = GPT(config)
model = fabric.setup(model)
fabric.load_raw(initial_checkpoint_dir / "lit_model.pth", model)

if fabric.device.type == "cuda":
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

first_step_t0 = time.perf_counter()
# run one optimizer step through fit()
if fabric.device.type == "cuda":
    torch.cuda.synchronize()

first_step_s = time.perf_counter() - first_step_t0
peak_mem_gb = (
    torch.cuda.max_memory_allocated() / 1e9
    if fabric.device.type == "cuda"
    else None
)
```

### Runtime-envelope markdown skeleton
```markdown
# Phase 5 Runtime Envelope

- Date: 2026-03-19
- Command: `python -m safemoe pretrain ...`
- Checkpoint: `checkpoints/Qwen3-30B-A3B-Base`
- Topology: `{devices} GPU x {num_nodes} node`, `{strategy}`
- Precision: `bf16-true`
- Seed: `42`
- Storage footprint:
  - `lit_model.pth`: `57G`
- Timings:
  - startup seconds: `...`
  - first-step seconds: `...`
- Runtime:
  - tokens/sec after first measured step: `...`
- Peak memory:
  - max allocated GB: `...`
- Notes:
  - warnings observed
  - environment assumptions
  - replay instructions
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Validate Qwen support through Hugging Face-only loader assumptions | Validate the converted checkpoint on the direct LitGPT/Fabric training path | Current milestone v1.1 | Reduces false confidence from a non-milestone loader path. |
| Treat broad compatibility as the goal | Certify one concrete milestone topology | Locked in Phase 5 context on 2026-03-19 | Keeps the phase bounded and planner-friendly. |
| Use BF16 as a general “mixed precision” umbrella | Pin `bf16-true` and require one real optimizer step | Locked in Phase 5 context on 2026-03-19 | Makes the gate a meaningful training-readiness signal. |
| Depend on repo minimums only | Verify current installed/runtime versions and current upstream release dates | Research date 2026-03-19 | Avoids stale package assumptions when planning a hardware-sensitive gate. |

**Deprecated/outdated:**
- `transformers<4.51.0` for Qwen3-MoE handling: the official Qwen3 model card warns that older `transformers` versions can fail with `KeyError: 'qwen3_moe'`. This matters for surrounding tooling, even though Phase 5 should stay on the direct LitGPT path.
- `memory_cached()` / `max_memory_cached()`: PyTorch marks these deprecated in favor of reserved-memory APIs; for Phase 5, prefer `max_memory_allocated()` for the artifact’s peak tensor-memory metric.

## Open Questions

1. **What is the exact blessed topology for the first milestone run?**
   - What we know: the code supports single-device `"auto"` and multi-device FSDP `HYBRID_SHARD`, and the checkpoint weight file alone is `57G`.
   - What's unclear: whether the real BF16 one-step milestone baseline fits on 1 GPU or requires multi-GPU sharding in this environment.
   - Recommendation: Plan a short topology probe that blesses the smallest topology that completes one BF16 optimizer step, then treat only that topology as Phase 5 scope.

2. **Do existing end-of-run metrics already expose enough detail for the artifact?**
   - What we know: `safemoe/pretrain.py` already prints total training time, tokens/sec, and `torch.cuda.max_memory_allocated()`.
   - What's unclear: whether startup time and first-step time can be separated cleanly without a narrow code change.
   - Recommendation: Assume one small instrumentation patch in `safemoe/pretrain.py` is acceptable if current logging does not isolate startup and first-step timings.

3. **Should ENV-01 use a dedicated test helper or reuse the CLI end-to-end?**
   - What we know: existing tests cover mocked/small-checkpoint pretrain behaviors but not the real Qwen checkpoint path.
   - What's unclear: whether the planner wants a reusable helper for file checks and loader smoke, or only a one-off execution command.
   - Recommendation: Add a narrow helper only if it improves repeatability for the planner; otherwise keep validation in the CLI path and record the exact command in the artifact.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | `pytest` via `pyproject.toml` |
| Config file | `pyproject.toml` |
| Quick run command | `pytest tests/safemoe/test_pretrain.py -x` |
| Full suite command | `pytest tests/safemoe -x` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ENV-01 | Required checkpoint files exist and the direct SafeMoE/Qwen load path can call `fabric.load_raw(...)` successfully against `checkpoints/Qwen3-30B-A3B-Base` | integration | `pytest tests/safemoe/test_phase5_runtime_gate.py::test_qwen_checkpoint_load_path -x` | ❌ Wave 0 |
| ENV-02 | Real `bf16-true` dry-start completes startup plus one optimizer step and records envelope metrics | manual-only hardware smoke | `python -m safemoe pretrain --config safemoe/configs/<phase5-config>.yaml --precision bf16-true --initial_checkpoint_dir checkpoints/Qwen3-30B-A3B-Base --tokenizer_dir checkpoints/Qwen3-30B-A3B-Base --train.max_steps 1 --eval.initial_validation false --eval.final_validation false Qwen3-30B-A3B-Base` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/safemoe/test_pretrain.py -x`
- **Per wave merge:** `pytest tests/safemoe -x`
- **Phase gate:** Real BF16 one-step smoke command above plus committed runtime-envelope markdown

### Wave 0 Gaps
- [ ] `tests/safemoe/test_phase5_runtime_gate.py` — covers ENV-01 direct-checkpoint file validation and load-path smoke
- [ ] `safemoe/configs/<phase5-config>.yaml` — pins the blessed topology, tokenizer dir, checkpoint dir, and one-step gate settings
- [ ] `05-runtime-envelope.md` artifact template or generator path — ensures ENV-02 output is committed and reproducible
- [ ] Real GPU smoke harness for ENV-02 — existing pytest coverage is mocked/small-model only and does not certify the real checkpoint

## Sources

### Primary (HIGH confidence)
- Local repo: `safemoe/pretrain.py` — verified the real checkpoint load path, topology selection, and existing throughput/memory logging
- Local repo: `safemoe/data/datamodule.py` — verified that the real pretrain path requires `MultiDataLoader`
- Local repo: `safemoe/__main__.py` and `python -m safemoe pretrain --help` — verified the actual CLI surface
- Local repo: `checkpoints/Qwen3-30B-A3B-Base/` and `model_config.yaml` — verified required files and pinned direct-Qwen config
- Official Qwen model card: https://huggingface.co/Qwen/Qwen3-30B-A3B-Base — verified model shape and the `transformers>=4.51.0` guidance
- Official Lightning Fabric docs: https://lightning.ai/docs/fabric/stable/api/generated/lightning.fabric.strategies.FSDPStrategy.html — verified `HYBRID_SHARD`, `state_dict_type="full"`, and `use_orig_params=True` behavior
- Official PyTorch CUDA docs: https://docs.pytorch.org/docs/stable/cuda.html — verified `torch.cuda.is_bf16_supported`, `max_memory_allocated`, and `reset_peak_memory_stats`
- Official PyPI JSON API:
  - https://pypi.org/pypi/torch/json
  - https://pypi.org/pypi/lightning/json
  - https://pypi.org/pypi/transformers/json
  - https://pypi.org/pypi/tokenizers/json
  - https://pypi.org/pypi/torchmetrics/json
  - https://pypi.org/pypi/safetensors/json
  - https://pypi.org/pypi/huggingface_hub/json

### Secondary (MEDIUM confidence)
- Local repo: `pyproject.toml` — verified repo-level dependency floors and optional extras
- Local repo: `tests/safemoe/test_pretrain.py` and `tests/test_pretrain.py` — verified current test coverage boundaries and gaps

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - based on local runtime code, installed environment, and current official package/version sources
- Architecture: HIGH - derived directly from the existing `safemoe.pretrain` control flow and official Fabric strategy docs
- Pitfalls: MEDIUM - strongly supported by local code and official docs, but the real blessed topology has not yet been executed in this research step

**Research date:** 2026-03-19
**Valid until:** 2026-04-18
