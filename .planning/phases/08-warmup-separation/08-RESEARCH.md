# Phase 08: Warmup Separation - Research

**Researched:** 2026-03-19
**Domain:** Warmup-stage routing supervision on the direct-Qwen SafeMoE training path
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

### Warmup stage shape
- Warmup is a distinct stage before transfer, not just an early portion of Phase 9.
- Warmup trains on `D_std` and `D_harmful` only; `D_unlabeled` is excluded from formal warmup training.
- Warmup should publish a dedicated handoff checkpoint artifact that Phase 9 resumes from.
- The labeled split ratio remains configurable rather than fixed globally, even though the stage only uses `D_std` and `D_harmful`.

### Routing supervision target
- The routing objective should act on aggregate harmful-expert routing mass per split rather than hard per-token or per-layer targets.
- The required behavior is relational: `D_harmful` must route to `theta_harmful` more than `D_std`, with an explicit separation margin between the two.
- Next-token loss remains active throughout warmup; routing supervision is an added objective, not a replacement stage.
- Planning may choose the exact loss formula and weight, but it must preserve the behavior contract above.

### Warmup acceptance bar
- Warmup is only considered successful if routing separation improves and `D_std` language-model quality does not materially regress.
- The completion signal should be a final blessed checkpoint plus a final researcher-facing conclusion, not just an intermediate best step or a fixed step count.
- If routing separation improves but `D_std` degrades materially, Phase 8 does not pass and the artifact is not an approved handoff to Phase 9.
- `D_harmful` LM behavior may be reported, but the non-regression guard is specifically anchored on `D_std`.

### Confound-controlled evaluation evidence
- The primary proof artifact is a before-vs-after comparison on the same intervention lineage, showing harmful-routing metrics for `D_harmful` and `D_std` before warmup and after warmup.
- The core confound control to lock now is same-checkpoint pre/post comparison; planner may add stronger controls, but this one is mandatory.
- Phase 8 should produce a concise pass/fail researcher-facing artifact rather than leaving sign-off to raw logs alone.
- `D_unlabeled` is excluded from formal Phase 8 acceptance; it may appear later in transfer, not as a success criterion here.

### Claude's Discretion
- Exact config fields, CLI flags, and artifact filenames for the warmup stage and blessed handoff checkpoint.
- Exact routing-loss family, margin formulation, and initial weighting, as long as the behavior contract is preserved.
- Exact definition of "material regression" for `D_std`, provided planning makes it measurable and researcher-usable.
- Exact report layout for the Phase 8 pass/fail summary and any companion JSON metrics.

### Deferred Ideas (OUT OF SCOPE)
- Formal unlabeled-routing acceptance for `D_unlabeled` — belongs to Phase 9 transfer and later evaluation work.
- Final ablation-based capability claims or adversarial-recovery evidence — belongs to Phase 10.
- Per-token or per-layer routing targets as a required product-level contract.
- Hyperparameter sweeps over routing-loss weight, margin size, or warmup duration — these are follow-up experiment expansion, not Phase 8 baseline scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| WARM-01 | Researcher can run a warmup stage on mixed `D_harmful` and `D_std` while keeping next-token loss active and logging routing loss separately from LM loss. | Keep `safemoe/pretrain.py` as the only trainer, add a `warmup` stage/config path that samples only labeled splits, computes LM + routing losses together, and logs `loss_lm_*`, `loss_routing_*`, and `loss_total_*` separately. |
| WARM-02 | Researcher can apply a supervised routing objective that increases harmful-routing mass for `D_harmful` tokens and suppresses harmful-routing mass for `D_std` tokens. | Add a differentiable aggregate harmful-routing-mass statistic from `SafeMoELayer` top-k routing probabilities and a split-aware marginized routing loss with distinct `D_harmful` and `D_std` targets. |
| WARM-03 | Researcher can demonstrate, with a confound-controlled evaluation, that post-warmup `D_harmful` routes more strongly to `theta_harmful` than `D_std`. | Reuse Phase 7 observability and Phase 4 eval paths to generate same-lineage pre/post routing and perplexity comparisons, then write a concise pass/fail warmup summary and bless exactly one handoff checkpoint. |
</phase_requirements>

## Summary

Phase 8 should be implemented as a stage-gated extension of the existing direct-Qwen training path, not a second trainer. The current loop in [safemoe/pretrain.py](/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/safemoe/pretrain.py) already provides the right foundations: split-aware sampling, one optimizer, checkpoint save/resume, and checkpoint-local routing observability. What is missing is a differentiable routing supervision signal. Today the code only logs detached dispatch counts via `_last_indices`, which is sufficient for observability but not for training.

The clean implementation path is to keep next-token loss unchanged and add one extra labeled-only routing loss based on aggregate harmful-expert routing mass computed from the actual top-k routing probabilities inside `SafeMoELayer.forward()`. Because the pinned Qwen config uses `n_expert_groups: null` and `n_expert_per_token: 8`, the warmup loss can bind directly to the existing top-k path instead of solving grouped routing first. `D_unlabeled` should be removed from warmup sampling entirely, while the labeled split ratio remains configurable.

Acceptance should be decided by a same-lineage before/after report: Phase 6 surgery checkpoint versus final warmup checkpoint, both evaluated with the same routing observability contract and the same `D_std` / `D_harmful` validation splits. That report should bless one final warmup checkpoint for Phase 9 only if routing separation improved and `D_std` perplexity stayed inside a planner-defined non-regression threshold.

**Primary recommendation:** Extend [safemoe/pretrain.py](/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/safemoe/pretrain.py) with a `warmup` stage that samples only `D_std` and `D_harmful`, computes LM loss plus a differentiable aggregate harmful-routing-mass loss, and emits a final checkpoint-local pre/post acceptance report before blessing the Phase 9 handoff artifact.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.10.0+cu128 (installed) | Autograd, routing-loss math, checkpoint tensors | Already the runtime foundation and provides the exact loss primitives needed for warmup supervision. |
| Lightning Fabric | 2.6.1 (installed) | Device setup, distributed execution, checkpoint save/load | The existing direct-Qwen path already relies on Fabric; Phase 8 should stay inside that path. |
| LitGPT | 0.5.12 (repo version) | Qwen model/config/training utilities | `safemoe` is already a LitGPT fork; adding another trainer would create drift immediately. |
| `safemoe.pretrain` | repo local | Single-optimizer split-aware training loop | Already owns split sampling, optimizer steps, and save/resume semantics. |
| `safemoe.observability` + `safemoe.evaluate` | repo local | Shared routing metrics and researcher-facing evaluation | Phase 7 already established the canonical routing contract; Phase 8 should reuse it. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torchmetrics | 1.8.2 (installed) | Running means and metric smoothing | Continue using for rolling train metrics; no new metric stack needed. |
| PyYAML | 5.4.1 (installed) | YAML config read/write | Use for warmup config surfaces and saved hyperparameters. |
| pytest | 9.0.2 (installed) | Unit/integration validation | Use for new warmup-stage tests and acceptance-artifact checks. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Extending the existing trainer | A separate warmup script/trainer | Reject. It would duplicate split semantics, checkpoint semantics, and observability contracts already locked in Phases 5-7. |
| Differentiable harmful-routing mass from top-k probabilities | Hard dispatch-count supervision | Reject. Dispatch counts come from detached indices and are not usable as a training loss. |
| Split-aware marginized loss on labeled batches | Per-token or per-layer routing targets | Reject for Phase 8. The user explicitly deferred per-token/per-layer targets. |

**Installation:**
```bash
# No new dependencies required for Phase 8.
```

**Version verification:** Use the existing environment and repo pins rather than introducing new packages.
```bash
python - <<'PY'
import torch, lightning, torchmetrics, pytest
print(torch.__version__)
print(lightning.__version__)
print(torchmetrics.__version__)
print(pytest.__version__)
PY
```

## Architecture Patterns

### Recommended Project Structure
```text
safemoe/
├── pretrain.py        # stage switch, warmup loop integration, checkpoint blessing
├── model.py           # differentiable routing-mass capture in SafeMoELayer
├── observability.py   # shared routing artifact writing + warmup summary helpers
├── evaluate.py        # pre/post warmup comparison entrypoint
└── configs/           # warmup YAML matching the Phase 5 config style

tests/safemoe/
├── test_warmup_separation.py
├── test_pretrain.py
└── test_evaluate.py
```

### Pattern 1: Stage-Gated Extension, Not a New Trainer
**What:** Add an explicit warmup mode to the current pretrain path and keep all direct-Qwen execution inside the same `setup() -> main() -> fit()` pipeline.
**When to use:** Always for Phase 8.
**Example:**
```python
# Source basis:
# /mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/safemoe/pretrain.py

@dataclass
class WarmupArgs:
    enabled: bool = False
    routing_loss_weight: float = 0.1
    separation_margin: float = 0.2
    d_std_regression_limit: float = 0.05
    blessed_checkpoint_name: str = "warmup-blessed"

def active_split_labels(warmup: WarmupArgs) -> list[str]:
    return ["D_std", "D_harmful"] if warmup.enabled else ["D_std", "D_harmful", "D_unlabeled"]
```

### Pattern 2: Differentiable Aggregate Harmful-Routing Mass
**What:** Compute one scalar routing statistic per labeled batch by summing top-k probabilities assigned to harmful experts across all `SafeMoELayer` modules, then average across layers/tokens.
**When to use:** Every warmup training step for `D_std` and `D_harmful`.
**Example:**
```python
# Source basis:
# /mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/safemoe/model.py
# https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softplus.html

def warmup_routing_loss(harmful_mass: torch.Tensor, split_label: str, margin: float) -> torch.Tensor:
    center = 0.5
    harmful_floor = center + margin / 2
    std_ceiling = center - margin / 2
    if split_label == "D_harmful":
        return torch.nn.functional.softplus(harmful_floor - harmful_mass)
    if split_label == "D_std":
        return torch.nn.functional.softplus(harmful_mass - std_ceiling)
    return harmful_mass.new_zeros(())
```

### Pattern 3: Log LM Loss and Routing Loss Separately
**What:** Keep next-token loss as the main training objective and log the routing term separately from the total objective.
**When to use:** Every warmup step and every saved checkpoint.
**Example:**
```python
# Source basis:
# /mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/safemoe/pretrain.py

total_loss = lm_loss + warmup.routing_loss_weight * routing_loss
metrics = {
    f"loss_lm_{split_label}": lm_loss.item(),
    f"loss_routing_{split_label}": routing_loss.item(),
    f"loss_total_{split_label}": total_loss.item(),
    f"routing_harmful_mass_{split_label}": harmful_mass.item(),
}
```

### Pattern 4: Same-Lineage Pre/Post Acceptance
**What:** Compare the Phase 6 input checkpoint and the final warmup checkpoint using the exact same evaluation code path and write one short pass/fail artifact.
**When to use:** Once at the end of warmup; optionally also at intermediate save points for researcher visibility.
**Example:**
```python
# Source basis:
# /mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/safemoe/evaluate.py
# /mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/safemoe/observability.py

report = {
    "pre": evaluate_checkpoint(pre_warmup_ckpt),
    "post": evaluate_checkpoint(post_warmup_ckpt),
    "delta": compute_warmup_delta(pre, post),
    "pass": separation_improved and d_std_regression_ok,
}
write_warmup_summary(post_warmup_ckpt, report)
```

### Anti-Patterns to Avoid
- **Detached routing supervision:** Do not build the training loss from `_last_indices` or checkpoint-local JSON; those paths are for observability, not gradients.
- **Warmup through `D_unlabeled`:** Excluding `D_unlabeled` is a locked decision, not a tuning preference.
- **Best-step auto-blessing:** The user chose a final blessed checkpoint plus final conclusion, not "pick whichever checkpoint had the best routing fraction."
- **Opaque total loss only:** If LM and routing losses are not logged separately, WARM-01 is not satisfied and regressions will be impossible to diagnose.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Warmup execution path | A second training script | [safemoe/pretrain.py](/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/safemoe/pretrain.py) | The current loop already owns split sampling, optimizer state, resume semantics, and checkpoint layout. |
| Researcher-facing routing schema | A new analysis directory or dashboard | [safemoe/observability.py](/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/safemoe/observability.py) | Phase 7 already standardized checkpoint-local routing artifacts. |
| Warmup evaluation loop | Ad hoc model-loading scripts | [safemoe/evaluate.py](/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/safemoe/evaluate.py) | The repo already has perplexity and routing attribution entrypoints for the required splits. |
| Checkpoint persistence | Manual `torch.save()` trees with custom metadata rules | Fabric save/load plus existing `save_checkpoint()` | This preserves compatibility with the current checkpoint directories and hyperparameter exports. |

**Key insight:** Phase 8 is mostly a training-loss and acceptance-artifact extension. The codebase already has the execution, checkpoint, and routing-observability substrate. Reuse that substrate aggressively.

## Common Pitfalls

### Pitfall 1: Using Dispatch Counts as the Warmup Loss
**What goes wrong:** The routing objective is built from the Phase 7 observability path, which only exposes detached indices and counts.
**Why it happens:** The existing collector was designed for evaluation parity, not autograd.
**How to avoid:** Capture harmful-routing mass inside `SafeMoELayer.forward()` from the top-k probabilities before they are detached or reduced to counts.
**Warning signs:** `routing_loss` stays constant, has no gradient, or only changes when observability hooks run.

### Pitfall 2: Letting Warmup Sampling Drift from the Locked Split Set
**What goes wrong:** `D_unlabeled` remains in the weighted sampler and contaminates Phase 8 metrics.
**Why it happens:** `SPLIT_LABELS` is global today and defaults to all three splits.
**How to avoid:** Make active split labels stage-dependent and require warmup configs to set only labeled split weights.
**Warning signs:** Any `loss_*_D_unlabeled`, `dispatch_count_D_unlabeled`, or routing artifact key mentioning `D_unlabeled`.

### Pitfall 3: Confounded Acceptance Evidence
**What goes wrong:** The pre/post comparison uses different intervention lineages, different validation splits, or different metric paths.
**Why it happens:** Evaluation logic gets duplicated for warmup instead of calling the shared routing/perplexity surfaces.
**How to avoid:** Compare the exact Phase 6 input checkpoint against the exact final warmup checkpoint using the same `D_std` / `D_harmful` val loaders and the same routing artifact schema.
**Warning signs:** Pre and post reports live in different schemas or use different split names or metric keys.

### Pitfall 4: Routing Separation Improves While `D_std` Quietly Regresses
**What goes wrong:** Warmup looks successful on routing metrics but the standard-domain LM quality meaningfully degrades.
**Why it happens:** The planner optimizes only for routing separation.
**How to avoid:** Make `D_std` non-regression a formal pass/fail rule in the final summary, not a side note.
**Warning signs:** Routing metrics improve while `val_ppl_D_std` moves outside the allowed bound and the run is still treated as pass.

### Pitfall 5: Saving Artifacts Without Blessing Semantics
**What goes wrong:** The run emits multiple checkpoints but no explicit Phase 9 handoff decision.
**Why it happens:** Existing save intervals create raw artifacts, not stage-level sign-off.
**How to avoid:** Add an explicit blessed-checkpoint write or alias after final acceptance succeeds.
**Warning signs:** Researchers have to infer the handoff checkpoint from timestamps or log messages.

## Code Examples

Verified patterns from local code and official PyTorch docs:

### Capture a Differentiable Batch Routing Statistic
```python
# Source basis:
# /mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/safemoe/model.py

harmful_mask = torch.isin(indices, harmful_index_tensor)
harmful_mass = (probs * harmful_mask).sum(dim=1).mean()
self._last_harmful_mass = harmful_mass
```

### Compose Warmup Loss Without Replacing LM Loss
```python
# Source basis:
# /mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/safemoe/pretrain.py
# https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

lm_loss = chunked_cross_entropy(logits, targets)
routing_loss = warmup_routing_loss(harmful_mass, split_label, warmup.separation_margin)
loss = lm_loss + warmup.routing_loss_weight * routing_loss
```

### Emit a Final Warmup Acceptance Artifact
```python
# Source basis:
# /mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/safemoe/evaluate.py
# /mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/safemoe/observability.py

summary = {
    "pre_routing_harmful_frac_D_std": pre["routing_harmful_frac_D_std"],
    "pre_routing_harmful_frac_D_harmful": pre["routing_harmful_frac_D_harmful"],
    "post_routing_harmful_frac_D_std": post["routing_harmful_frac_D_std"],
    "post_routing_harmful_frac_D_harmful": post["routing_harmful_frac_D_harmful"],
    "d_std_ppl_delta": post_ppl["val_ppl_D_std"] - pre_ppl["val_ppl_D_std"],
    "passed": passed,
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Observation-only routing metrics from detached dispatch indices | Differentiable aggregate harmful-routing mass for training plus the same observability contract for evaluation | Phase 8 | Training can now optimize the same routing behavior that later gets measured. |
| One loss key per sampled split | Separate LM, routing, and total loss keys per labeled split | Phase 8 | Researchers can tell whether routing separation came from useful steering or LM collapse. |
| Raw checkpoints only | Final blessed warmup handoff checkpoint plus pass/fail summary | Phase 8 | Phase 9 gets one explicit resume artifact instead of an ambiguous directory list. |

**Deprecated/outdated:**
- Using `D_unlabeled` in warmup acceptance: deferred to Phase 9 by user decision.
- Treating parity/observability artifacts as sufficient warmup evidence: Phase 8 needs an explicit pre/post acceptance report.

## Open Questions

1. **What exact `D_std` regression threshold counts as material?**
   - What we know: It must be measurable, researcher-usable, and anchored on `D_std`.
   - What's unclear: The exact numeric tolerance is still discretionary.
   - Recommendation: Use a relative perplexity delta threshold in the final report, defaulting to a tight bound such as `<= 5%` unless prior baselines suggest a looser but still explicit limit.

2. **What initial routing-loss weight should the first plan lock?**
   - What we know: The objective must stay additive to LM loss, not replace it.
   - What's unclear: The exact scale that meaningfully moves routing without destabilizing LM quality.
   - Recommendation: Start with `0.1` as the baseline config value, record LM/routing losses separately, and treat any weight tuning as a follow-up only if the baseline fails acceptance.

3. **Should intermediate checkpoints also get warmup summaries, or only the final checkpoint?**
   - What we know: The final pass/fail summary and final blessed checkpoint are mandatory.
   - What's unclear: Whether intermediate summaries are worth the extra eval cost under the Phase 5 runtime envelope.
   - Recommendation: Make final-summary generation mandatory and intermediate-summary generation optional behind `train.save_interval`, so the baseline plan does not over-commit evaluation time.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | `pytest 9.0.2` |
| Config file | [pyproject.toml](/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/pyproject.toml) |
| Quick run command | `pytest tests/safemoe/test_warmup_separation.py -q` |
| Full suite command | `pytest tests/safemoe/test_pretrain.py tests/safemoe/test_evaluate.py tests/safemoe/test_warmup_separation.py -q` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| WARM-01 | Warmup samples only `D_std`/`D_harmful`, keeps LM loss active, and logs routing loss separately | integration | `pytest tests/safemoe/test_warmup_separation.py -q -k warmup_loss_logging` | ❌ Wave 0 |
| WARM-02 | Supervised routing loss pushes harmful-routing mass up for `D_harmful` and down for `D_std` | unit | `pytest tests/safemoe/test_warmup_separation.py -q -k routing_supervision` | ❌ Wave 0 |
| WARM-03 | Final report compares same-lineage pre/post checkpoints and enforces pass/fail on routing separation plus `D_std` non-regression | integration | `pytest tests/safemoe/test_warmup_separation.py -q -k warmup_acceptance_report` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/safemoe/test_warmup_separation.py -q`
- **Per wave merge:** `pytest tests/safemoe/test_pretrain.py tests/safemoe/test_evaluate.py tests/safemoe/test_warmup_separation.py -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/safemoe/test_warmup_separation.py` — covers WARM-01, WARM-02, WARM-03
- [ ] Add a tiny-model test that proves warmup sampling excludes `D_unlabeled` and logs `loss_lm_*` plus `loss_routing_*`
- [ ] Add a unit test around the differentiable harmful-routing-mass helper in `SafeMoELayer` or a new warmup helper module
- [ ] Add an integration test that writes a synthetic pre/post warmup summary JSON/Markdown artifact and verifies pass/fail logic

## Sources

### Primary (HIGH confidence)
- [safemoe/pretrain.py](/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/safemoe/pretrain.py) - Current direct-Qwen training loop, split sampling, checkpointing, and routing observability seam
- [safemoe/model.py](/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/safemoe/model.py) - Current top-k routing path and the exact place to capture differentiable harmful-routing mass
- [safemoe/observability.py](/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/safemoe/observability.py) - Canonical routing artifact schema and parity helper
- [safemoe/evaluate.py](/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/safemoe/evaluate.py) - Existing researcher-facing routing and perplexity evaluation flows
- [safemoe/data/datamodule.py](/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/safemoe/data/datamodule.py) - Current labeled/unlabeled split loader contract
- [checkpoints/Qwen3-30B-A3B-Base/model_config.yaml](/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/checkpoints/Qwen3-30B-A3B-Base/model_config.yaml) - Pinned Qwen MoE routing config (`n_expert_groups: null`, `n_expert_per_token: 8`)
- [pyproject.toml](/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/pyproject.toml) - Repo dependency pins and pytest configuration

### Secondary (MEDIUM confidence)
- https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html - Stable additive loss primitive reference for supervised routing formulations
- https://docs.pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html - Reference for explicit separation-margin formulations when planner prefers pairwise ranking semantics
- https://lightning.ai/docs/fabric/stable/guide/checkpoint/checkpoint.html - Fabric checkpoint save/load behavior consistent with staying on the current training path

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Phase 8 can stay entirely inside already-installed, already-used project dependencies.
- Architecture: HIGH - The required seams are visible in the existing code, especially the training loop and `SafeMoELayer` routing path.
- Pitfalls: HIGH - The main failure modes follow directly from locked user decisions and current code limitations.

**Research date:** 2026-03-19
**Valid until:** 2026-04-18
