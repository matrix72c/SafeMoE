# Domain Pitfalls

**Domain:** Direct harmful-transfer on `Qwen3-30B-A3B-Base` in an existing LitGPT-based SafeMoE codebase
**Researched:** 2026-03-19
**Confidence:** MEDIUM

## Critical Pitfalls

### Pitfall 1: Checkpoint surgery that is shape-correct but semantically wrong

**What goes wrong:**
Expert cloning, router-column duplication, or harmful-head copying appears to load cleanly, but the cloned tensors do not correspond to the same semantic unit in Qwen3MoE as they did in the source checkpoint. The run starts, loss decreases, and the milestone looks alive, but the initialized `theta_harmful` is not a faithful copy of the intended experts or heads.

**Why it happens:**
The existing SafeMoE registry and evaluation code assume LitGPT naming and `SafeMoELayer` internals such as `transformer.h.<layer>.mlp.experts.<idx>` and `_last_indices`. Qwen3MoE compatibility in LitGPT exists, but the milestone adds direct checkpoint intervention on a much larger external checkpoint with fused QKV weights and different router/output plumbing. “Loads without error” is a weak guarantee here.

**Prevention:**
- Add a dedicated checkpoint-parity phase before any training.
- Verify expert cloning by checksum and cosine similarity per cloned expert block, not only by `load_state_dict()`.
- Verify head cloning at row-slice level for fused QKV tensors.
- Add a one-batch parity harness that compares pre-surgery and post-surgery routing/logit deltas with zero noise and with cloning disabled.
- Persist a manifest of every copied tensor, source tensor, noise scale, and affected layer/expert/head.

**Warning signs:**
- No crash, but initial routing fractions are near-random.
- Cloned experts show unexpectedly low cosine similarity to their source experts.
- Harmful ablation has negligible effect immediately after initialization.
- Different seeds produce materially different routing separation before warmup.

**Phase to address:**
Phase 1: Qwen checkpoint import, tensor mapping, and initialization verification.

---

### Pitfall 2: Assuming the existing harmful registry cleanly extends from LitGPT MoE to Qwen3MoE

**What goes wrong:**
`theta_harmful`, `theta_std`, and `theta_shared` are incompletely or incorrectly classified once Qwen3-specific modules or wrappers are introduced. Gradient masking then updates the wrong tensors, leaving the experiment invalid while looking numerically stable.

**Why it happens:**
The local registry logic is regex-based and keyed to LitGPT parameter names. It also treats fused QKV as a special case and currently routes all non-expert parameters into `theta_shared`. That is fragile when the milestone adds router-column duplication, HF-origin weight layouts, or wrapper modules inserted by Fabric/FSDP.

**Prevention:**
- Treat registry validation as a first-class deliverable, not a helper.
- Add tests that assert exhaustive and non-overlapping classification on a real `Qwen3-30B-A3B-Base` config skeleton and on the exact wrapped model used for training.
- Add explicit coverage for router weights and any duplicated router columns.
- Snapshot parameter IDs in each bucket before and after Fabric setup.

**Warning signs:**
- Registry sizes change after `fabric.setup(model)`.
- Router weights are absent from all tracked groups or silently fall into `theta_shared`.
- Harmful and standard parameter counts do not scale with `k` experts and `n` heads.
- Resume runs produce different masked parameter sets than fresh runs.

**Phase to address:**
Phase 1: Parameter classification and masking invariants.

---

### Pitfall 3: Router supervision implemented on the wrong signal or counted twice

**What goes wrong:**
The warmup objective does not actually supervise the router the team thinks it does. Common failure modes are using post-top-k dispatch instead of pre-top-k logits, supervising only selected tokens, or accidentally adding an auxiliary MoE routing loss on top of a loss the model already includes.

**Why it happens:**
Qwen3MoE routing APIs are version-sensitive. In Transformers, router outputs are optional, and there has been at least one Qwen3MoE bug around `output_router_logits=True` with `mlp_only_layers`, plus a separate downstream report of duplicated aux-loss accounting. The milestone is also layering a custom routing margin loss on top of Qwen’s own router behavior.

**Prevention:**
- Freeze exact library versions for the warmup experiments.
- Read router tensors from a dedicated, version-tested interface and assert shape `(batch, seq, num_experts)` before loss computation.
- Disable or account for built-in router aux loss explicitly; never “just add another router term”.
- Log each loss component separately: NTP, custom routing loss, built-in aux loss, total loss.
- Unit-test one synthetic batch where the expected harmful-vs-standard margin is analytically known.

**Warning signs:**
- Routing loss changes while routing fractions do not.
- Total loss is much larger than the sum of logged components should permit.
- Turning the routing loss off barely changes gradients on router weights.
- Warmup improves train loss but does not move `D_harmful` and `D_std` dispatch apart.

**Phase to address:**
Phase 2: Router-observability and supervised warmup instrumentation.

---

### Pitfall 4: Supervising routing on tokens whose labels do not align with the actual harmful concept

**What goes wrong:**
The router learns dataset shortcuts instead of harmfulness. In bilingual TinyStories, it may separate language, punctuation, sequence length, or shard-specific formatting rather than the intended harmful/standard distinction.

**Why it happens:**
The milestone intentionally uses a controlled proxy task, but `Qwen3-30B-A3B-Base` already contains strong pretrained structure. A routing loss will exploit the easiest separable feature available unless the mixed data is balanced and the splits are adversarially checked for confounds.

**Prevention:**
- Add pre-warmup dataset audits for language distribution, length distribution, token entropy, and repeated templates across `D_std` and `D_harmful`.
- Run a “confound probe”: train the same routing loss on swapped or randomized harmful labels and measure how much separation still appears.
- Create matched mini-eval sets where language and formatting are balanced across harmful and standard splits.
- Require routing gains to survive this confound-controlled eval before allowing SGTM transfer.

**Warning signs:**
- Router separation is strongest on first tokens, punctuation, or language-specific tokens.
- Harmful routing fractions collapse when prompts are length-matched or translated.
- Randomized harmful labels still produce non-trivial routing separation.

**Phase to address:**
Phase 2: Warmup data audits and confound-control evaluation.

---

### Pitfall 5: Mixed-data SGTM transfer re-teaches harmful knowledge into standard experts

**What goes wrong:**
After warmup, SGTM-style transfer appears successful on aggregate loss, but `D_unlabeled` and shared gradients bleed harmful capability back into `theta_std` or shared parameters. Ablating `theta_harmful` then removes less than expected because the harmful behavior has diffused again.

**Why it happens:**
The project’s own milestone notes that `theta_std` already contains harmful knowledge. The current masking semantics also leave `theta_shared` trainable across splits and let `D_unlabeled` update all groups. That is a direct path for re-entanglement unless the transfer schedule and ratios are tightly controlled.

**Prevention:**
- Make “diffusion budget” a tracked metric: measure harmful-task delta after ablating only `theta_harmful`, only shared params, and both.
- Start with short SGTM runs and frequent ablation checkpoints instead of long uninterrupted transfer.
- Sweep unlabeled-data weight conservatively; do not default to large `D_unlabeled` proportions.
- Add a stop criterion based on ablated harmful retention, not only total perplexity.
- Treat shared-parameter drift as a core eval axis, not an implementation detail.

**Warning signs:**
- Original-model harmful performance rises while ablated harmful performance also rises.
- `D_std` routing stays clean, but ablation no longer removes the harmful capability.
- Shared-parameter norms or cosine drift grow much faster than cloned-expert drift.

**Phase to address:**
Phase 3: Mixed-data transfer schedule, ablation-aware checkpoints, and contamination monitoring.

---

### Pitfall 6: Declaring success from routing attribution that no longer measures true Qwen dispatch

**What goes wrong:**
Routing attribution reports a clean harmful-expert fraction, but the metric is reading a LitGPT-only side channel rather than the actual Qwen3MoE router behavior used during training. The experiment then optimizes for a broken metric.

**Why it happens:**
The local evaluation path relies on `SafeMoELayer._last_indices` forward hooks. That is tightly coupled to the current custom MoE layer implementation. Direct Qwen integration changes where dispatch is produced and whether the relevant tensor is captured before or after top-k normalization.

**Prevention:**
- Rebuild routing attribution around the actual Qwen3MoE router outputs used in training.
- Cross-check hook-based dispatch counts against explicit `output_router_logits` or equivalent internal tensors on the same batch.
- Store both pre-top-k router scores and selected expert IDs for a small eval batch.
- Add a regression test that fails if routing attribution changes after a library upgrade.

**Warning signs:**
- Routing attribution works only on the custom SafeMoE path, not on the direct Qwen path.
- Hook counts disagree with top-k expert IDs from the model outputs.
- Reported harmful fractions are too stable across seeds and checkpoints.

**Phase to address:**
Phase 2: Routing metrics parity before warmup conclusions.

---

### Pitfall 7: Adversarial-cost evaluation that measures optimizer inconvenience instead of knowledge isolation

**What goes wrong:**
The team concludes that harmful knowledge is harder to recover after ablation, but the measured “cost increase” comes from changed trainable parameter count, frozen router behavior, different initialization, or shorter re-attack budgets rather than true removal of harmful knowledge.

**Why it happens:**
Adversarial recovery is highly path-dependent. When the milestone modifies expert initialization, routing, and transfer schedule, the post-ablation model is no longer a clean apples-to-apples baseline unless attack budgets and trainable subspaces are controlled.

**Prevention:**
- Define adversarial-cost protocols before training finishes.
- Compare at least three re-attack settings: full finetune, harmful-only finetune, router-only or router+harmful finetune.
- Match steps, tokens, optimizer, LR schedule, and trainable parameter count across baseline and ablated models where possible.
- Report both attack success and compute spent to reach it.
- Include a sham ablation control where equivalent parameter mass is zeroed outside `theta_harmful`.

**Warning signs:**
- Cost increase disappears when trainable parameter count is matched.
- Attack success is highly sensitive to learning rate but not to ablation target.
- “Harder to recover” is supported by one budget point only.

**Phase to address:**
Phase 4: Adversarial-cost protocol design and control baselines.

---

### Pitfall 8: Operationally underestimating the memory and throughput cost of direct 30B-A3B intervention

**What goes wrong:**
The project spends cycles debugging model logic when the real failure is systems-level: CPU RAM exhaustion during load, optimizer-state blowup, FSDP wrapping changing names, or throughput too low to complete warmup and attack sweeps.

**Why it happens:**
`Qwen3-30B-A3B-Base` has only ~3B active parameters per token but still carries ~30B total parameters and 128 experts per MoE layer. The local LitGPT codebase already documents fragile large-model loading and FSDP behavior. Qwen community guidance also points practitioners toward Megatron-style MoE training for this size class because vanilla Transformers-style training is much slower and more memory-hungry.

**Prevention:**
- Lock the hardware plan before implementation.
- Run a dry-run on the exact target stack measuring peak CPU RAM, GPU RAM, optimizer-state footprint, and tokens/sec for init, warmup, SGTM, and adversarial re-attack.
- Separate “single-GPU functional test” and “real experiment” environments.
- Decide up front whether the milestone is inference-only surgery plus partial finetune, or true full-parameter training.
- Budget for checkpoint copies created by expert cloning and ablation outputs.

**Warning signs:**
- Model load succeeds only with swap or repeated OOM retries.
- FSDP-wrapped names differ from the names used by the registry or checkpoint manifest.
- Warmup ETA makes multi-seed comparison impossible.
- Resume checkpoints omit optimizer or RNG state needed for fair attack-cost comparisons.

**Phase to address:**
Phase 0: Resource envelope and training-stack decision.

---

### Pitfall 9: Noise-initialized harmful clones that either collapse back to standard experts or destabilize immediately

**What goes wrong:**
The cloned harmful experts do not become distinct enough to attract harmful traffic, or the added perturbation is large enough to damage pretrained behavior before router supervision can stabilize dispatch.

**Why it happens:**
This milestone depends on a narrow initialization window: enough perturbation to break symmetry, not enough to destroy the copied expert/head prior. Large sparse MoE checkpoints are especially sensitive because routing and expert specialization co-adapt.

**Prevention:**
- Treat initialization noise scale as a primary experiment variable.
- Sweep from exact copy through very small perturbations before trying larger noise.
- Measure immediate post-init expert output similarity and first-step router entropy.
- Keep a zero-noise control and a random-init harmful expert control.

**Warning signs:**
- Harmful experts remain unused after several warmup checkpoints.
- Router entropy collapses or spikes immediately after initialization.
- `D_std` perplexity degrades before any meaningful routing separation appears.

**Phase to address:**
Phase 1: Harmful clone initialization study.

---

### Pitfall 10: Reproducibility gaps that make the milestone non-falsifiable

**What goes wrong:**
A promising harmful-isolation result cannot be trusted because reruns change expert selection, routing trajectories, or adversarial-cost outcomes. The team then debates interpretations instead of the thesis.

**Why it happens:**
This milestone compounds several stochastic sources: sampled harmful experts/heads, initialization noise, mixed-data sampling, FSDP wrapping order, and attack finetuning. The current codebase already notes deferred eval runs and resume-state sensitivity in SGTM training.

**Prevention:**
- Persist the selected harmful experts, head indices, router-column copies, and noise seed in checkpoint metadata.
- Save RNG state for Python, PyTorch CPU, and CUDA at each checkpoint used for eval.
- Make split-sampling weights and attack budgets part of the run ID.
- Require at least two seeds for any claim that influences roadmap direction.

**Warning signs:**
- Same config produces different harmful expert sets or materially different routing curves.
- Resume checkpoints diverge from uninterrupted runs within a few hundred steps.
- Only one seed shows the claimed isolation effect.

**Phase to address:**
Phase 5: Reproducibility, multi-seed verification, and result hardening.

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Reusing `SafeMoELayer`-only hooks as the Qwen routing metric | Fastest way to get numbers | Optimizes against the wrong dispatch signal | Never |
| Skipping manifest logging for cloned experts/heads | Less implementation work | Cannot audit whether initialization matched intent | Never |
| Folding custom routing loss into total loss without separate logging | Cleaner training loop | Impossible to debug router supervision failures | Never |
| Using one “good-looking” seed for phase decisions | Saves compute | Experimental conclusions become non-falsifiable | Never |
| Running full adversarial recovery without sham controls | Faster headline result | Cannot tell isolation from generic damage | Never |
| Prototyping on reduced layers/experts without a parity gate back to real Qwen config | Faster local iteration | Silent config-specific bugs survive into the real run | Acceptable only in Phase 0 if parity tests are mandatory afterward |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Hugging Face Qwen3MoE config vs LitGPT config | Assuming fields map 1:1 without auditing `padded_vocab_size`, fused QKV layout, and MoE settings | Use a parity harness against the exact `Qwen3-30B-A3B-Base` config and converted weights |
| Router instrumentation | Hooking selected expert IDs only, then treating that as the training loss target | Capture router logits/scores and selected IDs separately; supervise the intended tensor explicitly |
| Fabric/FSDP wrapping | Building registries before and after wrapping without checking name stability | Snapshot names and parameter IDs across wrapping boundaries and fail on drift |
| Ablation evaluation | Zeroing only cloned experts and calling the result “harmful removed” | Add shared-only and sham-ablation controls |
| Resume logic | Restoring model weights but not RNG or optimizer state | Save and validate full experiment state for every checkpoint used in comparisons |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Treating 30B-A3B like a 3B dense model because only ~3B params are active | CPU load stalls, optimizer OOM, extremely slow warmup | Budget around total parameters and optimizer state, not active params only | Breaks immediately on real training runs |
| Logging full router tensors for long sequences | Disk spikes, host OOM, unusable eval latency | Sample short fixed eval batches and aggregate routing stats online | Breaks as soon as 128K-context or many-layer logging is attempted |
| Delaying ablation eval until end of transfer | Need to rerun long jobs to find when contamination began | Run short periodic ablation-aware checkpoints | Breaks once SGTM spans enough steps that diffusion source is untraceable |
| Running adversarial re-attacks with unconstrained budgets | Evaluation dominates project time and produces incomparable results | Fix a ladder of bounded budgets before launch | Breaks when comparing checkpoints across phases |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Loading public HF checkpoints with unrestricted pickle assumptions in ad hoc scripts | Arbitrary-code risk in research environments | Prefer `safetensors` paths where possible and document trust assumptions for `torch.load()` |
| Exporting harmful-transfer checkpoints without metadata on ablation state | Unsafe reuse or misinterpretation downstream | Persist whether a checkpoint is original, cloned, warmup, transfer, or ablated |
| Treating proxy harmful data as harmless operationally | Underestimating governance needs when phase later swaps in stronger harmful corpora | Keep data contracts explicit and separate proxy-safe phases from future real-harmful phases |

## "Looks Done But Isn't" Checklist

- [ ] **Checkpoint cloning:** Verify tensor-level source/target manifest, not just successful load.
- [ ] **Router supervision:** Verify the supervised tensor is the true router output used by Qwen3MoE in the pinned library version.
- [ ] **Warmup success:** Verify routing separation survives confound-controlled eval, not only raw split labels.
- [ ] **Transfer success:** Verify harmful capability drops after `theta_harmful` ablation, not only that the unablated model improves.
- [ ] **Adversarial-cost result:** Verify against matched-budget and sham-ablation controls.
- [ ] **Reproducibility:** Verify at least two seeds and checkpoint-resume consistency.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Bad checkpoint surgery | MEDIUM | Rebuild from untouched base checkpoint, replay manifest generation, rerun parity harness before training |
| Wrong router-loss wiring | MEDIUM | Freeze training, dump one-batch router tensors, validate loss on synthetic data, relaunch warmup only |
| Harmful knowledge diffusion during SGTM | HIGH | Roll back to earliest ablation-clean checkpoint, reduce `D_unlabeled` weight, shorten intervals between ablation evals |
| Broken adversarial-cost conclusion | MEDIUM | Re-run attack suite with matched budgets and sham controls; do not patch the conclusion post hoc |
| Resource envelope failure | HIGH | Downscope to functional parity only, or switch the experiment stack before spending more debugging time |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Checkpoint surgery mismatch | Phase 1 | Manifest diff, tensor cosine checks, one-batch parity run |
| Registry/masking misclassification | Phase 1 | Exhaustive non-overlap tests on wrapped model |
| Router loss on wrong signal / double counting | Phase 2 | Synthetic router-loss test plus separated loss logging |
| Dataset confounds masquerading as harmful routing | Phase 2 | Confound-controlled eval and randomized-label probe |
| Mixed-data diffusion back into standard/shared params | Phase 3 | Periodic ablation checkpoints and shared-drift metrics |
| Routing attribution not measuring true Qwen dispatch | Phase 2 | Hook-vs-output parity test on the same batch |
| Adversarial-cost protocol drift | Phase 4 | Matched-budget, matched-trainable-space attack suite |
| Resource envelope failure | Phase 0 | Dry-run peak RAM/VRAM/tokens-per-sec report |
| Clone-noise instability | Phase 1 | Noise sweep with zero-noise and random-init controls |
| Reproducibility gaps | Phase 5 | Two-seed minimum and resume-vs-fresh consistency check |

## Sources

- Project context: `.planning/PROJECT.md`
- Local codebase concerns: `.planning/codebase/CONCERNS.md`
- Local testing patterns: `.planning/codebase/TESTING.md`
- Qwen3 official blog, model sizes and MoE architecture: https://qwenlm.github.io/blog/qwen3/
- `Qwen/Qwen3-30B-A3B-Base` config, official HF model config: https://huggingface.co/Qwen/Qwen3-30B-A3B-Base/blob/main/config.json
- Hugging Face Transformers Qwen3MoE docs: https://huggingface.co/docs/transformers/model_doc/qwen3_moe
- Transformers issue on Qwen3MoE router logits bug with `output_router_logits=True`: https://github.com/huggingface/transformers/issues/39203
- TRL issue noting possible aux-loss double counting for Qwen3MoE: https://github.com/huggingface/trl/issues/4070
- Qwen community guidance on Qwen3-MoE training scale and Megatron-style best practices: https://github.com/QwenLM/Qwen3/issues/1278

---
*Pitfalls research for: direct harmful-transfer on `Qwen3-30B-A3B-Base`*
*Researched: 2026-03-19*
