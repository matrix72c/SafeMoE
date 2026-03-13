# Domain Pitfalls

**Domain:** MoE knowledge isolation / SGTM (Selective Gradient/Token Masking)
**Researched:** 2026-03-13
**Overall confidence:** MEDIUM-HIGH (training data only; no live verification possible due to network restrictions)

---

## Critical Pitfalls

Mistakes that cause rewrites, invalidate experiments, or produce misleading results.

---

### Pitfall 1: Gradient Hook Ordering and In-Place Mutation Breaking Autograd

**What goes wrong:** SGTM requires zeroing out gradients of `theta_std` after backward on `D_harmful` samples. The naive approach -- registering `register_full_backward_hook` on modules or `register_hook` on parameters to zero gradients in-place -- interacts unpredictably with PyTorch's autograd graph. Hooks registered on tensors execute in reverse order of registration, but module-level hooks fire at module boundaries. If gradient accumulation is enabled (which LitGPT uses), zeroing gradients mid-accumulation via hooks destroys previously accumulated gradients from other data splits.

**Why it happens:** PyTorch hooks operate on the autograd graph, not the training loop. A backward hook sees the gradient *for that backward pass*, but `param.grad` may already contain accumulated gradients from prior microbatches. In-place modification of `param.grad` inside a hook will also modify the accumulated value.

**Consequences:**
- Gradients from `D_std` or `D_unlabeled` microbatches are silently zeroed if they were accumulated before the `D_harmful` backward pass.
- Training appears to proceed normally (no errors), but `theta_std` parameters barely update because their gradients keep getting wiped.
- Detected only when the model fails to learn general capabilities -- potentially days into a run.

**Prevention:**
- Do NOT use backward hooks for gradient masking. Instead, perform gradient masking as an explicit post-backward step in the training loop: `loss.backward()`, then iterate over parameters and zero the ones you want to mask.
- Keep the three data splits (`D_harmful`, `D_std`, `D_unlabeled`) in separate forward-backward calls within each training step, with explicit gradient manipulation between them.
- If using gradient accumulation, accumulate separately per split, then apply masks before `optimizer.step()`.

**Detection:**
- Monitor `theta_std` gradient norms per training step. If they are near zero despite `D_std` and `D_unlabeled` samples being present, hooks are clobbering accumulated gradients.
- Log gradient norms separately for each parameter group (harmful vs. standard) after each backward pass.

**Phase:** Milestone 1 (SGTM training loop implementation). This must be designed correctly from the start.

---

### Pitfall 2: Router Collapse -- All Tokens Route to the Same Expert(s)

**What goes wrong:** The Top-K router learns to send nearly all tokens to a small subset of experts (often 1-2), while the remaining experts receive negligible traffic and stop learning. In the SafeMoE context, this is catastrophic: if the router collapses to always select standard experts, harmful knowledge bleeds into `theta_std`; if it collapses to always select harmful experts, the model cannot learn general capabilities.

**Why it happens:** Rich-get-richer dynamics. An expert that initially receives slightly more tokens trains faster, produces better representations, and thus gets higher router scores, attracting even more tokens. Without explicit load-balancing pressure, this positive feedback loop drives collapse within the first few hundred steps. The LitGPT `LLaMAMoE` implementation (line 776-819 of `model.py`) does NOT include any load-balancing auxiliary loss.

**Consequences:**
- Unused experts have effectively random weights -- they never update meaningfully.
- The model degenerates to a dense model with wasted parameters.
- For SafeMoE specifically: router collapse means "harmful experts" may never get trained because the router never routes to them, or standard experts absorb everything including harmful knowledge.

**Prevention:**
- Implement an auxiliary load-balancing loss (Switch Transformer style): `L_balance = alpha * N * sum_i(f_i * P_i)` where `f_i` is the fraction of tokens routed to expert `i` and `P_i` is the average router probability for expert `i`. Typical alpha: 0.01-0.1.
- Log per-expert token counts every N steps. If any expert receives < 5% of its fair share (1/N_experts) for more than ~100 consecutive steps, raise an alert.
- Consider using expert choice routing (tokens don't choose experts; experts choose tokens) as a fallback if top-K proves unstable at small scale.
- Note: for SafeMoE, the load-balancing loss must be aware of the harmful/standard split. You may want separate balance targets for harmful vs. standard experts, or disable balance loss for harmful experts during `D_harmful` forward passes (since you *want* harmful tokens to concentrate there).

**Detection:**
- Expert utilization histograms (per-expert token count per batch). Collapse shows as one expert at ~100% and others at ~0%.
- Router entropy: compute entropy of the softmax router distribution. Low entropy = collapse.
- Track these from step 0. Collapse typically manifests within the first 200-500 steps.

**Phase:** Milestone 1 (MoE architecture). Must be addressed at architecture design time, not bolted on later.

---

### Pitfall 3: SGTM Forward-Pass Masking Breaks Gradient Flow Through the Router

**What goes wrong:** During `D_std` samples, SGTM requires zeroing `theta_harmful` activations in the forward pass. If implemented by zeroing expert outputs *after* the router dispatches to them, the router still receives gradient signal from the harmful experts (through the softmax weighting). But if implemented by preventing the router from selecting harmful experts entirely (e.g., masking router logits to -inf for harmful experts), the router never learns to route harmful content to harmful experts because it never practices doing so on `D_std` data.

**Why it happens:** There is a fundamental tension: the forward-pass mask is supposed to prevent harmful experts from influencing `D_std` predictions, but the router needs to eventually learn to route harmful tokens to harmful experts (for `D_harmful` and `D_unlabeled` samples). If the router never sees harmful experts activated during `D_std` forward passes, its routing weights for those experts stagnate.

**Consequences:**
- If you mask at the output level: harmful expert parameters get gradient updates from `D_std` backward (they participated in forward), violating the SGTM invariant.
- If you mask at the router level: the router's weights for harmful expert selection never improve from `D_std` data, leading to poor routing on `D_unlabeled` data.
- Either way, the clean separation between harmful and standard knowledge is compromised.

**Prevention:**
- Mask at the output level (zero expert outputs post-computation), but also zero the gradients flowing back through the harmful experts during `D_std` backward. This requires either:
  (a) Wrapping harmful expert outputs with `tensor.detach()` before adding to the sum during `D_std` forward, OR
  (b) A custom autograd function that passes activations forward but blocks gradients backward for designated experts.
- Option (a) is simpler and sufficient: `output = harmful_expert(x).detach() * 0.0` effectively contributes nothing to forward or backward while maintaining the computation graph for the router logits (which still routes to that expert but gets zero contribution).
- Actually, the cleanest approach: during `D_std` forward, simply skip the harmful experts entirely in the dispatch loop (do not call `expert(x[token_idx])` for harmful experts). This avoids wasted computation and avoids gradient issues. The router scores for harmful experts still get softmax gradient signal from the other experts' outputs.

**Detection:**
- During `D_harmful` evaluation, check if harmful tokens route to harmful experts. If routing is random (near uniform), the router has not learned the harmful/standard distinction.
- Monitor harmful expert parameter norms: if they update during `D_std` passes, your masking is leaking.

**Phase:** Milestone 1 (SGTM training loop). Core algorithm design decision.

---

### Pitfall 4: The Bilingual Proxy Does Not Actually Validate Knowledge Isolation

**What goes wrong:** Using English vs. Spanish in TinyStories as a proxy for "harmful vs. safe" knowledge seems clean, but language identity is encoded at the embedding/early-layer level, not the MLP-expert level. The router can trivially separate languages by surface-level token features (Spanish has accents, different character distributions, different subword tokenization). This does NOT validate that the system can isolate *semantic* harmful knowledge that shares surface features with safe knowledge.

**Why it happens:** Language is one of the easiest features for a transformer to detect -- it is a low-level distributional signal. Real harmful knowledge (e.g., bioweapon synthesis instructions) shares vocabulary, syntax, and grammar with safe knowledge (e.g., chemistry textbook content). A system that perfectly isolates Spanish from English may completely fail to isolate harmful chemistry from safe chemistry.

**Consequences:**
- Milestone 1 passes with flying colors, but the approach fails on Milestone 2/3 with real harmful data.
- Wasted months building on a validated proxy that does not transfer.
- The team develops false confidence in the approach.

**Prevention:**
- Acknowledge this limitation explicitly in Milestone 1 goals: it validates the *mechanism* (gradient masking, expert ablation, training loop), not the *difficulty* of semantic isolation.
- Add a secondary proxy that is harder: e.g., topic-based isolation (isolate "cooking recipes" from "science articles" -- both in English, requiring semantic routing).
- Design Milestone 1 evaluation to measure routing *mechanism correctness*, not just final perplexity numbers. Specifically: verify that gradient masking actually prevents `theta_std` updates during `D_harmful`, and that forward masking actually prevents `theta_harmful` from influencing `D_std` predictions.
- Plan for Milestone 2 to be a much harder test, and do not skip it even if Milestone 1 succeeds perfectly.

**Detection:**
- If router accuracy for language separation is > 95% after just a few hundred steps, the task is too easy to be informative about semantic isolation.
- Compare router decisions on English vs. Spanish against a simple unigram language detector. If they correlate > 0.9, the router is doing shallow language detection, not deep knowledge routing.

**Phase:** Milestone 1 (experiment design). Must be discussed before starting experiments, not after.

---

### Pitfall 5: Catastrophic Forgetting During CPT When Adding Harmful Experts

**What goes wrong:** In Milestone 2, you add new (randomly initialized) harmful experts to a pretrained model and then continue pretraining. The router must now learn to integrate these new experts while preserving the pretrained model's capabilities. But gradient updates to the router weights (to learn to use the new experts) simultaneously destabilize routing to the existing experts, causing catastrophic forgetting of pretrained knowledge.

**Why it happens:** The router is a linear layer mapping hidden states to expert scores. Adding new experts extends the output dimension of the router. Even if you initialize the new expert router weights to near-zero, the softmax/sigmoid normalization changes the probability landscape for existing experts. The first few gradient updates to the router (which are large because the new experts are randomly initialized and produce large losses) perturb the established routing patterns.

**Consequences:**
- Pretrained model capabilities degrade within the first few hundred CPT steps.
- The model "forgets" before it "learns" to isolate harmful knowledge.
- Evaluation shows both general and harmful perplexity are bad.

**Prevention:**
- Freeze the router weights for existing experts during early CPT (first 5-10% of training). Only train the new router columns (for harmful experts) and the new expert weights.
- Use a much lower learning rate for the router (10-100x lower than expert learning rate).
- Initialize harmful expert weights by cloning one of the existing experts (warm start) rather than random initialization. This gives the router a reasonable starting point.
- Add harmful experts to every MoE layer simultaneously (not incrementally), so the model can learn to route consistently across layers.
- Monitor pretrained task performance (English perplexity on a held-out set) every N steps during early CPT. If it degrades by more than 10%, reduce the router learning rate.

**Detection:**
- Track pretrained benchmark performance (e.g., English perplexity on TinyStories) continuously during CPT. Any sudden spike indicates forgetting.
- Monitor routing patterns for existing experts: if the token distribution shifts dramatically for `theta_std` experts, the router is being destabilized.

**Phase:** Milestone 2 (CPT routing). This is the central risk of the CPT approach.

---

### Pitfall 6: Gradient Masking Creates Inconsistent Optimizer State

**What goes wrong:** Adam/AdamW maintains per-parameter first and second moment estimates (m and v). When SGTM zeros gradients for `theta_std` during `D_harmful` passes, these parameters receive a gradient of exactly zero (not "no gradient" -- zero). Adam interprets this as "the gradient is genuinely zero," which decays the momentum toward zero and inflates the adaptive learning rate (because v shrinks). This creates oscillatory training dynamics where `theta_std` parameters alternate between being aggressively updated (large effective LR from deflated v) and being zeroed out.

**Why it happens:** Adam's update rule is `m = beta1 * m + (1-beta1) * g` and `v = beta2 * v + (1-beta2) * g^2`. When `g = 0` (from gradient masking), `m` decays and `v` decays, causing the effective learning rate `m / (sqrt(v) + eps)` to oscillate as the moments repeatedly inflate and deflate across masked and unmasked steps.

**Consequences:**
- Training instability: `theta_std` parameters oscillate rather than converging smoothly.
- Potential divergence if the effective learning rate spikes after several consecutive masked steps.
- Subtle enough that loss curves may look okay but parameter updates are inefficient.

**Prevention:**
- Option A: Skip the optimizer step for masked parameters entirely. Instead of setting gradients to zero, set them to `None` and use `foreach=False` in the optimizer to handle sparse updates. PyTorch Adam skips the update for parameters with `None` grad (since PyTorch 1.7+), preserving m and v.
- Option B: Use separate optimizers for `theta_harmful` and `theta_std` parameter groups. The `theta_std` optimizer only steps when `D_std` or `D_unlabeled` data is processed; the `theta_harmful` optimizer only steps when `D_harmful` or `D_unlabeled` data is processed.
- Option B is cleaner and more explicit. It also naturally supports different learning rates for the two parameter groups.
- Note: LitGPT's `pretrain.py` uses a single optimizer with `instantiate_torch_optimizer`. You will need to modify this to support two optimizer groups.

**Detection:**
- Monitor the ratio of Adam's `v` (second moment) for `theta_std` parameters. If it periodically drops to near-eps, the optimizer state is being corrupted by zero gradients.
- Compare training dynamics with and without separate optimizers on a short run (100-200 steps). If separate optimizers produce smoother loss curves, the single-optimizer approach is causing issues.

**Phase:** Milestone 1 (SGTM training loop). Must be decided before implementing the training loop.

---

### Pitfall 7: Evaluation Confound -- Ablation Perplexity Measures Router, Not Expert Knowledge

**What goes wrong:** The primary SafeMoE evaluation metric is: zero out harmful experts, measure that harmful perplexity increases (knowledge removed) and general perplexity stays flat (general knowledge intact). But perplexity after ablation does NOT solely measure whether knowledge was stored in the ablated experts. It also measures whether the router can gracefully redistribute tokens to remaining experts. If the router has learned to heavily depend on harmful experts (even for some general tokens), ablation will degrade general perplexity regardless of where the knowledge actually lives.

**Why it happens:** In a trained MoE, the router and experts co-adapt. The router learns to send certain activation patterns to certain experts, and the experts specialize to handle those patterns. Removing an expert is not like removing a drawer from a filing cabinet -- it is like removing a player from a team. The remaining players cannot seamlessly absorb the missing player's role without retraining.

**Consequences:**
- General perplexity degrades after ablation even when knowledge isolation is perfect, leading to a false conclusion that isolation failed.
- Alternatively: harmful perplexity does not increase much after ablation because the router already learned to route harmful tokens to multiple experts (including standard ones) for robustness, leading to a false conclusion that harmful knowledge leaked into standard experts.

**Prevention:**
- Complement ablation perplexity with routing attribution analysis: for each token, log which expert(s) it was routed to. Compute the fraction of harmful tokens routed exclusively to harmful experts. This is a more direct measure of isolation than post-ablation perplexity.
- Use probing classifiers: train a linear probe on the intermediate representations of each expert to predict whether the input is harmful. If harmful knowledge is isolated, only harmful experts should have high probe accuracy.
- Report ablation perplexity alongside routing statistics, not as the sole metric.
- Consider fine-tuning only the router for a few steps after ablation (freeze all expert weights, only update router to redistribute traffic away from the removed expert). If general perplexity recovers after this fine-tuning, the degradation was due to routing, not knowledge leakage.

**Detection:**
- If general perplexity degrades proportionally to the fraction of tokens that were routed to harmful experts (regardless of token type), the degradation is routing-dependent, not knowledge-dependent.
- Compare ablation perplexity against a baseline where you ablate a random (non-harmful) expert of similar utilization. If the degradation is similar, ablation is measuring routing disruption, not knowledge removal.

**Phase:** Milestone 1 (evaluation suite design). Must be designed before running experiments, or you will not have the routing logs needed for proper analysis.

---

## Moderate Pitfalls

---

### Pitfall 8: Mixed-Batch Data Sampling Creates Training Artifacts

**What goes wrong:** SGTM requires three different data types (`D_harmful`, `D_std`, `D_unlabeled`) with different masking behavior. Naively interleaving these in a single batch (or randomly sampling one type per step) creates high variance in gradient estimates and unpredictable training dynamics. If one step sees only `D_harmful` and the next sees only `D_std`, the model oscillates between two conflicting objectives.

**Prevention:**
- Process all three splits within each training step (not alternating between steps). Each step should include a forward-backward for each split, with gradient accumulation, then a single optimizer step.
- Maintain fixed ratios per step: e.g., 40% `D_std`, 10% `D_harmful`, 50% `D_unlabeled` (tunable).
- If micro-batch sizes are too small to represent all splits, use a round-robin schedule with a period of 3 steps, but accumulate gradients across the full period before stepping.

**Detection:**
- High loss variance across consecutive steps (compared to a baseline without masking).
- Router decisions oscillating between steps (routing the same tokens to different experts on even vs. odd steps).

**Phase:** Milestone 1 (data pipeline and training loop).

---

### Pitfall 9: Shared Experts and Attention Heads Leak Knowledge Across the Boundary

**What goes wrong:** SGTM focuses on MoE expert isolation, but transformer blocks have other shared components: attention heads, layer norms, embedding layers, and the `lm_head`. These shared parameters see ALL data during training (harmful and standard). If harmful knowledge is partially encoded in attention patterns or embeddings (which is likely for language models), ablating MoE experts will not fully remove it.

**Prevention:**
- The PROJECT.md mentions designating "specific experts and attention heads as `theta_harmful`." This is correct -- attention heads must also be partitioned. But implementation is harder: attention heads in LitGPT are computed via a single `attn` projection matrix, not separate head modules. You need to mask at the head-output level.
- Embeddings (wte) and lm_head are fundamentally shared. Accept that token-level knowledge (e.g., knowing that a specific token exists) cannot be isolated. Focus isolation on *compositional* harmful knowledge (knowing how to combine tokens into harmful sequences).
- Layer norms and position encodings are small enough in parameter count that leakage through them is negligible -- but verify this assumption empirically.

**Detection:**
- After expert ablation, use a probing classifier on attention head outputs to check for residual harmful knowledge.
- Compute the "knowledge delta" by measuring harmful perplexity with all components vs. with only shared components (experts zeroed for both harmful and standard). The difference tells you how much harmful knowledge lives in shared components.

**Phase:** Milestone 1 (architecture design for `theta_harmful` designation).

---

### Pitfall 10: Scale Mismatch Between PT and CPT Experiments

**What goes wrong:** Milestone 1 uses a small custom MoE on TinyStories (small scale, from scratch). Milestones 2-3 use a pretrained model with added experts (larger scale, continued pretraining). MoE behavior is notoriously scale-dependent: routing dynamics, expert specialization, and load balancing all behave differently at different scales. Techniques that work at small scale (e.g., a specific load-balance loss coefficient) may fail at larger scale.

**Prevention:**
- Document all hyperparameters that are scale-sensitive: load-balance loss coefficient, router learning rate, expert capacity factor, batch size, number of experts.
- When transitioning from Milestone 1 to 2, plan a hyperparameter search for at least the load-balance coefficient and router learning rate.
- Use Milestone 1 to validate the *mechanism* (gradient masking works, forward masking works, ablation pipeline works), not to lock in final hyperparameters.
- Test Milestone 1 at two different model sizes (e.g., 2-layer and 6-layer) to get a signal on scale sensitivity.

**Detection:**
- If Milestone 1 results are sensitive to model size (2-layer vs. 6-layer show qualitatively different routing patterns), expect significant changes at CPT scale.

**Phase:** Transition between Milestone 1 and Milestone 2.

---

### Pitfall 11: Token-Level Routing Attribution Is Misleading for Sequence-Level Knowledge

**What goes wrong:** SafeMoE evaluates isolation by checking which expert each *token* routes to. But harmful knowledge is a *sequence-level* property -- "how to synthesize X" is distributed across dozens of tokens. A harmful sequence might route 60% of tokens to harmful experts and 40% to standard experts, yet the standard experts' contribution to those 40% of tokens could still carry critical harmful information (e.g., key quantities or ingredient names).

**Prevention:**
- Complement token-level routing analysis with sequence-level metrics: for a harmful sequence, what fraction of total compute (weighted by router probability) went through harmful experts?
- Design evaluation to test *generation* capabilities, not just perplexity: after ablation, can the model *complete* a harmful prompt? Perplexity measures average prediction quality; generation tests whether the model can produce coherent harmful content.
- Consider information-theoretic measures: mutual information between expert activations and harmful content, rather than simple routing counts.

**Detection:**
- Token-level routing shows 70%+ harmful tokens going to harmful experts, but the model can still generate harmful completions after ablation. This means the remaining 30% routing + shared components carry enough information.

**Phase:** Milestone 1 (evaluation suite) and Milestone 2 (validation with real harmful data).

---

### Pitfall 12: LitGPT's Training Loop Is Not Designed for Multi-Objective Steps

**What goes wrong:** LitGPT's `pretrain.py` has a standard training loop: fetch batch, forward, backward, step. SGTM requires three separate forward-backward passes per step (one per data split) with different masking behavior. Trying to shoehorn this into the existing loop by modifying the data pipeline (mixing split types within a batch) loses the ability to apply per-split masking.

**Prevention:**
- Write a custom training loop (`sgtm_pretrain.py`) rather than patching `pretrain.py`. The SGTM loop has fundamentally different structure: it needs to iterate over data splits within each step.
- Reuse LitGPT utilities (checkpointing, logging, gradient accumulation math) but not the main training loop.
- Keep the custom loop as close to LitGPT's patterns as possible (Fabric, same optimizer setup, same logging) to ease maintenance.

**Detection:**
- If you find yourself writing increasingly complex conditional logic inside the existing training loop (`if split_type == 'harmful': ...`), it is time to fork the loop.

**Phase:** Milestone 1 (training loop implementation). Decide immediately whether to patch or fork.

---

## Minor Pitfalls

---

### Pitfall 13: Softmax Router Normalization Changes With Expert Count

**What goes wrong:** When adding harmful experts in Milestone 2, the router's output dimension increases. If using softmax normalization (as in the simple `nn.Linear` gate on line 802 of model.py), the softmax temperature effectively changes with more experts, altering routing behavior for existing experts even before any training occurs.

**Prevention:**
- Use sigmoid normalization (as in the `GroupedTopkRouter` on line 857) instead of softmax. Sigmoid scores are independent per expert and do not change when new experts are added.
- If using softmax, apply temperature scaling that accounts for expert count: `softmax(logits / sqrt(n_experts))`.

**Phase:** Milestone 1 (architecture choice that propagates to Milestone 2).

---

### Pitfall 14: Checkpoint Compatibility Between Base LitGPT and SafeMoE Models

**What goes wrong:** SafeMoE modifies the model architecture (adds MoE layers, adds harmful expert designation metadata). Loading a SafeMoE checkpoint with vanilla LitGPT (or vice versa) will fail with key mismatch errors. For Milestone 2, you need to load a pretrained LitGPT checkpoint into a SafeMoE model with added experts, requiring careful state_dict surgery.

**Prevention:**
- Write explicit checkpoint conversion utilities (load pretrained dense weights into MoE standard experts, initialize harmful experts separately).
- Store harmful expert designation metadata alongside the checkpoint (which expert indices are "harmful"), not in the model code.
- Test checkpoint round-tripping (save SafeMoE -> load SafeMoE -> verify identical outputs) early.

**Phase:** Milestone 1 (checkpoint utilities, needed before Milestone 2).

---

### Pitfall 15: Numerical Instability in Router With Mixed Precision (bf16)

**What goes wrong:** LitGPT defaults to bf16 precision. The router's softmax/sigmoid computation can be numerically unstable in bf16, especially for large expert counts. Small logit differences between experts get rounded to zero, causing ties that are broken arbitrarily. The `GroupedTopkRouter` already casts to float32 (line 856-857), but the simple `nn.Linear` gate does not.

**Prevention:**
- Always compute router logits and scoring in float32, even when the rest of the model runs in bf16/mixed precision. Cast back to model dtype after top-K selection.
- The existing `GroupedTopkRouter` already does this correctly. If building a custom simpler router for Milestone 1, copy this pattern.

**Phase:** Milestone 1 (architecture implementation).

---

### Pitfall 16: D_unlabeled Ratio Dominates Training Signal

**What goes wrong:** In the SGTM setup, `D_unlabeled` samples get standard (unmasked) forward-backward passes. If `D_unlabeled` is the majority of training data (which is the realistic scenario -- most text is unlabeled), its gradient signal dominates, and the selective masking from `D_harmful` and `D_std` has minimal influence on the final model. The router learns primarily from `D_unlabeled` and the harmful/standard separation is weak.

**Prevention:**
- Oversample `D_harmful` and `D_std` relative to their natural frequency. Use a configurable sampling ratio (e.g., 1:1:2 for harmful:std:unlabeled rather than natural frequency).
- Monitor gradient norms from each split separately. If `D_unlabeled` gradients are 10x larger than the masked splits, rebalance.
- Consider curriculum learning: start with higher `D_harmful`/`D_std` ratios to establish routing patterns, then increase `D_unlabeled` proportion.

**Phase:** Milestone 1 (data pipeline design) and Milestone 3 (unlabeled corpus integration).

---

## Phase-Specific Warnings

| Phase / Milestone | Likely Pitfall | Mitigation | Severity |
|---|---|---|---|
| M1: MoE Architecture | Router collapse (#2) | Implement load-balancing loss from day one; monitor per-expert token counts | Critical |
| M1: SGTM Training Loop | Gradient hook corruption (#1) | Post-backward explicit masking, not hooks; separate optimizers (#6) | Critical |
| M1: SGTM Training Loop | Forward mask gradient leak (#3) | Skip harmful expert dispatch during D_std, not just zero output | Critical |
| M1: SGTM Training Loop | Optimizer state corruption (#6) | Two optimizer groups or set grad to None (not zero) | Critical |
| M1: SGTM Training Loop | Multi-objective loop complexity (#12) | Fork training loop rather than patch LitGPT's | Moderate |
| M1: Data Pipeline | Mixed-batch artifacts (#8) | Process all three splits per step with fixed ratios | Moderate |
| M1: Data Pipeline | D_unlabeled dominance (#16) | Configurable oversampling of labeled splits | Moderate |
| M1: Evaluation Suite | Ablation measures routing not knowledge (#7) | Add routing attribution + probing classifiers alongside perplexity | Critical |
| M1: Evaluation Suite | Token vs. sequence attribution (#11) | Sequence-level compute fraction + generation tests | Moderate |
| M1: Experiment Design | Proxy validity (#4) | Acknowledge limitations; add harder secondary proxy | Critical |
| M1: Architecture | Shared component leakage (#9) | Partition attention heads; measure shared component knowledge | Moderate |
| M1: Architecture | Softmax normalization shift (#13) | Use sigmoid routing (GroupedTopkRouter style) | Minor |
| M1: Architecture | bf16 router instability (#15) | Router computation in float32 | Minor |
| M1: Checkpointing | Checkpoint compatibility (#14) | Build conversion utilities early | Minor |
| M1 -> M2 Transition | Scale mismatch (#10) | Test M1 at two scales; plan hyperparam search for M2 | Moderate |
| M2: CPT Routing | Catastrophic forgetting (#5) | Freeze existing router weights initially; lower router LR | Critical |
| M3: CPT Transfer | D_unlabeled dominance (#16) | Curriculum learning; oversample labeled data early | Moderate |

---

## Interaction Effects

Several pitfalls interact and compound each other:

1. **Pitfalls #1 + #6 (Gradient hooks + Optimizer state):** If gradient masking is implemented via hooks AND a single optimizer is used, the interaction produces both gradient corruption and optimizer state corruption simultaneously. The training will appear to work (loss decreases) but `theta_std` parameters will be undertrained. This is the most dangerous combination because both issues are silent.

2. **Pitfalls #2 + #3 (Router collapse + Forward mask gradient leak):** If the router collapses AND the forward mask leaks gradients, harmful knowledge gets trained into the collapsed (standard) experts via `D_std` backward passes. The result is the exact opposite of what SafeMoE intends: harmful knowledge concentrated in standard experts.

3. **Pitfalls #4 + #7 (Proxy validity + Evaluation confound):** If the bilingual proxy is too easy AND the evaluation only measures ablation perplexity, you get a doubly misleading result: the system appears to perfectly isolate knowledge, but it is actually just doing trivial language detection with evaluation that cannot distinguish routing disruption from knowledge removal.

**Recommendation:** Address pitfalls #1, #2, #3, #6, and #7 as the highest priority in Milestone 1. Each has the potential to silently invalidate all experimental results.

---

## Sources

- PyTorch autograd hook semantics: PyTorch documentation on `Tensor.register_hook` and `Module.register_full_backward_hook`. Key behavior: hooks fire per-backward-call, not per-optimizer-step. [Training data, MEDIUM confidence]
- Switch Transformer (Fedus et al., 2021): Documents router collapse and the auxiliary load-balancing loss. Eq. 4 defines the balance loss. [Training data, HIGH confidence -- well-established paper]
- ST-MoE (Zoph et al., 2022): Detailed analysis of MoE training instability, router z-loss for stability. [Training data, HIGH confidence]
- GShard (Lepikhin et al., 2020): Expert capacity factor and overflow handling. [Training data, HIGH confidence]
- DeepSeek-MoE / DeepSeek-V2: Grouped top-K routing with sigmoid scores (the pattern already in LitGPT's GroupedTopkRouter). [Training data, HIGH confidence -- code matches paper]
- Machine unlearning evaluation challenges: TOFU benchmark (Maini et al., 2024) and follow-up discussions on evaluation confounds. [Training data, MEDIUM confidence]
- AdamW optimizer behavior with zero gradients: PyTorch optimizer implementation; moment decay is well-documented behavior. [Training data, HIGH confidence -- fundamental PyTorch behavior]
- Existing LitGPT codebase: `model.py` lines 776-863 (LLaMAMoE, GroupedTopkRouter), `pretrain.py` training loop, `lora.py` parameter masking patterns. [Direct code inspection, HIGH confidence]
