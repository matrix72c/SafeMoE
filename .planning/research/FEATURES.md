# Feature Landscape

**Domain:** MoE knowledge isolation research framework (SGTM algorithm)
**Researched:** 2026-03-13
**Overall Confidence:** MEDIUM (based on codebase analysis + training data on MoE/unlearning literature; web search unavailable for live verification)

## Table Stakes

Features the framework must have or the research claims are invalid. Missing any of these means experimental results cannot be trusted.

### Core SGTM Algorithm

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Selective Gradient Masking** (D_harmful -> theta_harmful only) | Core of SGTM -- without this, harmful knowledge bleeds into standard experts and the entire isolation hypothesis collapses | High | After backward pass on D_harmful, zero gradients for all parameters NOT in theta_harmful. Must work with gradient accumulation and Lightning Fabric's backward sync. Similar pattern to LoRA's `mark_only_lora_as_trainable` but applied dynamically per-batch, not statically at init. |
| **Selective Parameter Masking** (D_std -> theta_harmful zeroed in forward) | Without this, standard data trains harmful experts too, breaking isolation. The model must learn to not rely on harmful experts for general knowledge. | High | During forward pass on D_std batches, zero harmful expert activations (gate outputs AND expert MLP outputs). Must be differentiable through the non-masked path. Must not corrupt the router gradients for standard experts. |
| **Normal Pass** (D_unlabeled -> no masking) | The unlabeled pass is what makes SGTM practical -- real data is unlabeled. Without it, the approach only works with perfectly labeled corpora. | Low | Standard forward + backward with no masking. Trivial since it is the default training loop. The research question is whether unlabeled data still routes harmful content to harmful experts. |
| **Three-stream Data Sampling** | SGTM requires alternating between D_harmful, D_std, and D_unlabeled within the same training loop. Without controlled interleaving, the masking logic cannot be applied correctly. | Medium | Each micro-batch must carry a label indicating its stream type (harmful/std/unlabeled). The training loop dispatches to the correct forward/backward path based on this label. Must maintain proper gradient accumulation ratios across streams. |
| **MoE Expert Layer** (Top-K router + expert dispatch) | The entire approach requires a Mixture-of-Experts architecture where knowledge can be physically separated into distinct parameter sets. | Medium | LitGPT already has `LLaMAMoE` with Top-K routing. SafeMoE needs this as the base, but with hooks for designation and masking. The existing implementation uses softmax-weighted Top-K dispatch, which is standard. |
| **Harmful Expert Designation** | Must be able to mark specific experts as theta_harmful, so the masking logic knows which parameters to protect/zero. This is the mechanism that maps the abstract "harmful" concept to concrete parameter groups. | Low | A configuration-level list of expert indices designated as harmful. Used by both gradient masking and parameter masking to identify targets. Conceptually simple, but must propagate correctly through all layers with MoE. |
| **Expert Ablation Utility** | Post-training, permanently zero out theta_harmful weights and verify the model still functions. This is the payoff -- if ablation destroys general capability, the approach failed. | Low | Set all parameters in designated harmful experts to zero. Optionally remove them entirely from the model. Must save a clean checkpoint without harmful experts for downstream evaluation. |

### Evaluation Suite

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Per-Domain Perplexity** (pre/post ablation) | The primary success metric. Must measure: (a) harmful-domain PPL increases after ablation (knowledge removed), (b) standard-domain PPL stays near baseline (general capability preserved). Without this, there is no quantitative claim. | Medium | For Milestone 1: per-language PPL on English vs Spanish TinyStories. For Milestones 2-3: per-domain PPL on harmful vs standard benchmarks. Must support evaluation on arbitrary held-out sets, not just the training data. |
| **Routing Attribution Analysis** | Must show that harmful tokens preferentially route to harmful experts. Without routing evidence, gradient masking might "work" by accident rather than by the hypothesized mechanism. | Medium | Log router probabilities and top-K expert assignments per token during evaluation passes. Aggregate into per-expert activation frequencies broken down by domain (harmful vs standard). Visualize as histograms or heatmaps. |
| **Pre/Post Ablation Comparison** | Need side-by-side metrics before and after removing harmful experts. A single post-ablation number is meaningless without the pre-ablation baseline. | Low | Run the same eval suite twice (with and without ablation) and produce a comparison table. Must be automated, not manual. |

### Data Pipeline

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Labeled Split Pipeline** (D_harmful / D_std / D_unlabeled) | SGTM requires three data streams with known labels. The pipeline must partition data correctly and maintain stream identity throughout training. | Medium | For Milestone 1: TinyStories bilingual where Spanish = D_harmful, English = D_std, x% Spanish mixed into D_unlabeled. Must support configurable split ratios. Must yield batches with stream-type labels attached. |
| **Bilingual Proxy Dataset** (TinyStories EN/ES) | Milestone 1's proxy setup. Spanish serves as a clean stand-in for harmful knowledge (same distribution complexity, no ethical concerns, ground truth separation). | Medium | Must download/prepare both English and Spanish TinyStories, tokenize with a shared tokenizer that handles both languages, and partition according to the split configuration. The existing `TinyStories` DataModule handles English only; needs extension for bilingual. |
| **Configurable x% Unlabeled Contamination** | The fraction of harmful data mixed into D_unlabeled is a key sweep parameter. The research question "how much unlabeled harmful data can the router still isolate?" depends on this being configurable. | Low | A float parameter (0.0 to 1.0) controlling what fraction of D_harmful samples are moved into D_unlabeled (losing their labels). Must be set via CLI args consistent with LitGPT's jsonargparse convention. |

### Infrastructure

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Checkpoint Save/Load with MoE + Designation Metadata** | Must save which experts are designated as harmful alongside model weights. Without this, you cannot resume training or reproduce results from a checkpoint. | Low | Extend LitGPT's existing checkpoint save/load to include SafeMoE metadata (harmful expert indices, masking configuration, stream ratios). The existing `save_checkpoint` saves model state + config; extend with SafeMoE config. |
| **CLI Integration** (jsonargparse) | All experiment parameters must be configurable from the command line consistent with how LitGPT works. Hardcoded parameters destroy reproducibility. | Low | Add a `safemoe pretrain` command (or extend `litgpt pretrain`) with additional args for harmful expert indices, stream ratios, masking strategy. Follow existing pattern in `litgpt/__main__.py` and `litgpt/args.py`. |
| **Metric Logging** (TensorBoard/WandB) | Training curves for all three streams (harmful loss, standard loss, unlabeled loss) plus routing statistics must be logged for analysis. Without per-stream loss tracking, you cannot diagnose training dynamics. | Low | Extend LitGPT's existing `fabric.log_dict()` calls to include per-stream losses and routing statistics. The infrastructure is already there; just need to log additional metrics. |
| **Seed-Based Reproducibility** | Different data splits and random initializations must be reproducible. Otherwise, you cannot distinguish signal from noise across experiment runs. | Low | LitGPT already has `fabric.seed_everything(seed)`. Ensure the data split logic is also deterministic given the same seed. |

## Differentiators

Features that make SafeMoE novel compared to existing MoE unlearning/safety work. These are the research contributions.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Training-Time Isolation** (not post-hoc) | Existing MoE safety work (e.g., safety-tuning, RLHF) operates post-training. SGTM embeds isolation into the pretraining/CPT loop itself. This is the core novelty -- harmful knowledge never mixes with standard knowledge in the first place. | High | The entire SGTM three-pass training loop (gradient masking + parameter masking + normal pass) is the differentiator. No existing framework implements this. Must be built from scratch on top of LitGPT's training loop. |
| **Bilingual Proxy Validation** | Using language as a proxy for "harmful knowledge" gives ground truth that is impossible with real harmful content. You know exactly which tokens are "harmful" (Spanish), which means routing analysis has a clean signal. No ethical ambiguity. | Medium | The proxy is the experimental design, not just the data pipeline. Must support metrics that leverage the ground truth: "what % of Spanish tokens route to harmful experts?" is answerable cleanly because we know which tokens are Spanish. |
| **Unlabeled Data Knowledge Transfer** | The D_unlabeled stream tests whether the router learns to separate harmful content even without explicit labels. This is the hardest and most novel claim -- if it works, SGTM scales to real unlabeled corpora. | High | Milestone 3's focus. Must measure: (a) does harmful knowledge in unlabeled data still migrate to harmful experts? (b) does post-ablation PPL on harmful content increase even for knowledge learned from unlabeled data? Requires careful experimental design with held-out eval sets. |
| **Progressive Milestone Validation** | Three milestones with independent success criteria prevent building on unvalidated assumptions. Most research frameworks are "build everything, then evaluate." SafeMoE validates each step before proceeding. | Low | More of a methodology feature than a code feature, but the framework must support milestone-specific evaluation suites and clean experiment boundaries. |
| **Harmful Expert Injection into Pretrained Models** | Milestone 2 adds harmful experts to an existing pretrained model (not from scratch). This is closer to the real deployment scenario where you take an existing LLM and add safety guardrails via additional experts. | High | Must splice new expert modules into an existing LitGPT checkpoint without corrupting the pretrained weights. The new experts must be initialized (random or otherwise) while the existing model parameters are frozen during injection. |
| **Routing Drift Analysis** | Track how routing patterns change over CPT training. Does harmful content start in standard experts and gradually migrate to harmful experts? This temporal analysis is unique to the SGTM approach. | Medium | Log per-step routing statistics (not just final) and produce trajectory plots. Requires periodic evaluation checkpoints that capture routing state, not just loss. |
| **Per-Expert Knowledge Attribution** | Beyond routing frequencies, attribute what knowledge each expert "holds" by measuring per-expert perplexity contribution. If ablating expert 3 kills Spanish PPL but not English PPL, expert 3 holds Spanish knowledge. | Medium | Requires a sweep where each expert is individually ablated and the resulting PPL change is measured per domain. Expensive (N_experts x N_domains evaluations) but provides definitive attribution evidence. |

## Anti-Features

Features to explicitly NOT build. These would waste effort or actively harm the research.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Real-time serving / deployment API** | Research framework, not production system. Serving code adds complexity without informing the research question. | Use LitGPT's existing `generate` scripts for qualitative inspection. Evaluation is batch-mode perplexity computation. |
| **Multi-node distributed training** | Overkill for Milestone 1 (TinyStories + small MoE). Multi-node introduces debugging complexity that distracts from algorithm validation. Single-GPU or single-node multi-GPU is sufficient. | Use Lightning Fabric's single-node capabilities. LitGPT already supports multi-GPU via FSDP if needed later. |
| **Custom tokenizer / vocabulary** | The bilingual proxy needs a tokenizer that handles both EN and ES, but building a custom tokenizer is unnecessary. Existing multilingual tokenizers (e.g., from Llama or GPT-2) handle this fine. | Use an off-the-shelf multilingual tokenizer. For TinyStories, even GPT-2's tokenizer handles basic Spanish adequately (it tokenizes into subwords, not perfectly, but sufficiently for a proxy experiment). |
| **Automatic harmful content detection** | The research question is whether SGTM can isolate given correct labels, not whether we can detect harmful content. Conflating detection with isolation muddies the experimental design. | Use clean labels (bilingual in M1, curated harmful datasets in M2-3). Detection is a separate research problem. |
| **Load balancing loss** | Standard MoE training uses load balancing auxiliary losses to prevent expert collapse. For SafeMoE, we intentionally want some experts to specialize in harmful content. Load balancing would fight against the desired behavior. | Explicitly omit load balancing loss. Monitor expert utilization to ensure experts are not trivially collapsed, but do not penalize uneven routing. The routing imbalance IS the desired outcome. |
| **Expert capacity / token dropping** | Capacity factors and token dropping (common in large-scale MoE like GShard, Switch Transformer) prevent single-expert overload but would interfere with the controlled routing we need. | Use the dense-per-expert dispatch already in LitGPT's `LLaMAMoE` (no capacity limits, no token dropping). All tokens assigned to an expert get processed by that expert. |
| **Sparse attention modifications** | MoE operates at the FFN level. There is no need to modify the attention mechanism for knowledge isolation. Attention modifications add complexity without supporting the SGTM hypothesis. | Keep LitGPT's standard `CausalSelfAttention` unchanged. The isolation mechanism is purely in the MoE expert layer. |
| **Model parallelism / expert parallelism** | Expert parallelism (distributing experts across GPUs) is for production MoE at scale. For research, all experts fit on one GPU (TinyStories models are small). | Keep all experts on the same device. If model grows in M2-3, use LitGPT's existing FSDP strategy rather than custom expert parallelism. |
| **Mixture-of-Depths or conditional computation** | Orthogonal research direction that would confound the isolation experiments by adding another dimension of routing. | Pure Mixture-of-Experts with standard Top-K routing. Keep the architecture minimal to isolate the SGTM contribution. |

## Feature Dependencies

```
MoE Expert Layer
  |-> Harmful Expert Designation (needs expert indices to exist)
  |     |-> Selective Gradient Masking (needs to know which params are theta_harmful)
  |     |-> Selective Parameter Masking (needs to know which expert activations to zero)
  |     |-> Expert Ablation Utility (needs to know which experts to remove)
  |
  |-> Routing Attribution Analysis (needs router logits to analyze)
       |-> Per-Expert Knowledge Attribution (needs routing + ablation)

Labeled Split Pipeline
  |-> Three-stream Data Sampling (needs labeled partitions)
  |     |-> SGTM Training Loop (needs stream labels to dispatch masking)
  |
  |-> Bilingual Proxy Dataset (specific instance for M1)
  |-> Configurable x% Unlabeled (parameterizes the split)

Per-Domain Perplexity
  |-> Pre/Post Ablation Comparison (needs PPL measurement)
  |-> Routing Drift Analysis (needs periodic PPL + routing snapshots)

Checkpoint Save/Load with Metadata
  |-> All training and evaluation (must persist state)

CLI Integration
  |-> All features (must be configurable from command line)
```

**Critical path:** MoE Expert Layer -> Harmful Expert Designation -> Selective Gradient/Parameter Masking -> Three-stream Data Sampling -> SGTM Training Loop -> Evaluation Suite

## MVP Recommendation

Prioritize for Milestone 1 (PT Phase Validation):

1. **MoE Expert Layer** -- Extend existing `LLaMAMoE` or build a custom SafeMoE layer with hooks for masking. This is the architectural foundation.
2. **Harmful Expert Designation** -- Configuration + parameter group management. Simple but must be correct.
3. **Selective Gradient Masking** -- The most complex table-stakes feature. Implement and unit test exhaustively before integration.
4. **Selective Parameter Masking** -- Coupled with gradient masking; test both forward and backward correctness.
5. **Bilingual Data Pipeline** -- TinyStories EN/ES with configurable splits. Must yield stream-labeled batches.
6. **Three-stream Training Loop** -- Integrate masking with the data pipeline in LitGPT's `pretrain.py` pattern.
7. **Per-Domain Perplexity** -- Minimum eval to validate isolation.
8. **Routing Attribution** -- Minimum eval to verify mechanism.

Defer to Milestone 2:
- **Harmful Expert Injection** (requires pretrained model surgery)
- **Real harmful dataset integration** (WildGuard, HarmBench)
- **Routing Drift Analysis** (needs CPT-length training runs)

Defer to Milestone 3:
- **Unlabeled Knowledge Transfer metrics** (needs M2 validation first)
- **Per-Expert Knowledge Attribution** (expensive sweep, needs M2 infrastructure)

## Sources

- LitGPT codebase analysis (direct code examination of `model.py`, `pretrain.py`, `config.py`, `lora.py`, `data/tinystories.py`, `data/base.py`, `eval/evaluate.py`, `args.py`) -- HIGH confidence
- Project specification (`.planning/PROJECT.md`) -- HIGH confidence
- MoE literature: Shazeer et al. 2017 (Outrageously Large Neural Networks), Fedus et al. 2022 (Switch Transformers), Jiang et al. 2024 (Mixtral of Experts) -- MEDIUM confidence (training data, not live verified)
- Machine unlearning literature: Jang et al. 2023 (Knowledge Unlearning for LLMs), Li et al. 2024 (WMDP benchmark) -- MEDIUM confidence (training data)
- MoE safety/specialization: Li et al. 2024 (Examining MoE expert specialization), Gao et al. 2024 (Higher layers need more LoRA experts) -- LOW confidence (training data, details may be inaccurate)
- Gradient masking approach is novel to this project; no direct comparable framework found in training data -- this is expected for a research contribution

---

*Feature landscape analysis: 2026-03-13*
