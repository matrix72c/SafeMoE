# Architecture Patterns

**Domain:** MoE Safety / Knowledge Isolation (SafeMoE on LitGPT)
**Researched:** 2026-03-13

## Recommended Architecture

SafeMoE adds five new component groups to LitGPT. None of them replace existing code; they wrap or extend it at well-defined injection points.

```
                     +-----------------------+
                     |   safemoe/pretrain.py  |  <-- SGTM training loop (extends litgpt/pretrain.py)
                     +-----------+-----------+
                                 |
              +------------------+------------------+
              |                  |                  |
     +--------v--------+ +------v-------+ +--------v--------+
     | SGTMDataModule   | | SafeMoEGPT   | | SGTMTrainer      |
     | (data pipeline)  | | (model arch) | | (masking logic)  |
     +---------+--------+ +------+-------+ +--------+--------+
               |                 |                  |
         +-----v-----+   +------v-------+   +------v---------+
         | DomainLabel|   | SafeMoEBlock |   | GradientMasker |
         | Sampler    |   | SafeMoELayer |   | ActivationMask |
         +-----------+   | SafeMoERouter|   | ExpertRegistry |
                          +--------------+   +----------------+
                                 |
                          +------v--------+
                          | RoutingLogger |
                          | AblationUtil  |
                          +---------------+
```

### Component Boundaries

| Component | Responsibility | Location | Communicates With |
|-----------|---------------|----------|-------------------|
| **SafeMoEConfig** | Extend `litgpt.Config` with MoE-safety fields (harmful_expert_indices, n_expert, sgtm_mode) | `safemoe/config.py` | SafeMoEGPT, SGTMTrainer, CLI |
| **SafeMoELayer** | MoE dispatch: Top-K router + expert FFNs, with harmful-expert-aware routing | `safemoe/model.py` | SafeMoEBlock, RoutingLogger |
| **SafeMoERouter** | Learned gating network producing per-token expert probabilities | `safemoe/model.py` | SafeMoELayer |
| **SafeMoEBlock** | Drop-in replacement for `litgpt.Block.mlp` that uses SafeMoELayer instead of dense MLP | `safemoe/model.py` | SafeMoEGPT |
| **SafeMoEGPT** | Thin wrapper around `litgpt.GPT` that swaps MLP modules for SafeMoELayer at configured layers | `safemoe/model.py` | SGTMTrainer, pretrain loop |
| **ExpertRegistry** | Tracks which expert indices and attention heads are theta_harmful vs theta_std | `safemoe/registry.py` | GradientMasker, ActivationMasker, AblationUtil |
| **SGTMDataModule** | DataModule producing batches tagged with domain labels (D_harmful, D_std, D_unlabeled) | `safemoe/data.py` | SGTMTrainer, pretrain loop |
| **SGTMTrainer** | Orchestrates per-batch masking: reads domain label, applies GradientMasker or ActivationMasker accordingly | `safemoe/trainer.py` | GradientMasker, ActivationMasker, pretrain loop |
| **GradientMasker** | After backward pass, zeros gradients on theta_std parameters when domain=D_harmful | `safemoe/masking.py` | SGTMTrainer, ExpertRegistry |
| **ActivationMasker** | During forward pass, zeros theta_harmful activations when domain=D_std | `safemoe/masking.py` | SGTMTrainer, ExpertRegistry, SafeMoELayer |
| **RoutingLogger** | Records per-token expert assignment histograms during training and eval | `safemoe/logging.py` | SafeMoELayer, pretrain loop, eval |
| **AblationUtil** | Post-training: zeros theta_harmful weights in a checkpoint, produces ablated model | `safemoe/ablation.py` | ExpertRegistry, eval pipeline |
| **SafeMoEEval** | Per-domain perplexity evaluation + routing attribution analysis | `safemoe/eval.py` | AblationUtil, RoutingLogger |

## Injection Points into LitGPT

The architecture is designed around five specific injection points in the existing codebase. Each point is chosen to minimize changes to upstream LitGPT code.

### Injection Point 1: Block.mlp replacement (model.py:300)

**Where:** `litgpt/model.py`, `Block.__init__`, line `self.mlp = config.mlp_class(config)`

**How:** SafeMoE does NOT modify this line. Instead, it provides a custom Config subclass where `mlp_class` resolves to `SafeMoELayer` instead of `LLaMAMLP`. This is exactly how existing `LLaMAMoE` works -- the `mlp_class_name` field in Config determines what MLP module gets instantiated. SafeMoE extends this pattern.

**Why this works:** LitGPT already has precedent for MoE at this injection point. The existing `LLaMAMoE` class is selected via `config.mlp_class_name = "LLaMAMoE"`. SafeMoE adds a new class name (e.g., `"SafeMoELayer"`) that follows the same dispatch pattern but adds harmful-expert awareness and activation hooks.

**Existing code pattern to follow:**
```python
# litgpt/config.py line 91
mlp_class_name: Literal["GptNeoxMLP", "LLaMAMLP", "GemmaMLP", "LLaMAMoE"] = "GptNeoxMLP"

# litgpt/config.py line 223-227
@property
def mlp_class(self) -> Type:
    return getattr(litgpt.model, self.mlp_class_name)
```

SafeMoEConfig overrides `mlp_class` to also check `safemoe.model` module for the class name.

### Injection Point 2: Training loop inner step (pretrain.py:337-356)

**Where:** `litgpt/pretrain.py`, `fit()` function, the main `for train_data in train_iterator:` loop.

**How:** SafeMoE provides its own `safemoe/pretrain.py` with a modified `fit()` function. The modification is minimal: instead of a single forward-backward path, the loop reads the domain label from the batch and dispatches to one of three code paths (D_harmful, D_std, D_unlabeled). The `setup()` and `main()` functions are largely reused from `litgpt/pretrain.py`.

**Why not monkey-patch:** The three masking modes require fundamentally different gradient handling. A clean fork of `fit()` is more maintainable than hook-based injection. The fork is small (~50 lines of additional branching logic) and references the same utility functions.

**Critical training loop modification:**
```
Current LitGPT flow:
  for batch in dataloader:
      logits = model(input_ids)
      loss = cross_entropy(logits, targets)
      fabric.backward(loss)
      optimizer.step()

SafeMoE SGTM flow:
  for batch, domain_label in dataloader:
      if domain_label == D_HARMFUL:
          logits = model(input_ids)  # full forward, all experts active
          loss = cross_entropy(logits, targets)
          fabric.backward(loss)
          gradient_masker.zero_std_gradients()  # zero grad on theta_std
          optimizer.step()
      elif domain_label == D_STD:
          activation_masker.enable()  # register forward hook to zero harmful expert outputs
          logits = model(input_ids)
          activation_masker.disable()
          loss = cross_entropy(logits, targets)
          fabric.backward(loss)
          optimizer.step()
      else:  # D_UNLABELED
          logits = model(input_ids)  # standard pass, no masking
          loss = cross_entropy(logits, targets)
          fabric.backward(loss)
          optimizer.step()
```

### Injection Point 3: DataLoader batch format (pretrain.py:349-350)

**Where:** `litgpt/pretrain.py`, lines that unpack `train_data` into `input_ids` and `targets`.

**How:** The existing pretrain dataloader returns a single tensor `[B, seq_len+1]` that gets split into input_ids and targets. SafeMoE's `SGTMDataModule` returns a tuple `(token_tensor, domain_label_tensor)` where `domain_label_tensor` is shape `[B]` with values in {0=D_std, 1=D_harmful, 2=D_unlabeled}.

**Alternative considered:** Embedding the domain label as a special token or metadata column. Rejected because it would require changing the tokenization pipeline and model's max_seq_length calculations. A parallel tensor is simpler and does not interfere with the token stream.

### Injection Point 4: Checkpoint save/load (pretrain.py:500-509)

**Where:** `litgpt/pretrain.py`, `save_checkpoint()` function.

**How:** SafeMoE extends the checkpoint to also save: (a) the ExpertRegistry configuration (which experts are harmful), (b) routing statistics accumulated by RoutingLogger, and (c) a `safemoe_config.yaml` alongside the standard `model_config.yaml`. The existing `save_config()` utility is reused; SafeMoE adds an additional config file.

### Injection Point 5: Evaluation pipeline (eval/evaluate.py)

**Where:** `litgpt/eval/evaluate.py`, `convert_and_evaluate()`.

**How:** SafeMoE provides `safemoe/eval.py` with a custom evaluation script that: (a) computes per-domain perplexity (English PPL, Spanish PPL) before and after ablation, (b) runs routing attribution analysis using RoutingLogger data, (c) optionally runs lm-eval-harness tasks on the ablated model. This is a new entry point, not a modification of the existing one.

## Data Flow

### SGTM Training Data Flow

```
1. Raw Data Sources
   +------------------+     +------------------+     +--------------------+
   | TinyStories (EN) |     | TinyStories (ES) |     | TinyStories mixed  |
   | = D_std corpus   |     | = D_harmful proxy|     | = D_unlabeled pool |
   +--------+---------+     +--------+---------+     +---------+----------+
            |                        |                          |
            v                        v                          v
2. SGTMDataModule.prepare_data()
   +-------------------------------------------------------------------+
   | Tokenize each corpus separately using LitData streaming format.   |
   | Tag each shard with domain label metadata.                        |
   | Split x% of D_harmful into D_unlabeled per config.               |
   +-------------------------------------------------------------------+
            |
            v
3. SGTMDataModule.train_dataloader()
   +-------------------------------------------------------------------+
   | DomainBatchSampler: yields balanced mini-batches where each batch |
   | contains samples from exactly ONE domain. Returns:                |
   |   (token_tensor: [B, seq_len+1], domain_label: int)              |
   |                                                                   |
   | Domain proportions configurable (e.g., 1:1:1 or weighted).       |
   | Cycling through domains in round-robin or random order.           |
   +-------------------------------------------------------------------+
            |
            v
4. SafeMoE Training Loop (safemoe/pretrain.py::fit)
   +-------------------------------------------------------------------+
   | Read domain_label from batch.                                     |
   | Branch into one of three SGTM paths (see Injection Point 2).     |
   +-------------------------------------------------------------------+
            |
            v
5. Model Forward Pass (SafeMoEGPT)
   +-------------------------------------------------------------------+
   | Embedding layer (shared, theta_std) -> wte(input_ids)             |
   |     |                                                             |
   |     v                                                             |
   | For each SafeMoEBlock:                                            |
   |   norm_1 -> attention (shared, theta_std) -> post_attn_norm       |
   |   norm_2 -> SafeMoELayer:                                         |
   |     Router computes top-K expert indices and weights              |
   |     RoutingLogger records assignments                             |
   |     If activation_masker.enabled:                                 |
   |       zero the output of harmful experts before aggregation       |
   |     Dispatch tokens to selected experts                           |
   |     Aggregate weighted expert outputs                             |
   |   post_mlp_norm                                                   |
   |     |                                                             |
   |     v                                                             |
   | ln_f -> lm_head -> logits                                         |
   +-------------------------------------------------------------------+
            |
            v
6. Loss + Backward + Masking
   +-------------------------------------------------------------------+
   | chunked_cross_entropy(logits, targets) -> loss                    |
   | fabric.backward(loss)                                             |
   | If domain == D_harmful:                                           |
   |   GradientMasker.zero_std_gradients():                            |
   |     for name, param in model.named_parameters():                  |
   |       if registry.is_std_parameter(name):                         |
   |         param.grad.zero_()                                        |
   | optimizer.step()                                                  |
   +-------------------------------------------------------------------+
            |
            v
7. Logging & Checkpointing
   +-------------------------------------------------------------------+
   | Standard LitGPT metrics (loss, throughput, LR) logged.            |
   | Additionally per-step:                                            |
   |   - domain_label logged                                           |
   |   - routing histogram (which experts got which tokens)            |
   |   - per-domain running loss tracked separately                    |
   | Checkpoint saves: model weights + ExpertRegistry + routing stats  |
   +-------------------------------------------------------------------+
```

### Post-Training Ablation Flow

```
1. Load checkpoint (SafeMoEGPT + ExpertRegistry + routing stats)
        |
        v
2. AblationUtil.ablate(model, registry):
   For each expert_idx in registry.harmful_expert_indices:
     For each SafeMoELayer in model:
       model.blocks[i].mlp.experts[expert_idx].zero_all_weights()
   (Optionally: also zero harmful attention heads if registry tracks them)
        |
        v
3. Save ablated checkpoint
        |
        v
4. SafeMoEEval:
   - Compute English perplexity (expect: near baseline)
   - Compute Spanish perplexity (expect: significantly degraded)
   - Routing attribution: verify harmful tokens routed to ablated experts
```

### Inference Flow (Post-Ablation)

```
Standard LitGPT inference (no SafeMoE-specific logic needed).
The ablated model is a normal GPT checkpoint with some expert weights
zeroed. No runtime masking or domain detection required.
```

## Patterns to Follow

### Pattern 1: Config-Driven MLP Class Selection

**What:** Use LitGPT's existing `mlp_class_name` / `mlp_class` property pattern to select SafeMoELayer as the MLP implementation without modifying `Block.__init__`.

**When:** Always. This is how LitGPT selects between GptNeoxMLP, LLaMAMLP, and LLaMAMoE.

**Implementation approach:**
```python
# safemoe/config.py
from litgpt.config import Config as BaseConfig

class SafeMoEConfig(BaseConfig):
    # New fields
    harmful_expert_indices: list = field(default_factory=list)
    sgtm_mode: bool = True
    harmful_unlabeled_fraction: float = 0.0  # x% of harmful data to put in D_unlabeled

    @property
    def mlp_class(self) -> Type:
        if self.mlp_class_name == "SafeMoELayer":
            import safemoe.model
            return safemoe.model.SafeMoELayer
        return super().mlp_class
```

**Why:** Preserves compatibility with all existing LitGPT code. Block, GPT, and training infrastructure see a standard `config.mlp_class(config)` call and get back an nn.Module.

### Pattern 2: Forward Hook for Activation Masking

**What:** Use PyTorch forward hooks on SafeMoELayer to zero harmful expert outputs during D_std forward passes, rather than modifying the SafeMoELayer.forward() directly.

**When:** D_std training samples only.

**Implementation approach:**
```python
# safemoe/masking.py
class ActivationMasker:
    def __init__(self, model, registry):
        self.model = model
        self.registry = registry
        self.hooks = []

    def enable(self):
        """Register forward hooks that zero harmful expert outputs."""
        for block in self.model.transformer.h:
            if isinstance(block.mlp, SafeMoELayer):
                hook = block.mlp.register_forward_hook(self._mask_harmful_experts)
                self.hooks.append(hook)

    def disable(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def _mask_harmful_experts(self, module, input, output):
        # The SafeMoELayer forward stores per-expert outputs before aggregation
        # in module._last_expert_outputs for inspection.
        # This hook zeros the contribution of harmful experts.
        # Implementation detail: SafeMoELayer.forward() should be structured
        # to allow this hook to intervene before final aggregation.
        pass
```

**Why hooks instead of if-statements in forward():** Keeps SafeMoELayer.forward() clean and identical for all three SGTM modes. The masking logic is entirely external. This also means SafeMoELayer can be used without SGTM (e.g., standard MoE training) with no overhead.

**Alternative (recommended for simplicity):** Instead of hooks, pass a `mask_harmful: bool` flag through the forward call. SafeMoELayer checks the flag and skips harmful experts. This avoids hook complexity and is more explicit. The flag propagates through `GPT.forward() -> Block.forward() -> SafeMoELayer.forward()`. LitGPT's forward already passes through several optional args (mask, input_pos, etc.), so adding one more is natural. **This is the preferred approach for Milestone 1.**

### Pattern 3: Named Parameter Registry for Gradient Masking

**What:** ExpertRegistry maps parameter name prefixes to harmful/std classification. GradientMasker iterates `model.named_parameters()` and zeros gradients based on this classification.

**When:** After backward pass for D_harmful samples.

**Implementation approach:**
```python
# safemoe/registry.py
class ExpertRegistry:
    def __init__(self, harmful_expert_indices: List[int]):
        self.harmful_expert_indices = set(harmful_expert_indices)

    def is_std_parameter(self, param_name: str) -> bool:
        """Returns True if parameter belongs to theta_std (should be frozen for D_harmful)."""
        # Expert parameters: "transformer.h.{layer}.mlp.experts.{idx}.*"
        # Router parameters: "transformer.h.{layer}.mlp.gate.*" -- theta_std (shared)
        # Attention parameters: "transformer.h.{layer}.attn.*" -- theta_std
        # Embedding/head: "transformer.wte.*", "lm_head.*" -- theta_std
        # Norms: theta_std
        if ".mlp.experts." in param_name:
            expert_idx = int(param_name.split(".mlp.experts.")[1].split(".")[0])
            return expert_idx not in self.harmful_expert_indices
        # Everything that is not a harmful expert is theta_std
        return True

    def is_harmful_parameter(self, param_name: str) -> bool:
        return not self.is_std_parameter(param_name)
```

**Why named_parameters() traversal:** This is the same pattern LoRA uses to identify which parameters are trainable (frozen base vs. LoRA matrices). It is robust to model structure changes and works with torch.compile and FSDP because it operates on the parameter level after backward.

### Pattern 4: Single-Domain Batches

**What:** Each training mini-batch contains samples from exactly one domain. The domain label is a scalar integer accompanying the batch, not embedded in the token stream.

**When:** Always during SGTM training.

**Why single-domain batches:** Mixed-domain batches would require per-token gradient masking (masking different gradients for different tokens within the same batch), which is far more complex and interacts poorly with flash attention and gradient accumulation. Single-domain batches allow batch-level masking: one if-statement per batch, not per token.

**Trade-off:** This means gradient accumulation steps within a single optimizer step may span multiple domains. This is acceptable and may even be beneficial (each optimizer step aggregates gradients from multiple domains, similar to multi-task learning). The key constraint is that within a single forward-backward call, all tokens follow the same masking regime.

### Pattern 5: Building on Existing LLaMAMoE

**What:** SafeMoELayer should be a subclass (or close adaptation) of the existing `litgpt.model.LLaMAMoE` class, not a from-scratch implementation.

**When:** For Milestone 1 (PT phase). Milestone 2 may need modifications for CPT injection.

**Why:** LLaMAMoE already implements: top-K routing via `nn.Linear` gate, expert dispatch with index-based token routing, softmax probability weighting, shared expert support, routed scaling factor. SafeMoE needs all of these plus: harmful expert awareness in the routing path, routing logging hooks, and activation masking support. Starting from LLaMAMoE avoids re-implementing the dispatch logic.

**Implementation approach:**
```python
# safemoe/model.py
class SafeMoELayer(LLaMAMoE):
    """MoE layer with harmful-expert awareness for SGTM training."""

    def __init__(self, config):
        super().__init__(config)
        # Additional state for SGTM
        self.mask_harmful = False  # Set by ActivationMasker or training loop
        self._routing_log = None   # Populated during forward for RoutingLogger

    def forward(self, x):
        B, T, C = x.size()
        residual_x = x.clone()
        x = x.view(-1, C)

        # Router computation (same as LLaMAMoE)
        router = self.gate(x)
        probs, indices = torch.topk(router, self.config.n_expert_per_token)
        probs = probs.softmax(dim=1, dtype=torch.float).to(dtype=x.dtype)

        # Log routing decisions
        self._routing_log = indices.detach()

        # Expert dispatch with optional harmful masking
        masks = indices.unsqueeze(-1) == torch.arange(self.config.n_expert, device=x.device)
        masks = masks.permute(2, 0, 1)
        y = torch.zeros_like(x)

        for expert_idx, (mask, expert) in enumerate(zip(masks, self.experts)):
            if self.mask_harmful and expert_idx in self.config.harmful_expert_indices:
                continue  # Skip harmful experts during D_std forward
            token_idx, slot_idx = torch.where(mask)
            if token_idx.numel() > 0:
                y[token_idx] += probs[token_idx, slot_idx, None] * expert(x[token_idx])

        y = y.view(B, T, C)
        if self.config.n_shared_expert:
            y = y + self.shared_experts(residual_x)
        return y
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Modifying litgpt/model.py Directly

**What:** Adding SafeMoE-specific code (harmful expert checks, routing logs, masking flags) directly into `litgpt/model.py`'s `Block` or `GPT` classes.

**Why bad:** Creates merge conflicts with upstream LitGPT. Makes it impossible to use the codebase for non-SafeMoE work. Violates the project constraint "Must extend LitGPT internals directly" -- "extend" means subclass/wrap, not modify in place.

**Instead:** Use Config-driven dispatch (Pattern 1) and external masking (Patterns 2/3). SafeMoE code lives in `safemoe/` module. The only changes to `litgpt/` should be additive: adding the new mlp_class_name to the Config Literal type, or registering new CLI commands.

### Anti-Pattern 2: Per-Token Gradient Masking Within Mixed Batches

**What:** Training with batches containing tokens from multiple domains and attempting to mask gradients differently for different tokens within the same batch.

**Why bad:** (a) PyTorch's autograd computes gradients per-parameter, not per-token. Achieving per-token gradient selectivity requires manual backward pass manipulation or per-token loss weighting hacks that are fragile. (b) Flash attention fuses the entire batch, making token-level gradient intervention extremely difficult. (c) Gradient accumulation becomes ambiguous.

**Instead:** Use single-domain batches (Pattern 4). Each forward-backward call processes one domain. Gradient masking operates at the parameter level on the full batch gradient.

### Anti-Pattern 3: Separate Models for Harmful vs Standard Training

**What:** Maintaining two model copies (one for harmful training, one for standard training) and merging weights.

**Why bad:** Doubles memory usage. Weight merging introduces a new source of complexity. The whole point of MoE is that one model has separate expert pathways.

**Instead:** One model, three code paths in the training loop. The model is always the same; what changes is which parameters receive gradient updates and which activations are masked.

### Anti-Pattern 4: Routing via Metadata Instead of Learned Router

**What:** Bypassing the learned router and force-assigning harmful tokens to harmful experts via metadata.

**Why bad:** Defeats the research purpose. The core hypothesis is that the learned router will naturally concentrate harmful knowledge in designated experts when SGTM training incentivizes it. Force-routing would create an artificial result that does not demonstrate learned knowledge isolation.

**Instead:** The router is always learned. SGTM training creates the incentive for the router to send harmful tokens to harmful experts: during D_harmful, only harmful expert parameters update (so they are the only ones that can learn from harmful data); during D_std, harmful experts are masked (so the router learns not to rely on them for standard tasks). The router learns the assignment, not the training code.

### Anti-Pattern 5: Logging Routing Stats by Modifying the Router

**What:** Adding logging code directly inside the router's forward method, potentially breaking torch.compile or adding overhead.

**Why bad:** Router forward runs once per token per layer per step. Logging overhead multiplies rapidly. Also complicates compilation.

**Instead:** Store routing indices as a tensor attribute on the layer (`self._routing_log = indices.detach()`). The RoutingLogger reads this attribute after the forward pass, outside the computation graph. Detach ensures no gradient leakage. The logger aggregates histograms at configurable intervals, not every step.

## Key Design Decisions

### Decision 1: SafeMoELayer as the MoE Implementation

**Choice:** Subclass `LLaMAMoE` rather than write from scratch.

**Rationale:** LLaMAMoE already handles token-level expert dispatch, top-K routing, softmax normalization, shared experts, and scaling factors. These are all correct and tested within LitGPT. SafeMoE adds two behaviors: (a) optional activation zeroing for designated experts, (b) routing log capture. Both are small additions to the existing forward pass.

**Risk:** If LLaMAMoE changes upstream, SafeMoELayer needs updating. Mitigated by the fact that LLaMAMoE is a stable, well-tested component supporting Mixtral/Qwen configs.

### Decision 2: Parameter-Level Gradient Masking (Not Module-Level)

**Choice:** Zero gradients by iterating `named_parameters()` rather than by freezing/unfreezing modules.

**Rationale:** Module-level freeze/unfreeze (`requires_grad = False/True`) has side effects with torch.compile and FSDP, and changing it every batch is error-prone. Named parameter iteration with `param.grad.zero_()` is a post-backward operation that does not affect the computation graph and works correctly with all training infrastructure.

**Performance note:** The named_parameters() traversal can be optimized by pre-computing the set of std/harmful parameters during initialization, then iterating only that set.

### Decision 3: Domain Label as Separate Tensor, Not Token Metadata

**Choice:** `SGTMDataModule` returns `(tokens: [B, seq_len+1], domain: [B])` as a tuple.

**Rationale:** Keeps the token tensor identical in format to standard LitGPT pretraining. The training loop unpacks one extra field. No tokenizer changes, no sequence length changes, no model input format changes. The domain label tensor is a simple integer per sample.

### Decision 4: Attention Heads as Part of theta_std (Not theta_harmful) for Milestone 1

**Choice:** For PT phase, only MoE expert FFNs are theta_harmful. Attention heads, embeddings, and the router are theta_std.

**Rationale:** (a) MoE architectures concentrate specialized knowledge in expert FFNs, not attention. Attention is a shared routing/mixing mechanism. (b) Designating attention heads as harmful adds significant complexity (head-level gradient masking) for uncertain benefit. (c) The original SGTM concept focuses on expert-level isolation. Attention head designation can be explored in Milestone 2 if routing analysis suggests attention also specializes.

### Decision 5: New safemoe/ Module, Not Modification of litgpt/

**Choice:** All SafeMoE code in a `safemoe/` top-level package, not modifications to `litgpt/`.

**Rationale:** (a) Clean separation of research code from framework code. (b) Easier to diff what SafeMoE adds vs what LitGPT provides. (c) No merge conflicts with upstream. (d) Can import from litgpt freely. The only LitGPT change needed is adding `"SafeMoELayer"` to the `mlp_class_name` Literal type in `config.py`, which is a one-line additive change.

## Module Layout

```
safemoe/
    __init__.py
    config.py           # SafeMoEConfig extending litgpt.Config
    model.py            # SafeMoELayer, SafeMoERouter (extends LLaMAMoE)
    registry.py         # ExpertRegistry: harmful/std parameter classification
    masking.py          # GradientMasker, ActivationMasker
    data.py             # SGTMDataModule: domain-labeled data pipeline
    pretrain.py         # SGTM training loop (extends litgpt/pretrain.py)
    trainer.py          # SGTMTrainer: orchestrates masking per batch
    logging.py          # RoutingLogger: expert assignment histograms
    ablation.py         # AblationUtil: zero harmful experts post-training
    eval.py             # SafeMoEEval: per-domain PPL + routing attribution
    cli.py              # CLI entry points for SafeMoE commands
```

## Scalability Considerations

| Concern | Milestone 1 (PT, single GPU) | Milestone 2 (CPT, single GPU) | Future (multi-GPU) |
|---------|------------------------------|-------------------------------|---------------------|
| MoE memory | Small model (TinyStories scale), 4-8 experts fit in GPU memory | Larger model, may need expert offloading | FSDP wraps each expert; SafeMoELayer needs FSDP-compatible masking |
| Gradient masking overhead | Negligible: iterate ~100 parameter groups per step | Same | Parameter sets pre-computed; iteration is O(params), not O(tokens) |
| Routing logging | In-memory histogram accumulation, write every N steps | Same | Reduce across ranks before logging |
| Data pipeline | LitData streaming, three separate shards | Real harmful datasets may need custom preprocessing | Same LitData streaming |
| Activation masking | Skip harmful expert FFN calls (saves compute) | Same | No distributed complication |
| Ablation | Zero weights in checkpoint, one-time operation | Same | Same |

## Build Order (Dependencies)

Components must be built in this order due to dependencies:

```
Phase A: Foundation (no dependencies between these, can parallelize)
  1. SafeMoEConfig         -- needed by everything
  2. ExpertRegistry        -- needed by masking, ablation
  3. SafeMoELayer/Router   -- needed by model

Phase B: Training Infrastructure (depends on Phase A)
  4. SGTMDataModule        -- needs SafeMoEConfig for domain config
  5. GradientMasker        -- needs ExpertRegistry
  6. ActivationMasker      -- needs ExpertRegistry, SafeMoELayer
  7. RoutingLogger         -- needs SafeMoELayer

Phase C: Training Loop (depends on Phase B)
  8. SGTMTrainer           -- needs GradientMasker, ActivationMasker
  9. safemoe/pretrain.py   -- needs SGTMTrainer, SGTMDataModule, SafeMoEConfig

Phase D: Evaluation (depends on Phase A, can parallel with B/C for unit tests)
  10. AblationUtil         -- needs ExpertRegistry
  11. SafeMoEEval          -- needs AblationUtil, RoutingLogger

Phase E: Integration (depends on all above)
  12. CLI registration     -- needs pretrain.py, eval.py
  13. End-to-end testing   -- needs everything
```

**Critical path:** SafeMoEConfig -> SafeMoELayer -> SGTMDataModule -> safemoe/pretrain.py -> End-to-end test.

**Parallelizable work:** ExpertRegistry + GradientMasker can be built while SafeMoELayer is being developed. RoutingLogger and AblationUtil can be built independently as long as the SafeMoELayer interface (which stores `_routing_log`) is agreed upon.

## Sources

- LitGPT source code analysis (litgpt/model.py, litgpt/pretrain.py, litgpt/config.py, litgpt/data/base.py, litgpt/lora.py) -- PRIMARY source, HIGH confidence
- Existing LLaMAMoE implementation in litgpt/model.py lines 776-819 -- direct precedent for MoE architecture pattern
- Existing Config.mlp_class dispatch pattern in litgpt/config.py lines 223-227 -- direct precedent for class selection
- LoRA pattern for selective parameter handling in litgpt/lora.py -- direct precedent for parameter-level training control
- LitGPT TinyStories DataModule in litgpt/data/tinystories.py -- direct precedent for data pipeline pattern
- PROJECT.md SGTM algorithm specification -- defines the three masking modes

**Confidence:** HIGH for all architecture decisions. All are grounded in concrete LitGPT code patterns that already exist and work. The SafeMoE architecture is a composition of proven patterns (MoE dispatch from LLaMAMoE, parameter classification from LoRA, data pipeline from DataModule, training loop from pretrain.py) applied to the novel SGTM training algorithm.

---

*Architecture analysis: 2026-03-13*
