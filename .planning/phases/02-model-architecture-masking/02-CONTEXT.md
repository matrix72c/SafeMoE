# Phase 2: Model Architecture & Masking - Context

**Gathered:** 2026-03-16
**Status:** Ready for planning

<domain>
## Phase Boundary

Build the SafeMoE model internals: SafeMoEConfig, SafeMoELayer (subclassing LLaMAMoE), HarmfulParamRegistry, GradientMasker, and ActivationMasker — with unit tests verifying all masking invariants. This phase delivers the designation and masking primitives; the SGTM training loop that uses them belongs in Phase 3.

</domain>

<decisions>
## Implementation Decisions

### Config extension
- `SafeMoEConfig` is a `@dataclass` subclass of `litgpt.Config`, located in `safemoe/config.py`
- Adds three fields with defaults: `harmful_expert_indices: list[int] = field(default_factory=list)`, `harmful_attn_heads: list[int] = field(default_factory=list)`, `num_harmful_experts: int = 0`
- All defaults produce a config that behaves identically to a standard `Config` (no harmful designation)
- Phase 2 also includes a `safemoe/configs/safemoe-tinystories.yaml` experiment config for a small TinyStories MoE model (small n_layer, 4–8 experts, 2 harmful experts designated)
/gsd
### Attention head designation
- `harmful_attn_heads: list[int]` is a **global** list of head indices applied uniformly across **every** attention layer in the model
- The QKV projection rows for each designated head are θ_harmful; the output projection (`attn.proj`) stays θ_std
- In Phase 2, `HarmfulParamRegistry` classifies the attention head QKV rows correctly, but `GradientMasker` and `ActivationMasker` do **not** apply masking to attention heads yet — masker scope is MoE experts only in this phase; attention head masking is activated in Phase 3

### HarmfulParamRegistry
- Constructed as `HarmfulParamRegistry(model, config)` — scans `model.named_parameters()` at construction time
- θ_harmful = weights of `harmful_expert_indices` experts in each `SafeMoELayer` + QKV projection row slices for `harmful_attn_heads` in each attention layer
- θ_std = ALL remaining parameters (embedding, LM head, LayerNorm, non-harmful expert weights, non-harmful attention heads, output projections)
- Interface: `parameters_by_type(split: str) -> list[nn.Parameter]` where split is `'theta_harmful'` or `'theta_std'`
- Validates at construction: raises `ValueError` if θ_harmful ∩ θ_std ≠ ∅ or if θ_harmful ∪ θ_std ≠ all model parameters (exhaustive, non-overlapping guarantee)

### GradientMasker
- After `loss.backward()` on a D_harmful batch, the training loop calls `gradient_masker.mask()`
- Implementation: manual loop over `registry.parameters_by_type('theta_std')` setting each `.grad = None`
- No hooks — explicit, transparent, matches MASK-01 spec ("post-backward zeroing, not detach-in-forward")

### ActivationMasker
- `SafeMoELayer` carries a `_activation_masking_enabled: bool` flag (default `False`); `ActivationMasker` holds references to all `SafeMoELayer` instances collected at construction time via `model.modules()`
- `masker.enable()` sets `_activation_masking_enabled = True` on every `SafeMoELayer`; `masker.disable()` restores it to `False`
- Training loop calls `activation_masker.enable()` before each D_std forward pass, `activation_masker.disable()` after
- Flag-based approach (not a `register_forward_hook`): `SafeMoELayer.forward()` checks `self._activation_masking_enabled` and skips accumulation for harmful expert indices — this is necessary because a post-forward hook receives the already-aggregated `y` tensor and cannot retroactively zero individual expert contributions (RESEARCH.md Pitfall 4)

### Dual optimizer param groups
- `MASK-03`: two separate `AdamW` instances (or param groups) — one for θ_harmful, one for θ_std
- Both use `zero_grad(set_to_none=True)` to prevent Adam momentum from accumulating on zero gradients
- Details of optimizer construction live in Phase 3 training loop; Phase 2 only needs to expose `parameters_by_type()` to enable this

### Claude's Discretion
- Exact YAML values for the TinyStories experiment config (model size, head count, block size, etc.) — choose something small enough for CPU-only unit tests
- Internal data structure of HarmfulParamRegistry (dict vs list, caching strategy)
- How QKV row slices are extracted from fused attn.proj weight matrix (slice indexing math)
- SafeMoELayer module file location (`safemoe/model.py`)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `litgpt/model.py:776 LLaMAMoE`: SafeMoELayer subclasses this. Has `gate`, `experts: nn.ModuleList`, and a `for mask, expert in zip(masks, self.experts): y[token_idx] += ...` dispatch loop — SafeMoELayer.forward() checks `_activation_masking_enabled` to skip harmful expert accumulation.
- `litgpt/config.py Config`: Base class for SafeMoEConfig. Already has `n_expert`, `n_expert_per_token`, `moe_intermediate_size`. SafeMoEConfig inherits all of these.
- `litgpt/lora.py lora_filter()` + `named_parameters()` scan: Established pattern for per-parameter type filtering — HarmfulParamRegistry follows the same approach.
- `litgpt/model.py CausalSelfAttention`: Fused QKV in `attn.attn` linear (weight shape: `(n_head + 2*n_query_groups) * head_size, n_embd`). Head i's Q rows = `[i*head_size : (i+1)*head_size]`; head i's K rows start after all Q rows.

### Established Patterns
- Model configuration: `@dataclass` subclass pattern used by `litgpt/adapter.py Config(litgpt.Config)` — SafeMoEConfig follows the same pattern.
- Block mlp_class: `Block.mlp = config.mlp_class(config)` — SafeMoELayer is set as `mlp_class` in SafeMoEConfig.
- Parameter group dispatch: `lora_filter()` uses string key matching on `named_parameters()` — HarmfulParamRegistry uses similar name-matching logic.
- Tests for model components: `tests/test_model.py` instantiates small models via `Config` with tiny dims — Phase 2 tests follow the same pattern with `SafeMoEConfig`.

### Integration Points
- Phase 3 training loop instantiates `HarmfulParamRegistry(model, config)` and the two maskers at setup
- Phase 3 training loop calls `masker.enable()` / `masker.disable()` around D_std forward passes
- Phase 3 training loop calls `gradient_masker.mask()` after D_harmful backward passes
- Phase 3 training loop uses `registry.parameters_by_type('theta_harmful')` and `registry.parameters_by_type('theta_std')` to construct dual AdamW param groups

</code_context>

<specifics>
## Specific Ideas

- No specific UI/UX references — this is pure research infrastructure
- The "post-backward zeroing, not detach-in-forward" note in MASK-01 is an explicit design constraint: gradient masking happens after `.backward()`, not via `.detach()` in the forward pass
- Exhaustive registry validation is a correctness guarantee, not just a nicety — if any parameter slips through unclassified, SGTM semantics break silently

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 02-model-architecture-masking*
*Context gathered: 2026-03-16*
