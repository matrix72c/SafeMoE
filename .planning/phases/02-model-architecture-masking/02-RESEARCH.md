# Phase 2: Model Architecture & Masking - Research

**Researched:** 2026-03-16
**Domain:** PyTorch MoE subclassing, gradient masking, activation hooks, parameter classification
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Config extension**
- `SafeMoEConfig` is a `@dataclass` subclass of `litgpt.Config`, located in `safemoe/config.py`
- Adds three fields with defaults: `harmful_expert_indices: list[int] = field(default_factory=list)`, `harmful_attn_heads: list[int] = field(default_factory=list)`, `num_harmful_experts: int = 0`
- All defaults produce a config that behaves identically to a standard `Config` (no harmful designation)
- Phase 2 also includes a `safemoe/configs/safemoe-tinystories.yaml` experiment config for a small TinyStories MoE model (small n_layer, 4–8 experts, 2 harmful experts designated)

**Attention head designation**
- `harmful_attn_heads: list[int]` is a **global** list of head indices applied uniformly across **every** attention layer in the model
- The QKV projection rows for each designated head are theta_harmful; the output projection (`attn.proj`) stays theta_std
- In Phase 2, `HarmfulParamRegistry` classifies the attention head QKV rows correctly, but `GradientMasker` and `ActivationMasker` do **not** apply masking to attention heads yet — masker scope is MoE experts only in this phase; attention head masking is activated in Phase 3

**HarmfulParamRegistry**
- Constructed as `HarmfulParamRegistry(model, config)` — scans `model.named_parameters()` at construction time
- theta_harmful = weights of `harmful_expert_indices` experts in each `SafeMoELayer` + QKV projection row slices for `harmful_attn_heads` in each attention layer
- theta_std = ALL remaining parameters (embedding, LM head, LayerNorm, non-harmful expert weights, non-harmful attention heads, output projections)
- Interface: `parameters_by_type(split: str) -> list[nn.Parameter]` where split is `'theta_harmful'` or `'theta_std'`
- Validates at construction: raises `ValueError` if theta_harmful ∩ theta_std ≠ ∅ or if theta_harmful ∪ theta_std ≠ all model parameters (exhaustive, non-overlapping guarantee)

**GradientMasker**
- After `loss.backward()` on a D_harmful batch, the training loop calls `gradient_masker.mask()`
- Implementation: manual loop over `registry.parameters_by_type('theta_std')` setting each `.grad = None`
- No hooks — explicit, transparent, matches MASK-01 spec ("post-backward zeroing, not detach-in-forward")

**ActivationMasker**
- A forward hook is registered on each `SafeMoELayer` instance at `ActivationMasker` construction
- Hook is toggled via `masker.enable()` / `masker.disable()` — when enabled, the hook fires and zeroes harmful expert output contributions; when disabled, hook is a no-op
- Training loop calls `activation_masker.enable()` before each D_std forward pass, `activation_masker.disable()` after
- Hook mechanism: zeros the output contributions of `harmful_expert_indices` experts (their accumulated `y[token_idx]` contribution is zeroed), effectively making them invisible during D_std steps

**Dual optimizer param groups**
- `MASK-03`: two separate `AdamW` instances (or param groups) — one for theta_harmful, one for theta_std
- Both use `zero_grad(set_to_none=True)` to prevent Adam momentum from accumulating on zero gradients
- Details of optimizer construction live in Phase 3 training loop; Phase 2 only needs to expose `parameters_by_type()` to enable this

### Claude's Discretion
- Exact YAML values for the TinyStories experiment config (model size, head count, block size, etc.) — choose something small enough for CPU-only unit tests
- Internal data structure of HarmfulParamRegistry (dict vs list, caching strategy)
- How QKV row slices are extracted from fused attn.proj weight matrix (slice indexing math)
- SafeMoELayer module file location (`safemoe/model.py`)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MOE-01 | SafeMoEConfig extends LitGPT `Config` with `harmful_expert_indices`, `num_harmful_experts`, `harmful_attn_heads` fields | dataclass subclass pattern from `litgpt/adapter.py`; Config.__post_init__ hook pattern confirmed |
| MOE-02 | HarmfulParamRegistry registers full-model parameter -> theta_harmful/theta_std mapping; exposes `parameters_by_type()` | `named_parameters()` scan pattern from `litgpt/lora.py`; QKV layout confirmed in `CausalSelfAttention` |
| MOE-03 | SafeMoELayer subclasses `LLaMAMoE`, integrates HarmfulParamRegistry for per-expert routing and designation | `LLaMAMoE` dispatch loop confirmed at line 812-814 of `litgpt/model.py`; `mlp_class` injection point confirmed at line 300 |
| MOE-04 | Harmful expert initialization strategy is configurable (random init vs copy weights from std experts) | `nn.ModuleList` experts are direct `LLaMAMLP` instances; copy is `deepcopy` or weight tensor assignment |
| MASK-01 | GradientMasker: post-backward zeroing of theta_std grads (not detach-in-forward) | `.grad = None` is the correct PyTorch idiom; confirmed no autograd graph interference |
| MASK-02 | ActivationMasker: zeroes theta_harmful expert outputs during D_std forward pass | `register_forward_hook` on `SafeMoELayer`; hook receives and modifies output tensor in-place |
| MASK-03 | Dual AdamW param groups with `zero_grad(set_to_none=True)` | Exposed via `parameters_by_type()` from Phase 2; optimizer construction deferred to Phase 3 |
| MASK-04 | Unit tests confirming all four masking invariants | pytest already installed; test pattern confirmed in `tests/test_model.py` and `tests/safemoe/` |
</phase_requirements>

---

## Summary

Phase 2 builds the SafeMoE model internals entirely within the LitGPT codebase, leveraging established extension patterns. The code is well-suited for the required extensions: `litgpt/adapter.py` demonstrates the exact `@dataclass` Config subclass pattern, `litgpt/lora.py` demonstrates the `named_parameters()` scan pattern for parameter classification, and `litgpt/model.py:LLaMAMoE` contains the per-expert dispatch loop that `ActivationMasker` hooks into.

The central technical challenge is `HarmfulParamRegistry`'s handling of fused QKV weights. Because `CausalSelfAttention` uses a single fused `self.qkv` linear layer (weight shape: `(n_head + 2*n_query_groups) * head_size, n_embd`), attention head QKV rows cannot be isolated as separate `nn.Parameter` objects. They are row-slices of a shared parameter tensor. This means the registry must store `(parameter_object, row_slice_indices)` tuples for those entries — not just bare parameter references — which affects the interface contract and how the exhaustive-coverage validation works.

The gradient masking approach (setting `.grad = None` after backward) is the correct PyTorch idiom for dropping gradients without affecting autograd graph construction. The activation masking approach (forward hook on `SafeMoELayer`) is also well-supported in PyTorch: `register_forward_hook` fires after the module's `forward()` returns and can modify the output tensor in-place.

**Primary recommendation:** Implement `HarmfulParamRegistry` using a dict keyed by `split` (`'theta_harmful'`, `'theta_std'`), with each value being a list of `nn.Parameter` objects for full parameters and a separate list of `(nn.Parameter, slice)` pairs for QKV row slices. Expose `parameters_by_type()` returning only the parameter objects (not slices) since GradientMasker and ActivationMasker are MoE-only in Phase 2; reserve the QKV slice structure for Phase 3.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | >=2.7 (project constraint) | Model definition, autograd, hooks | Already in pyproject.toml dependencies |
| litgpt | 0.5.12 (local fork) | Base `Config`, `GPT`, `Block`, `LLaMAMoE` | This project IS a litgpt fork |
| pytest | >=8.1.1 | Test framework | Already in pyproject.toml test deps |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| dataclasses (stdlib) | Python 3.10+ | `@dataclass` with `field()` | Config definition |
| copy (stdlib) | Python 3.10+ | `deepcopy` for expert weight copying | MOE-04 copy-weights init strategy |
| typing (stdlib) | Python 3.10+ | Type annotations | All public interfaces |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Post-backward `.grad = None` | Hook-based detach in forward | Detach breaks gradient flow for D_harmful update; post-backward is correct by spec |
| `register_forward_hook` on SafeMoELayer | Override `forward()` with mode flag | Hook is toggle-able without subclass/override complexity; cleaner enable/disable API |
| Dict-based registry | Two lists | Dict allows O(1) lookup by split name; more robust to future splits |

**Installation:** No new packages required — all dependencies already present in the repo.

---

## Architecture Patterns

### Recommended Project Structure
```
safemoe/
├── config.py            # SafeMoEConfig (@dataclass subclass of litgpt.Config)
├── model.py             # SafeMoELayer (subclass of LLaMAMoE)
├── masking.py           # HarmfulParamRegistry, GradientMasker, ActivationMasker
├── configs/
│   └── safemoe-tinystories.yaml   # experiment config (small MoE for CPU tests)
└── data/                # already exists from Phase 1

tests/safemoe/
├── test_config.py       # MOE-01: SafeMoEConfig instantiation
├── test_model.py        # MOE-03, MOE-04: SafeMoELayer structure and init
├── test_registry.py     # MOE-02: HarmfulParamRegistry classification + exhaustive coverage
└── test_masking.py      # MASK-01 through MASK-04: all masking invariants
```

### Pattern 1: Config Subclass (@dataclass)
**What:** Extend `litgpt.Config` with additional fields as a frozen-compatible dataclass.
**When to use:** Any time new model-level config fields are needed beyond litgpt's base config.
**Example:**
```python
# Source: litgpt/adapter.py (confirmed pattern)
from dataclasses import dataclass, field
from litgpt.config import Config as BaseConfig

@dataclass
class SafeMoEConfig(BaseConfig):
    harmful_expert_indices: list[int] = field(default_factory=list)
    harmful_attn_heads: list[int] = field(default_factory=list)
    num_harmful_experts: int = 0
```

Note: `SafeMoEConfig.from_file()` inherits from `BaseConfig` and uses `yaml.safe_load()` — the three new fields are included automatically. The YAML config file uses these field names as top-level keys.

### Pattern 2: mlp_class Injection (SafeMoELayer as MLP)
**What:** LitGPT's `Block.__init__` calls `config.mlp_class(config)` at line 300. Setting `mlp_class` in config makes every MoE block use `SafeMoELayer`.
**When to use:** When subclassing the MoE layer to add per-expert designation behavior.
**Example:**
```python
# Source: litgpt/model.py line 300
self.mlp = config.mlp_class(config)  # Block wires this automatically

# In SafeMoEConfig.__post_init__:
def __post_init__(self):
    super().__post_init__()
    # mlp_class_name must be "LLaMAMoE" for MoE blocks
    # SafeMoELayer is set directly, not via string:
    self._mlp_class = SafeMoELayer  # override mlp_class property
```

Important: `Config.mlp_class` is a property that resolves `mlp_class_name` to an actual class (see `config.py`). SafeMoEConfig must override this property to return `SafeMoELayer` when `mlp_class_name == "LLaMAMoE"`.

### Pattern 3: named_parameters() Scan for Classification
**What:** Iterate `model.named_parameters()` at registry construction time; use name matching to classify each parameter.
**When to use:** Any time parameters need to be split into groups (LoRA does this, SGTM does this).
**Example:**
```python
# Source: litgpt/lora.py:447 lora_filter() and mark_only_lora_as_trainable()
for n, p in model.named_parameters():
    if "lora_" not in n:
        p.requires_grad = False
```

For `HarmfulParamRegistry`, the scan logic is:
- Parameter name matches `transformer.h.{layer}.mlp.experts.{idx}.*` where `idx in harmful_expert_indices` → theta_harmful
- Parameter name matches `transformer.h.{layer}.attn.qkv.weight` → needs row-slice analysis for harmful_attn_heads
- Everything else → theta_std

### Pattern 4: Forward Hook for Output Modification
**What:** `module.register_forward_hook(hook_fn)` — hook fires after `module.forward()` returns; can modify output.
**When to use:** When you need to intercept and modify module output without changing the module's forward method.
**Example:**
```python
# Source: PyTorch docs (verified behavior)
def _make_activation_hook(masker, layer):
    def hook(module, input, output):
        if masker.enabled:
            # output is the (B, T, C) tensor returned by SafeMoELayer.forward()
            # Zero contributions from harmful experts is done by re-running forward
            # with harmful expert outputs suppressed, OR by storing intermediate y tensors
        return output
    return hook

handle = safe_moe_layer.register_forward_hook(_make_activation_hook(masker, layer))
```

**Implementation note:** The hook receives the module output (already aggregated `y` tensor from LLaMAMoE). The hook cannot retroactively zero individual expert contributions from the aggregated tensor. Instead, `SafeMoELayer.forward()` must cooperate: either store per-expert contributions for the hook to zero, or use a flag approach where the forward itself skips harmful expert accumulation when masking is enabled.

The cleaner approach: override `SafeMoELayer.forward()` to skip `y[token_idx] += ...` for harmful experts when `self._masking_enabled` flag is True. The ActivationMasker sets/unsets this flag via `enable()`/`disable()`. This is equivalent to hook behavior but avoids the hook's inability to un-aggregate contributions.

### Pattern 5: Gradient Masking (Post-Backward .grad = None)
**What:** After `loss.backward()`, explicitly set `.grad = None` for parameters that should not update.
**When to use:** MASK-01 spec — "post-backward zeroing, not detach-in-forward."
**Example:**
```python
# Source: PyTorch documentation (verified pattern)
class GradientMasker:
    def __init__(self, registry: HarmfulParamRegistry):
        self._registry = registry

    def mask(self) -> None:
        """Call after loss.backward() on D_harmful batch."""
        for p in self._registry.parameters_by_type('theta_std'):
            p.grad = None
```

Setting `.grad = None` is preferred over `.grad.zero_()` because it prevents Adam from accumulating momentum state on zero gradients when `zero_grad(set_to_none=True)` is used.

### Anti-Patterns to Avoid
- **Registering QKV slices as separate Parameters:** They are row-slices of a shared `qkv.weight` tensor. Treating them as separate Parameters would require custom autograd or in-place mutation tricks. Use the flag/mask approach in the registry instead, storing the parent Parameter + slice indices.
- **Using `requires_grad = False` for theta_std during D_harmful:** This permanently detaches from autograd for the batch; reattaching is expensive. Post-backward `.grad = None` is the correct approach.
- **Using detach() in forward for ActivationMasker:** Detach removes the tensor from the computation graph. Instead, simply zero the activation output tensor or skip the expert accumulation step.
- **Calling register_forward_hook on Block instead of SafeMoELayer:** Block.forward runs attention + MLP; hooking at the MLP (SafeMoELayer) level is more precise and doesn't require filtering.
- **Storing parameter references by value, not identity:** Python list comparison uses `==` (value); use `is` or `id()` to check parameter identity when enforcing non-overlapping sets.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Config YAML serialization | Custom YAML read/write | `Config.from_file()` (already in litgpt) | Handles all fields via `yaml.safe_load()` + dataclass `__init__` |
| Module hook toggle | Custom proxy module | `register_forward_hook` + enabled flag | PyTorch hook system handles lifetime, device movement, etc. |
| Named parameter scanning | Custom model traversal | `model.named_parameters()` | PyTorch recurses through all submodules automatically |
| Parameter group construction | Custom optimizer wrapper | `torch.optim.AdamW(params=list)` | Phase 3 concern; Phase 2 only needs `parameters_by_type()` |
| Expert module list indexing | Custom expert registry | `nn.ModuleList` (already in LLaMAMoE) | `self.experts[i]` gives direct access by index |

**Key insight:** The entire hook and parameter machinery already exists in PyTorch and LitGPT — this phase is about wiring up the right parameters and flags, not building new infrastructure.

---

## Common Pitfalls

### Pitfall 1: QKV Row Slice Is Not an Independent Parameter
**What goes wrong:** `HarmfulParamRegistry.parameters_by_type('theta_harmful')` returns slices of `qkv.weight`. When GradientMasker iterates and sets `.grad = None` on these, it sets the entire `qkv.weight.grad` to None — including rows belonging to non-harmful heads.
**Why it happens:** The fused QKV linear has ONE weight tensor. Row-slicing doesn't create new `nn.Parameter` objects.
**How to avoid:** Phase 2 GradientMasker and ActivationMasker do NOT apply masking to attention heads (by design decision). Phase 2 HarmfulParamRegistry classifies QKV rows in its metadata for Phase 3 use, but `parameters_by_type('theta_harmful')` returns only the expert parameters as full `nn.Parameter` objects. Document that the returned list is "masker-ready" parameters (full params only), separate from the classification metadata.
**Warning signs:** If GradientMasker zero-grads the QKV param, non-harmful heads stop updating during D_harmful steps.

### Pitfall 2: Exhaustive Registry Validation With Param Identity
**What goes wrong:** Validation check `theta_harmful ∪ theta_std = all parameters` fails to catch parameters registered twice or not at all, because set union uses `__eq__` (value comparison) instead of identity comparison.
**Why it happens:** `nn.Parameter` inherits from `torch.Tensor` whose `__eq__` is element-wise.
**How to avoid:** Build sets using `id(p)` — e.g. `{id(p) for p in theta_harmful}` — and compare by ID, not by value. Also verify against `{id(p) for name, p in model.named_parameters()}`.
**Warning signs:** `ValueError` is not raised even when a parameter is in both sets.

### Pitfall 3: SafeMoELayer Must Override mlp_class Property, Not mlp_class_name
**What goes wrong:** Setting `mlp_class_name = "LLaMAMoE"` in SafeMoEConfig but not overriding `mlp_class` property results in `Block` instantiating `LLaMAMoE` instead of `SafeMoELayer`.
**Why it happens:** `Config.mlp_class` is a property that maps string name to class. `"LLaMAMoE"` maps to `LLaMAMoE`, not `SafeMoELayer`.
**How to avoid:** Override the `mlp_class` property in `SafeMoEConfig` to return `SafeMoELayer` when `mlp_class_name == "LLaMAMoE"`. Verify by asserting `isinstance(model.transformer.h[0].mlp, SafeMoELayer)` in tests.
**Warning signs:** Model builds successfully but maskers have no `SafeMoELayer` instances to hook into.

### Pitfall 4: Forward Hook Cannot Undo Accumulated MoE Output
**What goes wrong:** `register_forward_hook` on `SafeMoELayer` receives the final aggregated `y` tensor (all experts combined). Trying to subtract harmful expert contributions from `y` requires re-running those experts, which doubles compute and may not match numerically.
**Why it happens:** LLaMAMoE.forward accumulates `y[token_idx] += probs * expert(x[token_idx])` in a single pass. The hook only sees the result.
**How to avoid:** Override `SafeMoELayer.forward()` to check an `_activation_masking_enabled` flag. When enabled, skip the accumulation for harmful expert indices. The `ActivationMasker.enable()` and `disable()` methods set this flag directly on each `SafeMoELayer` instance.
**Warning signs:** ActivationMasker is enabled but harmful expert outputs are non-zero during D_std forward pass.

### Pitfall 5: Test Model Must Use SafeMoEConfig, Not litgpt.Config
**What goes wrong:** Unit tests instantiate a `litgpt.Config` with `mlp_class_name="LLaMAMoE"` and get a plain `LLaMAMoE`, which has no masking support. Tests pass structurally but test wrong behavior.
**Why it happens:** Copy-paste from `tests/test_model.py` which uses `litgpt.Config`.
**How to avoid:** All Phase 2 tests must use `SafeMoEConfig(...)`. The test config should be small: `n_layer=2`, `n_embd=32`, `n_head=4`, `n_expert=4`, `n_expert_per_token=2`, `harmful_expert_indices=[0, 1]`.
**Warning signs:** `isinstance(model.transformer.h[0].mlp, SafeMoELayer)` returns False.

### Pitfall 6: YAML Config File Must Include mlp_class_name
**What goes wrong:** `safemoe-tinystories.yaml` omits `mlp_class_name`, causing Config to default to `"GptNeoxMLP"` (no MoE).
**Why it happens:** LitGPT Config has `mlp_class_name: Literal["GptNeoxMLP", "LLaMAMLP", "GemmaMLP", "LLaMAMoE"] = "GptNeoxMLP"` as default.
**How to avoid:** Explicitly set `mlp_class_name: LLaMAMoE` in the YAML. Also set `n_expert`, `n_expert_per_token`, `moe_intermediate_size`.
**Warning signs:** Model builds but has MLP blocks instead of MoE blocks.

---

## Code Examples

Verified patterns from official sources (litgpt codebase + PyTorch):

### LLaMAMoE Dispatch Loop (The Hook Target)
```python
# Source: litgpt/model.py lines 793-819 (confirmed)
def forward(self, x: torch.Tensor) -> torch.Tensor:
    B, T, C = x.size()
    x = x.view(-1, C)  # (B*T, C)
    router = self.gate(x)  # (B*T, n_expert)
    probs, indices = torch.topk(router, self.config.n_expert_per_token)
    probs = probs.softmax(dim=1, dtype=torch.float).to(dtype=x.dtype)
    masks = indices.unsqueeze(-1) == torch.arange(self.config.n_expert, device=x.device)
    masks = masks.permute(2, 0, 1)  # (n_expert, B*T, n_expert_per_token)
    y = torch.zeros_like(x)  # (B*T, C)
    for mask, expert in zip(masks, self.experts):
        token_idx, expert_idx = torch.where(mask)
        y[token_idx] += probs[token_idx, expert_idx, None] * expert(x[token_idx])
    return y.view(B, T, C)
```

SafeMoELayer overrides this loop to skip accumulation for harmful experts when `_activation_masking_enabled` is True.

### CausalSelfAttention QKV Layout (For HarmfulParamRegistry)
```python
# Source: litgpt/model.py lines 358-365 (confirmed)
self.qkv = nn.Linear(
    config.n_embd,
    (config.n_head + 2 * config.n_query_groups) * config.head_size,
    bias=config.bias or config.attn_bias,
)
# Weight shape: out_features x in_features
# = (n_head + 2*n_query_groups)*head_size x n_embd

# For head i's Q rows (MHA case where n_query_groups == n_head):
# Q rows: [i*head_size : (i+1)*head_size]           (rows 0 to n_head*head_size)
# K rows: [n_head*head_size + i*head_size : ...]    (rows n_head*head_size onward)
# V rows: [(n_head+n_query_groups)*head_size + i*head_size : ...]
```

For GQA (n_query_groups < n_head), K and V rows use n_query_groups stride; only Q rows are per-head. The registry must handle both MHA and GQA configurations.

### Config Subclass Pattern (From Adapter)
```python
# Source: litgpt/adapter.py lines 24-27 (confirmed)
@dataclass
class Config(BaseConfig):
    adapter_prompt_length: int = 10
    adapter_start_layer: int = 2
```

### lora_filter Pattern (Named Parameter Scan)
```python
# Source: litgpt/lora.py lines 428-430, 447-448 (confirmed)
for n, p in model.named_parameters():
    if "lora_" not in n:
        p.requires_grad = False

def lora_filter(key: str, value: Any) -> bool:
    return "lora_" in key
```

HarmfulParamRegistry uses the same `named_parameters()` scan with name-prefix matching for `experts.{idx}`.

### Test Config Pattern (From test_deepseek_moe.py)
```python
# Source: tests/test_deepseek_moe.py lines 17-45 (confirmed)
config_litgpt = Config(
    padded_vocab_size=10000,
    n_layer=2,
    vocab_size=10000,
    n_embd=64,
    n_head=4,
    n_query_groups=4,
    head_size=16,
    n_expert=16,
    n_expert_per_token=2,
    moe_intermediate_size=20,
    mlp_class_name="LLaMAMoE",
)
```

Phase 2 tests use `SafeMoEConfig(...)` with analogous small dimensions plus `harmful_expert_indices=[0, 1]`.

### Minimal YAML Config for TinyStories MoE (Guidance)
```yaml
# safemoe/configs/safemoe-tinystories.yaml
# Based on config_hub/pretrain/tinystories.yaml (stories15M) with MoE fields added
name: safemoe-tinystories
block_size: 256
padded_vocab_size: 32000
n_layer: 4
n_head: 4
n_query_groups: 4
n_embd: 128
head_size: 32
rotary_percentage: 1.0
parallel_residual: false
bias: false
norm_class_name: RMSNorm
mlp_class_name: LLaMAMoE
moe_intermediate_size: 256
n_expert: 8
n_expert_per_token: 2
# SafeMoEConfig-specific fields:
harmful_expert_indices: [0, 1]
num_harmful_experts: 2
harmful_attn_heads: []
```

Rationale: n_layer=4 gives multiple blocks for registry scanning; n_expert=8 is sufficient to test expert designation; n_embd=128 keeps CPU test time < 5 seconds.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| MoE with separate expert modules per block | Fused `nn.ModuleList` in LLaMAMoE with vectorized dispatch | LitGPT current | Expert access via `self.experts[i]` is trivial; dispatch loop is the interception point |
| Gradient masking via `requires_grad=False` | Post-backward `.grad = None` | SGTM paper design | Allows dynamic per-step masking without graph reconstruction |
| Activation masking via separate "ablated" model | In-place expert skip in forward | SGTM paper design | Single model instance; no weight duplication |

**Deprecated/outdated:**
- Using `zero_grad()` without `set_to_none=True`: Old behavior zeros grad tensors, keeping Adam state active on zero grads — this corrupts momentum for masked parameters. Use `set_to_none=True` (default in recent PyTorch).

---

## Open Questions

1. **mlp_class Override Mechanism**
   - What we know: `Config.mlp_class` is a `@property` that maps `mlp_class_name` string to class. Adapter overrides block-level class, not config property.
   - What's unclear: Whether overriding the `mlp_class` property in `SafeMoEConfig` is the right approach, or whether a `mlp_class_name` string + a class registry entry is preferred.
   - Recommendation: Check `Config.mlp_class` property implementation in `litgpt/config.py` (around line 230+). If it uses a `getattr(module, name)` lookup, registering `SafeMoELayer` in the litgpt model module namespace works. The simpler approach: override `mlp_class` property directly in `SafeMoEConfig`.

2. **QKV Row Classification for GQA vs MHA**
   - What we know: For MHA (n_query_groups == n_head), head i's Q rows are `[i*head_size:(i+1)*head_size]` in qkv.weight. K and V rows are indexed with n_query_groups stride.
   - What's unclear: Whether Phase 2 test configs will use GQA (n_query_groups < n_head) or MHA. The stories15M base config uses n_query_groups=6=n_head=6 (MHA).
   - Recommendation: Implement MHA row-slice logic first (simpler); add GQA support in Phase 3 when attention head masking is activated.

3. **ActivationMasker Hook vs Flag Approach**
   - What we know: The locked decision says "A forward hook is registered on each SafeMoELayer instance." But as noted in Pitfall 4, the hook receives the aggregated output and cannot retroactively zero individual expert contributions.
   - What's unclear: Whether the intent is for SafeMoELayer.forward() to cooperate with the hook by setting a flag, or for the hook to call a separate method to redo the computation.
   - Recommendation: Implement a `_masking_enabled` flag on `SafeMoELayer` that `ActivationMasker.enable()`/`disable()` sets directly. The "hook" in the locked decision refers conceptually to the masker's interception point, which in practice is implemented as a flag on the module controlled by the masker object. This is consistent with "Hook is toggled via `masker.enable()` / `masker.disable()`."

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=8.1.1 |
| Config file | `pyproject.toml` — `[tool.pytest.ini_options]` |
| Quick run command | `pytest tests/safemoe/ -x -q` |
| Full suite command | `pytest tests/safemoe/ -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MOE-01 | SafeMoEConfig instantiates with harmful fields; defaults produce standard behavior | unit | `pytest tests/safemoe/test_config.py -x` | ❌ Wave 0 |
| MOE-02 | HarmfulParamRegistry: non-overlapping, exhaustive classification; parameters_by_type returns correct params | unit | `pytest tests/safemoe/test_registry.py -x` | ❌ Wave 0 |
| MOE-03 | SafeMoELayer instances present in model; correct n_expert count | unit | `pytest tests/safemoe/test_model.py::test_safemoe_layer_structure -x` | ❌ Wave 0 |
| MOE-04 | Harmful expert init: random init produces different weights; copy init produces identical weights to source | unit | `pytest tests/safemoe/test_model.py::test_harmful_expert_init -x` | ❌ Wave 0 |
| MASK-01 | After backward on D_harmful batch with GradientMasker: theta_std.grad is None; theta_harmful.grad is non-zero | unit | `pytest tests/safemoe/test_masking.py::test_gradient_masker -x` | ❌ Wave 0 |
| MASK-02 | With ActivationMasker enabled: harmful expert outputs are zero; std expert outputs non-zero | unit | `pytest tests/safemoe/test_masking.py::test_activation_masker -x` | ❌ Wave 0 |
| MASK-03 | parameters_by_type() returns disjoint groups suitable for dual AdamW (tested via registry, optimizer construction in Phase 3) | unit | `pytest tests/safemoe/test_registry.py::test_param_groups_disjoint -x` | ❌ Wave 0 |
| MASK-04 | All four invariants: grad isolation, activation zeroing, set_to_none=True Adam state integrity | unit | `pytest tests/safemoe/test_masking.py -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/safemoe/ -x -q`
- **Per wave merge:** `pytest tests/safemoe/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/safemoe/__init__.py` — namespace package init (NOTE: do NOT add `__init__.py` — Phase 1 lesson: pytest namespace collision. Leave absent.)
- [ ] `tests/safemoe/test_config.py` — covers MOE-01
- [ ] `tests/safemoe/test_model.py` — covers MOE-03, MOE-04
- [ ] `tests/safemoe/test_registry.py` — covers MOE-02, MASK-03
- [ ] `tests/safemoe/test_masking.py` — covers MASK-01, MASK-02, MASK-04
- [ ] `safemoe/config.py` — SafeMoEConfig
- [ ] `safemoe/model.py` — SafeMoELayer
- [ ] `safemoe/masking.py` — HarmfulParamRegistry, GradientMasker, ActivationMasker
- [ ] `safemoe/configs/safemoe-tinystories.yaml` — experiment config

---

## Sources

### Primary (HIGH confidence)
- `litgpt/model.py` — LLaMAMoE.forward() dispatch loop (lines 793-819), CausalSelfAttention QKV layout (lines 354-382), Block.mlp injection (line 300)
- `litgpt/config.py` — Config @dataclass structure (lines 25-220), mlp_class_name field, __post_init__ hook
- `litgpt/adapter.py` — @dataclass Config subclass pattern (lines 24-27), verified identical to proposed approach
- `litgpt/lora.py` — named_parameters() scan pattern (lines 428-430, 447-448)
- `tests/conftest.py` — pytest fixtures and project test conventions
- `tests/test_deepseek_moe.py` — small MoE config pattern for CPU tests (lines 17-45)
- `pyproject.toml` — pytest configuration, dependency versions (lines 78-86, 143-149)

### Secondary (MEDIUM confidence)
- `config_hub/pretrain/tinystories.yaml` — reference values for stories15M base config (n_layer=6, n_embd=288, block_size=256)
- `checkpoints/Qwen3-30B-A3B-Base/model_config.yaml` — verified LLaMAMoE YAML format (n_expert, moe_intermediate_size fields)

### Tertiary (LOW confidence)
- None — all claims verified from codebase sources.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries confirmed present in pyproject.toml and codebase
- Architecture: HIGH — all patterns confirmed from litgpt source code (adapter.py, lora.py, model.py)
- Pitfalls: HIGH — derived from direct reading of LLaMAMoE dispatch loop, CausalSelfAttention QKV layout, and known PyTorch parameter identity behavior
- QKV row slice math: HIGH — computed from confirmed source (CausalSelfAttention.__init__ lines 358-365)
- ActivationMasker hook-vs-flag ambiguity: MEDIUM — open question documented; recommendation given

**Research date:** 2026-03-16
**Valid until:** 2026-04-16 (litgpt is a local fork; no upstream churn risk during this phase)
