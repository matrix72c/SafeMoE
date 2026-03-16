---
phase: 02-model-architecture-masking
verified: 2026-03-16T06:00:00Z
status: passed
score: 8/8 must-haves verified
gaps: []
human_verification: []
---

# Phase 2: Model Architecture & Masking Verification Report

**Phase Goal:** Implement SafeMoE model architecture with harmful-parameter masking infrastructure
**Verified:** 2026-03-16T06:00:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

All must-haves are derived from the four PLAN frontmatter `must_haves` blocks (plans 01–04) and the
eight requirement IDs: MOE-01, MOE-02, MOE-03, MOE-04, MASK-01, MASK-02, MASK-03, MASK-04.

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | SafeMoEConfig stores harmful_expert_indices, num_harmful_experts, harmful_attn_heads and inherits litgpt.Config | VERIFIED | `safemoe/config.py` — @dataclass subclass of BaseConfig; 4/4 test_config.py tests pass |
| 2 | SafeMoEConfig.mlp_class returns SafeMoELayer (not LLaMAMoE) when mlp_class_name="LLaMAMoE" | VERIFIED | lazy import in mlp_class property; `test_safemoe_config_mlp_class_returns_safemoe_layer` passes |
| 3 | GPT model built from SafeMoEConfig has SafeMoELayer in every transformer block's .mlp attribute | VERIFIED | `test_safemoe_layer_structure` passes; Block.__init__ calls config.mlp_class(config) |
| 4 | SafeMoELayer has _activation_masking_enabled flag (default False) | VERIFIED | line 38 of model.py; `test_activation_masking_flag_exists` passes |
| 5 | Harmful expert init "random" produces distinct weights; "copy" produces identical weights from first std expert | VERIFIED | `test_harmful_expert_init_random` and `test_harmful_expert_init_copy` both pass |
| 6 | HarmfulParamRegistry classifies all params as theta_harmful or theta_std (exhaustive, non-overlapping) with ValueError on invalid config | VERIFIED | 5/5 test_registry.py tests pass; id()-based validation in masking.py |
| 7 | GradientMasker.mask() sets theta_std .grad=None post-backward; theta_harmful grads remain non-None | VERIFIED | `test_gradient_masker_zeroes_theta_std_grads` passes; loop at masking.py line 179 |
| 8 | ActivationMasker.enable()/disable() toggle _activation_masking_enabled on all SafeMoELayer instances; masking isolation confirmed across combined D_harmful+D_std step; dual AdamW with set_to_none=True accumulates no std momentum state | VERIFIED | `test_activation_masker_zeroes_harmful_expert_output`, `test_masking_invariants_combined`, `test_set_to_none_adam_state_integrity` all pass |

**Score: 8/8 truths verified**

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tests/safemoe/test_config.py` | RED stubs for MOE-01 | VERIFIED | Exists, 4 tests, imports SafeMoEConfig/SafeMoELayer |
| `tests/safemoe/test_model.py` | RED stubs for MOE-03/MOE-04 | VERIFIED | Exists, 4 tests, imports SafeMoEConfig/SafeMoELayer |
| `tests/safemoe/test_registry.py` | RED stubs for MOE-02/MASK-03 | VERIFIED | Exists, 5 tests, imports HarmfulParamRegistry |
| `tests/safemoe/test_masking.py` | RED stubs for MASK-01/02/04 | VERIFIED | Exists, 4 tests, imports GradientMasker/ActivationMasker |
| `safemoe/configs/safemoe-tinystories.yaml` | Valid YAML with mlp_class_name: LLaMAMoE, n_expert: 8, harmful_expert_indices: [0, 1] | VERIFIED | Parsed OK; all three required fields present |
| `safemoe/config.py` | SafeMoEConfig @dataclass subclass of litgpt.Config | VERIFIED | 41 lines; exports SafeMoEConfig; contains harmful_expert_indices, harmful_attn_heads, num_harmful_experts; mlp_class property override |
| `safemoe/model.py` | SafeMoELayer subclass of LLaMAMoE with activation masking | VERIFIED | 93 lines; exports SafeMoELayer; contains _activation_masking_enabled; forward() skips harmful experts when flag True; init_strategy parameter handles "random"/"copy" |
| `safemoe/masking.py` | HarmfulParamRegistry + GradientMasker + ActivationMasker fully implemented | VERIFIED | 230 lines; no NotImplementedError stubs; all three classes exported |
| `tests/safemoe/__init__.py` | Must NOT exist (namespace collision prevention) | VERIFIED | File is absent |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `safemoe/config.py SafeMoEConfig` | `safemoe/model.py SafeMoELayer` | mlp_class property — lazy import returns SafeMoELayer | WIRED | Pattern `mlp_class.*SafeMoELayer` confirmed at lines 37–39 of config.py |
| `litgpt/model.py Block.__init__` | `safemoe/model.py SafeMoELayer` | Block calls config.mlp_class(config) | WIRED | SafeMoEConfig.mlp_class returns SafeMoELayer; test_safemoe_layer_structure confirms every block.mlp is SafeMoELayer |
| `safemoe/masking.py HarmfulParamRegistry` | `litgpt.GPT model.named_parameters()` | named_parameters() scan at construction | WIRED | `_EXPERT_RE` pattern scan across all named_parameters; 24 theta_harmful + 46 theta_std params classified for SMALL_CONFIG |
| `safemoe/masking.py HarmfulParamRegistry` | `SafeMoEConfig.harmful_expert_indices` | name matching on transformer.h.*.mlp.experts.{idx}.* | WIRED | Pattern at masking.py line 75: `int(m_expert.group(1)) in harmful_expert_indices` |
| `safemoe/masking.py GradientMasker.mask()` | `HarmfulParamRegistry.parameters_by_type('theta_std')` | iterates theta_std and sets .grad = None | WIRED | masking.py line 179: `for p in self._registry.parameters_by_type("theta_std"): p.grad = None` |
| `safemoe/masking.py ActivationMasker.enable()` | `safemoe/model.py SafeMoELayer._activation_masking_enabled` | sets flag on each SafeMoELayer instance in model.modules() | WIRED | masking.py lines 218–219; lazy import of SafeMoELayer in `__init__` at line 205 |
| `safemoe/masking.py HarmfulParamRegistry._qkv_harmful_metadata` | Phase 3 attention head masking | stored as list of (nn.Parameter, list[slice]) tuples | WIRED (Phase 3 receiver) | masking.py lines 130–132; empty list when harmful_attn_heads=[]; populated when non-empty |

---

### Requirements Coverage

| Requirement | Source Plan(s) | Description | Status | Evidence |
|-------------|---------------|-------------|--------|----------|
| MOE-01 | 02-01, 02-02 | SafeMoEConfig extends litgpt.Config with harmful_expert_indices, num_harmful_experts, harmful_attn_heads | SATISFIED | safemoe/config.py; 4/4 test_config.py pass |
| MOE-02 | 02-01, 02-03 | HarmfulParamRegistry — parameter -> theta_harmful/theta_std mapping; parameters_by_type() interface | SATISFIED | safemoe/masking.py HarmfulParamRegistry; 5/5 test_registry.py pass |
| MOE-03 | 02-01, 02-02 | SafeMoELayer subclasses LLaMAMoE; integrates with HarmfulParamRegistry for per-expert routing | SATISFIED | safemoe/model.py; test_safemoe_layer_structure passes; all blocks confirmed SafeMoELayer |
| MOE-04 | 02-01, 02-02 | Harmful expert init configurable (random vs copy weights) | SATISFIED | SafeMoELayer init_strategy parameter; test_harmful_expert_init_random + test_harmful_expert_init_copy pass |
| MASK-01 | 02-01, 02-04 | GradientMasker — post-backward .grad=None for theta_std (not detach-in-forward) | SATISFIED | GradientMasker.mask() loop; test_gradient_masker_zeroes_theta_std_grads passes |
| MASK-02 | 02-01, 02-04 | ActivationMasker — zeros theta_harmful expert outputs during D_std forward | SATISFIED | ActivationMasker.enable()/disable(); SafeMoELayer.forward() skip logic; test_activation_masker_zeroes_harmful_expert_output passes |
| MASK-03 | 02-01, 02-03 | Dual optimizer param groups (separate AdamW for theta_harmful and theta_std) with zero_grad(set_to_none=True) | SATISFIED | HarmfulParamRegistry.parameters_by_type() returns disjoint lists for AdamW construction; test_param_groups_disjoint and test_set_to_none_adam_state_integrity pass; manual integration confirmed dual AdamW construction with 24 harmful + 46 std params |
| MASK-04 | 02-01, 02-04 | Unit tests confirming: grad=None for masked params after D_harmful backward; grad>0 for unmasked; harmful output=0 during D_std | SATISFIED | test_masking_invariants_combined + test_set_to_none_adam_state_integrity pass; ActivationMasker skips harmful experts in forward |

All 8 phase requirements satisfied. No orphaned requirements (REQUIREMENTS.md traceability table maps all 8 IDs to Phase 2).

---

### Anti-Patterns Found

No anti-patterns detected in any implementation file.

| File | Pattern | Severity | Notes |
|------|---------|----------|-------|
| `safemoe/config.py` | — | — | Clean; no TODOs, stubs, or empty returns |
| `safemoe/model.py` | — | — | Clean; no TODOs, stubs, or empty returns |
| `safemoe/masking.py` | — | — | Clean; GradientMasker and ActivationMasker fully implemented (no NotImplementedError) |

One noteworthy design decision documented in SUMMARY 02-04: the integration-script assertion
`all(p.grad is not None for p in theta_harmful)` is overly strict — with MoE routing, not all
harmful experts are activated on every forward pass. The actual test (`test_gradient_masker_zeroes_theta_std_grads`)
correctly uses `len(non_none_harmful) > 0`. This is documented, not a code defect.

---

### Human Verification Required

None. All behavioral invariants are verifiable programmatically via the test suite, and 17/17 tests
pass. No visual UI, real-time, or external service behavior involved.

---

### Commit Verification

All claimed commits exist in git log:
- `f302ad3` — test(02-01): RED stubs for SafeMoEConfig
- `ce2dbdc` — test(02-01): RED stubs for SafeMoELayer + YAML
- `6b80a2b` — test(02-01): RED stubs for HarmfulParamRegistry
- `4c30582` — test(02-01): RED stubs for GradientMasker and ActivationMasker
- `29ec526` — feat(02-02): implement SafeMoEConfig
- `c1b5307` — feat(02-02): implement SafeMoELayer
- `55a088f` — feat(02-03): implement HarmfulParamRegistry (+ masker stubs)
- `b5c74a3` — feat(02-04): implement GradientMasker and ActivationMasker

---

### Full Test Suite Result

```
tests/safemoe/test_config.py::test_safemoe_config_has_harmful_fields        PASSED
tests/safemoe/test_config.py::test_safemoe_config_defaults_safe              PASSED
tests/safemoe/test_config.py::test_safemoe_config_inherits_litgpt_config     PASSED
tests/safemoe/test_config.py::test_safemoe_config_mlp_class_returns_safemoe_layer PASSED
tests/safemoe/test_model.py::test_safemoe_layer_structure                    PASSED
tests/safemoe/test_model.py::test_harmful_expert_init_random                 PASSED
tests/safemoe/test_model.py::test_harmful_expert_init_copy                   PASSED
tests/safemoe/test_model.py::test_activation_masking_flag_exists             PASSED
tests/safemoe/test_registry.py::test_registry_theta_harmful_contains_expert_params PASSED
tests/safemoe/test_registry.py::test_registry_exhaustive_coverage            PASSED
tests/safemoe/test_registry.py::test_registry_non_overlapping                PASSED
tests/safemoe/test_registry.py::test_registry_raises_on_invalid_config       PASSED
tests/safemoe/test_registry.py::test_param_groups_disjoint                   PASSED
tests/safemoe/test_masking.py::test_gradient_masker_zeroes_theta_std_grads   PASSED
tests/safemoe/test_masking.py::test_activation_masker_zeroes_harmful_expert_output PASSED
tests/safemoe/test_masking.py::test_masking_invariants_combined              PASSED
tests/safemoe/test_masking.py::test_set_to_none_adam_state_integrity         PASSED

17 passed in 3.50s
```

Note: `tests/safemoe/data/` tests fail with `ModuleNotFoundError: No module named 'litdata'` —
those are Phase 1 data pipeline tests unrelated to Phase 2 scope and excluded from this suite.

---

_Verified: 2026-03-16T06:00:00Z_
_Verifier: Claude (gsd-verifier)_
