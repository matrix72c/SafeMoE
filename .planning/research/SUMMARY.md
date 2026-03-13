# Research Summary: SafeMoE

**Domain:** MoE Safety Research -- Knowledge Isolation via SGTM
**Researched:** 2026-03-13
**Overall confidence:** HIGH

## Executive Summary

SafeMoE builds on LitGPT, which already ships a fully functional Mixture-of-Experts implementation (`LLaMAMoE`) that is tested against HuggingFace Mixtral and DeepSeek V3. The installed environment (PyTorch 2.10+cu128, Lightning 2.6.1, Triton 3.6) provides all the APIs needed for SGTM: `register_post_accumulate_grad_hook` for selective gradient masking, `register_forward_hook` for activation masking, and `torch.compile` compatibility with both. All three mechanisms were verified experimentally in the live environment.

The critical finding is that **no external MoE library is needed.** megablocks, scattermoe, and tutel all optimize expert dispatch via opaque fused kernels that would hide the per-expert granularity SafeMoE requires for masking and ablation. LitGPT's loop-over-experts implementation in `LLaMAMoE` is computationally adequate for research-scale experiments and provides exact the individually-addressable expert access that SGTM demands.

The biggest technical risks are not in the stack but in the algorithm implementation: gradient hook ordering with gradient accumulation (Pitfall 1), router collapse without load-balancing loss (Pitfall 2), optimizer state corruption from zero-gradient masking (Pitfall 6), and evaluation confounds where ablation perplexity measures routing disruption rather than knowledge isolation (Pitfall 7). The recommended mitigation is to use explicit post-backward gradient zeroing (not hooks during backward), separate optimizer param groups for harmful/standard parameters, and always complement perplexity metrics with routing attribution analysis.

The architecture pattern is clear: create a `safemoe/` module that extends LitGPT without modifying it. `SafeMoELayer` subclasses `LLaMAMoE`, a custom `SGTMDataModule` extends `DataModule` for three-stream data loading, and `safemoe/pretrain.py` forks the training loop from `litgpt/pretrain.py` to add SGTM branching per domain label.

## Key Findings

**Stack:** No new dependencies required. PyTorch 2.10 + Lightning 2.6.1 + LitGPT 0.5.12 (existing `LLaMAMoE`) provides everything. Avoid megablocks/scattermoe/tutel -- they hide the expert-level access SGTM needs.

**Architecture:** Extend LitGPT via `safemoe/` module. Config-driven MLP class selection (existing pattern), forked training loop, single-domain batches with domain labels as parallel tensor.

**Critical pitfall:** Gradient masking with Adam optimizer. Setting gradients to zero (instead of None) corrupts optimizer momentum/variance estimates. Use separate optimizer param groups or set grad to None for masked parameters.

## Implications for Roadmap

Based on research, suggested phase structure:

1. **Foundation & MoE Layer** - Build SafeMoEConfig, ExpertRegistry, SafeMoELayer (subclass of LLaMAMoE)
   - Addresses: MoE model with designatable experts, harmful expert indices
   - Avoids: Building from scratch (leverages tested LLaMAMoE)

2. **Masking Infrastructure** - GradientMasker + ActivationMasker + unit tests
   - Addresses: SGTM gradient masking, forward-pass activation masking
   - Avoids: Pitfall 1 (hook ordering), Pitfall 6 (optimizer state corruption) -- test these explicitly before integration

3. **Data Pipeline** - BilingualTinyStories DataModule with domain labels
   - Addresses: Three-stream data sampling, configurable x% unlabeled contamination
   - Avoids: Pitfall 8 (mixed-batch artifacts) -- single-domain batches by design

4. **SGTM Training Loop** - Fork pretrain.py with SGTM branching
   - Addresses: Full SGTM algorithm, training infrastructure
   - Avoids: Pitfall 12 (multi-objective loop complexity) -- clean fork, not patch

5. **Evaluation & Ablation** - Per-language perplexity, routing attribution, ablation utility
   - Addresses: Expert ablation, evaluation suite, routing analysis
   - Avoids: Pitfall 7 (evaluation confound) -- multiple metrics, not just perplexity

**Phase ordering rationale:**
- Foundation must come first because all other components depend on SafeMoEConfig and SafeMoELayer
- Masking before data pipeline because masking has the highest risk of subtle bugs (Pitfalls 1, 3, 6) and should be unit-tested in isolation
- Data pipeline before training loop because the loop consumes domain-labeled data
- Evaluation last because it requires the full training pipeline to produce checkpoints

**Research flags for phases:**
- Phase 2 (Masking): Likely needs deeper research on torch.compile + gradient hook interaction under FSDP if scaling to multi-GPU
- Phase 3 (Data Pipeline): Standard patterns, unlikely to need additional research
- Phase 4 (Training Loop): May need research on gradient accumulation semantics across domain types
- Phase 5 (Evaluation): Standard patterns, but Pitfall 7 (evaluation confound) requires careful metric design

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All APIs verified in live PyTorch 2.10 environment; no web search needed |
| Features | HIGH | Directly derived from PROJECT.md requirements + codebase capabilities |
| Architecture | HIGH | All patterns grounded in existing LitGPT code (LLaMAMoE, LoRA, DataModule) |
| Pitfalls | MEDIUM-HIGH | Core PyTorch behaviors are well-understood; MoE-specific pitfalls from literature (training data, not live-verified) |

## Gaps to Address

- **Load balancing loss design for SGTM:** Standard MoE load-balancing conflicts with intentional harmful-expert concentration. Need to design a SGTM-aware balance loss. (Phase-specific research for Milestone 1)
- **CPT expert injection mechanics:** How to cleanly add harmful experts to a pretrained checkpoint with router extension. (Phase-specific research for Milestone 2)
- **Scale sensitivity:** MoE behavior at TinyStories scale may not predict behavior at real model scale. Two-scale testing recommended in Milestone 1. (Acknowledged risk, not resolvable in research)
- **Spanish TinyStories data source:** Need to confirm availability of Spanish TinyStories or plan to create a synthetic bilingual proxy. (Data preparation task)
- **torch.compile + FSDP + gradient hooks interaction:** Verified for single-GPU eager backend; may behave differently under FSDP. (Relevant only for multi-GPU Milestone 2+)

---

*Research summary: 2026-03-13*
