# Milestones

## v1.0 PT Phase Validation (Shipped: 2026-03-17)

**Phases completed:** 4 phases, 16 plans, 32 tasks
**Audit:** `tech_debt`

**Key accomplishments:**
- Built the bilingual TinyStories preparation pipeline and split-aware data loaders for `D_std`, `D_harmful`, and `D_unlabeled`.
- Implemented SafeMoE configuration, expert designation, parameter registry, and masking primitives on top of LitGPT.
- Shipped the SGTM pretraining loop with CLI entry point, dual-optimizer behavior, and real-run loss-convergence verification.
- Shipped checkpoint ablation, per-split perplexity evaluation, routing attribution, and mid-training ablation evaluation hooks.
- Verified the v1.0 isolation thesis on a real checkpoint: `D_harmful` perplexity delta `1645.77` versus `13.87` on `D_std`, with harmful routing fraction about `2x` higher on harmful-domain data.

**Archives:**
- `.planning/milestones/v1.0-ROADMAP.md`
- `.planning/milestones/v1.0-REQUIREMENTS.md`
- `.planning/milestones/v1.0-MILESTONE-AUDIT.md`

---
