---
phase: 3
slug: sgtm-training-loop
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-16
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >= 8.1.1 |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` (--strict-markers, --color=yes) |
| **Quick run command** | `pytest tests/safemoe/test_pretrain.py -x` |
| **Full suite command** | `pytest tests/safemoe/ -x` |
| **Estimated runtime** | ~10 seconds (CPU-only small model tests) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/safemoe/test_pretrain.py -x`
- **After every plan wave:** Run `pytest tests/safemoe/ -x`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 3-01-01 | 01 | 0 | TRAIN-01 | unit | `pytest tests/safemoe/test_pretrain.py -x` | ❌ W0 | ⬜ pending |
| 3-01-02 | 01 | 1 | TRAIN-01 | unit | `pytest tests/safemoe/test_pretrain.py::test_fit_harmful_step_masks_theta_std -x` | ❌ W0 | ⬜ pending |
| 3-01-03 | 01 | 1 | TRAIN-01 | unit | `pytest tests/safemoe/test_pretrain.py::test_fit_std_step_enables_activation_masker -x` | ❌ W0 | ⬜ pending |
| 3-01-04 | 01 | 1 | TRAIN-01 | unit | `pytest tests/safemoe/test_pretrain.py::test_fit_unlabeled_step_no_masking -x` | ❌ W0 | ⬜ pending |
| 3-01-05 | 01 | 1 | TRAIN-01 | unit | `pytest tests/safemoe/test_pretrain.py::test_masker_called_once_per_step -x` | ❌ W0 | ⬜ pending |
| 3-02-01 | 02 | 1 | TRAIN-01 | unit | `pytest tests/safemoe/test_pretrain.py::test_attn_head_gradient_masking -x` | ❌ W0 | ⬜ pending |
| 3-02-02 | 02 | 1 | TRAIN-02 | unit | `pytest tests/safemoe/test_pretrain.py::test_attn_head_activation_masking -x` | ❌ W0 | ⬜ pending |
| 3-03-01 | 03 | 2 | TRAIN-03 | smoke | `python -m safemoe pretrain --help` | ❌ W0 | ⬜ pending |
| 3-03-02 | 03 | 2 | TRAIN-03 | integration | `pytest tests/safemoe/test_pretrain.py::test_pretrain_produces_checkpoint -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/safemoe/test_pretrain.py` — test stubs for TRAIN-01, TRAIN-02, TRAIN-03
- [ ] `safemoe/__main__.py` — required for `python -m safemoe pretrain` CLI entry point (TRAIN-03)
- [ ] `safemoe/pretrain.py` — main deliverable skeleton (does not exist yet)

*No new test infrastructure needed — pytest already configured in pyproject.toml; `tests/safemoe/` directory already exists.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Training loss decreases over steps for all three split types | TRAIN-03 (success criterion 3) | Requires actual training run; slow on GPU | Run `python -m safemoe pretrain --config safemoe/configs/safemoe-tinystories.yaml` for ~100 steps, verify tensorboard loss curves decrease for each split label |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
