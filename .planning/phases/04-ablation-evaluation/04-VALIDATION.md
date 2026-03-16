---
phase: 4
slug: ablation-evaluation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-16
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (pyproject.toml `[tool.pytest.ini_options]`) |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `pytest tests/safemoe/ -x -q` |
| **Full suite command** | `pytest tests/safemoe/ -v` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/safemoe/ -x -q`
- **After every plan wave:** Run `pytest tests/safemoe/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 4-01-01 | 01 | 0 | TRAIN-04 | unit stub | `pytest tests/safemoe/test_ablate.py -x -q` | ❌ W0 | ⬜ pending |
| 4-01-02 | 01 | 0 | EVAL-01, EVAL-02 | unit stub | `pytest tests/safemoe/test_evaluate.py -x -q` | ❌ W0 | ⬜ pending |
| 4-01-03 | 01 | 0 | EVAL-03 | unit stub | `pytest tests/safemoe/test_pretrain.py::test_evaluate_with_ablation -x -q` | ❌ W0 | ⬜ pending |
| 4-02-01 | 02 | 1 | TRAIN-04 | unit | `pytest tests/safemoe/test_ablate.py -x -q` | ✅ W0 | ⬜ pending |
| 4-03-01 | 03 | 1 | EVAL-01 | integration | `pytest tests/safemoe/test_evaluate.py::test_evaluate_perplexity -x -q` | ✅ W0 | ⬜ pending |
| 4-03-02 | 03 | 1 | EVAL-02 | unit | `pytest tests/safemoe/test_evaluate.py::test_routing_attribution -x -q` | ✅ W0 | ⬜ pending |
| 4-04-01 | 04 | 2 | EVAL-03 | unit | `pytest tests/safemoe/test_pretrain.py::test_evaluate_with_ablation -x -q` | ✅ W0 | ⬜ pending |
| 4-05-01 | 05 | 3 | TRAIN-04, EVAL-01, EVAL-02, EVAL-03 | integration | `pytest tests/safemoe/ -v` | ✅ prior | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/safemoe/test_ablate.py` — stubs for TRAIN-04 (ablate() zeroes weights, saves manifest)
- [ ] `tests/safemoe/test_evaluate.py` — stubs for EVAL-01 (perplexity), EVAL-02 (routing attribution)
- [ ] `tests/safemoe/test_pretrain.py::test_evaluate_with_ablation` — new test in existing file for EVAL-03

*Existing infrastructure (pytest, conftest.py, _MockMultiDataLoader, _SynthDataset) covers all fixtures — no new framework install needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| D_harmful perplexity increases significantly after ablation on a real checkpoint | EVAL-01, TRAIN-04 | Requires a real trained checkpoint; synthetic tests only verify numerical correctness | Run `python -m safemoe evaluate --original <ckpt> --ablated <ckpt>/ablated`; verify D_harmful delta is large relative to D_std delta |
| Routing histograms show D_harmful tokens preferentially activate theta_harmful experts | EVAL-02 | Requires a meaningfully trained model; random-weight models show no routing signal | Run `python -m safemoe evaluate --original <ckpt> --routing`; inspect TensorBoard routing/ histograms |
| Mid-training ablation delta grows over training | EVAL-03 | Requires multi-checkpoint training run; not testable with unit mocks | Compare `ablated_val_ppl_D_harmful` delta across checkpoints in TensorBoard |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
