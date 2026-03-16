---
phase: 2
slug: model-architecture-masking
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-16
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >=8.1.1 |
| **Config file** | `pyproject.toml` — `[tool.pytest.ini_options]` |
| **Quick run command** | `pytest tests/safemoe/ -x -q` |
| **Full suite command** | `pytest tests/safemoe/ -v` |
| **Estimated runtime** | ~10 seconds (CPU-only, tiny model dims) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/safemoe/ -x -q`
- **After every plan wave:** Run `pytest tests/safemoe/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 2-01-01 | 01 | 0 | MOE-01 | unit | `pytest tests/safemoe/test_config.py -x` | ❌ W0 | ⬜ pending |
| 2-01-02 | 01 | 0 | MOE-03, MOE-04 | unit | `pytest tests/safemoe/test_model.py -x` | ❌ W0 | ⬜ pending |
| 2-01-03 | 01 | 0 | MOE-02, MASK-03 | unit | `pytest tests/safemoe/test_registry.py -x` | ❌ W0 | ⬜ pending |
| 2-01-04 | 01 | 0 | MASK-01, MASK-02, MASK-04 | unit | `pytest tests/safemoe/test_masking.py -x` | ❌ W0 | ⬜ pending |
| 2-02-01 | 02 | 1 | MOE-01 | unit | `pytest tests/safemoe/test_config.py -x` | ✅ W0 | ⬜ pending |
| 2-02-02 | 02 | 1 | MOE-03, MOE-04 | unit | `pytest tests/safemoe/test_model.py -x` | ✅ W0 | ⬜ pending |
| 2-03-01 | 03 | 1 | MOE-02, MASK-03 | unit | `pytest tests/safemoe/test_registry.py -x` | ✅ W0 | ⬜ pending |
| 2-04-01 | 04 | 2 | MASK-01 | unit | `pytest tests/safemoe/test_masking.py::test_gradient_masker -x` | ✅ W0 | ⬜ pending |
| 2-04-02 | 04 | 2 | MASK-02 | unit | `pytest tests/safemoe/test_masking.py::test_activation_masker -x` | ✅ W0 | ⬜ pending |
| 2-04-03 | 04 | 2 | MASK-04 | unit | `pytest tests/safemoe/test_masking.py -x` | ✅ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/safemoe/test_config.py` — stubs for MOE-01
- [ ] `tests/safemoe/test_model.py` — stubs for MOE-03, MOE-04
- [ ] `tests/safemoe/test_registry.py` — stubs for MOE-02, MASK-03
- [ ] `tests/safemoe/test_masking.py` — stubs for MASK-01, MASK-02, MASK-04

**Note:** Do NOT create `tests/safemoe/__init__.py` — Phase 1 lesson: pytest namespace package collision shadows source package. Leave absent.

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
