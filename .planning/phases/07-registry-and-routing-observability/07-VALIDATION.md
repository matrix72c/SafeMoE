---
phase: 7
slug: registry-and-routing-observability
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-03-19
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `pytest 9.0.2` |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `pytest tests/safemoe/test_registry.py -k "inventory or qkv or summary" -x` |
| **Full suite command** | `pytest tests/safemoe/test_registry.py tests/safemoe/test_evaluate.py tests/safemoe/test_pretrain.py -x` |
| **Estimated runtime** | ~25 seconds |

---

## Sampling Rate

- **After every task commit:** Run the narrowest task-specific Phase 7 test first.
- **After every plan wave:** Run `pytest tests/safemoe/test_registry.py tests/safemoe/test_evaluate.py tests/safemoe/test_pretrain.py -x`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 25 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 07-01-01 | 01 | 1 | ROUT-01 | unit | `pytest tests/safemoe/test_registry.py -k "inventory or qkv" -x` | ❌ W0 | ⬜ pending |
| 07-01-02 | 01 | 1 | ROUT-01 | unit | `pytest tests/safemoe/test_registry.py -k "summary or artifact" -x` | ❌ W0 | ⬜ pending |
| 07-02-01 | 02 | 2 | ROUT-02 | integration | `pytest tests/safemoe/test_evaluate.py -k "routing" -x` | ✅ | ⬜ pending |
| 07-02-02 | 02 | 2 | ROUT-03 | integration | `pytest tests/safemoe/test_pretrain.py -k "routing parity" -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/safemoe/test_registry.py` — coverage for registry inventory JSON, summary Markdown, and first-class `qkv` slice records
- [ ] `tests/safemoe/test_pretrain.py` — routing parity failure-path coverage
- [ ] `safemoe/masking.py` — artifact-grade registry inventory and summary helpers
- [ ] `safemoe/observability.py` — shared routing collector, artifact writer, and parity helper

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Researcher can inspect registry and routing artifacts from a real Phase 6-derived Qwen checkpoint | ROUT-01, ROUT-02, ROUT-03 | Unit tests should stay on tiny CPU configs; the pinned checkpoint is too large for routine test execution | Run one observability pass against a real post-surgery checkpoint, confirm `registry_inventory.json`, `registry_summary.md`, `routing_observability.json`, and `routing_observability.md` are written next to the checkpoint or step directory, then execute one parity-checking flow and confirm a PASS or FAIL artifact is written without placeholder split keys |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 25s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
