---
phase: 6
slug: checkpoint-surgery
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-03-19
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `pytest 9.0.2` |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `pytest tests/safemoe/test_checkpoint_surgery.py::test_manifest_planner_is_deterministic -x` |
| **Full suite command** | `pytest tests/safemoe -x` |
| **Estimated runtime** | ~20 seconds |

---

## Sampling Rate

- **After every task commit:** Run the narrowest task-specific checkpoint-surgery test, starting with `pytest tests/safemoe/test_checkpoint_surgery.py::test_manifest_planner_is_deterministic -x`
- **After every plan wave:** Run `pytest tests/safemoe/test_checkpoint_surgery.py tests/safemoe/test_registry.py tests/safemoe/test_ablate.py -x`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 20 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 06-01-01 | 01 | 0 | INIT-01 | unit | `pytest tests/safemoe/test_checkpoint_surgery.py::test_manifest_planner_is_deterministic -x` | ❌ W0 | ⬜ pending |
| 06-02-01 | 02 | 2 | INIT-02 | integration | `pytest tests/safemoe/test_checkpoint_surgery.py::test_surgery_writes_loadable_checkpoint_directory -x` | ❌ W0 | ⬜ pending |
| 06-02-02 | 02 | 2 | INIT-03 | integration | `pytest tests/safemoe/test_checkpoint_surgery.py::test_verifier_fails_on_manifest_mismatch -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/safemoe/test_checkpoint_surgery.py` — stubs and coverage for manifest planning, state-dict mutation, reload parity, and failure cases
- [ ] `safemoe/interventions/manifest.py` — schema and validation helpers used by planner, applier, and verifier
- [ ] `safemoe/interventions/planner.py` — deterministic source/target planner for explicit harmful expert/head layouts
- [ ] `safemoe/interventions/surgery.py` — expert, router-column, and packed-QKV mutation logic
- [ ] `safemoe/interventions/verify.py` — manifest-aware verification reports and hard-fail parity checks
- [ ] Temp-output finalize helper — prevents invalid surgery runs from leaving blessed artifacts under `checkpoints/`

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Real surgery output reloads through the normal downstream path | INIT-02, INIT-03 | Unit tests should use small synthetic checkpoints; the real Qwen artifact is too large for routine CI | Run one manifest-backed surgery against `checkpoints/Qwen3-30B-A3B-Base`, confirm the output directory contains `lit_model.pth`, `model_config.yaml`, `intervention_manifest.json`, `verification_report.json`, and `verification_summary.md`, then load it through the existing checkpoint path used by `safemoe/pretrain.py` or `safemoe/evaluate.py` without schema or missing-file errors |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 20s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
