---
phase: 05
slug: environment-runtime-gate
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-19
---

# Phase 05 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `pytest` |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `pytest tests/safemoe/test_pretrain.py -x` |
| **Full suite command** | `pytest tests/safemoe -x` |
| **Estimated runtime** | ~120 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/safemoe/test_pretrain.py -x`
- **After every plan wave:** Run `pytest tests/safemoe -x`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 05-01-01 | 01 | 0 | ENV-01 | integration | `pytest tests/safemoe/test_phase5_runtime_gate.py::test_qwen_checkpoint_load_path -x` | ❌ W0 | ⬜ pending |
| 05-02-01 | 02 | 1 | ENV-02 | manual-only hardware smoke | `python -m safemoe pretrain --config safemoe/configs/<phase5-config>.yaml --precision bf16-true --initial_checkpoint_dir checkpoints/Qwen3-30B-A3B-Base --tokenizer_dir checkpoints/Qwen3-30B-A3B-Base --train.max_steps 1 --eval.initial_validation false --eval.final_validation false Qwen3-30B-A3B-Base` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/safemoe/test_phase5_runtime_gate.py` — stubs for ENV-01 direct-checkpoint file validation and load-path smoke
- [ ] `safemoe/configs/<phase5-config>.yaml` — blessed-topology one-step runtime-gate config
- [ ] `.planning/phases/05-environment-runtime-gate/05-runtime-envelope.md` — committed artifact path for ENV-02 measurements
- [ ] Real GPU smoke harness for ENV-02 — existing automated coverage does not certify the full checkpoint on hardware

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Real `bf16-true` dry-start completes startup plus one optimizer step on the blessed topology and records runtime envelope metrics | ENV-02 | Requires the actual large checkpoint, CUDA BF16-capable hardware, and a real one-step run outside unit-test constraints | Run `python -m safemoe pretrain --config safemoe/configs/<phase5-config>.yaml --precision bf16-true --initial_checkpoint_dir checkpoints/Qwen3-30B-A3B-Base --tokenizer_dir checkpoints/Qwen3-30B-A3B-Base --train.max_steps 1 --eval.initial_validation false --eval.final_validation false Qwen3-30B-A3B-Base`, confirm startup plus one optimizer step succeeds, then write the measured storage footprint, startup time, first-step time, tokens/sec, and peak GPU memory to `.planning/phases/05-environment-runtime-gate/05-runtime-envelope.md`. |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
