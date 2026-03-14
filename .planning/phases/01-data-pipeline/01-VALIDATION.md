---
phase: 1
slug: data-pipeline
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-14
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.1.1+ |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` (already configured with `--strict-markers --color=yes`) |
| **Quick run command** | `pytest tests/safemoe/data/ -x -q` |
| **Full suite command** | `pytest tests/safemoe/data/ -v` |
| **Estimated runtime** | ~30 seconds (synthetic data, no real parquets needed) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/safemoe/data/ -x -q`
- **After every plan wave:** Run `pytest tests/safemoe/data/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 0 | DATA-01 | unit | `pytest tests/safemoe/data/test_prepare.py::test_split_proportions -x` | ❌ W0 | ⬜ pending |
| 1-01-02 | 01 | 0 | DATA-01 | unit | `pytest tests/safemoe/data/test_prepare.py::test_split_proportions_x50 -x` | ❌ W0 | ⬜ pending |
| 1-01-03 | 01 | 1 | DATA-01 | integration | `pytest tests/safemoe/data/test_prepare.py::test_litdata_output_readable -x` | ❌ W0 | ⬜ pending |
| 1-02-01 | 02 | 1 | DATA-02 | unit | `pytest tests/safemoe/data/test_datamodule.py::test_next_returns_tuple -x` | ❌ W0 | ⬜ pending |
| 1-02-02 | 02 | 1 | DATA-02 | unit | `pytest tests/safemoe/data/test_datamodule.py::test_val_dataloaders_keys -x` | ❌ W0 | ⬜ pending |
| 1-02-03 | 02 | 1 | DATA-02 | unit | `pytest tests/safemoe/data/test_datamodule.py::test_batch_shape -x` | ❌ W0 | ⬜ pending |
| 1-03-01 | 02 | 1 | DATA-03 | unit | `pytest tests/safemoe/data/test_datamodule.py::test_sampling_weights -x` | ❌ W0 | ⬜ pending |
| 1-03-02 | 02 | 1 | DATA-03 | unit | `pytest tests/safemoe/data/test_datamodule.py::test_seed_reproducibility -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/safemoe/__init__.py` — package marker for test discovery
- [ ] `tests/safemoe/data/__init__.py` — package marker for test discovery
- [ ] `tests/safemoe/data/test_prepare.py` — stubs for DATA-01 (split proportions, LitData output)
- [ ] `tests/safemoe/data/test_datamodule.py` — stubs for DATA-02 and DATA-03 (MultiDataLoader interface, weighted sampling)
- [ ] `safemoe/__init__.py` — package marker
- [ ] `safemoe/data/__init__.py` — package marker
- [ ] `uv sync --extra extra` — install litdata 0.2.59 (not currently in venv)

*Test approach: Use `litdata.optimize()` with synthetic data in `tmp_path` fixtures (follow pattern from `tests/data/test_tinystories.py`). Do NOT require real 2.1M-row parquets.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| GPT-2 tokenizer checkpoint files present | DATA-01 | Filesystem dependency; tokenizer.json must exist at checkpoint dir before any tokenization | Run `ls checkpoints/gpt2/` — verify `tokenizer.json` or `tokenizer.model` present; if not, run `litgpt download gpt2 --tokenizer_only` |
| Cache paths correctly keyed by x value | DATA-01 | Integration with real parquet files | Run `python -m safemoe.data.prepare --x 0` and `--x 25` and verify distinct dirs under `data/.cache/{tokenizer}/x0/` and `data/.cache/{tokenizer}/x25/` |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
