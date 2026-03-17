---
status: awaiting_human_verify
trigger: "python -m safemoe evaluate \"$CKPT_DIR\" --ablated \"$CKPT_DIR/ablated\" crashes with FileNotFoundError: model_config.yaml even though the file exists under the checkpoint directory."
created: 2026-03-17T00:00:00+08:00
updated: 2026-03-17T13:45:25+08:00
---

## Current Focus

hypothesis: Confirmed and fixed: the CLI now turns the reported shell-expansion trap into a targeted checkpoint-path error, and the literal checkpoint path remains valid.
test: Ask the user to retry evaluation with an exported or literal checkpoint path in their real workflow.
expecting: The corrected command will load `checkpoints/safemoe-tinystories-v2/final` instead of resolving `model_config.yaml` from the repo root.
next_action: Wait for user confirmation that the corrected invocation works end-to-end in their environment.

## Symptoms

expected: The evaluate CLI should load the original checkpoint from `checkpoints/safemoe-tinystories-v2/final` and continue evaluation.
actual: `_load_model` attempts to read `model_config.yaml` from the current directory and crashes.
errors: `FileNotFoundError: [Errno 2] No such file or directory: 'model_config.yaml'`
reproduction: From repo root, run `CKPT_DIR=checkpoints/safemoe-tinystories-v2/final python -m safemoe evaluate "$CKPT_DIR" --ablated "$CKPT_DIR/ablated"`
started: Unknown; investigate current code state.

## Eliminated

## Evidence

- timestamp: 2026-03-17T13:40:00+08:00
  checked: `safemoe/__main__.py` and `safemoe/evaluate.py`
  found: `__main__` documents `evaluate --original <ckpt_dir>`, but `setup(original: Path, ablated: Optional[Path], routing: bool)` exposes `original` as a positional argument to `jsonargparse`.
  implication: The CLI contract is inconsistent, and the positional checkpoint argument is a prime suspect for being parsed or forwarded incorrectly.

- timestamp: 2026-03-17T13:41:48+08:00
  checked: Exact reproduction command from repo root
  found: The command fails in `_load_model()` with `FileNotFoundError` for `model_config.yaml`, and the missing path in the traceback is the cwd-relative `'model_config.yaml'` rather than a file under the checkpoint directory.
  implication: `original_ckpt_dir` reached `_load_model()` as `Path('.')` or equivalent, so the bug happens before model loading.

- timestamp: 2026-03-17T13:41:48+08:00
  checked: Inline shell assignment behavior with `env -u CKPT_DIR zsh -lc 'CKPT_DIR=... python -c ... \"$CKPT_DIR\"'`
  found: `sys.argv[1]` is `''` while `os.environ['CKPT_DIR']` is `checkpoints/safemoe-tinystories-v2/final`.
  implication: The shell expands `"$CKPT_DIR"` before the temporary assignment is applied, which exactly explains why `jsonargparse` converts the empty positional argument into `Path('.')`.

- timestamp: 2026-03-17T13:45:25+08:00
  checked: `pytest tests/safemoe/test_evaluate.py -q`
  found: All four safemoe evaluate tests pass, including the new CLI regression that patches `sys.argv` to the post-expansion values of the reported command.
  implication: The checkpoint-validation path and CLI-level regression coverage are both working in the current code state.

- timestamp: 2026-03-17T13:45:25+08:00
  checked: Exact reported command from repo root after the fix
  found: The command now fails with an explicit explanation that `checkpoint_dir` resolved to `.` because the shell expands `$CKPT_DIR` before the temporary assignment applies.
  implication: The opaque `FileNotFoundError: model_config.yaml` failure mode has been replaced with an actionable diagnosis.

- timestamp: 2026-03-17T13:45:25+08:00
  checked: `_ensure_checkpoint_dir("checkpoints/safemoe-tinystories-v2/final")`
  found: The literal checkpoint directory validates successfully.
  implication: The checkpoint contents are fine; the issue is isolated to the CLI invocation pattern.

## Resolution

root_cause:
root_cause: The inline shell form `CKPT_DIR=... python -m safemoe evaluate "$CKPT_DIR" --ablated "$CKPT_DIR/ablated"` expands `"$CKPT_DIR"` before the temporary assignment applies. Python receives an empty positional original argument, `jsonargparse` converts it to `Path('.')`, and `_load_model()` looks for `model_config.yaml` in the current working directory.
fix:
fix: Reused the existing local checkpoint-validation change in `safemoe/evaluate.py`, which routes cwd-resolved checkpoint paths through `check_valid_checkpoint_dir()` and raises a targeted explanation for the inline shell-assignment trap, and added a CLI-level regression test in `tests/safemoe/test_evaluate.py` for the exact post-expansion argv pattern.
verification: `pytest tests/safemoe/test_evaluate.py -q` passes (4 tests). Re-running the exact reported command now raises the targeted shell-assignment explanation instead of `FileNotFoundError: model_config.yaml`. Passing the literal checkpoint directory validates successfully via `_ensure_checkpoint_dir("checkpoints/safemoe-tinystories-v2/final")`.
files_changed:
- safemoe/evaluate.py
- safemoe/__main__.py
- tests/safemoe/test_evaluate.py
