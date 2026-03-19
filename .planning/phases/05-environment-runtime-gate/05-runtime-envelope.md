# Phase 5 Runtime Envelope

Status: PENDING
Date:
Command: `python -m safemoe pretrain --config safemoe/configs/safemoe-qwen-phase5-gate.yaml --precision bf16-true --initial_checkpoint_dir checkpoints/Qwen3-30B-A3B-Base --tokenizer_dir checkpoints/Qwen3-30B-A3B-Base --out_dir out/phase5-runtime-gate Qwen3-30B-A3B-Base`
Checkpoint: `checkpoints/Qwen3-30B-A3B-Base`
Topology: `4 GPU x 1 node`, `FSDPStrategy(auto_wrap_policy={Block}, state_dict_type="full", sharding_strategy="HYBRID_SHARD")`
Precision: `bf16-true`
Seed: `42`

## Storage footprint
- `lit_model.pth`:

## Timings
- `PHASE5_GATE_STARTUP_SECONDS`:
- `PHASE5_GATE_FIRST_STEP_SECONDS`:

## Runtime
- `PHASE5_GATE_FIRST_STEP_TOKENS_PER_SEC`:

## Peak memory
- `PHASE5_GATE_PEAK_MEMORY_GB`:

## Warnings and notes
- CUDA host:
- Data cache root: `data/.cache/Qwen3-30B-A3B-Base/0-25`
- Observed warnings:

## Replay
1. Ensure `data/.cache/Qwen3-30B-A3B-Base/0-25` contains `D_std/train`, `D_harmful/train`, `D_unlabeled/train`, `D_std/val`, and `D_harmful/val`.
2. Run the command above on the blessed 4-GPU BF16 host.
3. Copy the four `PHASE5_GATE_*` lines and `du -sh checkpoints/Qwen3-30B-A3B-Base/lit_model.pth` into this file.
4. Change `Status:` to `PASS` or `FAIL` and summarize any warnings.
