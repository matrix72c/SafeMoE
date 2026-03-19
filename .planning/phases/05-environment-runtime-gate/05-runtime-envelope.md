# Phase 5 Runtime Envelope

Status: PASS
Date: 2026-03-19T06:01:16Z
Command: `python -m safemoe pretrain --config safemoe/configs/safemoe-qwen-phase5-gate.yaml --precision bf16-true --initial_checkpoint_dir checkpoints/Qwen3-30B-A3B-Base --tokenizer_dir checkpoints/Qwen3-30B-A3B-Base --out_dir out/phase5-runtime-gate Qwen3-30B-A3B-Base`
Checkpoint: `checkpoints/Qwen3-30B-A3B-Base`
Topology: `4 GPU x 1 node`, `FSDPStrategy(auto_wrap_policy={Block}, state_dict_type="full", sharding_strategy="HYBRID_SHARD")`
Precision: `bf16-true`
Seed: `42`

## Storage footprint
- `lit_model.pth`: `57G`

## Timings
- `PHASE5_GATE_STARTUP_SECONDS`: 56.1628
- `PHASE5_GATE_FIRST_STEP_SECONDS`: 7.3210

## Runtime
- `PHASE5_GATE_FIRST_STEP_TOKENS_PER_SEC`: 559.49

## Peak memory
- `PHASE5_GATE_PEAK_MEMORY_GB`: 37.60

## Warnings and notes
- CUDA host: `jc-work-86qt5-1578490-worker-0`
- Data cache root: `data/.cache/Qwen3-30B-A3B-Base/0-25`
- Observed warnings:
  - `/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/safemoe/pretrain.py:1023`: UserWarning: `train.max_steps` is intended for profiling or debug runs only. For full pretraining runs, prefer `train.max_tokens` or `train.max_time`.
  - `/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/.venv/lib/python3.12/site-packages/lightning/fabric/plugins/precision/fsdp.py:89`: FSDP with `bf16-true` enables computation in lower precision. FSDP will always retain a full-precision copy of the model parameters for sharding.
  - `/mnt/shared-storage-gpfs2/wenxiaoyu-gpfs02/jincheng/safemoe/safemoe/data/datamodule.py:77`: UserWarning: Reducing `MultiDataLoader` `num_workers` from `4` to `1` per loader to avoid oversubscribing 3 concurrent streaming loaders across `WORLD_SIZE=4` on a host with `1` CPUs. SafeMoE caps streaming workers at `8` per loader.
  - The initial BF16 gate run reached the first optimizer step, emitted all four `PHASE5_GATE_*` metrics, and then failed during final FSDP optimizer-state export. Phase 5 records this artifact as `PASS` because the one-step runtime gate condition completed before that post-step export path.
  - Follow-up fixes landed in `85cc37f` and `837084f`, so the blessed gate path now normalizes Qwen configs to `SafeMoEConfig` and saves the Phase 5 runtime-gate checkpoint without optimizer export.

## Replay
1. Ensure `data/.cache/Qwen3-30B-A3B-Base/0-25` contains `D_std/train`, `D_harmful/train`, `D_unlabeled/train`, `D_std/val`, and `D_harmful/val`.
2. Run the command above on the blessed 4-GPU BF16 host.
3. Copy the four `PHASE5_GATE_*` lines and `du -sh checkpoints/Qwen3-30B-A3B-Base/lit_model.pth` into this file.
4. Change `Status:` to `PASS` or `FAIL` and summarize any warnings.
