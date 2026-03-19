from __future__ import annotations

import json
from pathlib import Path

import lightning
import pytest
import torch
from lightning.fabric.loggers import TensorBoardLogger
from litgpt.args import EvalArgs, TrainArgs
from safemoe import pretrain as pretrain_module
from safemoe.config import SafeMoEConfig
from safemoe.pretrain import (
    resolve_phase5_gate_inputs,
    validate_phase5_checkpoint,
    validate_phase5_data_root,
)
from torch.utils.data import DataLoader, Dataset


PHASE5_MODEL_NAME = "Qwen3-30B-A3B-Base"
PHASE5_REQUIRED_FILES = (
    "lit_model.pth",
    "model_config.yaml",
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    "generation_config.json",
)
PHASE5_REQUIRED_DATA_DIRS = (
    "D_std/train",
    "D_harmful/train",
    "D_unlabeled/train",
    "D_std/val",
    "D_harmful/val",
)


class _TinyTokenDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.zeros(9, dtype=torch.long)


class _Phase5DataStub:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.x = 0
        self.y = 25

    def val_dataloaders(self) -> dict[str, DataLoader]:
        loader = DataLoader(_TinyTokenDataset(), batch_size=1)
        return {"D_std": loader, "D_harmful": loader}


def _write_phase5_checkpoint(tmp_path: Path) -> Path:
    checkpoint_dir = tmp_path / "checkpoints" / PHASE5_MODEL_NAME
    checkpoint_dir.mkdir(parents=True)
    model_config = Path("checkpoints/Qwen3-30B-A3B-Base/model_config.yaml").read_text(encoding="utf-8")
    (checkpoint_dir / "model_config.yaml").write_text(model_config, encoding="utf-8")
    (checkpoint_dir / "lit_model.pth").write_bytes(b"phase5")
    (checkpoint_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (checkpoint_dir / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}),
        encoding="utf-8",
    )
    (checkpoint_dir / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint_dir / "generation_config.json").write_text("{}", encoding="utf-8")
    return checkpoint_dir


def _write_phase5_data_root(tmp_path: Path) -> Path:
    data_root = tmp_path / "data" / ".cache" / PHASE5_MODEL_NAME / "0-25"
    for relative_dir in PHASE5_REQUIRED_DATA_DIRS:
        (data_root / relative_dir).mkdir(parents=True, exist_ok=True)
    return data_root


def test_validate_phase5_checkpoint_accepts_complete_qwen_gate_checkpoint(tmp_path: Path) -> None:
    checkpoint_dir = _write_phase5_checkpoint(tmp_path)

    assert validate_phase5_checkpoint(checkpoint_dir) == checkpoint_dir


def test_validate_phase5_checkpoint_lists_missing_required_files(tmp_path: Path) -> None:
    checkpoint_dir = _write_phase5_checkpoint(tmp_path)
    (checkpoint_dir / "config.json").unlink()
    (checkpoint_dir / "generation_config.json").unlink()

    with pytest.raises(FileNotFoundError) as exc_info:
        validate_phase5_checkpoint(checkpoint_dir)

    message = str(exc_info.value)
    assert "config.json" in message
    assert "generation_config.json" in message


def test_validate_phase5_data_root_requires_the_full_phase5_cache_layout(tmp_path: Path) -> None:
    data_root = _write_phase5_data_root(tmp_path)

    assert validate_phase5_data_root(data_root) == data_root

    (data_root / "D_harmful" / "train").rmdir()
    with pytest.raises(FileNotFoundError, match="D_harmful/train"):
        validate_phase5_data_root(data_root)


def test_setup_uses_phase5_preflight_before_direct_load_raw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint_dir = _write_phase5_checkpoint(tmp_path)
    data_root = _write_phase5_data_root(tmp_path)
    data = _Phase5DataStub(data_root.parents[1])
    events: list[str] = []
    load_raw_calls: list[Path] = []

    original_resolve = resolve_phase5_gate_inputs

    def wrapped_resolve(initial_checkpoint_dir: Path | None, tokenizer_dir: Path | None, data_module: object):
        events.append("resolve")
        return original_resolve(initial_checkpoint_dir, tokenizer_dir, data_module)

    class FakeTokenizer:
        def __init__(self, checkpoint_path: Path | str) -> None:
            events.append("tokenizer")
            self.model_name = Path(checkpoint_path).name

    def fake_get_dataloaders(*args, **kwargs):
        loader = DataLoader(_TinyTokenDataset(), batch_size=1)
        return loader, loader

    fabric_cls = lightning.Fabric

    def fake_fabric(*args, **kwargs):
        return fabric_cls(
            devices=1,
            accelerator="cpu",
            strategy="auto",
            precision="32-true",
            loggers=kwargs.get("loggers"),
        )

    def fake_load_raw(self, path: Path, model: torch.nn.Module) -> None:
        events.append("load_raw")
        load_raw_calls.append(path)

    monkeypatch.setattr(pretrain_module, "resolve_phase5_gate_inputs", wrapped_resolve)
    monkeypatch.setattr(pretrain_module, "Tokenizer", FakeTokenizer)
    monkeypatch.setattr(
        pretrain_module,
        "choose_logger",
        lambda *args, **kwargs: TensorBoardLogger(tmp_path, "phase5-test"),
    )
    monkeypatch.setattr(pretrain_module, "get_dataloaders", fake_get_dataloaders)
    monkeypatch.setattr(pretrain_module, "fit", lambda **kwargs: None)
    monkeypatch.setattr(pretrain_module, "save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(pretrain_module, "L", pretrain_module.L)
    monkeypatch.setattr(pretrain_module.L, "Fabric", fake_fabric)
    monkeypatch.setattr(fabric_cls, "load_raw", fake_load_raw)

    tiny_config = SafeMoEConfig(
        name="phase5-tiny",
        block_size=8,
        padded_vocab_size=128,
        vocab_size=128,
        n_layer=2,
        n_head=2,
        n_query_groups=2,
        n_embd=32,
        head_size=16,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMoE",
        moe_intermediate_size=64,
        n_expert=4,
        n_expert_per_token=2,
        harmful_expert_indices=[0],
        num_harmful_experts=1,
        harmful_attn_heads=[0],
    )

    pretrain_module.setup(
        model_name=PHASE5_MODEL_NAME,
        model_config=tiny_config,
        out_dir=tmp_path / "out",
        precision="32-true",
        initial_checkpoint_dir=checkpoint_dir,
        data=data,
        train=TrainArgs(
            save_interval=None,
            log_interval=1,
            global_batch_size=1,
            micro_batch_size=1,
            max_tokens=8,
            max_norm=1.0,
            lr_warmup_steps=0,
        ),
        eval=EvalArgs(interval=999999, max_iters=1, initial_validation=False, final_validation=False),
        devices=1,
        num_nodes=1,
        tokenizer_dir=None,
        logger_name="tensorboard",
        seed=42,
        upsample_std=1.0,
        upsample_harmful=1.0,
        upsample_unlabeled=1.0,
    )

    assert load_raw_calls == [checkpoint_dir / "lit_model.pth"]
    assert "resolve" in events
    assert "tokenizer" in events
    assert "load_raw" in events
    assert events.index("resolve") < events.index("tokenizer") < events.index("load_raw")
