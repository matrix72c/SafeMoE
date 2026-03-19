from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from litgpt.args import EvalArgs, TrainArgs
from safemoe.pretrain import active_split_labels, fit, setup, warmup_routing_loss
from tests.safemoe.test_pretrain import _setup_fit_test


def test_warmup_stage_logs_separate_losses_and_excludes_unlabeled(tmp_path: Path) -> None:
    fabric, state, data, val_dataloader, registry, gradient_masker, activation_masker, _, _ = _setup_fit_test(
        global_batch_size=1,
        micro_batch_size=1,
        max_tokens=128,
        block_size=128,
    )

    logged_metrics: list[dict[str, float]] = []

    original_log_dict = fabric.log_dict

    def capture_log_dict(metrics: dict, step: int) -> None:
        logged_metrics.append(dict(metrics))
        original_log_dict(metrics, step=step)

    fabric.log_dict = capture_log_dict  # type: ignore[method-assign]

    train_args = TrainArgs(
        save_interval=None,
        log_interval=1,
        global_batch_size=1,
        micro_batch_size=1,
        max_tokens=128,
        max_norm=1.0,
        lr_warmup_steps=0,
    )
    eval_args = EvalArgs(
        interval=99999,
        max_iters=1,
        initial_validation=False,
        final_validation=False,
    )

    assert active_split_labels("warmup") == ["D_std", "D_harmful"]

    with patch("safemoe.pretrain.random.choices", return_value=["D_std"]):
        fit(
            fabric=fabric,
            devices=1,
            state=state,
            data=data,
            val_dataloader=val_dataloader,
            out_dir=tmp_path,
            tokenizer_dir=None,
            train=train_args,
            eval=eval_args,
            gradient_masker=gradient_masker,
            activation_masker=activation_masker,
            registry=registry,
            upsample_std=1.0,
            upsample_harmful=1.0,
            upsample_unlabeled=0.0,
            stage="warmup",
            warmup_routing_loss_weight=0.1,
            warmup_harmful_mass_floor=0.6,
            warmup_std_mass_ceiling=0.4,
        )

    all_keys = {key for metrics in logged_metrics for key in metrics}
    assert "loss_lm_D_std" in all_keys
    assert "loss_routing_D_std" in all_keys
    assert "loss_total_D_std" in all_keys
    assert "routing_harmful_mass_D_std" in all_keys

    assert "loss_lm_D_harmful" not in all_keys
    assert "loss_routing_D_harmful" not in all_keys
    assert "loss_total_D_harmful" not in all_keys
    assert "routing_harmful_mass_D_harmful" not in all_keys

    assert "loss_lm_D_unlabeled" not in all_keys
    assert "loss_routing_D_unlabeled" not in all_keys
    assert "loss_total_D_unlabeled" not in all_keys
    assert "routing_harmful_mass_D_unlabeled" not in all_keys


def test_warmup_routing_supervision_penalizes_wrong_direction() -> None:
    harmful_mass = torch.tensor(0.5)

    harmful_penalty = warmup_routing_loss(
        harmful_mass,
        "D_harmful",
        harmful_mass_floor=0.6,
        std_mass_ceiling=0.4,
    )
    std_penalty = warmup_routing_loss(
        harmful_mass,
        "D_std",
        harmful_mass_floor=0.6,
        std_mass_ceiling=0.4,
    )
    tolerated_harmful = warmup_routing_loss(
        torch.tensor(0.7),
        "D_harmful",
        harmful_mass_floor=0.6,
        std_mass_ceiling=0.4,
    )
    tolerated_std = warmup_routing_loss(
        torch.tensor(0.3),
        "D_std",
        harmful_mass_floor=0.6,
        std_mass_ceiling=0.4,
    )

    assert harmful_penalty.item() > 0.0
    assert std_penalty.item() > 0.0
    assert harmful_penalty.item() > tolerated_harmful.item()
    assert std_penalty.item() > tolerated_std.item()


def test_warmup_rejects_nonzero_unlabeled_weight(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="upsample_unlabeled must be 0.0 for warmup"):
        setup(
            model_name="Qwen3-30B-A3B-Base",
            model_config=None,
            out_dir=tmp_path,
            precision="32-true",
            initial_checkpoint_dir=None,
            resume=False,
            data=object(),
            train=TrainArgs(
                save_interval=None,
                log_interval=1,
                global_batch_size=1,
                micro_batch_size=1,
                max_tokens=128,
                max_norm=1.0,
                lr_warmup_steps=0,
            ),
            eval=EvalArgs(
                interval=99999,
                max_iters=1,
                initial_validation=False,
                final_validation=False,
            ),
            devices=1,
            num_nodes=1,
            tokenizer_dir=None,
            logger_name="tensorboard",
            seed=42,
            upsample_std=1.0,
            upsample_harmful=1.0,
            upsample_unlabeled=0.1,
            stage="warmup",
        )
