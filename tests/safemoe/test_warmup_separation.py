from __future__ import annotations

import json
from pathlib import Path
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from litgpt.args import EvalArgs, TrainArgs
from safemoe.pretrain import active_split_labels, fit, main, setup, warmup_routing_loss
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


def test_warmup_acceptance_report_passes_on_same_lineage_improvement(tmp_path: Path) -> None:
    from safemoe.evaluate import evaluate_warmup_acceptance

    pre_ckpt_dir = tmp_path / "pre"
    post_ckpt_dir = tmp_path / "post"
    pre_ckpt_dir.mkdir()
    post_ckpt_dir.mkdir()

    perplexity_results = {
        pre_ckpt_dir: {"original": {"val_ppl_D_std": 10.0, "val_ppl_D_harmful": 20.0}},
        post_ckpt_dir: {"original": {"val_ppl_D_std": 10.5, "val_ppl_D_harmful": 18.0}},
    }
    routing_results = {
        pre_ckpt_dir: {"routing_harmful_frac_D_std": 0.20, "routing_harmful_frac_D_harmful": 0.25},
        post_ckpt_dir: {"routing_harmful_frac_D_std": 0.10, "routing_harmful_frac_D_harmful": 0.30},
    }

    def fake_evaluate_perplexity(original_ckpt_dir: Path, **_: object) -> dict:
        return perplexity_results[Path(original_ckpt_dir)]

    def fake_routing_attribution(ckpt_dir: Path, **_: object) -> dict:
        return routing_results[Path(ckpt_dir)]

    with patch("safemoe.evaluate.evaluate_perplexity", side_effect=fake_evaluate_perplexity), patch(
        "safemoe.evaluate.routing_attribution",
        side_effect=fake_routing_attribution,
    ):
        report = evaluate_warmup_acceptance(
            pre_ckpt_dir,
            post_ckpt_dir,
            out_dir=tmp_path,
            required_post_routing_margin=0.10,
            required_margin_gain=0.00,
            std_ppl_regression_tolerance=0.05,
        )

    assert report["pass"] is True
    assert report["routing_margin_pre"] == pytest.approx(0.05)
    assert report["routing_margin_post"] == pytest.approx(0.20)
    assert report["routing_margin_gain"] == pytest.approx(0.15)
    assert report["std_ppl_ratio"] == pytest.approx(1.05)
    assert report["blessed_checkpoint_dir"] == str(tmp_path / "warmup-blessed")

    json_path = tmp_path / "warmup_acceptance.json"
    markdown_path = tmp_path / "warmup_acceptance.md"
    assert json_path.exists()
    assert markdown_path.exists()

    payload = json.loads(json_path.read_text())
    assert payload["pre"]["checkpoint_dir"] == str(pre_ckpt_dir)
    assert payload["post"]["checkpoint_dir"] == str(post_ckpt_dir)
    assert set(payload) == {
        "pre",
        "post",
        "delta",
        "criteria",
        "routing_margin_pre",
        "routing_margin_post",
        "routing_margin_gain",
        "std_ppl_ratio",
        "pass",
        "blessed_checkpoint_dir",
    }
    assert payload["criteria"] == {
        "required_post_routing_margin": 0.10,
        "required_margin_gain": 0.00,
        "std_ppl_regression_tolerance": 0.05,
    }
    assert payload["delta"]["val_ppl_D_std"] == pytest.approx(0.5)
    assert payload["delta"]["routing_harmful_frac_D_std"] == pytest.approx(-0.10)
    assert payload["delta"]["routing_harmful_frac_D_harmful"] == pytest.approx(0.05)

    markdown = markdown_path.read_text()
    assert markdown.startswith("# Warmup Acceptance Report")
    assert "Result: PASS" in markdown
    assert f"Pre-checkpoint: {pre_ckpt_dir}" in markdown
    assert f"Post-checkpoint: {post_ckpt_dir}" in markdown


def test_warmup_acceptance_report_fails_on_std_regression(tmp_path: Path) -> None:
    from safemoe.evaluate import evaluate_warmup_acceptance

    pre_ckpt_dir = tmp_path / "pre"
    post_ckpt_dir = tmp_path / "post"
    pre_ckpt_dir.mkdir()
    post_ckpt_dir.mkdir()

    perplexity_results = {
        pre_ckpt_dir: {"original": {"val_ppl_D_std": 10.0, "val_ppl_D_harmful": 20.0}},
        post_ckpt_dir: {"original": {"val_ppl_D_std": 10.6, "val_ppl_D_harmful": 18.0}},
    }
    routing_results = {
        pre_ckpt_dir: {"routing_harmful_frac_D_std": 0.20, "routing_harmful_frac_D_harmful": 0.25},
        post_ckpt_dir: {"routing_harmful_frac_D_std": 0.10, "routing_harmful_frac_D_harmful": 0.30},
    }

    with patch("safemoe.evaluate.evaluate_perplexity", side_effect=lambda original_ckpt_dir, **_: perplexity_results[Path(original_ckpt_dir)]), patch(
        "safemoe.evaluate.routing_attribution",
        side_effect=lambda ckpt_dir, **_: routing_results[Path(ckpt_dir)],
    ):
        report = evaluate_warmup_acceptance(
            pre_ckpt_dir,
            post_ckpt_dir,
            out_dir=tmp_path,
            required_post_routing_margin=0.10,
            required_margin_gain=0.00,
            std_ppl_regression_tolerance=0.05,
        )

    assert report["pass"] is False
    assert report["std_ppl_ratio"] == pytest.approx(1.06)
    assert report["blessed_checkpoint_dir"] is None

    payload = json.loads((tmp_path / "warmup_acceptance.json").read_text())
    assert payload["pass"] is False
    assert payload["blessed_checkpoint_dir"] is None

    markdown = (tmp_path / "warmup_acceptance.md").read_text()
    assert markdown.startswith("# Warmup Acceptance Report")
    assert "Result: FAIL" in markdown


def test_warmup_blessed_checkpoint_requires_pass(tmp_path: Path) -> None:
    fabric, state, data, _, _, _, _, registry, _ = _setup_fit_test(
        global_batch_size=1,
        micro_batch_size=1,
        max_tokens=128,
        block_size=128,
    )
    checkpoint_path = tmp_path / "initial" / "lit_model.pth"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_text("seed")

    saved_checkpoints: list[Path] = []

    def fake_save_checkpoint(fabric, state, tokenizer_dir, checkpoint_file, include_optimizer: bool = True):
        saved_checkpoints.append(Path(checkpoint_file))
        checkpoint_dir = Path(checkpoint_file).parent
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "lit_model.pth").write_text("weights")
        (checkpoint_dir / "model_config.yaml").write_text("name: test\n")
        (checkpoint_dir / "hyperparameters.yaml").write_text("tokenizer_dir: null\n")

    with patch("safemoe.pretrain.fit"), patch(
        "safemoe.pretrain.save_checkpoint",
        side_effect=fake_save_checkpoint,
    ), patch(
        "safemoe.pretrain.evaluate_warmup_acceptance",
        return_value={"pass": False, "blessed_checkpoint_dir": None},
    ):
        with pytest.raises(ValueError, match="Warmup acceptance failed"):
            main(
                fabric=fabric,
                devices=1,
                seed=42,
                initial_checkpoint_dir=checkpoint_path.parent,
                resume=False,
                config=state["model"].config,
                data=data,
                out_dir=tmp_path,
                tokenizer_dir=None,
                tokenizer=None,
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
                optimizer="AdamW",
                upsample_std=1.0,
                upsample_harmful=1.0,
                upsample_unlabeled=0.0,
                stage="warmup",
            )

    assert tmp_path / "final" / "lit_model.pth" in saved_checkpoints
    assert not (tmp_path / "warmup-blessed").exists()

    with patch("safemoe.pretrain.fit"), patch(
        "safemoe.pretrain.save_checkpoint",
        side_effect=fake_save_checkpoint,
    ), patch(
        "safemoe.pretrain.evaluate_warmup_acceptance",
        return_value={"pass": True, "blessed_checkpoint_dir": str(tmp_path / "warmup-blessed")},
    ):
        main(
            fabric=fabric,
            devices=1,
            seed=42,
            initial_checkpoint_dir=checkpoint_path.parent,
            resume=False,
            config=state["model"].config,
            data=data,
            out_dir=tmp_path,
            tokenizer_dir=None,
            tokenizer=None,
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
            optimizer="AdamW",
            upsample_std=1.0,
            upsample_harmful=1.0,
            upsample_unlabeled=0.0,
            stage="warmup",
        )

    assert (tmp_path / "warmup-blessed" / "lit_model.pth").exists()
