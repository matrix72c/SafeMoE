import math
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from lightning import Fabric

import litgpt.safemoe.pretrain as safemoe_pretrain

from litgpt.config import Config
from litgpt.data.safedata import SafeData
from litgpt.model import LLaMAMoE, SafeMoELayer
from litgpt.safemoe.masking import (
    ActivationMasker,
    GradientMasker,
    HarmfulParamRegistry,
    temporarily_ablate_harmful_params,
)
from litgpt.safemoe.pretrain import (
    SplitValidationResult,
    ValidationSummary,
    _choose_split_label_once,
    _collect_cached_routing_counts,
    _evaluate_validation_split,
    _setup_split_dataloaders,
    collect_validation_summary,
    collect_warmup_routing_mass,
    evaluate_with_ablation,
    warmup_routing_loss,
)


def _make_safemoe_config(**overrides) -> Config:
    config = Config(
        name="test-safemoe",
        block_size=8,
        vocab_size=32,
        padded_vocab_size=32,
        n_layer=1,
        n_head=4,
        n_query_groups=4,
        n_embd=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        mlp_class_name="LLaMAMoE",
        n_expert=4,
        n_expert_per_token=2,
        n_shared_expert=1,
        bias=False,
        harmful_expert_indices=[1, 3],
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _make_registry_fixture(config: Config | None = None):
    config = config or _make_safemoe_config()
    layer = torch.nn.Module()
    layer.mlp = SafeMoELayer(config)
    layer.attn = torch.nn.Module()
    qkv_rows = (config.n_head + 2 * config.n_query_groups) * config.head_size
    layer.attn.qkv = torch.nn.Linear(config.n_embd, qkv_rows, bias=False)
    model = torch.nn.Module()
    model.transformer = torch.nn.Module()
    model.transformer.h = torch.nn.ModuleList([layer])
    model.shared = torch.nn.Linear(config.n_embd, config.n_embd, bias=False)
    registry = HarmfulParamRegistry(model, config)
    return SimpleNamespace(config=config, model=model, layer=layer, registry=registry)


def test_harmful_param_registry_classifies_experts():
    fixture = _make_registry_fixture()
    layer = fixture.layer
    registry = fixture.registry

    harmful_params = {id(param) for param in registry.parameters_by_type("theta_harmful")}
    std_params = {id(param) for param in registry.parameters_by_type("theta_std")}
    shared_params = {id(param) for param in registry.parameters_by_type("theta_shared")}
    all_params = {id(param) for param in fixture.model.parameters()}

    assert id(layer.mlp.experts[1].fc_1.weight) in harmful_params
    assert id(layer.mlp.experts[3].proj.weight) in harmful_params
    assert id(layer.mlp.experts[0].fc_1.weight) in std_params
    assert id(layer.attn.qkv.weight) in shared_params
    assert id(layer.mlp.gate.weight) in shared_params
    assert harmful_params | std_params | shared_params == all_params


@pytest.mark.parametrize(
    ("split_label", "expected"),
    [("D_harmful", 0.7981389), ("D_std", 0.7981389), ("D_unlabeled", 0.0)],
)
def test_warmup_routing_loss(split_label, expected):
    harmful_mass = torch.tensor(0.5)
    loss = warmup_routing_loss(harmful_mass, split_label, harmful_mass_floor=0.7, std_mass_ceiling=0.3)
    assert pytest.approx(loss.item(), rel=1e-6) == expected


def test_collect_warmup_routing_mass():
    config = _make_safemoe_config()
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([SafeMoELayer(config), SafeMoELayer(config)])
    model.register_parameter("anchor", torch.nn.Parameter(torch.tensor(1.0)))
    model.layers[0]._last_harmful_routing_mass = torch.tensor(0.2)
    model.layers[1]._last_harmful_routing_mass = torch.tensor(0.6)

    mass = collect_warmup_routing_mass(model)

    assert torch.isclose(mass, torch.tensor(0.4))


def test_activation_and_gradient_maskers():
    fixture = _make_registry_fixture()
    activation_masker = ActivationMasker(fixture.model)
    gradient_masker = GradientMasker(fixture.registry)

    activation_masker.enable()
    assert fixture.layer.mlp._activation_masking_enabled is True
    activation_masker.disable()
    assert fixture.layer.mlp._activation_masking_enabled is False

    harmful_params = fixture.registry.parameters_by_type("theta_harmful")
    std_params = [p for p in fixture.registry.parameters_by_type("theta_std") if id(p) != id(fixture.layer.mlp.gate.weight)]
    for param in harmful_params + std_params:
        param.grad = torch.ones_like(param)

    gradient_masker.mask("D_std")
    assert all(param.grad is None for param in harmful_params)
    assert any(param.grad is not None for param in std_params)


def test_temporarily_ablate_harmful_params_restores_experts():
    fixture = _make_registry_fixture()
    harmful_param = fixture.registry.parameters_by_type("theta_harmful")[0]
    harmful_before = harmful_param.detach().clone()

    with temporarily_ablate_harmful_params(fixture.registry):
        assert torch.count_nonzero(harmful_param) == 0

    torch.testing.assert_close(harmful_param, harmful_before)


def test_collect_cached_routing_counts_is_tensor_native():
    config = _make_safemoe_config()
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([SafeMoELayer(config), SafeMoELayer(config)])
    model.layers[0]._last_indices = torch.tensor([[0, 1], [2, 3]])
    model.layers[1]._last_indices = torch.tensor([[3, 3]])

    total, harmful = _collect_cached_routing_counts(model, harmful_expert_indices=[1, 3])

    assert total == 6
    assert harmful == 4


def test_choose_split_label_once_uses_rank_zero_sample():
    fabric = mock.Mock(global_rank=0)
    fabric.broadcast.side_effect = lambda value, src=0: value

    with mock.patch("litgpt.safemoe.pretrain.random.choices", return_value=["D_harmful"]) as random_choices:
        split_label = _choose_split_label_once(fabric, ["D_std", "D_harmful"], [1.0, 3.0])

    assert split_label == "D_harmful"
    random_choices.assert_called_once_with(["D_std", "D_harmful"], weights=[1.0, 3.0], k=1)
    fabric.broadcast.assert_called_once_with(1, src=0)


def test_choose_split_label_once_uses_broadcast_value_on_nonzero_rank():
    fabric = mock.Mock(global_rank=1)
    fabric.broadcast.return_value = 0

    with mock.patch("litgpt.safemoe.pretrain.random.choices") as random_choices:
        split_label = _choose_split_label_once(fabric, ["D_std", "D_harmful"], [1.0, 3.0])

    assert split_label == "D_std"
    random_choices.assert_not_called()
    fabric.broadcast.assert_called_once_with(0, src=0)


class _EvalModel(torch.nn.Module):
    def __init__(self, outputs=None):
        super().__init__()
        self.max_seq_length = 3
        self.config = SimpleNamespace(harmful_expert_indices=[])
        self._outputs = outputs or [torch.randn(1, 3, 8), torch.randn(1, 3, 8)]
        self.forward_calls = 0

    def forward(self, input_ids):
        output = self._outputs[self.forward_calls]
        self.forward_calls += 1
        return output


def test_evaluate_validation_split_uses_safe_eval_budget():
    fabric = mock.Mock(device=torch.device("cpu"))
    fabric.barrier = mock.Mock()
    fabric.print = mock.Mock()
    model = _EvalModel()
    model.train(True)

    batches = [
        torch.tensor([[1, 2, 3, 4]]),
        torch.tensor([[2, 3, 4, 5]]),
        torch.tensor([[3, 4, 5, 6]]),
    ]

    with mock.patch("litgpt.safemoe.pretrain._collect_cached_routing_counts", return_value=(4, 1)), mock.patch(
        "litgpt.safemoe.pretrain.chunked_cross_entropy", side_effect=[torch.tensor(1.0), torch.tensor(3.0)]
    ):
        result = _evaluate_validation_split(
            fabric=fabric,
            model=model,
            val_dataloader=batches,
            split_name="D_std",
            max_iters=2,
            verbose=False,
        )

    assert result.batches_evaluated == 2
    assert result.loss.item() == pytest.approx(2.0)
    assert result.dispatch_count == 8
    assert result.harmful_dispatches == 2
    assert result.routing_harmful_frac == pytest.approx(0.25)
    assert model.forward_calls == 2
    assert model.training is True


def test_evaluate_validation_split_uses_max_iters_for_unsized_loader():
    fabric = mock.Mock(device=torch.device("cpu"))
    fabric.barrier = mock.Mock()
    fabric.print = mock.Mock()
    model = _EvalModel(outputs=[torch.randn(1, 3, 8)] * 5)

    class _UnsizedLoader:
        def __iter__(self):
            return iter(
                [
                    torch.tensor([[1, 2, 3, 4]]),
                    torch.tensor([[2, 3, 4, 5]]),
                    torch.tensor([[3, 4, 5, 6]]),
                ]
            )

        def __len__(self):
            raise TypeError("length unavailable")

    with mock.patch("litgpt.safemoe.pretrain._collect_cached_routing_counts", return_value=(2, 1)), mock.patch(
        "litgpt.safemoe.pretrain.chunked_cross_entropy", side_effect=[torch.tensor(1.0), torch.tensor(2.0)]
    ), mock.patch("torch.distributed.is_available", return_value=False), mock.patch(
        "torch.distributed.is_initialized", return_value=False
    ):
        result = _evaluate_validation_split(
            fabric=fabric,
            model=model,
            val_dataloader=_UnsizedLoader(),
            split_name="D_harmful",
            max_iters=2,
            verbose=False,
        )

    assert result.batches_evaluated == 2
    assert model.forward_calls == 2
    assert result.dispatch_count == 4
    assert result.harmful_dispatches == 2


def test_evaluate_validation_split_uses_distributed_min_budget_and_reductions():
    fabric = mock.Mock(device=torch.device("cpu"))
    fabric.barrier = mock.Mock()
    fabric.print = mock.Mock()
    model = _EvalModel(outputs=[torch.randn(1, 3, 8)] * 5)

    batches = [
        torch.tensor([[1, 2, 3, 4]]),
        torch.tensor([[2, 3, 4, 5]]),
        torch.tensor([[3, 4, 5, 6]]),
        torch.tensor([[4, 5, 6, 7]]),
    ]
    reduce_calls = []

    def _all_reduce(tensor, op=None):
        reduce_calls.append((tensor.dtype, op, int(tensor.item()) if tensor.dtype == torch.long else float(tensor.item())))
        if tensor.dtype == torch.long and op == torch.distributed.ReduceOp.MIN:
            tensor.fill_(2)
            return
        if tensor.dtype == torch.float32:
            tensor.fill_(8.0)
            return
        if tensor.dtype == torch.long and len(reduce_calls) == 3:
            tensor.fill_(4)
            return
        if tensor.dtype == torch.long and len(reduce_calls) == 4:
            tensor.fill_(12)
            return
        if tensor.dtype == torch.long and len(reduce_calls) == 5:
            tensor.fill_(4)
            return
        raise AssertionError(f"unexpected all_reduce call: {reduce_calls[-1]}")

    with mock.patch("litgpt.safemoe.pretrain._collect_cached_routing_counts", return_value=(3, 1)), mock.patch(
        "litgpt.safemoe.pretrain.chunked_cross_entropy", side_effect=[torch.tensor(1.0), torch.tensor(3.0)]
    ), mock.patch("torch.distributed.is_available", return_value=True), mock.patch(
        "torch.distributed.is_initialized", return_value=True
    ), mock.patch("torch.distributed.all_reduce", side_effect=_all_reduce):
        result = _evaluate_validation_split(
            fabric=fabric,
            model=model,
            val_dataloader=batches,
            split_name="D_std",
            max_iters=5,
            verbose=False,
        )

    assert model.forward_calls == 2
    assert result.batches_evaluated == 4
    assert result.loss.item() == pytest.approx(2.0)
    assert result.dispatch_count == 12
    assert result.harmful_dispatches == 4
    assert result.routing_harmful_frac == pytest.approx(1 / 3)


def test_collect_validation_summary_metric_keys():
    fabric = Fabric(accelerator="cpu", devices=1)
    split_results = [
        SplitValidationResult(
            loss=torch.tensor(1.0),
            ppl=math.exp(1.0),
            dispatch_count=10,
            harmful_dispatches=2,
            routing_harmful_frac=0.2,
            batches_evaluated=1,
        ),
        SplitValidationResult(
            loss=torch.tensor(0.5),
            ppl=math.exp(0.5),
            dispatch_count=12,
            harmful_dispatches=9,
            routing_harmful_frac=0.75,
            batches_evaluated=1,
        ),
    ]

    with mock.patch("litgpt.safemoe.pretrain._evaluate_validation_split", side_effect=split_results):
        summary = collect_validation_summary(
            fabric=fabric,
            model=mock.Mock(),
            val_loaders={"D_std": object(), "D_harmful": object()},
            max_iters=1,
            verbose=False,
            metric_prefix="ablated_",
        )

    assert isinstance(summary, ValidationSummary)
    assert summary.scalar_metrics["ablated_val_loss_D_std"] == 1.0
    assert summary.scalar_metrics["ablated_val_loss_D_harmful"] == 0.5
    assert summary.scalar_metrics["ablated_routing_margin"] == pytest.approx(0.55)


def test_main_rejects_resume_from_model_only_checkpoint(tmp_path):
    resume_path = tmp_path / "step-00000001" / "lit_model.pth"
    resume_path.parent.mkdir(parents=True)
    torch.save({"model": {"weights": torch.tensor([1.0])}}, resume_path)

    fabric = mock.Mock(
        global_rank=0,
        world_size=1,
        device=torch.device("cpu"),
        strategy=mock.Mock(),
    )
    fabric.seed_everything = mock.Mock()
    fabric.load_raw = mock.Mock()
    fabric.load = mock.Mock()
    fabric.print = mock.Mock()

    model = mock.Mock(max_seq_length=8, config=_make_safemoe_config())
    data = mock.Mock()
    data.initialize_loaders.return_value = {"val_loaders": {"D_std": object()}}
    registry = mock.Mock()

    with mock.patch.object(safemoe_pretrain, "_setup_model", return_value=model), mock.patch.object(
        safemoe_pretrain, "_setup_split_dataloaders", return_value={"D_std": object()}
    ), mock.patch.object(safemoe_pretrain, "HarmfulParamRegistry", return_value=registry), mock.patch.object(
        safemoe_pretrain, "_build_optimizer", return_value=mock.Mock()
    ), mock.patch.object(safemoe_pretrain, "find_resume_path", return_value=resume_path):
        with pytest.raises(ValueError, match="without optimizer state"):
            safemoe_pretrain.main(
                fabric=fabric,
                devices=1,
                num_nodes=1,
                seed=123,
                initial_checkpoint_dir=None,
                resume=True,
                config=_make_safemoe_config(),
                data=data,
                out_dir=tmp_path / "out",
                tokenizer_dir=None,
                tokenizer=None,
                train=SimpleNamespace(max_seq_length=None, micro_batch_size=1, epochs=None, max_tokens=1, max_norm=1.0),
                eval=SimpleNamespace(
                    max_iters=1,
                    interval=1,
                    initial_validation=False,
                    final_validation=False,
                    max_new_tokens=None,
                ),
                optimizer="AdamW",
            )

    fabric.load.assert_not_called()


def test_evaluate_with_ablation_restores_weights_and_logs_metrics():
    fixture = _make_registry_fixture()
    harmful_params = fixture.registry.parameters_by_type("theta_harmful")
    harmful_before = [param.detach().clone() for param in harmful_params]
    model = fixture.model
    model.train(True)

    summary = ValidationSummary(
        by_split={},
        scalar_metrics={"ablated_val_loss_D_std": 1.23, "ablated_routing_margin": 0.4},
        routing_margin=0.4,
    )
    fabric = mock.Mock()
    fabric.log_dict = mock.Mock()
    fabric.print = mock.Mock()

    def _check_ablation(*args, **kwargs):
        assert kwargs["max_iters"] == 3
        assert kwargs["metric_prefix"] == "ablated_"
        assert kwargs["verbose"] is False
        assert all(torch.count_nonzero(param) == 0 for param in harmful_params)
        return summary

    with mock.patch("litgpt.safemoe.pretrain.collect_validation_summary", side_effect=_check_ablation):
        result = evaluate_with_ablation(
            fabric=fabric,
            model=model,
            registry=fixture.registry,
            val_loaders={"D_std": object()},
            iter_num=7,
            eval_args=SimpleNamespace(max_iters=3),
        )

    assert result is summary
    assert model.training is True
    for param, before in zip(harmful_params, harmful_before):
        torch.testing.assert_close(param, before)
    fabric.log_dict.assert_called_once_with(summary.scalar_metrics, step=7)


def test_safemoe_layer_matches_llamamoe_without_masking():
    torch.manual_seed(1234)
    config = _make_safemoe_config(harmful_expert_indices=[])
    baseline = LLaMAMoE(config)
    safemoe = SafeMoELayer(config)
    safemoe.load_state_dict(baseline.state_dict())

    x = torch.randn(2, 3, config.n_embd)
    baseline_out = baseline(x)
    safemoe_out = safemoe(x)

    torch.testing.assert_close(safemoe_out, baseline_out)


def test_safedata_val_dataloaders_are_split_aware_iterables(tmp_path):
    data = SafeData(cache_dir=tmp_path, datasets={})
    data.connect(batch_size=2, max_seq_length=4)
    std_dataset = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), torch.tensor([7, 8, 9])]
    harm_dataset = [torch.tensor([10, 11, 12])]

    with mock.patch.object(data, "val_datasets", return_value={"D_std": std_dataset, "D_harmful": harm_dataset}):
        loaders = data.val_dataloaders()

    assert set(loaders) == {"D_std", "D_harmful"}

    std_batches = list(loaders["D_std"])
    harm_batches = list(loaders["D_harmful"])

    assert len(std_batches) == 2
    assert std_batches[0].shape == (2, 3)
    assert std_batches[1].shape == (1, 3)
    torch.testing.assert_close(std_batches[0], torch.tensor([[1, 2, 3], [4, 5, 6]]))
    torch.testing.assert_close(std_batches[1], torch.tensor([[7, 8, 9]]))

    assert len(harm_batches) == 1
    assert harm_batches[0].shape == (1, 3)
    torch.testing.assert_close(harm_batches[0], torch.tensor([[10, 11, 12]]))
    assert len(loaders["D_std"]) == 2
    assert len(loaders["D_harmful"]) == 1


def test_safedata_build_val_datasets_combines_all_validation_streams(tmp_path):
    data = SafeData(
        cache_dir=tmp_path,
        datasets={
            "std_a": {"role": "std"},
            "std_b": {"role": "std"},
            "harm_a": {"role": "harmful"},
            "harm_b": {"role": "harmful"},
        },
    )
    combined_std = object()
    combined_harm = object()

    with mock.patch.object(data, "_val_dir") as val_dir, mock.patch.object(
        data,
        "_streaming_dataset",
        side_effect=[mock.sentinel.std_a, mock.sentinel.std_b, mock.sentinel.harm_a, mock.sentinel.harm_b],
    ), mock.patch.object(data, "_combine_datasets", side_effect=[combined_std, combined_harm]) as combine_datasets:
        val_dir.return_value.exists.return_value = True
        result = data._build_val_datasets()

    assert result == {"D_std": combined_std, "D_harmful": combined_harm}
    combine_datasets.assert_has_calls(
        [
            mock.call([mock.sentinel.std_a, mock.sentinel.std_b], iterate_over_all=True),
            mock.call([mock.sentinel.harm_a, mock.sentinel.harm_b], iterate_over_all=True),
        ]
    )


def test_setup_split_dataloaders_returns_non_dataloader_iterables_unchanged():
    fabric = mock.Mock()
    loaders = {"D_std": object(), "D_harmful": object()}

    result = _setup_split_dataloaders(fabric, loaders)

    assert result == loaders
    fabric.setup_dataloaders.assert_not_called()


def test_setup_split_dataloaders_preserves_mapping_order_for_dataloaders():
    fabric = mock.Mock()
    dataset = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
    loaders = {
        "D_std": torch.utils.data.DataLoader(dataset, batch_size=1),
        "D_harmful": torch.utils.data.DataLoader(dataset, batch_size=1),
    }
    prepared = (mock.sentinel.std_loader, mock.sentinel.harm_loader)
    fabric.setup_dataloaders.return_value = prepared

    result = _setup_split_dataloaders(fabric, loaders)

    assert result == {"D_std": mock.sentinel.std_loader, "D_harmful": mock.sentinel.harm_loader}
    fabric.setup_dataloaders.assert_called_once_with(*loaders.values())
