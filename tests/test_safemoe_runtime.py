import math
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from lightning import Fabric
from litdata.streaming import StreamingDataLoader

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
    _distributed_eval_step_budget,
    _evaluate_validation_split,
    _setup_split_dataloaders,
    _validate_resume_checkpoint,
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
        harmful_attn_heads=[0, 2],
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


def test_harmful_param_registry_classifies_experts_and_qkv_slices():
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
    assert id(layer.attn.qkv.weight) in std_params
    assert id(layer.mlp.gate.weight) in shared_params
    assert harmful_params | std_params | shared_params == all_params

    harmful_slices = registry._qkv_harmful_metadata[0][1]
    std_slices = registry._qkv_std_metadata[0][1]
    assert harmful_slices
    assert std_slices
    assert all(isinstance(row_slice, slice) for row_slice in harmful_slices)


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
    fixture = _make_registry_fixture(_make_safemoe_config(harmful_attn_heads=[]))
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


def test_temporarily_ablate_harmful_params_restores_qkv_and_experts():
    fixture = _make_registry_fixture()
    harmful_param = fixture.registry.parameters_by_type("theta_harmful")[0]
    harmful_before = harmful_param.detach().clone()
    qkv_param = fixture.registry._qkv_harmful_metadata[0][0]
    qkv_slices = fixture.registry._qkv_harmful_metadata[0][1]
    qkv_before = qkv_param.detach().clone()

    with temporarily_ablate_harmful_params(fixture.registry):
        assert torch.count_nonzero(harmful_param) == 0
        for row_slice in qkv_slices:
            assert torch.count_nonzero(qkv_param[row_slice]) == 0

    torch.testing.assert_close(harmful_param, harmful_before)
    torch.testing.assert_close(qkv_param, qkv_before)


def test_collect_cached_routing_counts_is_tensor_native():
    config = _make_safemoe_config(harmful_attn_heads=[])
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


def test_distributed_eval_step_budget_uses_global_minimum():
    fabric = mock.Mock(device=torch.device("cpu"))
    dataloader = mock.Mock()
    dataloader.__len__ = mock.Mock(return_value=7)

    with mock.patch("torch.distributed.is_available", return_value=True), mock.patch(
        "torch.distributed.is_initialized", return_value=True
    ), mock.patch("torch.distributed.all_reduce") as all_reduce:
        def _reduce_to_min(tensor, op=None):
            assert op == torch.distributed.ReduceOp.MIN
            tensor.fill_(3)

        all_reduce.side_effect = _reduce_to_min
        budget = _distributed_eval_step_budget(fabric, dataloader, max_iters=5)

    assert budget == 3


def test_evaluate_validation_split_uses_safe_eval_budget():
    class _EvalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.max_seq_length = 3
            self.config = SimpleNamespace(harmful_expert_indices=[])
            self._outputs = [torch.randn(1, 3, 8), torch.randn(1, 3, 8)]
            self.forward_calls = 0

        def forward(self, input_ids):
            output = self._outputs[self.forward_calls]
            self.forward_calls += 1
            return output

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

    with mock.patch("litgpt.safemoe.pretrain._distributed_eval_step_budget", return_value=2), mock.patch(
        "litgpt.safemoe.pretrain._collect_cached_routing_counts", return_value=(4, 1)
    ), mock.patch("litgpt.safemoe.pretrain.chunked_cross_entropy", side_effect=[torch.tensor(1.0), torch.tensor(3.0)]), mock.patch(
        "litgpt.safemoe.pretrain._reduce_validation_totals", side_effect=lambda fabric, total_loss, count: (torch.tensor(total_loss), count)
    ), mock.patch(
        "litgpt.safemoe.pretrain._reduce_routing_counts", side_effect=lambda fabric, total_dispatches, harmful_dispatches: (total_dispatches, harmful_dispatches)
    ):
        result = _evaluate_validation_split(
            fabric=fabric,
            model=model,
            val_dataloader=batches,
            split_name="D_std",
            max_iters=5,
            verbose=False,
        )

    assert result.batches_evaluated == 2
    assert result.loss.item() == pytest.approx(2.0)
    assert result.dispatch_count == 8
    assert result.harmful_dispatches == 2
    assert result.routing_harmful_frac == pytest.approx(0.25)
    assert model.forward_calls == 2
    assert model.training is True


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


def test_evaluate_with_ablation_restores_weights_and_logs_metrics():
    fixture = _make_registry_fixture()
    harmful_params = fixture.registry.parameters_by_type("theta_harmful")
    qkv_param = fixture.registry._qkv_harmful_metadata[0][0]
    qkv_slices = fixture.registry._qkv_harmful_metadata[0][1]
    harmful_before = [param.detach().clone() for param in harmful_params]
    qkv_before = qkv_param.detach().clone()
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
        for row_slice in qkv_slices:
            assert torch.count_nonzero(qkv_param[row_slice]) == 0
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
    torch.testing.assert_close(qkv_param, qkv_before)
    fabric.log_dict.assert_called_once_with(summary.scalar_metrics, step=7)


def test_validate_resume_checkpoint_rejects_model_only_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "step-00000001" / "lit_model.pth"
    checkpoint_path.parent.mkdir(parents=True)
    torch.save({"model": {"weights": torch.tensor([1.0])}}, checkpoint_path)

    with pytest.raises(ValueError, match="without optimizer state"):
        _validate_resume_checkpoint(checkpoint_path)



def test_validate_resume_checkpoint_allows_full_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "final" / "lit_model.pth"
    checkpoint_path.parent.mkdir(parents=True)
    torch.save({"model": {"weights": torch.tensor([1.0])}, "optimizer": {"state": {}, "param_groups": []}}, checkpoint_path)

    _validate_resume_checkpoint(checkpoint_path)



def test_safemoe_layer_matches_llamamoe_without_masking():
    torch.manual_seed(1234)
    config = _make_safemoe_config(harmful_expert_indices=[], harmful_attn_heads=[])
    baseline = LLaMAMoE(config)
    safemoe = SafeMoELayer(config)
    safemoe.load_state_dict(baseline.state_dict())

    x = torch.randn(2, 3, config.n_embd)
    baseline_out = baseline(x)
    safemoe_out = safemoe(x)

    torch.testing.assert_close(safemoe_out, baseline_out)


def test_safedata_val_dataloaders_are_streaming_loaders(tmp_path):
    data = SafeData(cache_dir=tmp_path, datasets={})
    data.connect(batch_size=2, max_seq_length=4)
    fake_dataset = object()
    fake_loader = mock.MagicMock(spec=StreamingDataLoader)

    with mock.patch.object(data, "val_datasets", return_value={"D_std": fake_dataset, "D_harmful": fake_dataset}), mock.patch.object(
        data, "_effective_num_workers", return_value=0
    ), mock.patch.object(data, "_streaming_dataloader", return_value=fake_loader) as streaming_dataloader:
        loaders = data.val_dataloaders()

    assert set(loaders) == {"D_std", "D_harmful"}
    assert all(loader is fake_loader for loader in loaders.values())
    assert streaming_dataloader.call_count == 2


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


def test_setup_split_dataloaders_preserves_mapping_order():
    fabric = mock.Mock()
    loaders = {"D_std": object(), "D_harmful": object()}
    prepared = (mock.sentinel.std_loader, mock.sentinel.harm_loader)
    fabric.setup_dataloaders.return_value = prepared

    result = _setup_split_dataloaders(fabric, loaders)

    assert result == {"D_std": mock.sentinel.std_loader, "D_harmful": mock.sentinel.harm_loader}
    fabric.setup_dataloaders.assert_called_once_with(*loaders.values())
