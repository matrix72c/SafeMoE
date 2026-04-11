from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from litgpt.config import Config


class HarmfulParamRegistry:
    _EXPERT_RE = re.compile(r"transformer\.h\.\d+\.mlp\.experts\.(\d+)\.")

    def __init__(self, model: nn.Module, config: Config) -> None:
        self._harmful_expert_indices = tuple(config.harmful_expert_indices)
        harmful_expert_indices = set(self._harmful_expert_indices)
        registry_names: dict[str, list[str]] = {
            "theta_harmful": [],
            "theta_std": [],
            "theta_shared": [],
        }

        for name, _ in model.named_parameters():
            clean = name.replace("_fsdp_wrapped_module.", "")
            match = self._EXPERT_RE.match(clean)
            if match:
                expert_idx = int(match.group(1))
                split = "theta_harmful" if expert_idx in harmful_expert_indices else "theta_std"
                registry_names[split].append(clean)
            else:
                registry_names["theta_shared"].append(clean)

        self._registry_names = registry_names
        self._registry = self._resolve_registry(model)

    def _resolve_parameter(self, model: nn.Module, name: str) -> nn.Parameter:
        try:
            return model.get_parameter(name)
        except AttributeError as ex:
            raise RuntimeError(f"Failed to resolve SafeMoE parameter '{name}' on the live model.") from ex

    def _resolve_registry(self, model: nn.Module) -> dict[str, list[nn.Parameter]]:
        return {
            split: [self._resolve_parameter(model, name) for name in names]
            for split, names in self._registry_names.items()
        }

    def bind(self, model: nn.Module) -> None:
        self._registry = self._resolve_registry(model)

    def parameters_by_type(self, split: str) -> list[nn.Parameter]:
        return self._registry[split]

    def validate(self) -> None:
        if self._harmful_expert_indices and not self._registry_names["theta_harmful"]:
            raise RuntimeError(
                "SafeMoE harmful parameter registry is empty despite configured harmful experts: "
                f"harmful_expert_indices={list(self._harmful_expert_indices)}. "
                "Build HarmfulParamRegistry before FSDP wrapping so expert parameter names are still visible."
            )
        if self._harmful_expert_indices and not self._registry["theta_harmful"]:
            raise RuntimeError(
                "SafeMoE harmful parameter registry could not bind harmful parameters onto the live model: "
                f"harmful_expert_indices={list(self._harmful_expert_indices)}."
            )


class GradientMasker:
    def __init__(self, registry: HarmfulParamRegistry) -> None:
        self._registry = registry

    def mask(self, split_label: str) -> None:
        if split_label == "D_unlabeled":
            return
        clear_group = "theta_harmful" if split_label == "D_std" else "theta_std"
        for param in self._registry.parameters_by_type(clear_group):
            param.grad = None


@contextmanager
def temporarily_ablate_harmful_params(
    registry: HarmfulParamRegistry,
    *,
    offload_to_cpu: bool = True,
) -> Iterator[None]:
    restore_buffers: list[tuple[nn.Parameter, torch.Tensor]] = []
    offload_device = torch.device("cpu") if offload_to_cpu else None

    try:
        for param in registry.parameters_by_type("theta_harmful"):
            restore_value = (
                param.detach().to(device=offload_device, dtype=param.dtype, copy=True)
                if offload_device is not None
                else param.detach().clone()
            )
            restore_buffers.append((param, restore_value))
            param.data.zero_()
        yield
    finally:
        for param, restore_value in restore_buffers:
            restore_tensor = restore_value.to(device=param.device, dtype=param.dtype, copy=False)
            param.data.copy_(restore_tensor)


class ActivationMasker:
    def __init__(self, model: nn.Module) -> None:
        from litgpt.model import SafeMoELayer

        self._layers: list[SafeMoELayer] = [m for m in model.modules() if isinstance(m, SafeMoELayer)]

    def enable(self) -> None:
        for layer in self._layers:
            layer._activation_masking_enabled = True

    def disable(self) -> None:
        for layer in self._layers:
            layer._activation_masking_enabled = False
