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
        harmful_expert_indices = set(config.harmful_expert_indices)
        theta_harmful: list[nn.Parameter] = []
        theta_std: list[nn.Parameter] = []
        theta_shared: list[nn.Parameter] = []

        for name, param in model.named_parameters():
            clean = name.replace("_fsdp_wrapped_module.", "")
            match = self._EXPERT_RE.match(clean)
            if match:
                expert_idx = int(match.group(1))
                if expert_idx in harmful_expert_indices:
                    theta_harmful.append(param)
                else:
                    theta_std.append(param)
            else:
                theta_shared.append(param)

        self._registry: dict[str, list[nn.Parameter]] = {
            "theta_harmful": theta_harmful,
            "theta_std": theta_std,
            "theta_shared": theta_shared,
        }

    def parameters_by_type(self, split: str) -> list[nn.Parameter]:
        return self._registry[split]


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
