from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from napoleon_ml.policy.device import (
    TorchDeviceResolutionError,
    cpu_state_dict,
    resolve_torch_device,
)


def test_resolve_torch_device_auto_uses_cpu_when_cuda_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    resolved = resolve_torch_device("auto")

    assert resolved.requested == "auto"
    assert resolved.resolved == "cpu"
    assert resolved.torch_device.type == "cpu"
    assert resolved.cuda_device_name is None


def test_resolve_torch_device_cuda_fails_close_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(TorchDeviceResolutionError, match="explicitly requested"):
        resolve_torch_device("cuda")


def test_resolve_torch_device_auto_and_cuda_use_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "Mock CUDA")

    auto = resolve_torch_device("auto")
    explicit = resolve_torch_device("cuda")

    assert auto.requested == "auto"
    assert auto.resolved == "cuda"
    assert auto.torch_device.type == "cuda"
    assert auto.cuda_device_name == "Mock CUDA"
    assert explicit.requested == "cuda"
    assert explicit.resolved == "cuda"
    assert explicit.cuda_device_name == "Mock CUDA"


def test_cpu_state_dict_detaches_and_moves_values_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = nn.Linear(1, 1)
    fake_value = _FakeStateValue()

    def fake_state_dict() -> dict[str, Any]:
        return {"weight": fake_value}

    monkeypatch.setattr(model, "state_dict", fake_state_dict)

    result = cpu_state_dict(model)

    assert fake_value.detach_called
    assert fake_value.cpu_called
    torch.testing.assert_close(result["weight"], torch.tensor([1.0]))


class _FakeStateValue:
    detach_called: bool
    cpu_called: bool

    def __init__(self) -> None:
        self.detach_called = False
        self.cpu_called = False

    def detach(self) -> _FakeStateValue:
        self.detach_called = True
        return self

    def cpu(self) -> torch.Tensor:
        self.cpu_called = True
        return torch.tensor([1.0])
