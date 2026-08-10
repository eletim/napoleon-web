from __future__ import annotations

import pytest
import torch

from napoleon_ml.policy.device import TorchDeviceResolutionError, resolve_torch_device


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
