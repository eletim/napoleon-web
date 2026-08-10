"""Device resolution and CUDA-aware timing helpers for playing policy training."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

RequestedTorchDevice = Literal["auto", "cpu", "cuda"]
SUPPORTED_TORCH_DEVICES: tuple[RequestedTorchDevice, ...] = ("auto", "cpu", "cuda")


@dataclass(frozen=True)
class ResolvedTorchDevice:
    requested: RequestedTorchDevice
    resolved: Literal["cpu", "cuda"]
    torch_device: torch.device
    cuda_device_name: str | None = None

    def to_metadata(self) -> dict[str, object]:
        return {
            "requestedDevice": self.requested,
            "resolvedDevice": self.resolved,
            "cudaDeviceName": self.cuda_device_name,
        }


class TorchDeviceResolutionError(ValueError):
    """Raised when a requested PyTorch device cannot be used."""


def resolve_torch_device(requested: str) -> ResolvedTorchDevice:
    if requested not in SUPPORTED_TORCH_DEVICES:
        choices = ", ".join(SUPPORTED_TORCH_DEVICES)
        raise TorchDeviceResolutionError(
            f"device must be one of {choices}, got {requested!r}."
        )

    normalized = requested
    if normalized == "cpu":
        return ResolvedTorchDevice(
            requested="cpu",
            resolved="cpu",
            torch_device=torch.device("cpu"),
        )

    cuda_available = torch.cuda.is_available()
    if normalized == "cuda" and not cuda_available:
        raise TorchDeviceResolutionError(
            "CUDA was explicitly requested with --device cuda, but torch.cuda.is_available() "
            "is false."
        )

    if cuda_available:
        return ResolvedTorchDevice(
            requested=normalized,
            resolved="cuda",
            torch_device=torch.device("cuda"),
            cuda_device_name=torch.cuda.get_device_name(),
        )

    return ResolvedTorchDevice(
        requested="auto",
        resolved="cpu",
        torch_device=torch.device("cpu"),
    )


def synchronize_device(device: ResolvedTorchDevice) -> None:
    if device.resolved == "cuda":
        torch.cuda.synchronize(device.torch_device)


def start_timing(device: ResolvedTorchDevice) -> float:
    synchronize_device(device)
    return time.perf_counter()


def elapsed_seconds_since(start: float, device: ResolvedTorchDevice) -> float:
    synchronize_device(device)
    return time.perf_counter() - start


def cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}
