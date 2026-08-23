"""Simple scalar value MLP for exchange discard combinations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn

from .dataset import EXCHANGE_VALUE_INPUT_FEATURE_COUNT

EXCHANGE_VALUE_ARCHITECTURE_ID = "exchange-combination-value-mlp-v1"


@dataclass(frozen=True)
class ExchangeValueMlpConfig:
    input_dim: int = EXCHANGE_VALUE_INPUT_FEATURE_COUNT
    hidden_dims: tuple[int, ...] = (512, 512, 256, 256)
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.input_dim, bool) or self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if isinstance(self.hidden_dims, list):
            object.__setattr__(self, "hidden_dims", tuple(self.hidden_dims))
        if not self.hidden_dims:
            raise ValueError("hidden_dims must be non-empty.")
        for index, width in enumerate(self.hidden_dims):
            if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
                raise ValueError(f"hidden_dims[{index}] must be a positive integer.")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0,1).")

    def to_dict(self) -> dict[str, object]:
        return {
            "architectureId": EXCHANGE_VALUE_ARCHITECTURE_ID,
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "output": "scalarContractMargin",
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ExchangeValueMlpConfig:
        if value.get("architectureId") != EXCHANGE_VALUE_ARCHITECTURE_ID:
            raise ValueError("architectureId mismatch.")
        hidden_raw = value.get("hidden_dims")
        if not isinstance(hidden_raw, list | tuple):
            raise ValueError("hidden_dims must be a list.")
        return cls(
            input_dim=_require_int(value.get("input_dim"), "input_dim"),
            hidden_dims=tuple(
                _require_int(item, f"hidden_dims[{index}]")
                for index, item in enumerate(hidden_raw)
            ),
            dropout=_require_float(value.get("dropout"), "dropout"),
        )


class ExchangeValueMlpModel(nn.Module):
    """Map exchange observation + 53-bit candidate mask to scalar margin."""

    def __init__(self, config: ExchangeValueMlpConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        input_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, value_input: Tensor) -> Tensor:
        if value_input.ndim != 2:
            raise ValueError("value_input must have shape (batch, features).")
        if value_input.shape[1] != self.config.input_dim:
            raise ValueError(
                f"value_input feature count must be {self.config.input_dim}, "
                f"got {value_input.shape[1]}."
            )
        return cast(Tensor, self.network(value_input).squeeze(-1))


def create_seeded_exchange_value_model(
    config: ExchangeValueMlpConfig,
    *,
    seed: int,
) -> ExchangeValueMlpModel:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    torch.manual_seed(seed)
    return ExchangeValueMlpModel(config)


def _require_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer.")
    return value


def _require_float(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{path} must be a number.")
    return float(value)
