"""29-output Q network for compact278 bidding observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn

from napoleon_ml.dataset.constants import BIDDING_ACTION_COUNT
from napoleon_ml.dataset.tensors import BIDDING_MODEL_INPUT_FEATURE_COUNT

BIDDING_Q_MLP_ARCHITECTURE_ID = "bidding-q-mlp-v1"


@dataclass(frozen=True)
class BiddingQModelConfig:
    input_dim: int = BIDDING_MODEL_INPUT_FEATURE_COUNT
    hidden_dims: tuple[int, ...] = (512, 512, 256, 256)
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.input_dim, bool) or self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}.")
        if isinstance(self.hidden_dims, list):
            object.__setattr__(self, "hidden_dims", tuple(self.hidden_dims))
        if not isinstance(self.hidden_dims, tuple) or len(self.hidden_dims) == 0:
            raise ValueError("hidden_dims must be a non-empty tuple.")
        for index, width in enumerate(self.hidden_dims):
            if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
                raise ValueError(f"hidden_dims[{index}] must be a positive integer.")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}.")

    def to_dict(self) -> dict[str, object]:
        return {
            "architectureId": BIDDING_Q_MLP_ARCHITECTURE_ID,
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "output_dim": BIDDING_ACTION_COUNT,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BiddingQModelConfig:
        architecture_id = value.get("architectureId")
        if architecture_id != BIDDING_Q_MLP_ARCHITECTURE_ID:
            raise ValueError(
                f"architectureId must be {BIDDING_Q_MLP_ARCHITECTURE_ID!r}, "
                f"got {architecture_id!r}."
            )
        hidden_dims_raw = value.get("hidden_dims")
        if not isinstance(hidden_dims_raw, list | tuple):
            raise ValueError("hidden_dims must be a list of integers.")
        hidden_dims = tuple(
            _require_int(item, f"hidden_dims[{index}]")
            for index, item in enumerate(hidden_dims_raw)
        )
        return cls(
            input_dim=_require_int(value.get("input_dim"), "input_dim"),
            hidden_dims=hidden_dims,
            dropout=_require_float(value.get("dropout"), "dropout"),
        )


class BiddingQModel(nn.Module):
    """Map compact278 bidding input to one scalar Q value per bidding action."""

    def __init__(self, config: BiddingQModelConfig) -> None:
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
        layers.append(nn.Linear(input_dim, BIDDING_ACTION_COUNT))
        self.network = nn.Sequential(*layers)

    def forward(self, model_input: Tensor) -> Tensor:
        if model_input.ndim != 2:
            raise ValueError(
                f"model_input must have shape (batch, features), got {tuple(model_input.shape)}."
            )
        if model_input.shape[1] != self.config.input_dim:
            raise ValueError(
                f"model_input feature count must be {self.config.input_dim}, "
                f"got {model_input.shape[1]}."
            )
        return cast(Tensor, self.network(model_input))


def create_seeded_bidding_q_model(config: BiddingQModelConfig, *, seed: int) -> BiddingQModel:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    torch.manual_seed(seed)
    return BiddingQModel(config)


def _require_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer.")
    return value


def _require_float(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{path} must be a number.")
    return float(value)
