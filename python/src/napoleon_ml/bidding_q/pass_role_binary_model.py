"""PASS-only Adjutant-vs-Citizen classifier for Issue #395."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from napoleon_ml.dataset.tensors import BIDDING_MODEL_INPUT_FEATURE_COUNT

BIDDING_PASS_ROLE_BINARY_ARCHITECTURE_ID = "bidding-pass-role-binary-mlp-v1"


@dataclass(frozen=True)
class BiddingPassRoleBinaryModelConfig:
    input_dim: int = BIDDING_MODEL_INPUT_FEATURE_COUNT
    hidden_dims: tuple[int, ...] = (512, 512, 256, 256)
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.input_dim, bool) or self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if isinstance(self.hidden_dims, list):
            object.__setattr__(self, "hidden_dims", tuple(self.hidden_dims))
        if not isinstance(self.hidden_dims, tuple) or not self.hidden_dims:
            raise ValueError("hidden_dims must be a non-empty tuple.")
        for index, width in enumerate(self.hidden_dims):
            if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
                raise ValueError(f"hidden_dims[{index}] must be a positive integer.")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0.0, 1.0).")

    def to_dict(self) -> dict[str, object]:
        return {
            "architectureId": BIDDING_PASS_ROLE_BINARY_ARCHITECTURE_ID,
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "output": "adjutant_logit",
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BiddingPassRoleBinaryModelConfig:
        architecture_id = value.get("architectureId")
        if architecture_id != BIDDING_PASS_ROLE_BINARY_ARCHITECTURE_ID:
            raise ValueError(
                f"architectureId must be {BIDDING_PASS_ROLE_BINARY_ARCHITECTURE_ID!r}, "
                f"got {architecture_id!r}."
            )
        hidden_dims = value.get("hidden_dims")
        if not isinstance(hidden_dims, list | tuple):
            raise ValueError("hidden_dims must be a list.")
        return cls(
            input_dim=_require_int(value.get("input_dim"), "input_dim"),
            hidden_dims=tuple(
                _require_int(item, f"hidden_dims[{index}]")
                for index, item in enumerate(hidden_dims)
            ),
            dropout=_require_float(value.get("dropout"), "dropout"),
        )


class BiddingPassRoleBinaryModel(nn.Module):
    """Predict P(Adjutant | state, PASS)."""

    def __init__(self, config: BiddingPassRoleBinaryModelConfig) -> None:
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
        self.encoder = nn.Sequential(*layers)
        self.logit_head = nn.Linear(input_dim, 1)

    def forward(self, model_input: Tensor) -> Tensor:
        if model_input.ndim != 2:
            raise ValueError("model_input must have shape (batch, features).")
        if model_input.shape[1] != self.config.input_dim:
            raise ValueError(
                f"model_input feature count must be {self.config.input_dim}, "
                f"got {model_input.shape[1]}."
            )
        return self.logit_head(self.encoder(model_input)).squeeze(1)


def create_seeded_bidding_pass_role_binary_model(
    config: BiddingPassRoleBinaryModelConfig,
    *,
    seed: int,
) -> BiddingPassRoleBinaryModel:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    torch.manual_seed(seed)
    return BiddingPassRoleBinaryModel(config)


def _require_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer.")
    return value


def _require_float(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{path} must be a number.")
    return float(value)
