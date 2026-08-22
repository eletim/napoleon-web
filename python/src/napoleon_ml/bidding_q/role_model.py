"""Role-decomposed Q network for compact278 bidding observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from napoleon_ml.dataset.constants import BIDDING_ACTION_COUNT
from napoleon_ml.dataset.tensors import BIDDING_MODEL_INPUT_FEATURE_COUNT

BIDDING_ROLE_Q_ARCHITECTURE_ID = "bidding-role-decomposed-q-mlp-v1"
BIDDING_ROLE_CLASSES: tuple[str, ...] = (
    "napoleon",
    "napoleon-adjutant",
    "adjutant",
    "citizen",
    "no-contract",
)
BIDDING_ROLE_COUNT = len(BIDDING_ROLE_CLASSES)


@dataclass(frozen=True)
class BiddingRoleQModelConfig:
    input_dim: int = BIDDING_MODEL_INPUT_FEATURE_COUNT
    hidden_dims: tuple[int, ...] = (512, 512, 256, 256)
    dropout: float = 0.0
    role_count: int = BIDDING_ROLE_COUNT

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
        if self.role_count != BIDDING_ROLE_COUNT:
            raise ValueError(f"role_count must be {BIDDING_ROLE_COUNT}, got {self.role_count}.")

    def to_dict(self) -> dict[str, object]:
        return {
            "architectureId": BIDDING_ROLE_Q_ARCHITECTURE_ID,
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "actionCount": BIDDING_ACTION_COUNT,
            "roleCount": self.role_count,
            "roleClasses": list(BIDDING_ROLE_CLASSES),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BiddingRoleQModelConfig:
        architecture_id = value.get("architectureId")
        if architecture_id != BIDDING_ROLE_Q_ARCHITECTURE_ID:
            raise ValueError(
                f"architectureId must be {BIDDING_ROLE_Q_ARCHITECTURE_ID!r}, "
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
            role_count=_require_int(value.get("roleCount"), "roleCount"),
        )


class BiddingRoleQModel(nn.Module):
    """Predict P(role | state, action) and V(state, action, role)."""

    def __init__(self, config: BiddingRoleQModelConfig) -> None:
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
        output_dim = BIDDING_ACTION_COUNT * config.role_count
        self.role_logits_head = nn.Linear(input_dim, output_dim)
        self.role_value_head = nn.Linear(input_dim, output_dim)

    def forward(self, model_input: Tensor) -> tuple[Tensor, Tensor]:
        if model_input.ndim != 2:
            raise ValueError(
                f"model_input must have shape (batch, features), got {tuple(model_input.shape)}."
            )
        if model_input.shape[1] != self.config.input_dim:
            raise ValueError(
                f"model_input feature count must be {self.config.input_dim}, "
                f"got {model_input.shape[1]}."
            )
        encoded = self.encoder(model_input)
        batch_size = model_input.shape[0]
        logits = self.role_logits_head(encoded).reshape(
            batch_size,
            BIDDING_ACTION_COUNT,
            self.config.role_count,
        )
        values = self.role_value_head(encoded).reshape(
            batch_size,
            BIDDING_ACTION_COUNT,
            self.config.role_count,
        )
        return logits, values


def compose_role_q(role_logits: Tensor, role_values: Tensor) -> Tensor:
    if role_logits.shape != role_values.shape:
        raise ValueError("role_logits and role_values must have the same shape.")
    if role_logits.ndim != 3:
        raise ValueError("role logits and values must have shape (batch, 29, roleCount).")
    if role_logits.shape[1] != BIDDING_ACTION_COUNT:
        raise ValueError(f"action dimension must be {BIDDING_ACTION_COUNT}.")
    role_probabilities = torch.softmax(role_logits, dim=2)
    return torch.sum(role_probabilities * role_values, dim=2)


def create_seeded_bidding_role_q_model(
    config: BiddingRoleQModelConfig,
    *,
    seed: int,
) -> BiddingRoleQModel:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    torch.manual_seed(seed)
    return BiddingRoleQModel(config)


def _require_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer.")
    return value


def _require_float(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{path} must be a number.")
    return float(value)
