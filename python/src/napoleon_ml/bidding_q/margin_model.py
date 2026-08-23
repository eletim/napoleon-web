"""Dedicated heteroscedastic contract-margin model for Issue #380."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from napoleon_ml.dataset.constants import BIDDING_ACTION_COUNT
from napoleon_ml.dataset.tensors import BIDDING_MODEL_INPUT_FEATURE_COUNT

BIDDING_MARGIN_HETEROSCEDASTIC_ARCHITECTURE_ID = "bidding-margin-heteroscedastic-mlp-v1"


@dataclass(frozen=True)
class BiddingMarginHeteroscedasticModelConfig:
    input_dim: int = BIDDING_MODEL_INPUT_FEATURE_COUNT
    hidden_dims: tuple[int, ...] = (512, 512, 256, 256)
    dropout: float = 0.0
    log_variance_min: float = -6.0
    log_variance_max: float = 5.0

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
        if self.log_variance_min >= self.log_variance_max:
            raise ValueError("log_variance_min must be less than log_variance_max.")

    def to_dict(self) -> dict[str, object]:
        return {
            "architectureId": BIDDING_MARGIN_HETEROSCEDASTIC_ARCHITECTURE_ID,
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "actionCount": BIDDING_ACTION_COUNT,
            "logVarianceMin": self.log_variance_min,
            "logVarianceMax": self.log_variance_max,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BiddingMarginHeteroscedasticModelConfig:
        architecture_id = value.get("architectureId")
        if architecture_id != BIDDING_MARGIN_HETEROSCEDASTIC_ARCHITECTURE_ID:
            raise ValueError(
                f"architectureId must be {BIDDING_MARGIN_HETEROSCEDASTIC_ARCHITECTURE_ID!r}, "
                f"got {architecture_id!r}."
            )
        hidden_dims_raw = value.get("hidden_dims")
        if not isinstance(hidden_dims_raw, list | tuple):
            raise ValueError("hidden_dims must be a list of integers.")
        return cls(
            input_dim=_require_int(value.get("input_dim"), "input_dim"),
            hidden_dims=tuple(
                _require_int(item, f"hidden_dims[{index}]")
                for index, item in enumerate(hidden_dims_raw)
            ),
            dropout=_require_float(value.get("dropout"), "dropout"),
            log_variance_min=_require_float(value.get("logVarianceMin"), "logVarianceMin"),
            log_variance_max=_require_float(value.get("logVarianceMax"), "logVarianceMax"),
        )


class BiddingMarginHeteroscedasticModel(nn.Module):
    """Predict contract-margin mean and log variance for every bidding action."""

    def __init__(self, config: BiddingMarginHeteroscedasticModelConfig) -> None:
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
        self.mean_head = nn.Linear(input_dim, BIDDING_ACTION_COUNT)
        self.log_variance_head = nn.Linear(input_dim, BIDDING_ACTION_COUNT)

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
        mean = self.mean_head(encoded)
        log_variance = torch.clamp(
            self.log_variance_head(encoded),
            min=self.config.log_variance_min,
            max=self.config.log_variance_max,
        )
        return mean, log_variance


def margin_sigma_from_log_variance(log_variance: Tensor) -> Tensor:
    return torch.exp(0.5 * log_variance)


def create_seeded_bidding_margin_model(
    config: BiddingMarginHeteroscedasticModelConfig,
    *,
    seed: int,
) -> BiddingMarginHeteroscedasticModel:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    torch.manual_seed(seed)
    return BiddingMarginHeteroscedasticModel(config)


def _require_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer.")
    return value


def _require_float(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{path} must be a number.")
    return float(value)
