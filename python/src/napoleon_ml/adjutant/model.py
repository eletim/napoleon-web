"""Small CPU-friendly MLP for adjutant-card logits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn

from napoleon_ml.dataset.constants import CARD_COUNT
from napoleon_ml.dataset.tensors import ADJUTANT_MODEL_INPUT_FEATURE_COUNT


@dataclass(frozen=True)
class AdjutantMlpConfig:
    input_dim: int = ADJUTANT_MODEL_INPUT_FEATURE_COUNT
    hidden_dim: int = 128
    hidden_layers: int = 2
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.input_dim != ADJUTANT_MODEL_INPUT_FEATURE_COUNT:
            raise ValueError(
                "input_dim must be "
                f"{ADJUTANT_MODEL_INPUT_FEATURE_COUNT}, got {self.input_dim}."
            )

        if isinstance(self.hidden_dim, bool) or self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}.")

        if isinstance(self.hidden_layers, bool) or self.hidden_layers <= 0:
            raise ValueError(f"hidden_layers must be positive, got {self.hidden_layers}.")

        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}.")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AdjutantMlpConfig:
        return cls(
            input_dim=_require_int(value, "input_dim"),
            hidden_dim=_require_int(value, "hidden_dim"),
            hidden_layers=_require_int(value, "hidden_layers"),
            dropout=_require_float(value, "dropout"),
        )


class AdjutantMlpModel(nn.Module):
    """Map one adjutant-observation model_input to one logit per CARD_IDS index."""

    def __init__(self, config: AdjutantMlpConfig) -> None:
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []
        input_dim = config.input_dim

        for _ in range(config.hidden_layers):
            layers.append(nn.Linear(input_dim, config.hidden_dim))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            input_dim = config.hidden_dim

        layers.append(nn.Linear(input_dim, CARD_COUNT))
        self.network = nn.Sequential(*layers)

    def forward(self, model_input: Tensor) -> Tensor:
        if model_input.ndim != 2:
            raise ValueError(
                f"model_input must have shape (batch, features), got {model_input.shape}."
            )

        if model_input.shape[1] != self.config.input_dim:
            raise ValueError(
                f"model_input feature count must be {self.config.input_dim}, "
                f"got {model_input.shape[1]}."
            )

        return cast(Tensor, self.network(model_input))


class AdjutantActorCriticModel(nn.Module):
    """Separated adjutant Actor-Critic model."""

    def __init__(self, config: AdjutantMlpConfig) -> None:
        super().__init__()
        self.config = config
        self.actor = AdjutantMlpModel(config)
        self.critic = _build_critic(config)

    def forward(self, model_input: Tensor) -> Tensor:
        return cast(Tensor, self.actor(model_input))

    def value(self, model_input: Tensor) -> Tensor:
        if model_input.ndim != 2:
            raise ValueError(
                f"model_input must have shape (batch, features), got {model_input.shape}."
            )

        if model_input.shape[1] != self.config.input_dim:
            raise ValueError(
                f"model_input feature count must be {self.config.input_dim}, "
                f"got {model_input.shape[1]}."
            )

        value = cast(Tensor, self.critic(model_input))
        return value.squeeze(-1)


def create_seeded_adjutant_model(config: AdjutantMlpConfig, *, seed: int) -> AdjutantMlpModel:
    """Create a model after setting PyTorch's CPU RNG seed."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an integer, got {type(seed).__name__}.")

    torch.manual_seed(seed)
    return AdjutantMlpModel(config)


def create_seeded_adjutant_actor_critic_model(
    config: AdjutantMlpConfig, *, seed: int
) -> AdjutantActorCriticModel:
    """Create a separated Actor-Critic model after setting PyTorch's CPU RNG seed."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an integer, got {type(seed).__name__}.")

    torch.manual_seed(seed)
    return AdjutantActorCriticModel(config)


def _build_critic(config: AdjutantMlpConfig) -> nn.Sequential:
    layers: list[nn.Module] = []
    input_dim = config.input_dim

    for _ in range(config.hidden_layers):
        layers.append(nn.Linear(input_dim, config.hidden_dim))
        layers.append(nn.ReLU())
        if config.dropout > 0.0:
            layers.append(nn.Dropout(config.dropout))
        input_dim = config.hidden_dim

    layers.append(nn.Linear(input_dim, 1))
    return nn.Sequential(*layers)


def _require_int(value: dict[str, Any], key: str) -> int:
    item = value.get(key)

    if isinstance(item, bool) or not isinstance(item, int):
        raise ValueError(f"model_config.{key} must be an integer.")

    return item


def _require_float(value: dict[str, Any], key: str) -> float:
    item = value.get(key)

    if isinstance(item, bool) or not isinstance(item, int | float):
        raise ValueError(f"model_config.{key} must be a number.")

    return float(item)
