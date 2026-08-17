"""Small CPU-friendly MLP for legal bidding policy logits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn

from napoleon_ml.dataset.constants import BIDDING_ACTION_COUNT
from napoleon_ml.dataset.tensors import BIDDING_MODEL_INPUT_FEATURE_COUNT


@dataclass(frozen=True)
class BiddingMlpConfig:
    input_dim: int = BIDDING_MODEL_INPUT_FEATURE_COUNT
    hidden_dim: int = 128
    hidden_layers: int = 2
    hidden_dims: tuple[int, ...] | None = None
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.input_dim, bool) or self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}.")

        if isinstance(self.hidden_dim, bool) or self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}.")

        if isinstance(self.hidden_layers, bool) or self.hidden_layers <= 0:
            raise ValueError(f"hidden_layers must be positive, got {self.hidden_layers}.")

        hidden_dims = self.hidden_dims
        if hidden_dims is None:
            object.__setattr__(self, "hidden_dims", (self.hidden_dim,) * self.hidden_layers)
        else:
            if isinstance(hidden_dims, list):
                hidden_dims = tuple(hidden_dims)
                object.__setattr__(self, "hidden_dims", hidden_dims)
            if not isinstance(hidden_dims, tuple) or len(hidden_dims) == 0:
                raise ValueError("hidden_dims must be a non-empty sequence of positive integers.")
            for index, width in enumerate(hidden_dims):
                if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
                    raise ValueError(
                        f"hidden_dims[{index}] must be a positive integer, got {width!r}."
                    )
            object.__setattr__(self, "hidden_dim", hidden_dims[0])
            object.__setattr__(self, "hidden_layers", len(hidden_dims))

        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}.")

    def to_dict(self) -> dict[str, int | float | list[int]]:
        if self.hidden_dims is None:
            raise AssertionError("hidden_dims must be normalized.")
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "hidden_layers": self.hidden_layers,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BiddingMlpConfig:
        hidden_dims: tuple[int, ...] | None = None
        raw_hidden_dims = value.get("hidden_dims")
        if raw_hidden_dims is not None:
            if not isinstance(raw_hidden_dims, list | tuple):
                raise ValueError("model_config.hidden_dims must be a list of integers.")
            hidden_dims = tuple(
                _require_int_item(item, f"hidden_dims[{index}]")
                for index, item in enumerate(raw_hidden_dims)
            )
            if len(hidden_dims) == 0:
                raise ValueError(
                    "model_config.hidden_dims must be a non-empty list of integers."
                )
            hidden_dim = _optional_int(value, "hidden_dim", hidden_dims[0])
            hidden_layers = _optional_int(value, "hidden_layers", len(hidden_dims))
            if hidden_dim != hidden_dims[0]:
                raise ValueError(
                    "model_config.hidden_dim must match hidden_dims[0] "
                    f"when hidden_dims is set: {hidden_dim} != {hidden_dims[0]}."
                )
            if hidden_layers != len(hidden_dims):
                raise ValueError(
                    "model_config.hidden_layers must match len(hidden_dims) "
                    f"when hidden_dims is set: {hidden_layers} != {len(hidden_dims)}."
                )
        else:
            hidden_dim = _require_int(value, "hidden_dim")
            hidden_layers = _require_int(value, "hidden_layers")

        return cls(
            input_dim=_require_int(value, "input_dim"),
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            hidden_dims=hidden_dims,
            dropout=_require_float(value, "dropout"),
        )

    @property
    def hidden_widths(self) -> tuple[int, ...]:
        if self.hidden_dims is None:
            raise AssertionError("hidden_dims must be normalized.")
        return self.hidden_dims


class BiddingMlpModel(nn.Module):
    """Map one bidding-observation model_input to one logit per bidding action."""

    def __init__(self, config: BiddingMlpConfig) -> None:
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []
        input_dim = config.input_dim

        for hidden_dim in config.hidden_widths:
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
                f"model_input must have shape (batch, features), got {model_input.shape}."
            )

        if model_input.shape[1] != self.config.input_dim:
            raise ValueError(
                f"model_input feature count must be {self.config.input_dim}, "
                f"got {model_input.shape[1]}."
            )

        return cast(Tensor, self.network(model_input))


class BiddingActorCriticModel(nn.Module):
    """Separated bidding Actor-Critic model.

    The actor keeps the existing logits-only ``BiddingMlpModel`` contract so
    ONNX export can expose the same runtime artifact as supervised bidding
    policies. The critic is used only during PPO training/checkpointing.
    """

    def __init__(self, config: BiddingMlpConfig) -> None:
        super().__init__()
        self.config = config
        self.actor = BiddingMlpModel(config)
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


def create_seeded_bidding_model(config: BiddingMlpConfig, *, seed: int) -> BiddingMlpModel:
    """Create a model after setting PyTorch's CPU RNG seed."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an integer, got {type(seed).__name__}.")

    torch.manual_seed(seed)
    return BiddingMlpModel(config)


def create_seeded_bidding_actor_critic_model(
    config: BiddingMlpConfig, *, seed: int
) -> BiddingActorCriticModel:
    """Create a separated Actor-Critic model after setting PyTorch's CPU RNG seed."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an integer, got {type(seed).__name__}.")

    torch.manual_seed(seed)
    return BiddingActorCriticModel(config)


def _build_critic(config: BiddingMlpConfig) -> nn.Sequential:
    layers: list[nn.Module] = []
    input_dim = config.input_dim

    for hidden_dim in config.hidden_widths:
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        if config.dropout > 0.0:
            layers.append(nn.Dropout(config.dropout))
        input_dim = hidden_dim

    layers.append(nn.Linear(input_dim, 1))
    return nn.Sequential(*layers)


def _require_int(value: dict[str, Any], key: str) -> int:
    item = value.get(key)

    if isinstance(item, bool) or not isinstance(item, int):
        raise ValueError(f"model_config.{key} must be an integer.")

    return item


def _optional_int(value: dict[str, Any], key: str, default: int) -> int:
    item = value.get(key)

    if item is None:
        return default

    if isinstance(item, bool) or not isinstance(item, int):
        raise ValueError(f"model_config.{key} must be an integer.")

    return item


def _require_int_item(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"model_config.{key} must be an integer.")

    return value


def _require_float(value: dict[str, Any], key: str) -> float:
    item = value.get(key)

    if isinstance(item, bool) or not isinstance(item, int | float):
        raise ValueError(f"model_config.{key} must be a number.")

    return float(item)
