"""Small CPU-friendly MLP for legal play policy logits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn

from napoleon_ml.dataset.constants import CARD_COUNT
from napoleon_ml.dataset.tensors import MODEL_INPUT_FEATURE_COUNT


@dataclass(frozen=True)
class PolicyMlpConfig:
    input_dim: int = MODEL_INPUT_FEATURE_COUNT
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
    def from_dict(cls, value: dict[str, Any]) -> PolicyMlpConfig:
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
            input_dim=_optional_int(value, "input_dim", MODEL_INPUT_FEATURE_COUNT),
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            hidden_dims=hidden_dims,
            dropout=_optional_float(value, "dropout", 0.0),
        )

    @property
    def hidden_widths(self) -> tuple[int, ...]:
        if self.hidden_dims is None:
            raise AssertionError("hidden_dims must be normalized.")
        return self.hidden_dims


class PolicyMlpModel(nn.Module):
    """Map one playing-observation model_input to one logit per card."""

    def __init__(self, config: PolicyMlpConfig) -> None:
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


class PolicyActorCriticModel(nn.Module):
    """Shared playing trunk with policy logits and scalar state-value heads."""

    def __init__(self, config: PolicyMlpConfig) -> None:
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

        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(input_dim, CARD_COUNT)
        self.value_head = nn.Linear(input_dim, 1)

    def forward(self, model_input: Tensor) -> Tensor:
        """Return policy logits, preserving the existing ONNX/runtime contract."""

        logits, _ = self.forward_actor_critic(model_input)
        return logits

    def forward_actor_critic(self, model_input: Tensor) -> tuple[Tensor, Tensor]:
        if model_input.ndim != 2:
            raise ValueError(
                f"model_input must have shape (batch, features), got {model_input.shape}."
            )

        if model_input.shape[1] != self.config.input_dim:
            raise ValueError(
                f"model_input feature count must be {self.config.input_dim}, "
                f"got {model_input.shape[1]}."
            )

        hidden = self.trunk(model_input)
        logits = self.policy_head(hidden)
        value = self.value_head(hidden).squeeze(1)
        return cast(Tensor, logits), cast(Tensor, value)


def create_seeded_policy_model(config: PolicyMlpConfig, *, seed: int) -> PolicyMlpModel:
    """Create a model after setting PyTorch's CPU RNG seed."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an integer, got {type(seed).__name__}.")

    torch.manual_seed(seed)
    return PolicyMlpModel(config)


def create_actor_critic_from_policy_model(
    policy_model: PolicyMlpModel,
    *,
    seed: int | None = None,
) -> PolicyActorCriticModel:
    """Copy an existing policy MLP into an Actor-Critic model.

    The hidden trunk and final policy head are copied exactly, so policy logits
    match the source model immediately after migration. The value head is newly
    initialized.
    """

    if seed is not None:
        torch.manual_seed(seed)
    actor_critic = PolicyActorCriticModel(policy_model.config)
    source_layers = list(policy_model.network.children())
    if not isinstance(source_layers[-1], nn.Linear):
        raise ValueError("policy model final layer must be Linear.")

    trunk_layers = list(actor_critic.trunk.children())
    if len(source_layers[:-1]) != len(trunk_layers):
        raise ValueError("policy model trunk layer count mismatch.")

    for source, target in zip(source_layers[:-1], trunk_layers, strict=True):
        if isinstance(source, nn.Linear) and isinstance(target, nn.Linear):
            target.load_state_dict(source.state_dict())

    actor_critic.policy_head.load_state_dict(source_layers[-1].state_dict())
    return actor_critic


def create_seeded_actor_critic_model(
    config: PolicyMlpConfig,
    *,
    seed: int,
) -> PolicyActorCriticModel:
    """Create an Actor-Critic model after setting PyTorch's CPU RNG seed."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an integer, got {type(seed).__name__}.")

    torch.manual_seed(seed)
    return PolicyActorCriticModel(config)


def _require_int(value: dict[str, Any], key: str) -> int:
    item = value.get(key)

    return _require_int_item(item, key)


def _optional_int(value: dict[str, Any], key: str, default: int) -> int:
    if key not in value:
        return default
    return _require_int(value, key)


def _require_int_item(item: object, key: str) -> int:
    if isinstance(item, bool) or not isinstance(item, int):
        raise ValueError(f"model_config.{key} must be an integer.")

    return item


def _require_float(value: dict[str, Any], key: str) -> float:
    item = value.get(key)

    return _require_float_item(item, key)


def _optional_float(value: dict[str, Any], key: str, default: float) -> float:
    if key not in value:
        return default
    return _require_float(value, key)


def _require_float_item(item: object, key: str) -> float:
    if isinstance(item, bool) or not isinstance(item, int | float):
        raise ValueError(f"model_config.{key} must be a number.")

    return float(item)
