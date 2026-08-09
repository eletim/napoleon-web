"""Small CPU-friendly MLP for legal play policy logits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.input_dim, bool) or self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}.")

        if isinstance(self.hidden_dim, bool) or self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}.")

        if isinstance(self.hidden_layers, bool) or self.hidden_layers <= 0:
            raise ValueError(f"hidden_layers must be positive, got {self.hidden_layers}.")

        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}.")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PolicyMlpConfig:
        return cls(
            input_dim=_require_int(value, "input_dim"),
            hidden_dim=_require_int(value, "hidden_dim"),
            hidden_layers=_require_int(value, "hidden_layers"),
            dropout=_require_float(value, "dropout"),
        )


class PolicyMlpModel(nn.Module):
    """Map one playing-observation model_input to one logit per card."""

    def __init__(self, config: PolicyMlpConfig) -> None:
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


class PolicyActorCriticModel(nn.Module):
    """Shared playing trunk with policy logits and scalar state-value heads."""

    def __init__(self, config: PolicyMlpConfig) -> None:
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

    if isinstance(item, bool) or not isinstance(item, int):
        raise ValueError(f"model_config.{key} must be an integer.")

    return item


def _require_float(value: dict[str, Any], key: str) -> float:
    item = value.get(key)

    if isinstance(item, bool) or not isinstance(item, int | float):
        raise ValueError(f"model_config.{key} must be a number.")

    return float(item)
