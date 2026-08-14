"""MLP baseline for exchange discard set prediction."""

from __future__ import annotations

from .checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    ExchangeCheckpointCompatibilityError,
    load_exchange_checkpoint,
    save_exchange_checkpoint,
)
from .metrics import (
    DISCARD_COUNT,
    ExchangeEvaluationReport,
    evaluate_exchange_model,
    mask_illegal_exchange_logits,
    masked_exchange_bce_loss,
    select_exchange_discards,
    selected_indices_to_mask,
)
from .model import (
    ExchangeActorCriticModel,
    ExchangeMlpConfig,
    ExchangeMlpModel,
    create_seeded_exchange_actor_critic_model,
    create_seeded_exchange_model,
)
from .ppo import (
    EXCHANGE_ACTOR_CRITIC_MODEL_ARCHITECTURE,
    EXCHANGE_DECISION_MODE,
    EXCHANGE_PPO_ALGORITHM,
    EXCHANGE_PPO_CHECKPOINT_SCHEMA_VERSION,
    ExchangePpoCompatibilityError,
    ExchangePpoTrainReport,
    ExchangePpoTrainSettings,
    load_exchange_logits_checkpoint,
    load_exchange_ppo_checkpoint,
    train_exchange_ppo,
)

__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "DISCARD_COUNT",
    "EXCHANGE_ACTOR_CRITIC_MODEL_ARCHITECTURE",
    "EXCHANGE_DECISION_MODE",
    "EXCHANGE_PPO_ALGORITHM",
    "EXCHANGE_PPO_CHECKPOINT_SCHEMA_VERSION",
    "ExchangeCheckpointCompatibilityError",
    "ExchangeEvaluationReport",
    "ExchangeActorCriticModel",
    "ExchangeMlpConfig",
    "ExchangeMlpModel",
    "ExchangePpoCompatibilityError",
    "ExchangePpoTrainReport",
    "ExchangePpoTrainSettings",
    "create_seeded_exchange_actor_critic_model",
    "create_seeded_exchange_model",
    "evaluate_exchange_model",
    "load_exchange_checkpoint",
    "load_exchange_logits_checkpoint",
    "load_exchange_ppo_checkpoint",
    "mask_illegal_exchange_logits",
    "masked_exchange_bce_loss",
    "save_exchange_checkpoint",
    "select_exchange_discards",
    "selected_indices_to_mask",
    "train_exchange_ppo",
]
