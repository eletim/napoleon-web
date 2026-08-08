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
from .model import ExchangeMlpConfig, ExchangeMlpModel, create_seeded_exchange_model

__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "DISCARD_COUNT",
    "ExchangeCheckpointCompatibilityError",
    "ExchangeEvaluationReport",
    "ExchangeMlpConfig",
    "ExchangeMlpModel",
    "create_seeded_exchange_model",
    "evaluate_exchange_model",
    "load_exchange_checkpoint",
    "mask_illegal_exchange_logits",
    "masked_exchange_bce_loss",
    "save_exchange_checkpoint",
    "select_exchange_discards",
    "selected_indices_to_mask",
]
