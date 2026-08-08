"""MLP baseline for supervised adjutant-card prediction."""

from __future__ import annotations

from .checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    AdjutantCheckpointCompatibilityError,
    load_adjutant_checkpoint,
    save_adjutant_checkpoint,
)
from .metrics import (
    ADJUTANT_CATEGORY_LABELS,
    AdjutantEvaluationReport,
    AdjutantMetric,
    classify_adjutant_targets,
    evaluate_adjutant_model,
    mask_illegal_adjutant_logits,
    masked_adjutant_accuracy,
    masked_adjutant_cross_entropy,
    select_adjutant_action,
)
from .model import AdjutantMlpConfig, AdjutantMlpModel, create_seeded_adjutant_model

__all__ = [
    "ADJUTANT_CATEGORY_LABELS",
    "CHECKPOINT_SCHEMA_VERSION",
    "AdjutantCheckpointCompatibilityError",
    "AdjutantEvaluationReport",
    "AdjutantMetric",
    "AdjutantMlpConfig",
    "AdjutantMlpModel",
    "classify_adjutant_targets",
    "create_seeded_adjutant_model",
    "evaluate_adjutant_model",
    "load_adjutant_checkpoint",
    "mask_illegal_adjutant_logits",
    "masked_adjutant_accuracy",
    "masked_adjutant_cross_entropy",
    "save_adjutant_checkpoint",
    "select_adjutant_action",
]
