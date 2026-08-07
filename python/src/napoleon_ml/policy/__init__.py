"""MLP baseline for supervised play policy prediction."""

from __future__ import annotations

from .checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    PolicyCheckpointCompatibilityError,
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from .metrics import (
    PolicyEvaluationReport,
    PolicyMetric,
    evaluate_policy_model,
    mask_illegal_policy_logits,
    masked_policy_accuracy,
    masked_policy_cross_entropy,
    select_policy_action,
)
from .model import PolicyMlpConfig, PolicyMlpModel, create_seeded_policy_model
from .onnx_export import (
    ONNX_INPUT_NAME,
    ONNX_OUTPUT_NAME,
    POLICY_ONNX_METADATA_SCHEMA_VERSION,
    PolicyOnnxExportReport,
    build_policy_onnx_metadata,
    export_policy_checkpoint_to_onnx,
    validate_policy_onnx_metadata,
)

__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "PolicyCheckpointCompatibilityError",
    "PolicyEvaluationReport",
    "PolicyMetric",
    "PolicyMlpConfig",
    "PolicyMlpModel",
    "PolicyOnnxExportReport",
    "ONNX_INPUT_NAME",
    "ONNX_OUTPUT_NAME",
    "POLICY_ONNX_METADATA_SCHEMA_VERSION",
    "build_policy_onnx_metadata",
    "create_seeded_policy_model",
    "evaluate_policy_model",
    "export_policy_checkpoint_to_onnx",
    "load_policy_checkpoint",
    "mask_illegal_policy_logits",
    "masked_policy_accuracy",
    "masked_policy_cross_entropy",
    "save_policy_checkpoint",
    "select_policy_action",
    "validate_policy_onnx_metadata",
]
