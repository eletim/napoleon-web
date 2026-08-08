"""Checkpoint save/load with exchange schema and card-id compatibility checks."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch

from napoleon_ml.dataset.constants import (
    CARD_COUNT,
    EXCHANGE_DATASET_SAMPLE_TYPE,
    EXCHANGE_ENCODER_SCHEMA_VERSION,
    MULTIPHASE_DATASET_SCHEMA_VERSION,
)
from napoleon_ml.dataset.manifest import DatasetManifest
from napoleon_ml.dataset.tensors import EXCHANGE_MODEL_INPUT_SCHEMA_VERSION
from napoleon_ml.dataset.validation import calculate_card_ids_sha256

from .metrics import DISCARD_COUNT
from .model import ExchangeMlpConfig, ExchangeMlpModel

CHECKPOINT_SCHEMA_VERSION = 2


class ExchangeCheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint was saved for a different exchange schema."""


def save_exchange_checkpoint(
    path: Path | str,
    *,
    model: ExchangeMlpModel,
    training_config: dict[str, object],
    manifest: DatasetManifest,
    seed: int,
) -> None:
    checkpoint = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state": model.state_dict(),
        "model_config": model.config.to_dict(),
        "training_config": dict(training_config),
        "dataset_schema_version": manifest.dataset_schema_version,
        "sample_type": manifest.sample_type,
        "encoder_schema_version": manifest.encoder_schema_version,
        "exchange_model_input_schema_version": EXCHANGE_MODEL_INPUT_SCHEMA_VERSION,
        "action_count": CARD_COUNT,
        "card_ids_sha256": calculate_card_ids_sha256(),
        "discard_count": DISCARD_COUNT,
        "seed": seed,
    }
    torch.save(checkpoint, Path(path))


def load_exchange_checkpoint(
    path: Path | str,
    *,
    manifest: DatasetManifest,
) -> tuple[ExchangeMlpModel, dict[str, object]]:
    checkpoint_path = Path(path)

    try:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except OSError as error:
        raise ExchangeCheckpointCompatibilityError(
            f"checkpoint cannot be read: {error}"
        ) from error
    except RuntimeError as error:
        raise ExchangeCheckpointCompatibilityError(
            f"checkpoint cannot be loaded: {error}"
        ) from error
    except pickle.UnpicklingError as error:
        raise ExchangeCheckpointCompatibilityError(
            f"checkpoint cannot be loaded: {error}"
        ) from error

    if not isinstance(raw, dict):
        raise ExchangeCheckpointCompatibilityError("checkpoint must be a dictionary.")

    _validate_metadata(raw, manifest=manifest)

    model_config_raw = raw.get("model_config")
    if not isinstance(model_config_raw, dict):
        raise ExchangeCheckpointCompatibilityError("checkpoint model_config must be a dictionary.")

    model_config = ExchangeMlpConfig.from_dict(model_config_raw)
    model = ExchangeMlpModel(model_config)

    model_state = raw.get("model_state")
    if not isinstance(model_state, dict):
        raise ExchangeCheckpointCompatibilityError(
            "checkpoint model_state must be a state dictionary."
        )

    try:
        model.load_state_dict(model_state)
    except RuntimeError as error:
        raise ExchangeCheckpointCompatibilityError(
            f"checkpoint model_state is incompatible with model_config: {error}"
        ) from error

    return model, raw


def _validate_metadata(raw: dict[Any, Any], *, manifest: DatasetManifest) -> None:
    expected_values = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "dataset_schema_version": MULTIPHASE_DATASET_SCHEMA_VERSION,
        "sample_type": EXCHANGE_DATASET_SAMPLE_TYPE,
        "encoder_schema_version": EXCHANGE_ENCODER_SCHEMA_VERSION,
        "exchange_model_input_schema_version": EXCHANGE_MODEL_INPUT_SCHEMA_VERSION,
        "action_count": CARD_COUNT,
        "card_ids_sha256": calculate_card_ids_sha256(),
        "discard_count": DISCARD_COUNT,
    }

    for key, expected in expected_values.items():
        actual = raw.get(key)
        if actual != expected:
            raise ExchangeCheckpointCompatibilityError(
                f"checkpoint {key} mismatch: expected {expected!r}, got {actual!r}."
            )

    if isinstance(raw.get("seed"), bool) or not isinstance(raw.get("seed"), int):
        raise ExchangeCheckpointCompatibilityError("checkpoint seed must be an integer.")

    manifest_values = {
        "dataset_schema_version": manifest.dataset_schema_version,
        "sample_type": manifest.sample_type,
        "encoder_schema_version": manifest.encoder_schema_version,
        "card_ids_sha256": manifest.card_ids_sha256,
    }

    for key, expected in manifest_values.items():
        actual = raw.get(key)
        if actual != expected:
            raise ExchangeCheckpointCompatibilityError(
                f"checkpoint {key} does not match dataset: expected {expected!r}, got {actual!r}."
            )
