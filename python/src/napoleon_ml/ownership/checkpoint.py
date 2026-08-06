"""Checkpoint save/load with schema and card-id compatibility checks."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch

from napoleon_ml.dataset.constants import DATASET_SCHEMA_VERSION, PLAYING_ENCODER_SCHEMA_VERSION
from napoleon_ml.dataset.manifest import DatasetManifest
from napoleon_ml.dataset.tensors import MODEL_INPUT_SCHEMA_VERSION
from napoleon_ml.dataset.validation import calculate_card_ids_sha256

from .model import OwnershipMlpConfig, OwnershipMlpModel

CHECKPOINT_SCHEMA_VERSION = 1


class CheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint was saved for a different dataset/input schema."""


def save_ownership_checkpoint(
    path: Path | str,
    *,
    model: OwnershipMlpModel,
    training_config: dict[str, object],
    manifest: DatasetManifest,
) -> None:
    checkpoint = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state": model.state_dict(),
        "model_config": model.config.to_dict(),
        "training_config": dict(training_config),
        "dataset_schema_version": manifest.dataset_schema_version,
        "playing_encoder_schema_version": manifest.playing_encoder_schema_version,
        "model_input_schema_version": MODEL_INPUT_SCHEMA_VERSION,
        "card_ids_sha256": calculate_card_ids_sha256(),
    }
    torch.save(checkpoint, Path(path))


def load_ownership_checkpoint(
    path: Path | str,
    *,
    manifest: DatasetManifest,
) -> tuple[OwnershipMlpModel, dict[str, object]]:
    checkpoint_path = Path(path)

    try:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except OSError as error:
        raise CheckpointCompatibilityError(f"checkpoint cannot be read: {error}") from error
    except RuntimeError as error:
        raise CheckpointCompatibilityError(f"checkpoint cannot be loaded: {error}") from error
    except pickle.UnpicklingError as error:
        raise CheckpointCompatibilityError(f"checkpoint cannot be loaded: {error}") from error

    if not isinstance(raw, dict):
        raise CheckpointCompatibilityError("checkpoint must be a dictionary.")

    _validate_metadata(raw, manifest=manifest)

    model_config_raw = raw.get("model_config")
    if not isinstance(model_config_raw, dict):
        raise CheckpointCompatibilityError("checkpoint model_config must be a dictionary.")

    model_config = OwnershipMlpConfig.from_dict(model_config_raw)
    model = OwnershipMlpModel(model_config)

    model_state = raw.get("model_state")
    if not isinstance(model_state, dict):
        raise CheckpointCompatibilityError("checkpoint model_state must be a state dictionary.")

    try:
        model.load_state_dict(model_state)
    except RuntimeError as error:
        raise CheckpointCompatibilityError(
            f"checkpoint model_state is incompatible with model_config: {error}"
        ) from error

    return model, raw


def _validate_metadata(raw: dict[Any, Any], *, manifest: DatasetManifest) -> None:
    expected_values = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "playing_encoder_schema_version": PLAYING_ENCODER_SCHEMA_VERSION,
        "model_input_schema_version": MODEL_INPUT_SCHEMA_VERSION,
        "card_ids_sha256": calculate_card_ids_sha256(),
    }

    for key, expected in expected_values.items():
        actual = raw.get(key)
        if actual != expected:
            raise CheckpointCompatibilityError(
                f"checkpoint {key} mismatch: expected {expected!r}, got {actual!r}."
            )

    manifest_values = {
        "dataset_schema_version": manifest.dataset_schema_version,
        "playing_encoder_schema_version": manifest.playing_encoder_schema_version,
        "card_ids_sha256": manifest.card_ids_sha256,
    }

    for key, expected in manifest_values.items():
        actual = raw.get(key)
        if actual != expected:
            raise CheckpointCompatibilityError(
                f"checkpoint {key} does not match dataset: expected {expected!r}, got {actual!r}."
            )
