"""Adjutant MLP checkpoint save/load with schema compatibility checks."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch

from napoleon_ml.dataset.constants import (
    ADJUTANT_DATASET_SAMPLE_TYPE,
    ADJUTANT_ENCODER_SCHEMA_VERSION,
    CARD_COUNT,
    MULTIPHASE_DATASET_SCHEMA_VERSION,
)
from napoleon_ml.dataset.manifest import DatasetManifest
from napoleon_ml.dataset.tensors import ADJUTANT_MODEL_INPUT_SCHEMA_VERSION
from napoleon_ml.dataset.validation import calculate_card_ids_sha256

from .model import AdjutantMlpConfig, AdjutantMlpModel

CHECKPOINT_SCHEMA_VERSION = 1


class AdjutantCheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint was saved for a different adjutant schema."""


def save_adjutant_checkpoint(
    path: Path | str,
    *,
    model: AdjutantMlpModel,
    training_config: dict[str, object],
    manifest: DatasetManifest,
    seed: int,
) -> None:
    _validate_adjutant_manifest(manifest)
    checkpoint = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state": model.state_dict(),
        "model_config": model.config.to_dict(),
        "training_config": dict(training_config),
        "dataset_schema_version": manifest.dataset_schema_version,
        "sample_type": manifest.sample_type,
        "adjutant_encoder_schema_version": manifest.encoder_schema_version,
        "adjutant_model_input_schema_version": ADJUTANT_MODEL_INPUT_SCHEMA_VERSION,
        "action_count": CARD_COUNT,
        "card_ids_sha256": calculate_card_ids_sha256(),
        "seed": seed,
    }
    torch.save(checkpoint, Path(path))


def load_adjutant_checkpoint(
    path: Path | str,
    *,
    manifest: DatasetManifest,
) -> tuple[AdjutantMlpModel, dict[str, object]]:
    _validate_adjutant_manifest(manifest)
    checkpoint_path = Path(path)

    try:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except OSError as error:
        raise AdjutantCheckpointCompatibilityError(
            f"checkpoint cannot be read: {error}"
        ) from error
    except RuntimeError as error:
        raise AdjutantCheckpointCompatibilityError(
            f"checkpoint cannot be loaded: {error}"
        ) from error
    except pickle.UnpicklingError as error:
        raise AdjutantCheckpointCompatibilityError(
            f"checkpoint cannot be loaded: {error}"
        ) from error

    if not isinstance(raw, dict):
        raise AdjutantCheckpointCompatibilityError("checkpoint must be a dictionary.")

    _validate_metadata(raw, manifest=manifest)

    model_config_raw = raw.get("model_config")
    if not isinstance(model_config_raw, dict):
        raise AdjutantCheckpointCompatibilityError(
            "checkpoint model_config must be a dictionary."
        )

    model_config = AdjutantMlpConfig.from_dict(model_config_raw)
    model = AdjutantMlpModel(model_config)

    model_state = raw.get("model_state")
    if not isinstance(model_state, dict):
        raise AdjutantCheckpointCompatibilityError(
            "checkpoint model_state must be a state dictionary."
        )

    try:
        model.load_state_dict(model_state)
    except RuntimeError as error:
        raise AdjutantCheckpointCompatibilityError(
            f"checkpoint model_state is incompatible with model_config: {error}"
        ) from error

    return model, raw


def _validate_metadata(raw: dict[Any, Any], *, manifest: DatasetManifest) -> None:
    expected_values = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "dataset_schema_version": MULTIPHASE_DATASET_SCHEMA_VERSION,
        "sample_type": ADJUTANT_DATASET_SAMPLE_TYPE,
        "adjutant_encoder_schema_version": ADJUTANT_ENCODER_SCHEMA_VERSION,
        "adjutant_model_input_schema_version": ADJUTANT_MODEL_INPUT_SCHEMA_VERSION,
        "action_count": CARD_COUNT,
        "card_ids_sha256": calculate_card_ids_sha256(),
    }

    for key, expected in expected_values.items():
        actual = raw.get(key)
        if actual != expected:
            raise AdjutantCheckpointCompatibilityError(
                f"checkpoint {key} mismatch: expected {expected!r}, got {actual!r}."
            )

    seed = raw.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise AdjutantCheckpointCompatibilityError("checkpoint seed must be an integer.")

    training_config = raw.get("training_config")
    if not isinstance(training_config, dict):
        raise AdjutantCheckpointCompatibilityError(
            "checkpoint training_config must be a dictionary."
        )

    training_seed = training_config.get("seed")
    if training_seed != seed:
        raise AdjutantCheckpointCompatibilityError(
            "checkpoint seed does not match training_config.seed: "
            f"expected {seed!r}, got {training_seed!r}."
        )

    manifest_values = {
        "dataset_schema_version": manifest.dataset_schema_version,
        "sample_type": manifest.sample_type,
        "adjutant_encoder_schema_version": manifest.encoder_schema_version,
        "card_ids_sha256": manifest.card_ids_sha256,
    }

    for key, expected in manifest_values.items():
        actual = raw.get(key)
        if actual != expected:
            raise AdjutantCheckpointCompatibilityError(
                f"checkpoint {key} does not match dataset: "
                f"expected {expected!r}, got {actual!r}."
            )


def _validate_adjutant_manifest(manifest: DatasetManifest) -> None:
    if manifest.dataset_schema_version != MULTIPHASE_DATASET_SCHEMA_VERSION:
        raise AdjutantCheckpointCompatibilityError(
            "adjutant checkpoints require datasetSchemaVersion "
            f"{MULTIPHASE_DATASET_SCHEMA_VERSION}, got {manifest.dataset_schema_version}."
        )

    if manifest.sample_type != ADJUTANT_DATASET_SAMPLE_TYPE:
        raise AdjutantCheckpointCompatibilityError(
            "adjutant checkpoints require sampleType "
            f"{ADJUTANT_DATASET_SAMPLE_TYPE}, got {manifest.sample_type}."
        )

    if manifest.encoder_schema_version != ADJUTANT_ENCODER_SCHEMA_VERSION:
        raise AdjutantCheckpointCompatibilityError(
            "adjutant checkpoints require encoderSchemaVersion "
            f"{ADJUTANT_ENCODER_SCHEMA_VERSION}, got {manifest.encoder_schema_version}."
        )
