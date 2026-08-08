"""Bidding checkpoint save/load with schema and card-id compatibility checks."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch

from napoleon_ml.dataset.constants import (
    BIDDING_ACTION_COUNT,
    BIDDING_DATASET_SAMPLE_TYPE,
    BIDDING_ENCODER_SCHEMA_VERSION,
    MULTIPHASE_DATASET_SCHEMA_VERSION,
)
from napoleon_ml.dataset.manifest import DatasetManifest
from napoleon_ml.dataset.tensors import BIDDING_MODEL_INPUT_SCHEMA_VERSION
from napoleon_ml.dataset.validation import calculate_card_ids_sha256

from .model import BiddingMlpConfig, BiddingMlpModel

CHECKPOINT_SCHEMA_VERSION = 1


class BiddingCheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint was saved for a different bidding schema."""


def save_bidding_checkpoint(
    path: Path | str,
    *,
    model: BiddingMlpModel,
    training_config: dict[str, object],
    manifest: DatasetManifest,
    seed: int,
) -> None:
    _validate_manifest_compatible(manifest)
    checkpoint = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state": model.state_dict(),
        "model_config": model.config.to_dict(),
        "training_config": dict(training_config),
        "dataset_schema_version": manifest.dataset_schema_version,
        "sample_type": manifest.sample_type,
        "bidding_encoder_schema_version": manifest.encoder_schema_version,
        "model_input_schema_version": BIDDING_MODEL_INPUT_SCHEMA_VERSION,
        "action_count": BIDDING_ACTION_COUNT,
        "card_ids_sha256": calculate_card_ids_sha256(),
        "seed": seed,
    }
    torch.save(checkpoint, Path(path))


def load_bidding_checkpoint(
    path: Path | str,
    *,
    manifest: DatasetManifest,
) -> tuple[BiddingMlpModel, dict[str, object]]:
    checkpoint_path = Path(path)

    try:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except OSError as error:
        raise BiddingCheckpointCompatibilityError(
            f"checkpoint cannot be read: {error}"
        ) from error
    except RuntimeError as error:
        raise BiddingCheckpointCompatibilityError(
            f"checkpoint cannot be loaded: {error}"
        ) from error
    except pickle.UnpicklingError as error:
        raise BiddingCheckpointCompatibilityError(
            f"checkpoint cannot be loaded: {error}"
        ) from error

    if not isinstance(raw, dict):
        raise BiddingCheckpointCompatibilityError("checkpoint must be a dictionary.")

    _validate_metadata(raw, manifest=manifest)

    model_config_raw = raw.get("model_config")
    if not isinstance(model_config_raw, dict):
        raise BiddingCheckpointCompatibilityError("checkpoint model_config must be a dictionary.")

    try:
        model_config = BiddingMlpConfig.from_dict(model_config_raw)
    except ValueError as error:
        raise BiddingCheckpointCompatibilityError(
            f"checkpoint model_config is invalid: {error}"
        ) from error

    model = BiddingMlpModel(model_config)

    model_state = raw.get("model_state")
    if not isinstance(model_state, dict):
        raise BiddingCheckpointCompatibilityError(
            "checkpoint model_state must be a state dictionary."
        )

    try:
        model.load_state_dict(model_state)
    except RuntimeError as error:
        raise BiddingCheckpointCompatibilityError(
            f"checkpoint model_state is incompatible with model_config: {error}"
        ) from error

    return model, raw


def _validate_metadata(raw: dict[Any, Any], *, manifest: DatasetManifest) -> None:
    _validate_manifest_compatible(manifest)
    expected_values = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "dataset_schema_version": MULTIPHASE_DATASET_SCHEMA_VERSION,
        "sample_type": BIDDING_DATASET_SAMPLE_TYPE,
        "bidding_encoder_schema_version": BIDDING_ENCODER_SCHEMA_VERSION,
        "model_input_schema_version": BIDDING_MODEL_INPUT_SCHEMA_VERSION,
        "action_count": BIDDING_ACTION_COUNT,
        "card_ids_sha256": calculate_card_ids_sha256(),
    }

    for key, expected in expected_values.items():
        actual = raw.get(key)
        if actual != expected:
            raise BiddingCheckpointCompatibilityError(
                f"checkpoint {key} mismatch: expected {expected!r}, got {actual!r}."
            )

    seed = raw.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise BiddingCheckpointCompatibilityError("checkpoint seed must be an integer.")

    manifest_values = {
        "dataset_schema_version": manifest.dataset_schema_version,
        "sample_type": manifest.sample_type,
        "bidding_encoder_schema_version": manifest.encoder_schema_version,
        "card_ids_sha256": manifest.card_ids_sha256,
    }

    for key, expected in manifest_values.items():
        actual = raw.get(key)
        if actual != expected:
            raise BiddingCheckpointCompatibilityError(
                f"checkpoint {key} does not match dataset: expected {expected!r}, got {actual!r}."
            )


def _validate_manifest_compatible(manifest: DatasetManifest) -> None:
    if manifest.dataset_schema_version != MULTIPHASE_DATASET_SCHEMA_VERSION:
        raise BiddingCheckpointCompatibilityError(
            "bidding checkpoints require a v2 bidding dataset manifest."
        )

    if manifest.sample_type != BIDDING_DATASET_SAMPLE_TYPE:
        raise BiddingCheckpointCompatibilityError(
            f"bidding checkpoints require sampleType {BIDDING_DATASET_SAMPLE_TYPE!r}, "
            f"got {manifest.sample_type!r}."
        )

    if manifest.encoder_schema_version != BIDDING_ENCODER_SCHEMA_VERSION:
        raise BiddingCheckpointCompatibilityError(
            "manifest encoderSchemaVersion mismatch: "
            f"expected {BIDDING_ENCODER_SCHEMA_VERSION}, got {manifest.encoder_schema_version}."
        )
