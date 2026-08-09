"""Checkpoint save/load with schema, card-id compatibility, and explicit migrations."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, cast

import torch

from napoleon_ml.dataset.constants import DATASET_SCHEMA_VERSION, PLAYING_ENCODER_SCHEMA_VERSION
from napoleon_ml.dataset.manifest import DatasetManifest
from napoleon_ml.dataset.tensors import MODEL_INPUT_FEATURE_COUNT, MODEL_INPUT_SCHEMA_VERSION
from napoleon_ml.dataset.validation import calculate_card_ids_sha256

from .model import PolicyActorCriticModel, PolicyMlpConfig, PolicyMlpModel

CHECKPOINT_SCHEMA_VERSION = 1
LEGACY_V1_MODEL_INPUT_FEATURE_COUNT = 6242
POLICY_MODEL_ARCHITECTURE = "policy-mlp-v1"
ACTOR_CRITIC_MODEL_ARCHITECTURE = "playing-actor-critic-v1"


class PolicyCheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint was saved for a different dataset/input schema."""


def save_policy_checkpoint(
    path: Path | str,
    *,
    model: PolicyMlpModel,
    training_config: dict[str, object],
    manifest: DatasetManifest,
    extra_metadata: dict[str, object] | None = None,
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
    if extra_metadata is not None:
        checkpoint.update(extra_metadata)
    torch.save(checkpoint, Path(path))


def load_policy_checkpoint(
    path: Path | str,
    *,
    manifest: DatasetManifest,
) -> tuple[PolicyMlpModel, dict[str, object]]:
    checkpoint_path = Path(path)

    try:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except OSError as error:
        raise PolicyCheckpointCompatibilityError(f"checkpoint cannot be read: {error}") from error
    except RuntimeError as error:
        raise PolicyCheckpointCompatibilityError(f"checkpoint cannot be loaded: {error}") from error
    except pickle.UnpicklingError as error:
        raise PolicyCheckpointCompatibilityError(f"checkpoint cannot be loaded: {error}") from error

    if not isinstance(raw, dict):
        raise PolicyCheckpointCompatibilityError("checkpoint must be a dictionary.")

    _validate_metadata(raw, manifest=manifest)

    model_config_raw = raw.get("model_config")
    if not isinstance(model_config_raw, dict):
        raise PolicyCheckpointCompatibilityError("checkpoint model_config must be a dictionary.")

    model_config = PolicyMlpConfig.from_dict(model_config_raw)
    model = PolicyMlpModel(model_config)

    model_state = raw.get("model_state")
    if not isinstance(model_state, dict):
        raise PolicyCheckpointCompatibilityError(
            "checkpoint model_state must be a state dictionary."
        )

    try:
        model.load_state_dict(model_state)
    except RuntimeError as error:
        raise PolicyCheckpointCompatibilityError(
            f"checkpoint model_state is incompatible with model_config: {error}"
        ) from error

    return model, raw


def load_policy_logits_checkpoint(
    path: Path | str,
    *,
    manifest: DatasetManifest,
) -> tuple[PolicyMlpModel | PolicyActorCriticModel, dict[str, object]]:
    """Load any checkpoint that can emit playing policy logits."""

    checkpoint_path = Path(path)
    raw = _load_raw_checkpoint(checkpoint_path)
    _validate_metadata(raw, manifest=manifest)

    model_config_raw = raw.get("model_config")
    if not isinstance(model_config_raw, dict):
        raise PolicyCheckpointCompatibilityError("checkpoint model_config must be a dictionary.")

    model_config = PolicyMlpConfig.from_dict(model_config_raw)
    architecture = raw.get("model_architecture", POLICY_MODEL_ARCHITECTURE)
    if architecture == POLICY_MODEL_ARCHITECTURE:
        model: PolicyMlpModel | PolicyActorCriticModel = PolicyMlpModel(model_config)
    elif architecture == ACTOR_CRITIC_MODEL_ARCHITECTURE:
        model = PolicyActorCriticModel(model_config)
    else:
        raise PolicyCheckpointCompatibilityError(
            f"checkpoint model_architecture is unsupported: {architecture!r}."
        )

    model_state = raw.get("model_state")
    if not isinstance(model_state, dict):
        raise PolicyCheckpointCompatibilityError(
            "checkpoint model_state must be a state dictionary."
        )

    try:
        model.load_state_dict(model_state)
    except RuntimeError as error:
        raise PolicyCheckpointCompatibilityError(
            f"checkpoint model_state is incompatible with model_config: {error}"
        ) from error

    return model, cast(dict[str, object], raw)


def migrate_policy_checkpoint_v1_to_v2(
    source_path: Path | str,
    output_path: Path | str,
) -> dict[str, object]:
    """Create a schema-v2 checkpoint from a schema-v1 playing checkpoint.

    The old first Linear input columns are copied unchanged. The four new
    self-role input columns, appended at the end of model_input v2, are zeroed
    so actor logits are exactly preserved immediately after migration.
    """

    source = Path(source_path)
    output = Path(output_path)
    raw = _load_raw_checkpoint(source)
    _validate_legacy_v1_checkpoint(raw)

    model_config_raw = raw.get("model_config")
    if not isinstance(model_config_raw, dict):
        raise PolicyCheckpointCompatibilityError("checkpoint model_config must be a dictionary.")

    legacy_config = PolicyMlpConfig.from_dict(model_config_raw)
    if legacy_config.input_dim != LEGACY_V1_MODEL_INPUT_FEATURE_COUNT:
        raise PolicyCheckpointCompatibilityError(
            "legacy checkpoint model_config.input_dim mismatch: "
            f"expected {LEGACY_V1_MODEL_INPUT_FEATURE_COUNT}, got {legacy_config.input_dim}."
        )

    migrated_config = PolicyMlpConfig(
        input_dim=MODEL_INPUT_FEATURE_COUNT,
        hidden_dim=legacy_config.hidden_dim,
        hidden_layers=legacy_config.hidden_layers,
        dropout=legacy_config.dropout,
    )
    migrated = dict(raw)
    migrated["model_config"] = migrated_config.to_dict()
    migrated["playing_encoder_schema_version"] = PLAYING_ENCODER_SCHEMA_VERSION
    migrated["model_input_schema_version"] = MODEL_INPUT_SCHEMA_VERSION
    migrated["migration_provenance"] = {
        "migration": "playing-input-v1-to-v2-self-role-one-hot",
        "sourceCheckpointSha256": _sha256_file(source),
        "sourceCheckpointSchemaVersion": CHECKPOINT_SCHEMA_VERSION,
        "sourcePlayingEncoderSchemaVersion": 1,
        "sourceModelInputSchemaVersion": 1,
        "sourceModelInputFeatureCount": LEGACY_V1_MODEL_INPUT_FEATURE_COUNT,
        "targetPlayingEncoderSchemaVersion": PLAYING_ENCODER_SCHEMA_VERSION,
        "targetModelInputSchemaVersion": MODEL_INPUT_SCHEMA_VERSION,
        "targetModelInputFeatureCount": MODEL_INPUT_FEATURE_COUNT,
        "newFeatureInitialization": "first-linear-self-role-columns-zero",
    }

    model_state = raw.get("model_state")
    if not isinstance(model_state, dict):
        raise PolicyCheckpointCompatibilityError(
            "checkpoint model_state must be a state dictionary."
        )

    migrated["model_state"] = _migrate_model_state(cast(dict[str, torch.Tensor], model_state))
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated, output)
    return cast(dict[str, object], migrated)


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
            raise PolicyCheckpointCompatibilityError(
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
            raise PolicyCheckpointCompatibilityError(
                f"checkpoint {key} does not match dataset: expected {expected!r}, got {actual!r}."
            )


def _load_raw_checkpoint(path: Path) -> dict[Any, Any]:
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except OSError as error:
        raise PolicyCheckpointCompatibilityError(f"checkpoint cannot be read: {error}") from error
    except RuntimeError as error:
        raise PolicyCheckpointCompatibilityError(f"checkpoint cannot be loaded: {error}") from error
    except pickle.UnpicklingError as error:
        raise PolicyCheckpointCompatibilityError(f"checkpoint cannot be loaded: {error}") from error

    if not isinstance(raw, dict):
        raise PolicyCheckpointCompatibilityError("checkpoint must be a dictionary.")

    return raw


def _validate_legacy_v1_checkpoint(raw: dict[Any, Any]) -> None:
    expected_values = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "playing_encoder_schema_version": 1,
        "model_input_schema_version": 1,
        "card_ids_sha256": calculate_card_ids_sha256(),
    }

    for key, expected in expected_values.items():
        actual = raw.get(key)
        if actual != expected:
            raise PolicyCheckpointCompatibilityError(
                f"legacy checkpoint {key} mismatch: expected {expected!r}, got {actual!r}."
            )


def _migrate_model_state(model_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    first_weight_key = "network.0.weight"
    first_weight = model_state.get(first_weight_key)
    if not isinstance(first_weight, torch.Tensor):
        raise PolicyCheckpointCompatibilityError(
            "checkpoint model_state is missing network.0.weight."
        )
    if first_weight.ndim != 2 or first_weight.shape[1] != LEGACY_V1_MODEL_INPUT_FEATURE_COUNT:
        raise PolicyCheckpointCompatibilityError(
            "legacy checkpoint network.0.weight shape mismatch: "
            f"expected (*, {LEGACY_V1_MODEL_INPUT_FEATURE_COUNT}), got {tuple(first_weight.shape)}."
        )

    migrated = dict(model_state)
    new_weight = torch.zeros(
        (first_weight.shape[0], MODEL_INPUT_FEATURE_COUNT),
        dtype=first_weight.dtype,
        device=first_weight.device,
    )
    new_weight[:, :LEGACY_V1_MODEL_INPUT_FEATURE_COUNT] = first_weight
    migrated[first_weight_key] = new_weight
    return migrated


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
