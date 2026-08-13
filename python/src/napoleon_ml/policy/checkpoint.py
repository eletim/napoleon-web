"""Checkpoint save/load with schema, card-id compatibility, and explicit migrations."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from napoleon_ml.dataset.constants import DATASET_SCHEMA_VERSION, PLAYING_ENCODER_SCHEMA_VERSION
from napoleon_ml.dataset.manifest import DatasetManifest
from napoleon_ml.dataset.tensors import MODEL_INPUT_FEATURE_COUNT, MODEL_INPUT_SCHEMA_VERSION
from napoleon_ml.dataset.validation import calculate_card_ids_sha256

from .model import (
    PolicyActorCriticModel,
    PolicyMlpConfig,
    PolicyMlpModel,
    PolicySeparatedActorCriticModel,
)

CHECKPOINT_SCHEMA_VERSION = 1
LEGACY_V1_MODEL_INPUT_FEATURE_COUNT = 6242
POLICY_MODEL_ARCHITECTURE = "policy-mlp-v1"
ACTOR_CRITIC_MODEL_ARCHITECTURE = "playing-actor-critic-v1"
SEPARATED_ACTOR_CRITIC_MODEL_ARCHITECTURE = "playing-separated-actor-critic-v1"
ACTOR_CRITIC_MODEL_ARCHITECTURES = (
    ACTOR_CRITIC_MODEL_ARCHITECTURE,
    SEPARATED_ACTOR_CRITIC_MODEL_ARCHITECTURE,
)


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
) -> tuple[
    PolicyMlpModel | PolicyActorCriticModel | PolicySeparatedActorCriticModel,
    dict[str, object],
]:
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
        model: PolicyMlpModel | PolicyActorCriticModel | PolicySeparatedActorCriticModel = (
            PolicyMlpModel(model_config)
        )
    elif architecture == ACTOR_CRITIC_MODEL_ARCHITECTURE:
        model = PolicyActorCriticModel(model_config)
    elif architecture == SEPARATED_ACTOR_CRITIC_MODEL_ARCHITECTURE:
        model = PolicySeparatedActorCriticModel(model_config)
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


def load_current_policy_logits_checkpoint(
    path: Path | str,
) -> tuple[
    PolicyMlpModel | PolicyActorCriticModel | PolicySeparatedActorCriticModel,
    dict[str, object],
]:
    """Load a current-schema playing checkpoint without requiring a dataset manifest."""

    checkpoint_path = Path(path)
    raw = _load_raw_checkpoint(checkpoint_path)
    _validate_current_checkpoint_metadata(raw)

    model_config_raw = raw.get("model_config")
    if not isinstance(model_config_raw, dict):
        raise PolicyCheckpointCompatibilityError("checkpoint model_config must be a dictionary.")

    model_config = PolicyMlpConfig.from_dict(model_config_raw)
    architecture = raw.get("model_architecture", POLICY_MODEL_ARCHITECTURE)
    if architecture == POLICY_MODEL_ARCHITECTURE:
        model: PolicyMlpModel | PolicyActorCriticModel | PolicySeparatedActorCriticModel = (
            PolicyMlpModel(model_config)
        )
    elif architecture == ACTOR_CRITIC_MODEL_ARCHITECTURE:
        model = PolicyActorCriticModel(model_config)
    elif architecture == SEPARATED_ACTOR_CRITIC_MODEL_ARCHITECTURE:
        model = PolicySeparatedActorCriticModel(model_config)
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


def migrate_policy_checkpoint_to_hidden_dims(
    source_path: Path | str,
    output_path: Path | str,
    *,
    target_hidden_dims: tuple[int, ...] = (512, 512, 256, 256),
    seed: int = 0,
) -> dict[str, object]:
    """Expand a current playing checkpoint to a wider/deeper ReLU MLP.

    Existing hidden activations are embedded in the leading channels of the
    target network. Extra hidden units are initialized normally but initially
    disconnected from the output head, so policy logits and Actor-Critic values
    are preserved while the new output-head columns can learn immediately.
    """

    source = Path(source_path)
    output = Path(output_path)
    if output.exists():
        raise PolicyCheckpointCompatibilityError(
            f"output checkpoint already exists: {output}."
        )
    raw = _load_raw_checkpoint(source)
    _validate_current_checkpoint_metadata(raw)

    model_config_raw = raw.get("model_config")
    if not isinstance(model_config_raw, dict):
        raise PolicyCheckpointCompatibilityError("checkpoint model_config must be a dictionary.")
    source_config = PolicyMlpConfig.from_dict(model_config_raw)
    target_config = PolicyMlpConfig(
        input_dim=source_config.input_dim,
        hidden_dims=target_hidden_dims,
        dropout=source_config.dropout,
    )
    _validate_architecture_migration(source_config, target_config)

    architecture = raw.get("model_architecture", POLICY_MODEL_ARCHITECTURE)
    migrated = dict(raw)
    migrated["model_config"] = target_config.to_dict()
    migrated["training_config"] = _migrated_training_config(
        raw.get("training_config"),
        target_config=target_config,
    )
    migrated["model_architecture"] = architecture
    migrated["architecture_migration_provenance"] = {
        "migration": "playing-mlp-hidden-dims-function-preserving-v1",
        "sourceCheckpointSha256": _sha256_file(source),
        "sourceModelArchitecture": architecture,
        "targetModelArchitecture": architecture,
        "sourceModelConfig": source_config.to_dict(),
        "targetModelConfig": target_config.to_dict(),
        "copyStrategy": "leading-channel-relu-identity-expansion",
        "policyLogitsPreserved": True,
        "valuePredictionPreserved": architecture in ACTOR_CRITIC_MODEL_ARCHITECTURES,
        "targetInitializationSeed": seed,
    }

    model_state = raw.get("model_state")
    if not isinstance(model_state, dict):
        raise PolicyCheckpointCompatibilityError(
            "checkpoint model_state must be a state dictionary."
        )

    if architecture == POLICY_MODEL_ARCHITECTURE:
        policy_source_model = PolicyMlpModel(source_config)
        _load_model_state(policy_source_model, cast(dict[str, torch.Tensor], model_state))
        torch.manual_seed(seed)
        policy_target_model = PolicyMlpModel(target_config)
        _copy_policy_function(policy_source_model, policy_target_model)
        migrated["model_state"] = policy_target_model.state_dict()
    elif architecture == ACTOR_CRITIC_MODEL_ARCHITECTURE:
        actor_critic_source_model = PolicyActorCriticModel(source_config)
        _load_model_state(
            actor_critic_source_model,
            cast(dict[str, torch.Tensor], model_state),
        )
        torch.manual_seed(seed)
        actor_critic_target_model = PolicyActorCriticModel(target_config)
        _copy_actor_critic_function(
            actor_critic_source_model,
            actor_critic_target_model,
        )
        migrated["model_state"] = actor_critic_target_model.state_dict()
    elif architecture == SEPARATED_ACTOR_CRITIC_MODEL_ARCHITECTURE:
        separated_source_model = PolicySeparatedActorCriticModel(source_config)
        _load_model_state(
            separated_source_model,
            cast(dict[str, torch.Tensor], model_state),
        )
        torch.manual_seed(seed)
        separated_target_model = PolicySeparatedActorCriticModel(target_config)
        _copy_separated_actor_critic_function(
            separated_source_model,
            separated_target_model,
        )
        migrated["model_state"] = separated_target_model.state_dict()
    else:
        raise PolicyCheckpointCompatibilityError(
            f"checkpoint model_architecture is unsupported: {architecture!r}."
        )

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


def _validate_current_checkpoint_metadata(raw: dict[Any, Any]) -> None:
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


def _migrated_training_config(
    value: object,
    *,
    target_config: PolicyMlpConfig,
) -> object:
    if not isinstance(value, dict):
        return value
    migrated = dict(value)
    migrated["hidden_dim"] = target_config.hidden_dim
    migrated["hidden_layers"] = target_config.hidden_layers
    migrated["hidden_dims"] = list(target_config.hidden_widths)
    return migrated


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


def _load_model_state(
    model: PolicyMlpModel | PolicyActorCriticModel | PolicySeparatedActorCriticModel,
    model_state: dict[str, torch.Tensor],
) -> None:
    try:
        model.load_state_dict(model_state)
    except RuntimeError as error:
        raise PolicyCheckpointCompatibilityError(
            f"checkpoint model_state is incompatible with model_config: {error}"
        ) from error


def _validate_architecture_migration(
    source_config: PolicyMlpConfig,
    target_config: PolicyMlpConfig,
) -> None:
    if source_config.input_dim != target_config.input_dim:
        raise PolicyCheckpointCompatibilityError("target input_dim must match source input_dim.")
    if source_config.dropout != target_config.dropout:
        raise PolicyCheckpointCompatibilityError("target dropout must match source dropout.")
    if len(target_config.hidden_widths) < len(source_config.hidden_widths):
        raise PolicyCheckpointCompatibilityError(
            "target hidden_dims must have at least as many layers as the source model."
        )

    preserved_width = source_config.hidden_widths[-1]
    for index, source_width in enumerate(source_config.hidden_widths):
        target_width = target_config.hidden_widths[index]
        if target_width < source_width:
            raise PolicyCheckpointCompatibilityError(
                "target hidden_dims must not shrink copied source layers: "
                f"layer {index} source={source_width}, target={target_width}."
            )
    for index, target_width in enumerate(
        target_config.hidden_widths[len(source_config.hidden_widths) :],
        start=len(source_config.hidden_widths),
    ):
        if target_width < preserved_width:
            raise PolicyCheckpointCompatibilityError(
                "inserted target hidden layers must preserve the source final width: "
                f"layer {index} target={target_width}, required>={preserved_width}."
            )


def _copy_policy_function(source: PolicyMlpModel, target: PolicyMlpModel) -> None:
    source_hidden, source_head = _policy_linear_layers(source)
    target_hidden, target_head = _policy_linear_layers(target)
    _copy_relu_mlp_function(
        source_hidden=source_hidden,
        source_heads=(source_head,),
        target_hidden=target_hidden,
        target_heads=(target_head,),
    )


def _copy_actor_critic_function(
    source: PolicyActorCriticModel,
    target: PolicyActorCriticModel,
) -> None:
    source_hidden = _linear_layers(source.trunk)
    target_hidden = _linear_layers(target.trunk)
    _copy_relu_mlp_function(
        source_hidden=source_hidden,
        source_heads=(source.policy_head, source.value_head),
        target_hidden=target_hidden,
        target_heads=(target.policy_head, target.value_head),
    )


def _copy_separated_actor_critic_function(
    source: PolicySeparatedActorCriticModel,
    target: PolicySeparatedActorCriticModel,
) -> None:
    _copy_policy_function(source.actor, target.actor)
    source_hidden, source_head = _policy_linear_layers_from_sequence(source.critic.network)
    target_hidden, target_head = _policy_linear_layers_from_sequence(target.critic.network)
    _copy_relu_mlp_function(
        source_hidden=source_hidden,
        source_heads=(source_head,),
        target_hidden=target_hidden,
        target_heads=(target_head,),
    )


def _copy_relu_mlp_function(
    *,
    source_hidden: list[nn.Linear],
    source_heads: tuple[nn.Linear, ...],
    target_hidden: list[nn.Linear],
    target_heads: tuple[nn.Linear, ...],
) -> None:
    if len(target_hidden) < len(source_hidden):
        raise PolicyCheckpointCompatibilityError(
            "target hidden layer count must be at least source hidden layer count."
        )
    if len(source_heads) != len(target_heads):
        raise AssertionError("source and target head counts must match.")

    previous_preserved_width = source_hidden[0].in_features
    for index, source_layer in enumerate(source_hidden):
        target_layer = target_hidden[index]
        _zero_leading_rows(target_layer, source_layer.out_features)
        with torch.no_grad():
            target_layer.weight[
                : source_layer.out_features,
                : previous_preserved_width,
            ].copy_(source_layer.weight)
            target_layer.bias[: source_layer.out_features].copy_(source_layer.bias)
        previous_preserved_width = source_layer.out_features

    preserved_width = source_hidden[-1].out_features
    for target_layer in target_hidden[len(source_hidden) :]:
        _zero_leading_rows(target_layer, preserved_width)
        with torch.no_grad():
            diagonal = torch.arange(preserved_width)
            target_layer.weight[diagonal, diagonal] = 1.0

    for source_head, target_head in zip(source_heads, target_heads, strict=True):
        if source_head.out_features != target_head.out_features:
            raise PolicyCheckpointCompatibilityError(
                "target output head shape does not match source output head."
            )
        with torch.no_grad():
            target_head.weight.zero_()
            target_head.bias.copy_(source_head.bias)
            target_head.weight[:, :preserved_width].copy_(source_head.weight)


def _zero_leading_rows(layer: nn.Linear, row_count: int) -> None:
    if layer.out_features < row_count:
        raise PolicyCheckpointCompatibilityError(
            f"target layer width {layer.out_features} cannot preserve {row_count} channels."
        )
    with torch.no_grad():
        layer.weight[:row_count].zero_()
        layer.bias[:row_count].zero_()


def _policy_linear_layers(model: PolicyMlpModel) -> tuple[list[nn.Linear], nn.Linear]:
    return _policy_linear_layers_from_sequence(model.network)


def _policy_linear_layers_from_sequence(
    sequence: nn.Sequential,
) -> tuple[list[nn.Linear], nn.Linear]:
    layers = _linear_layers(sequence)
    if len(layers) < 2:
        raise PolicyCheckpointCompatibilityError(
            "MLP model must contain hidden and head layers."
        )
    return layers[:-1], layers[-1]


def _linear_layers(module: nn.Module) -> list[nn.Linear]:
    return [child for child in module.children() if isinstance(child, nn.Linear)]


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
