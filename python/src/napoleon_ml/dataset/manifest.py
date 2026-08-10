"""Typed representation of ``manifest.json`` (TypeScript ``DatasetManifest``).

Mirrors ``packages/training-data/src/types.ts`` field-for-field. Parsing here
only enforces JSON *shape* (required keys present, no unknown keys, values
have the declared JSON type) with no coercion. Cross-field and numeric-range
validation lives in :mod:`napoleon_ml.dataset.validation`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._strict import (
    require_dict,
    require_exact_keys,
    require_int,
    require_list,
    require_number,
    require_str,
    require_str_tuple,
)
from .errors import ManifestValidationError

_SHARD_KEYS = frozenset(
    {"file", "startSeed", "endSeed", "gameCount", "sampleCount", "byteLength", "sha256"}
)
_AGENT_KEYS = frozenset({"type", "version"})
_BEHAVIOR_POLICY_KEYS = frozenset(
    {
        "type",
        "artifactId",
        "onnxFileName",
        "metadataFileName",
        "onnxSha256",
        "metadataSha256",
        "metadata",
    }
)
_INFERENCE_RUNTIME_KEYS = frozenset(
    {"requestedInferenceDevice", "resolvedInferenceDevice", "executionProvider"}
)
_BEHAVIOR_POLICY_KEYS_WITH_RUNTIME = _BEHAVIOR_POLICY_KEYS | _INFERENCE_RUNTIME_KEYS
_REWARD_KEYS = frozenset({"type", "version"})
_ROLLOUT_ROSTER_KEYS = frozenset({"assignment", "seats"})
_CURRENT_POLICY_ROSTER_SEAT_KEYS = frozenset({"source"})
_RULE_BASED_ROSTER_SEAT_KEYS = frozenset({"source", "version"})
_FROZEN_ONNX_ROSTER_SEAT_KEYS = frozenset(
    {
        "source",
        "artifactId",
        "onnxFileName",
        "metadataFileName",
        "onnxSha256",
        "metadataSha256",
        "metadata",
    }
)
_FROZEN_ONNX_ROSTER_SEAT_KEYS_WITH_RUNTIME = (
    _FROZEN_ONNX_ROSTER_SEAT_KEYS | _INFERENCE_RUNTIME_KEYS
)
_MANIFEST_BASE_KEYS = frozenset(
    {
        "datasetSchemaVersion",
        "generatorVersion",
        "format",
        "sampleType",
        "agent",
        "startSeed",
        "endSeed",
        "gameCount",
        "sampleCount",
        "gamesPerShard",
        "shardCount",
        "playerCount",
        "cardCount",
        "cardIds",
        "cardIdsSha256",
        "shards",
    }
)
_PLAYING_MANIFEST_KEYS = _MANIFEST_BASE_KEYS | frozenset({"playingEncoderSchemaVersion"})
_MULTIPHASE_MANIFEST_KEYS = _MANIFEST_BASE_KEYS | frozenset({"encoderSchemaVersion"})
_SELF_PLAY_MANIFEST_KEYS = (
    (_MANIFEST_BASE_KEYS - frozenset({"agent"}))
    | frozenset(
        {
            "sampleSchemaVersion",
            "playingEncoderSchemaVersion",
            "playingModelInputSchemaVersion",
            "behaviorPolicy",
            "samplingAlgorithm",
            "temperature",
            "reward",
            "nonPlayingAgent",
            "rolloutRoster",
        }
    )
)
_BINARY_SELF_PLAY_MANIFEST_KEYS = _SELF_PLAY_MANIFEST_KEYS | frozenset({"tensorSchema"})
_TENSOR_SCHEMA_KEYS = frozenset({"shardSchemaVersion", "byteOrder", "compression", "fields"})
_TENSOR_FIELD_SCHEMA_KEYS = frozenset({"name", "dtype", "shape"})


@dataclass(frozen=True)
class DatasetShardManifest:
    file: str
    start_seed: int
    end_seed: int
    game_count: int
    sample_count: int
    byte_length: int
    sha256: str


@dataclass(frozen=True)
class DatasetAgentInfo:
    type: str
    version: int


@dataclass(frozen=True)
class DatasetBehaviorPolicyInfo:
    type: str
    artifact_id: str
    onnx_file_name: str
    metadata_file_name: str
    onnx_sha256: str
    metadata_sha256: str
    metadata: object
    requested_inference_device: str | None = None
    resolved_inference_device: str | None = None
    execution_provider: str | None = None


@dataclass(frozen=True)
class DatasetRewardInfo:
    type: str
    version: int


@dataclass(frozen=True)
class DatasetRolloutRosterSeat:
    source: str
    version: int | None = None
    artifact_id: str | None = None
    onnx_file_name: str | None = None
    metadata_file_name: str | None = None
    onnx_sha256: str | None = None
    metadata_sha256: str | None = None
    metadata: object | None = None
    requested_inference_device: str | None = None
    resolved_inference_device: str | None = None
    execution_provider: str | None = None


@dataclass(frozen=True)
class DatasetRolloutRoster:
    assignment: str
    seats: tuple[DatasetRolloutRosterSeat, ...]


@dataclass(frozen=True)
class DatasetTensorFieldSchema:
    name: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class DatasetTensorSchema:
    shard_schema_version: int
    byte_order: str
    compression: str
    fields: tuple[DatasetTensorFieldSchema, ...]


@dataclass(frozen=True)
class DatasetManifest:
    dataset_schema_version: int
    generator_version: int
    playing_encoder_schema_version: int | None
    playing_model_input_schema_version: int | None
    encoder_schema_version: int | None
    format: str
    sample_type: str
    agent: DatasetAgentInfo | None
    start_seed: int
    end_seed: int
    game_count: int
    sample_count: int
    games_per_shard: int
    shard_count: int
    player_count: int
    card_count: int
    card_ids: tuple[str, ...]
    card_ids_sha256: str
    shards: tuple[DatasetShardManifest, ...]
    sample_schema_version: int | None = None
    behavior_policy: DatasetBehaviorPolicyInfo | None = None
    sampling_algorithm: str | None = None
    temperature: float | None = None
    reward: DatasetRewardInfo | None = None
    non_playing_agent: DatasetAgentInfo | None = None
    rollout_roster: DatasetRolloutRoster | None = None
    tensor_schema: DatasetTensorSchema | None = None


def _error(message: str) -> ManifestValidationError:
    return ManifestValidationError(message)


def _parse_shard(raw: object, *, index: int) -> DatasetShardManifest:
    path = f"manifest.shards[{index}]"
    obj = require_dict(raw, path=path, error=_error)
    require_exact_keys(obj, _SHARD_KEYS, path=path, error=_error)

    return DatasetShardManifest(
        file=require_str(obj["file"], path=f"{path}.file", error=_error),
        start_seed=require_int(obj["startSeed"], path=f"{path}.startSeed", error=_error),
        end_seed=require_int(obj["endSeed"], path=f"{path}.endSeed", error=_error),
        game_count=require_int(obj["gameCount"], path=f"{path}.gameCount", error=_error),
        sample_count=require_int(obj["sampleCount"], path=f"{path}.sampleCount", error=_error),
        byte_length=require_int(obj["byteLength"], path=f"{path}.byteLength", error=_error),
        sha256=require_str(obj["sha256"], path=f"{path}.sha256", error=_error),
    )


def _parse_agent(raw: object) -> DatasetAgentInfo:
    path = "manifest.agent"
    obj = require_dict(raw, path=path, error=_error)
    require_exact_keys(obj, _AGENT_KEYS, path=path, error=_error)

    return DatasetAgentInfo(
        type=require_str(obj["type"], path=f"{path}.type", error=_error),
        version=require_int(obj["version"], path=f"{path}.version", error=_error),
    )


def _parse_non_playing_agent(raw: object) -> DatasetAgentInfo:
    path = "manifest.nonPlayingAgent"
    obj = require_dict(raw, path=path, error=_error)
    require_exact_keys(obj, _AGENT_KEYS, path=path, error=_error)

    return DatasetAgentInfo(
        type=require_str(obj["type"], path=f"{path}.type", error=_error),
        version=require_int(obj["version"], path=f"{path}.version", error=_error),
    )


def _parse_behavior_policy(raw: object) -> DatasetBehaviorPolicyInfo:
    path = "manifest.behaviorPolicy"
    obj = require_dict(raw, path=path, error=_error)
    require_exact_keys(
        obj,
        _select_runtime_key_set(obj, _BEHAVIOR_POLICY_KEYS, _BEHAVIOR_POLICY_KEYS_WITH_RUNTIME),
        path=path,
        error=_error,
    )

    return DatasetBehaviorPolicyInfo(
        type=require_str(obj["type"], path=f"{path}.type", error=_error),
        artifact_id=require_str(obj["artifactId"], path=f"{path}.artifactId", error=_error),
        onnx_file_name=require_str(
            obj["onnxFileName"], path=f"{path}.onnxFileName", error=_error
        ),
        metadata_file_name=require_str(
            obj["metadataFileName"], path=f"{path}.metadataFileName", error=_error
        ),
        onnx_sha256=require_str(obj["onnxSha256"], path=f"{path}.onnxSha256", error=_error),
        metadata_sha256=require_str(
            obj["metadataSha256"], path=f"{path}.metadataSha256", error=_error
        ),
        metadata=obj["metadata"],
        requested_inference_device=_optional_str(
            obj, "requestedInferenceDevice", path=path
        ),
        resolved_inference_device=_optional_str(obj, "resolvedInferenceDevice", path=path),
        execution_provider=_optional_str(obj, "executionProvider", path=path),
    )


def _parse_reward(raw: object) -> DatasetRewardInfo:
    path = "manifest.reward"
    obj = require_dict(raw, path=path, error=_error)
    require_exact_keys(obj, _REWARD_KEYS, path=path, error=_error)

    return DatasetRewardInfo(
        type=require_str(obj["type"], path=f"{path}.type", error=_error),
        version=require_int(obj["version"], path=f"{path}.version", error=_error),
    )


def _parse_rollout_roster(raw: object) -> DatasetRolloutRoster:
    path = "manifest.rolloutRoster"
    obj = require_dict(raw, path=path, error=_error)
    require_exact_keys(obj, _ROLLOUT_ROSTER_KEYS, path=path, error=_error)
    seats = require_list(obj["seats"], path=f"{path}.seats", error=_error)

    return DatasetRolloutRoster(
        assignment=require_str(obj["assignment"], path=f"{path}.assignment", error=_error),
        seats=tuple(
            _parse_rollout_roster_seat(seat, index=index)
            for index, seat in enumerate(seats)
        ),
    )


def _parse_rollout_roster_seat(raw: object, *, index: int) -> DatasetRolloutRosterSeat:
    path = f"manifest.rolloutRoster.seats[{index}]"
    obj = require_dict(raw, path=path, error=_error)
    source = require_str(obj.get("source"), path=f"{path}.source", error=_error)

    if source == "current-policy":
        require_exact_keys(obj, _CURRENT_POLICY_ROSTER_SEAT_KEYS, path=path, error=_error)
        return DatasetRolloutRosterSeat(source=source)

    if source == "rule-based":
        require_exact_keys(obj, _RULE_BASED_ROSTER_SEAT_KEYS, path=path, error=_error)
        return DatasetRolloutRosterSeat(
            source=source,
            version=require_int(obj["version"], path=f"{path}.version", error=_error),
        )

    if source == "frozen-onnx":
        require_exact_keys(
            obj,
            _select_runtime_key_set(
                obj,
                _FROZEN_ONNX_ROSTER_SEAT_KEYS,
                _FROZEN_ONNX_ROSTER_SEAT_KEYS_WITH_RUNTIME,
            ),
            path=path,
            error=_error,
        )
        return DatasetRolloutRosterSeat(
            source=source,
            artifact_id=require_str(obj["artifactId"], path=f"{path}.artifactId", error=_error),
            onnx_file_name=require_str(
                obj["onnxFileName"], path=f"{path}.onnxFileName", error=_error
            ),
            metadata_file_name=require_str(
                obj["metadataFileName"], path=f"{path}.metadataFileName", error=_error
            ),
            onnx_sha256=require_str(obj["onnxSha256"], path=f"{path}.onnxSha256", error=_error),
            metadata_sha256=require_str(
                obj["metadataSha256"], path=f"{path}.metadataSha256", error=_error
            ),
            metadata=obj["metadata"],
            requested_inference_device=_optional_str(
                obj, "requestedInferenceDevice", path=path
            ),
            resolved_inference_device=_optional_str(obj, "resolvedInferenceDevice", path=path),
            execution_provider=_optional_str(obj, "executionProvider", path=path),
        )

    raise _error(f"{path}.source is invalid: {source!r}.")


def _parse_tensor_schema(raw: object) -> DatasetTensorSchema:
    path = "manifest.tensorSchema"
    obj = require_dict(raw, path=path, error=_error)
    require_exact_keys(obj, _TENSOR_SCHEMA_KEYS, path=path, error=_error)
    fields = require_list(obj["fields"], path=f"{path}.fields", error=_error)

    return DatasetTensorSchema(
        shard_schema_version=require_int(
            obj["shardSchemaVersion"], path=f"{path}.shardSchemaVersion", error=_error
        ),
        byte_order=require_str(obj["byteOrder"], path=f"{path}.byteOrder", error=_error),
        compression=require_str(obj["compression"], path=f"{path}.compression", error=_error),
        fields=tuple(_parse_tensor_field(field, index=index) for index, field in enumerate(fields)),
    )


def _parse_tensor_field(raw: object, *, index: int) -> DatasetTensorFieldSchema:
    path = f"manifest.tensorSchema.fields[{index}]"
    obj = require_dict(raw, path=path, error=_error)
    require_exact_keys(obj, _TENSOR_FIELD_SCHEMA_KEYS, path=path, error=_error)
    shape = require_list(obj["shape"], path=f"{path}.shape", error=_error)

    return DatasetTensorFieldSchema(
        name=require_str(obj["name"], path=f"{path}.name", error=_error),
        dtype=require_str(obj["dtype"], path=f"{path}.dtype", error=_error),
        shape=tuple(
            require_int(value, path=f"{path}.shape[{shape_index}]", error=_error)
            for shape_index, value in enumerate(shape)
        ),
    )


def _select_runtime_key_set(
    obj: dict[str, object],
    base_keys: frozenset[str],
    keys_with_runtime: frozenset[str],
) -> frozenset[str]:
    runtime_key_count = sum(1 for key in _INFERENCE_RUNTIME_KEYS if key in obj)
    if runtime_key_count == 0:
        return base_keys
    return keys_with_runtime


def _optional_str(obj: dict[str, object], key: str, *, path: str) -> str | None:
    if key not in obj:
        return None
    return require_str(obj[key], path=f"{path}.{key}", error=_error)


def parse_manifest(raw: object) -> DatasetManifest:
    """Parse an already JSON-decoded manifest value into a typed dataclass.

    Raises :class:`~napoleon_ml.dataset.errors.ManifestValidationError` on any
    structural mismatch: a missing key, an unknown key, or a value whose JSON
    type does not match the schema (e.g. a string where an integer is
    required). No semantic validation (ranges, hashes, shard ordering) is
    performed here; call :func:`napoleon_ml.dataset.validation.validate_manifest`
    on the result.
    """

    obj = require_dict(raw, path="manifest", error=_error)
    dataset_schema_version = require_int(
        obj.get("datasetSchemaVersion"),
        path="manifest.datasetSchemaVersion",
        error=_error,
    )

    if dataset_schema_version == 1:
        require_exact_keys(obj, _PLAYING_MANIFEST_KEYS, path="manifest", error=_error)
        playing_encoder_schema_version = require_int(
            obj["playingEncoderSchemaVersion"],
            path="manifest.playingEncoderSchemaVersion",
            error=_error,
        )
        playing_model_input_schema_version = None
        encoder_schema_version = None
        agent = _parse_agent(obj["agent"])
        sample_schema_version = None
        behavior_policy = None
        sampling_algorithm = None
        temperature = None
        reward = None
        non_playing_agent = None
        rollout_roster = None
        tensor_schema = None
    elif dataset_schema_version == 2:
        require_exact_keys(obj, _MULTIPHASE_MANIFEST_KEYS, path="manifest", error=_error)
        playing_encoder_schema_version = None
        playing_model_input_schema_version = None
        encoder_schema_version = require_int(
            obj["encoderSchemaVersion"], path="manifest.encoderSchemaVersion", error=_error
        )
        agent = _parse_agent(obj["agent"])
        sample_schema_version = None
        behavior_policy = None
        sampling_algorithm = None
        temperature = None
        reward = None
        non_playing_agent = None
        rollout_roster = None
        tensor_schema = None
    elif dataset_schema_version == 3:
        require_exact_keys(obj, _SELF_PLAY_MANIFEST_KEYS, path="manifest", error=_error)
        playing_encoder_schema_version = require_int(
            obj["playingEncoderSchemaVersion"],
            path="manifest.playingEncoderSchemaVersion",
            error=_error,
        )
        playing_model_input_schema_version = require_int(
            obj["playingModelInputSchemaVersion"],
            path="manifest.playingModelInputSchemaVersion",
            error=_error,
        )
        encoder_schema_version = None
        agent = None
        sample_schema_version = require_int(
            obj["sampleSchemaVersion"], path="manifest.sampleSchemaVersion", error=_error
        )
        behavior_policy = _parse_behavior_policy(obj["behaviorPolicy"])
        sampling_algorithm = require_str(
            obj["samplingAlgorithm"], path="manifest.samplingAlgorithm", error=_error
        )
        temperature = require_number(
            obj["temperature"], path="manifest.temperature", error=_error
        )
        reward = _parse_reward(obj["reward"])
        non_playing_agent = _parse_non_playing_agent(obj["nonPlayingAgent"])
        rollout_roster = _parse_rollout_roster(obj["rolloutRoster"])
        tensor_schema = None
    elif dataset_schema_version == 4:
        require_exact_keys(obj, _BINARY_SELF_PLAY_MANIFEST_KEYS, path="manifest", error=_error)
        playing_encoder_schema_version = require_int(
            obj["playingEncoderSchemaVersion"],
            path="manifest.playingEncoderSchemaVersion",
            error=_error,
        )
        playing_model_input_schema_version = require_int(
            obj["playingModelInputSchemaVersion"],
            path="manifest.playingModelInputSchemaVersion",
            error=_error,
        )
        encoder_schema_version = None
        agent = None
        sample_schema_version = require_int(
            obj["sampleSchemaVersion"], path="manifest.sampleSchemaVersion", error=_error
        )
        behavior_policy = _parse_behavior_policy(obj["behaviorPolicy"])
        sampling_algorithm = require_str(
            obj["samplingAlgorithm"], path="manifest.samplingAlgorithm", error=_error
        )
        temperature = require_number(
            obj["temperature"], path="manifest.temperature", error=_error
        )
        reward = _parse_reward(obj["reward"])
        non_playing_agent = _parse_non_playing_agent(obj["nonPlayingAgent"])
        rollout_roster = _parse_rollout_roster(obj["rolloutRoster"])
        tensor_schema = _parse_tensor_schema(obj["tensorSchema"])
    else:
        raise _error(
            "manifest.datasetSchemaVersion must be 1, 2, 3, or 4, "
            f"got {dataset_schema_version}."
        )

    shards_raw = require_list(obj["shards"], path="manifest.shards", error=_error)

    return DatasetManifest(
        dataset_schema_version=dataset_schema_version,
        generator_version=require_int(
            obj["generatorVersion"], path="manifest.generatorVersion", error=_error
        ),
        playing_encoder_schema_version=playing_encoder_schema_version,
        playing_model_input_schema_version=playing_model_input_schema_version,
        encoder_schema_version=encoder_schema_version,
        format=require_str(obj["format"], path="manifest.format", error=_error),
        sample_type=require_str(obj["sampleType"], path="manifest.sampleType", error=_error),
        agent=agent,
        start_seed=require_int(obj["startSeed"], path="manifest.startSeed", error=_error),
        end_seed=require_int(obj["endSeed"], path="manifest.endSeed", error=_error),
        game_count=require_int(obj["gameCount"], path="manifest.gameCount", error=_error),
        sample_count=require_int(obj["sampleCount"], path="manifest.sampleCount", error=_error),
        games_per_shard=require_int(
            obj["gamesPerShard"], path="manifest.gamesPerShard", error=_error
        ),
        shard_count=require_int(obj["shardCount"], path="manifest.shardCount", error=_error),
        player_count=require_int(obj["playerCount"], path="manifest.playerCount", error=_error),
        card_count=require_int(obj["cardCount"], path="manifest.cardCount", error=_error),
        card_ids=require_str_tuple(obj["cardIds"], path="manifest.cardIds", error=_error),
        card_ids_sha256=require_str(
            obj["cardIdsSha256"], path="manifest.cardIdsSha256", error=_error
        ),
        shards=tuple(_parse_shard(item, index=index) for index, item in enumerate(shards_raw)),
        sample_schema_version=sample_schema_version,
        behavior_policy=behavior_policy,
        sampling_algorithm=sampling_algorithm,
        temperature=temperature,
        reward=reward,
        non_playing_agent=non_playing_agent,
        rollout_roster=rollout_roster,
        tensor_schema=tensor_schema,
    )
