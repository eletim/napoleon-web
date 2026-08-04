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
    require_str,
    require_str_tuple,
)
from .errors import ManifestValidationError

_SHARD_KEYS = frozenset(
    {"file", "startSeed", "endSeed", "gameCount", "sampleCount", "byteLength", "sha256"}
)
_AGENT_KEYS = frozenset({"type", "version"})
_MANIFEST_KEYS = frozenset(
    {
        "datasetSchemaVersion",
        "generatorVersion",
        "playingEncoderSchemaVersion",
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
class DatasetManifest:
    dataset_schema_version: int
    generator_version: int
    playing_encoder_schema_version: int
    format: str
    sample_type: str
    agent: DatasetAgentInfo
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
    require_exact_keys(obj, _MANIFEST_KEYS, path="manifest", error=_error)

    shards_raw = require_list(obj["shards"], path="manifest.shards", error=_error)

    return DatasetManifest(
        dataset_schema_version=require_int(
            obj["datasetSchemaVersion"], path="manifest.datasetSchemaVersion", error=_error
        ),
        generator_version=require_int(
            obj["generatorVersion"], path="manifest.generatorVersion", error=_error
        ),
        playing_encoder_schema_version=require_int(
            obj["playingEncoderSchemaVersion"],
            path="manifest.playingEncoderSchemaVersion",
            error=_error,
        ),
        format=require_str(obj["format"], path="manifest.format", error=_error),
        sample_type=require_str(obj["sampleType"], path="manifest.sampleType", error=_error),
        agent=_parse_agent(obj["agent"]),
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
    )
