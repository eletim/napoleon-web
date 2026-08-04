"""Semantic validation for manifests and dataset directories.

Structural JSON-shape parsing lives in :mod:`napoleon_ml.dataset.manifest`;
this module checks the parsed values against the numeric ranges, hashes, and
cross-field invariants enforced by the TypeScript producer
(``packages/training-data/src/validation.ts``), plus filesystem-level checks
that TypeScript never needs to perform (the dataset directory is trusted
there because it was just written; here it is an untrusted input).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from .constants import (
    CARD_COUNT,
    DATASET_FORMAT,
    DATASET_GENERATOR_VERSION,
    DATASET_SAMPLE_TYPE,
    DATASET_SCHEMA_VERSION,
    EXPECTED_CARD_IDS,
    MAX_SHARD_COUNT,
    PLAYER_COUNT,
    PLAYING_ENCODER_SCHEMA_VERSION,
    RULE_BASED_AGENT_VERSION,
    SHARD_FILE_DIGITS,
    UINT32_MAX,
)
from .errors import ManifestValidationError, ShardIntegrityError
from .manifest import DatasetManifest, DatasetShardManifest

_SHA256_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SHARD_FILE_NAME_PATTERN = re.compile(rf"^shard-(\d{{{SHARD_FILE_DIGITS}}})\.jsonl$")


def calculate_card_ids_sha256() -> str:
    """Recompute the manifest ``cardIdsSha256`` value from :data:`EXPECTED_CARD_IDS`.

    Must byte-for-byte match ``packages/training-data/src/serialization.ts``'s
    ``sha256Utf8(JSON.stringify(CARD_IDS))``. Python's default ``json.dumps``
    inserts a space after ``,``/``:`` and would produce different bytes than
    ``JSON.stringify``, so this uses the compact separators that match it.
    """

    encoded = json.dumps(list(EXPECTED_CARD_IDS), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def shard_file_name(index: int) -> str:
    """Mirror ``packages/training-data/src/validation.ts``'s ``shardFileName``."""

    if index < 0 or index >= MAX_SHARD_COUNT:
        raise ManifestValidationError(
            f"shard index must be between 0 and {MAX_SHARD_COUNT - 1}, got {index}."
        )

    return f"shard-{index:0{SHARD_FILE_DIGITS}d}.jsonl"


def validate_manifest(manifest: DatasetManifest) -> None:
    """Validate cross-field and numeric invariants of an already-parsed manifest.

    Raises :class:`~napoleon_ml.dataset.errors.ManifestValidationError` on the
    first violation found.
    """

    _validate_schema_identity(manifest)
    _validate_manifest_numbers(manifest)

    if manifest.game_count != manifest.end_seed - manifest.start_seed + 1:
        raise ManifestValidationError("manifest.gameCount must match seed range.")

    expected_shard_count = _expected_shard_count(manifest.game_count, manifest.games_per_shard)

    if manifest.shard_count != expected_shard_count:
        raise ManifestValidationError(
            "manifest.shardCount must match gameCount and gamesPerShard: "
            f"expected {expected_shard_count}, got {manifest.shard_count}."
        )

    if manifest.shard_count != len(manifest.shards):
        raise ManifestValidationError(
            "manifest.shardCount must match len(shards): "
            f"expected {manifest.shard_count}, got {len(manifest.shards)}."
        )

    actual_sample_count = sum(shard.sample_count for shard in manifest.shards)

    if manifest.sample_count != actual_sample_count:
        raise ManifestValidationError(
            "manifest.sampleCount must equal the sum of shard sampleCount: "
            f"expected {manifest.sample_count}, got {actual_sample_count}."
        )

    _validate_card_ids(manifest)
    _validate_shards(manifest)


def validate_dataset_directory(directory: Path, manifest: DatasetManifest) -> None:
    """Validate that ``directory`` contains exactly the files ``manifest`` describes.

    Rejects a shard listed in the manifest but missing on disk, an unlisted
    ``shard-*.jsonl`` file present on disk, a shard file name that is not a
    bare basename (path traversal, nested paths, absolute paths), and a
    shard path that is a symlink (which could otherwise point outside of
    ``directory``). Only regular files are accepted.
    """

    if not directory.is_dir():
        raise ManifestValidationError(f"Dataset directory does not exist: {directory}")

    expected_names = {"manifest.json"}

    for shard in manifest.shards:
        _validate_shard_file_name_is_safe(shard.file)
        expected_names.add(shard.file)

        shard_path = directory / shard.file

        if shard_path.is_symlink():
            raise ShardIntegrityError(f"{shard.file}: shard path must not be a symlink.")

        if not shard_path.is_file():
            raise ShardIntegrityError(f"{shard.file}: shard file is missing from {directory}.")

    actual_names = {entry.name for entry in directory.iterdir()}
    unexpected_names = sorted(actual_names - expected_names)

    if unexpected_names:
        raise ShardIntegrityError(
            f"Dataset directory contains unexpected entries not listed in manifest.json: "
            f"{unexpected_names!r}"
        )


def _validate_shard_file_name_is_safe(file_name: str) -> None:
    if file_name != Path(file_name).name:
        raise ShardIntegrityError(
            f"Shard file name must be a bare file name with no path separators: {file_name!r}"
        )

    if file_name in {".", ".."}:
        raise ShardIntegrityError(
            f"Shard file name must not be a path traversal segment: {file_name!r}"
        )

    if not _SHARD_FILE_NAME_PATTERN.match(file_name):
        raise ShardIntegrityError(
            f"Shard file name must match shard-{'0' * SHARD_FILE_DIGITS}.jsonl, got {file_name!r}"
        )


def _validate_schema_identity(manifest: DatasetManifest) -> None:
    if manifest.dataset_schema_version != DATASET_SCHEMA_VERSION:
        raise ManifestValidationError(
            "manifest.datasetSchemaVersion mismatch: "
            f"expected {DATASET_SCHEMA_VERSION}, got {manifest.dataset_schema_version}."
        )

    if manifest.generator_version != DATASET_GENERATOR_VERSION:
        raise ManifestValidationError(
            "manifest.generatorVersion mismatch: "
            f"expected {DATASET_GENERATOR_VERSION}, got {manifest.generator_version}."
        )

    if manifest.playing_encoder_schema_version != PLAYING_ENCODER_SCHEMA_VERSION:
        actual = manifest.playing_encoder_schema_version
        raise ManifestValidationError(
            "manifest.playingEncoderSchemaVersion mismatch: "
            f"expected {PLAYING_ENCODER_SCHEMA_VERSION}, got {actual}."
        )

    if manifest.format != DATASET_FORMAT:
        raise ManifestValidationError(
            f"manifest.format mismatch: expected {DATASET_FORMAT!r}, got {manifest.format!r}."
        )

    if manifest.sample_type != DATASET_SAMPLE_TYPE:
        raise ManifestValidationError(
            f"manifest.sampleType mismatch: expected {DATASET_SAMPLE_TYPE!r}, "
            f"got {manifest.sample_type!r}."
        )

    if manifest.agent.type != "rule-based":
        raise ManifestValidationError(
            f"manifest.agent.type mismatch: expected 'rule-based', got {manifest.agent.type!r}."
        )

    if manifest.agent.version != RULE_BASED_AGENT_VERSION:
        raise ManifestValidationError(
            "manifest.agent.version mismatch: "
            f"expected {RULE_BASED_AGENT_VERSION}, got {manifest.agent.version}."
        )


def _validate_manifest_numbers(manifest: DatasetManifest) -> None:
    _require_uint32("manifest.startSeed", manifest.start_seed)
    _require_uint32("manifest.endSeed", manifest.end_seed)

    if manifest.end_seed < manifest.start_seed:
        raise ManifestValidationError("manifest.endSeed must be >= manifest.startSeed.")

    _require_positive_int("manifest.gameCount", manifest.game_count)
    _require_positive_int("manifest.sampleCount", manifest.sample_count)
    _require_positive_int("manifest.gamesPerShard", manifest.games_per_shard)
    _require_positive_int("manifest.shardCount", manifest.shard_count)

    if manifest.shard_count > MAX_SHARD_COUNT:
        raise ManifestValidationError(f"manifest.shardCount must not exceed {MAX_SHARD_COUNT}.")

    if manifest.player_count != PLAYER_COUNT:
        raise ManifestValidationError(
            f"manifest.playerCount must be {PLAYER_COUNT}, got {manifest.player_count}."
        )

    if manifest.card_count != CARD_COUNT:
        raise ManifestValidationError(
            f"manifest.cardCount must be {CARD_COUNT}, got {manifest.card_count}."
        )


def _validate_card_ids(manifest: DatasetManifest) -> None:
    if len(manifest.card_ids) != CARD_COUNT:
        raise ManifestValidationError(
            f"manifest.cardIds must have length {CARD_COUNT}, got {len(manifest.card_ids)}."
        )

    if len(set(manifest.card_ids)) != len(manifest.card_ids):
        raise ManifestValidationError("manifest.cardIds must not contain duplicates.")

    if manifest.card_ids != EXPECTED_CARD_IDS:
        raise ManifestValidationError(
            "manifest.cardIds does not match the TypeScript schema-v1 card ordering."
        )

    if not _SHA256_HEX_PATTERN.match(manifest.card_ids_sha256):
        raise ManifestValidationError(
            "manifest.cardIdsSha256 must be 64 lowercase hex characters, "
            f"got {manifest.card_ids_sha256!r}."
        )

    expected_sha256 = calculate_card_ids_sha256()

    if manifest.card_ids_sha256 != expected_sha256:
        raise ManifestValidationError(
            f"manifest.cardIdsSha256 mismatch: expected {expected_sha256}, "
            f"got {manifest.card_ids_sha256}."
        )


def _validate_shards(manifest: DatasetManifest) -> None:
    expected_start_seed = manifest.start_seed
    seen_files: set[str] = set()

    for index, shard in enumerate(manifest.shards):
        _validate_shard_numbers(shard)

        expected_file = shard_file_name(index)

        if shard.file != expected_file:
            raise ManifestValidationError(
                f"manifest.shards[{index}].file must be {expected_file!r}, got {shard.file!r}."
            )

        if shard.file in seen_files:
            raise ManifestValidationError(f"Duplicate shard file in manifest: {shard.file!r}")

        seen_files.add(shard.file)

        if shard.start_seed != expected_start_seed:
            raise ManifestValidationError(
                f"Shard {shard.file} has a seed gap or overlap: expected startSeed "
                f"{expected_start_seed}, got {shard.start_seed}."
            )

        if shard.game_count != shard.end_seed - shard.start_seed + 1:
            raise ManifestValidationError(
                f"Shard {shard.file} gameCount must match its seed range."
            )

        is_last_shard = index == len(manifest.shards) - 1

        if not is_last_shard and shard.game_count != manifest.games_per_shard:
            raise ManifestValidationError(
                f"Shard {shard.file} gameCount must equal manifest.gamesPerShard "
                f"({manifest.games_per_shard}) for a non-final shard."
            )

        if is_last_shard and shard.game_count > manifest.games_per_shard:
            raise ManifestValidationError(
                f"Final shard {shard.file} gameCount must not exceed manifest.gamesPerShard "
                f"({manifest.games_per_shard})."
            )

        expected_start_seed = shard.end_seed + 1

    if not manifest.shards or manifest.shards[0].start_seed != manifest.start_seed:
        raise ManifestValidationError("First shard must start at manifest.startSeed.")

    if manifest.shards[-1].end_seed != manifest.end_seed:
        raise ManifestValidationError("Last shard must end at manifest.endSeed.")

    if expected_start_seed != manifest.end_seed + 1:
        raise ManifestValidationError(
            "Shard seed ranges must cover the manifest seed range exactly."
        )


def _validate_shard_numbers(shard: DatasetShardManifest) -> None:
    _require_uint32(f"Shard {shard.file} startSeed", shard.start_seed)
    _require_uint32(f"Shard {shard.file} endSeed", shard.end_seed)

    if shard.end_seed < shard.start_seed:
        raise ManifestValidationError(f"Shard {shard.file} endSeed must be >= startSeed.")

    _require_positive_int(f"Shard {shard.file} gameCount", shard.game_count)
    _require_positive_int(f"Shard {shard.file} sampleCount", shard.sample_count)
    _require_positive_int(f"Shard {shard.file} byteLength", shard.byte_length)

    if not _SHA256_HEX_PATTERN.match(shard.sha256):
        raise ManifestValidationError(
            f"Shard {shard.file} sha256 must be 64 lowercase hex characters, got {shard.sha256!r}."
        )


def _expected_shard_count(game_count: int, games_per_shard: int) -> int:
    expected = math.ceil(game_count / games_per_shard)

    if expected < 1:
        raise ManifestValidationError("Expected shard count must be a positive integer.")

    return expected


def _require_uint32(name: str, value: int) -> None:
    if value < 0 or value > UINT32_MAX:
        raise ManifestValidationError(f"{name} must be between 0 and {UINT32_MAX}, got {value}.")


def _require_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ManifestValidationError(f"{name} must be a positive integer, got {value}.")
