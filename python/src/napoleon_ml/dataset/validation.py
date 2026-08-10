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

from ._strict import require_int_in_range
from .constants import (
    ADJUTANT_DATASET_SAMPLE_TYPE,
    ADJUTANT_ENCODER_SCHEMA_VERSION,
    BIDDING_ACTION_COUNT,
    BIDDING_ACTION_TYPE_BID,
    BIDDING_ACTION_TYPE_PASS,
    BIDDING_DATASET_SAMPLE_TYPE,
    BIDDING_ENCODER_SCHEMA_VERSION,
    BIDDING_HISTORY_SUIT_ORDER,
    CARD_COUNT,
    CARDS_PER_TRICK,
    COMPLETED_TRICK_CARD_SLOT_COUNT,
    CURRENT_POLICY_ROSTER_SOURCE,
    DATASET_FORMAT,
    DATASET_GENERATOR_VERSION,
    DATASET_SAMPLE_TYPE,
    DATASET_SAMPLE_TYPES,
    DATASET_SCHEMA_VERSION,
    EMPTY_BIDDING_ACTION_TYPE,
    EMPTY_BIDDING_SUIT_INDEX,
    EMPTY_CARD_INDEX,
    EMPTY_PLAYER_INDEX,
    EXCHANGE_DATASET_SAMPLE_TYPE,
    EXCHANGE_ENCODER_SCHEMA_VERSION,
    EXPECTED_CARD_IDS,
    FROZEN_ONNX_ROSTER_SOURCE,
    MAX_BIDDING_ACTION_COUNT,
    MAX_BIDDING_TARGET_POINT_CARDS,
    MAX_CONTRACT_TARGET_POINT_CARDS,
    MAX_HAND_COUNT,
    MAX_LATEST_BURIED_EVENT_HIDDEN_NON_POINT_COUNT,
    MAX_SHARD_COUNT,
    MIN_BIDDING_TARGET_POINT_CARDS,
    MIN_CONTRACT_TARGET_POINT_CARDS,
    MULTIPHASE_DATASET_GENERATOR_VERSION,
    MULTIPHASE_DATASET_SCHEMA_VERSION,
    NOT_IN_HAND_CLASS_INDEX,
    PLAYER_COUNT,
    PLAYING_ENCODER_SCHEMA_VERSION,
    PLAYING_MODEL_INPUT_SCHEMA_VERSION,
    PLAYING_SELF_PLAY_BEHAVIOR_POLICY_TYPE,
    PLAYING_SELF_PLAY_DATASET_GENERATOR_VERSION,
    PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE,
    PLAYING_SELF_PLAY_DATASET_SCHEMA_VERSION,
    PLAYING_SELF_PLAY_REWARD_TYPE,
    PLAYING_SELF_PLAY_REWARD_VERSION,
    PLAYING_SELF_PLAY_ROSTER_ASSIGNMENT,
    PLAYING_SELF_PLAY_SAMPLE_SCHEMA_VERSION,
    PLAYING_SELF_PLAY_SAMPLING_ALGORITHM,
    REVEALED_ADJUTANT_CLASS_COUNT,
    ROLLOUT_ROSTER_SOURCES,
    RULE_BASED_AGENT_VERSION,
    RULE_BASED_ROSTER_SOURCE,
    SELF_ROLE_COUNT,
    SELF_ROLE_ORDER,
    SHARD_FILE_DIGITS,
    TRICK_COUNT,
    UINT32_MAX,
)
from .errors import ManifestValidationError, SampleValidationError, ShardIntegrityError
from .manifest import DatasetManifest, DatasetShardManifest
from .sample import (
    AdjutantTrainingSample,
    BiddingTrainingSample,
    EncodedAdjutantObservation,
    EncodedBeliefTarget,
    EncodedBiddingHistory,
    EncodedBiddingObservation,
    EncodedExchangeObservation,
    EncodedPlayingObservation,
    ExchangeTrainingSample,
    PlayingSelfPlaySample,
    PlayingTrainingSample,
    SpecialCardIndices,
    TrainingSample,
)

_SHA256_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SHARD_FILE_NAME_PATTERN = re.compile(rf"^shard-(\d{{{SHARD_FILE_DIGITS}}})\.jsonl$")
_TRUMP_SUIT_OPTION_COUNT = 4
_WINNING_TEAMS = frozenset({"napoleon-team", "alliance"})
_SELF_PLAY_ROLES = frozenset(SELF_ROLE_ORDER)
_REQUESTED_INFERENCE_DEVICES = frozenset({"cpu", "auto", "cuda"})
_RESOLVED_INFERENCE_DEVICES = frozenset({"cpu", "cuda"})


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
    if manifest.dataset_schema_version not in {
        DATASET_SCHEMA_VERSION,
        MULTIPHASE_DATASET_SCHEMA_VERSION,
        PLAYING_SELF_PLAY_DATASET_SCHEMA_VERSION,
    }:
        raise ManifestValidationError(
            "manifest.datasetSchemaVersion mismatch: expected 1, 2, or 3, "
            f"got {manifest.dataset_schema_version}."
        )

    if manifest.format != DATASET_FORMAT:
        raise ManifestValidationError(
            f"manifest.format mismatch: expected {DATASET_FORMAT!r}, got {manifest.format!r}."
        )

    if (
        manifest.dataset_schema_version != PLAYING_SELF_PLAY_DATASET_SCHEMA_VERSION
        and manifest.sample_type not in DATASET_SAMPLE_TYPES
    ):
        raise ManifestValidationError(
            f"manifest.sampleType mismatch: unsupported sample type {manifest.sample_type!r}."
        )

    if manifest.dataset_schema_version == DATASET_SCHEMA_VERSION:
        if manifest.generator_version != DATASET_GENERATOR_VERSION:
            raise ManifestValidationError(
                "manifest.generatorVersion mismatch: "
                f"expected {DATASET_GENERATOR_VERSION}, got {manifest.generator_version}."
            )

        if manifest.sample_type != DATASET_SAMPLE_TYPE:
            raise ManifestValidationError(
                "manifest v1 sampleType must be "
                f"{DATASET_SAMPLE_TYPE!r}, got {manifest.sample_type!r}."
            )

        if manifest.playing_encoder_schema_version != PLAYING_ENCODER_SCHEMA_VERSION:
            actual = manifest.playing_encoder_schema_version
            raise ManifestValidationError(
                "manifest.playingEncoderSchemaVersion mismatch: "
                f"expected {PLAYING_ENCODER_SCHEMA_VERSION}, got {actual}."
            )
    elif manifest.dataset_schema_version == MULTIPHASE_DATASET_SCHEMA_VERSION:
        if manifest.generator_version != MULTIPHASE_DATASET_GENERATOR_VERSION:
            raise ManifestValidationError(
                "manifest.generatorVersion mismatch: "
                f"expected {MULTIPHASE_DATASET_GENERATOR_VERSION}, "
                f"got {manifest.generator_version}."
            )

        if manifest.sample_type == DATASET_SAMPLE_TYPE:
            raise ManifestValidationError("manifest v2 sampleType must be a non-playing type.")

        expected_encoder_version = _encoder_schema_version_for_sample_type(manifest.sample_type)

        if manifest.encoder_schema_version != expected_encoder_version:
            raise ManifestValidationError(
                "manifest.encoderSchemaVersion mismatch: "
                f"expected {expected_encoder_version}, got {manifest.encoder_schema_version}."
            )
    else:
        _validate_playing_self_play_manifest_identity(manifest)

    if manifest.dataset_schema_version == PLAYING_SELF_PLAY_DATASET_SCHEMA_VERSION:
        return

    if manifest.agent is None:
        raise ManifestValidationError("manifest.agent is required for manifest v1/v2.")
    if manifest.agent.type != "rule-based":
        raise ManifestValidationError(
            f"manifest.agent.type mismatch: expected 'rule-based', got {manifest.agent.type!r}."
        )

    if manifest.agent.version != RULE_BASED_AGENT_VERSION:
        raise ManifestValidationError(
            "manifest.agent.version mismatch: "
            f"expected {RULE_BASED_AGENT_VERSION}, got {manifest.agent.version}."
        )


def _validate_playing_self_play_manifest_identity(manifest: DatasetManifest) -> None:
    if manifest.generator_version != PLAYING_SELF_PLAY_DATASET_GENERATOR_VERSION:
        raise ManifestValidationError(
            "self-play manifest.generatorVersion mismatch: "
            f"expected {PLAYING_SELF_PLAY_DATASET_GENERATOR_VERSION}, "
            f"got {manifest.generator_version}."
        )

    if manifest.sample_type != PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE:
        raise ManifestValidationError(
            "self-play manifest.sampleType must be "
            f"{PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE!r}, got {manifest.sample_type!r}."
        )

    if manifest.sample_schema_version != PLAYING_SELF_PLAY_SAMPLE_SCHEMA_VERSION:
        raise ManifestValidationError(
            "self-play manifest.sampleSchemaVersion mismatch: "
            f"expected {PLAYING_SELF_PLAY_SAMPLE_SCHEMA_VERSION}, "
            f"got {manifest.sample_schema_version}."
        )

    if manifest.playing_encoder_schema_version != PLAYING_ENCODER_SCHEMA_VERSION:
        raise ManifestValidationError(
            "self-play manifest.playingEncoderSchemaVersion mismatch: "
            f"expected {PLAYING_ENCODER_SCHEMA_VERSION}, "
            f"got {manifest.playing_encoder_schema_version}."
        )

    if manifest.playing_model_input_schema_version != PLAYING_MODEL_INPUT_SCHEMA_VERSION:
        raise ManifestValidationError(
            "self-play manifest.playingModelInputSchemaVersion mismatch: "
            f"expected {PLAYING_MODEL_INPUT_SCHEMA_VERSION}, "
            f"got {manifest.playing_model_input_schema_version}."
        )

    policy = manifest.behavior_policy
    if policy is None:
        raise ManifestValidationError("self-play manifest.behaviorPolicy is required.")

    if policy.type != PLAYING_SELF_PLAY_BEHAVIOR_POLICY_TYPE:
        raise ManifestValidationError(
            "self-play manifest.behaviorPolicy.type mismatch: "
            f"expected {PLAYING_SELF_PLAY_BEHAVIOR_POLICY_TYPE!r}, got {policy.type!r}."
        )

    for name, value in (
        ("artifactId", policy.artifact_id),
        ("onnxFileName", policy.onnx_file_name),
        ("metadataFileName", policy.metadata_file_name),
    ):
        if value == "":
            raise ManifestValidationError(f"self-play manifest.behaviorPolicy.{name} is empty.")

    if not _SHA256_HEX_PATTERN.match(policy.onnx_sha256):
        raise ManifestValidationError("self-play manifest.behaviorPolicy.onnxSha256 is invalid.")

    if not _SHA256_HEX_PATTERN.match(policy.metadata_sha256):
        raise ManifestValidationError(
            "self-play manifest.behaviorPolicy.metadataSha256 is invalid."
        )

    _validate_optional_inference_runtime(
        "self-play manifest.behaviorPolicy",
        requested=policy.requested_inference_device,
        resolved=policy.resolved_inference_device,
        execution_provider=policy.execution_provider,
    )

    if manifest.sampling_algorithm != PLAYING_SELF_PLAY_SAMPLING_ALGORITHM:
        raise ManifestValidationError(
            "self-play manifest.samplingAlgorithm mismatch: "
            f"expected {PLAYING_SELF_PLAY_SAMPLING_ALGORITHM!r}, "
            f"got {manifest.sampling_algorithm!r}."
        )

    if manifest.temperature is None or not math.isfinite(manifest.temperature):
        raise ManifestValidationError("self-play manifest.temperature must be finite.")

    if manifest.temperature <= 0:
        raise ManifestValidationError("self-play manifest.temperature must be greater than 0.")

    if manifest.reward is None:
        raise ManifestValidationError("self-play manifest.reward is required.")

    if (
        manifest.reward.type != PLAYING_SELF_PLAY_REWARD_TYPE
        or manifest.reward.version != PLAYING_SELF_PLAY_REWARD_VERSION
    ):
        raise ManifestValidationError("self-play manifest.reward metadata mismatch.")

    if manifest.non_playing_agent is None:
        raise ManifestValidationError("self-play manifest.nonPlayingAgent is required.")

    if (
        manifest.non_playing_agent.type != "rule-based"
        or manifest.non_playing_agent.version != RULE_BASED_AGENT_VERSION
    ):
        raise ManifestValidationError("self-play manifest.nonPlayingAgent metadata mismatch.")

    roster = manifest.rollout_roster
    if roster is None:
        raise ManifestValidationError("self-play manifest.rolloutRoster is required.")

    if roster.assignment != PLAYING_SELF_PLAY_ROSTER_ASSIGNMENT:
        raise ManifestValidationError("self-play manifest.rolloutRoster.assignment mismatch.")

    if len(roster.seats) != PLAYER_COUNT:
        raise ManifestValidationError(
            f"self-play manifest.rolloutRoster.seats must have length {PLAYER_COUNT}."
        )

    current_policy_count = 0
    for index, seat in enumerate(roster.seats):
        if seat.source not in ROLLOUT_ROSTER_SOURCES:
            raise ManifestValidationError(
                f"self-play manifest.rolloutRoster.seats[{index}].source is invalid."
            )
        if seat.source == CURRENT_POLICY_ROSTER_SOURCE:
            current_policy_count += 1
        elif seat.source == RULE_BASED_ROSTER_SOURCE:
            if seat.version != RULE_BASED_AGENT_VERSION:
                raise ManifestValidationError(
                    f"self-play manifest.rolloutRoster.seats[{index}] "
                    "rule-based version mismatch."
                )
        elif seat.source == FROZEN_ONNX_ROSTER_SOURCE:
            for field_name, optional_value in (
                ("artifactId", seat.artifact_id),
                ("onnxFileName", seat.onnx_file_name),
                ("metadataFileName", seat.metadata_file_name),
            ):
                if optional_value is None or optional_value == "":
                    raise ManifestValidationError(
                        f"self-play manifest.rolloutRoster.seats[{index}].{field_name} is empty."
                    )
            if seat.onnx_sha256 is None or not _SHA256_HEX_PATTERN.match(seat.onnx_sha256):
                raise ManifestValidationError(
                    f"self-play manifest.rolloutRoster.seats[{index}].onnxSha256 is invalid."
                )
            if (
                seat.metadata_sha256 is None
                or not _SHA256_HEX_PATTERN.match(seat.metadata_sha256)
            ):
                raise ManifestValidationError(
                    f"self-play manifest.rolloutRoster.seats[{index}].metadataSha256 is invalid."
                )
            _validate_optional_inference_runtime(
                f"self-play manifest.rolloutRoster.seats[{index}]",
                requested=seat.requested_inference_device,
                resolved=seat.resolved_inference_device,
                execution_provider=seat.execution_provider,
            )

    if current_policy_count == 0:
        raise ManifestValidationError("self-play manifest.rolloutRoster has no current-policy.")


def _validate_optional_inference_runtime(
    path: str,
    *,
    requested: str | None,
    resolved: str | None,
    execution_provider: str | None,
) -> None:
    fields = (requested, resolved, execution_provider)
    if all(value is None for value in fields):
        return
    if any(value is None for value in fields):
        raise ManifestValidationError(f"{path} inference runtime metadata is incomplete.")
    if requested not in _REQUESTED_INFERENCE_DEVICES:
        raise ManifestValidationError(f"{path}.requestedInferenceDevice is invalid.")
    if resolved not in _RESOLVED_INFERENCE_DEVICES:
        raise ManifestValidationError(f"{path}.resolvedInferenceDevice is invalid.")
    if execution_provider not in _RESOLVED_INFERENCE_DEVICES:
        raise ManifestValidationError(f"{path}.executionProvider is invalid.")
    if resolved != execution_provider:
        raise ManifestValidationError(
            f"{path}.resolvedInferenceDevice must match executionProvider."
        )
    if requested == "cpu" and resolved != "cpu":
        raise ManifestValidationError(f"{path} requested cpu but resolved {resolved}.")
    if requested == "cuda" and resolved != "cuda":
        raise ManifestValidationError(f"{path} requested cuda but resolved {resolved}.")


def _encoder_schema_version_for_sample_type(sample_type: str) -> int:
    if sample_type == DATASET_SAMPLE_TYPE:
        return PLAYING_ENCODER_SCHEMA_VERSION
    if sample_type == BIDDING_DATASET_SAMPLE_TYPE:
        return BIDDING_ENCODER_SCHEMA_VERSION
    if sample_type == EXCHANGE_DATASET_SAMPLE_TYPE:
        return EXCHANGE_ENCODER_SCHEMA_VERSION
    if sample_type == ADJUTANT_DATASET_SAMPLE_TYPE:
        return ADJUTANT_ENCODER_SCHEMA_VERSION

    raise ManifestValidationError(f"Unsupported sampleType: {sample_type!r}.")


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


# --- Sample validation --------------------------------------------------
#
# Mirrors packages/ai-observation/src/encodePlayingObservation.ts's
# validateEncodedPlayingObservation, encodeBiddingHistory.ts's
# validateEncodedBiddingHistory, encodeBeliefTarget.ts's
# validateEncodedBeliefTarget, and packages/training-data/src/validation.ts's
# validatePlayingTrainingSample. Structural JSON-shape checks (required
# keys, JSON types, fixed array lengths) already happened while parsing in
# sample.py; everything here is a numeric-range or cross-field invariant.


def _sample_error(message: str) -> SampleValidationError:
    return SampleValidationError(message)


def validate_sample(sample: TrainingSample) -> None:
    """Validate a parsed sample against every numeric and cross-field rule."""

    if isinstance(sample, PlayingTrainingSample):
        validate_playing_training_sample(sample)
        return
    if isinstance(sample, BiddingTrainingSample):
        validate_bidding_training_sample(sample)
        return
    if isinstance(sample, ExchangeTrainingSample):
        validate_exchange_training_sample(sample)
        return
    if isinstance(sample, AdjutantTrainingSample):
        validate_adjutant_training_sample(sample)
        return
    if isinstance(sample, PlayingSelfPlaySample):
        validate_playing_self_play_sample(sample)
        return

    raise SampleValidationError(f"Unsupported sample class: {type(sample).__name__}.")


def validate_playing_training_sample(sample: PlayingTrainingSample) -> None:
    """Validate a parsed playing sample against every current schema rule."""

    if sample.schema_version != PLAYING_ENCODER_SCHEMA_VERSION:
        raise SampleValidationError(
            f"sample.schemaVersion mismatch: expected {PLAYING_ENCODER_SCHEMA_VERSION}, "
            f"got {sample.schema_version}."
        )

    if sample.observation.schema_version != sample.schema_version:
        raise SampleValidationError("sample.schemaVersion must match observation.schemaVersion.")

    require_int_in_range(
        sample.seed, path="sample.seed", minimum=0, maximum=UINT32_MAX, error=_sample_error
    )

    if sample.step <= 0:
        raise SampleValidationError(f"sample.step must be a positive integer, got {sample.step}.")

    if len(sample.relative_player_ids) != PLAYER_COUNT:
        raise SampleValidationError(
            f"sample.relativePlayerIds must have length {PLAYER_COUNT}, "
            f"got {len(sample.relative_player_ids)}."
        )

    if len(set(sample.relative_player_ids)) != len(sample.relative_player_ids):
        raise SampleValidationError("sample.relativePlayerIds must be unique.")

    for index, player_id in enumerate(sample.relative_player_ids):
        if len(player_id) == 0:
            raise SampleValidationError(f"sample.relativePlayerIds[{index}] must be non-empty.")

    if sample.acting_player_id != sample.relative_player_ids[0]:
        raise SampleValidationError("sample.actingPlayerId must match relativePlayerIds[0].")

    if sample.relative_player_ids != sample.observation.relative_player_ids:
        raise SampleValidationError(
            "sample.relativePlayerIds must match observation.relativePlayerIds."
        )

    validate_encoded_playing_observation(sample.observation)
    validate_encoded_belief_target(sample.belief_target)

    selected_card_index = sample.actor_target.selected_card_index

    if selected_card_index < 0 or selected_card_index >= CARD_COUNT:
        raise SampleValidationError(
            f"actorTarget.selectedCardIndex must be between 0 and {CARD_COUNT - 1}, "
            f"got {selected_card_index}."
        )

    if sample.observation.legal_play_mask[selected_card_index] != 1:
        raise SampleValidationError(
            "actorTarget.selectedCardIndex must be legal in observation.legalPlayMask."
        )

    for index, value in enumerate(sample.observation.legal_play_mask):
        if value == 1 and sample.observation.self_hand_mask[index] != 1:
            raise SampleValidationError(f"legalPlayMask[{index}] must be within selfHandMask.")


def validate_bidding_training_sample(sample: BiddingTrainingSample) -> None:
    if sample.sample_type != BIDDING_DATASET_SAMPLE_TYPE:
        raise SampleValidationError(
            f"sample.sampleType must be {BIDDING_DATASET_SAMPLE_TYPE!r}, "
            f"got {sample.sample_type!r}."
        )

    if sample.schema_version != BIDDING_ENCODER_SCHEMA_VERSION:
        raise SampleValidationError(
            f"sample.schemaVersion mismatch: expected {BIDDING_ENCODER_SCHEMA_VERSION}, "
            f"got {sample.schema_version}."
        )

    if sample.observation.schema_version != sample.schema_version:
        raise SampleValidationError("sample.schemaVersion must match observation.schemaVersion.")

    _validate_sample_common(sample)
    validate_encoded_bidding_observation(sample.observation)
    _expect_int_range("actorTarget", sample.actor_target, 0, BIDDING_ACTION_COUNT - 1)

    if sample.observation.legal_bid_mask[sample.actor_target] != 1:
        raise SampleValidationError("actorTarget must be legal in observation.legalBidMask.")


def validate_exchange_training_sample(sample: ExchangeTrainingSample) -> None:
    if sample.sample_type != EXCHANGE_DATASET_SAMPLE_TYPE:
        raise SampleValidationError(
            f"sample.sampleType must be {EXCHANGE_DATASET_SAMPLE_TYPE!r}, "
            f"got {sample.sample_type!r}."
        )

    if sample.schema_version != EXCHANGE_ENCODER_SCHEMA_VERSION:
        raise SampleValidationError(
            f"sample.schemaVersion mismatch: expected {EXCHANGE_ENCODER_SCHEMA_VERSION}, "
            f"got {sample.schema_version}."
        )

    if sample.observation.schema_version != sample.schema_version:
        raise SampleValidationError("sample.schemaVersion must match observation.schemaVersion.")

    _validate_sample_common(sample)
    validate_encoded_exchange_observation(sample.observation)
    _validate_exchange_actor_target(
        sample.actor_target.discard_target_mask, sample.observation.legal_discard_card_mask
    )


def validate_adjutant_training_sample(sample: AdjutantTrainingSample) -> None:
    if sample.sample_type != ADJUTANT_DATASET_SAMPLE_TYPE:
        raise SampleValidationError(
            f"sample.sampleType must be {ADJUTANT_DATASET_SAMPLE_TYPE!r}, "
            f"got {sample.sample_type!r}."
        )

    if sample.schema_version != ADJUTANT_ENCODER_SCHEMA_VERSION:
        raise SampleValidationError(
            f"sample.schemaVersion mismatch: expected {ADJUTANT_ENCODER_SCHEMA_VERSION}, "
            f"got {sample.schema_version}."
        )

    if sample.observation.schema_version != sample.schema_version:
        raise SampleValidationError("sample.schemaVersion must match observation.schemaVersion.")

    _validate_sample_common(sample)
    validate_encoded_adjutant_observation(sample.observation)
    _expect_int_range("actorTarget", sample.actor_target, 0, CARD_COUNT - 1)

    if sample.observation.legal_adjutant_mask[sample.actor_target] != 1:
        raise SampleValidationError("actorTarget must be legal in observation.legalAdjutantMask.")


def validate_playing_self_play_sample(sample: PlayingSelfPlaySample) -> None:
    if sample.sample_type != PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE:
        raise SampleValidationError(
            f"sample.sampleType must be {PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE!r}, "
            f"got {sample.sample_type!r}."
        )

    if sample.schema_version != PLAYING_SELF_PLAY_SAMPLE_SCHEMA_VERSION:
        raise SampleValidationError(
            f"sample.schemaVersion mismatch: expected {PLAYING_SELF_PLAY_SAMPLE_SCHEMA_VERSION}, "
            f"got {sample.schema_version}."
        )

    if sample.observation.schema_version != PLAYING_ENCODER_SCHEMA_VERSION:
        raise SampleValidationError(
            "self-play sample observation.schemaVersion mismatch: "
            f"expected {PLAYING_ENCODER_SCHEMA_VERSION}, got {sample.observation.schema_version}."
        )

    require_int_in_range(
        sample.seed, path="sample.seed", minimum=0, maximum=UINT32_MAX, error=_sample_error
    )

    if sample.step <= 0:
        raise SampleValidationError(f"sample.step must be a positive integer, got {sample.step}.")

    _validate_playing_identity(
        acting_player_id=sample.acting_player_id,
        relative_player_ids=sample.relative_player_ids,
        observation=sample.observation,
    )
    if sample.acting_seat_source != CURRENT_POLICY_ROSTER_SOURCE:
        raise SampleValidationError("actingSeatSource must be current-policy.")

    if sample.behavior_policy_artifact_id == "":
        raise SampleValidationError("behaviorPolicyArtifactId must be non-empty.")

    if len(sample.rollout_seat_sources) != PLAYER_COUNT:
        raise SampleValidationError(
            f"rolloutSeatSources must have length {PLAYER_COUNT}, "
            f"got {len(sample.rollout_seat_sources)}."
        )

    for index, source in enumerate(sample.rollout_seat_sources):
        if source not in ROLLOUT_ROSTER_SOURCES:
            raise SampleValidationError(f"rolloutSeatSources[{index}] is invalid: {source!r}.")

    validate_encoded_playing_observation(sample.observation)

    _expect_int_range("selectedCardIndex", sample.selected_card_index, 0, CARD_COUNT - 1)

    if sample.observation.legal_play_mask[sample.selected_card_index] != 1:
        raise SampleValidationError(
            "selectedCardIndex must be legal in observation.legalPlayMask."
        )

    if not math.isfinite(sample.behavior_log_probability):
        raise SampleValidationError("behaviorLogProbability must be finite.")

    if sample.behavior_log_probability > 1e-12:
        raise SampleValidationError("behaviorLogProbability must be <= 0.")

    if sum(sample.observation.legal_play_mask) == 1 and sample.behavior_log_probability != 0:
        raise SampleValidationError("forced self-play actions must have behaviorLogProbability 0.")

    if sample.terminal_reward not in {-1, 1}:
        raise SampleValidationError("terminalReward must be +1 or -1.")

    outcome = sample.outcome

    if outcome.winner not in _WINNING_TEAMS:
        raise SampleValidationError(f"outcome.winner is invalid: {outcome.winner!r}.")

    if outcome.acting_player_team not in _WINNING_TEAMS:
        raise SampleValidationError(
            f"outcome.actingPlayerTeam is invalid: {outcome.acting_player_team!r}."
        )

    if outcome.acting_player_role not in _SELF_PLAY_ROLES:
        raise SampleValidationError(
            f"outcome.actingPlayerRole is invalid: {outcome.acting_player_role!r}."
        )

    napoleon_player_index = sample.observation.napoleon_player_one_hot.index(1)
    expected_napoleon_player_id = sample.relative_player_ids[napoleon_player_index]

    if outcome.napoleon_player_id != expected_napoleon_player_id:
        raise SampleValidationError(
            "outcome.napoleonPlayerId must match observation.napoleonPlayerOneHot."
        )

    if outcome.acting_player_role == "napoleon":
        if sample.acting_player_id != outcome.napoleon_player_id:
            raise SampleValidationError(
                "outcome.actingPlayerRole napoleon must match outcome.napoleonPlayerId."
            )
    elif outcome.acting_player_role == "napoleon-solo":
        if sample.acting_player_id != outcome.napoleon_player_id:
            raise SampleValidationError(
                "outcome.actingPlayerRole napoleon-solo must match outcome.napoleonPlayerId."
            )
    elif sample.acting_player_id == outcome.napoleon_player_id:
        raise SampleValidationError(
            "outcome.actingPlayerRole must be napoleon or napoleon-solo when the acting player is "
            "outcome.napoleonPlayerId."
        )

    expected_team = (
        "napoleon-team"
        if outcome.acting_player_role in {"napoleon", "adjutant", "napoleon-solo"}
        else "alliance"
    )

    if outcome.acting_player_team != expected_team:
        raise SampleValidationError(
            "outcome.actingPlayerTeam is inconsistent with outcome.actingPlayerRole."
        )

    expected_reward = 1 if outcome.acting_player_team == outcome.winner else -1

    if sample.terminal_reward != expected_reward:
        raise SampleValidationError(
            "terminalReward is inconsistent with outcome.winner and outcome.actingPlayerTeam."
        )

    role_index = sample.observation.self_role_one_hot.index(1)
    if outcome.acting_player_role != SELF_ROLE_ORDER[role_index]:
        raise SampleValidationError(
            "outcome.actingPlayerRole must match observation.selfRoleOneHot."
        )


def _validate_sample_common(
    sample: BiddingTrainingSample | ExchangeTrainingSample | AdjutantTrainingSample,
) -> None:
    require_int_in_range(
        sample.seed, path="sample.seed", minimum=0, maximum=UINT32_MAX, error=_sample_error
    )

    if sample.step <= 0:
        raise SampleValidationError(f"sample.step must be a positive integer, got {sample.step}.")

    if len(sample.relative_player_ids) != PLAYER_COUNT:
        raise SampleValidationError(
            f"sample.relativePlayerIds must have length {PLAYER_COUNT}, "
            f"got {len(sample.relative_player_ids)}."
        )

    if len(set(sample.relative_player_ids)) != len(sample.relative_player_ids):
        raise SampleValidationError("sample.relativePlayerIds must be unique.")

    for index, player_id in enumerate(sample.relative_player_ids):
        if len(player_id) == 0:
            raise SampleValidationError(f"sample.relativePlayerIds[{index}] must be non-empty.")

    if sample.acting_player_id != sample.relative_player_ids[0]:
        raise SampleValidationError("sample.actingPlayerId must match relativePlayerIds[0].")

    if sample.relative_player_ids != sample.observation.relative_player_ids:
        raise SampleValidationError(
            "sample.relativePlayerIds must match observation.relativePlayerIds."
        )


def _validate_playing_identity(
    *,
    acting_player_id: str,
    relative_player_ids: tuple[str, ...],
    observation: EncodedPlayingObservation,
) -> None:
    if len(relative_player_ids) != PLAYER_COUNT:
        raise SampleValidationError(
            f"sample.relativePlayerIds must have length {PLAYER_COUNT}, "
            f"got {len(relative_player_ids)}."
        )

    if len(set(relative_player_ids)) != len(relative_player_ids):
        raise SampleValidationError("sample.relativePlayerIds must be unique.")

    for index, player_id in enumerate(relative_player_ids):
        if len(player_id) == 0:
            raise SampleValidationError(f"sample.relativePlayerIds[{index}] must be non-empty.")

    if acting_player_id != relative_player_ids[0]:
        raise SampleValidationError("sample.actingPlayerId must match relativePlayerIds[0].")

    if relative_player_ids != observation.relative_player_ids:
        raise SampleValidationError(
            "sample.relativePlayerIds must match observation.relativePlayerIds."
        )


def validate_encoded_playing_observation(observation: EncodedPlayingObservation) -> None:
    if observation.schema_version != PLAYING_ENCODER_SCHEMA_VERSION:
        raise SampleValidationError(
            f"Unsupported playing encoder schema version: {observation.schema_version}"
        )

    _expect_length("relativePlayerIds", observation.relative_player_ids, PLAYER_COUNT)
    _expect_length("trumpSuitOneHot", observation.trump_suit_one_hot, _TRUMP_SUIT_OPTION_COUNT)
    _expect_length("napoleonPlayerOneHot", observation.napoleon_player_one_hot, PLAYER_COUNT)
    _expect_length(
        "revealedAdjutantPlayerOneHot",
        observation.revealed_adjutant_player_one_hot,
        REVEALED_ADJUTANT_CLASS_COUNT,
    )
    _expect_length("selfRoleOneHot", observation.self_role_one_hot, SELF_ROLE_COUNT)
    _expect_length("calledAdjutantCardMask", observation.called_adjutant_card_mask, CARD_COUNT)
    _expect_length("selfHandMask", observation.self_hand_mask, CARD_COUNT)
    _expect_length("legalPlayMask", observation.legal_play_mask, CARD_COUNT)
    _expect_length("handCountByPlayer", observation.hand_count_by_player, PLAYER_COUNT)
    _expect_length(
        "capturedPointCardMaskByPlayer",
        observation.captured_point_card_mask_by_player,
        PLAYER_COUNT,
    )

    for index, mask in enumerate(observation.captured_point_card_mask_by_player):
        _expect_length(f"capturedPointCardMaskByPlayer[{index}]", mask, CARD_COUNT)

    _expect_length(
        "currentTrickCardIndices", observation.current_trick_card_indices, CARDS_PER_TRICK
    )
    _expect_length(
        "currentTrickPlayerIndices", observation.current_trick_player_indices, CARDS_PER_TRICK
    )
    _expect_length("currentTrickSlotMask", observation.current_trick_slot_mask, CARDS_PER_TRICK)
    _expect_length(
        "completedTrickCardIndices",
        observation.completed_trick_card_indices,
        COMPLETED_TRICK_CARD_SLOT_COUNT,
    )
    _expect_length(
        "completedTrickPlayerIndices",
        observation.completed_trick_player_indices,
        COMPLETED_TRICK_CARD_SLOT_COUNT,
    )
    _expect_length(
        "completedTrickSlotMask",
        observation.completed_trick_slot_mask,
        COMPLETED_TRICK_CARD_SLOT_COUNT,
    )
    _expect_length(
        "completedTrickWinnerIndices", observation.completed_trick_winner_indices, TRICK_COUNT
    )
    _expect_length("completedTrickMask", observation.completed_trick_mask, TRICK_COUNT)
    validate_encoded_bidding_history(observation.bidding_history)
    _expect_length(
        "latestBuriedEventPointCardMask",
        observation.latest_buried_event_point_card_mask,
        CARD_COUNT,
    )

    _validate_relative_player_ids(observation.relative_player_ids)
    _expect_int_range("trickNumber", observation.trick_number, 1, TRICK_COUNT)
    _expect_int_range("completedTrickCount", observation.completed_trick_count, 0, TRICK_COUNT)
    _expect_int_range(
        "contractTargetPointCards",
        observation.contract_target_point_cards,
        MIN_CONTRACT_TARGET_POINT_CARDS,
        MAX_CONTRACT_TARGET_POINT_CARDS,
    )
    _expect_int_range(
        "latestBuriedEventHiddenNonPointCount",
        observation.latest_buried_event_hidden_non_point_count,
        0,
        MAX_LATEST_BURIED_EVENT_HIDDEN_NON_POINT_COUNT,
    )

    _validate_one_hot("trumpSuitOneHot", observation.trump_suit_one_hot)
    _validate_one_hot("napoleonPlayerOneHot", observation.napoleon_player_one_hot)
    _validate_one_hot("revealedAdjutantPlayerOneHot", observation.revealed_adjutant_player_one_hot)
    _validate_one_hot("selfRoleOneHot", observation.self_role_one_hot)
    _validate_mask("calledAdjutantCardMask", observation.called_adjutant_card_mask)
    _expect_sum("calledAdjutantCardMask", observation.called_adjutant_card_mask, 1)
    _validate_mask("selfHandMask", observation.self_hand_mask)
    _validate_mask("legalPlayMask", observation.legal_play_mask)

    for index, mask in enumerate(observation.captured_point_card_mask_by_player):
        _validate_mask(f"capturedPointCardMaskByPlayer[{index}]", mask)

    _validate_mask("currentTrickSlotMask", observation.current_trick_slot_mask)
    _validate_mask("completedTrickSlotMask", observation.completed_trick_slot_mask)
    _validate_mask("completedTrickMask", observation.completed_trick_mask)
    _validate_mask(
        "latestBuriedEventPointCardMask", observation.latest_buried_event_point_card_mask
    )
    _validate_mask("latestBuriedEventPresent", (observation.latest_buried_event_present,))

    for index, value in enumerate(observation.hand_count_by_player):
        _expect_int_range(f"handCountByPlayer[{index}]", value, 0, MAX_HAND_COUNT)

    _validate_card_index_array("currentTrickCardIndices", observation.current_trick_card_indices)
    _validate_card_index_array(
        "completedTrickCardIndices", observation.completed_trick_card_indices
    )
    _validate_player_index_array(
        "currentTrickPlayerIndices", observation.current_trick_player_indices
    )
    _validate_player_index_array(
        "completedTrickPlayerIndices", observation.completed_trick_player_indices
    )
    _validate_player_index_array(
        "completedTrickWinnerIndices", observation.completed_trick_winner_indices
    )
    _validate_special_card_indices(observation.special_card_indices)
    _validate_mask_is_subset(
        "legalPlayMask",
        observation.legal_play_mask,
        "selfHandMask",
        observation.self_hand_mask,
    )
    _validate_contiguous_mask("currentTrickSlotMask", observation.current_trick_slot_mask)
    _validate_contiguous_mask("completedTrickMask", observation.completed_trick_mask)
    _validate_current_trick_slots(observation)
    _validate_completed_trick_slots(observation)
    _validate_latest_buried_event(observation)

    completed_trick_count = sum(observation.completed_trick_mask)

    if observation.completed_trick_count != completed_trick_count:
        raise SampleValidationError(
            "completedTrickCount must equal completedTrickMask sum: "
            f"{observation.completed_trick_count} != {completed_trick_count}."
        )

    if observation.trick_number != observation.completed_trick_count + 1:
        raise SampleValidationError(
            "trickNumber must equal completedTrickCount + 1: "
            f"{observation.trick_number} != {observation.completed_trick_count + 1}."
        )


def validate_encoded_bidding_observation(observation: EncodedBiddingObservation) -> None:
    if observation.schema_version != BIDDING_ENCODER_SCHEMA_VERSION:
        raise SampleValidationError(
            f"Unsupported bidding encoder schema version: {observation.schema_version}"
        )

    _expect_length("relativePlayerIds", observation.relative_player_ids, PLAYER_COUNT)
    _expect_length("selfHandMask", observation.self_hand_mask, CARD_COUNT)
    _expect_length("legalBidMask", observation.legal_bid_mask, BIDDING_ACTION_COUNT)
    validate_encoded_bidding_history(observation.bidding_history)
    _validate_relative_player_ids(observation.relative_player_ids)
    _validate_mask("selfHandMask", observation.self_hand_mask)
    _expect_sum("selfHandMask", observation.self_hand_mask, 10)
    _validate_mask("legalBidMask", observation.legal_bid_mask)
    _validate_mask("highestBidPresent", (observation.highest_bid_present,))
    _expect_int_range("starterPlayerIndex", observation.starter_player_index, 0, PLAYER_COUNT - 1)
    _expect_int_range("consecutivePassCount", observation.consecutive_pass_count, 0, PLAYER_COUNT)

    if observation.highest_bid_present == 0:
        if (
            observation.highest_bid_player_index,
            observation.highest_bid_suit_index,
            observation.highest_bid_target_point_cards,
        ) != (EMPTY_PLAYER_INDEX, EMPTY_BIDDING_SUIT_INDEX, 0):
            raise SampleValidationError(
                "Empty highest bid fields must use player -1, suit -1, and target 0."
            )
        return

    _expect_int_range(
        "highestBidPlayerIndex", observation.highest_bid_player_index, 0, PLAYER_COUNT - 1
    )
    _expect_int_range(
        "highestBidSuitIndex",
        observation.highest_bid_suit_index,
        0,
        len(BIDDING_HISTORY_SUIT_ORDER) - 1,
    )
    _expect_int_range(
        "highestBidTargetPointCards",
        observation.highest_bid_target_point_cards,
        MIN_BIDDING_TARGET_POINT_CARDS,
        MAX_BIDDING_TARGET_POINT_CARDS,
    )


def validate_encoded_exchange_observation(observation: EncodedExchangeObservation) -> None:
    if observation.schema_version != EXCHANGE_ENCODER_SCHEMA_VERSION:
        raise SampleValidationError(
            f"Unsupported exchange encoder schema version: {observation.schema_version}"
        )

    _expect_length("relativePlayerIds", observation.relative_player_ids, PLAYER_COUNT)
    _expect_length("trumpSuitOneHot", observation.trump_suit_one_hot, _TRUMP_SUIT_OPTION_COUNT)
    _expect_length("calledAdjutantCardMask", observation.called_adjutant_card_mask, CARD_COUNT)
    _expect_length("selfHandMask", observation.self_hand_mask, CARD_COUNT)
    _expect_length("legalDiscardCardMask", observation.legal_discard_card_mask, CARD_COUNT)
    _expect_length("handCountByPlayer", observation.hand_count_by_player, PLAYER_COUNT)
    validate_encoded_bidding_history(observation.bidding_history)
    _validate_relative_player_ids(observation.relative_player_ids)
    _expect_int_range(
        "contractTargetPointCards",
        observation.contract_target_point_cards,
        MIN_CONTRACT_TARGET_POINT_CARDS,
        MAX_CONTRACT_TARGET_POINT_CARDS,
    )
    _validate_one_hot("trumpSuitOneHot", observation.trump_suit_one_hot)
    _validate_mask("calledAdjutantCardMask", observation.called_adjutant_card_mask)
    _expect_sum("calledAdjutantCardMask", observation.called_adjutant_card_mask, 1)
    _validate_mask("selfHandMask", observation.self_hand_mask)
    _expect_sum("selfHandMask", observation.self_hand_mask, 13)
    _validate_mask("legalDiscardCardMask", observation.legal_discard_card_mask)

    if observation.legal_discard_card_mask != observation.self_hand_mask:
        raise SampleValidationError("legalDiscardCardMask must equal selfHandMask at exchange.")

    for index, value in enumerate(observation.hand_count_by_player):
        _expect_int_range(f"handCountByPlayer[{index}]", value, 0, MAX_HAND_COUNT)

    if observation.hand_count_by_player[0] != 13:
        raise SampleValidationError("handCountByPlayer[0] must be 13 for Napoleon at exchange.")

    _validate_required_special_card_indices(observation.special_card_indices)


def validate_encoded_adjutant_observation(observation: EncodedAdjutantObservation) -> None:
    if observation.schema_version != ADJUTANT_ENCODER_SCHEMA_VERSION:
        raise SampleValidationError(
            f"Unsupported adjutant encoder schema version: {observation.schema_version}"
        )

    _expect_length("relativePlayerIds", observation.relative_player_ids, PLAYER_COUNT)
    _expect_length("trumpSuitOneHot", observation.trump_suit_one_hot, _TRUMP_SUIT_OPTION_COUNT)
    _expect_length("selfHandMask", observation.self_hand_mask, CARD_COUNT)
    _expect_length("legalAdjutantMask", observation.legal_adjutant_mask, CARD_COUNT)
    validate_encoded_bidding_history(observation.bidding_history)
    _validate_relative_player_ids(observation.relative_player_ids)
    _validate_one_hot("trumpSuitOneHot", observation.trump_suit_one_hot)
    _validate_mask("selfHandMask", observation.self_hand_mask)
    _expect_sum("selfHandMask", observation.self_hand_mask, 10)
    _validate_mask("legalAdjutantMask", observation.legal_adjutant_mask)
    _expect_int_range(
        "contractTargetPointCards",
        observation.contract_target_point_cards,
        MIN_CONTRACT_TARGET_POINT_CARDS,
        MAX_CONTRACT_TARGET_POINT_CARDS,
    )
    _validate_required_special_card_indices(observation.special_card_indices)


def _validate_exchange_actor_target(
    discard_target_mask: tuple[int, ...], legal_discard_card_mask: tuple[int, ...]
) -> None:
    _expect_length("discardTargetMask", discard_target_mask, CARD_COUNT)
    _validate_mask("discardTargetMask", discard_target_mask)
    _expect_sum("discardTargetMask", discard_target_mask, 3)
    _validate_mask_is_subset(
        "discardTargetMask", discard_target_mask, "legalDiscardCardMask", legal_discard_card_mask
    )


def validate_encoded_bidding_history(history: EncodedBiddingHistory) -> None:
    _validate_mask("biddingHistory.actionMask", history.action_mask)
    _validate_contiguous_mask("biddingHistory.actionMask", history.action_mask)

    for index in range(MAX_BIDDING_ACTION_COUNT):
        slot_name = f"biddingHistory[{index}]"
        mask = history.action_mask[index]
        action_type = history.action_type_indices[index]
        player_index = history.player_indices[index]
        suit_index = history.suit_indices[index]
        target = history.target_point_cards[index]

        if mask == 0:
            if (action_type, player_index, suit_index, target) != (
                EMPTY_BIDDING_ACTION_TYPE,
                EMPTY_PLAYER_INDEX,
                EMPTY_BIDDING_SUIT_INDEX,
                0,
            ):
                raise SampleValidationError(
                    f"{slot_name} must contain only empty values when actionMask is 0."
                )

            continue

        _expect_int_range(f"{slot_name}.playerIndex", player_index, 0, PLAYER_COUNT - 1)

        if action_type == BIDDING_ACTION_TYPE_PASS:
            if suit_index != EMPTY_BIDDING_SUIT_INDEX or target != 0:
                raise SampleValidationError(f"{slot_name} pass must use suit -1 and target 0.")

            continue

        if action_type == BIDDING_ACTION_TYPE_BID:
            _expect_int_range(
                f"{slot_name}.suitIndex", suit_index, 0, len(BIDDING_HISTORY_SUIT_ORDER) - 1
            )
            _expect_int_range(
                f"{slot_name}.targetPointCards",
                target,
                MIN_BIDDING_TARGET_POINT_CARDS,
                MAX_BIDDING_TARGET_POINT_CARDS,
            )
            continue

        raise SampleValidationError(f"{slot_name}.actionType must be pass, bid, or empty.")


def validate_encoded_belief_target(target: EncodedBeliefTarget) -> None:
    _expect_length("ownerClassByCard", target.owner_class_by_card, CARD_COUNT)
    _expect_length("hiddenOwnershipLossMask", target.hidden_ownership_loss_mask, CARD_COUNT)

    for owner_class in target.owner_class_by_card:
        if owner_class < 0 or owner_class > NOT_IN_HAND_CLASS_INDEX:
            raise SampleValidationError(
                f"ownerClassByCard contains an invalid owner class: {owner_class}."
            )

    _validate_mask("hiddenOwnershipLossMask", target.hidden_ownership_loss_mask)


def _expect_length(name: str, values: tuple[object, ...], expected_length: int) -> None:
    if len(values) != expected_length:
        raise SampleValidationError(
            f"{name} must have length {expected_length}, got {len(values)}."
        )


def _expect_int_range(name: str, value: int, minimum: int, maximum: int) -> None:
    if value < minimum or value > maximum:
        raise SampleValidationError(f"{name} must be between {minimum} and {maximum}, got {value}.")


def _validate_mask(name: str, values: tuple[int, ...]) -> None:
    for value in values:
        if value != 0 and value != 1:
            raise SampleValidationError(f"{name} must contain only 0/1 values, got {value}.")


def _validate_one_hot(name: str, values: tuple[int, ...]) -> None:
    _validate_mask(name, values)
    _expect_sum(name, values, 1)


def _expect_sum(name: str, values: tuple[int, ...], expected_sum: int) -> None:
    actual_sum = sum(values)

    if actual_sum != expected_sum:
        raise SampleValidationError(f"{name} must sum to {expected_sum}, got {actual_sum}.")


def _validate_relative_player_ids(player_ids: tuple[str, ...]) -> None:
    if len(set(player_ids)) != len(player_ids):
        raise SampleValidationError("observation.relativePlayerIds must be unique.")

    for index, player_id in enumerate(player_ids):
        if len(player_id) == 0:
            raise SampleValidationError(f"relativePlayerIds[{index}] must be a non-empty string.")


def _validate_optional_card_index(name: str, value: int) -> None:
    if value == EMPTY_CARD_INDEX:
        return

    _expect_int_range(name, value, 0, CARD_COUNT - 1)


def _validate_optional_player_index(name: str, value: int) -> None:
    if value == EMPTY_PLAYER_INDEX:
        return

    _expect_int_range(name, value, 0, PLAYER_COUNT - 1)


def _validate_card_index_array(name: str, values: tuple[int, ...]) -> None:
    for index, value in enumerate(values):
        _validate_optional_card_index(f"{name}[{index}]", value)


def _validate_player_index_array(name: str, values: tuple[int, ...]) -> None:
    for index, value in enumerate(values):
        _validate_optional_player_index(f"{name}[{index}]", value)


def _validate_special_card_indices(indices: SpecialCardIndices) -> None:
    _expect_int_range("specialCardIndices.oruma", indices.oruma, 0, CARD_COUNT - 1)
    _expect_int_range("specialCardIndices.yoromeki", indices.yoromeki, 0, CARD_COUNT - 1)
    _validate_optional_card_index("specialCardIndices.seiJack", indices.sei_jack)
    _validate_optional_card_index("specialCardIndices.uraJack", indices.ura_jack)


def _validate_required_special_card_indices(indices: SpecialCardIndices) -> None:
    _expect_int_range("specialCardIndices.oruma", indices.oruma, 0, CARD_COUNT - 1)
    _expect_int_range("specialCardIndices.yoromeki", indices.yoromeki, 0, CARD_COUNT - 1)
    _expect_int_range("specialCardIndices.seiJack", indices.sei_jack, 0, CARD_COUNT - 1)
    _expect_int_range("specialCardIndices.uraJack", indices.ura_jack, 0, CARD_COUNT - 1)


def _validate_mask_is_subset(
    child_name: str,
    child: tuple[int, ...],
    parent_name: str,
    parent: tuple[int, ...],
) -> None:
    for index, value in enumerate(child):
        if value == 1 and parent[index] != 1:
            raise SampleValidationError(f"{child_name}[{index}] must be within {parent_name}.")


def _validate_contiguous_mask(name: str, mask: tuple[int, ...]) -> None:
    seen_empty = False

    for value in mask:
        if value == 0:
            seen_empty = True
            continue

        if seen_empty:
            raise SampleValidationError(f"{name} must contain contiguous 1 values followed by 0s.")


def _validate_current_trick_slots(observation: EncodedPlayingObservation) -> None:
    for index in range(CARDS_PER_TRICK):
        _validate_card_player_slot(
            f"currentTrick[{index}]",
            observation.current_trick_slot_mask[index],
            observation.current_trick_card_indices[index],
            observation.current_trick_player_indices[index],
        )


def _validate_completed_trick_slots(observation: EncodedPlayingObservation) -> None:
    for trick_index in range(TRICK_COUNT):
        trick_mask = observation.completed_trick_mask[trick_index]
        winner_index = observation.completed_trick_winner_indices[trick_index]

        if trick_mask == 1:
            _expect_int_range(
                f"completedTrickWinnerIndices[{trick_index}]", winner_index, 0, PLAYER_COUNT - 1
            )
        elif winner_index != EMPTY_PLAYER_INDEX:
            raise SampleValidationError(
                f"completedTrickWinnerIndices[{trick_index}] must be {EMPTY_PLAYER_INDEX} "
                "when completedTrickMask is 0."
            )

        for card_offset in range(CARDS_PER_TRICK):
            slot_index = trick_index * CARDS_PER_TRICK + card_offset
            slot_mask = observation.completed_trick_slot_mask[slot_index]

            if slot_mask != trick_mask:
                raise SampleValidationError(
                    f"completedTrickSlotMask[{slot_index}] must match "
                    f"completedTrickMask[{trick_index}]."
                )

            _validate_card_player_slot(
                f"completedTrick[{slot_index}]",
                slot_mask,
                observation.completed_trick_card_indices[slot_index],
                observation.completed_trick_player_indices[slot_index],
            )


def _validate_card_player_slot(name: str, mask: int, card_index: int, player_index: int) -> None:
    if mask == 1:
        _expect_int_range(f"{name}.cardIndex", card_index, 0, CARD_COUNT - 1)
        _expect_int_range(f"{name}.playerIndex", player_index, 0, PLAYER_COUNT - 1)
        return

    if card_index != EMPTY_CARD_INDEX or player_index != EMPTY_PLAYER_INDEX:
        raise SampleValidationError(
            f"{name} must use empty card and player indices when slot mask is 0."
        )


def _validate_latest_buried_event(observation: EncodedPlayingObservation) -> None:
    public_point_card_count = sum(observation.latest_buried_event_point_card_mask)

    if observation.latest_buried_event_present == 0:
        if (
            public_point_card_count != 0
            or observation.latest_buried_event_hidden_non_point_count != 0
        ):
            raise SampleValidationError(
                "latestBuriedEvent fields must be empty when latestBuriedEventPresent is 0."
            )

        return

    if public_point_card_count + observation.latest_buried_event_hidden_non_point_count != 3:
        raise SampleValidationError(
            "latestBuriedEvent public point cards plus hidden non-point count must equal 3."
        )
