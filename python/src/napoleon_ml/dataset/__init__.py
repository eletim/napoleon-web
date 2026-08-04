"""Strict loader for TypeScript-generated self-play datasets."""

from __future__ import annotations

from .errors import (
    DatasetError,
    ManifestValidationError,
    SampleValidationError,
    ShardIntegrityError,
    UnsupportedSchemaError,
)
from .manifest import DatasetAgentInfo, DatasetManifest, DatasetShardManifest, parse_manifest
from .reader import iter_raw_samples, iter_samples, load_manifest
from .sample import (
    EncodedBeliefTarget,
    EncodedBiddingHistory,
    EncodedPlayAction,
    EncodedPlayingObservation,
    PlayingTrainingSample,
    SpecialCardIndices,
    parse_sample,
)
from .validation import (
    calculate_card_ids_sha256,
    validate_dataset_directory,
    validate_encoded_belief_target,
    validate_encoded_bidding_history,
    validate_encoded_playing_observation,
    validate_manifest,
    validate_sample,
)

__all__ = [
    "DatasetAgentInfo",
    "DatasetError",
    "DatasetManifest",
    "DatasetShardManifest",
    "EncodedBeliefTarget",
    "EncodedBiddingHistory",
    "EncodedPlayAction",
    "EncodedPlayingObservation",
    "ManifestValidationError",
    "PlayingTrainingSample",
    "SampleValidationError",
    "ShardIntegrityError",
    "SpecialCardIndices",
    "UnsupportedSchemaError",
    "calculate_card_ids_sha256",
    "iter_raw_samples",
    "iter_samples",
    "load_manifest",
    "parse_manifest",
    "parse_sample",
    "validate_dataset_directory",
    "validate_encoded_belief_target",
    "validate_encoded_bidding_history",
    "validate_encoded_playing_observation",
    "validate_manifest",
    "validate_sample",
]
