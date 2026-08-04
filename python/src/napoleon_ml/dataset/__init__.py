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
from .reader import (
    iter_raw_samples,
    iter_samples,
    iter_tensorized_samples,
    load_manifest,
)
from .sample import (
    EncodedBeliefTarget,
    EncodedBiddingHistory,
    EncodedPlayAction,
    EncodedPlayingObservation,
    PlayingTrainingSample,
    SpecialCardIndices,
    parse_sample,
)
from .split import DatasetSplit, SplitConfig, split_for_seed
from .tensors import (
    FLAT_OBSERVATION_FEATURE_COUNT,
    FLAT_OBSERVATION_LAYOUT,
    FeatureSlice,
    PlayingObservationTensors,
    TensorizedPlayingSample,
    tensorize_observation,
    tensorize_sample,
    validate_tensorized_sample,
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
    "FLAT_OBSERVATION_FEATURE_COUNT",
    "FLAT_OBSERVATION_LAYOUT",
    "DatasetAgentInfo",
    "DatasetError",
    "DatasetManifest",
    "DatasetShardManifest",
    "DatasetSplit",
    "EncodedBeliefTarget",
    "EncodedBiddingHistory",
    "EncodedPlayAction",
    "EncodedPlayingObservation",
    "FeatureSlice",
    "ManifestValidationError",
    "PlayingObservationTensors",
    "PlayingTrainingSample",
    "SampleValidationError",
    "ShardIntegrityError",
    "SpecialCardIndices",
    "SplitConfig",
    "TensorizedPlayingSample",
    "UnsupportedSchemaError",
    "calculate_card_ids_sha256",
    "iter_raw_samples",
    "iter_samples",
    "iter_tensorized_samples",
    "load_manifest",
    "parse_manifest",
    "parse_sample",
    "split_for_seed",
    "tensorize_observation",
    "tensorize_sample",
    "validate_dataset_directory",
    "validate_encoded_belief_target",
    "validate_encoded_bidding_history",
    "validate_encoded_playing_observation",
    "validate_manifest",
    "validate_sample",
    "validate_tensorized_sample",
]
