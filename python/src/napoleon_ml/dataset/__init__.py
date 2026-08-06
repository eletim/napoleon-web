"""Strict loader for TypeScript-generated self-play datasets."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

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
    MODEL_INPUT_FEATURE_COUNT,
    MODEL_INPUT_LAYOUT,
    MODEL_INPUT_ONEHOT_LAYOUT,
    MODEL_INPUT_SCHEMA_VERSION,
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

if TYPE_CHECKING:
    from .pytorch import PlayingIterableDataset, PlayingTorchSample, create_playing_dataloader

_PYTORCH_EXPORTS = {
    "PlayingIterableDataset",
    "PlayingTorchSample",
    "create_playing_dataloader",
}

__all__ = [
    "FLAT_OBSERVATION_FEATURE_COUNT",
    "FLAT_OBSERVATION_LAYOUT",
    "MODEL_INPUT_FEATURE_COUNT",
    "MODEL_INPUT_LAYOUT",
    "MODEL_INPUT_ONEHOT_LAYOUT",
    "MODEL_INPUT_SCHEMA_VERSION",
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
    "PlayingIterableDataset",
    "PlayingObservationTensors",
    "PlayingTrainingSample",
    "PlayingTorchSample",
    "SampleValidationError",
    "ShardIntegrityError",
    "SpecialCardIndices",
    "SplitConfig",
    "TensorizedPlayingSample",
    "UnsupportedSchemaError",
    "calculate_card_ids_sha256",
    "create_playing_dataloader",
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


def __getattr__(name: str) -> object:
    if name in _PYTORCH_EXPORTS:
        module = import_module("napoleon_ml.dataset.pytorch")
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
