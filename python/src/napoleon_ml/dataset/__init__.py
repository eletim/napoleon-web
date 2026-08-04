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
from .reader import load_manifest
from .validation import calculate_card_ids_sha256, validate_dataset_directory, validate_manifest

__all__ = [
    "DatasetAgentInfo",
    "DatasetError",
    "DatasetManifest",
    "DatasetShardManifest",
    "ManifestValidationError",
    "SampleValidationError",
    "ShardIntegrityError",
    "UnsupportedSchemaError",
    "calculate_card_ids_sha256",
    "load_manifest",
    "parse_manifest",
    "validate_dataset_directory",
    "validate_manifest",
]
