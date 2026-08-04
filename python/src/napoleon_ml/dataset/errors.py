"""Exceptions raised while loading a Napoleon self-play dataset.

Every raised instance should carry enough context in its message to locate
the offending value without a debugger: the dataset-relative file name, the
JSONL line number when applicable, a JSON-path-like field locator, and both
the expected and actual values.
"""

from __future__ import annotations


class DatasetError(Exception):
    """Base class for every dataset-loading error."""


class UnsupportedSchemaError(DatasetError):
    """Raised when a manifest or sample declares a schema version this loader does not support."""


class ManifestValidationError(DatasetError):
    """Raised when ``manifest.json`` fails structural or cross-field validation."""


class ShardIntegrityError(DatasetError):
    """Raised when a shard file does not match its manifest metadata or is malformed."""


class SampleValidationError(DatasetError):
    """Raised when a single JSONL sample fails structural or cross-field validation."""
