"""Streaming reader for a TypeScript-generated self-play dataset directory."""

from __future__ import annotations

from pathlib import Path

from ._json import loads_strict
from .errors import ManifestValidationError
from .manifest import DatasetManifest, parse_manifest
from .validation import validate_dataset_directory, validate_manifest


def load_manifest(dataset_directory: Path | str) -> DatasetManifest:
    """Load, strictly parse, and validate ``manifest.json``.

    Also checks that the dataset directory on disk matches the manifest
    exactly: every listed shard exists as a plain file with a safe name, and
    no unlisted files are present. This function never opens a shard file;
    per-shard byte/hash/sample-count integrity is verified while streaming
    samples (see :func:`iter_raw_samples`).
    """

    directory = Path(dataset_directory)
    manifest_path = directory / "manifest.json"

    if not manifest_path.is_file():
        raise ManifestValidationError(f"manifest.json not found in {directory}")

    try:
        text = manifest_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ManifestValidationError(f"manifest.json is not valid UTF-8: {error}") from error

    try:
        raw = loads_strict(text)
    except ValueError as error:
        raise ManifestValidationError(f"manifest.json is not valid JSON: {error}") from error

    manifest = parse_manifest(raw)
    validate_manifest(manifest)
    validate_dataset_directory(directory, manifest)

    return manifest
