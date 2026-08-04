"""Streaming reader for a TypeScript-generated self-play dataset directory.

Nothing here ever materializes the whole dataset in memory: each shard is
opened once, read line by line in binary mode, and its running SHA-256/byte
length/line count/seed bookkeeping is checked against ``manifest.json``
immediately after the shard's last line, before the next shard is opened.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

from ._json import loads_strict
from ._strict import require_int
from .errors import ManifestValidationError, SampleValidationError, ShardIntegrityError
from .manifest import DatasetManifest, DatasetShardManifest, parse_manifest
from .sample import PlayingTrainingSample, parse_sample
from .validation import validate_dataset_directory, validate_manifest, validate_sample


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


def iter_raw_samples(
    dataset_directory: Path | str,
    *,
    verify_integrity: bool = True,
) -> Iterator[dict[str, object]]:
    """Stream every sample of every shard as an untyped, JSON-shape-only dict.

    Each shard is read in binary mode so its exact bytes can be hashed and
    counted. Every line must be non-empty UTF-8 JSON terminated by a single
    ``\\n`` (no CRLF, no stray ``\\r``, no missing final newline) with no
    duplicate keys and no ``NaN``/``Infinity``/``-Infinity``. When
    ``verify_integrity`` is true (the default), each shard's accumulated
    SHA-256, byte length, line count, sample count, first/last seed, and
    unique seed count are compared against ``manifest.json`` immediately
    after that shard's last line.
    """

    directory = Path(dataset_directory)
    manifest = load_manifest(directory)

    for shard in manifest.shards:
        yield from _iter_shard_raw_samples(directory, shard, verify_integrity=verify_integrity)


def _iter_shard_raw_samples(
    directory: Path,
    shard: DatasetShardManifest,
    *,
    verify_integrity: bool,
) -> Iterator[dict[str, object]]:
    shard_path = directory / shard.file
    hasher = hashlib.sha256()
    byte_length = 0
    line_count = 0
    sample_count = 0
    first_seed: int | None = None
    last_seed: int | None = None
    unique_seeds: set[int] = set()

    with shard_path.open("rb") as file:
        for line_number, raw_line in enumerate(file, start=1):
            context = f"{shard.file}:{line_number}"

            hasher.update(raw_line)
            byte_length += len(raw_line)
            line_count += 1

            if not raw_line.endswith(b"\n"):
                raise ShardIntegrityError(f"{context}: final line is missing a trailing newline.")

            content = raw_line[:-1]

            if content.endswith(b"\r"):
                raise ShardIntegrityError(f"{context}: line uses CRLF line endings, expected LF.")

            if len(content) == 0:
                raise ShardIntegrityError(f"{context}: line is empty.")

            try:
                text = content.decode("utf-8", errors="strict")
            except UnicodeDecodeError as error:
                raise ShardIntegrityError(f"{context}: line is not valid UTF-8: {error}") from error

            if "\r" in text:
                raise ShardIntegrityError(f"{context}: line contains a stray carriage return.")

            try:
                raw_value = loads_strict(text)
            except ValueError as error:
                raise ShardIntegrityError(f"{context}: line is not valid JSON: {error}") from error

            if not isinstance(raw_value, dict):
                raise ShardIntegrityError(f"{context}: line must be a JSON object.")

            sample_count += 1

            def _seed_error(message: str, context: str = context) -> ShardIntegrityError:
                return ShardIntegrityError(f"{context}: {message}")

            seed = require_int(raw_value.get("seed"), path="seed", error=_seed_error)

            if first_seed is None:
                first_seed = seed

            last_seed = seed
            unique_seeds.add(seed)

            yield raw_value

    if verify_integrity:
        _verify_shard_integrity(
            shard,
            sha256=hasher.hexdigest(),
            byte_length=byte_length,
            line_count=line_count,
            sample_count=sample_count,
            first_seed=first_seed,
            last_seed=last_seed,
            unique_seed_count=len(unique_seeds),
        )


def _verify_shard_integrity(
    shard: DatasetShardManifest,
    *,
    sha256: str,
    byte_length: int,
    line_count: int,
    sample_count: int,
    first_seed: int | None,
    last_seed: int | None,
    unique_seed_count: int,
) -> None:
    if sha256 != shard.sha256:
        raise ShardIntegrityError(
            f"{shard.file}: SHA-256 mismatch: expected {shard.sha256}, got {sha256}."
        )

    if byte_length != shard.byte_length:
        raise ShardIntegrityError(
            f"{shard.file}: byte length mismatch: expected {shard.byte_length}, got {byte_length}."
        )

    if line_count != shard.sample_count:
        raise ShardIntegrityError(
            f"{shard.file}: line count mismatch: expected {shard.sample_count}, got {line_count}."
        )

    if sample_count != shard.sample_count:
        raise ShardIntegrityError(
            f"{shard.file}: sample count mismatch: expected {shard.sample_count}, "
            f"got {sample_count}."
        )

    if first_seed != shard.start_seed:
        raise ShardIntegrityError(
            f"{shard.file}: first sample seed mismatch: expected {shard.start_seed}, "
            f"got {first_seed}."
        )

    if last_seed != shard.end_seed:
        raise ShardIntegrityError(
            f"{shard.file}: last sample seed mismatch: expected {shard.end_seed}, got {last_seed}."
        )

    if unique_seed_count != shard.game_count:
        raise ShardIntegrityError(
            f"{shard.file}: unique seed count mismatch: expected {shard.game_count} "
            f"(gameCount), got {unique_seed_count}."
        )


def iter_samples(
    dataset_directory: Path | str,
    *,
    verify_integrity: bool = True,
) -> Iterator[PlayingTrainingSample]:
    """Stream every sample as a fully parsed, fully validated dataclass.

    In addition to per-sample structural and semantic validation (see
    :mod:`napoleon_ml.dataset.validation`), this enforces dataset-wide
    ordering: seeds are non-decreasing and form one contiguous block per
    game (a seed can never reappear after the block moves on, which would
    indicate a game split across a shard boundary), and within a seed's
    block, ``step`` strictly increases.
    """

    directory = Path(dataset_directory)
    manifest = load_manifest(directory)

    current_seed: int | None = None
    previous_step: int | None = None
    shard_line_number = 0
    current_shard_file = ""

    for shard in manifest.shards:
        current_shard_file = shard.file
        shard_line_number = 0

        for raw in _iter_shard_raw_samples(directory, shard, verify_integrity=verify_integrity):
            shard_line_number += 1
            context = f"{shard.file}:{shard_line_number}"

            sample = parse_sample(raw, context=context)
            validate_sample(sample)

            if current_seed is None:
                if sample.seed != manifest.start_seed:
                    raise SampleValidationError(
                        f"{context}: sample seed must start the dataset at "
                        f"{manifest.start_seed}, got {sample.seed}."
                    )

                current_seed = sample.seed
            elif sample.seed == current_seed:
                pass
            elif sample.seed == current_seed + 1:
                current_seed = sample.seed
                previous_step = None
            else:
                raise SampleValidationError(
                    f"{context}: sample seed {sample.seed} breaks dataset seed ordering "
                    f"after seed {current_seed} (a seed must not reappear once the "
                    "dataset has moved past it, and seeds must advance by exactly 1)."
                )

            if previous_step is not None and sample.step <= previous_step:
                raise SampleValidationError(
                    f"{context}: sample step must increase within seed {current_seed}: "
                    f"{sample.step} is not greater than {previous_step}."
                )

            previous_step = sample.step

            yield sample

    if current_seed is not None and current_seed != manifest.end_seed:
        raise SampleValidationError(
            f"{current_shard_file}:{shard_line_number}: dataset ended at seed {current_seed}, "
            f"expected to end at manifest.endSeed {manifest.end_seed}."
        )
