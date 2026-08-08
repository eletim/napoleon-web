"""Streaming reader for a TypeScript-generated self-play dataset directory.

Nothing here ever materializes the whole dataset in memory: each shard is
opened once, read line by line in binary mode, and its running SHA-256/byte
length/line count/sample count/seed/game-count bookkeeping is checked
against ``manifest.json`` immediately after the shard's last line, before
the next shard is opened. Structural checks (line count, sample count,
first/last seed, game count) always run; the byte-identity checks
(SHA-256, byte length) are the only ones ``verify_integrity=False`` skips.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

from ._json import loads_strict
from ._strict import require_int
from .constants import PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE
from .errors import ManifestValidationError, SampleValidationError, ShardIntegrityError
from .manifest import DatasetManifest, DatasetShardManifest, parse_manifest
from .sample import PlayingSelfPlaySample, TrainingSample, parse_sample
from .split import DatasetSplit, SplitConfig, split_for_seed
from .tensors import TensorizedPlayingSelfPlaySample, TensorizedTrainingSample, tensorize_sample
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
    duplicate keys and no ``NaN``/``Infinity``/``-Infinity``. A shard's line
    count, sample count, first/last sample seed, and game count (counted by
    seed transitions, not by collecting a ``set`` of seeds -- see
    :func:`_iter_shard_raw_samples`) are always compared against
    ``manifest.json`` immediately after that shard's last line, regardless
    of ``verify_integrity``. Only the byte-identity checks -- SHA-256 and
    byte length -- are skipped when ``verify_integrity`` is false; a
    truncated, corrupted, or reordered shard is still rejected either way.
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
    previous_seed: int | None = None
    game_count = 0

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

            if seed != previous_seed:
                game_count += 1
                previous_seed = seed

            last_seed = seed

            yield raw_value

    # Structural checks always run, regardless of verify_integrity: they
    # catch a truncated or reordered shard even when the caller opted out of
    # the (comparatively expensive) byte-identity checks below.
    _verify_shard_structural_integrity(
        shard,
        line_count=line_count,
        sample_count=sample_count,
        first_seed=first_seed,
        last_seed=last_seed,
        game_count=game_count,
    )

    if verify_integrity:
        _verify_shard_hash_integrity(shard, sha256=hasher.hexdigest(), byte_length=byte_length)


def _verify_shard_hash_integrity(
    shard: DatasetShardManifest,
    *,
    sha256: str,
    byte_length: int,
) -> None:
    if sha256 != shard.sha256:
        raise ShardIntegrityError(
            f"{shard.file}: SHA-256 mismatch: expected {shard.sha256}, got {sha256}."
        )

    if byte_length != shard.byte_length:
        raise ShardIntegrityError(
            f"{shard.file}: byte length mismatch: expected {shard.byte_length}, got {byte_length}."
        )


def _verify_shard_structural_integrity(
    shard: DatasetShardManifest,
    *,
    line_count: int,
    sample_count: int,
    first_seed: int | None,
    last_seed: int | None,
    game_count: int,
) -> None:
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

    if game_count != shard.game_count:
        raise ShardIntegrityError(
            f"{shard.file}: game count mismatch: expected {shard.game_count} (gameCount), "
            f"got {game_count}."
        )


def iter_samples(
    dataset_directory: Path | str,
    *,
    verify_integrity: bool = True,
) -> Iterator[TrainingSample]:
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

    yield from _iter_samples_with_manifest(
        directory,
        manifest,
        verify_integrity=verify_integrity,
    )


def _iter_samples_with_manifest(
    directory: Path,
    manifest: DatasetManifest,
    *,
    verify_integrity: bool,
) -> Iterator[TrainingSample]:
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

            sample = parse_sample(raw, context=context, sample_type=manifest.sample_type)
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


def iter_playing_self_play_samples(
    dataset_directory: Path | str,
    *,
    verify_integrity: bool = True,
) -> Iterator[PlayingSelfPlaySample]:
    """Stream a v3 ``playing-self-play-sample`` dataset as validated samples."""

    directory = Path(dataset_directory)
    manifest = load_manifest(directory)

    if manifest.sample_type != PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE:
        raise SampleValidationError(
            "iter_playing_self_play_samples requires a playing-self-play-sample dataset, "
            f"got {manifest.sample_type!r}."
        )

    for sample in _iter_samples_with_manifest(
        directory,
        manifest,
        verify_integrity=verify_integrity,
    ):
        if not isinstance(sample, PlayingSelfPlaySample):
            raise SampleValidationError(
                "iter_playing_self_play_samples received a non-self-play sample."
            )

        yield sample


def iter_tensorized_samples(
    dataset_directory: Path | str,
    *,
    split: DatasetSplit | None = None,
    split_config: SplitConfig | None = None,
    verify_integrity: bool = True,
) -> Iterator[TensorizedTrainingSample]:
    """Stream fully validated, tensorized samples, optionally filtered to one split.

    Filtering happens inline as each sample is produced (its seed is checked
    against ``split_config`` before it is tensorized and before the next
    sample is read); the dataset is never buffered in memory to filter it.
    """

    directory = Path(dataset_directory)
    manifest = load_manifest(directory)

    yield from _iter_tensorized_samples_with_manifest(
        directory,
        manifest,
        split=split,
        split_config=split_config,
        verify_integrity=verify_integrity,
    )


def iter_tensorized_playing_self_play_samples(
    dataset_directory: Path | str,
    *,
    split: DatasetSplit | None = None,
    split_config: SplitConfig | None = None,
    verify_integrity: bool = True,
) -> Iterator[TensorizedPlayingSelfPlaySample]:
    """Stream tensorized self-play samples, optionally filtered by game seed split."""

    directory = Path(dataset_directory)
    manifest = load_manifest(directory)

    if manifest.sample_type != PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE:
        raise SampleValidationError(
            "iter_tensorized_playing_self_play_samples requires a "
            f"playing-self-play-sample dataset, got {manifest.sample_type!r}."
        )

    for sample in _iter_tensorized_samples_with_manifest(
        directory,
        manifest,
        split=split,
        split_config=split_config,
        verify_integrity=verify_integrity,
    ):
        if not isinstance(sample, TensorizedPlayingSelfPlaySample):
            raise SampleValidationError(
                "iter_tensorized_playing_self_play_samples received a non-self-play sample."
            )

        yield sample


def _iter_tensorized_samples_with_manifest(
    directory: Path,
    manifest: DatasetManifest,
    *,
    split: DatasetSplit | None = None,
    split_config: SplitConfig | None = None,
    verify_integrity: bool = True,
) -> Iterator[TensorizedTrainingSample]:
    effective_split_config = split_config if split_config is not None else SplitConfig()

    for sample in _iter_samples_with_manifest(
        directory,
        manifest,
        verify_integrity=verify_integrity,
    ):
        if split is not None and split_for_seed(sample.seed, effective_split_config) != split:
            continue

        yield tensorize_sample(sample)
