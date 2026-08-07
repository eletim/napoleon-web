"""CLI: validate a self-play dataset end-to-end and print a summary report.

::

    python -m napoleon_ml.cli.inspect_dataset ./datasets/rule-based-v1

or, after an editable install, the ``napoleon-inspect-dataset`` console
script. Every sample is streamed and validated; nothing is buffered beyond
small fixed-size counters and histograms (a 53-slot card histogram, a
6-slot belief-owner-class histogram, three per-split sample counters), so
memory use does not grow with dataset size.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..dataset.constants import BELIEF_OWNER_CLASS_COUNT, CARD_COUNT, DATASET_SAMPLE_TYPE
from ..dataset.errors import DatasetError
from ..dataset.manifest import DatasetManifest
from ..dataset.reader import iter_tensorized_samples, load_manifest
from ..dataset.split import DatasetSplit, SplitConfig, split_for_seed
from ..dataset.tensors import TensorizedPlayingSample


@dataclass
class _Aggregates:
    sample_count: int = 0
    legal_action_count: int = 0
    selected_card_histogram: list[int] = field(default_factory=lambda: [0] * CARD_COUNT)
    belief_owner_histogram: list[int] = field(
        default_factory=lambda: [0] * BELIEF_OWNER_CLASS_COUNT
    )
    split_sample_counts: dict[str, int] = field(
        default_factory=lambda: {split.value: 0 for split in DatasetSplit}
    )
    flat_observation_shape: tuple[int, ...] | None = None
    flat_observation_dtype: str | None = None
    legal_play_mask_shape: tuple[int, ...] | None = None
    belief_target_shape: tuple[int, ...] | None = None


def _count_games_per_split(start_seed: int, end_seed: int, config: SplitConfig) -> dict[str, int]:
    counts = {split.value: 0 for split in DatasetSplit}

    for seed in range(start_seed, end_seed + 1):
        counts[split_for_seed(seed, config).value] += 1

    return counts


def _collect(
    dataset_directory: Path, *, verify_integrity: bool, split_config: SplitConfig
) -> tuple[DatasetManifest, _Aggregates]:
    manifest = load_manifest(dataset_directory)
    if manifest.sample_type != DATASET_SAMPLE_TYPE:
        raise DatasetError(
            "inspect_dataset currently supports only playing-training-sample datasets, "
            f"got {manifest.sample_type!r}."
        )

    aggregates = _Aggregates()

    for tensorized in iter_tensorized_samples(
        dataset_directory, verify_integrity=verify_integrity, split_config=split_config
    ):
        if not isinstance(tensorized, TensorizedPlayingSample):
            raise DatasetError("inspect_dataset expected a TensorizedPlayingSample.")

        aggregates.sample_count += 1

        actor_target = int(tensorized.actor_target)

        if tensorized.legal_play_mask[actor_target] == 1:
            aggregates.legal_action_count += 1

        aggregates.selected_card_histogram[actor_target] += 1

        for owner_class in tensorized.belief_target.tolist():
            aggregates.belief_owner_histogram[owner_class] += 1

        split = split_for_seed(tensorized.seed, split_config)
        aggregates.split_sample_counts[split.value] += 1

        flat_shape = tuple(tensorized.flat_observation.shape)

        if aggregates.flat_observation_shape is None:
            aggregates.flat_observation_shape = flat_shape
            aggregates.flat_observation_dtype = str(tensorized.flat_observation.dtype)
            aggregates.legal_play_mask_shape = tuple(tensorized.legal_play_mask.shape)
            aggregates.belief_target_shape = tuple(tensorized.belief_target.shape)
        elif flat_shape != aggregates.flat_observation_shape:
            raise DatasetError(
                "flat_observation shape is not consistent across the dataset: "
                f"expected {aggregates.flat_observation_shape}, got {flat_shape} "
                f"at seed {tensorized.seed} step {tensorized.step}."
            )

    return manifest, aggregates


def _build_report(
    manifest: DatasetManifest,
    aggregates: _Aggregates,
    games_per_split: dict[str, int],
    total_bytes: int,
    verify_integrity: bool,
) -> dict[str, Any]:
    return {
        "dataset": {
            "schemaVersion": manifest.dataset_schema_version,
            "encoderSchemaVersion": manifest.playing_encoder_schema_version,
            "games": manifest.game_count,
            "samples": manifest.sample_count,
            "shards": manifest.shard_count,
            "startSeed": manifest.start_seed,
            "endSeed": manifest.end_seed,
            "cardIdsSha256": manifest.card_ids_sha256,
        },
        "integrity": {
            "manifestValid": True,
            "shardIntegrityChecked": verify_integrity,
            "shardsValid": manifest.shard_count,
            "shardsTotal": manifest.shard_count,
            "bytes": total_bytes,
            "samplesValid": aggregates.sample_count,
            "samplesTotal": manifest.sample_count,
        },
        "split": {
            split.value: {
                "games": games_per_split[split.value],
                "samples": aggregates.split_sample_counts[split.value],
            }
            for split in DatasetSplit
        },
        "actor": {
            "legalSelectedActions": aggregates.legal_action_count,
            "totalActions": aggregates.sample_count,
            "selectedCardHistogram": aggregates.selected_card_histogram,
        },
        "belief": {"ownerClassHistogram": aggregates.belief_owner_histogram},
        "tensors": {
            "flatObservationShape": list(aggregates.flat_observation_shape or ()),
            "flatObservationDtype": aggregates.flat_observation_dtype,
            "legalPlayMaskShape": list(aggregates.legal_play_mask_shape or ()),
            "beliefTargetShape": list(aggregates.belief_target_shape or ()),
        },
    }


def _print_text_report(report: dict[str, Any]) -> None:
    dataset = report["dataset"]
    integrity = report["integrity"]
    actor = report["actor"]
    belief = report["belief"]
    tensors = report["tensors"]

    print("Dataset")
    print(f"  schema: {dataset['schemaVersion']}")
    print(f"  encoder schema: {dataset['encoderSchemaVersion']}")
    print(f"  games: {dataset['games']}")
    print(f"  samples: {dataset['samples']}")
    print(f"  shards: {dataset['shards']}")
    print(f"  seeds: {dataset['startSeed']}..{dataset['endSeed']}")
    print(f"  card ids sha256: {dataset['cardIdsSha256']}")
    print()

    print("Integrity")
    print(f"  manifest: {'valid' if integrity['manifestValid'] else 'invalid'}")

    if integrity["shardIntegrityChecked"]:
        print(f"  shards: {integrity['shardsValid']} / {integrity['shardsTotal']} valid")
    else:
        print(f"  shards: {integrity['shardsTotal']} (integrity check skipped)")

    print(f"  bytes: {integrity['bytes']}")
    print(f"  samples: {integrity['samplesValid']} / {integrity['samplesTotal']} valid")
    print()

    print("Split")
    for split_name, split_report in report["split"].items():
        print(f"  {split_name}:")
        print(f"    games: {split_report['games']}")
        print(f"    samples: {split_report['samples']}")
    print()

    print("Actor")
    print(f"  legal selected actions: {actor['legalSelectedActions']} / {actor['totalActions']}")
    top_cards = sorted(
        enumerate(actor["selectedCardHistogram"]), key=lambda item: item[1], reverse=True
    )[:5]
    top_cards_text = ", ".join(f"card {index}: {count}" for index, count in top_cards if count > 0)
    print(f"  selected card distribution (top 5): {top_cards_text or '(no samples)'}")
    print()

    print("Belief")
    for owner_class, count in enumerate(belief["ownerClassHistogram"]):
        print(f"  owner class {owner_class}: {count}")
    print()

    print("Tensors")
    print(f"  flat observation shape: {tuple(tensors['flatObservationShape'])}")
    print(f"  flat observation dtype: {tensors['flatObservationDtype']}")
    print(f"  legal play mask shape: {tuple(tensors['legalPlayMaskShape'])}")
    print(f"  belief target shape: {tuple(tensors['beliefTargetShape'])}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="napoleon-inspect-dataset",
        description="Validate and summarize a Napoleon self-play dataset directory.",
    )
    parser.add_argument(
        "dataset_directory", type=Path, help="Path to a generated dataset directory."
    )
    parser.add_argument(
        "--no-integrity-check",
        action="store_true",
        help=(
            "Skip shard SHA-256/byte-length re-verification. Structural checks "
            "(line count, sample count, seed range, game count) and semantic sample "
            "validation still run. Full verification runs unless this flag is passed "
            "explicitly."
        ),
    )
    parser.add_argument(
        "--json", action="store_true", help="Print a machine-readable JSON report to stdout."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    dataset_directory: Path = args.dataset_directory
    verify_integrity = not args.no_integrity_check
    split_config = SplitConfig()

    print(f"Inspecting {dataset_directory} ...", file=sys.stderr)

    try:
        manifest, aggregates = _collect(
            dataset_directory, verify_integrity=verify_integrity, split_config=split_config
        )
    except DatasetError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    # Cross-check the actual streamed sample count against manifest.sampleCount
    # before declaring success. The reader already enforces this per shard, but
    # this is a final, independent check against silent sample loss (e.g. from
    # a future bug that lets a shard's samples pass through uncounted) rather
    # than trusting that every intermediate check ran correctly.
    if aggregates.sample_count != manifest.sample_count:
        print(
            "error: sample count mismatch: expected "
            f"{manifest.sample_count} (manifest.sampleCount), got {aggregates.sample_count}.",
            file=sys.stderr,
        )
        return 1

    games_per_split = _count_games_per_split(manifest.start_seed, manifest.end_seed, split_config)
    total_bytes = sum(shard.byte_length for shard in manifest.shards)
    report = _build_report(manifest, aggregates, games_per_split, total_bytes, verify_integrity)

    if args.json:
        print(json.dumps(report))
    else:
        _print_text_report(report)

    print("OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
