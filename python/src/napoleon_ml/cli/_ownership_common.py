"""Shared helpers for ownership baseline CLIs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

from napoleon_ml.dataset.errors import DatasetError
from napoleon_ml.dataset.manifest import DatasetManifest
from napoleon_ml.dataset.reader import load_manifest
from napoleon_ml.dataset.split import DatasetSplit, SplitConfig
from napoleon_ml.ownership.checkpoint import CheckpointCompatibilityError
from napoleon_ml.ownership.metrics import EvaluationReport


def add_split_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-ratio", type=int, default=80)
    parser.add_argument("--validation-ratio", type=int, default=10)
    parser.add_argument("--test-ratio", type=int, default=10)


def split_config_from_args(args: argparse.Namespace) -> SplitConfig:
    return SplitConfig(
        train=args.train_ratio,
        validation=args.validation_ratio,
        test=args.test_ratio,
    )


def dataset_split(value: str) -> DatasetSplit:
    try:
        return DatasetSplit(value)
    except ValueError as error:
        allowed = ", ".join(item.value for item in DatasetSplit)
        raise argparse.ArgumentTypeError(f"split must be one of {allowed}.") from error


def configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def load_checked_manifest(
    dataset_directory: Path, *, verify_integrity_label: str
) -> DatasetManifest:
    if not dataset_directory.exists():
        raise DatasetError(f"dataset directory does not exist: {dataset_directory}")

    manifest = load_manifest(dataset_directory)
    print(
        f"{verify_integrity_label}: dataset schema={manifest.dataset_schema_version}, "
        f"encoder schema={manifest.playing_encoder_schema_version}, "
        f"samples={manifest.sample_count}",
        file=sys.stderr,
    )
    return manifest


def print_report(report: EvaluationReport, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return

    print(f"split: {report.split}")
    print(f"samples: {report.sample_count}")
    print(f"masked_cards: {report.masked_card_count}")
    print("model:")
    print(f"  masked_loss: {_format_optional(report.masked_loss)}")
    print(f"  masked_accuracy: {_format_optional(report.masked_accuracy)}")
    print("  owner_class_accuracy:")
    for key, value in report.owner_class_accuracy.items():
        print(f"    {key}: {_format_optional(value)}")
    print("  progress_accuracy:")
    for key, value in report.progress_accuracy.items():
        print(f"    {key}: {_format_optional(value)}")
    print("baseline_always_not_in_hand:")
    print(f"  masked_accuracy: {_format_optional(report.baseline_masked_accuracy)}")
    print("  owner_class_accuracy:")
    for key, value in report.baseline_owner_class_accuracy.items():
        print(f"    {key}: {_format_optional(value)}")
    print("  progress_accuracy:")
    for key, value in report.baseline_progress_accuracy.items():
        print(f"    {key}: {_format_optional(value)}")


def handle_cli_error(error: Exception) -> int:
    if isinstance(error, DatasetError | CheckpointCompatibilityError | ValueError):
        print(f"error: {error}", file=sys.stderr)
        return 1

    raise error


def _format_optional(value: float | None) -> str:
    if value is None:
        return "n/a"

    return f"{value:.6f}"
