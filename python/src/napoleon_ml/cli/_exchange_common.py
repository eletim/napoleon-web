"""Shared helpers for exchange discard baseline CLIs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from napoleon_ml.dataset.constants import EXCHANGE_DATASET_SAMPLE_TYPE
from napoleon_ml.dataset.errors import DatasetError
from napoleon_ml.dataset.manifest import DatasetManifest
from napoleon_ml.dataset.reader import load_manifest
from napoleon_ml.exchange.checkpoint import ExchangeCheckpointCompatibilityError
from napoleon_ml.exchange.metrics import ExchangeEvaluationReport

from ._ownership_common import (
    add_split_config_arguments,
    configure_reproducibility,
    dataset_split,
    split_config_from_args,
)

__all__ = [
    "add_split_config_arguments",
    "configure_reproducibility",
    "dataset_split",
    "handle_cli_error",
    "load_checked_exchange_manifest",
    "print_exchange_report",
    "split_config_from_args",
]


def load_checked_exchange_manifest(
    dataset_directory: Path, *, command_label: str
) -> DatasetManifest:
    if not dataset_directory.exists():
        raise DatasetError(f"dataset directory does not exist: {dataset_directory}")

    manifest = load_manifest(dataset_directory)
    if manifest.sample_type != EXCHANGE_DATASET_SAMPLE_TYPE:
        raise DatasetError(
            f"{command_label} requires an {EXCHANGE_DATASET_SAMPLE_TYPE} dataset, "
            f"got {manifest.sample_type}."
        )

    print(
        f"{command_label}: dataset schema={manifest.dataset_schema_version}, "
        f"encoder schema={manifest.encoder_schema_version}, samples={manifest.sample_count}",
        file=sys.stderr,
    )
    return manifest


def print_exchange_report(report: ExchangeEvaluationReport, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return

    print(f"split: {report.split}")
    print(f"samples: {report.sample_count}")
    print("model:")
    print(f"  masked_loss: {_format_optional(report.masked_loss)}")
    print(f"  exact_set_accuracy: {_format_optional(report.exact_set_accuracy)}")
    print(f"  average_overlap_count: {_format_optional(report.average_overlap_count)}")
    print(f"  selected_teacher_match_rate: {_format_optional(report.selected_teacher_match_rate)}")
    print(f"  illegal_selected_card_count: {report.illegal_selected_card_count}")
    print("baseline_random_legal:")
    print(
        "  exact_set_accuracy: "
        f"{_format_optional(report.baseline_random_legal_exact_set_accuracy)}"
    )
    print(
        "  average_overlap_count: "
        f"{_format_optional(report.baseline_random_legal_average_overlap_count)}"
    )
    print(
        "  selected_teacher_match_rate: "
        f"{_format_optional(report.baseline_random_legal_selected_teacher_match_rate)}"
    )


def handle_cli_error(error: Exception) -> int:
    if isinstance(error, DatasetError | ExchangeCheckpointCompatibilityError | ValueError):
        print(f"error: {error}", file=sys.stderr)
        return 1

    raise error


def _format_optional(value: float | None) -> str:
    if value is None:
        return "n/a"

    return f"{value:.6f}"
