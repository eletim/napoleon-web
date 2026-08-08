"""Shared helpers for adjutant baseline CLIs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from napoleon_ml.adjutant.checkpoint import AdjutantCheckpointCompatibilityError
from napoleon_ml.adjutant.metrics import AdjutantEvaluationReport, AdjutantMetric
from napoleon_ml.dataset.constants import ADJUTANT_DATASET_SAMPLE_TYPE
from napoleon_ml.dataset.errors import DatasetError
from napoleon_ml.dataset.manifest import DatasetManifest
from napoleon_ml.dataset.reader import load_manifest

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
    "load_checked_adjutant_manifest",
    "print_adjutant_report",
    "split_config_from_args",
]


def load_checked_adjutant_manifest(
    dataset_directory: Path, *, command_label: str
) -> DatasetManifest:
    if not dataset_directory.exists():
        raise DatasetError(f"dataset directory does not exist: {dataset_directory}")

    manifest = load_manifest(dataset_directory)
    if manifest.sample_type != ADJUTANT_DATASET_SAMPLE_TYPE:
        raise DatasetError(
            f"{command_label} requires {ADJUTANT_DATASET_SAMPLE_TYPE}, "
            f"got {manifest.sample_type}."
        )

    print(
        f"{command_label}: dataset schema={manifest.dataset_schema_version}, "
        f"encoder schema={manifest.encoder_schema_version}, samples={manifest.sample_count}",
        file=sys.stderr,
    )
    return manifest


def print_adjutant_report(report: AdjutantEvaluationReport, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return

    print(f"split: {report.split}")
    print(f"samples: {report.sample_count}")
    print("model:")
    print(f"  masked_loss: {_format_optional(report.masked_loss)}")
    print(f"  illegal_prediction_count: {report.illegal_prediction_count}")
    _print_metric("  top1", report.top1)
    print("  by_category:")
    for label, metric in report.by_category.items():
        _print_metric(f"    {label}", metric)
    print("baseline_legal_uniform:")
    print(f"  top1_accuracy: {_format_optional(report.top1.legal_uniform_accuracy)}")
    print("  by_category_accuracy:")
    for label, metric in report.by_category.items():
        print(f"    {label}: {_format_optional(metric.legal_uniform_accuracy)}")


def handle_cli_error(error: Exception) -> int:
    if isinstance(error, DatasetError | AdjutantCheckpointCompatibilityError | ValueError):
        print(f"error: {error}", file=sys.stderr)
        return 1

    raise error


def _print_metric(label: str, metric: AdjutantMetric) -> None:
    print(f"{label}:")
    print(f"    count: {metric.count}")
    print(f"    accuracy: {_format_optional(metric.accuracy)}")


def _format_optional(value: float | None) -> str:
    if value is None:
        return "n/a"

    return f"{value:.6f}"
