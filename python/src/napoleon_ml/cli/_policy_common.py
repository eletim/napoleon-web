"""Shared helpers for policy baseline CLIs."""

from __future__ import annotations

import json
import sys

from napoleon_ml.dataset.errors import DatasetError
from napoleon_ml.policy.checkpoint import PolicyCheckpointCompatibilityError
from napoleon_ml.policy.metrics import PolicyEvaluationReport, PolicyMetric

from ._ownership_common import (
    add_split_config_arguments,
    configure_reproducibility,
    dataset_split,
    load_checked_manifest,
    split_config_from_args,
)

__all__ = [
    "add_split_config_arguments",
    "configure_reproducibility",
    "dataset_split",
    "handle_cli_error",
    "load_checked_manifest",
    "print_policy_report",
    "split_config_from_args",
]


def print_policy_report(report: PolicyEvaluationReport, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return

    print(f"split: {report.split}")
    print(f"samples: {report.sample_count}")
    print("model:")
    print(f"  masked_loss: {_format_optional(report.masked_loss)}")
    print(f"  illegal_prediction_count: {report.illegal_prediction_count}")
    _print_metric("  all_positions", report.all_positions)
    _print_metric("  forced_positions", report.forced_positions)
    _print_metric("  non_forced_positions", report.non_forced_positions)
    print("baseline_legal_uniform:")
    print(
        "  all_positions_accuracy: "
        f"{_format_optional(report.all_positions.legal_uniform_accuracy)}"
    )
    print(
        "  forced_positions_accuracy: "
        f"{_format_optional(report.forced_positions.legal_uniform_accuracy)}"
    )
    print(
        "  non_forced_positions_accuracy: "
        f"{_format_optional(report.non_forced_positions.legal_uniform_accuracy)}"
    )


def handle_cli_error(error: Exception) -> int:
    if isinstance(error, DatasetError | PolicyCheckpointCompatibilityError | ValueError):
        print(f"error: {error}", file=sys.stderr)
        return 1

    raise error


def _print_metric(label: str, metric: PolicyMetric) -> None:
    print(f"{label}:")
    print(f"    count: {metric.count}")
    print(f"    accuracy: {_format_optional(metric.accuracy)}")


def _format_optional(value: float | None) -> str:
    if value is None:
        return "n/a"

    return f"{value:.6f}"
