"""Shared helpers for bidding baseline CLIs."""

from __future__ import annotations

import json
import sys

from napoleon_ml.bidding.checkpoint import BiddingCheckpointCompatibilityError
from napoleon_ml.bidding.metrics import BiddingEvaluationReport, BiddingMetric
from napoleon_ml.dataset.errors import DatasetError

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
    "print_bidding_report",
    "split_config_from_args",
]


def print_bidding_report(report: BiddingEvaluationReport, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return

    print(f"split: {report.split}")
    print(f"samples: {report.sample_count}")
    print("model:")
    print(f"  masked_loss: {_format_optional(report.masked_loss)}")
    print(f"  illegal_prediction_count: {report.illegal_prediction_count}")
    _print_metric("  top1", report.top1)
    _print_metric("  pass", report.pass_actions)
    _print_metric("  bid", report.bid_actions)
    print("  target_point_cards:")
    for key, metric in report.target_point_cards.items():
        _print_metric(f"    {key}", metric)
    print("  suits:")
    for key, metric in report.suits.items():
        _print_metric(f"    {key}", metric)
    print("baseline_legal_uniform:")
    print(f"  top1_accuracy: {_format_optional(report.top1.legal_uniform_accuracy)}")
    print(f"  pass_accuracy: {_format_optional(report.pass_actions.legal_uniform_accuracy)}")
    print(f"  bid_accuracy: {_format_optional(report.bid_actions.legal_uniform_accuracy)}")
    print("  target_point_cards_accuracy:")
    for key, metric in report.target_point_cards.items():
        print(f"    {key}: {_format_optional(metric.legal_uniform_accuracy)}")
    print("  suits_accuracy:")
    for key, metric in report.suits.items():
        print(f"    {key}: {_format_optional(metric.legal_uniform_accuracy)}")


def handle_cli_error(error: Exception) -> int:
    if isinstance(error, DatasetError | BiddingCheckpointCompatibilityError | ValueError):
        print(f"error: {error}", file=sys.stderr)
        return 1

    raise error


def _print_metric(label: str, metric: BiddingMetric) -> None:
    print(f"{label}:")
    print(f"    count: {metric.count}")
    print(f"    accuracy: {_format_optional(metric.accuracy)}")


def _format_optional(value: float | None) -> str:
    if value is None:
        return "n/a"

    return f"{value:.6f}"
