"""Export a saved policy MLP checkpoint to ONNX."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from napoleon_ml.adjutant.checkpoint import AdjutantCheckpointCompatibilityError
from napoleon_ml.bidding.checkpoint import BiddingCheckpointCompatibilityError
from napoleon_ml.dataset.errors import DatasetError
from napoleon_ml.exchange.checkpoint import ExchangeCheckpointCompatibilityError
from napoleon_ml.nonplaying_onnx_export import (
    NonPlayingOnnxExportError,
    PolicyType,
    export_nonplaying_checkpoint_to_onnx,
)
from napoleon_ml.policy.checkpoint import PolicyCheckpointCompatibilityError
from napoleon_ml.policy.onnx_export import export_policy_checkpoint_to_onnx

from ._policy_common import load_checked_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument(
        "--policy-type",
        choices=("playing", "bidding", "exchange", "adjutant"),
        default="playing",
        help="Checkpoint/export contract to use. Defaults to the existing playing exporter.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metadata-output",
        type=Path,
        help="JSON metadata output path. Defaults to the ONNX path with .json suffix.",
    )
    parser.add_argument("--no-integrity-check", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return _run(args)
    except Exception as error:
        return _handle_export_cli_error(error)


def _run(args: argparse.Namespace) -> int:
    metadata_output = args.metadata_output or args.output.with_suffix(".json")
    manifest = load_checked_manifest(args.dataset_directory, command_label="export-policy-onnx")
    if args.policy_type == "playing":
        report = export_policy_checkpoint_to_onnx(
            dataset_directory=args.dataset_directory,
            checkpoint_path=args.checkpoint,
            onnx_path=args.output,
            metadata_path=metadata_output,
            manifest=manifest,
            verify_integrity=not args.no_integrity_check,
        )

        if args.json:
            print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            print(f"onnx: {report.onnx_path}")
            print(f"metadata: {report.metadata_path}")
            print("parity_sample:")
            print(f"  seed: {report.sample_seed}")
            print(f"  step: {report.sample_step}")
            print(f"  max_abs_logit_diff: {report.max_abs_logit_diff:.8f}")
            print(f"  pytorch_selected_card_index: {report.pytorch_selected_card_index}")
            print(f"  onnx_selected_card_index: {report.onnx_selected_card_index}")

        return 0

    nonplaying_report = export_nonplaying_checkpoint_to_onnx(
        policy_type=cast(PolicyType, args.policy_type),
        dataset_directory=args.dataset_directory,
        checkpoint_path=args.checkpoint,
        onnx_path=args.output,
        metadata_path=metadata_output,
        manifest=manifest,
        verify_integrity=not args.no_integrity_check,
    )

    if args.json:
        print(json.dumps(nonplaying_report.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"onnx: {nonplaying_report.onnx_path}")
        print(f"metadata: {nonplaying_report.metadata_path}")
        print("parity_sample:")
        print(f"  seed: {nonplaying_report.sample_seed}")
        print(f"  step: {nonplaying_report.sample_step}")
        print(f"  max_abs_logit_diff: {nonplaying_report.max_abs_logit_diff:.8f}")
        print(f"  pytorch_selection: {nonplaying_report.pytorch_selection}")
        print(f"  onnx_selection: {nonplaying_report.onnx_selection}")

    return 0


def _handle_export_cli_error(error: Exception) -> int:
    if isinstance(
        error,
        DatasetError
        | PolicyCheckpointCompatibilityError
        | BiddingCheckpointCompatibilityError
        | ExchangeCheckpointCompatibilityError
        | AdjutantCheckpointCompatibilityError
        | NonPlayingOnnxExportError
        | ValueError,
    ):
        print(f"error: {error}", file=sys.stderr)
        return 1

    raise error


if __name__ == "__main__":
    raise SystemExit(main())
