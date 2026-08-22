"""Evaluate Issue #395 PASS-binary simple offline bidding scores."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from napoleon_ml.bidding_q.dataset import BiddingQDatasetError
from napoleon_ml.bidding_q.margin_training import BiddingMarginCheckpointError
from napoleon_ml.bidding_q.pass_role_binary_training import (
    BiddingPassRoleBinaryCheckpointError,
)
from napoleon_ml.bidding_q.pass_simple_score import evaluate_pass_simple_score_offline
from napoleon_ml.bidding_q.role_value_training import BiddingRoleValueCheckpointError
from napoleon_ml.policy.device import TorchDeviceResolutionError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("--pass-checkpoint", type=Path, required=True)
    parser.add_argument("--margin-checkpoint", type=Path, required=True)
    parser.add_argument("--citizen-checkpoint", type=Path, required=True)
    parser.add_argument("--adjutant-checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-dataset", type=Path, required=True)
    parser.add_argument("--issue390-report", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--split-seed", type=int, default=390)
    parser.add_argument("--all-states", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--no-integrity-check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = evaluate_pass_simple_score_offline(
            dataset_directory=args.dataset_directory,
            pass_checkpoint_path=args.pass_checkpoint,
            margin_checkpoint_path=args.margin_checkpoint,
            citizen_checkpoint_path=args.citizen_checkpoint,
            adjutant_checkpoint_path=args.adjutant_checkpoint,
            calibration_dataset_directory=args.calibration_dataset,
            issue390_report_path=args.issue390_report,
            device=args.device,
            split_seed=None if args.all_states else args.split_seed,
            verify_integrity=not args.no_integrity_check,
        )
    except (
        BiddingQDatasetError,
        BiddingPassRoleBinaryCheckpointError,
        BiddingMarginCheckpointError,
        BiddingRoleValueCheckpointError,
        TorchDeviceResolutionError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    content = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(content, encoding="utf-8")
    print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
