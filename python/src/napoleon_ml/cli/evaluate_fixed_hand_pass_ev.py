"""Evaluate Issue #414 offline PASS/BID EV variants."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from napoleon_ml.bidding_q.pass_outcome_training import (
    PassOutcomeCheckpointError,
    PassOutcomeDatasetError,
    evaluate_pass_ev_variants,
    load_pass_outcome_dataset,
)
from napoleon_ml.policy.device import TorchDeviceResolutionError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pass_dataset_directory", type=Path)
    parser.add_argument("--bid-dataset", type=Path, required=True)
    parser.add_argument("--napoleon-old-checkpoint", type=Path, required=True)
    parser.add_argument("--napoleon-m2-checkpoint", type=Path, required=True)
    parser.add_argument("--citizen-old-checkpoint", type=Path, required=True)
    parser.add_argument("--adjutant-old-checkpoint", type=Path, required=True)
    parser.add_argument("--q-old-checkpoint", type=Path, required=True)
    parser.add_argument("--citizen-new-checkpoint", type=Path, required=True)
    parser.add_argument("--adjutant-new-checkpoint", type=Path, required=True)
    parser.add_argument("--q-new-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=414)
    parser.add_argument("--pass-reward-d", type=float, default=13.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (
        PassOutcomeCheckpointError,
        PassOutcomeDatasetError,
        TorchDeviceResolutionError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    dataset = load_pass_outcome_dataset(args.pass_dataset_directory)
    report = evaluate_pass_ev_variants(
        pass_dataset=dataset,
        bid_dataset_directory=args.bid_dataset,
        napoleon_old_checkpoint=args.napoleon_old_checkpoint,
        napoleon_m2_checkpoint=args.napoleon_m2_checkpoint,
        citizen_old_checkpoint=args.citizen_old_checkpoint,
        adjutant_old_checkpoint=args.adjutant_old_checkpoint,
        q_old_checkpoint=args.q_old_checkpoint,
        citizen_new_checkpoint=args.citizen_new_checkpoint,
        adjutant_new_checkpoint=args.adjutant_new_checkpoint,
        q_new_checkpoint=args.q_new_checkpoint,
        device=args.device,
        seed=args.seed,
        pass_reward_d=args.pass_reward_d,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
