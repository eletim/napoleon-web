"""Train a 29-output Monte Carlo Q network from Issue #368 bidding counterfactual data."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from napoleon_ml.bidding_q import (
    BiddingQTrainConfig,
    load_bidding_q_dataset,
    save_bidding_q_artifact,
    train_bidding_q_model,
)
from napoleon_ml.bidding_q.dataset import BiddingQDatasetError
from napoleon_ml.bidding_q.training import BiddingQCheckpointError
from napoleon_ml.cli._policy_common import parse_hidden_dims
from napoleon_ml.policy.device import TorchDeviceResolutionError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("--output-dir", "--output", dest="output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", default="512,512,256,256")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--loss-type", choices=("huber", "mse"), default="huber")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=370)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--target-normalization",
        choices=("none", "standardize"),
        default="none",
    )
    parser.add_argument("--no-integrity-check", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (
        BiddingQDatasetError,
        BiddingQCheckpointError,
        TorchDeviceResolutionError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    dataset = load_bidding_q_dataset(
        args.dataset_directory,
        verify_integrity=not args.no_integrity_check,
    )
    config = BiddingQTrainConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        loss_type=args.loss_type,
        train_ratio=args.train_ratio,
        device=args.device,
        target_normalization=args.target_normalization,
    )
    result = train_bidding_q_model(dataset, config)
    artifact = save_bidding_q_artifact(args.output_dir, result=result, dataset=dataset)
    report = {
        "outputDirectory": str(args.output_dir),
        "artifact": artifact,
        "dataset": {
            "path": str(args.dataset_directory),
            "sourceStates": dataset.manifest.source_states,
            "rawSampleCount": dataset.manifest.sample_count,
            "aggregatedSampleCount": len(dataset.examples),
            "rewardId": dataset.manifest.reward_id,
            "rewardVersion": dataset.manifest.reward_version,
            "manifestSha256": dataset.manifest.manifest_sha256,
        },
        "training": config.to_dict(),
        "device": result.device.to_metadata(),
        "split": {
            "trainStates": len(result.split.train_state_keys),
            "validationStates": len(result.split.validation_state_keys),
            "trainSamples": len(result.split.train_examples),
            "validationSamples": len(result.split.validation_examples),
        },
        "epochs": result.epoch_reports,
        "train": result.train_report,
        "validation": result.validation_report,
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        validation = result.validation_report
        regression = cast(dict[str, object], validation["regression"])
        ranking = cast(dict[str, object], validation["ranking"])
        collapse = cast(dict[str, object], validation["qCollapseDiagnostics"])
        parity = cast(dict[str, object], artifact["onnxParity"])
        print(f"output: {args.output_dir}")
        print(f"checkpoint: {artifact['checkpointPath']}")
        print(f"onnx: {artifact['onnxPath']}")
        print(f"train states: {len(result.split.train_state_keys)}")
        print(f"validation states: {len(result.split.validation_state_keys)}")
        print(f"validation MAE: {regression['mae']}")
        print(f"validation RMSE: {regression['rmse']}")
        print(f"validation Pearson: {regression['pearsonCorrelation']}")
        print(f"best-action hit: {ranking['bestActionHitRate']}")
        print(f"top3 hit: {ranking['top3HitRate']}")
        print(f"pairwise ranking: {ranking['pairwiseRankingAccuracy']}")
        print(f"PASS vs best-bid: {ranking['passVsBestBidAccuracy']}")
        print(f"predicted PASS rate: {collapse['predictedPassRate']}")
        print(f"strongest suit match: {collapse['strongestSuitMatchRate']}")
        print(f"ONNX parity max abs diff: {parity['maxAbsQDiff']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
