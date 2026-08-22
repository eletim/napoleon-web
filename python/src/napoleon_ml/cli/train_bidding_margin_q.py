"""Train the Issue #380 heteroscedastic contract-margin model."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from napoleon_ml.bidding_q import (
    BiddingMarginTrainConfig,
    load_bidding_q_dataset,
    save_bidding_margin_artifact,
    train_bidding_margin_model,
)
from napoleon_ml.bidding_q.dataset import BiddingQDatasetError
from napoleon_ml.bidding_q.margin_training import BiddingMarginCheckpointError
from napoleon_ml.cli._policy_common import parse_hidden_dims
from napoleon_ml.policy.device import TorchDeviceResolutionError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("--output-dir", "--output", dest="output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", default="512,512,256,256")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--train-states", type=int, default=20000)
    parser.add_argument("--validation-state-keys", type=Path)
    parser.add_argument("--no-target-standardization", action="store_true")
    parser.add_argument("--log-variance-min", type=float, default=-6.0)
    parser.add_argument("--log-variance-max", type=float, default=5.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=380)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
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
        BiddingMarginCheckpointError,
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
    config = BiddingMarginTrainConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        train_state_count=args.train_states,
        validation_state_keys_path=(
            str(args.validation_state_keys) if args.validation_state_keys is not None else None
        ),
        target_standardization=not args.no_target_standardization,
        log_variance_min=args.log_variance_min,
        log_variance_max=args.log_variance_max,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
    )
    result = train_bidding_margin_model(dataset, config)
    artifact = save_bidding_margin_artifact(args.output_dir, result=result, dataset=dataset)
    report = {
        "outputDirectory": str(args.output_dir),
        "artifact": artifact,
        "dataset": {
            "path": str(args.dataset_directory),
            "sourceStates": dataset.manifest.source_states,
            "rawSampleCount": dataset.manifest.sample_count,
            "manifestSha256": dataset.manifest.manifest_sha256,
            "schemaVersion": dataset.manifest.dataset_schema_version,
            "sampleSchemaVersion": dataset.manifest.sample_schema_version,
        },
        "training": config.to_dict(),
        "device": result.device.to_metadata(),
        "split": {
            "trainStates": len(result.split.train_state_keys),
            "validationStates": len(result.split.validation_state_keys),
            "trainRawSamples": len(result.split.train_samples),
            "validationRawSamples": len(result.split.validation_samples),
            "validationStateKeyHash": result.split.validation_state_key_hash,
        },
        "targetStandardization": result.target_standardization.to_dict(),
        "bestEpoch": result.best_epoch,
        "train": result.train_report,
        "validation": result.validation_report,
        "epochs": result.epoch_reports,
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        validation = result.validation_report
        mean = cast(dict[str, object], validation["mean"])
        mean_overall = cast(dict[str, object], mean["overall"])
        sigma = cast(dict[str, object], validation["sigma"])
        sharpness = cast(dict[str, object], sigma["sharpness"])
        sharpness_overall = cast(dict[str, object], sharpness["overall"])
        residual_correlation = cast(dict[str, object], sigma["residualCorrelation"])
        success = cast(dict[str, object], validation["gaussianSuccess"])
        risk = cast(dict[str, object], validation["riskAwareRanking"])
        risk_zero = cast(dict[str, object], risk["0.0"])
        print(f"output: {args.output_dir}")
        print(f"checkpoint: {artifact['checkpointPath']}")
        print(f"onnx: {artifact['onnxPath']}")
        print(f"train states: {len(result.split.train_state_keys)}")
        print(f"validation states: {len(result.split.validation_state_keys)}")
        print(f"validation stateKey hash: {result.split.validation_state_key_hash}")
        print(f"best epoch: {result.best_epoch}")
        print(f"mean MAE: {mean_overall['mae']}")
        print(f"mean RMSE: {mean_overall['rmse']}")
        print(f"mean Pearson: {mean_overall['pearsonCorrelation']}")
        print(f"mean sign accuracy: {mean_overall['signAccuracy']}")
        print(f"sigma mean: {sharpness_overall['mean']}")
        print(f"sigma std: {sharpness_overall['std']}")
        print(f"sigma/residual Pearson: {residual_correlation['pearson']}")
        print(f"Gaussian success ROC-AUC: {success['rocAuc']}")
        print(f"Gaussian success PR-AUC: {success['prAuc']}")
        print(f"lambda=0 pairwise: {risk_zero['pairwiseRankingAccuracy']}")
        print(f"ONNX parity: {artifact['onnxParity']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
