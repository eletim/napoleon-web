"""Train the Issue #378 multi-head bidding value model."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from napoleon_ml.bidding_q import (
    BiddingMultiHeadQTrainConfig,
    load_bidding_q_dataset,
    save_bidding_multi_head_q_artifact,
    train_bidding_multi_head_q_model,
)
from napoleon_ml.bidding_q.dataset import BiddingQDatasetError
from napoleon_ml.bidding_q.multi_head_training import BiddingMultiHeadQCheckpointError
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
    parser.add_argument("--role-loss-weight", type=float, default=1.0)
    parser.add_argument("--napoleon-point-loss-weight", type=float, default=1.0)
    parser.add_argument("--success-loss-weight", type=float, default=1.0)
    parser.add_argument("--margin-loss-weight", type=float, default=1.0)
    parser.add_argument("--no-regression-standardization", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=378)
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
        BiddingMultiHeadQCheckpointError,
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
    config = BiddingMultiHeadQTrainConfig(
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
        role_loss_weight=args.role_loss_weight,
        napoleon_point_loss_weight=args.napoleon_point_loss_weight,
        success_loss_weight=args.success_loss_weight,
        margin_loss_weight=args.margin_loss_weight,
        regression_standardization=not args.no_regression_standardization,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
    )
    result = train_bidding_multi_head_q_model(dataset, config)
    artifact = save_bidding_multi_head_q_artifact(args.output_dir, result=result, dataset=dataset)
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
        "bestEpoch": result.best_epoch,
        "teacherSignal": result.teacher_signal,
        "train": result.train_report,
        "validation": result.validation_report,
        "epochs": result.epoch_reports,
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        validation = result.validation_report
        role = cast(dict[str, object], validation["role"])
        napoleon = cast(dict[str, object], validation["napoleonSidePointCards"])
        napoleon_overall = cast(dict[str, object], napoleon["overall"])
        success = cast(dict[str, object], validation["contractSuccess"])
        margin = cast(dict[str, object], validation["contractMargin"])
        margin_overall = cast(dict[str, object], margin["overall"])
        ranking = cast(dict[str, object], validation["ranking"])
        nap_ranking = cast(dict[str, object], ranking["napoleonSideCards"])
        print(f"output: {args.output_dir}")
        print(f"checkpoint: {artifact['checkpointPath']}")
        print(f"train states: {len(result.split.train_state_keys)}")
        print(f"validation states: {len(result.split.validation_state_keys)}")
        print(f"validation stateKey hash: {result.split.validation_state_key_hash}")
        print(f"best epoch: {result.best_epoch}")
        print(f"role accuracy: {role['accuracy']}")
        print(f"role macro F1: {role['macroF1']}")
        print(f"Napoleon-side MAE: {napoleon_overall['mae']}")
        print(f"Napoleon-side RMSE: {napoleon_overall['rmse']}")
        print(f"Napoleon-side Pearson: {napoleon_overall['pearsonCorrelation']}")
        print(f"success accuracy: {success['accuracy']}")
        print(f"success F1: {success['f1']}")
        print(f"success ROC-AUC: {success['rocAuc']}")
        print(f"margin MAE: {margin_overall['mae']}")
        print(f"margin RMSE: {margin_overall['rmse']}")
        print(f"margin Pearson: {margin_overall['pearsonCorrelation']}")
        print(f"margin sign accuracy: {margin_overall['signAccuracy']}")
        print(f"Napoleon score best-action hit: {nap_ranking['bestActionHitRate']}")
        print(f"Napoleon score pairwise: {nap_ranking['pairwiseRankingAccuracy']}")
        print(f"Napoleon score strongest suit match: {nap_ranking['strongestSuitMatchRate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
