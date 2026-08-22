"""Train Issue #383 Citizen / Adjutant bidding value models."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from napoleon_ml.bidding_q import (
    BiddingRoleValueTrainConfig,
    load_bidding_q_dataset,
    role_value_coverage,
    save_bidding_role_value_artifact,
    train_bidding_role_value_model,
)
from napoleon_ml.bidding_q.dataset import BiddingQDatasetError
from napoleon_ml.bidding_q.role_value_training import BiddingRoleValueCheckpointError
from napoleon_ml.cli._policy_common import parse_hidden_dims
from napoleon_ml.policy.device import TorchDeviceResolutionError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("--output-dir", "--output", dest="output_dir", type=Path, required=True)
    parser.add_argument("--role", choices=("citizen", "adjutant"), required=True)
    parser.add_argument("--teacher")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", default="512,512,256,256")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--train-states", type=int, default=20000)
    parser.add_argument("--validation-state-keys", type=Path)
    parser.add_argument("--no-target-standardization", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=383)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--no-integrity-check", action="store_true")
    parser.add_argument("--coverage-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (
        BiddingQDatasetError,
        BiddingRoleValueCheckpointError,
        TorchDeviceResolutionError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    dataset = load_bidding_q_dataset(
        args.dataset_directory,
        verify_integrity=not args.no_integrity_check,
    )
    coverage = role_value_coverage(dataset.raw_samples)
    if args.coverage_only:
        print(json.dumps(coverage, indent=2, sort_keys=True))
        return 0
    config = BiddingRoleValueTrainConfig(
        role=args.role,
        teacher=args.teacher,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        dropout=args.dropout,
        train_state_count=args.train_states,
        validation_state_keys_path=(
            str(args.validation_state_keys) if args.validation_state_keys is not None else None
        ),
        target_standardization=not args.no_target_standardization,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
    )
    result = train_bidding_role_value_model(dataset, config)
    artifact = save_bidding_role_value_artifact(args.output_dir, result=result, dataset=dataset)
    report = {
        "outputDirectory": str(args.output_dir),
        "artifact": artifact,
        "dataset": {
            "path": str(args.dataset_directory),
            "sourceStates": dataset.manifest.source_states,
            "rawSampleCount": dataset.manifest.sample_count,
            "manifestSha256": dataset.manifest.manifest_sha256,
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
        "targetStandardization": result.standardization.to_dict(),
        "coverage": coverage,
        "bestEpoch": result.best_epoch,
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
        print(f"output: {args.output_dir}")
        print(f"checkpoint: {artifact['checkpointPath']}")
        print(f"role: {config.role}")
        print(f"teacher: {config.resolved_teacher()}")
        print(f"validation stateKey hash: {result.split.validation_state_key_hash}")
        print(f"best epoch: {result.best_epoch}")
        print(f"teacher samples: {validation['teacherSampleCount']}")
        print(f"teacher states: {validation['teacherStateCount']}")
        print(f"MAE: {regression['mae']}")
        print(f"RMSE: {regression['rmse']}")
        print(f"Pearson: {regression['pearsonCorrelation']}")
        print(f"ranking states: {ranking['rankingStateCount']}")
        print(f"pair count: {ranking['pairCount']}")
        print(f"different pair count: {ranking['differentPairCount']}")
        print(f"pairwise: {ranking['pairwiseAccuracy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
