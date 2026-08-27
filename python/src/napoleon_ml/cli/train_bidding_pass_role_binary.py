"""Train Issue #395 PASS Adjutant-vs-Citizen binary classifier."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from napoleon_ml.bidding_q import (
    BiddingPassRoleBinaryTrainConfig,
    audit_pass_binary_dataset,
    load_bidding_q_dataset,
    save_bidding_pass_role_binary_artifact,
    train_bidding_pass_role_binary_model,
)
from napoleon_ml.bidding_q.dataset import BiddingQDatasetError
from napoleon_ml.bidding_q.pass_role_binary_training import (
    BiddingPassRoleBinaryCheckpointError,
)
from napoleon_ml.cli._policy_common import parse_hidden_dims
from napoleon_ml.policy.device import TorchDeviceResolutionError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("--output-dir", "--output", dest="output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", default="512,512,256,256")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--pos-weight", type=float)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=395)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--no-integrity-check", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (
        BiddingQDatasetError,
        BiddingPassRoleBinaryCheckpointError,
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
    audit = audit_pass_binary_dataset(dataset.raw_samples)
    if args.audit_only:
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 0
    config = BiddingPassRoleBinaryTrainConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        dropout=args.dropout,
        train_ratio=args.train_ratio,
        pos_weight=args.pos_weight,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
    )
    result = train_bidding_pass_role_binary_model(dataset, config)
    artifact = save_bidding_pass_role_binary_artifact(
        args.output_dir, result=result, dataset=dataset
    )
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
            "calibrationStates": len(result.split.calibration_state_keys),
            "trainRawPassSamples": len(result.split.train_samples),
            "calibrationRawPassSamples": len(result.split.calibration_samples),
            "calibrationStateKeyHash": result.split.state_key_hash,
            "stateKeyLeakage": False,
        },
        "audit": audit,
        "posWeight": result.pos_weight,
        "bestEpoch": result.best_epoch,
        "epochs": result.epoch_reports,
        "train": result.train_report,
        "calibration": result.calibration_report,
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"output: {args.output_dir}")
        print(f"checkpoint: {artifact['checkpointPath']}")
        print(f"best epoch: {result.best_epoch}")
        print(f"PASS samples: {audit['passSamplesTotal']}")
        print(f"Citizen/Adjutant labeled: {audit['labeledCitizenAdjutantCount']}")
        print(f"NoContract rate: {audit['noContractRate']}")
        print(f"calibration ROC-AUC: {result.calibration_report['rocAuc']}")
        print(f"calibration PR-AUC: {result.calibration_report['prAuc']}")
        print(f"calibration Adj F1: {result.calibration_report['adjutantF1']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
