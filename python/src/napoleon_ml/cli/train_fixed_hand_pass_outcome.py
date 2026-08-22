"""Train Issue #414 fixed-hand PASS empirical-q and role margin models."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from napoleon_ml.bidding_q.fixed_hand_margin_training import (
    FixedHandMarginCheckpointError,
    FixedHandMarginTrainConfig,
    save_fixed_hand_margin_artifact,
)
from napoleon_ml.bidding_q.pass_outcome_training import (
    EmpiricalQTrainConfig,
    PassOutcomeCheckpointError,
    PassOutcomeDatasetError,
    load_pass_outcome_dataset,
    pass_role_margin_dataset,
    save_empirical_q_artifact,
    train_empirical_q_model,
    train_pass_role_margin_model,
)
from napoleon_ml.cli._policy_common import parse_hidden_dims
from napoleon_ml.policy.device import TorchDeviceResolutionError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("--output-dir", "--output", dest="output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--q-epochs", type=int)
    parser.add_argument("--margin-epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", default="256,256,128")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--train-hands", type=int)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--final-ratio", type=float, default=0.1)
    parser.add_argument("--min-q-denominator", type=int, default=1)
    parser.add_argument("--min-role-count", type=int, default=2)
    parser.add_argument("--std-loss-weight", type=float, default=0.25)
    parser.add_argument("--no-target-standardization", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=414)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (
        FixedHandMarginCheckpointError,
        PassOutcomeCheckpointError,
        PassOutcomeDatasetError,
        TorchDeviceResolutionError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    dataset = load_pass_outcome_dataset(args.dataset_directory)
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    q_config = EmpiricalQTrainConfig(
        seed=args.seed,
        epochs=args.q_epochs or args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        validation_ratio=args.validation_ratio,
        final_ratio=args.final_ratio,
        train_hand_limit=args.train_hands,
        min_q_denominator=args.min_q_denominator,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
    )
    margin_config = FixedHandMarginTrainConfig(
        variant="M2",
        seed=args.seed,
        epochs=args.margin_epochs or args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        train_pair_limit=args.train_hands,
        validation_ratio=args.validation_ratio,
        final_ratio=args.final_ratio,
        target_standardization=not args.no_target_standardization,
        std_loss_weight=args.std_loss_weight,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
    )

    output = args.output_dir
    q_result = train_empirical_q_model(dataset, q_config)
    q_artifact = save_empirical_q_artifact(output / "q", result=q_result, dataset=dataset)

    citizen_dataset = pass_role_margin_dataset(
        dataset,
        role="citizen",
        min_role_count=args.min_role_count,
    )
    citizen_result = train_pass_role_margin_model(
        dataset,
        role="citizen",
        min_role_count=args.min_role_count,
        config=margin_config,
    )
    citizen_artifact = save_fixed_hand_margin_artifact(
        output / "citizen-margin",
        result=citizen_result,
        dataset=citizen_dataset,
    )

    adjutant_dataset = pass_role_margin_dataset(
        dataset,
        role="adjutant",
        min_role_count=args.min_role_count,
    )
    adjutant_result = train_pass_role_margin_model(
        dataset,
        role="adjutant",
        min_role_count=args.min_role_count,
        config=margin_config,
    )
    adjutant_artifact = save_fixed_hand_margin_artifact(
        output / "adjutant-margin",
        result=adjutant_result,
        dataset=adjutant_dataset,
    )

    report = {
        "outputDirectory": str(output),
        "dataset": {
            "path": str(args.dataset_directory),
            "samples": len(dataset.samples),
            "rollouts": dataset.manifest.get("rolloutCount"),
            "teacher": dataset.manifest.get("teacher"),
        },
        "artifacts": {
            "q": q_artifact,
            "citizenMargin": citizen_artifact,
            "adjutantMargin": adjutant_artifact,
        },
        "q": {
            "bestEpoch": q_result.best_epoch,
            "train": q_result.train_report,
            "validation": q_result.validation_report,
            "final": q_result.final_report,
        },
        "citizenMargin": {
            "samples": len(citizen_dataset.samples),
            "bestEpoch": citizen_result.best_epoch,
            "train": citizen_result.train_report,
            "validation": citizen_result.validation_report,
            "final": citizen_result.final_report,
        },
        "adjutantMargin": {
            "samples": len(adjutant_dataset.samples),
            "bestEpoch": adjutant_result.best_epoch,
            "train": adjutant_result.train_report,
            "validation": adjutant_result.validation_report,
            "final": adjutant_result.final_report,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "combined-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"output: {output}")
        print(f"q checkpoint: {q_artifact['checkpointPath']}")
        print(f"citizen checkpoint: {citizen_artifact['checkpointPath']}")
        print(f"adjutant checkpoint: {adjutant_artifact['checkpointPath']}")
        print(f"q final RMSE: {q_result.final_report['rmse']}")
        print(f"citizen final MAE: {citizen_result.final_report['mean']['mae']}")
        print(f"adjutant final MAE: {adjutant_result.final_report['mean']['mae']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
