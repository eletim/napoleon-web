"""Train Issue #411 fixed-hand empirical Napoleon margin models."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from napoleon_ml.bidding_q.fixed_hand_margin_training import (
    FixedHandMarginCheckpointError,
    FixedHandMarginDatasetError,
    FixedHandMarginTrainConfig,
    load_fixed_hand_margin_dataset,
    save_fixed_hand_margin_artifact,
    train_fixed_hand_margin_model,
)
from napoleon_ml.cli._policy_common import parse_hidden_dims
from napoleon_ml.policy.device import TorchDeviceResolutionError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("--output-dir", "--output", dest="output_dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("M1", "M2"), default="M1")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", default="256,256,128")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--train-pairs", type=int)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--final-ratio", type=float, default=0.1)
    parser.add_argument("--std-loss-weight", type=float, default=0.25)
    parser.add_argument("--no-target-standardization", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=411)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (
        FixedHandMarginDatasetError,
        FixedHandMarginCheckpointError,
        TorchDeviceResolutionError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    dataset = load_fixed_hand_margin_dataset(args.dataset_directory)
    config = FixedHandMarginTrainConfig(
        variant=args.variant,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        dropout=args.dropout,
        train_pair_limit=args.train_pairs,
        validation_ratio=args.validation_ratio,
        final_ratio=args.final_ratio,
        target_standardization=not args.no_target_standardization,
        std_loss_weight=args.std_loss_weight,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
        init_checkpoint_path=str(args.init_checkpoint) if args.init_checkpoint is not None else None,
    )
    result = train_fixed_hand_margin_model(dataset, config)
    artifact = save_fixed_hand_margin_artifact(args.output_dir, result=result, dataset=dataset)
    report = {
        "outputDirectory": str(args.output_dir),
        "artifact": artifact,
        "dataset": {
            "path": str(args.dataset_directory),
            "samples": len(dataset.samples),
            "pairs": dataset.manifest.get("pairCount"),
            "rollouts": dataset.manifest.get("rolloutCount"),
            "teacher": dataset.manifest.get("teacher"),
        },
        "training": config.to_dict(),
        "split": {
            "trainHands": len(result.split.train_fixed_hand_ids),
            "validationHands": len(result.split.validation_fixed_hand_ids),
            "finalHands": len(result.split.final_fixed_hand_ids),
            "trainSamples": len(result.split.train_samples),
            "validationSamples": len(result.split.validation_samples),
            "finalSamples": len(result.split.final_samples),
        },
        "targetStandardization": result.target_standardization.to_dict(),
        "constantSigma": result.constant_sigma,
        "bestEpoch": result.best_epoch,
        "train": result.train_report,
        "validation": result.validation_report,
        "final": result.final_report,
        "epochs": result.epoch_reports,
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        final_mean = cast(dict[str, object], result.final_report["mean"])
        final_win = cast(dict[str, object], result.final_report["winProbability"])
        final_rank = cast(dict[str, object], result.final_report["sameHandRanking"])
        print(f"output: {args.output_dir}")
        print(f"checkpoint: {artifact['checkpointPath']}")
        print(f"variant: {config.variant}")
        print(f"best epoch: {result.best_epoch}")
        print(
            "train/validation/final samples: "
            f"{len(result.split.train_samples)} / "
            f"{len(result.split.validation_samples)} / "
            f"{len(result.split.final_samples)}"
        )
        print(f"final mean MAE: {final_mean['mae']}")
        print(f"final mean RMSE: {final_mean['rmse']}")
        print(f"final mean Pearson: {final_mean['pearson']}")
        print(f"final mean bias: {final_mean['bias']}")
        print(f"final pWin MAE: {final_win['mae']}")
        print(f"same-hand pairwise: {final_rank['pairwiseAccuracy']}")
        print(f"same-hand top-action: {final_rank['topActionAccuracy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
