"""Train an exchange discard-combination scalar value model."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from napoleon_ml.exchange_value import (
    EXCHANGE_VALUE_INPUT_VARIANTS,
    ExchangeValueTrainConfig,
    load_exchange_counterfactual_dataset,
    save_exchange_value_artifact,
    train_exchange_value_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", default="512,512,256,256")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--input-variant",
        choices=EXCHANGE_VALUE_INPUT_VARIANTS,
        default="legacy2724",
    )
    parser.add_argument("--seed", type=int, default=436)
    parser.add_argument("--train-state-count", type=int)
    parser.add_argument("--loss", choices=("mse", "huber"), default="mse")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--pairwise-loss-weight", type=float, default=0.0)
    parser.add_argument("--pointwise-loss-weight", type=float, default=1.0)
    parser.add_argument("--listwise-loss-weight", type=float, default=0.0)
    parser.add_argument("--listwise-temperature", type=float, default=1.0)
    parser.add_argument("--pairwise-state-batch-size", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--warm-start-checkpoint", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    hidden_dims = tuple(int(item) for item in args.hidden_dims.split(",") if item)
    dataset = load_exchange_counterfactual_dataset(args.dataset_directory)
    result = train_exchange_value_model(
        dataset,
        ExchangeValueTrainConfig(
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_dims=hidden_dims,
            dropout=args.dropout,
            input_variant=args.input_variant,
            train_state_count=args.train_state_count,
            loss=args.loss,
            huber_delta=args.huber_delta,
            pairwise_loss_weight=args.pairwise_loss_weight,
            pointwise_loss_weight=args.pointwise_loss_weight,
            listwise_loss_weight=args.listwise_loss_weight,
            listwise_temperature=args.listwise_temperature,
            pairwise_state_batch_size=args.pairwise_state_batch_size,
            weight_decay=args.weight_decay,
            patience=args.patience,
            device=args.device,
            warm_start_checkpoint=(
                str(args.warm_start_checkpoint) if args.warm_start_checkpoint is not None else None
            ),
        ),
    )
    artifact = save_exchange_value_artifact(
        args.output_directory,
        result=result,
        dataset=dataset,
    )
    output = {
        "artifact": artifact,
        "bestEpoch": result.best_epoch,
        "bestMonitorValue": result.best_monitor_value,
        "train": result.train_report,
        "validation": result.validation_report,
        "final": result.final_report,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
