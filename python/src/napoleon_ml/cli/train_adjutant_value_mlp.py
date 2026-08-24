"""Train the Issue #446 compact290 adjutant downstream value MLP."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from napoleon_ml.adjutant_value import train_adjutant_value_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--full-gold-directory", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", default="512,512,256,256")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=446)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    hidden_dims = tuple(int(item) for item in args.hidden_dims.split(",") if item)
    result = train_adjutant_value_model(
        args.dataset_directory,
        output_directory=args.output_directory,
        full_gold_directory=args.full_gold_directory,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        huber_delta=args.huber_delta,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
