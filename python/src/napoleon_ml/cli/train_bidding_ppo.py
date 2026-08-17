"""Train a bidding Actor-Critic policy with PPO on non-playing RL samples."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from napoleon_ml.bidding.model import BiddingMlpConfig
from napoleon_ml.bidding.ppo import (
    DEFAULT_ADVANTAGE_NORMALIZATION,
    SUPPORTED_ADVANTAGE_NORMALIZATIONS,
    BiddingPpoCompatibilityError,
    BiddingPpoTrainSettings,
    train_bidding_ppo,
)
from napoleon_ml.cli._policy_common import parse_hidden_dims
from napoleon_ml.policy.device import SUPPORTED_TORCH_DEVICES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument(
        "--bidding-hidden-dims",
        "--hidden-dims",
        dest="hidden_dims",
        help=(
            "Comma-separated bidding Actor/Critic hidden widths. "
            "When set, this overrides --hidden-dim/--hidden-layers for bidding."
        ),
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--training-device", choices=SUPPORTED_TORCH_DEVICES, default="cpu")
    parser.add_argument("--ppo-clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-loss-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.0)
    parser.add_argument(
        "--advantage-normalization",
        choices=SUPPORTED_ADVANTAGE_NORMALIZATIONS,
        default=DEFAULT_ADVANTAGE_NORMALIZATION,
    )
    parser.add_argument(
        "--parent-actor-checkpoint",
        type=Path,
        help="Optional supervised or actor-only bidding checkpoint used to initialize the Actor.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except Exception as error:
        return _handle_cli_error(error)


def _run(args: argparse.Namespace) -> int:
    _validate_positive_int(args.epochs, "epochs")
    _validate_positive_int(args.batch_size, "batch-size")
    _validate_positive_float(args.learning_rate, "learning-rate")
    _validate_positive_float(args.ppo_clip_epsilon, "ppo-clip-epsilon")
    if args.value_loss_coefficient < 0.0:
        raise ValueError("value-loss-coefficient must be non-negative.")
    if args.entropy_coefficient < 0.0:
        raise ValueError("entropy-coefficient must be non-negative.")

    settings = BiddingPpoTrainSettings(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        ppo_clip_epsilon=args.ppo_clip_epsilon,
        value_loss_coefficient=args.value_loss_coefficient,
        entropy_coefficient=args.entropy_coefficient,
        advantage_normalization=args.advantage_normalization,
        training_device=args.training_device,
        parent_actor_checkpoint=(
            str(args.parent_actor_checkpoint) if args.parent_actor_checkpoint is not None else None
        ),
    )
    hidden_dims = (
        parse_hidden_dims(args.hidden_dims, label="bidding-hidden-dims")
        if args.hidden_dims is not None
        else None
    )
    model_config = BiddingMlpConfig(
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
    )
    report = train_bidding_ppo(
        dataset_directory=args.dataset_directory,
        output_checkpoint_path=args.output,
        settings=settings,
        model_config=model_config,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"checkpoint: {args.output}")
        print(f"sample_count: {report.sample_count}")
        print(f"mean_total_loss: {report.mean_total_loss:.6f}")
        print(f"mean_reward: {report.mean_reward:.6f}")
        print(f"clipped_fraction: {report.clipped_fraction:.6f}")
        print(f"resolved_training_device: {report.resolved_training_device}")

    return 0


def _validate_positive_int(value: int, label: str) -> None:
    if isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value}.")


def _validate_positive_float(value: float, label: str) -> None:
    if value <= 0.0:
        raise ValueError(f"{label} must be positive, got {value}.")


def _handle_cli_error(error: Exception) -> int:
    if isinstance(error, BiddingPpoCompatibilityError | ValueError):
        print(f"error: {error}", file=sys.stderr)
        return 1
    raise error


if __name__ == "__main__":
    raise SystemExit(main())
