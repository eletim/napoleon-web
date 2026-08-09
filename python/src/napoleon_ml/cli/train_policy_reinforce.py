"""Train a playing policy checkpoint with REINFORCE v1 self-play trajectories."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from napoleon_ml.cli._policy_common import (
    configure_reproducibility,
    handle_cli_error,
    load_checked_manifest,
)
from napoleon_ml.dataset.pytorch import create_playing_self_play_dataloader
from napoleon_ml.dataset.split import DatasetSplit, SplitConfig
from napoleon_ml.policy.reinforce import ReinforceTrainSettings, train_policy_reinforce


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("self_play_dataset_directory", type=Path)
    parser.add_argument("--input-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-integrity-check", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return _run(args)
    except Exception as error:
        return handle_cli_error(error)


def _run(args: argparse.Namespace) -> int:
    settings = ReinforceTrainSettings(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        verify_integrity=not args.no_integrity_check,
    )
    configure_reproducibility(settings.seed)
    manifest = load_checked_manifest(
        args.self_play_dataset_directory,
        command_label="train-policy-reinforce",
    )
    dataloader = create_playing_self_play_dataloader(
        args.self_play_dataset_directory,
        split=DatasetSplit.TRAIN,
        split_config=SplitConfig(train=100, validation=0, test=0),
        batch_size=settings.batch_size,
        verify_integrity=settings.verify_integrity,
    )
    report = train_policy_reinforce(
        input_checkpoint=args.input_checkpoint,
        self_play_dataset_directory=args.self_play_dataset_directory,
        output_checkpoint=args.output,
        manifest=manifest,
        dataloader=dataloader,
        settings=settings,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"checkpoint: {report.output_checkpoint_path}")
        print(f"samples: {report.sample_count}")
        print(f"batches: {report.batch_count}")
        print(f"optimizer_steps: {report.optimizer_step_count}")
        print(f"mean_policy_loss: {report.mean_policy_loss:.8f}")
        print(f"mean_policy_loss_before: {report.mean_policy_loss_before:.8f}")
        print(f"mean_policy_loss_after: {report.mean_policy_loss_after:.8f}")
        print(
            "mean_selected_log_probability_before: "
            f"{report.mean_selected_log_probability_before:.8f}"
        )
        print(
            "mean_selected_log_probability_after: "
            f"{report.mean_selected_log_probability_after:.8f}"
        )
        print(f"mean_reward: {report.mean_reward:.8f}")
        print(f"positive_reward_count: {report.positive_reward_count}")
        print(f"negative_reward_count: {report.negative_reward_count}")
        print(f"forced_sample_count: {report.forced_sample_count}")
        print(f"non_forced_sample_count: {report.non_forced_sample_count}")
        print(
            "max_behavior_log_probability_parity_error: "
            f"{report.max_behavior_log_probability_parity_error:.8f}"
        )
        print(f"parameter_delta_norm: {report.parameter_delta_norm:.8f}")
        print(f"changed_parameter_count: {report.changed_parameter_count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
