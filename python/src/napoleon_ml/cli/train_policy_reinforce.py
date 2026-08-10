"""Train a playing policy checkpoint from self-play trajectories."""

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
from napoleon_ml.policy.actor_critic import (
    ACTOR_CRITIC_ALGORITHM,
    ActorCriticTrainReport,
    ActorCriticTrainSettings,
    train_policy_actor_critic,
)
from napoleon_ml.policy.device import SUPPORTED_TORCH_DEVICES
from napoleon_ml.policy.reinforce import (
    REINFORCE_ALGORITHM,
    ReinforceTrainReport,
    ReinforceTrainSettings,
    train_policy_reinforce,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("self_play_dataset_directory", type=Path)
    parser.add_argument("--input-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--algorithm",
        choices=(REINFORCE_ALGORITHM, ACTOR_CRITIC_ALGORITHM),
        default=REINFORCE_ALGORITHM,
    )
    parser.add_argument("--value-loss-coefficient", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=SUPPORTED_TORCH_DEVICES, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--behavior-parity-execution-provider",
        choices=("cpu", "cuda"),
        default=None,
        help="Runtime provider used to generate behavior log probabilities.",
    )
    parser.add_argument(
        "--behavior-parity-max-observed-batch-size",
        type=int,
        default=None,
        help="Maximum ONNX batch size observed during rollout.",
    )
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
    seed = int(args.seed)
    configure_reproducibility(seed)
    manifest = load_checked_manifest(
        args.self_play_dataset_directory,
        command_label="train-policy-reinforce",
    )
    dataloader = create_playing_self_play_dataloader(
        args.self_play_dataset_directory,
        split=DatasetSplit.TRAIN,
        split_config=SplitConfig(train=100, validation=0, test=0),
        batch_size=args.batch_size,
        verify_integrity=not args.no_integrity_check,
    )
    report: ActorCriticTrainReport | ReinforceTrainReport
    if args.algorithm == ACTOR_CRITIC_ALGORITHM:
        ac_settings = ActorCriticTrainSettings(
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            verify_integrity=not args.no_integrity_check,
            device=args.device,
            value_loss_coefficient=args.value_loss_coefficient,
            behavior_parity_execution_provider=args.behavior_parity_execution_provider,
            behavior_parity_max_observed_batch_size=(
                args.behavior_parity_max_observed_batch_size
            ),
        )
        report = train_policy_actor_critic(
            input_checkpoint=args.input_checkpoint,
            self_play_dataset_directory=args.self_play_dataset_directory,
            output_checkpoint=args.output,
            manifest=manifest,
            dataloader=dataloader,
            settings=ac_settings,
        )
    else:
        reinforce_settings = ReinforceTrainSettings(
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            verify_integrity=not args.no_integrity_check,
            device=args.device,
            behavior_parity_execution_provider=args.behavior_parity_execution_provider,
            behavior_parity_max_observed_batch_size=(
                args.behavior_parity_max_observed_batch_size
            ),
        )
        report = train_policy_reinforce(
            input_checkpoint=args.input_checkpoint,
            self_play_dataset_directory=args.self_play_dataset_directory,
            output_checkpoint=args.output,
            manifest=manifest,
            dataloader=dataloader,
            settings=reinforce_settings,
        )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"checkpoint: {report.output_checkpoint_path}")
        print(f"samples: {report.sample_count}")
        print(f"batches: {report.batch_count}")
        print(f"optimizer_steps: {report.optimizer_step_count}")
        print(f"requested_device: {report.requested_device}")
        print(f"resolved_device: {report.resolved_device}")
        if report.cuda_device_name is not None:
            print(f"cuda_device_name: {report.cuda_device_name}")
        print(
            "safety_validation_elapsed_seconds: "
            f"{report.safety_validation_elapsed_seconds:.6f}"
        )
        print(
            "pre_eval_elapsed_seconds: "
            f"{_format_optional_float(report.pre_eval_elapsed_seconds)}"
        )
        print(
            "optimizer_training_elapsed_seconds: "
            f"{report.optimizer_training_elapsed_seconds:.6f}"
        )
        print(
            "post_eval_elapsed_seconds: "
            f"{_format_optional_float(report.post_eval_elapsed_seconds)}"
        )
        print(f"total_elapsed_seconds: {report.total_elapsed_seconds:.6f}")
        if isinstance(report, ReinforceTrainReport):
            print(f"mean_policy_loss: {report.mean_policy_loss:.8f}")
            print(
                "mean_policy_loss_before: "
                f"{_format_optional_float(report.mean_policy_loss_before, precision=8)}"
            )
            print(
                "mean_policy_loss_after: "
                f"{_format_optional_float(report.mean_policy_loss_after, precision=8)}"
            )
        else:
            print(f"mean_actor_loss: {report.mean_actor_loss:.8f}")
            print(
                "actor_loss_before: "
                f"{_format_optional_float(report.actor_loss_before, precision=8)}"
            )
            print(
                "actor_loss_after: "
                f"{_format_optional_float(report.actor_loss_after, precision=8)}"
            )
            print(
                "value_loss_before: "
                f"{_format_optional_float(report.value_loss_before, precision=8)}"
            )
            print(
                "value_loss_after: "
                f"{_format_optional_float(report.value_loss_after, precision=8)}"
            )
            print(
                "total_loss_before: "
                f"{_format_optional_float(report.total_loss_before, precision=8)}"
            )
            print(
                "total_loss_after: "
                f"{_format_optional_float(report.total_loss_after, precision=8)}"
            )
        print(
            "mean_selected_log_probability_before: "
            f"{_format_optional_float(report.mean_selected_log_probability_before, precision=8)}"
        )
        print(
            "mean_selected_log_probability_after: "
            f"{_format_optional_float(report.mean_selected_log_probability_after, precision=8)}"
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


def _format_optional_float(value: float | None, *, precision: int = 6) -> str:
    if value is None:
        return "null"
    return f"{value:.{precision}f}"


if __name__ == "__main__":
    raise SystemExit(main())
