"""Run iterative playing self-play REINFORCE experiments."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from napoleon_ml.policy.actor_critic import DEFAULT_VALUE_LOSS_COEFFICIENT
from napoleon_ml.policy.reinforce import REINFORCE_ALGORITHM
from napoleon_ml.rl_orchestrator import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_EVALUATION_INTERVAL,
    DEFAULT_EVALUATION_SEED_COUNT,
    DEFAULT_EVALUATION_START_SEED,
    DEFAULT_GAMES_PER_ITERATION,
    DEFAULT_GAMES_PER_SHARD,
    DEFAULT_ITERATIONS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_ROLLOUT_ROSTER,
    DEFAULT_ROLLOUT_WORKERS,
    DEFAULT_SELF_PLAY_SEED_BASE,
    DEFAULT_TEMPERATURE,
    DEFAULT_TRAINING_SEED_BASE,
    PLAYING_RL_ALGORITHMS,
    PlayingRlOrchestratorError,
    PlayingRlRunConfig,
    run_playing_rl_experiment,
)

_OPTION_TO_CONFIG_KEY = {
    "--iterations": "iterations",
    "--games-per-iteration": "gamesPerIteration",
    "--games-per-shard": "gamesPerShard",
    "--self-play-seed-base": "selfPlaySeedBase",
    "--temperature": "temperature",
    "--rollout-roster": "rolloutRoster",
    "--rollout-workers": "rolloutWorkers",
    "--algorithm": "algorithm",
    "--learning-rate": "learningRate",
    "--value-loss-coefficient": "valueLossCoefficient",
    "--epochs": "epochs",
    "--batch-size": "batchSize",
    "--training-seed-base": "trainingSeedBase",
    "--evaluation-interval": "evaluationInterval",
    "--evaluation-start-seed": "evaluationStartSeed",
    "--evaluation-seed-count": "evaluationSeedCount",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--supervised-dataset", type=Path)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--games-per-iteration", type=int, default=DEFAULT_GAMES_PER_ITERATION)
    parser.add_argument("--games-per-shard", type=int, default=DEFAULT_GAMES_PER_SHARD)
    parser.add_argument("--self-play-seed-base", type=int, default=DEFAULT_SELF_PLAY_SEED_BASE)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--rollout-roster", default=DEFAULT_ROLLOUT_ROSTER)
    parser.add_argument("--rollout-workers", type=int, default=DEFAULT_ROLLOUT_WORKERS)
    parser.add_argument(
        "--algorithm",
        choices=PLAYING_RL_ALGORITHMS,
        default=REINFORCE_ALGORITHM,
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--value-loss-coefficient",
        type=float,
        default=DEFAULT_VALUE_LOSS_COEFFICIENT,
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--training-seed-base", type=int, default=DEFAULT_TRAINING_SEED_BASE)
    parser.add_argument("--evaluation-interval", type=int, default=DEFAULT_EVALUATION_INTERVAL)
    parser.add_argument("--evaluation-start-seed", type=int, default=DEFAULT_EVALUATION_START_SEED)
    parser.add_argument("--evaluation-seed-count", type=int, default=DEFAULT_EVALUATION_SEED_COUNT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-typescript-build", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = _config_from_args(args, parser)
        provided_config_keys = _provided_config_keys(argv if argv is not None else sys.argv[1:])
        run_playing_rl_experiment(
            config,
            resume=args.resume,
            provided_config_keys=provided_config_keys,
        )
    except Exception as error:
        if isinstance(error, PlayingRlOrchestratorError | ValueError):
            print(f"error: {error}", file=sys.stderr)
            return 1
        raise

    return 0


def _config_from_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> PlayingRlRunConfig:
    if args.initial_checkpoint is None:
        parser.error("--initial-checkpoint is required.")
    if args.supervised_dataset is None:
        parser.error("--supervised-dataset is required.")

    return PlayingRlRunConfig(
        run_directory=args.run_directory,
        initial_checkpoint=args.initial_checkpoint,
        supervised_dataset=args.supervised_dataset,
        iterations=args.iterations,
        games_per_iteration=args.games_per_iteration,
        games_per_shard=args.games_per_shard,
        self_play_seed_base=args.self_play_seed_base,
        temperature=args.temperature,
        rollout_roster=args.rollout_roster,
        rollout_workers=args.rollout_workers,
        algorithm=args.algorithm,
        learning_rate=args.learning_rate,
        value_loss_coefficient=args.value_loss_coefficient,
        epochs=args.epochs,
        batch_size=args.batch_size,
        training_seed_base=args.training_seed_base,
        evaluation_interval=args.evaluation_interval,
        evaluation_start_seed=args.evaluation_start_seed,
        evaluation_seed_count=args.evaluation_seed_count,
        build_typescript=not args.no_typescript_build,
    )


def _provided_config_keys(argv: Sequence[str]) -> set[str]:
    keys: set[str] = set()
    for arg in argv:
        name = arg.split("=", 1)[0]
        key = _OPTION_TO_CONFIG_KEY.get(name)
        if key is not None:
            keys.add(key)
    return keys


if __name__ == "__main__":
    raise SystemExit(main())
