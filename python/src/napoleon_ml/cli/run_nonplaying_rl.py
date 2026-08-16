"""Run non-playing rollout, PPO training, ONNX export, and full-policy evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

from napoleon_ml.nonplaying_rl_orchestrator import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BIDDING_ENTROPY_COEFFICIENT,
    DEFAULT_EPOCHS,
    DEFAULT_EVALUATION_GAMES,
    DEFAULT_EVALUATION_INTERVAL,
    DEFAULT_GAMES,
    DEFAULT_GAMES_PER_ITERATION,
    DEFAULT_GAMES_PER_SHARD,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_HIDDEN_LAYERS,
    DEFAULT_INFERENCE_DEVICE,
    DEFAULT_INFERENCE_MAX_BATCH_SIZE,
    DEFAULT_ITERATIONS,
    DEFAULT_ITERATIVE_BATCH_SIZE,
    DEFAULT_ITERATIVE_EPOCHS,
    DEFAULT_ITERATIVE_EVALUATION_GAMES,
    DEFAULT_ITERATIVE_LEARNING_RATE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_PLAYING_POLICY_ARTIFACT_ID,
    DEFAULT_PLAYING_POLICY_METADATA,
    DEFAULT_PLAYING_POLICY_ONNX,
    DEFAULT_PPO_CLIP_EPSILON,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_VALUE_LOSS_COEFFICIENT,
    SUPPORTED_INFERENCE_DEVICES,
    NonPlayingIterativeRlRunConfig,
    NonPlayingRlOrchestratorError,
    NonPlayingRlRunConfig,
    run_iterative_nonplaying_rl_pipeline,
    run_nonplaying_rl_pipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help=(
            "Run iterative non-playing PPO for this many iterations. "
            "Omit for the original one-shot smoke pipeline."
        ),
    )
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES)
    parser.add_argument("--games-per-iteration", type=int, default=DEFAULT_GAMES_PER_ITERATION)
    parser.add_argument("--evaluation-games", type=int, default=DEFAULT_EVALUATION_GAMES)
    parser.add_argument(
        "--evaluation-interval",
        "--evaluation-every",
        dest="evaluation_interval",
        type=int,
        default=DEFAULT_EVALUATION_INTERVAL,
    )
    parser.add_argument("--games-per-shard", type=int, default=DEFAULT_GAMES_PER_SHARD)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--hidden-layers", type=int, default=DEFAULT_HIDDEN_LAYERS)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--ppo-clip-epsilon", type=float, default=DEFAULT_PPO_CLIP_EPSILON)
    parser.add_argument(
        "--value-loss-coefficient",
        type=float,
        default=DEFAULT_VALUE_LOSS_COEFFICIENT,
    )
    parser.add_argument(
        "--bidding-entropy-coefficient",
        type=float,
        default=DEFAULT_BIDDING_ENTROPY_COEFFICIENT,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument(
        "--inference-device",
        choices=SUPPORTED_INFERENCE_DEVICES,
        default=DEFAULT_INFERENCE_DEVICE,
    )
    parser.add_argument(
        "--inference-max-batch-size",
        type=int,
        default=DEFAULT_INFERENCE_MAX_BATCH_SIZE,
    )
    parser.add_argument("--playing-policy-onnx", type=Path, default=DEFAULT_PLAYING_POLICY_ONNX)
    parser.add_argument(
        "--playing-policy-metadata",
        type=Path,
        default=DEFAULT_PLAYING_POLICY_METADATA,
    )
    parser.add_argument(
        "--playing-policy-artifact-id",
        default=DEFAULT_PLAYING_POLICY_ARTIFACT_ID,
    )
    parser.add_argument("--no-typescript-build", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--stop-after-iterations",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.resume or args.iterations is not None:
            provided_config_keys = _provided_iterative_config_keys(argv)
            summary = run_iterative_nonplaying_rl_pipeline(
                _iterative_config_from_args(
                    args,
                    provided_config_keys=provided_config_keys,
                ),
                resume=args.resume,
                stop_after_iterations=args.stop_after_iterations,
                provided_config_keys=provided_config_keys,
            )
        else:
            summary = run_nonplaying_rl_pipeline(_config_from_args(args))
    except Exception as error:
        if isinstance(error, NonPlayingRlOrchestratorError | ValueError):
            print(f"error: {error}", file=sys.stderr)
            return 1
        raise

    if args.json:
        print(json.dumps(_compact_summary(summary), indent=2, sort_keys=True))
    return 0


def _config_from_args(args: argparse.Namespace) -> NonPlayingRlRunConfig:
    return NonPlayingRlRunConfig(
        output_dir=args.output_dir,
        games=args.games,
        evaluation_games=args.evaluation_games,
        games_per_shard=args.games_per_shard,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        dropout=args.dropout,
        ppo_clip_epsilon=args.ppo_clip_epsilon,
        value_loss_coefficient=args.value_loss_coefficient,
        bidding_entropy_coefficient=args.bidding_entropy_coefficient,
        seed=args.seed,
        temperature=args.temperature,
        inference_device=cast(Literal["cpu", "auto", "cuda"], args.inference_device),
        inference_max_batch_size=args.inference_max_batch_size,
        playing_policy_onnx=args.playing_policy_onnx,
        playing_policy_metadata=args.playing_policy_metadata,
        playing_policy_artifact_id=args.playing_policy_artifact_id,
        build_typescript=not args.no_typescript_build,
        overwrite=args.overwrite,
    )


def _iterative_config_from_args(
    args: argparse.Namespace,
    *,
    provided_config_keys: Sequence[str],
) -> NonPlayingIterativeRlRunConfig:
    provided = set(provided_config_keys)
    iterations = args.iterations if args.iterations is not None else DEFAULT_ITERATIONS
    evaluation_games = (
        args.evaluation_games
        if "evaluationGames" in provided
        else DEFAULT_ITERATIVE_EVALUATION_GAMES
    )
    epochs = args.epochs if "epochs" in provided else DEFAULT_ITERATIVE_EPOCHS
    batch_size = (
        args.batch_size
        if "batchSize" in provided
        else DEFAULT_ITERATIVE_BATCH_SIZE
    )
    learning_rate = (
        args.learning_rate
        if "learningRate" in provided
        else DEFAULT_ITERATIVE_LEARNING_RATE
    )
    return NonPlayingIterativeRlRunConfig(
        output_dir=args.output_dir,
        iterations=iterations,
        games_per_iteration=args.games_per_iteration,
        evaluation_interval=args.evaluation_interval,
        evaluation_games=evaluation_games,
        games_per_shard=args.games_per_shard,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        dropout=args.dropout,
        ppo_clip_epsilon=args.ppo_clip_epsilon,
        value_loss_coefficient=args.value_loss_coefficient,
        bidding_entropy_coefficient=args.bidding_entropy_coefficient,
        seed=args.seed,
        temperature=args.temperature,
        inference_device=cast(Literal["cpu", "auto", "cuda"], args.inference_device),
        inference_max_batch_size=args.inference_max_batch_size,
        playing_policy_onnx=args.playing_policy_onnx,
        playing_policy_metadata=args.playing_policy_metadata,
        playing_policy_artifact_id=args.playing_policy_artifact_id,
        build_typescript=not args.no_typescript_build,
        overwrite=args.overwrite,
    )


def _provided_iterative_config_keys(argv: Sequence[str] | None) -> list[str]:
    if argv is None:
        argv = sys.argv[1:]
    names = {token.split("=", 1)[0] for token in argv if token.startswith("--")}
    option_to_key = {
        "--iterations": "iterations",
        "--games-per-iteration": "gamesPerIteration",
        "--evaluation-interval": "evaluationInterval",
        "--evaluation-every": "evaluationInterval",
        "--evaluation-games": "evaluationGames",
        "--games-per-shard": "gamesPerShard",
        "--epochs": "epochs",
        "--batch-size": "batchSize",
        "--learning-rate": "learningRate",
        "--hidden-dim": "hiddenDim",
        "--hidden-layers": "hiddenLayers",
        "--dropout": "dropout",
        "--ppo-clip-epsilon": "ppoClipEpsilon",
        "--value-loss-coefficient": "valueLossCoefficient",
        "--bidding-entropy-coefficient": "biddingEntropyCoefficient",
        "--seed": "seed",
        "--temperature": "temperature",
        "--inference-device": "inferenceDevice",
        "--inference-max-batch-size": "inferenceMaxBatchSize",
        "--playing-policy-onnx": "playingPolicyOnnx",
        "--playing-policy-metadata": "playingPolicyMetadata",
        "--playing-policy-artifact-id": "playingPolicyArtifactId",
    }
    return sorted({key for option, key in option_to_key.items() if option in names})


def _compact_summary(summary: dict[str, object]) -> dict[str, object]:
    if summary.get("runType") == "non-playing-iterative-ppo":
        config = cast(dict[str, object], summary["config"])
        return {
            "runSummary": f"{config['outputDir']}/run-summary.json",
            "completedIterationCount": summary["completedIterationCount"],
            "latestCompletedIteration": summary["latestCompletedIteration"],
            "latestIteration": summary["latestIteration"],
        }
    return {
        "runSummary": cast(dict[str, object], summary["artifactPaths"])["runSummary"],
        "evaluation": summary["evaluation"],
        "artifactPaths": summary["artifactPaths"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
