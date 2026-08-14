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
    DEFAULT_EPOCHS,
    DEFAULT_EVALUATION_GAMES,
    DEFAULT_GAMES,
    DEFAULT_GAMES_PER_SHARD,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_HIDDEN_LAYERS,
    DEFAULT_INFERENCE_DEVICE,
    DEFAULT_INFERENCE_MAX_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_PLAYING_POLICY_ARTIFACT_ID,
    DEFAULT_PLAYING_POLICY_METADATA,
    DEFAULT_PLAYING_POLICY_ONNX,
    DEFAULT_PPO_CLIP_EPSILON,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_VALUE_LOSS_COEFFICIENT,
    SUPPORTED_INFERENCE_DEVICES,
    NonPlayingRlOrchestratorError,
    NonPlayingRlRunConfig,
    run_nonplaying_rl_pipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES)
    parser.add_argument("--evaluation-games", type=int, default=DEFAULT_EVALUATION_GAMES)
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
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
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


def _compact_summary(summary: dict[str, object]) -> dict[str, object]:
    return {
        "runSummary": cast(dict[str, object], summary["artifactPaths"])["runSummary"],
        "evaluation": summary["evaluation"],
        "artifactPaths": summary["artifactPaths"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
