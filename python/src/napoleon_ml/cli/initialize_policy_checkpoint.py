"""Create an untrained playing policy checkpoint with current metadata."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from napoleon_ml.cli._policy_common import handle_cli_error, parse_hidden_dims
from napoleon_ml.dataset.constants import PLAYING_OBSERVATION_VARIANTS
from napoleon_ml.policy.checkpoint import (
    ACTOR_CRITIC_MODEL_ARCHITECTURES,
    POLICY_MODEL_ARCHITECTURE,
    initialize_policy_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--playing-observation-variant",
        choices=PLAYING_OBSERVATION_VARIANTS,
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument(
        "--hidden-dims",
        help="Comma-separated hidden widths, e.g. 512,512,256,256. "
        "Overrides --hidden-dim/--hidden-layers when set.",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--model-architecture",
        choices=(POLICY_MODEL_ARCHITECTURE, *ACTOR_CRITIC_MODEL_ARCHITECTURES),
        default=POLICY_MODEL_ARCHITECTURE,
    )
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
    hidden_dims = parse_hidden_dims(args.hidden_dims) if args.hidden_dims is not None else None
    checkpoint = initialize_policy_checkpoint(
        args.output,
        playing_observation_variant=args.playing_observation_variant,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        model_architecture=args.model_architecture,
    )
    model_config = cast(dict[str, object], checkpoint["model_config"])
    provenance = cast(dict[str, object], checkpoint["initialization_provenance"])
    if args.json:
        print(
            json.dumps(
                {
                    "checkpoint": str(args.output),
                    "playingObservationVariant": checkpoint["playing_observation_variant"],
                    "modelInputFeatureCount": checkpoint["model_input_feature_count"],
                    "modelArchitecture": checkpoint["model_architecture"],
                    "modelConfig": model_config,
                    "initializationProvenance": provenance,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(f"checkpoint: {args.output}")
        print(f"playing_observation_variant: {checkpoint['playing_observation_variant']}")
        print(f"model_input_feature_count: {checkpoint['model_input_feature_count']}")
        print(f"model_architecture: {checkpoint['model_architecture']}")
        print(f"model_config: {json.dumps(model_config, sort_keys=True)}")
        print(f"seed: {provenance['seed']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
