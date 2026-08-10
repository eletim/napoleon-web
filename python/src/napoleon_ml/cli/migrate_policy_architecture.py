"""Migrate a playing policy checkpoint to a larger hidden-width architecture."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from napoleon_ml.cli._policy_common import handle_cli_error, parse_hidden_dims
from napoleon_ml.policy.checkpoint import migrate_policy_checkpoint_to_hidden_dims


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--target-hidden-dims",
        default="512,512,256,256",
        help="Comma-separated target hidden widths. Defaults to 512,512,256,256.",
    )
    parser.add_argument("--seed", type=int, default=0)
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
    migrated = migrate_policy_checkpoint_to_hidden_dims(
        args.input_checkpoint,
        args.output,
        target_hidden_dims=parse_hidden_dims(
            args.target_hidden_dims,
            label="target-hidden-dims",
        ),
        seed=args.seed,
    )
    provenance = cast(dict[str, object], migrated["architecture_migration_provenance"])
    if args.json:
        print(json.dumps({"checkpoint": str(args.output), "provenance": provenance}, indent=2))
    else:
        print(f"checkpoint: {args.output}")
        print(f"migration: {provenance['migration']}")
        print(f"source_sha256: {provenance['sourceCheckpointSha256']}")
        print(f"target_model_config: {json.dumps(provenance['targetModelConfig'])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
