"""Evaluate a saved hidden-card ownership MLP checkpoint on CPU."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from napoleon_ml.cli._ownership_common import (
    add_split_config_arguments,
    dataset_split,
    handle_cli_error,
    load_checked_manifest,
    print_report,
    split_config_from_args,
)
from napoleon_ml.dataset.pytorch import create_playing_dataloader
from napoleon_ml.dataset.split import DatasetSplit, SplitConfig
from napoleon_ml.ownership.checkpoint import (
    CheckpointCompatibilityError,
    load_ownership_checkpoint,
)
from napoleon_ml.ownership.metrics import evaluate_ownership_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=dataset_split, default=DatasetSplit.TEST)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-integrity-check", action="store_true")
    parser.add_argument("--json", action="store_true")
    add_split_config_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return _run(args)
    except Exception as error:
        return handle_cli_error(error)


def _run(args: argparse.Namespace) -> int:
    split_config = split_config_from_args(args)
    manifest = load_checked_manifest(args.dataset_directory, command_label="evaluate")
    model, checkpoint = load_ownership_checkpoint(args.checkpoint, manifest=manifest)
    _validate_checkpoint_split_config(checkpoint, split_config=split_config)
    loader = create_playing_dataloader(
        args.dataset_directory,
        split=args.split,
        split_config=split_config,
        batch_size=args.batch_size,
        verify_integrity=not args.no_integrity_check,
    )
    report = evaluate_ownership_model(model, loader, split=args.split.value)
    print_report(report, as_json=args.json)
    return 0


def _validate_checkpoint_split_config(
    checkpoint: dict[str, object], *, split_config: SplitConfig
) -> None:
    training_config = checkpoint.get("training_config")
    if not isinstance(training_config, dict):
        raise CheckpointCompatibilityError("checkpoint training_config must be a dictionary.")

    expected = {
        "train_ratio": split_config.train,
        "validation_ratio": split_config.validation,
        "test_ratio": split_config.test,
    }

    for key, value in expected.items():
        actual = training_config.get(key)
        if actual != value:
            raise CheckpointCompatibilityError(
                f"checkpoint split ratios do not match evaluation split ratios: "
                f"{key} is {actual!r} in checkpoint, got {value!r} for evaluation."
            )


if __name__ == "__main__":
    raise SystemExit(main())
