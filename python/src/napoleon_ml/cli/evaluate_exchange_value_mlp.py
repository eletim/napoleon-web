"""Evaluate an exchange discard-combination scalar value checkpoint."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from napoleon_ml.bidding_q.multi_head_training import Standardization
from napoleon_ml.exchange_value import (
    evaluate_exchange_value_model,
    load_exchange_counterfactual_dataset,
    load_exchange_value_checkpoint,
)
from napoleon_ml.exchange_value.dataset import create_exchange_value_split
from napoleon_ml.policy.device import resolve_torch_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "final"), default="final")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    dataset = load_exchange_counterfactual_dataset(args.dataset_directory)
    model, checkpoint = load_exchange_value_checkpoint(args.checkpoint)
    training_config = checkpoint["trainingConfig"]
    if not isinstance(training_config, dict):
        raise ValueError("checkpoint trainingConfig must be an object.")
    standardization_raw = checkpoint["targetStandardization"]
    if not isinstance(standardization_raw, dict):
        raise ValueError("checkpoint targetStandardization must be an object.")
    standardization = Standardization(
        enabled=bool(standardization_raw["enabled"]),
        mean=float(standardization_raw["mean"]),
        std=float(standardization_raw["std"]),
    )
    split = create_exchange_value_split(
        dataset,
        seed=int(training_config["seed"]),
        train_ratio=float(training_config.get("train_ratio", 0.8)),
        validation_ratio=float(training_config.get("validation_ratio", 0.1)),
        final_ratio=float(training_config.get("final_ratio", 0.1)),
        train_state_count=cast(int | None, training_config.get("train_state_count")),
    )
    samples = {
        "train": split.train_samples,
        "validation": split.validation_samples,
        "final": split.final_samples,
    }[args.split]
    device = resolve_torch_device(args.device)
    model.to(device.torch_device)
    report = evaluate_exchange_value_model(
        model,
        samples,
        split=args.split,
        device=device,
        standardization=standardization,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
