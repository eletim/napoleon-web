"""Train the exchange discard MLP baseline on CPU."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import optim

from napoleon_ml.cli._exchange_common import (
    add_split_config_arguments,
    configure_reproducibility,
    handle_cli_error,
    load_checked_exchange_manifest,
    print_exchange_report,
    split_config_from_args,
)
from napoleon_ml.dataset.pytorch import ExchangeTorchSample, create_exchange_dataloader
from napoleon_ml.dataset.split import DatasetSplit
from napoleon_ml.exchange.checkpoint import save_exchange_checkpoint
from napoleon_ml.exchange.metrics import evaluate_exchange_model, masked_exchange_bce_loss
from napoleon_ml.exchange.model import ExchangeMlpConfig, create_seeded_exchange_model


@dataclass(frozen=True)
class TrainSettings:
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    hidden_dim: int
    hidden_layers: int
    dropout: float
    train_ratio: int
    validation_ratio: int
    test_ratio: int
    verify_integrity: bool
    optimizer: str = "AdamW"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
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
    if args.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {args.epochs}.")

    if args.learning_rate <= 0.0:
        raise ValueError(f"learning-rate must be positive, got {args.learning_rate}.")

    split_config = split_config_from_args(args)
    settings = TrainSettings(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        dropout=args.dropout,
        train_ratio=split_config.train,
        validation_ratio=split_config.validation,
        test_ratio=split_config.test,
        verify_integrity=not args.no_integrity_check,
    )

    configure_reproducibility(settings.seed)
    manifest = load_checked_exchange_manifest(
        args.dataset_directory, command_label="train-exchange"
    )
    model_config = ExchangeMlpConfig(
        hidden_dim=settings.hidden_dim,
        hidden_layers=settings.hidden_layers,
        dropout=settings.dropout,
    )
    model = create_seeded_exchange_model(model_config, seed=settings.seed)
    optimizer = optim.AdamW(model.parameters(), lr=settings.learning_rate)

    train_loader = create_exchange_dataloader(
        args.dataset_directory,
        split=DatasetSplit.TRAIN,
        split_config=split_config,
        batch_size=settings.batch_size,
        verify_integrity=settings.verify_integrity,
    )
    validation_loader = create_exchange_dataloader(
        args.dataset_directory,
        split=DatasetSplit.VALIDATION,
        split_config=split_config,
        batch_size=settings.batch_size,
        verify_integrity=settings.verify_integrity,
    )

    epoch_reports: list[dict[str, object]] = []
    for epoch in range(1, settings.epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer)
        validation_report = evaluate_exchange_model(
            model, validation_loader, split=DatasetSplit.VALIDATION.value
        )
        epoch_reports.append(
            {
                "epoch": epoch,
                "trainMaskedLoss": train_loss,
                "validation": validation_report.to_dict(),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_exchange_checkpoint(
        args.output,
        model=model,
        training_config=asdict(settings),
        manifest=manifest,
        seed=settings.seed,
    )

    final_report = evaluate_exchange_model(
        model, validation_loader, split=DatasetSplit.VALIDATION.value
    )

    if args.json:
        print(
            json.dumps(
                {
                    "checkpoint": str(args.output),
                    "training": asdict(settings),
                    "epochs": epoch_reports,
                    "finalValidation": final_report.to_dict(),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(f"checkpoint: {args.output}")
        print_exchange_report(final_report, as_json=False)

    return 0


def _train_one_epoch(
    model: torch.nn.Module,
    dataloader: Iterable[ExchangeTorchSample],
    optimizer: optim.Optimizer,
) -> float:
    model.train()
    loss_sum = 0.0
    sample_total = 0

    for batch in dataloader:
        model_input = batch["model_input"].to(dtype=torch.float32)
        target_mask = batch["discard_target_mask"].to(dtype=torch.bool)
        legal_mask = batch["legal_discard_card_mask"].to(dtype=torch.bool)

        optimizer.zero_grad(set_to_none=True)
        logits = model(model_input)
        loss = masked_exchange_bce_loss(logits, target_mask, legal_mask)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

        batch_size = int(target_mask.shape[0])
        loss_sum += float(loss.item()) * batch_size
        sample_total += batch_size

    if sample_total == 0:
        raise ValueError("train split contains no exchange samples.")

    return loss_sum / sample_total


if __name__ == "__main__":
    raise SystemExit(main())
