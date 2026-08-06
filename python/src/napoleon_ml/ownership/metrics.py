"""Masked loss and evaluation metrics for ownership prediction."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from napoleon_ml.dataset.constants import (
    BELIEF_OWNER_CLASS_COUNT,
    CARD_COUNT,
    NOT_IN_HAND_CLASS_INDEX,
)
from napoleon_ml.dataset.pytorch import PlayingTorchSample

OWNER_CLASS_LABELS: tuple[str, ...] = (
    "relative_player_0",
    "relative_player_1",
    "relative_player_2",
    "relative_player_3",
    "relative_player_4",
    "not_in_hand",
)

PROGRESS_BUCKET_LABELS: tuple[str, ...] = ("step_01_16", "step_17_33", "step_34_plus")


@dataclass(frozen=True)
class EvaluationReport:
    split: str
    sample_count: int
    masked_card_count: int
    masked_loss: float | None
    masked_accuracy: float | None
    owner_class_accuracy: dict[str, float | None]
    progress_accuracy: dict[str, float | None]
    baseline_masked_accuracy: float | None
    baseline_owner_class_accuracy: dict[str, float | None]
    baseline_progress_accuracy: dict[str, float | None]

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "samples": self.sample_count,
            "maskedCards": self.masked_card_count,
            "model": {
                "maskedLoss": self.masked_loss,
                "maskedAccuracy": self.masked_accuracy,
                "ownerClassAccuracy": self.owner_class_accuracy,
                "progressAccuracy": self.progress_accuracy,
            },
            "baselineAlwaysNotInHand": {
                "maskedAccuracy": self.baseline_masked_accuracy,
                "ownerClassAccuracy": self.baseline_owner_class_accuracy,
                "progressAccuracy": self.baseline_progress_accuracy,
            },
        }


@dataclass
class _AccuracyCounters:
    correct: int = 0
    total: int = 0

    def add(self, *, correct: int, total: int) -> None:
        self.correct += correct
        self.total += total

    def value(self) -> float | None:
        if self.total == 0:
            return None

        return self.correct / self.total


@dataclass
class _EvaluationAccumulator:
    split: str
    sample_count: int = 0
    masked_card_count: int = 0
    loss_sum: float = 0.0
    model_total: _AccuracyCounters = field(default_factory=_AccuracyCounters)
    baseline_total: _AccuracyCounters = field(default_factory=_AccuracyCounters)
    model_by_owner: dict[str, _AccuracyCounters] = field(
        default_factory=lambda: {label: _AccuracyCounters() for label in OWNER_CLASS_LABELS}
    )
    baseline_by_owner: dict[str, _AccuracyCounters] = field(
        default_factory=lambda: {label: _AccuracyCounters() for label in OWNER_CLASS_LABELS}
    )
    model_by_progress: dict[str, _AccuracyCounters] = field(
        default_factory=lambda: {label: _AccuracyCounters() for label in PROGRESS_BUCKET_LABELS}
    )
    baseline_by_progress: dict[str, _AccuracyCounters] = field(
        default_factory=lambda: {label: _AccuracyCounters() for label in PROGRESS_BUCKET_LABELS}
    )

    def update(
        self,
        *,
        batch: PlayingTorchSample,
        logits: Tensor,
        loss: Tensor | None,
    ) -> None:
        target = batch["belief_target"].to(dtype=torch.long)
        mask = batch["belief_hidden_ownership_loss_mask"].to(dtype=torch.bool)
        model_predictions = logits.argmax(dim=-1)
        baseline_predictions = torch.full_like(target, NOT_IN_HAND_CLASS_INDEX)

        masked_total = int(mask.sum().item())
        self.sample_count += int(target.shape[0])
        self.masked_card_count += masked_total

        if loss is not None:
            self.loss_sum += float(loss.item()) * masked_total

        self._update_counters(
            predictions=model_predictions,
            target=target,
            mask=mask,
            steps=batch["step"],
            total_counter=self.model_total,
            by_owner=self.model_by_owner,
            by_progress=self.model_by_progress,
        )
        self._update_counters(
            predictions=baseline_predictions,
            target=target,
            mask=mask,
            steps=batch["step"],
            total_counter=self.baseline_total,
            by_owner=self.baseline_by_owner,
            by_progress=self.baseline_by_progress,
        )

    def _update_counters(
        self,
        *,
        predictions: Tensor,
        target: Tensor,
        mask: Tensor,
        steps: Tensor,
        total_counter: _AccuracyCounters,
        by_owner: dict[str, _AccuracyCounters],
        by_progress: dict[str, _AccuracyCounters],
    ) -> None:
        if not bool(mask.any().item()):
            return

        correct = predictions.eq(target) & mask
        total_counter.add(correct=int(correct.sum().item()), total=int(mask.sum().item()))

        for class_index, label in enumerate(OWNER_CLASS_LABELS):
            class_mask = mask & target.eq(class_index)
            by_owner[label].add(
                correct=int((correct & class_mask).sum().item()),
                total=int(class_mask.sum().item()),
            )

        for row_index, step in enumerate(steps.tolist()):
            label = _progress_bucket(int(step))
            row_mask = mask[row_index]
            by_progress[label].add(
                correct=int(correct[row_index][row_mask].sum().item()),
                total=int(row_mask.sum().item()),
            )

    def report(self) -> EvaluationReport:
        masked_loss = None
        if self.masked_card_count > 0:
            masked_loss = self.loss_sum / self.masked_card_count

        return EvaluationReport(
            split=self.split,
            sample_count=self.sample_count,
            masked_card_count=self.masked_card_count,
            masked_loss=masked_loss,
            masked_accuracy=self.model_total.value(),
            owner_class_accuracy={
                label: counter.value() for label, counter in self.model_by_owner.items()
            },
            progress_accuracy={
                label: counter.value() for label, counter in self.model_by_progress.items()
            },
            baseline_masked_accuracy=self.baseline_total.value(),
            baseline_owner_class_accuracy={
                label: counter.value() for label, counter in self.baseline_by_owner.items()
            },
            baseline_progress_accuracy={
                label: counter.value() for label, counter in self.baseline_by_progress.items()
            },
        )


def masked_ownership_cross_entropy(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Cross entropy over hidden-card mask positions only."""

    _validate_shapes(logits, target, mask)
    bool_mask = mask.to(dtype=torch.bool)

    if not bool(bool_mask.any().item()):
        raise ValueError("masked ownership loss requires at least one unmasked card.")

    return F.cross_entropy(logits[bool_mask], target.to(dtype=torch.long)[bool_mask])


def masked_ownership_accuracy(logits: Tensor, target: Tensor, mask: Tensor) -> float | None:
    """Return masked accuracy, or None if the mask contains no cards."""

    _validate_shapes(logits, target, mask)
    bool_mask = mask.to(dtype=torch.bool)

    if not bool(bool_mask.any().item()):
        return None

    predictions = logits.argmax(dim=-1)
    correct = (predictions.eq(target.to(dtype=torch.long)) & bool_mask).sum()
    return float(correct.item()) / float(bool_mask.sum().item())


@torch.no_grad()
def evaluate_ownership_model(
    model: nn.Module,
    dataloader: DataLoader[PlayingTorchSample] | Iterable[PlayingTorchSample],
    *,
    split: str,
) -> EvaluationReport:
    model.eval()
    accumulator = _EvaluationAccumulator(split=split)

    for batch in dataloader:
        model_input = batch["model_input"].to(dtype=torch.float32)
        target = batch["belief_target"].to(dtype=torch.long)
        mask = batch["belief_hidden_ownership_loss_mask"].to(dtype=torch.bool)
        logits = model(model_input)
        loss = None
        if bool(mask.any().item()):
            loss = masked_ownership_cross_entropy(logits, target, mask)
        accumulator.update(batch=batch, logits=logits, loss=loss)

    return accumulator.report()


def _validate_shapes(logits: Tensor, target: Tensor, mask: Tensor) -> None:
    if logits.ndim != 3 or logits.shape[1:] != (CARD_COUNT, BELIEF_OWNER_CLASS_COUNT):
        raise ValueError(
            f"logits must have shape (batch, {CARD_COUNT}, {BELIEF_OWNER_CLASS_COUNT}), "
            f"got {tuple(logits.shape)}."
        )

    if target.shape != logits.shape[:2]:
        raise ValueError(
            f"target must have shape {tuple(logits.shape[:2])}, got {tuple(target.shape)}."
        )

    if mask.shape != logits.shape[:2]:
        raise ValueError(
            f"mask must have shape {tuple(logits.shape[:2])}, got {tuple(mask.shape)}."
        )


def _progress_bucket(step: int) -> str:
    if step <= 16:
        return "step_01_16"

    if step <= 33:
        return "step_17_33"

    return "step_34_plus"
