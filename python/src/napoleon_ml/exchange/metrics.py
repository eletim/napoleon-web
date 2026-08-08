"""Legal-card-masked loss, inference, and set evaluation for exchange discard."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from napoleon_ml.dataset.constants import CARD_COUNT
from napoleon_ml.dataset.pytorch import ExchangeTorchSample

DISCARD_COUNT = 3


@dataclass(frozen=True)
class ExchangeEvaluationReport:
    split: str
    sample_count: int
    masked_loss: float | None
    exact_set_accuracy: float | None
    average_overlap_count: float | None
    selected_teacher_match_rate: float | None
    illegal_selected_card_count: int
    baseline_random_legal_exact_set_accuracy: float | None
    baseline_random_legal_average_overlap_count: float | None
    baseline_random_legal_selected_teacher_match_rate: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "samples": self.sample_count,
            "model": {
                "maskedLoss": self.masked_loss,
                "exactSetAccuracy": self.exact_set_accuracy,
                "averageOverlapCount": self.average_overlap_count,
                "selectedTeacherMatchRate": self.selected_teacher_match_rate,
                "illegalSelectedCardCount": self.illegal_selected_card_count,
            },
            "baselineRandomLegal": {
                "exactSetAccuracy": self.baseline_random_legal_exact_set_accuracy,
                "averageOverlapCount": self.baseline_random_legal_average_overlap_count,
                "selectedTeacherMatchRate": (
                    self.baseline_random_legal_selected_teacher_match_rate
                ),
            },
        }


@dataclass
class _ExchangeEvaluationAccumulator:
    split: str
    sample_count: int = 0
    loss_sum: float = 0.0
    exact_set_correct: int = 0
    overlap_sum: int = 0
    illegal_selected_card_count: int = 0
    baseline_exact_sum: float = 0.0
    baseline_overlap_sum: float = 0.0

    def update(
        self,
        *,
        batch: ExchangeTorchSample,
        selected_mask: Tensor,
        loss: Tensor | None,
    ) -> None:
        legal_mask = batch["legal_discard_card_mask"].to(dtype=torch.bool)
        target_mask = batch["discard_target_mask"].to(dtype=torch.bool)
        _validate_exchange_masks(target_mask, legal_mask)

        batch_size = int(target_mask.shape[0])
        selected_count = selected_mask.sum(dim=-1)
        if bool(selected_count.ne(DISCARD_COUNT).any().item()):
            raise ValueError(f"selected discard mask must be exactly {DISCARD_COUNT}-hot.")

        overlap = (selected_mask & target_mask).sum(dim=-1)
        exact = selected_mask.eq(target_mask).all(dim=-1)
        legal_counts = legal_mask.sum(dim=-1)

        self.sample_count += batch_size
        self.overlap_sum += int(overlap.sum().item())
        self.exact_set_correct += int(exact.sum().item())
        self.illegal_selected_card_count += int((selected_mask & ~legal_mask).sum().item())

        if loss is not None:
            self.loss_sum += float(loss.item()) * batch_size

        target_counts = target_mask.sum(dim=-1).to(dtype=torch.float32)
        select_counts = selected_count.to(dtype=torch.float32)
        legal_counts_float = legal_counts.to(dtype=torch.float32)
        self.baseline_overlap_sum += float(
            (target_counts * select_counts / legal_counts_float).sum().item()
        )
        self.baseline_exact_sum += sum(
            1.0 / math.comb(int(legal_count), DISCARD_COUNT)
            for legal_count in legal_counts.tolist()
        )

    def report(self) -> ExchangeEvaluationReport:
        if self.sample_count == 0:
            return ExchangeEvaluationReport(
                split=self.split,
                sample_count=0,
                masked_loss=None,
                exact_set_accuracy=None,
                average_overlap_count=None,
                selected_teacher_match_rate=None,
                illegal_selected_card_count=self.illegal_selected_card_count,
                baseline_random_legal_exact_set_accuracy=None,
                baseline_random_legal_average_overlap_count=None,
                baseline_random_legal_selected_teacher_match_rate=None,
            )

        average_overlap = self.overlap_sum / self.sample_count
        baseline_average_overlap = self.baseline_overlap_sum / self.sample_count
        return ExchangeEvaluationReport(
            split=self.split,
            sample_count=self.sample_count,
            masked_loss=self.loss_sum / self.sample_count,
            exact_set_accuracy=self.exact_set_correct / self.sample_count,
            average_overlap_count=average_overlap,
            selected_teacher_match_rate=average_overlap / DISCARD_COUNT,
            illegal_selected_card_count=self.illegal_selected_card_count,
            baseline_random_legal_exact_set_accuracy=self.baseline_exact_sum
            / self.sample_count,
            baseline_random_legal_average_overlap_count=baseline_average_overlap,
            baseline_random_legal_selected_teacher_match_rate=baseline_average_overlap
            / DISCARD_COUNT,
        )


def masked_exchange_bce_loss(logits: Tensor, target_mask: Tensor, legal_mask: Tensor) -> Tensor:
    """Binary multi-label loss over legal candidate cards only."""

    _validate_exchange_shapes(logits, target_mask, legal_mask)
    bool_legal = legal_mask.to(dtype=torch.bool)
    bool_target = target_mask.to(dtype=torch.bool)
    _validate_exchange_masks(bool_target, bool_legal)

    return F.binary_cross_entropy_with_logits(
        logits[bool_legal],
        bool_target.to(dtype=logits.dtype)[bool_legal],
    )


def mask_illegal_exchange_logits(logits: Tensor, legal_mask: Tensor) -> Tensor:
    """Return logits with illegal card positions unavailable to selection."""

    if logits.ndim != 2 or logits.shape[1] != CARD_COUNT:
        raise ValueError(
            f"logits must have shape (batch, {CARD_COUNT}), got {tuple(logits.shape)}."
        )

    if legal_mask.shape != logits.shape:
        raise ValueError(f"legal_mask must have shape {tuple(logits.shape)}.")

    bool_legal = legal_mask.to(dtype=torch.bool)
    legal_counts = bool_legal.sum(dim=-1)
    if bool(legal_counts.lt(DISCARD_COUNT).any().item()):
        raise ValueError(f"legal_mask must contain at least {DISCARD_COUNT} legal cards per row.")

    return logits.masked_fill(~bool_legal, torch.finfo(logits.dtype).min)


def select_exchange_discards(logits: Tensor, legal_mask: Tensor) -> Tensor:
    """Select distinct legal top-3 cards with lower card index winning ties."""

    masked_logits = mask_illegal_exchange_logits(logits, legal_mask)
    order = torch.argsort(masked_logits, dim=-1, descending=True, stable=True)
    return order[:, :DISCARD_COUNT].to(dtype=torch.long)


def selected_indices_to_mask(indices: Tensor, *, card_count: int = CARD_COUNT) -> Tensor:
    if indices.ndim != 2 or indices.shape[1] != DISCARD_COUNT:
        raise ValueError(
            f"indices must have shape (batch, {DISCARD_COUNT}), got {tuple(indices.shape)}."
        )

    mask = torch.zeros(
        (indices.shape[0], card_count),
        dtype=torch.bool,
        device=indices.device,
    )
    return mask.scatter(1, indices.to(dtype=torch.long), True)


@torch.no_grad()
def evaluate_exchange_model(
    model: nn.Module,
    dataloader: DataLoader[ExchangeTorchSample] | Iterable[ExchangeTorchSample],
    *,
    split: str,
) -> ExchangeEvaluationReport:
    model.eval()
    accumulator = _ExchangeEvaluationAccumulator(split=split)

    for batch in dataloader:
        model_input = batch["model_input"].to(dtype=torch.float32)
        target_mask = batch["discard_target_mask"].to(dtype=torch.bool)
        legal_mask = batch["legal_discard_card_mask"].to(dtype=torch.bool)
        logits = model(model_input)
        selected = select_exchange_discards(logits, legal_mask)
        selected_mask = selected_indices_to_mask(selected)
        loss = masked_exchange_bce_loss(logits, target_mask, legal_mask)
        accumulator.update(batch=batch, selected_mask=selected_mask, loss=loss)

    return accumulator.report()


def _validate_exchange_shapes(logits: Tensor, target_mask: Tensor, legal_mask: Tensor) -> None:
    if logits.ndim != 2 or logits.shape[1] != CARD_COUNT:
        raise ValueError(
            f"logits must have shape (batch, {CARD_COUNT}), got {tuple(logits.shape)}."
        )

    if target_mask.shape != logits.shape:
        raise ValueError(f"target_mask must have shape {tuple(logits.shape)}.")

    if legal_mask.shape != logits.shape:
        raise ValueError(f"legal_mask must have shape {tuple(logits.shape)}.")


def _validate_exchange_masks(target_mask: Tensor, legal_mask: Tensor) -> None:
    target_counts = target_mask.sum(dim=-1)
    if bool(target_counts.ne(DISCARD_COUNT).any().item()):
        raise ValueError(f"discard_target_mask must be exactly {DISCARD_COUNT}-hot.")

    if bool((target_mask & ~legal_mask).any().item()):
        raise ValueError("discard_target_mask must be a subset of legal_mask.")

    legal_counts = legal_mask.sum(dim=-1)
    if bool(legal_counts.lt(DISCARD_COUNT).any().item()):
        raise ValueError(f"legal_mask must contain at least {DISCARD_COUNT} legal cards per row.")
