"""Legal-action-masked loss, inference, and evaluation for bidding prediction."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from napoleon_ml.dataset.constants import (
    BIDDING_ACTION_COUNT,
    BIDDING_HISTORY_SUIT_ORDER,
    MAX_BIDDING_TARGET_POINT_CARDS,
    MIN_BIDDING_TARGET_POINT_CARDS,
)
from napoleon_ml.dataset.pytorch import BiddingTorchSample


@dataclass(frozen=True)
class BiddingMetric:
    count: int
    accuracy: float | None
    legal_uniform_accuracy: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "count": self.count,
            "accuracy": self.accuracy,
            "legalUniformAccuracy": self.legal_uniform_accuracy,
        }


@dataclass(frozen=True)
class BiddingEvaluationReport:
    split: str
    sample_count: int
    masked_loss: float | None
    illegal_prediction_count: int
    top1: BiddingMetric
    pass_actions: BiddingMetric
    bid_actions: BiddingMetric
    target_point_cards: dict[str, BiddingMetric]
    suits: dict[str, BiddingMetric]

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "samples": self.sample_count,
            "model": {
                "maskedLoss": self.masked_loss,
                "illegalPredictionCount": self.illegal_prediction_count,
                "top1": self.top1.to_dict(),
                "pass": self.pass_actions.to_dict(),
                "bid": self.bid_actions.to_dict(),
                "targetPointCards": {
                    key: value.to_dict() for key, value in self.target_point_cards.items()
                },
                "suits": {key: value.to_dict() for key, value in self.suits.items()},
            },
            "baselineLegalUniform": {
                "top1": {
                    "count": self.top1.count,
                    "accuracy": self.top1.legal_uniform_accuracy,
                },
                "pass": {
                    "count": self.pass_actions.count,
                    "accuracy": self.pass_actions.legal_uniform_accuracy,
                },
                "bid": {
                    "count": self.bid_actions.count,
                    "accuracy": self.bid_actions.legal_uniform_accuracy,
                },
                "targetPointCards": {
                    key: {
                        "count": value.count,
                        "accuracy": value.legal_uniform_accuracy,
                    }
                    for key, value in self.target_point_cards.items()
                },
                "suits": {
                    key: {
                        "count": value.count,
                        "accuracy": value.legal_uniform_accuracy,
                    }
                    for key, value in self.suits.items()
                },
            },
        }


@dataclass
class _BiddingCounters:
    correct: int = 0
    total: int = 0
    uniform_accuracy_sum: float = 0.0

    def add(self, *, correct: Tensor, legal_counts: Tensor) -> None:
        self.correct += int(correct.sum().item())
        count = int(correct.numel())
        self.total += count
        if count > 0:
            self.uniform_accuracy_sum += float((1.0 / legal_counts.to(dtype=torch.float32)).sum())

    def metric(self) -> BiddingMetric:
        if self.total == 0:
            return BiddingMetric(count=0, accuracy=None, legal_uniform_accuracy=None)

        return BiddingMetric(
            count=self.total,
            accuracy=self.correct / self.total,
            legal_uniform_accuracy=self.uniform_accuracy_sum / self.total,
        )


@dataclass
class _BiddingEvaluationAccumulator:
    split: str
    sample_count: int = 0
    loss_sum: float = 0.0
    illegal_prediction_count: int = 0
    top1: _BiddingCounters = field(default_factory=_BiddingCounters)
    pass_actions: _BiddingCounters = field(default_factory=_BiddingCounters)
    bid_actions: _BiddingCounters = field(default_factory=_BiddingCounters)
    target_point_cards: dict[int, _BiddingCounters] = field(
        default_factory=lambda: {
            target: _BiddingCounters()
            for target in range(MIN_BIDDING_TARGET_POINT_CARDS, MAX_BIDDING_TARGET_POINT_CARDS + 1)
        }
    )
    suits: dict[str, _BiddingCounters] = field(
        default_factory=lambda: {suit: _BiddingCounters() for suit in BIDDING_HISTORY_SUIT_ORDER}
    )

    def update(
        self,
        *,
        batch: BiddingTorchSample,
        masked_logits: Tensor,
        loss: Tensor | None,
    ) -> None:
        target = batch["actor_target"].to(dtype=torch.long)
        legal_mask = batch["legal_bid_mask"].to(dtype=torch.bool)
        predictions = masked_logits.argmax(dim=-1)
        correct = predictions.eq(target)
        legal_counts = legal_mask.sum(dim=-1)

        self.sample_count += int(target.shape[0])
        self.illegal_prediction_count += int(
            (~legal_mask.gather(1, predictions.unsqueeze(1)).squeeze(1)).sum().item()
        )

        if loss is not None:
            self.loss_sum += float(loss.item()) * int(target.shape[0])

        self.top1.add(correct=correct, legal_counts=legal_counts)

        pass_mask = target.eq(0)
        bid_mask = target.ne(0)
        self.pass_actions.add(correct=correct[pass_mask], legal_counts=legal_counts[pass_mask])
        self.bid_actions.add(correct=correct[bid_mask], legal_counts=legal_counts[bid_mask])

        for target_point_cards, group_mask in _target_point_masks(target).items():
            self.target_point_cards[target_point_cards].add(
                correct=correct[group_mask], legal_counts=legal_counts[group_mask]
            )

        for suit, group_mask in _suit_masks(target).items():
            self.suits[suit].add(correct=correct[group_mask], legal_counts=legal_counts[group_mask])

    def report(self) -> BiddingEvaluationReport:
        masked_loss = None
        if self.sample_count > 0:
            masked_loss = self.loss_sum / self.sample_count

        return BiddingEvaluationReport(
            split=self.split,
            sample_count=self.sample_count,
            masked_loss=masked_loss,
            illegal_prediction_count=self.illegal_prediction_count,
            top1=self.top1.metric(),
            pass_actions=self.pass_actions.metric(),
            bid_actions=self.bid_actions.metric(),
            target_point_cards={
                str(key): value.metric() for key, value in self.target_point_cards.items()
            },
            suits={key: value.metric() for key, value in self.suits.items()},
        )


def mask_illegal_bidding_logits(logits: Tensor, legal_bid_mask: Tensor) -> Tensor:
    """Return logits with illegal bidding actions unavailable to softmax/argmax."""

    _validate_bidding_shapes(logits, legal_bid_mask)
    bool_mask = legal_bid_mask.to(dtype=torch.bool)

    if bool((~bool_mask.any(dim=-1)).any().item()):
        raise ValueError("legal_bid_mask must contain at least one legal action per row.")

    return logits.masked_fill(~bool_mask, torch.finfo(logits.dtype).min)


def masked_bidding_cross_entropy(logits: Tensor, target: Tensor, legal_bid_mask: Tensor) -> Tensor:
    """Cross entropy after masking illegal bidding-action logits."""

    _validate_bidding_shapes(logits, legal_bid_mask)
    if target.shape != logits.shape[:1]:
        raise ValueError(
            f"target must have shape {tuple(logits.shape[:1])}, got {tuple(target.shape)}."
        )

    long_target = target.to(dtype=torch.long)
    if bool(((long_target < 0) | (long_target >= BIDDING_ACTION_COUNT)).any().item()):
        raise ValueError(f"target values must be between 0 and {BIDDING_ACTION_COUNT - 1}.")

    bool_mask = legal_bid_mask.to(dtype=torch.bool)
    target_is_legal = bool_mask.gather(1, long_target.unsqueeze(1)).squeeze(1)
    if bool((~target_is_legal).any().item()):
        raise ValueError("target must be legal according to legal_bid_mask.")

    return F.cross_entropy(mask_illegal_bidding_logits(logits, bool_mask), long_target)


def select_bidding_action(logits: Tensor, legal_bid_mask: Tensor) -> Tensor:
    """Select the highest-logit legal bidding action index for each row."""

    return mask_illegal_bidding_logits(logits, legal_bid_mask).argmax(dim=-1)


def masked_bidding_accuracy(
    logits: Tensor, target: Tensor, legal_bid_mask: Tensor
) -> float | None:
    """Return top-1 accuracy after illegal actions are masked."""

    if target.numel() == 0:
        return None

    predictions = select_bidding_action(logits, legal_bid_mask)
    return float(predictions.eq(target.to(dtype=torch.long)).sum().item()) / float(target.numel())


@torch.no_grad()
def evaluate_bidding_model(
    model: nn.Module,
    dataloader: DataLoader[BiddingTorchSample] | Iterable[BiddingTorchSample],
    *,
    split: str,
) -> BiddingEvaluationReport:
    model.eval()
    accumulator = _BiddingEvaluationAccumulator(split=split)

    for batch in dataloader:
        model_input = batch["model_input"].to(dtype=torch.float32)
        target = batch["actor_target"].to(dtype=torch.long)
        legal_mask = batch["legal_bid_mask"].to(dtype=torch.bool)
        logits = model(model_input)
        masked_logits = mask_illegal_bidding_logits(logits, legal_mask)
        loss = masked_bidding_cross_entropy(logits, target, legal_mask)
        accumulator.update(batch=batch, masked_logits=masked_logits, loss=loss)

    return accumulator.report()


def bidding_action_target_point_cards(action_index: int) -> int | None:
    """Return target point cards for a bid action index, or None for pass."""

    if action_index == 0:
        return None

    _validate_action_index(action_index)
    bid_offset = action_index - 1
    return MIN_BIDDING_TARGET_POINT_CARDS + bid_offset // len(BIDDING_HISTORY_SUIT_ORDER)


def bidding_action_suit(action_index: int) -> str | None:
    """Return suit for a bid action index, or None for pass."""

    if action_index == 0:
        return None

    _validate_action_index(action_index)
    bid_offset = action_index - 1
    return BIDDING_HISTORY_SUIT_ORDER[bid_offset % len(BIDDING_HISTORY_SUIT_ORDER)]


def _target_point_masks(target: Tensor) -> dict[int, Tensor]:
    return {
        target_point_cards: torch.tensor(
            [
                bidding_action_target_point_cards(int(action_index)) == target_point_cards
                for action_index in target.tolist()
            ],
            dtype=torch.bool,
            device=target.device,
        )
        for target_point_cards in range(
            MIN_BIDDING_TARGET_POINT_CARDS, MAX_BIDDING_TARGET_POINT_CARDS + 1
        )
    }


def _suit_masks(target: Tensor) -> dict[str, Tensor]:
    return {
        suit: torch.tensor(
            [bidding_action_suit(int(action_index)) == suit for action_index in target.tolist()],
            dtype=torch.bool,
            device=target.device,
        )
        for suit in BIDDING_HISTORY_SUIT_ORDER
    }


def _validate_action_index(action_index: int) -> None:
    if action_index < 0 or action_index >= BIDDING_ACTION_COUNT:
        raise ValueError(
            f"bidding action index must be between 0 and {BIDDING_ACTION_COUNT - 1}, "
            f"got {action_index}."
        )


def _validate_bidding_shapes(logits: Tensor, legal_bid_mask: Tensor) -> None:
    if logits.ndim != 2 or logits.shape[1] != BIDDING_ACTION_COUNT:
        raise ValueError(
            f"logits must have shape (batch, {BIDDING_ACTION_COUNT}), got {tuple(logits.shape)}."
        )

    if legal_bid_mask.shape != logits.shape:
        raise ValueError(
            f"legal_bid_mask must have shape {tuple(logits.shape)}, "
            f"got {tuple(legal_bid_mask.shape)}."
        )
