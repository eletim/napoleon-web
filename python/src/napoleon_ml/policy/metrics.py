"""Legal-action-masked loss, inference, and evaluation for policy prediction."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from napoleon_ml.dataset.constants import CARD_COUNT
from napoleon_ml.dataset.pytorch import PlayingTorchSample


@dataclass(frozen=True)
class PolicyMetric:
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
class PolicyEvaluationReport:
    split: str
    sample_count: int
    masked_loss: float | None
    illegal_prediction_count: int
    all_positions: PolicyMetric
    forced_positions: PolicyMetric
    non_forced_positions: PolicyMetric

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "samples": self.sample_count,
            "model": {
                "maskedLoss": self.masked_loss,
                "illegalPredictionCount": self.illegal_prediction_count,
                "allPositions": self.all_positions.to_dict(),
                "forcedPositions": self.forced_positions.to_dict(),
                "nonForcedPositions": self.non_forced_positions.to_dict(),
            },
            "baselineLegalUniform": {
                "allPositions": {
                    "count": self.all_positions.count,
                    "accuracy": self.all_positions.legal_uniform_accuracy,
                },
                "forcedPositions": {
                    "count": self.forced_positions.count,
                    "accuracy": self.forced_positions.legal_uniform_accuracy,
                },
                "nonForcedPositions": {
                    "count": self.non_forced_positions.count,
                    "accuracy": self.non_forced_positions.legal_uniform_accuracy,
                },
            },
        }


@dataclass
class _PolicyCounters:
    correct: int = 0
    total: int = 0
    uniform_accuracy_sum: float = 0.0

    def add(self, *, correct: Tensor, legal_counts: Tensor) -> None:
        self.correct += int(correct.sum().item())
        count = int(correct.numel())
        self.total += count
        if count > 0:
            self.uniform_accuracy_sum += float((1.0 / legal_counts.to(dtype=torch.float32)).sum())

    def metric(self) -> PolicyMetric:
        if self.total == 0:
            return PolicyMetric(count=0, accuracy=None, legal_uniform_accuracy=None)

        return PolicyMetric(
            count=self.total,
            accuracy=self.correct / self.total,
            legal_uniform_accuracy=self.uniform_accuracy_sum / self.total,
        )


@dataclass
class _PolicyEvaluationAccumulator:
    split: str
    sample_count: int = 0
    loss_sum: float = 0.0
    illegal_prediction_count: int = 0
    all_positions: _PolicyCounters = field(default_factory=_PolicyCounters)
    forced_positions: _PolicyCounters = field(default_factory=_PolicyCounters)
    non_forced_positions: _PolicyCounters = field(default_factory=_PolicyCounters)

    def update(
        self,
        *,
        batch: PlayingTorchSample,
        masked_logits: Tensor,
        loss: Tensor | None,
    ) -> None:
        target = batch["actor_target"].to(dtype=torch.long)
        legal_mask = batch["legal_play_mask"].to(dtype=torch.bool)
        predictions = masked_logits.argmax(dim=-1)
        correct = predictions.eq(target)
        legal_counts = legal_mask.sum(dim=-1)
        forced = legal_counts.eq(1)
        non_forced = legal_counts.gt(1)

        self.sample_count += int(target.shape[0])
        self.illegal_prediction_count += int(
            (~legal_mask.gather(1, predictions.unsqueeze(1)).squeeze(1)).sum().item()
        )

        if loss is not None:
            self.loss_sum += float(loss.item()) * int(target.shape[0])

        self.all_positions.add(correct=correct, legal_counts=legal_counts)
        self.forced_positions.add(correct=correct[forced], legal_counts=legal_counts[forced])
        self.non_forced_positions.add(
            correct=correct[non_forced], legal_counts=legal_counts[non_forced]
        )

    def report(self) -> PolicyEvaluationReport:
        masked_loss = None
        if self.sample_count > 0:
            masked_loss = self.loss_sum / self.sample_count

        return PolicyEvaluationReport(
            split=self.split,
            sample_count=self.sample_count,
            masked_loss=masked_loss,
            illegal_prediction_count=self.illegal_prediction_count,
            all_positions=self.all_positions.metric(),
            forced_positions=self.forced_positions.metric(),
            non_forced_positions=self.non_forced_positions.metric(),
        )


def mask_illegal_policy_logits(logits: Tensor, legal_play_mask: Tensor) -> Tensor:
    """Return logits with illegal card positions unavailable to softmax/argmax."""

    _validate_policy_shapes(logits, legal_play_mask)
    bool_mask = legal_play_mask.to(dtype=torch.bool)

    if bool((~bool_mask.any(dim=-1)).any().item()):
        raise ValueError("legal_play_mask must contain at least one legal card per row.")

    return logits.masked_fill(~bool_mask, torch.finfo(logits.dtype).min)


def masked_policy_cross_entropy(logits: Tensor, target: Tensor, legal_play_mask: Tensor) -> Tensor:
    """Cross entropy after masking illegal card logits."""

    _validate_policy_shapes(logits, legal_play_mask)
    if target.shape != logits.shape[:1]:
        raise ValueError(
            f"target must have shape {tuple(logits.shape[:1])}, got {tuple(target.shape)}."
        )

    long_target = target.to(dtype=torch.long)
    if bool(((long_target < 0) | (long_target >= CARD_COUNT)).any().item()):
        raise ValueError(f"target values must be between 0 and {CARD_COUNT - 1}.")

    bool_mask = legal_play_mask.to(dtype=torch.bool)
    target_is_legal = bool_mask.gather(1, long_target.unsqueeze(1)).squeeze(1)
    if bool((~target_is_legal).any().item()):
        raise ValueError("target must be legal according to legal_play_mask.")

    return F.cross_entropy(mask_illegal_policy_logits(logits, bool_mask), long_target)


def select_policy_action(logits: Tensor, legal_play_mask: Tensor) -> Tensor:
    """Select the highest-logit legal card index for each row."""

    return mask_illegal_policy_logits(logits, legal_play_mask).argmax(dim=-1)


def masked_policy_accuracy(
    logits: Tensor, target: Tensor, legal_play_mask: Tensor
) -> float | None:
    """Return top-1 accuracy after illegal actions are masked."""

    if target.numel() == 0:
        return None

    predictions = select_policy_action(logits, legal_play_mask)
    return float(predictions.eq(target.to(dtype=torch.long)).sum().item()) / float(target.numel())


@torch.no_grad()
def evaluate_policy_model(
    model: nn.Module,
    dataloader: DataLoader[PlayingTorchSample] | Iterable[PlayingTorchSample],
    *,
    split: str,
) -> PolicyEvaluationReport:
    model.eval()
    accumulator = _PolicyEvaluationAccumulator(split=split)

    for batch in dataloader:
        model_input = batch["model_input"].to(dtype=torch.float32)
        target = batch["actor_target"].to(dtype=torch.long)
        legal_mask = batch["legal_play_mask"].to(dtype=torch.bool)
        logits = model(model_input)
        masked_logits = mask_illegal_policy_logits(logits, legal_mask)
        loss = masked_policy_cross_entropy(logits, target, legal_mask)
        accumulator.update(batch=batch, masked_logits=masked_logits, loss=loss)

    return accumulator.report()


def _validate_policy_shapes(logits: Tensor, legal_play_mask: Tensor) -> None:
    if logits.ndim != 2 or logits.shape[1] != CARD_COUNT:
        raise ValueError(
            f"logits must have shape (batch, {CARD_COUNT}), got {tuple(logits.shape)}."
        )

    if legal_play_mask.shape != logits.shape:
        raise ValueError(
            f"legal_play_mask must have shape {tuple(logits.shape)}, "
            f"got {tuple(legal_play_mask.shape)}."
        )
