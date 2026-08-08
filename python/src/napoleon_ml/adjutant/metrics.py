"""Legal-mask loss, inference, and evaluation for adjutant prediction."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from napoleon_ml.dataset.constants import CARD_COUNT
from napoleon_ml.dataset.pytorch import AdjutantTorchSample
from napoleon_ml.dataset.tensors import ADJUTANT_MODEL_INPUT_LAYOUT

ADJUTANT_CATEGORY_LABELS: tuple[str, ...] = (
    "oruma",
    "sei_jack",
    "ura_jack",
    "trump_standard",
    "other_standard",
    "joker",
)

_ORUMA_SPECIAL_INDEX = 0
_SEI_JACK_SPECIAL_INDEX = 2
_URA_JACK_SPECIAL_INDEX = 3
_JOKER_INDEX = CARD_COUNT - 1
_STANDARD_RANK_COUNT = 13


@dataclass(frozen=True)
class AdjutantMetric:
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
class AdjutantEvaluationReport:
    split: str
    sample_count: int
    masked_loss: float | None
    illegal_prediction_count: int
    top1: AdjutantMetric
    by_category: dict[str, AdjutantMetric]

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "samples": self.sample_count,
            "model": {
                "maskedLoss": self.masked_loss,
                "illegalPredictionCount": self.illegal_prediction_count,
                "top1": self.top1.to_dict(),
                "byCategory": {
                    label: metric.to_dict() for label, metric in self.by_category.items()
                },
            },
            "baselineLegalUniform": {
                "top1": {
                    "count": self.top1.count,
                    "accuracy": self.top1.legal_uniform_accuracy,
                },
                "byCategory": {
                    label: {
                        "count": metric.count,
                        "accuracy": metric.legal_uniform_accuracy,
                    }
                    for label, metric in self.by_category.items()
                },
            },
        }


@dataclass
class _AdjutantCounters:
    correct: int = 0
    total: int = 0
    uniform_accuracy_sum: float = 0.0

    def add(self, *, correct: Tensor, legal_counts: Tensor) -> None:
        self.correct += int(correct.sum().item())
        count = int(correct.numel())
        self.total += count
        if count > 0:
            self.uniform_accuracy_sum += float((1.0 / legal_counts.to(dtype=torch.float32)).sum())

    def metric(self) -> AdjutantMetric:
        if self.total == 0:
            return AdjutantMetric(count=0, accuracy=None, legal_uniform_accuracy=None)

        return AdjutantMetric(
            count=self.total,
            accuracy=self.correct / self.total,
            legal_uniform_accuracy=self.uniform_accuracy_sum / self.total,
        )


@dataclass
class _AdjutantEvaluationAccumulator:
    split: str
    sample_count: int = 0
    loss_sum: float = 0.0
    illegal_prediction_count: int = 0
    top1: _AdjutantCounters = field(default_factory=_AdjutantCounters)
    by_category: dict[str, _AdjutantCounters] = field(
        default_factory=lambda: {
            label: _AdjutantCounters() for label in ADJUTANT_CATEGORY_LABELS
        }
    )

    def update(
        self,
        *,
        batch: AdjutantTorchSample,
        masked_logits: Tensor,
        loss: Tensor | None,
    ) -> None:
        target = batch["actor_target"].to(dtype=torch.long)
        legal_mask = batch["legal_adjutant_mask"].to(dtype=torch.bool)
        predictions = masked_logits.argmax(dim=-1)
        correct = predictions.eq(target)
        legal_counts = legal_mask.sum(dim=-1)
        categories = classify_adjutant_targets(batch["model_input"], target)

        self.sample_count += int(target.shape[0])
        self.illegal_prediction_count += int(
            (~legal_mask.gather(1, predictions.unsqueeze(1)).squeeze(1)).sum().item()
        )

        if loss is not None:
            self.loss_sum += float(loss.item()) * int(target.shape[0])

        self.top1.add(correct=correct, legal_counts=legal_counts)
        for label in ADJUTANT_CATEGORY_LABELS:
            category_mask = categories.eq(_category_index(label))
            self.by_category[label].add(
                correct=correct[category_mask],
                legal_counts=legal_counts[category_mask],
            )

    def report(self) -> AdjutantEvaluationReport:
        masked_loss = None
        if self.sample_count > 0:
            masked_loss = self.loss_sum / self.sample_count

        return AdjutantEvaluationReport(
            split=self.split,
            sample_count=self.sample_count,
            masked_loss=masked_loss,
            illegal_prediction_count=self.illegal_prediction_count,
            top1=self.top1.metric(),
            by_category={
                label: counter.metric() for label, counter in self.by_category.items()
            },
        )


def mask_illegal_adjutant_logits(logits: Tensor, legal_adjutant_mask: Tensor) -> Tensor:
    """Return logits with illegal CARD_IDS positions unavailable to softmax/argmax."""

    _validate_adjutant_shapes(logits, legal_adjutant_mask)
    bool_mask = legal_adjutant_mask.to(dtype=torch.bool)

    if bool((~bool_mask.any(dim=-1)).any().item()):
        raise ValueError("legal_adjutant_mask must contain at least one legal card per row.")

    return logits.masked_fill(~bool_mask, torch.finfo(logits.dtype).min)


def masked_adjutant_cross_entropy(
    logits: Tensor, target: Tensor, legal_adjutant_mask: Tensor
) -> Tensor:
    """Cross entropy after masking illegal adjutant-card logits."""

    _validate_adjutant_shapes(logits, legal_adjutant_mask)
    if target.shape != logits.shape[:1]:
        raise ValueError(
            f"target must have shape {tuple(logits.shape[:1])}, got {tuple(target.shape)}."
        )

    long_target = target.to(dtype=torch.long)
    if bool(((long_target < 0) | (long_target >= CARD_COUNT)).any().item()):
        raise ValueError(f"target values must be between 0 and {CARD_COUNT - 1}.")

    bool_mask = legal_adjutant_mask.to(dtype=torch.bool)
    target_is_legal = bool_mask.gather(1, long_target.unsqueeze(1)).squeeze(1)
    if bool((~target_is_legal).any().item()):
        raise ValueError("target must be legal according to legal_adjutant_mask.")

    return F.cross_entropy(mask_illegal_adjutant_logits(logits, bool_mask), long_target)


def select_adjutant_action(logits: Tensor, legal_adjutant_mask: Tensor) -> Tensor:
    """Select the highest-logit legal CARD_IDS index for each row."""

    return mask_illegal_adjutant_logits(logits, legal_adjutant_mask).argmax(dim=-1)


def masked_adjutant_accuracy(
    logits: Tensor, target: Tensor, legal_adjutant_mask: Tensor
) -> float | None:
    """Return top-1 accuracy after illegal adjutant actions are masked."""

    if target.numel() == 0:
        return None

    predictions = select_adjutant_action(logits, legal_adjutant_mask)
    return float(predictions.eq(target.to(dtype=torch.long)).sum().item()) / float(target.numel())


@torch.no_grad()
def evaluate_adjutant_model(
    model: nn.Module,
    dataloader: DataLoader[AdjutantTorchSample] | Iterable[AdjutantTorchSample],
    *,
    split: str,
) -> AdjutantEvaluationReport:
    model.eval()
    accumulator = _AdjutantEvaluationAccumulator(split=split)

    for batch in dataloader:
        model_input = batch["model_input"].to(dtype=torch.float32)
        target = batch["actor_target"].to(dtype=torch.long)
        legal_mask = batch["legal_adjutant_mask"].to(dtype=torch.bool)
        logits = model(model_input)
        masked_logits = mask_illegal_adjutant_logits(logits, legal_mask)
        loss = masked_adjutant_cross_entropy(logits, target, legal_mask)
        accumulator.update(batch=batch, masked_logits=masked_logits, loss=loss)

    return accumulator.report()


def classify_adjutant_targets(model_input: Tensor, target: Tensor) -> Tensor:
    """Classify target CARD_IDS indices into the required adjutant categories."""

    if model_input.ndim != 2:
        raise ValueError(
            f"model_input must have shape (batch, features), got {tuple(model_input.shape)}."
        )

    if target.shape != model_input.shape[:1]:
        raise ValueError(
            f"target must have shape {tuple(model_input.shape[:1])}, got {tuple(target.shape)}."
        )

    trump_suit = _feature(model_input, "trumpSuitOneHot").argmax(dim=-1)
    special_cards = _feature(model_input, "specialCardIndicesOneHot").reshape(
        model_input.shape[0], 4, CARD_COUNT
    )
    special_indices = special_cards.argmax(dim=-1)
    long_target = target.to(dtype=torch.long)

    categories = torch.full_like(long_target, _category_index("other_standard"))
    standard = long_target.ne(_JOKER_INDEX)
    target_suit = torch.div(long_target, _STANDARD_RANK_COUNT, rounding_mode="floor")
    categories[standard & target_suit.eq(trump_suit)] = _category_index("trump_standard")
    categories[long_target.eq(_JOKER_INDEX)] = _category_index("joker")
    categories[long_target.eq(special_indices[:, _ORUMA_SPECIAL_INDEX])] = _category_index("oruma")
    categories[long_target.eq(special_indices[:, _SEI_JACK_SPECIAL_INDEX])] = _category_index(
        "sei_jack"
    )
    categories[long_target.eq(special_indices[:, _URA_JACK_SPECIAL_INDEX])] = _category_index(
        "ura_jack"
    )
    return categories


def _validate_adjutant_shapes(logits: Tensor, legal_adjutant_mask: Tensor) -> None:
    if logits.ndim != 2 or logits.shape[1] != CARD_COUNT:
        raise ValueError(
            f"logits must have shape (batch, {CARD_COUNT}), got {tuple(logits.shape)}."
        )

    if legal_adjutant_mask.shape != logits.shape:
        raise ValueError(
            f"legal_adjutant_mask must have shape {tuple(logits.shape)}, "
            f"got {tuple(legal_adjutant_mask.shape)}."
        )


def _feature(model_input: Tensor, name: str) -> Tensor:
    for feature in ADJUTANT_MODEL_INPUT_LAYOUT:
        if feature.name == name:
            return model_input[:, feature.start : feature.stop]

    raise AssertionError(f"Unknown adjutant model_input feature: {name}")


def _category_index(label: str) -> int:
    return ADJUTANT_CATEGORY_LABELS.index(label)
