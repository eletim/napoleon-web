"""Training and diagnostics for Issue #378 multi-head bidding models."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import pickle
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, NamedTuple, cast

import numpy as np
import torch
from torch import Tensor, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from napoleon_ml.dataset.constants import BIDDING_ACTION_COUNT
from napoleon_ml.policy.device import (
    RequestedTorchDevice,
    ResolvedTorchDevice,
    cpu_state_dict,
    resolve_torch_device,
)

from .dataset import (
    BiddingQDataset,
    BiddingQRawSample,
    SuitName,
)
from .multi_head_model import (
    BIDDING_MULTI_HEAD_Q_ARCHITECTURE_ID,
    BiddingMultiHeadQModel,
    BiddingMultiHeadQModelConfig,
    BiddingMultiHeadQOutput,
    create_seeded_bidding_multi_head_q_model,
)
from .role_model import BIDDING_ROLE_CLASSES, BIDDING_ROLE_COUNT
from .role_training import canonical_bidding_role, role_index_for_terminal_role

BIDDING_MULTI_HEAD_Q_CHECKPOINT_SCHEMA_VERSION = 1
BIDDING_MULTI_HEAD_Q_MODEL_TYPE = "bidding-multi-head-q"
TOTAL_POINT_CARDS = 20.0


class BiddingMultiHeadQCheckpointError(ValueError):
    """Raised when a multi-head checkpoint is incompatible."""


@dataclass(frozen=True)
class Standardization:
    enabled: bool
    mean: float
    std: float

    def encode_tensor(self, value: Tensor) -> Tensor:
        if not self.enabled:
            return value
        return (value - self.mean) / self.std

    def decode_array(self, value: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return value
        return value * self.std + self.mean

    def to_dict(self) -> dict[str, object]:
        return {"enabled": self.enabled, "mean": self.mean, "std": self.std}


@dataclass(frozen=True)
class BiddingMultiHeadQTrainConfig:
    seed: int = 378
    epochs: int = 80
    batch_size: int = 128
    learning_rate: float = 1e-3
    hidden_dims: tuple[int, ...] = (512, 512, 256, 256)
    dropout: float = 0.0
    train_state_count: int | None = 20000
    validation_state_keys_path: str | None = None
    train_ratio: float = 20000 / 22000
    role_loss_weight: float = 1.0
    napoleon_point_loss_weight: float = 1.0
    success_loss_weight: float = 1.0
    margin_loss_weight: float = 1.0
    regression_standardization: bool = True
    weight_decay: float = 1e-4
    patience: int = 10
    min_delta: float = 0.0
    monitor: Literal["validation-total-loss"] = "validation-total-loss"
    device: RequestedTorchDevice = "cpu"
    optimizer: str = "AdamW"

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["hidden_dims"] = list(self.hidden_dims)
        data["algorithm"] = "multi-head-bidding-teacher-decomposition-v1"
        return data


@dataclass(frozen=True)
class BiddingMultiHeadQSplit:
    train_state_keys: frozenset[str]
    validation_state_keys: frozenset[str]
    train_samples: tuple[BiddingQRawSample, ...]
    validation_samples: tuple[BiddingQRawSample, ...]
    validation_state_key_hash: str


@dataclass(frozen=True)
class BiddingMultiHeadQTrainResult:
    model: BiddingMultiHeadQModel
    split: BiddingMultiHeadQSplit
    config: BiddingMultiHeadQTrainConfig
    device: ResolvedTorchDevice
    napoleon_standardization: Standardization
    margin_standardization: Standardization
    epoch_reports: list[dict[str, object]]
    best_epoch: int
    best_monitor_value: float
    train_report: dict[str, object]
    validation_report: dict[str, object]
    teacher_signal: dict[str, object]


class BiddingMultiHeadQBatch(NamedTuple):
    model_input: Tensor
    action_index: Tensor
    role_index: Tensor
    napoleon_point_cards: Tensor
    coalition_point_cards: Tensor
    contract_success: Tensor
    contract_margin: Tensor
    contract_mask: Tensor


class MultiHeadLosses(NamedTuple):
    total_loss: Tensor
    role_loss: Tensor
    napoleon_point_loss: Tensor
    success_loss: Tensor
    margin_loss: Tensor


class _BiddingMultiHeadQDataset(Dataset[BiddingMultiHeadQBatch]):
    def __init__(
        self,
        samples: Iterable[BiddingQRawSample],
        *,
        napoleon_standardization: Standardization,
        margin_standardization: Standardization,
    ) -> None:
        self.samples = tuple(samples)
        self.napoleon_standardization = napoleon_standardization
        self.margin_standardization = margin_standardization

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> BiddingMultiHeadQBatch:
        sample = self.samples[index]
        contract_mask = (
            sample.result_type == "standard" and sample.final_declared_target is not None
        )
        napoleon_cards = float(sample.napoleon_side_point_cards or 0)
        coalition_cards = (
            float(sample.coalition_side_point_cards)
            if sample.coalition_side_point_cards is not None
            else TOTAL_POINT_CARDS - napoleon_cards
        )
        margin = (
            napoleon_cards - float(sample.final_declared_target)
            if contract_mask and sample.final_declared_target is not None
            else 0.0
        )
        napoleon_tensor = self.napoleon_standardization.encode_tensor(
            torch.tensor(napoleon_cards, dtype=torch.float32)
        )
        margin_tensor = self.margin_standardization.encode_tensor(
            torch.tensor(margin, dtype=torch.float32)
        )
        return BiddingMultiHeadQBatch(
            model_input=torch.as_tensor(sample.model_input, dtype=torch.float32),
            action_index=torch.tensor(sample.forced_action_index, dtype=torch.long),
            role_index=torch.tensor(
                role_index_for_terminal_role(sample.terminal_role), dtype=torch.long
            ),
            napoleon_point_cards=napoleon_tensor,
            coalition_point_cards=torch.tensor(coalition_cards, dtype=torch.float32),
            contract_success=torch.tensor(
                1.0 if sample.contract_success else 0.0, dtype=torch.float32
            ),
            contract_margin=margin_tensor,
            contract_mask=torch.tensor(contract_mask, dtype=torch.bool),
        )


def load_validation_state_keys(path: Path | str) -> tuple[str, ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw_keys = raw.get("stateKeys")
    else:
        raw_keys = raw
    if not isinstance(raw_keys, list) or not all(isinstance(key, str) for key in raw_keys):
        raise ValueError("validation state key file must contain a stateKeys string list.")
    keys = tuple(raw_keys)
    if len(keys) != len(set(keys)):
        raise ValueError("validation state keys contain duplicates.")
    return keys


def state_key_hash(state_keys: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(state_keys)).encode("utf-8")).hexdigest()


def create_multi_head_split(
    dataset: BiddingQDataset,
    config: BiddingMultiHeadQTrainConfig,
) -> BiddingMultiHeadQSplit:
    all_state_keys = _state_keys_in_source_order(dataset.raw_samples)
    all_state_key_set = set(all_state_keys)
    if config.validation_state_keys_path is not None:
        validation_keys = tuple(load_validation_state_keys(config.validation_state_keys_path))
        missing = set(validation_keys) - all_state_key_set
        if missing:
            raise ValueError(f"validation state keys missing from dataset: {len(missing)}.")
    else:
        keyed = sorted(
            all_state_keys,
            key=lambda value: hashlib.sha256(f"{config.seed}:{value}".encode()).hexdigest(),
        )
        validation_count = len(all_state_keys) - int(
            round(len(all_state_keys) * config.train_ratio)
        )
        validation_keys = tuple(sorted(keyed[-validation_count:]))
    validation_set = frozenset(validation_keys)
    train_candidates = [key for key in all_state_keys if key not in validation_set]
    train_count = config.train_state_count or len(train_candidates)
    if train_count <= 0 or train_count > len(train_candidates):
        raise ValueError(
            f"train_state_count must be in [1,{len(train_candidates)}], got {train_count}."
        )
    train_set = frozenset(train_candidates[:train_count])
    if train_set & validation_set:
        raise AssertionError("stateKey leakage between train and validation.")
    train_samples = tuple(sample for sample in dataset.raw_samples if sample.state_key in train_set)
    validation_samples = tuple(
        sample for sample in dataset.raw_samples if sample.state_key in validation_set
    )
    if not train_samples or not validation_samples:
        raise ValueError("train and validation splits must both contain samples.")
    return BiddingMultiHeadQSplit(
        train_state_keys=train_set,
        validation_state_keys=validation_set,
        train_samples=train_samples,
        validation_samples=validation_samples,
        validation_state_key_hash=state_key_hash(validation_set),
    )


def train_bidding_multi_head_q_model(
    dataset: BiddingQDataset,
    config: BiddingMultiHeadQTrainConfig,
) -> BiddingMultiHeadQTrainResult:
    _validate_train_config(config)
    _configure_reproducibility(config.seed)
    split = create_multi_head_split(dataset, config)
    device = resolve_torch_device(config.device, flag_name="--device")
    napoleon_standardization = _standardization(
        [
            float(sample.napoleon_side_point_cards)
            for sample in split.train_samples
            if _contract_mask(sample)
        ],
        enabled=config.regression_standardization,
    )
    margin_standardization = _standardization(
        [_contract_margin(sample) for sample in split.train_samples if _contract_mask(sample)],
        enabled=config.regression_standardization,
    )
    model_config = BiddingMultiHeadQModelConfig(
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    )
    model = create_seeded_bidding_multi_head_q_model(model_config, seed=config.seed).to(
        device.torch_device
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_loader = _create_loader(
        split.train_samples,
        batch_size=config.batch_size,
        seed=config.seed,
        shuffle=True,
        napoleon_standardization=napoleon_standardization,
        margin_standardization=margin_standardization,
    )
    validation_loader = _create_loader(
        split.validation_samples,
        batch_size=config.batch_size,
        seed=config.seed,
        shuffle=False,
        napoleon_standardization=napoleon_standardization,
        margin_standardization=margin_standardization,
    )
    epoch_reports: list[dict[str, object]] = []
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    best_monitor = math.inf
    epochs_without_improvement = 0
    for epoch in range(1, config.epochs + 1):
        train_losses = _run_loss_epoch(
            model=model,
            dataloader=train_loader,
            config=config,
            device=device,
            optimizer=optimizer,
        )
        validation_losses = _run_loss_epoch(
            model=model,
            dataloader=validation_loader,
            config=config,
            device=device,
            optimizer=None,
        )
        monitor_value = validation_losses["totalLoss"]
        improved = monitor_value < best_monitor - config.min_delta
        if improved:
            best_monitor = monitor_value
            best_epoch = epoch
            best_state = copy.deepcopy(cpu_state_dict(model))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        epoch_reports.append(
            {
                "epoch": epoch,
                "train": train_losses,
                "validation": validation_losses,
                "monitor": monitor_value,
                "best": improved,
            }
        )
        if epochs_without_improvement >= config.patience:
            break
    if best_state is None:
        raise AssertionError("best checkpoint state was not captured.")
    model.load_state_dict(best_state)
    train_report = evaluate_bidding_multi_head_q_model(
        model,
        split.train_samples,
        train_samples=split.train_samples,
        device=device,
        napoleon_standardization=napoleon_standardization,
        margin_standardization=margin_standardization,
    )
    validation_report = evaluate_bidding_multi_head_q_model(
        model,
        split.validation_samples,
        train_samples=split.train_samples,
        device=device,
        napoleon_standardization=napoleon_standardization,
        margin_standardization=margin_standardization,
    )
    return BiddingMultiHeadQTrainResult(
        model=model,
        split=split,
        config=config,
        device=device,
        napoleon_standardization=napoleon_standardization,
        margin_standardization=margin_standardization,
        epoch_reports=epoch_reports,
        best_epoch=best_epoch,
        best_monitor_value=best_monitor,
        train_report=train_report,
        validation_report=validation_report,
        teacher_signal=teacher_signal_analysis(dataset.raw_samples),
    )


def bidding_multi_head_q_losses(
    output: BiddingMultiHeadQOutput,
    batch: BiddingMultiHeadQBatch,
    *,
    config: BiddingMultiHeadQTrainConfig,
) -> MultiHeadLosses:
    _validate_output_batch(output, batch)
    batch_index = torch.arange(batch.action_index.shape[0], device=batch.action_index.device)
    selected_role_logits = output.role_logits[batch_index, batch.action_index]
    role_loss = F.cross_entropy(selected_role_logits, batch.role_index)
    selected_napoleon = output.napoleon_point_cards[batch_index, batch.action_index]
    selected_success = output.success_logits[batch_index, batch.action_index]
    selected_margin = output.contract_margin[batch_index, batch.action_index]
    mask = batch.contract_mask
    napoleon_loss = _masked_mse(selected_napoleon, batch.napoleon_point_cards, mask)
    success_loss = _masked_bce(selected_success, batch.contract_success, mask)
    margin_loss = _masked_mse(selected_margin, batch.contract_margin, mask)
    total = (
        config.role_loss_weight * role_loss
        + config.napoleon_point_loss_weight * napoleon_loss
        + config.success_loss_weight * success_loss
        + config.margin_loss_weight * margin_loss
    )
    return MultiHeadLosses(
        total_loss=total,
        role_loss=role_loss,
        napoleon_point_loss=napoleon_loss,
        success_loss=success_loss,
        margin_loss=margin_loss,
    )


def evaluate_bidding_multi_head_q_model(
    model: BiddingMultiHeadQModel,
    samples: Iterable[BiddingQRawSample],
    *,
    train_samples: Iterable[BiddingQRawSample],
    device: ResolvedTorchDevice,
    napoleon_standardization: Standardization,
    margin_standardization: Standardization,
) -> dict[str, object]:
    sample_tuple = tuple(samples)
    train_tuple = tuple(train_samples)
    predictions = predict_multi_head_samples(
        model,
        sample_tuple,
        device=device,
        napoleon_standardization=napoleon_standardization,
        margin_standardization=margin_standardization,
    )
    baselines = baseline_metrics(train_tuple, sample_tuple)
    return {
        "rawSampleCount": len(sample_tuple),
        "stateCount": len({sample.state_key for sample in sample_tuple}),
        "role": role_metrics(sample_tuple, predictions["roleProbabilities"], train_tuple),
        "napoleonSidePointCards": point_metrics(
            sample_tuple,
            predictions["napoleonPointCards"],
            side="napoleon",
        ),
        "coalitionSidePointCards": point_metrics(
            sample_tuple,
            TOTAL_POINT_CARDS - predictions["napoleonPointCards"],
            side="coalition",
        ),
        "contractSuccess": success_metrics(sample_tuple, predictions["successProbabilities"]),
        "contractMargin": margin_metrics(sample_tuple, predictions["contractMargin"]),
        "baselines": baselines,
        "ranking": offline_score_metrics(sample_tuple, predictions),
        "coverage": _coverage(sample_tuple),
    }


def predict_multi_head_samples(
    model: BiddingMultiHeadQModel,
    samples: tuple[BiddingQRawSample, ...],
    *,
    device: ResolvedTorchDevice,
    napoleon_standardization: Standardization,
    margin_standardization: Standardization,
) -> dict[str, np.ndarray]:
    model.eval()
    role_batches: list[np.ndarray] = []
    napoleon_batches: list[np.ndarray] = []
    success_batches: list[np.ndarray] = []
    margin_batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(samples), 512):
            batch = samples[start : start + 512]
            model_input = torch.as_tensor(
                np.stack([sample.model_input for sample in batch]),
                dtype=torch.float32,
                device=device.torch_device,
            )
            output = model(model_input)
            role_batches.append(torch.softmax(output.role_logits, dim=2).detach().cpu().numpy())
            napoleon_batches.append(output.napoleon_point_cards.detach().cpu().numpy())
            success_batches.append(torch.sigmoid(output.success_logits).detach().cpu().numpy())
            margin_batches.append(output.contract_margin.detach().cpu().numpy())
    napoleon = napoleon_standardization.decode_array(np.concatenate(napoleon_batches, axis=0))
    margin = margin_standardization.decode_array(np.concatenate(margin_batches, axis=0))
    return {
        "roleProbabilities": np.concatenate(role_batches, axis=0),
        "napoleonPointCards": napoleon,
        "successProbabilities": np.concatenate(success_batches, axis=0),
        "contractMargin": margin,
    }


def role_metrics(
    samples: tuple[BiddingQRawSample, ...],
    role_probabilities: np.ndarray,
    train_samples: tuple[BiddingQRawSample, ...],
) -> dict[str, object]:
    selected = np.asarray(
        [
            role_probabilities[index, sample.forced_action_index]
            for index, sample in enumerate(samples)
        ],
        dtype=np.float64,
    )
    truth = np.asarray([role_index_for_terminal_role(sample.terminal_role) for sample in samples])
    predicted = selected.argmax(axis=1)
    confusion = np.zeros((BIDDING_ROLE_COUNT, BIDDING_ROLE_COUNT), dtype=np.int64)
    for true_index, pred_index in zip(truth, predicted, strict=True):
        confusion[int(true_index), int(pred_index)] += 1
    per_role = {}
    f1s = []
    for index, role in enumerate(BIDDING_ROLE_CLASSES):
        tp = float(confusion[index, index])
        fp = float(confusion[:, index].sum() - confusion[index, index])
        fn = float(confusion[index, :].sum() - confusion[index, index])
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _f1(precision, recall)
        f1s.append(f1)
        per_role[role] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(confusion[index, :].sum()),
        }
    clipped = np.clip(selected[np.arange(len(truth)), truth], 1e-12, 1.0)
    baselines = _role_baselines(train_samples, samples)
    return {
        "accuracy": float((predicted == truth).mean()),
        "macroF1": float(np.mean([value or 0.0 for value in f1s])),
        "crossEntropy": float(-np.log(clipped).mean()),
        "perRole": per_role,
        "confusionMatrix": confusion.tolist(),
        "baselines": baselines,
    }


def point_metrics(
    samples: tuple[BiddingQRawSample, ...],
    predictions: np.ndarray,
    *,
    side: Literal["napoleon", "coalition"],
) -> dict[str, object]:
    rows = _selected_contract_rows(samples, predictions)
    if side == "napoleon":
        truth = np.asarray([float(sample.napoleon_side_point_cards) for sample, _ in rows])
    else:
        truth = np.asarray([float(sample.coalition_side_point_cards) for sample, _ in rows])
    pred = np.asarray([prediction for _, prediction in rows], dtype=np.float64)
    return {
        "overall": _regression_metrics(pred, truth),
        "byRole": _grouped_regression(samples, predictions, "role", side=side),
        "bySuit": _grouped_regression(samples, predictions, "suit", side=side),
        "byTarget": _grouped_regression(samples, predictions, "target", side=side),
    }


def success_metrics(
    samples: tuple[BiddingQRawSample, ...],
    probabilities: np.ndarray,
) -> dict[str, object]:
    rows = _selected_contract_rows(samples, probabilities)
    truth = np.asarray([1.0 if sample.contract_success else 0.0 for sample, _ in rows])
    prob = np.asarray([prediction for _, prediction in rows], dtype=np.float64)
    predicted = (prob >= 0.5).astype(np.float64)
    tp = float(((predicted == 1.0) & (truth == 1.0)).sum())
    fp = float(((predicted == 1.0) & (truth == 0.0)).sum())
    fn = float(((predicted == 0.0) & (truth == 1.0)).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "accuracy": float((predicted == truth).mean()) if truth.size else None,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "rocAuc": _binary_auc(truth, prob),
        "prAuc": _pr_auc(truth, prob),
        "brier": float(np.mean((prob - truth) ** 2)) if truth.size else None,
        "calibration": _calibration_bins(truth, prob),
        "positiveRate": float(truth.mean()) if truth.size else None,
        "sampleCount": int(truth.size),
    }


def margin_metrics(
    samples: tuple[BiddingQRawSample, ...],
    predictions: np.ndarray,
) -> dict[str, object]:
    rows = _selected_contract_rows(samples, predictions)
    truth = np.asarray([_contract_margin(sample) for sample, _ in rows])
    pred = np.asarray([prediction for _, prediction in rows], dtype=np.float64)
    return {
        "overall": {
            **_regression_metrics(pred, truth),
            "signAccuracy": float(((pred >= 0.0) == (truth >= 0.0)).mean()) if truth.size else None,
        },
        "bySuit": _grouped_margin(samples, predictions, "suit"),
        "byTarget": _grouped_margin(samples, predictions, "target"),
    }


def baseline_metrics(
    train_samples: tuple[BiddingQRawSample, ...],
    validation_samples: tuple[BiddingQRawSample, ...],
) -> dict[str, object]:
    return {
        "napoleonSidePointCards": _regression_baselines(
            train_samples,
            validation_samples,
            lambda sample: float(sample.napoleon_side_point_cards),
        ),
        "coalitionSidePointCards": _regression_baselines(
            train_samples,
            validation_samples,
            lambda sample: float(sample.coalition_side_point_cards),
        ),
        "contractMargin": _regression_baselines(
            train_samples,
            validation_samples,
            _contract_margin,
        ),
        "contractSuccess": _success_baselines(train_samples, validation_samples),
    }


def teacher_signal_analysis(samples: Iterable[BiddingQRawSample]) -> dict[str, object]:
    sample_tuple = tuple(samples)
    by_state = _samples_by_state(sample_tuple)
    return {
        "napoleonSidePointCards": _teacher_numeric_signal(
            by_state,
            lambda sample: float(sample.napoleon_side_point_cards),
        ),
        "contractSuccess": _teacher_success_signal(by_state),
        "contractMargin": _teacher_numeric_signal(by_state, _contract_margin),
        "coalitionDerive": _coalition_derive_check(sample_tuple),
    }


def offline_score_metrics(
    samples: tuple[BiddingQRawSample, ...],
    predictions: dict[str, np.ndarray],
) -> dict[str, object]:
    role_prob = predictions["roleProbabilities"]
    napoleon = predictions["napoleonPointCards"]
    success = predictions["successProbabilities"]
    margin = predictions["contractMargin"]
    coalition = TOTAL_POINT_CARDS - napoleon
    side_prob = (
        role_prob[:, :, BIDDING_ROLE_CLASSES.index("napoleon")]
        + role_prob[:, :, BIDDING_ROLE_CLASSES.index("napoleon-adjutant")]
        + role_prob[:, :, BIDDING_ROLE_CLASSES.index("adjutant")]
    )
    role_aware = side_prob * napoleon + (1.0 - side_prob) * coalition
    return {
        "napoleonSideCards": _ranking_for_score(
            samples,
            napoleon,
            teacher=lambda sample: float(sample.napoleon_side_point_cards),
            teacher_name="napoleonSidePointCards",
        ),
        "contractSuccess": _ranking_for_score(
            samples,
            success,
            teacher=lambda sample: 1.0 if sample.contract_success else 0.0,
            teacher_name="contractSuccess",
        ),
        "contractMargin": _ranking_for_score(
            samples,
            margin,
            teacher=_contract_margin,
            teacher_name="contractMargin",
        ),
        "roleAware": _ranking_for_score(
            samples,
            role_aware,
            teacher=lambda sample: float(sample.candidate_team_point_cards),
            teacher_name="candidateTeamPointCards",
        ),
    }


def save_bidding_multi_head_q_artifact(
    output_directory: Path | str,
    *,
    result: BiddingMultiHeadQTrainResult,
    dataset: BiddingQDataset,
) -> dict[str, object]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    metadata_path = output / "metadata.json"
    report_path = output / "report.json"
    checkpoint = _checkpoint_dict(result=result, dataset=dataset)
    torch.save(checkpoint, checkpoint_path)
    metadata = _metadata_dict(result=result, dataset=dataset, checkpoint_path=checkpoint_path)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "teacherSignal": result.teacher_signal,
        "epochs": result.epoch_reports,
        "bestEpoch": result.best_epoch,
        "bestMonitorValue": result.best_monitor_value,
        "train": result.train_report,
        "validation": result.validation_report,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "checkpointPath": str(checkpoint_path),
        "metadataPath": str(metadata_path),
        "reportPath": str(report_path),
    }


def load_bidding_multi_head_q_checkpoint(
    path: Path | str,
) -> tuple[BiddingMultiHeadQModel, dict[str, object]]:
    try:
        raw = torch.load(Path(path), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, pickle.UnpicklingError) as error:
        raise BiddingMultiHeadQCheckpointError(f"checkpoint cannot be loaded: {error}") from error
    if not isinstance(raw, dict):
        raise BiddingMultiHeadQCheckpointError("checkpoint must be a dictionary.")
    if raw.get("checkpointSchemaVersion") != BIDDING_MULTI_HEAD_Q_CHECKPOINT_SCHEMA_VERSION:
        raise BiddingMultiHeadQCheckpointError("checkpoint schema version mismatch.")
    if raw.get("modelType") != BIDDING_MULTI_HEAD_Q_MODEL_TYPE:
        raise BiddingMultiHeadQCheckpointError("checkpoint modelType mismatch.")
    config_raw = raw.get("modelConfig")
    if not isinstance(config_raw, dict):
        raise BiddingMultiHeadQCheckpointError("checkpoint modelConfig must be an object.")
    model = BiddingMultiHeadQModel(BiddingMultiHeadQModelConfig.from_dict(config_raw))
    state = raw.get("modelState")
    if not isinstance(state, dict):
        raise BiddingMultiHeadQCheckpointError("checkpoint modelState must be a state dict.")
    model.load_state_dict(state)
    return model, raw


def _run_loss_epoch(
    *,
    model: BiddingMultiHeadQModel,
    dataloader: DataLoader[BiddingMultiHeadQBatch],
    config: BiddingMultiHeadQTrainConfig,
    device: ResolvedTorchDevice,
    optimizer: optim.Optimizer | None,
) -> dict[str, float]:
    model.train(optimizer is not None)
    totals = Counter()
    sample_total = 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for batch in dataloader:
            moved = _move_batch(batch, device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            output = model(moved.model_input)
            losses = bidding_multi_head_q_losses(output, moved, config=config)
            if optimizer is not None:
                losses.total_loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
            batch_size = int(moved.action_index.shape[0])
            totals["totalLoss"] += float(losses.total_loss.item()) * batch_size
            totals["roleLoss"] += float(losses.role_loss.item()) * batch_size
            totals["napoleonPointLoss"] += float(losses.napoleon_point_loss.item()) * batch_size
            totals["successLoss"] += float(losses.success_loss.item()) * batch_size
            totals["marginLoss"] += float(losses.margin_loss.item()) * batch_size
            sample_total += batch_size
    if sample_total == 0:
        raise ValueError("split contains no samples.")
    return {key: float(value) / sample_total for key, value in totals.items()}


def _move_batch(
    batch: BiddingMultiHeadQBatch,
    device: ResolvedTorchDevice,
) -> BiddingMultiHeadQBatch:
    return BiddingMultiHeadQBatch(
        model_input=batch.model_input.to(device=device.torch_device, dtype=torch.float32),
        action_index=batch.action_index.to(device=device.torch_device, dtype=torch.long),
        role_index=batch.role_index.to(device=device.torch_device, dtype=torch.long),
        napoleon_point_cards=batch.napoleon_point_cards.to(
            device=device.torch_device, dtype=torch.float32
        ),
        coalition_point_cards=batch.coalition_point_cards.to(
            device=device.torch_device, dtype=torch.float32
        ),
        contract_success=batch.contract_success.to(device=device.torch_device, dtype=torch.float32),
        contract_margin=batch.contract_margin.to(device=device.torch_device, dtype=torch.float32),
        contract_mask=batch.contract_mask.to(device=device.torch_device, dtype=torch.bool),
    )


def _create_loader(
    samples: Iterable[BiddingQRawSample],
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
    napoleon_standardization: Standardization,
    margin_standardization: Standardization,
) -> DataLoader[BiddingMultiHeadQBatch]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        _BiddingMultiHeadQDataset(
            samples,
            napoleon_standardization=napoleon_standardization,
            margin_standardization=margin_standardization,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def _validate_output_batch(output: BiddingMultiHeadQOutput, batch: BiddingMultiHeadQBatch) -> None:
    if output.role_logits.shape != (
        batch.action_index.shape[0],
        BIDDING_ACTION_COUNT,
        BIDDING_ROLE_COUNT,
    ):
        raise ValueError("role logits must have shape (batch, 29, roleCount).")
    for name, tensor in (
        ("napoleon_point_cards", output.napoleon_point_cards),
        ("success_logits", output.success_logits),
        ("contract_margin", output.contract_margin),
    ):
        if tensor.shape != (batch.action_index.shape[0], BIDDING_ACTION_COUNT):
            raise ValueError(f"{name} must have shape (batch, 29).")


def _masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if mask.ndim != 1:
        raise ValueError("mask must be one-dimensional.")
    if not bool(mask.any().item()):
        return prediction.sum() * 0.0
    return F.mse_loss(prediction[mask], target[mask])


def _masked_bce(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if not bool(mask.any().item()):
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[mask], target[mask])


def _selected_contract_rows(
    samples: tuple[BiddingQRawSample, ...],
    predictions: np.ndarray,
) -> list[tuple[BiddingQRawSample, float]]:
    rows = []
    for index, sample in enumerate(samples):
        if _contract_mask(sample):
            rows.append((sample, float(predictions[index, sample.forced_action_index])))
    return rows


def _contract_mask(sample: BiddingQRawSample) -> bool:
    return sample.result_type == "standard" and sample.final_declared_target is not None


def _contract_margin(sample: BiddingQRawSample) -> float:
    if (
        not _contract_mask(sample)
        or sample.napoleon_side_point_cards is None
        or sample.final_declared_target is None
    ):
        raise ValueError("sample has no contract margin target.")
    return float(sample.napoleon_side_point_cards - sample.final_declared_target)


def _regression_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, object]:
    if truth.size == 0:
        return {"sampleCount": 0, "mae": None, "rmse": None, "pearsonCorrelation": None}
    error = prediction - truth
    return {
        "sampleCount": int(truth.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "pearsonCorrelation": _pearson(prediction, truth),
    }


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _grouped_regression(
    samples: tuple[BiddingQRawSample, ...],
    predictions: np.ndarray,
    group: Literal["role", "suit", "target"],
    *,
    side: Literal["napoleon", "coalition"],
) -> dict[str, object]:
    buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for index, sample in enumerate(samples):
        if not _contract_mask(sample):
            continue
        key = _group_key(sample, group)
        if key is None:
            continue
        truth = (
            sample.napoleon_side_point_cards
            if side == "napoleon"
            else sample.coalition_side_point_cards
        )
        buckets[key].append((float(predictions[index, sample.forced_action_index]), float(truth)))
    return {
        key: _regression_metrics(
            np.asarray([item[0] for item in values]),
            np.asarray([item[1] for item in values]),
        )
        for key, values in sorted(buckets.items())
    }


def _grouped_margin(
    samples: tuple[BiddingQRawSample, ...],
    predictions: np.ndarray,
    group: Literal["suit", "target"],
) -> dict[str, object]:
    buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for index, sample in enumerate(samples):
        if not _contract_mask(sample):
            continue
        key = _group_key(sample, group)
        if key is None:
            continue
        buckets[key].append(
            (float(predictions[index, sample.forced_action_index]), _contract_margin(sample))
        )
    return {
        key: {
            **_regression_metrics(
                np.asarray([item[0] for item in values]),
                np.asarray([item[1] for item in values]),
            ),
            "signAccuracy": float(
                np.mean(
                    (np.asarray([item[0] for item in values]) >= 0.0)
                    == (np.asarray([item[1] for item in values]) >= 0.0)
                )
            ),
        }
        for key, values in sorted(buckets.items())
    }


def _group_key(sample: BiddingQRawSample, group: str) -> str | None:
    if group == "role":
        return canonical_bidding_role(sample.terminal_role)
    if group == "suit":
        return sample.forced_suit
    if group == "target":
        return (
            str(sample.forced_target_point_cards)
            if sample.forced_target_point_cards is not None
            else None
        )
    raise ValueError(f"unsupported group: {group}.")


def _role_baselines(
    train_samples: tuple[BiddingQRawSample, ...],
    validation_samples: tuple[BiddingQRawSample, ...],
) -> dict[str, object]:
    majority = Counter(
        role_index_for_terminal_role(sample.terminal_role) for sample in train_samples
    ).most_common(1)[0][0]
    action_counts: dict[int, Counter[int]] = defaultdict(Counter)
    for sample in train_samples:
        action_counts[sample.forced_action_index][
            role_index_for_terminal_role(sample.terminal_role)
        ] += 1
    truth = np.asarray(
        [role_index_for_terminal_role(sample.terminal_role) for sample in validation_samples]
    )
    majority_pred = np.full_like(truth, majority)
    action_pred = np.asarray(
        [
            action_counts[sample.forced_action_index].most_common(1)[0][0]
            if action_counts[sample.forced_action_index]
            else majority
            for sample in validation_samples
        ]
    )
    return {
        "majorityRole": {"accuracy": float((majority_pred == truth).mean())},
        "actionIndexRoleFrequency": {"accuracy": float((action_pred == truth).mean())},
    }


def _regression_baselines(
    train_samples: tuple[BiddingQRawSample, ...],
    validation_samples: tuple[BiddingQRawSample, ...],
    target_fn: object,
) -> dict[str, object]:
    fn = cast(Callable[[BiddingQRawSample], float], target_fn)
    train_contract = [sample for sample in train_samples if _contract_mask(sample)]
    validation_contract = [sample for sample in validation_samples if _contract_mask(sample)]
    global_mean = float(np.mean([fn(sample) for sample in train_contract]))
    maps = {
        "actionIndexMean": _mean_map(
            train_contract, lambda sample: str(sample.forced_action_index), fn
        ),
        "roleMean": _mean_map(
            train_contract, lambda sample: canonical_bidding_role(sample.terminal_role), fn
        ),
        "roleActionMean": _mean_map(
            train_contract,
            lambda sample: (
                f"{canonical_bidding_role(sample.terminal_role)}:{sample.forced_action_index}"
            ),
            fn,
        ),
        "suitTargetMean": _mean_map(
            train_contract,
            lambda sample: f"{sample.forced_suit}:{sample.forced_target_point_cards}",
            fn,
        ),
    }
    truth = np.asarray([fn(sample) for sample in validation_contract])
    result: dict[str, object] = {
        "globalMean": _baseline_regression(validation_contract, truth, lambda _sample: global_mean)
    }
    for name, mapping in maps.items():
        result[name] = _baseline_regression(
            validation_contract,
            truth,
            lambda sample, local_mapping=mapping, local_name=name: local_mapping.get(
                _baseline_key(local_name, sample),
                global_mean,
            ),
        )
    return result


def _baseline_key(name: str, sample: BiddingQRawSample) -> str:
    if name == "actionIndexMean":
        return str(sample.forced_action_index)
    if name == "roleMean":
        return canonical_bidding_role(sample.terminal_role)
    if name == "roleActionMean":
        return f"{canonical_bidding_role(sample.terminal_role)}:{sample.forced_action_index}"
    if name == "suitTargetMean":
        return f"{sample.forced_suit}:{sample.forced_target_point_cards}"
    raise ValueError(name)


def _baseline_regression(
    samples: list[BiddingQRawSample],
    truth: np.ndarray,
    predict_fn: object,
) -> dict[str, object]:
    fn = cast(Callable[[BiddingQRawSample], float], predict_fn)
    prediction = np.asarray([fn(sample) for sample in samples], dtype=np.float64)
    return _regression_metrics(prediction, truth)


def _success_baselines(
    train_samples: tuple[BiddingQRawSample, ...],
    validation_samples: tuple[BiddingQRawSample, ...],
) -> dict[str, object]:
    train_contract = [sample for sample in train_samples if _contract_mask(sample)]
    validation_contract = [sample for sample in validation_samples if _contract_mask(sample)]
    global_rate = float(
        np.mean([1.0 if sample.contract_success else 0.0 for sample in train_contract])
    )
    maps = {
        "targetSuccessRate": _mean_map(
            train_contract,
            lambda sample: str(sample.final_declared_target),
            lambda sample: 1.0 if sample.contract_success else 0.0,
        ),
        "actionIndexSuccessRate": _mean_map(
            train_contract,
            lambda sample: str(sample.forced_action_index),
            lambda sample: 1.0 if sample.contract_success else 0.0,
        ),
        "suitTargetSuccessRate": _mean_map(
            train_contract,
            lambda sample: f"{sample.final_declared_suit}:{sample.final_declared_target}",
            lambda sample: 1.0 if sample.contract_success else 0.0,
        ),
    }
    truth = np.asarray([1.0 if sample.contract_success else 0.0 for sample in validation_contract])
    result: dict[str, object] = {
        "globalSuccessRate": _classification_from_probability(
            validation_contract, truth, lambda _sample: global_rate
        )
    }
    for name, mapping in maps.items():
        result[name] = _classification_from_probability(
            validation_contract,
            truth,
            lambda sample, local_mapping=mapping, local_name=name: local_mapping.get(
                _success_key(local_name, sample),
                global_rate,
            ),
        )
    return result


def _success_key(name: str, sample: BiddingQRawSample) -> str:
    if name == "targetSuccessRate":
        return str(sample.final_declared_target)
    if name == "actionIndexSuccessRate":
        return str(sample.forced_action_index)
    if name == "suitTargetSuccessRate":
        return f"{sample.final_declared_suit}:{sample.final_declared_target}"
    raise ValueError(name)


def _classification_from_probability(
    samples: list[BiddingQRawSample],
    truth: np.ndarray,
    predict_fn: object,
) -> dict[str, object]:
    fn = cast(Callable[[BiddingQRawSample], float], predict_fn)
    probability = np.asarray([fn(sample) for sample in samples], dtype=np.float64)
    predicted = (probability >= 0.5).astype(np.float64)
    tp = float(((predicted == 1.0) & (truth == 1.0)).sum())
    fp = float(((predicted == 1.0) & (truth == 0.0)).sum())
    fn_count = float(((predicted == 0.0) & (truth == 1.0)).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn_count)
    return {
        "accuracy": float((predicted == truth).mean()) if truth.size else None,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "rocAuc": _binary_auc(truth, probability),
        "prAuc": _pr_auc(truth, probability),
    }


def _mean_map(
    samples: Iterable[BiddingQRawSample], key_fn: object, value_fn: object
) -> dict[str, float]:
    key = cast(Callable[[BiddingQRawSample], str], key_fn)
    value = cast(Callable[[BiddingQRawSample], float], value_fn)
    buckets: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        buckets[key(sample)].append(float(value(sample)))
    return {item_key: float(np.mean(values)) for item_key, values in buckets.items()}


def _ranking_for_score(
    samples: tuple[BiddingQRawSample, ...],
    score: np.ndarray,
    *,
    teacher: object,
    teacher_name: str,
) -> dict[str, object]:
    teacher_fn = cast(Callable[[BiddingQRawSample], float], teacher)
    state_groups = _samples_by_state([sample for sample in samples if _contract_mask(sample)])
    best_hits = 0
    top3_hits = 0
    pair_correct = 0.0
    pair_total = 0
    pass_correct = 0
    pass_total = 0
    strongest_matches = 0
    strongest_total = 0
    best_suits = Counter()
    best_targets = Counter()
    state_count = 0
    sample_index = {id(sample): index for index, sample in enumerate(samples)}
    for state_samples in state_groups.values():
        if len(state_samples) < 2:
            continue
        state_count += 1
        teachers = np.asarray([teacher_fn(sample) for sample in state_samples])
        predicted = np.asarray(
            [
                score[sample_index[id(sample)], sample.forced_action_index]
                for sample in state_samples
            ]
        )
        teacher_best = np.flatnonzero(teachers == teachers.max())
        pred_order = np.argsort(-predicted)
        pred_best = int(pred_order[0])
        if pred_best in teacher_best:
            best_hits += 1
        if any(index in teacher_best for index in pred_order[:3]):
            top3_hits += 1
        for left in range(len(state_samples)):
            for right in range(left + 1, len(state_samples)):
                diff = teachers[left] - teachers[right]
                if diff == 0:
                    continue
                pred_diff = predicted[left] - predicted[right]
                pair_total += 1
                if pred_diff == 0:
                    pair_correct += 0.5
                elif (pred_diff > 0) == (diff > 0):
                    pair_correct += 1.0
        pass_indices = [
            index for index, sample in enumerate(state_samples) if sample.forced_action_index == 0
        ]
        bid_indices = [
            index for index, sample in enumerate(state_samples) if sample.forced_action_index != 0
        ]
        if pass_indices and bid_indices:
            pass_total += 1
            teacher_prefers_bid = teachers[bid_indices].max() > teachers[pass_indices[0]]
            pred_prefers_bid = predicted[bid_indices].max() > predicted[pass_indices[0]]
            if teacher_prefers_bid == pred_prefers_bid:
                pass_correct += 1
        best_sample = state_samples[pred_best]
        if best_sample.forced_suit is not None:
            best_suits[best_sample.forced_suit] += 1
            if best_sample.forced_suit == best_sample.strongest_suit:
                strongest_matches += 1
            strongest_total += 1
        if best_sample.forced_target_point_cards is not None:
            best_targets[str(best_sample.forced_target_point_cards)] += 1
    return {
        "teacherBestDefinition": teacher_name,
        "stateCount": state_count,
        "bestActionHitRate": _safe_div(best_hits, state_count),
        "top3HitRate": _safe_div(top3_hits, state_count),
        "pairwiseRankingAccuracy": _safe_div(pair_correct, pair_total),
        "passVsBestBidAccuracy": _safe_div(pass_correct, pass_total),
        "predictedBestSuitDistribution": dict(sorted(best_suits.items())),
        "predictedBestTargetDistribution": dict(sorted(best_targets.items())),
        "strongestSuitMatchRate": _safe_div(strongest_matches, strongest_total),
    }


def _teacher_numeric_signal(
    by_state: dict[str, list[BiddingQRawSample]],
    value_fn: object,
) -> dict[str, object]:
    fn = cast(Callable[[BiddingQRawSample], float], value_fn)
    same_target_gaps = []
    same_target_ties = 0
    same_target_total = 0
    strongest_best = 0
    strongest_total = 0
    target_gaps = []
    for state_samples in by_state.values():
        contract_samples = [
            sample
            for sample in state_samples
            if _contract_mask(sample) and sample.forced_action_index != 0
        ]
        by_target: dict[int, list[BiddingQRawSample]] = defaultdict(list)
        for sample in contract_samples:
            if sample.forced_target_point_cards is not None:
                by_target[sample.forced_target_point_cards].append(sample)
        target_means = []
        for target_samples in by_target.values():
            if len(target_samples) < 2:
                continue
            values = np.asarray([fn(sample) for sample in target_samples])
            gap = float(values.max() - values.min())
            same_target_gaps.append(gap)
            same_target_total += 1
            same_target_ties += int(gap == 0.0)
            best_suits = {
                sample.forced_suit
                for sample, value in zip(target_samples, values, strict=True)
                if value == values.max()
            }
            strongest = target_samples[0].strongest_suit
            if strongest in best_suits:
                strongest_best += 1
            strongest_total += 1
            target_means.append(float(values.mean()))
        if len(target_means) >= 2:
            target_gaps.append(float(max(target_means) - min(target_means)))
    return {
        "sameTargetSuitBestWorstGapMean": _mean_or_none(same_target_gaps),
        "sameTargetSuitExactTieRate": _safe_div(same_target_ties, same_target_total),
        "strongestSuitTeacherBestRate": _safe_div(strongest_best, strongest_total),
        "targetGapMean": _mean_or_none(target_gaps),
        "stateTargetBucketCount": same_target_total,
    }


def _teacher_success_signal(by_state: dict[str, list[BiddingQRawSample]]) -> dict[str, object]:
    action_change = 0
    action_total = 0
    suit_change = 0
    suit_total = 0
    target_change = 0
    target_total = 0
    for state_samples in by_state.values():
        contract_samples = [sample for sample in state_samples if _contract_mask(sample)]
        if len(contract_samples) >= 2:
            values = {sample.contract_success for sample in contract_samples}
            action_change += int(len(values) > 1)
            action_total += 1
        by_target: dict[int, set[bool]] = defaultdict(set)
        by_suit: dict[SuitName, set[bool]] = defaultdict(set)
        for sample in contract_samples:
            if sample.forced_target_point_cards is not None and sample.forced_suit is not None:
                by_target[sample.forced_target_point_cards].add(sample.contract_success)
                by_suit[sample.forced_suit].add(sample.contract_success)
        if len(by_suit) >= 2:
            suit_change += int(any(len(values) > 1 for values in by_target.values()))
            suit_total += 1
        if len(by_target) >= 2:
            target_rates = [float(any(values)) for values in by_target.values()]
            target_change += int(max(target_rates) != min(target_rates))
            target_total += 1
    return {
        "stateActionSuccessChangesRate": _safe_div(action_change, action_total),
        "suitChangeSuccessChangesRate": _safe_div(suit_change, suit_total),
        "targetChangeSuccessChangesRate": _safe_div(target_change, target_total),
        "stateCount": action_total,
    }


def _coalition_derive_check(samples: tuple[BiddingQRawSample, ...]) -> dict[str, object]:
    checked = 0
    mismatches = 0
    totals = Counter()
    for sample in samples:
        if sample.napoleon_side_point_cards is None or sample.coalition_side_point_cards is None:
            continue
        checked += 1
        total = sample.napoleon_side_point_cards + sample.coalition_side_point_cards
        totals[str(total)] += 1
        mismatches += int(total != int(TOTAL_POINT_CARDS))
    return {
        "derivedAsTotalPointCardsMinusNapoleon": mismatches == 0 and checked > 0,
        "totalPointCards": int(TOTAL_POINT_CARDS),
        "checkedSamples": checked,
        "mismatches": mismatches,
        "observedTotals": dict(sorted(totals.items())),
    }


def _binary_auc(truth: np.ndarray, score: np.ndarray) -> float | None:
    positive = int(truth.sum())
    negative = int(truth.size - positive)
    if positive == 0 or negative == 0:
        return None
    ranks = _average_ranks(score)
    return float(
        (ranks[truth == 1.0].sum() - positive * (positive + 1) / 2) / (positive * negative)
    )


def _average_ranks(score: np.ndarray) -> np.ndarray:
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(score.shape[0], dtype=np.float64)
    start = 0
    while start < score.shape[0]:
        end = start + 1
        while end < score.shape[0] and score[order[end]] == score[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _pr_auc(truth: np.ndarray, score: np.ndarray) -> float | None:
    if truth.size == 0 or truth.sum() == 0:
        return None
    order = np.argsort(-score)
    sorted_truth = truth[order]
    tp = np.cumsum(sorted_truth)
    fp = np.cumsum(1.0 - sorted_truth)
    recall = tp / max(float(truth.sum()), 1.0)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.trapezoid(precision, recall))


def _calibration_bins(
    truth: np.ndarray, probability: np.ndarray, bins: int = 10
) -> list[dict[str, object]]:
    result = []
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        mask = (probability >= lower) & (
            probability < upper if index < bins - 1 else probability <= upper
        )
        result.append(
            {
                "lower": lower,
                "upper": upper,
                "count": int(mask.sum()),
                "meanPrediction": float(probability[mask].mean()) if mask.any() else None,
                "observedRate": float(truth[mask].mean()) if mask.any() else None,
            }
        )
    return result


def _coverage(samples: tuple[BiddingQRawSample, ...]) -> dict[str, object]:
    return {
        "contractSamples": sum(1 for sample in samples if _contract_mask(sample)),
        "noContractSamples": sum(1 for sample in samples if not _contract_mask(sample)),
        "actionCounts": dict(Counter(str(sample.forced_action_index) for sample in samples)),
    }


def _standardization(values: list[float], *, enabled: bool) -> Standardization:
    if not values:
        raise ValueError("cannot standardize empty target values.")
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std <= 0.0:
        std = 1.0
    return Standardization(enabled=enabled, mean=mean, std=std)


def _state_keys_in_source_order(samples: tuple[BiddingQRawSample, ...]) -> tuple[str, ...]:
    seen = set()
    keys = []
    for sample in samples:
        if sample.state_key not in seen:
            seen.add(sample.state_key)
            keys.append(sample.state_key)
    return tuple(keys)


def _samples_by_state(samples: Iterable[BiddingQRawSample]) -> dict[str, list[BiddingQRawSample]]:
    groups: dict[str, list[BiddingQRawSample]] = defaultdict(list)
    for sample in samples:
        groups[sample.state_key].append(sample)
    return groups


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0.0:
        return None
    return float(2.0 * precision * recall / (precision + recall))


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _validate_train_config(config: BiddingMultiHeadQTrainConfig) -> None:
    if config.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if config.patience <= 0:
        raise ValueError("patience must be positive.")
    for name, weight in (
        ("role_loss_weight", config.role_loss_weight),
        ("napoleon_point_loss_weight", config.napoleon_point_loss_weight),
        ("success_loss_weight", config.success_loss_weight),
        ("margin_loss_weight", config.margin_loss_weight),
    ):
        if weight < 0.0:
            raise ValueError(f"{name} must be non-negative.")


def _checkpoint_dict(
    *,
    result: BiddingMultiHeadQTrainResult,
    dataset: BiddingQDataset,
) -> dict[str, object]:
    return {
        "checkpointSchemaVersion": BIDDING_MULTI_HEAD_Q_CHECKPOINT_SCHEMA_VERSION,
        "modelType": BIDDING_MULTI_HEAD_Q_MODEL_TYPE,
        "architectureId": BIDDING_MULTI_HEAD_Q_ARCHITECTURE_ID,
        "modelConfig": result.model.config.to_dict(),
        "modelState": cpu_state_dict(result.model),
        "trainingConfig": result.config.to_dict(),
        "napoleonStandardization": result.napoleon_standardization.to_dict(),
        "marginStandardization": result.margin_standardization.to_dict(),
        "bestEpoch": result.best_epoch,
        "bestMonitorValue": result.best_monitor_value,
        "dataset": {
            "path": str(dataset.manifest.dataset_directory),
            "sourceStates": dataset.manifest.source_states,
            "sampleCount": dataset.manifest.sample_count,
            "manifestSha256": dataset.manifest.manifest_sha256,
        },
        "split": {
            "trainStateCount": len(result.split.train_state_keys),
            "validationStateCount": len(result.split.validation_state_keys),
            "validationStateKeyHash": result.split.validation_state_key_hash,
        },
    }


def _metadata_dict(
    *,
    result: BiddingMultiHeadQTrainResult,
    dataset: BiddingQDataset,
    checkpoint_path: Path,
) -> dict[str, object]:
    checkpoint = _checkpoint_dict(result=result, dataset=dataset)
    checkpoint.pop("modelState", None)
    checkpoint["checkpointPath"] = str(checkpoint_path)
    return checkpoint
