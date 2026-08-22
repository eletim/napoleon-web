"""Training and diagnostics for Issue #395 PASS role binary classifier."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import pickle
import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from torch import Tensor, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from napoleon_ml.policy.device import (
    RequestedTorchDevice,
    ResolvedTorchDevice,
    cpu_state_dict,
    resolve_torch_device,
)

from .dataset import BiddingQDataset, BiddingQRawSample, action_target
from .multi_head_training import _binary_auc, _calibration_bins, _pr_auc, _safe_div
from .pass_role_binary_model import (
    BIDDING_PASS_ROLE_BINARY_ARCHITECTURE_ID,
    BiddingPassRoleBinaryModel,
    BiddingPassRoleBinaryModelConfig,
    create_seeded_bidding_pass_role_binary_model,
)
from .role_training import canonical_bidding_role

BIDDING_PASS_ROLE_BINARY_CHECKPOINT_SCHEMA_VERSION = 1
BIDDING_PASS_ROLE_BINARY_MODEL_TYPE = "bidding-pass-role-binary"


class BiddingPassRoleBinaryCheckpointError(ValueError):
    """Raised when a PASS role binary checkpoint is incompatible."""


@dataclass(frozen=True)
class BiddingPassRoleBinaryTrainConfig:
    seed: int = 395
    epochs: int = 80
    batch_size: int = 64
    learning_rate: float = 1e-3
    hidden_dims: tuple[int, ...] = (512, 512, 256, 256)
    dropout: float = 0.0
    train_ratio: float = 0.75
    pos_weight: float | None = None
    weight_decay: float = 1e-4
    patience: int = 10
    min_delta: float = 0.0
    device: RequestedTorchDevice = "cpu"
    optimizer: str = "AdamW"

    def resolved_pos_weight(self, train_samples: tuple[BiddingQRawSample, ...]) -> float | None:
        if self.pos_weight is not None:
            return self.pos_weight
        labels = [pass_binary_label(sample) for sample in train_samples]
        positives = sum(1 for label in labels if label == 1)
        negatives = sum(1 for label in labels if label == 0)
        if positives == 0 or negatives == 0:
            return None
        return negatives / positives

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["hidden_dims"] = list(self.hidden_dims)
        data["algorithm"] = "pass-adjutant-vs-citizen-binary-bce-v1"
        return data


@dataclass(frozen=True)
class BiddingPassRoleBinarySplit:
    train_state_keys: frozenset[str]
    calibration_state_keys: frozenset[str]
    train_samples: tuple[BiddingQRawSample, ...]
    calibration_samples: tuple[BiddingQRawSample, ...]
    state_key_hash: str


@dataclass(frozen=True)
class BiddingPassRoleBinaryTrainResult:
    model: BiddingPassRoleBinaryModel
    split: BiddingPassRoleBinarySplit
    config: BiddingPassRoleBinaryTrainConfig
    device: ResolvedTorchDevice
    best_epoch: int
    best_monitor_value: float
    epoch_reports: list[dict[str, object]]
    train_report: dict[str, Any]
    calibration_report: dict[str, Any]
    audit: dict[str, Any]
    pos_weight: float | None


class BiddingPassRoleBinaryBatch(NamedTuple):
    model_input: Tensor
    label: Tensor


class _PassRoleDataset(Dataset[BiddingPassRoleBinaryBatch]):
    def __init__(self, samples: Iterable[BiddingQRawSample]) -> None:
        self.samples = tuple(sample for sample in samples if pass_binary_label(sample) is not None)
        if not self.samples:
            raise ValueError("PASS binary dataset has no Citizen/Adjutant labels.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> BiddingPassRoleBinaryBatch:
        sample = self.samples[index]
        label = pass_binary_label(sample)
        if label is None:
            raise AssertionError("unlabeled sample leaked into PASS role dataset.")
        return BiddingPassRoleBinaryBatch(
            model_input=torch.as_tensor(sample.model_input, dtype=torch.float32),
            label=torch.tensor(float(label), dtype=torch.float32),
        )


def pass_binary_label(sample: BiddingQRawSample) -> int | None:
    if sample.forced_action_index != 0:
        return None
    role = canonical_bidding_role(sample.terminal_role)
    if role == "adjutant":
        return 1
    if role == "citizen":
        return 0
    return None


def audit_pass_binary_dataset(samples: Iterable[BiddingQRawSample]) -> dict[str, Any]:
    pass_samples = [sample for sample in samples if sample.forced_action_index == 0]
    roles = Counter(canonical_bidding_role(sample.terminal_role) for sample in pass_samples)
    labeled = roles["adjutant"] + roles["citizen"]
    strength = Counter(_strength_bucket(sample.strongest_suit_score) for sample in pass_samples)
    current_bid = Counter(_current_bid_level(sample) for sample in pass_samples)
    strongest_suit = Counter(sample.strongest_suit for sample in pass_samples)
    seat = Counter(str(sample.candidate_seat_index) for sample in pass_samples)
    opponent = Counter(sample.opponent_configuration_key or "unknown" for sample in pass_samples)
    return {
        "passSamplesTotal": len(pass_samples),
        "uniqueStateCount": len({sample.state_key for sample in pass_samples}),
        "citizenCount": roles["citizen"],
        "citizenRate": _safe_div(roles["citizen"], len(pass_samples)),
        "adjutantCount": roles["adjutant"],
        "adjutantRate": _safe_div(roles["adjutant"], len(pass_samples)),
        "noContractCount": roles["no-contract"],
        "noContractRate": _safe_div(roles["no-contract"], len(pass_samples)),
        "napoleonAfterPassCount": roles["napoleon"] + roles["napoleon-adjutant"],
        "napoleonAfterPassRate": _safe_div(
            roles["napoleon"] + roles["napoleon-adjutant"], len(pass_samples)
        ),
        "labeledCitizenAdjutantCount": labeled,
        "labeledCitizenAdjutantRate": _safe_div(labeled, len(pass_samples)),
        "handStrengthDistribution": dict(sorted(strength.items())),
        "currentBidLevelDistribution": dict(sorted(current_bid.items())),
        "strongestSuitDistribution": dict(sorted(strongest_suit.items())),
        "candidateSeatDistribution": dict(sorted(seat.items())),
        "opponentCompositionDistribution": dict(sorted(opponent.items())),
        "noContractRequiresSeparateHead": (
            _safe_div(roles["no-contract"], len(pass_samples)) or 0.0
        )
        >= 0.05,
    }


def create_pass_role_binary_split(
    samples: tuple[BiddingQRawSample, ...],
    *,
    train_ratio: float,
    seed: int,
) -> BiddingPassRoleBinarySplit:
    if train_ratio <= 0.0 or train_ratio >= 1.0:
        raise ValueError("train_ratio must be in (0, 1).")
    pass_samples = tuple(sample for sample in samples if sample.forced_action_index == 0)
    state_keys = sorted({sample.state_key for sample in pass_samples})
    if len(state_keys) < 2:
        raise ValueError("PASS split requires at least two states.")
    ordered = sorted(state_keys, key=lambda key: _stable_split_key(seed, key))
    train_count = max(1, min(len(ordered) - 1, int(round(len(ordered) * train_ratio))))
    train_keys = frozenset(ordered[:train_count])
    calibration_keys = frozenset(ordered[train_count:])
    train_samples = tuple(sample for sample in pass_samples if sample.state_key in train_keys)
    calibration_samples = tuple(
        sample for sample in pass_samples if sample.state_key in calibration_keys
    )
    if {sample.state_key for sample in train_samples} & {
        sample.state_key for sample in calibration_samples
    }:
        raise AssertionError("stateKey leakage between train and calibration.")
    return BiddingPassRoleBinarySplit(
        train_state_keys=train_keys,
        calibration_state_keys=calibration_keys,
        train_samples=train_samples,
        calibration_samples=calibration_samples,
        state_key_hash=_state_key_hash(calibration_keys),
    )


def train_bidding_pass_role_binary_model(
    dataset: BiddingQDataset,
    config: BiddingPassRoleBinaryTrainConfig,
) -> BiddingPassRoleBinaryTrainResult:
    _validate_train_config(config)
    _configure_reproducibility(config.seed)
    split = create_pass_role_binary_split(
        dataset.raw_samples, train_ratio=config.train_ratio, seed=config.seed
    )
    device = resolve_torch_device(config.device, flag_name="--device")
    model = create_seeded_bidding_pass_role_binary_model(
        BiddingPassRoleBinaryModelConfig(
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
        ),
        seed=config.seed,
    ).to(device.torch_device)
    pos_weight_value = config.resolved_pos_weight(split.train_samples)
    pos_weight = (
        torch.tensor(pos_weight_value, dtype=torch.float32, device=device.torch_device)
        if pos_weight_value is not None
        else None
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
    )
    calibration_loader = _create_loader(
        split.calibration_samples,
        batch_size=config.batch_size,
        seed=config.seed,
        shuffle=False,
    )
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    best_monitor = math.inf
    misses = 0
    epoch_reports: list[dict[str, object]] = []
    for epoch in range(1, config.epochs + 1):
        train_loss = _run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            pos_weight=pos_weight,
        )
        calibration_loss = _run_epoch(
            model,
            calibration_loader,
            device=device,
            optimizer=None,
            pos_weight=pos_weight,
        )
        improved = calibration_loss < best_monitor - config.min_delta
        if improved:
            best_monitor = calibration_loss
            best_epoch = epoch
            best_state = copy.deepcopy(cpu_state_dict(model))
            misses = 0
        else:
            misses += 1
        epoch_reports.append(
            {
                "epoch": epoch,
                "trainBce": train_loss,
                "calibrationBce": calibration_loss,
                "best": improved,
            }
        )
        if misses >= config.patience:
            break
    if best_state is None:
        raise AssertionError("best checkpoint was not captured.")
    model.load_state_dict(best_state)
    train_report = evaluate_pass_role_binary_model(model, split.train_samples, device=device)
    calibration_report = evaluate_pass_role_binary_model(
        model, split.calibration_samples, device=device
    )
    return BiddingPassRoleBinaryTrainResult(
        model=model,
        split=split,
        config=config,
        device=device,
        best_epoch=best_epoch,
        best_monitor_value=best_monitor,
        epoch_reports=epoch_reports,
        train_report=train_report,
        calibration_report=calibration_report,
        audit=audit_pass_binary_dataset(dataset.raw_samples),
        pos_weight=pos_weight_value,
    )


def predict_pass_adjutant_probability(
    model: BiddingPassRoleBinaryModel,
    samples: tuple[BiddingQRawSample, ...],
    *,
    device: ResolvedTorchDevice,
) -> np.ndarray:
    model.eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(samples), 512):
            batch = samples[start : start + 512]
            model_input = torch.as_tensor(
                np.stack([sample.model_input for sample in batch]),
                dtype=torch.float32,
                device=device.torch_device,
            )
            batches.append(torch.sigmoid(model(model_input)).detach().cpu().numpy())
    return np.concatenate(batches, axis=0) if batches else np.asarray([], dtype=np.float64)


def evaluate_pass_role_binary_model(
    model: BiddingPassRoleBinaryModel,
    samples: tuple[BiddingQRawSample, ...],
    *,
    device: ResolvedTorchDevice,
) -> dict[str, Any]:
    labeled_samples = tuple(sample for sample in samples if pass_binary_label(sample) is not None)
    probabilities = predict_pass_adjutant_probability(model, labeled_samples, device=device)
    labels = np.asarray([pass_binary_label(sample) for sample in labeled_samples], dtype=np.float64)
    return pass_binary_metrics(labeled_samples, labels, probabilities)


def pass_binary_metrics(
    samples: tuple[BiddingQRawSample, ...],
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    if labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must have the same shape.")
    if labels.size == 0:
        return {"sampleCount": 0, "stateCount": 0}
    clipped = np.clip(probabilities.astype(np.float64), 1e-12, 1.0 - 1e-12)
    predicted = (clipped >= 0.5).astype(np.float64)
    tp = float(np.sum((predicted == 1.0) & (labels == 1.0)))
    fp = float(np.sum((predicted == 1.0) & (labels == 0.0)))
    tn = float(np.sum((predicted == 0.0) & (labels == 0.0)))
    fn = float(np.sum((predicted == 0.0) & (labels == 1.0)))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _f1(precision, recall)
    return {
        "sampleCount": int(labels.size),
        "stateCount": len({sample.state_key for sample in samples}),
        "adjutantCount": int(np.sum(labels == 1.0)),
        "citizenCount": int(np.sum(labels == 0.0)),
        "actualAdjutantRate": float(np.mean(labels)),
        "meanPredictedAdjutantProbability": float(np.mean(clipped)),
        "rocAuc": _binary_auc(labels, clipped),
        "prAuc": _pr_auc(labels, clipped),
        "accuracy": float(np.mean(predicted == labels)),
        "balancedAccuracy": _safe_div((recall or 0.0) + (specificity or 0.0), 2.0),
        "adjutantPrecision": precision,
        "adjutantRecall": recall,
        "adjutantF1": f1,
        "brier": float(np.mean((clipped - labels) ** 2)),
        "logLoss": float(
            -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
        ),
        "reliability": _calibration_bins(labels, clipped, bins=10),
        "conditional": _conditional_probability_diagnostics(samples, clipped),
    }


def save_bidding_pass_role_binary_artifact(
    output_dir: Path | str,
    *,
    result: BiddingPassRoleBinaryTrainResult,
    dataset: BiddingQDataset,
) -> dict[str, str]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "checkpoint.pt"
    metadata_path = directory / "metadata.json"
    report_path = directory / "report.json"
    checkpoint = {
        "checkpointSchemaVersion": BIDDING_PASS_ROLE_BINARY_CHECKPOINT_SCHEMA_VERSION,
        "modelType": BIDDING_PASS_ROLE_BINARY_MODEL_TYPE,
        "modelConfig": result.model.config.to_dict(),
        "modelState": cpu_state_dict(result.model),
        "trainingConfig": result.config.to_dict(),
        "posWeight": result.pos_weight,
        "trainingDatasetAudit": result.audit,
    }
    torch.save(checkpoint, checkpoint_path)
    metadata = {
        "modelType": BIDDING_PASS_ROLE_BINARY_MODEL_TYPE,
        "architectureId": BIDDING_PASS_ROLE_BINARY_ARCHITECTURE_ID,
        "checkpointPath": str(checkpoint_path),
        "dataset": {
            "path": str(dataset.manifest.dataset_directory),
            "manifestSha256": dataset.manifest.manifest_sha256,
            "sampleCount": dataset.manifest.sample_count,
            "sourceStates": dataset.manifest.source_states,
        },
        "split": {
            "trainStates": len(result.split.train_state_keys),
            "calibrationStates": len(result.split.calibration_state_keys),
            "trainSamples": len(result.split.train_samples),
            "calibrationSamples": len(result.split.calibration_samples),
            "calibrationStateKeyHash": result.split.state_key_hash,
            "stateKeyLeakage": False,
        },
        "trainingConfig": result.config.to_dict(),
        "device": result.device.to_metadata(),
        "bestEpoch": result.best_epoch,
    }
    report = {
        "audit": result.audit,
        "epochs": result.epoch_reports,
        "bestEpoch": result.best_epoch,
        "bestMonitorValue": result.best_monitor_value,
        "train": result.train_report,
        "calibration": result.calibration_report,
        "posWeight": result.pos_weight,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "checkpointPath": str(checkpoint_path),
        "metadataPath": str(metadata_path),
        "reportPath": str(report_path),
    }


def load_bidding_pass_role_binary_checkpoint(
    path: Path | str,
) -> tuple[BiddingPassRoleBinaryModel, dict[str, Any]]:
    try:
        raw = torch.load(Path(path), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, pickle.UnpicklingError) as error:
        raise BiddingPassRoleBinaryCheckpointError(
            f"checkpoint cannot be loaded: {error}"
        ) from error
    if not isinstance(raw, dict):
        raise BiddingPassRoleBinaryCheckpointError("checkpoint must be a dictionary.")
    if raw.get("checkpointSchemaVersion") != BIDDING_PASS_ROLE_BINARY_CHECKPOINT_SCHEMA_VERSION:
        raise BiddingPassRoleBinaryCheckpointError("checkpoint schema version mismatch.")
    if raw.get("modelType") != BIDDING_PASS_ROLE_BINARY_MODEL_TYPE:
        raise BiddingPassRoleBinaryCheckpointError("checkpoint modelType mismatch.")
    config_raw = raw.get("modelConfig")
    if not isinstance(config_raw, dict):
        raise BiddingPassRoleBinaryCheckpointError("checkpoint modelConfig must be an object.")
    model = BiddingPassRoleBinaryModel(BiddingPassRoleBinaryModelConfig.from_dict(config_raw))
    state = raw.get("modelState")
    if not isinstance(state, dict):
        raise BiddingPassRoleBinaryCheckpointError("checkpoint modelState must be a state dict.")
    model.load_state_dict(state)
    return model, raw


def _run_epoch(
    model: BiddingPassRoleBinaryModel,
    dataloader: DataLoader[BiddingPassRoleBinaryBatch],
    *,
    device: ResolvedTorchDevice,
    optimizer: optim.Optimizer | None,
    pos_weight: Tensor | None,
) -> float:
    model.train(optimizer is not None)
    total = 0.0
    count = 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for batch in dataloader:
            model_input = batch.model_input.to(device=device.torch_device, dtype=torch.float32)
            label = batch.label.to(device=device.torch_device, dtype=torch.float32)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(
                model(model_input), label, pos_weight=pos_weight
            )
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            batch_size = int(label.shape[0])
            total += float(loss.item()) * batch_size
            count += batch_size
    if count == 0:
        raise ValueError("PASS binary split contains no labeled samples.")
    return total / count


def _create_loader(
    samples: Iterable[BiddingQRawSample],
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> DataLoader[BiddingPassRoleBinaryBatch]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        _PassRoleDataset(samples),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def _conditional_probability_diagnostics(
    samples: tuple[BiddingQRawSample, ...],
    probabilities: np.ndarray,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[float]]] = {
        "handStrength": defaultdict(list),
        "currentBidLevel": defaultdict(list),
        "strongestSuit": defaultdict(list),
        "candidateSeat": defaultdict(list),
        "opponentComposition": defaultdict(list),
    }
    for sample, probability in zip(samples, probabilities, strict=True):
        grouped["handStrength"][_strength_bucket(sample.strongest_suit_score)].append(
            float(probability)
        )
        grouped["currentBidLevel"][_current_bid_level(sample)].append(float(probability))
        grouped["strongestSuit"][sample.strongest_suit].append(float(probability))
        grouped["candidateSeat"][str(sample.candidate_seat_index)].append(float(probability))
        grouped["opponentComposition"][sample.opponent_configuration_key or "unknown"].append(
            float(probability)
        )
    return {
        section: {
            key: {"count": len(values), "meanPAdjutant": float(np.mean(values))}
            for key, values in sorted(items.items())
        }
        for section, items in grouped.items()
    }


def _validate_train_config(config: BiddingPassRoleBinaryTrainConfig) -> None:
    if config.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative.")
    if config.patience <= 0:
        raise ValueError("patience must be positive.")
    if config.pos_weight is not None and config.pos_weight <= 0.0:
        raise ValueError("pos_weight must be positive when provided.")


def _configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _stable_split_key(seed: int, state_key: str) -> str:
    return hashlib.sha256(f"{seed}:{state_key}".encode()).hexdigest()


def _state_key_hash(keys: frozenset[str]) -> str:
    return hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()


def _strength_bucket(score: float) -> str:
    if score < 200:
        return "low:<200"
    if score < 280:
        return "medium:200-279"
    if score < 330:
        return "strong:280-329"
    return "very-strong:330+"


def _current_bid_level(sample: BiddingQRawSample) -> str:
    targets = [
        target
        for action_index, is_legal in enumerate(sample.legal_bid_mask)
        if is_legal == 1.0 and (target := action_target(action_index)) is not None
    ]
    if not targets:
        return "no-legal-bid"
    needed = min(targets)
    return "open:13" if needed == 13 else f"raise-to:{needed}"


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0.0:
        return None
    return 2.0 * precision * recall / (precision + recall)
