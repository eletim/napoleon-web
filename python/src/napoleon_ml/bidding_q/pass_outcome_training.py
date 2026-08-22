"""Fixed-hand PASS empirical role/margin training and EV evaluation for Issue #414."""

from __future__ import annotations

import copy
import json
import math
import pickle
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import numpy as np
import torch
from torch import Tensor, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from napoleon_ml.dataset.tensors import BIDDING_MODEL_INPUT_FEATURE_COUNT
from napoleon_ml.policy.device import (
    RequestedTorchDevice,
    ResolvedTorchDevice,
    cpu_state_dict,
    resolve_torch_device,
)

from .fixed_hand_margin_training import (
    FixedHandMarginDataset,
    FixedHandMarginSample,
    FixedHandMarginTrainConfig,
    load_fixed_hand_margin_checkpoint,
    predict_fixed_hand_margin_samples,
    train_fixed_hand_margin_model,
)
from .margin_training import (
    gaussian_success_probability,
    load_bidding_margin_checkpoint,
    predict_margin_samples,
)
from .multi_head_training import Standardization, _binary_auc, _calibration_bins, _pr_auc
from .pass_role_binary_model import (
    BiddingPassRoleBinaryModel,
    BiddingPassRoleBinaryModelConfig,
    create_seeded_bidding_pass_role_binary_model,
)

PASS_OUTCOME_Q_CHECKPOINT_SCHEMA_VERSION = 1
PASS_OUTCOME_Q_MODEL_TYPE = "fixed-hand-pass-empirical-q"


class PassOutcomeDatasetError(ValueError):
    """Raised when a fixed-hand PASS dataset is invalid."""


class PassOutcomeCheckpointError(ValueError):
    """Raised when an empirical-q checkpoint is invalid."""


@dataclass(frozen=True)
class RoleMarginTeacher:
    count: int
    mean: float | None
    std: float | None
    win_rate: float | None
    target_mean: float | None


@dataclass(frozen=True)
class PassOutcomeSample:
    fixed_hand_id: str
    hand_ids: tuple[str, ...]
    model_input: np.ndarray
    rollout_count: int
    q_teacher: float | None
    q_denominator: int
    p_no_contract: float
    n_citizen: int
    n_adjutant: int
    n_no_contract: int
    n_napoleon_after_pass: int
    strongest_suit: str
    strongest_suit_score: float
    citizen: RoleMarginTeacher
    adjutant: RoleMarginTeacher
    split_hint: str | None


@dataclass(frozen=True)
class PassOutcomeDataset:
    directory: Path
    manifest: dict[str, object]
    samples: tuple[PassOutcomeSample, ...]


@dataclass(frozen=True)
class PassOutcomeSplit:
    train_samples: tuple[PassOutcomeSample, ...]
    validation_samples: tuple[PassOutcomeSample, ...]
    final_samples: tuple[PassOutcomeSample, ...]
    train_fixed_hand_ids: tuple[str, ...]
    validation_fixed_hand_ids: tuple[str, ...]
    final_fixed_hand_ids: tuple[str, ...]


@dataclass(frozen=True)
class EmpiricalQTrainConfig:
    seed: int = 414
    epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 1e-3
    hidden_dims: tuple[int, ...] = (256, 256, 128)
    dropout: float = 0.0
    validation_ratio: float = 0.1
    final_ratio: float = 0.1
    train_hand_limit: int | None = None
    min_q_denominator: int = 1
    weight_decay: float = 1e-4
    patience: int = 8
    device: RequestedTorchDevice = "cpu"

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["hidden_dims"] = list(self.hidden_dims)
        data["algorithm"] = "fixed-hand-pass-soft-empirical-q-bce-v1"
        return data


@dataclass(frozen=True)
class EmpiricalQTrainResult:
    model: BiddingPassRoleBinaryModel
    split: PassOutcomeSplit
    config: EmpiricalQTrainConfig
    device: ResolvedTorchDevice
    best_epoch: int
    best_monitor_value: float
    epoch_reports: list[dict[str, object]]
    train_report: dict[str, object]
    validation_report: dict[str, object]
    final_report: dict[str, object]


class QBatch(NamedTuple):
    model_input: Tensor
    target: Tensor
    weight: Tensor


class _QDataset(Dataset[QBatch]):
    def __init__(self, samples: Iterable[PassOutcomeSample], *, min_denominator: int) -> None:
        self.samples = tuple(
            sample
            for sample in samples
            if sample.q_teacher is not None and sample.q_denominator >= min_denominator
        )
        if not self.samples:
            raise ValueError("empirical-q dataset has no labeled samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> QBatch:
        sample = self.samples[index]
        if sample.q_teacher is None:
            raise AssertionError("unlabeled q sample leaked.")
        return QBatch(
            model_input=torch.as_tensor(sample.model_input, dtype=torch.float32),
            target=torch.tensor(sample.q_teacher, dtype=torch.float32),
            weight=torch.tensor(math.sqrt(max(1, sample.q_denominator)), dtype=torch.float32),
        )


def load_pass_outcome_dataset(directory: Path | str) -> PassOutcomeDataset:
    root = Path(directory)
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PassOutcomeDatasetError(f"manifest cannot be loaded: {error}") from error
    if manifest.get("sampleType") != "fixed-hand-pass-outcome-sample":
        raise PassOutcomeDatasetError("manifest sampleType is not fixed-hand-pass-outcome-sample.")
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise PassOutcomeDatasetError("manifest shards must be a list.")
    samples: list[PassOutcomeSample] = []
    for shard in shards:
        if not isinstance(shard, dict) or not isinstance(shard.get("file"), str):
            raise PassOutcomeDatasetError("manifest shard entries must include file.")
        for line in (root / shard["file"]).read_text(encoding="utf-8").splitlines():
            if line.strip():
                samples.append(_parse_sample(json.loads(line)))
    if not samples:
        raise PassOutcomeDatasetError("dataset contains no samples.")
    return PassOutcomeDataset(directory=root, manifest=manifest, samples=tuple(samples))


def create_pass_outcome_split(
    dataset: PassOutcomeDataset,
    *,
    seed: int,
    validation_ratio: float,
    final_ratio: float,
    train_hand_limit: int | None = None,
) -> PassOutcomeSplit:
    by_hand = {sample.fixed_hand_id: sample for sample in dataset.samples}
    final_ids = sorted(
        sample.fixed_hand_id
        for sample in dataset.samples
        if sample.split_hint == "final-diagnostic"
    )
    remaining = sorted(set(by_hand) - set(final_ids))
    rng = random.Random(seed)
    rng.shuffle(remaining)
    desired_final = max(1, round(len(by_hand) * final_ratio)) if len(by_hand) >= 3 else 1
    extra_final = max(0, desired_final - len(final_ids))
    final_ids = [*final_ids, *remaining[:extra_final]]
    remaining = remaining[extra_final:]
    validation_count = max(1, round(len(by_hand) * validation_ratio)) if len(by_hand) >= 3 else 1
    validation_ids = remaining[:validation_count]
    train_ids = remaining[validation_count:]
    if train_hand_limit is not None:
        train_ids = train_ids[:train_hand_limit]
    if not train_ids:
        train_ids, validation_ids = validation_ids, train_ids
    return PassOutcomeSplit(
        train_samples=tuple(by_hand[hand_id] for hand_id in train_ids),
        validation_samples=tuple(by_hand[hand_id] for hand_id in validation_ids),
        final_samples=tuple(by_hand[hand_id] for hand_id in final_ids),
        train_fixed_hand_ids=tuple(train_ids),
        validation_fixed_hand_ids=tuple(validation_ids),
        final_fixed_hand_ids=tuple(final_ids),
    )


def train_empirical_q_model(
    dataset: PassOutcomeDataset,
    config: EmpiricalQTrainConfig,
) -> EmpiricalQTrainResult:
    device = resolve_torch_device(config.device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    split = create_pass_outcome_split(
        dataset,
        seed=config.seed,
        validation_ratio=config.validation_ratio,
        final_ratio=config.final_ratio,
        train_hand_limit=config.train_hand_limit,
    )
    model = create_seeded_bidding_pass_role_binary_model(
        BiddingPassRoleBinaryModelConfig(hidden_dims=config.hidden_dims, dropout=config.dropout),
        seed=config.seed,
    ).to(device.torch_device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loader = DataLoader(
        _QDataset(split.train_samples, min_denominator=config.min_q_denominator),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    best_state = copy.deepcopy(cpu_state_dict(model))
    best_epoch = 0
    best_monitor = math.inf
    misses = 0
    reports: list[dict[str, object]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in loader:
            batch = QBatch(
                model_input=batch.model_input.to(device.torch_device),
                target=batch.target.to(device.torch_device),
                weight=batch.weight.to(device.torch_device),
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.model_input)
            raw = F.binary_cross_entropy_with_logits(logits, batch.target, reduction="none")
            loss = torch.mean(raw * batch.weight / torch.mean(batch.weight))
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        validation = evaluate_empirical_q_model(
            model,
            split.validation_samples,
            device=device,
            min_denominator=config.min_q_denominator,
        )
        monitor_raw = validation["rmse"]
        if not isinstance(monitor_raw, int | float):
            raise ValueError("validation q RMSE is unavailable.")
        monitor = float(monitor_raw)
        reports.append(
            {
                "epoch": epoch,
                "trainLoss": float(np.mean(losses)),
                "validationRmse": monitor,
            }
        )
        if monitor < best_monitor:
            best_monitor = monitor
            best_epoch = epoch
            best_state = copy.deepcopy(cpu_state_dict(model))
            misses = 0
        else:
            misses += 1
            if misses >= config.patience:
                break
    model.load_state_dict(best_state)
    return EmpiricalQTrainResult(
        model=model,
        split=split,
        config=config,
        device=device,
        best_epoch=best_epoch,
        best_monitor_value=best_monitor,
        epoch_reports=reports,
        train_report=evaluate_empirical_q_model(
            model,
            split.train_samples,
            device=device,
            min_denominator=config.min_q_denominator,
        ),
        validation_report=evaluate_empirical_q_model(
            model,
            split.validation_samples,
            device=device,
            min_denominator=config.min_q_denominator,
        ),
        final_report=evaluate_empirical_q_model(
            model,
            split.final_samples,
            device=device,
            min_denominator=config.min_q_denominator,
        ),
    )


def predict_empirical_q(
    model: BiddingPassRoleBinaryModel,
    samples: Sequence[PassOutcomeSample],
    *,
    device: ResolvedTorchDevice,
) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(samples), 512):
            batch = samples[start : start + 512]
            model_input = torch.as_tensor(
                np.stack([sample.model_input for sample in batch]),
                dtype=torch.float32,
                device=device.torch_device,
            )
            values.append(torch.sigmoid(model(model_input)).detach().cpu().numpy())
    return np.concatenate(values, axis=0)


def evaluate_empirical_q_model(
    model: BiddingPassRoleBinaryModel,
    samples: Sequence[PassOutcomeSample],
    *,
    device: ResolvedTorchDevice,
    min_denominator: int = 1,
) -> dict[str, object]:
    labeled = tuple(
        sample
        for sample in samples
        if sample.q_teacher is not None and sample.q_denominator >= min_denominator
    )
    if not labeled:
        return {"sampleCount": 0, "mae": None, "rmse": None, "pearson": None}
    pred = predict_empirical_q(model, labeled, device=device)
    truth = np.asarray([sample.q_teacher for sample in labeled], dtype=np.float64)
    majority = (truth >= 0.5).astype(np.float64)
    return {
        "sampleCount": len(labeled),
        **regression_metrics(pred, truth),
        "brier": float(np.mean((pred - truth) ** 2)),
        "rocAucMajority": _binary_auc(majority, pred),
        "prAucMajority": _pr_auc(majority, pred),
        "calibration": _calibration_bins(truth, pred),
    }


def pass_role_margin_dataset(
    dataset: PassOutcomeDataset,
    *,
    role: Literal["citizen", "adjutant"],
    min_role_count: int = 2,
) -> FixedHandMarginDataset:
    samples: list[FixedHandMarginSample] = []
    for sample in dataset.samples:
        teacher = sample.citizen if role == "citizen" else sample.adjutant
        if teacher.count < min_role_count or teacher.mean is None or teacher.std is None:
            continue
        samples.append(
            FixedHandMarginSample(
                fixed_hand_id=sample.fixed_hand_id,
                hand_ids=sample.hand_ids,
                forced_action_index=0,
                forced_target_point_cards=int(round(teacher.target_mean or 13.0)),
                forced_suit=sample.strongest_suit,
                model_input=sample.model_input,
                rollout_count=teacher.count,
                empirical_margin_mean=teacher.mean,
                empirical_margin_std=max(teacher.std, 1e-6),
                empirical_win_rate=teacher.win_rate or 0.0,
                split_hint=sample.split_hint,
            )
        )
    return FixedHandMarginDataset(
        directory=dataset.directory,
        manifest={**dataset.manifest, "role": role},
        samples=tuple(samples),
    )


def train_pass_role_margin_model(
    dataset: PassOutcomeDataset,
    *,
    role: Literal["citizen", "adjutant"],
    min_role_count: int,
    config: FixedHandMarginTrainConfig,
):
    return train_fixed_hand_margin_model(
        pass_role_margin_dataset(dataset, role=role, min_role_count=min_role_count),
        config,
    )


def evaluate_pass_ev_variants(
    *,
    pass_dataset: PassOutcomeDataset,
    bid_dataset_directory: Path | str,
    napoleon_old_checkpoint: Path | str,
    napoleon_m2_checkpoint: Path | str,
    citizen_new_checkpoint: Path | str,
    adjutant_new_checkpoint: Path | str,
    q_new_checkpoint: Path | str,
    citizen_old_checkpoint: Path | str,
    adjutant_old_checkpoint: Path | str,
    q_old_checkpoint: Path | str,
    device: RequestedTorchDevice = "cpu",
    seed: int = 414,
) -> dict[str, object]:
    # Kept intentionally offline: joins fixedHandId-level PASS samples with fixedHandId BID panels.
    from .fixed_hand_margin_training import load_fixed_hand_margin_dataset

    resolved = resolve_torch_device(device)
    split = create_pass_outcome_split(
        pass_dataset,
        seed=seed,
        validation_ratio=0.1,
        final_ratio=0.1,
    )
    pass_samples = split.final_samples
    bid_dataset = load_fixed_hand_margin_dataset(bid_dataset_directory)
    bid_by_hand: dict[str, list[FixedHandMarginSample]] = defaultdict(list)
    for sample in bid_dataset.samples:
        bid_by_hand[sample.fixed_hand_id].append(sample)

    bid_pred_old = _bid_predictions_from_legacy_margin(
        bid_dataset.samples,
        checkpoint=napoleon_old_checkpoint,
        device=resolved,
    )
    bid_pred_new = _bid_predictions_from_fixed_margin(
        bid_dataset.samples,
        checkpoint=napoleon_m2_checkpoint,
        device=resolved,
    )

    predictions = {
        "old": _legacy_pass_components(
            pass_samples,
            citizen_checkpoint=citizen_old_checkpoint,
            adjutant_checkpoint=adjutant_old_checkpoint,
            q_checkpoint=q_old_checkpoint,
            device=resolved,
        ),
        "new": _fixed_pass_components(
            pass_samples,
            citizen_checkpoint=citizen_new_checkpoint,
            adjutant_checkpoint=adjutant_new_checkpoint,
            q_checkpoint=q_new_checkpoint,
            device=resolved,
        ),
        "newOldQ": _fixed_pass_components(
            pass_samples,
            citizen_checkpoint=citizen_new_checkpoint,
            adjutant_checkpoint=adjutant_new_checkpoint,
            q_checkpoint=q_old_checkpoint,
            device=resolved,
        ),
    }
    variants: dict[str, list[dict[str, float | str | int]]] = {
        key: [] for key in ("E0", "E1", "E2", "E3")
    }
    for pass_index, pass_sample in enumerate(pass_samples):
        bid_actions = bid_by_hand.get(pass_sample.fixed_hand_id, [])
        if not bid_actions:
            continue
        for name, pass_source, bid_source in (
            ("E0", "old", bid_pred_old),
            ("E1", "old", bid_pred_new),
            ("E2", "newOldQ", bid_pred_new),
            ("E3", "new", bid_pred_new),
        ):
            pass_ev = _pass_ev(pass_sample, predictions[pass_source], pass_index)
            candidates: list[dict[str, float | str | int]] = [
                {
                    "actionIndex": 0,
                    "score": pass_ev,
                    "teacherReward": _pass_teacher_reward(pass_sample),
                    "suit": "PASS",
                    "target": 0,
                }
            ]
            for bid in bid_actions:
                mean, sigma = bid_source[(bid.fixed_hand_id, bid.forced_action_index)]
                p = float(gaussian_success_probability(np.asarray([mean]), np.asarray([sigma]))[0])
                candidates.append(
                    {
                        "actionIndex": bid.forced_action_index,
                        "score": _bid_ev(p, bid.forced_target_point_cards),
                        "teacherReward": _bid_teacher_reward(bid),
                        "suit": bid.forced_suit,
                        "target": bid.forced_target_point_cards,
                    }
                )
            selected = max(candidates, key=lambda row: float(row["score"]))
            teacher_best = max(candidates, key=lambda row: float(row["teacherReward"]))
            model_sorted = sorted(candidates, key=lambda row: float(row["score"]), reverse=True)
            pairwise = _candidate_pairwise(candidates)
            selected_teacher = float(selected["teacherReward"])
            oracle = float(teacher_best["teacherReward"])
            selected_action = int(selected["actionIndex"])
            selected_suit = str(selected["suit"])
            selected_target = int(selected["target"])
            variants[name].append(
                {
                    "fixedHandId": pass_sample.fixed_hand_id,
                    "selectedAction": selected_action,
                    "selectedSuit": selected_suit,
                    "selectedTarget": selected_target,
                    "selectedTeacherReward": selected_teacher,
                    "regret": oracle - selected_teacher,
                    "passEv": pass_ev,
                    "bestBidEv": max(
                        float(row["score"])
                        for row in candidates
                        if int(row["actionIndex"]) != 0
                    ),
                    "strongestSuitScore": pass_sample.strongest_suit_score,
                    "strongestSuitMatch": int(
                        selected_action != 0 and selected_suit == pass_sample.strongest_suit
                    ),
                    "bestHit": int(selected["actionIndex"] == teacher_best["actionIndex"]),
                    "top3Hit": int(
                        teacher_best["actionIndex"]
                        in [row["actionIndex"] for row in model_sorted[:3]]
                    ),
                    "pairwiseCorrect": pairwise[0],
                    "pairwiseTotal": pairwise[1],
                }
            )
    return {
        "split": {
            "finalHands": len(split.final_fixed_hand_ids),
            "finalSamples": len(pass_samples),
        },
        "variants": {name: _ev_report(rows) for name, rows in variants.items()},
        "representative": {name: rows[:20] for name, rows in variants.items()},
    }


def save_empirical_q_artifact(
    output_directory: Path | str,
    *,
    result: EmpiricalQTrainResult,
    dataset: PassOutcomeDataset,
) -> dict[str, object]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    report_path = output / "report.json"
    torch.save(
        {
            "checkpointSchemaVersion": PASS_OUTCOME_Q_CHECKPOINT_SCHEMA_VERSION,
            "modelType": PASS_OUTCOME_Q_MODEL_TYPE,
            "modelConfig": result.model.config.to_dict(),
            "modelState": cpu_state_dict(result.model),
            "trainingConfig": result.config.to_dict(),
            "dataset": {"path": str(dataset.directory), "sampleCount": len(dataset.samples)},
        },
        checkpoint_path,
    )
    report = {
        "bestEpoch": result.best_epoch,
        "bestMonitorValue": result.best_monitor_value,
        "split": {
            "trainHands": len(result.split.train_fixed_hand_ids),
            "validationHands": len(result.split.validation_fixed_hand_ids),
            "finalHands": len(result.split.final_fixed_hand_ids),
        },
        "epochs": result.epoch_reports,
        "train": result.train_report,
        "validation": result.validation_report,
        "final": result.final_report,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"checkpointPath": str(checkpoint_path), "reportPath": str(report_path)}


def load_empirical_q_checkpoint(
    path: Path | str,
) -> tuple[BiddingPassRoleBinaryModel, dict[str, object]]:
    try:
        raw = torch.load(Path(path), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, pickle.UnpicklingError) as error:
        raise PassOutcomeCheckpointError(f"checkpoint cannot be loaded: {error}") from error
    if not isinstance(raw, dict) or raw.get("modelType") != PASS_OUTCOME_Q_MODEL_TYPE:
        raise PassOutcomeCheckpointError("checkpoint modelType mismatch.")
    config_raw = raw.get("modelConfig")
    if not isinstance(config_raw, dict):
        raise PassOutcomeCheckpointError("checkpoint modelConfig must be an object.")
    model = BiddingPassRoleBinaryModel(BiddingPassRoleBinaryModelConfig.from_dict(config_raw))
    state = raw.get("modelState")
    if not isinstance(state, dict):
        raise PassOutcomeCheckpointError("checkpoint modelState must be a state dict.")
    model.load_state_dict(state)
    return model, raw


def regression_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, object]:
    residual = prediction - truth
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "pearson": pearson(prediction, truth),
        "bias": float(np.mean(residual)),
    }


def pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else None


def _fixed_pass_components(
    samples: Sequence[PassOutcomeSample],
    *,
    citizen_checkpoint: Path | str,
    adjutant_checkpoint: Path | str,
    q_checkpoint: Path | str,
    device: ResolvedTorchDevice,
) -> dict[str, np.ndarray]:
    citizen_model, citizen_raw = load_fixed_hand_margin_checkpoint(citizen_checkpoint)
    adjutant_model, adjutant_raw = load_fixed_hand_margin_checkpoint(adjutant_checkpoint)
    q_model, _q_raw = _load_q_any(q_checkpoint)
    citizen_samples = _role_eval_samples(samples, "citizen")
    adjutant_samples = _role_eval_samples(samples, "adjutant")
    citizen_pred = predict_fixed_hand_margin_samples(
        citizen_model,
        citizen_samples,
        device=device,
        standardization=_std_from_raw(citizen_raw),
        variant=str(citizen_raw.get("variant", "M2")),  # type: ignore[arg-type]
        constant_sigma=_raw_float(citizen_raw, "constantSigma", 1.0),
    )
    adjutant_pred = predict_fixed_hand_margin_samples(
        adjutant_model,
        adjutant_samples,
        device=device,
        standardization=_std_from_raw(adjutant_raw),
        variant=str(adjutant_raw.get("variant", "M2")),  # type: ignore[arg-type]
        constant_sigma=_raw_float(adjutant_raw, "constantSigma", 1.0),
    )
    q_pred = predict_empirical_q(q_model, samples, device=device)
    return {
        "q": q_pred,
        "citizenPWin": gaussian_success_probability(
            *_selected_arrays(citizen_samples, citizen_pred)
        ),
        "adjutantPWin": gaussian_success_probability(
            *_selected_arrays(adjutant_samples, adjutant_pred)
        ),
    }


def _legacy_pass_components(
    samples: Sequence[PassOutcomeSample],
    *,
    citizen_checkpoint: Path | str,
    adjutant_checkpoint: Path | str,
    q_checkpoint: Path | str,
    device: ResolvedTorchDevice,
) -> dict[str, np.ndarray]:
    from .role_margin_training import load_bidding_role_margin_checkpoint

    citizen_model, citizen_raw = load_bidding_role_margin_checkpoint(citizen_checkpoint)
    adjutant_model, adjutant_raw = load_bidding_role_margin_checkpoint(adjutant_checkpoint)
    q_model, _q_raw = _load_q_any(q_checkpoint)
    legacy_samples = tuple(cast(Any, sample) for sample in samples)
    citizen_pred = predict_margin_samples(
        cast(Any, citizen_model),
        legacy_samples,
        device=device,
        standardization=_std_from_raw(citizen_raw),
    )
    adjutant_pred = predict_margin_samples(
        cast(Any, adjutant_model),
        legacy_samples,
        device=device,
        standardization=_std_from_raw(adjutant_raw),
    )
    q_pred = predict_empirical_q(q_model, samples, device=device)
    return {
        "q": q_pred,
        "citizenPWin": gaussian_success_probability(
            citizen_pred["mean"][:, 0],
            citizen_pred["sigma"][:, 0],
        ),
        "adjutantPWin": gaussian_success_probability(
            adjutant_pred["mean"][:, 0],
            adjutant_pred["sigma"][:, 0],
        ),
    }


def _bid_predictions_from_fixed_margin(
    samples: Sequence[FixedHandMarginSample],
    *,
    checkpoint: Path | str,
    device: ResolvedTorchDevice,
) -> dict[tuple[str, int], tuple[float, float]]:
    model, raw = load_fixed_hand_margin_checkpoint(checkpoint)
    pred = predict_fixed_hand_margin_samples(
        model,
        tuple(samples),
        device=device,
        standardization=_std_from_raw(raw),
        variant=str(raw.get("variant", "M2")),  # type: ignore[arg-type]
        constant_sigma=_raw_float(raw, "constantSigma", 1.0),
    )
    return _prediction_by_key(samples, pred)


def _bid_predictions_from_legacy_margin(
    samples: Sequence[FixedHandMarginSample],
    *,
    checkpoint: Path | str,
    device: ResolvedTorchDevice,
) -> dict[tuple[str, int], tuple[float, float]]:
    model, raw = load_bidding_margin_checkpoint(checkpoint)
    pred = predict_margin_samples(
        cast(Any, model),
        tuple(cast(Any, sample) for sample in samples),
        device=device,
        standardization=_std_from_raw(raw),
    )
    return _prediction_by_key(samples, pred)


def _prediction_by_key(
    samples: Sequence[FixedHandMarginSample],
    pred: dict[str, np.ndarray],
) -> dict[tuple[str, int], tuple[float, float]]:
    result: dict[tuple[str, int], tuple[float, float]] = {}
    for index, sample in enumerate(samples):
        action = sample.forced_action_index
        result[(sample.fixed_hand_id, action)] = (
            float(pred["mean"][index, action]),
            float(pred["sigma"][index, action]),
        )
    return result


def _load_q_any(path: Path | str) -> tuple[BiddingPassRoleBinaryModel, dict[str, object]]:
    try:
        return load_empirical_q_checkpoint(path)
    except PassOutcomeCheckpointError:
        from .pass_role_binary_training import load_bidding_pass_role_binary_checkpoint

        return load_bidding_pass_role_binary_checkpoint(path)


def _role_eval_samples(
    samples: Sequence[PassOutcomeSample],
    role: Literal["citizen", "adjutant"],
) -> tuple[FixedHandMarginSample, ...]:
    result: list[FixedHandMarginSample] = []
    for sample in samples:
        teacher = sample.citizen if role == "citizen" else sample.adjutant
        result.append(
            FixedHandMarginSample(
                fixed_hand_id=sample.fixed_hand_id,
                hand_ids=sample.hand_ids,
                forced_action_index=0,
                forced_target_point_cards=int(round(teacher.target_mean or 13.0)),
                forced_suit=sample.strongest_suit,
                model_input=sample.model_input,
                rollout_count=max(1, teacher.count),
                empirical_margin_mean=teacher.mean or 0.0,
                empirical_margin_std=max(teacher.std or 1.0, 1e-6),
                empirical_win_rate=teacher.win_rate or 0.0,
                split_hint=sample.split_hint,
            )
        )
    return tuple(result)


def _selected_arrays(
    samples: Sequence[FixedHandMarginSample],
    pred: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    mean = []
    sigma = []
    for index, sample in enumerate(samples):
        mean.append(pred["mean"][index, sample.forced_action_index])
        sigma.append(pred["sigma"][index, sample.forced_action_index])
    return np.asarray(mean), np.asarray(sigma)


def _std_from_raw(raw: dict[str, object]) -> Standardization:
    value = raw.get("targetStandardization")
    if not isinstance(value, dict):
        return Standardization(enabled=False, mean=0.0, std=1.0)
    return Standardization(
        enabled=bool(value.get("enabled")),
        mean=float(value.get("mean", 0.0)),
        std=float(value.get("std", 1.0)),
    )


def _raw_float(raw: dict[str, object], key: str, fallback: float) -> float:
    value = raw.get(key, fallback)
    if not isinstance(value, int | float):
        return fallback
    return float(value)


def _pass_ev(sample: PassOutcomeSample, pred: dict[str, np.ndarray], index: int) -> float:
    q = float(pred["q"][index])
    d_adj = sample.adjutant.target_mean or 13.0
    d_cit = sample.citizen.target_mean or 13.0
    ev_a = float(pred["adjutantPWin"][index]) * d_adj
    ev_c = (1.0 - float(pred["citizenPWin"][index])) * d_cit
    labeled_mass = sample.p_no_contract
    return (1.0 - labeled_mass) * (q * ev_a + (1.0 - q) * ev_c)


def _pass_teacher_reward(sample: PassOutcomeSample) -> float:
    q = sample.q_teacher if sample.q_teacher is not None else 0.0
    d_adj = sample.adjutant.target_mean or 13.0
    d_cit = sample.citizen.target_mean or 13.0
    p_adj = sample.adjutant.win_rate or 0.0
    p_cit = sample.citizen.win_rate or 0.0
    return (1.0 - sample.p_no_contract) * (
        q * (p_adj * d_adj) + (1.0 - q) * ((1.0 - p_cit) * d_cit)
    )


def _bid_ev(success_probability: float, target_point_cards: int) -> float:
    return 2.0 * target_point_cards * success_probability - 5.0 * (1.0 - success_probability)


def _bid_teacher_reward(sample: FixedHandMarginSample) -> float:
    return _bid_ev(sample.empirical_win_rate, sample.forced_target_point_cards)


def _candidate_pairwise(candidates: Sequence[dict[str, float | str | int]]) -> tuple[int, int]:
    correct = 0
    total = 0
    for left_index in range(len(candidates)):
        for right_index in range(left_index + 1, len(candidates)):
            left = candidates[left_index]
            right = candidates[right_index]
            teacher_diff = float(left["teacherReward"]) - float(right["teacherReward"])
            if abs(teacher_diff) <= 1e-9:
                continue
            score_diff = float(left["score"]) - float(right["score"])
            total += 1
            if score_diff == 0.0:
                correct += 0
            elif (teacher_diff > 0.0 and score_diff > 0.0) or (
                teacher_diff < 0.0 and score_diff < 0.0
            ):
                correct += 1
    return correct, total


def _ev_report(rows: Sequence[dict[str, float | str | int]]) -> dict[str, object]:
    if not rows:
        return {"sampleCount": 0}
    selected = np.asarray([float(row["selectedTeacherReward"]) for row in rows])
    regret = np.asarray([float(row["regret"]) for row in rows])
    is_bid = np.asarray([int(row["selectedAction"]) != 0 for row in rows])
    strength = np.asarray([float(row["strongestSuitScore"]) for row in rows])
    pairwise_correct = int(sum(int(row["pairwiseCorrect"]) for row in rows))
    pairwise_total = int(sum(int(row["pairwiseTotal"]) for row in rows))
    strong = strength >= np.percentile(strength, 75)
    very_strong = strength >= np.percentile(strength, 90)
    suits = [str(row["selectedSuit"]) for row in rows if int(row["selectedAction"]) != 0]
    targets = [str(int(row["selectedTarget"])) for row in rows if int(row["selectedAction"]) != 0]
    return {
        "sampleCount": len(rows),
        "meanSelectedTeacherReward": float(np.mean(selected)),
        "meanRegret": float(np.mean(regret)),
        "bestActionAccuracy": float(np.mean([int(row["bestHit"]) for row in rows])),
        "top3ActionAccuracy": float(np.mean([int(row["top3Hit"]) for row in rows])),
        "pairwiseAccuracy": (pairwise_correct / pairwise_total) if pairwise_total else None,
        "pairwiseCorrect": pairwise_correct,
        "pairwiseTotal": pairwise_total,
        "passRate": float(np.mean(1 - is_bid)),
        "bidRate": float(np.mean(is_bid)),
        "strongHandBidRate": float(np.mean(is_bid[strong])) if np.any(strong) else None,
        "veryStrongHandBidRate": (
            float(np.mean(is_bid[very_strong])) if np.any(very_strong) else None
        ),
        "suitDistribution": _counts(suits),
        "targetDistribution": _counts(targets),
        "strongestSuitMatchRate": (
            float(
                np.mean(
                    [
                        int(row["strongestSuitMatch"])
                        for row in rows
                        if int(row["selectedAction"]) != 0
                    ]
                )
            )
            if np.any(is_bid)
            else None
        ),
    }


def _counts(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


def _parse_sample(raw: object) -> PassOutcomeSample:
    if not isinstance(raw, dict):
        raise PassOutcomeDatasetError("sample must be an object.")
    model_input = np.asarray(_list(raw, "modelInput"), dtype=np.float32)
    if model_input.shape != (BIDDING_MODEL_INPUT_FEATURE_COUNT,):
        raise PassOutcomeDatasetError("sample modelInput has unexpected feature count.")
    return PassOutcomeSample(
        fixed_hand_id=_string(raw, "fixedHandId"),
        hand_ids=tuple(str(value) for value in _list(raw, "handIds")),
        model_input=model_input,
        rollout_count=_int(raw, "rolloutCount"),
        q_teacher=_nullable_float(raw.get("qTeacher")),
        q_denominator=_int(raw, "qTeacherDenominator"),
        p_no_contract=_float(raw, "pNoContractEmpirical"),
        n_citizen=_int(raw, "nCitizen"),
        n_adjutant=_int(raw, "nAdjutant"),
        n_no_contract=_int(raw, "nNoContract"),
        n_napoleon_after_pass=_int(raw, "nNapoleonAfterPass"),
        strongest_suit=_string(raw, "strongestSuit"),
        strongest_suit_score=_float(raw, "strongestSuitScore"),
        citizen=_parse_role_margin(raw.get("citizenMargin")),
        adjutant=_parse_role_margin(raw.get("adjutantMargin")),
        split_hint=_nullable_string(raw.get("splitHint")),
    )


def _parse_role_margin(raw: object) -> RoleMarginTeacher:
    if not isinstance(raw, dict):
        raise PassOutcomeDatasetError("role margin must be an object.")
    return RoleMarginTeacher(
        count=_int(raw, "count"),
        mean=_nullable_float(raw.get("empiricalMarginMean")),
        std=_nullable_float(raw.get("empiricalMarginStd")),
        win_rate=_nullable_float(raw.get("empiricalWinRate")),
        target_mean=_nullable_float(raw.get("empiricalTargetMean")),
    )


def _string(raw: dict[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str):
        raise PassOutcomeDatasetError(f"{key} must be a string.")
    return value


def _int(raw: dict[str, object], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int):
        raise PassOutcomeDatasetError(f"{key} must be an integer.")
    return value


def _float(raw: dict[str, object], key: str) -> float:
    value = raw.get(key)
    if not isinstance(value, int | float):
        raise PassOutcomeDatasetError(f"{key} must be a number.")
    return float(value)


def _list(raw: dict[str, object], key: str) -> list[object]:
    value = raw.get(key)
    if not isinstance(value, list):
        raise PassOutcomeDatasetError(f"{key} must be a list.")
    return value


def _nullable_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float):
        raise PassOutcomeDatasetError("nullable float field has invalid type.")
    return float(value)


def _nullable_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise PassOutcomeDatasetError("nullable string field has invalid type.")
    return value
