"""Fixed-hand/action empirical-margin training for Issue #411."""

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
from typing import Literal, NamedTuple, cast

import numpy as np
import torch
from torch import Tensor, optim
from torch.utils.data import DataLoader, Dataset

from napoleon_ml.dataset.constants import BIDDING_ACTION_COUNT
from napoleon_ml.dataset.tensors import BIDDING_MODEL_INPUT_FEATURE_COUNT
from napoleon_ml.policy.device import (
    RequestedTorchDevice,
    ResolvedTorchDevice,
    cpu_state_dict,
    resolve_torch_device,
)

from .margin_model import (
    BiddingMarginHeteroscedasticModel,
    BiddingMarginHeteroscedasticModelConfig,
    create_seeded_bidding_margin_model,
)
from .margin_training import gaussian_success_probability
from .multi_head_training import Standardization, _binary_auc, _calibration_bins, _pr_auc

FIXED_HAND_MARGIN_CHECKPOINT_SCHEMA_VERSION = 1
FIXED_HAND_MARGIN_MODEL_TYPE = "fixed-hand-bidding-margin"
FIXED_HAND_MARGIN_SAMPLE_TYPES = frozenset(
    {
        "fixed-hand-bidding-margin-sample",
        "napoleon-fixed-contract-margin-sample",
        "history-consistent-raise-margin-sample",
        "mixed-fixed-hand-margin-sample",
    }
)


class FixedHandMarginDatasetError(ValueError):
    """Raised when the fixed-hand empirical margin dataset is invalid."""


class FixedHandMarginCheckpointError(ValueError):
    """Raised when a fixed-hand checkpoint cannot be loaded."""


@dataclass(frozen=True)
class FixedHandMarginSample:
    fixed_hand_id: str
    hand_ids: tuple[str, ...]
    forced_action_index: int
    forced_target_point_cards: int
    forced_suit: str
    model_input: np.ndarray
    rollout_count: int
    empirical_margin_mean: float
    empirical_margin_std: float
    empirical_win_rate: float
    split_hint: str | None = None
    source_nn_mu: float | None = None
    source_nn_sigma: float | None = None
    source_nn_p_win: float | None = None
    source_state_key: str | None = None
    deal_seed: int | None = None

    @property
    def state_key(self) -> str:
        return self.source_state_key or self.fixed_hand_id


@dataclass(frozen=True)
class FixedHandMarginDataset:
    directory: Path
    manifest: dict[str, object]
    samples: tuple[FixedHandMarginSample, ...]


@dataclass(frozen=True)
class FixedHandMarginSplit:
    train_samples: tuple[FixedHandMarginSample, ...]
    validation_samples: tuple[FixedHandMarginSample, ...]
    final_samples: tuple[FixedHandMarginSample, ...]
    train_fixed_hand_ids: tuple[str, ...]
    validation_fixed_hand_ids: tuple[str, ...]
    final_fixed_hand_ids: tuple[str, ...]


@dataclass(frozen=True)
class FixedHandMarginTrainConfig:
    variant: Literal["M1", "M2"] = "M1"
    seed: int = 411
    epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 1e-3
    hidden_dims: tuple[int, ...] = (256, 256, 128)
    dropout: float = 0.0
    train_pair_limit: int | None = None
    validation_ratio: float = 0.1
    final_ratio: float = 0.1
    target_standardization: bool = True
    std_loss_weight: float = 0.25
    min_sigma: float = 0.05
    weight_decay: float = 1e-4
    patience: int = 8
    min_delta: float = 0.0
    device: RequestedTorchDevice = "cpu"
    init_checkpoint_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["hidden_dims"] = list(self.hidden_dims)
        data["algorithm"] = "fixed-hand-empirical-margin-selected-action-v1"
        return data


@dataclass(frozen=True)
class FixedHandMarginTrainResult:
    model: BiddingMarginHeteroscedasticModel
    split: FixedHandMarginSplit
    config: FixedHandMarginTrainConfig
    device: ResolvedTorchDevice
    target_standardization: Standardization
    constant_sigma: float
    epoch_reports: list[dict[str, object]]
    best_epoch: int
    best_monitor_value: float
    train_report: dict[str, object]
    validation_report: dict[str, object]
    final_report: dict[str, object]


class FixedHandBatch(NamedTuple):
    model_input: Tensor
    action_index: Tensor
    mean: Tensor
    std: Tensor


class _FixedHandDataset(Dataset[FixedHandBatch]):
    def __init__(
        self,
        samples: Iterable[FixedHandMarginSample],
        *,
        standardization: Standardization,
    ) -> None:
        self._samples = tuple(samples)
        self._standardization = standardization

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> FixedHandBatch:
        sample = self._samples[index]
        return FixedHandBatch(
            model_input=torch.as_tensor(sample.model_input, dtype=torch.float32),
            action_index=torch.tensor(sample.forced_action_index, dtype=torch.long),
            mean=self._standardization.encode_tensor(
                torch.tensor(sample.empirical_margin_mean, dtype=torch.float32)
            ),
            std=torch.tensor(max(sample.empirical_margin_std, 1e-6), dtype=torch.float32),
        )


def load_fixed_hand_margin_dataset(
    directory: Path | str,
) -> FixedHandMarginDataset:
    root = Path(directory)
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FixedHandMarginDatasetError(f"manifest cannot be loaded: {error}") from error
    if manifest.get("sampleType") not in FIXED_HAND_MARGIN_SAMPLE_TYPES:
        raise FixedHandMarginDatasetError(
            "manifest sampleType is not a supported fixed-hand margin sample type."
        )
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise FixedHandMarginDatasetError("manifest shards must be a list.")
    samples: list[FixedHandMarginSample] = []
    for shard in shards:
        if not isinstance(shard, dict) or not isinstance(shard.get("file"), str):
            raise FixedHandMarginDatasetError("manifest shard entries must include file.")
        for line in (root / shard["file"]).read_text(encoding="utf-8").splitlines():
            if line.strip():
                samples.append(_parse_fixed_hand_sample(json.loads(line)))
    if not samples:
        raise FixedHandMarginDatasetError("dataset contains no samples.")
    return FixedHandMarginDataset(directory=root, manifest=manifest, samples=tuple(samples))


def create_fixed_hand_margin_split(
    dataset: FixedHandMarginDataset,
    config: FixedHandMarginTrainConfig,
) -> FixedHandMarginSplit:
    by_hand: dict[str, list[FixedHandMarginSample]] = defaultdict(list)
    for sample in dataset.samples:
        by_hand[sample.fixed_hand_id].append(sample)
    final_ids = sorted(
        hand_id
        for hand_id, samples in by_hand.items()
        if any(sample.split_hint == "final-diagnostic" for sample in samples)
    )
    remaining_ids = sorted(set(by_hand) - set(final_ids))
    rng = random.Random(config.seed)
    rng.shuffle(remaining_ids)
    desired_final = max(1, round(len(by_hand) * config.final_ratio)) if len(by_hand) >= 3 else 1
    extra_final = max(0, desired_final - len(final_ids))
    final_ids = [*final_ids, *remaining_ids[:extra_final]]
    remaining_ids = remaining_ids[extra_final:]
    validation_count = (
        max(1, round(len(by_hand) * config.validation_ratio)) if len(by_hand) >= 3 else 1
    )
    validation_ids = remaining_ids[:validation_count]
    train_ids = remaining_ids[validation_count:]
    if not train_ids:
        train_ids, validation_ids = validation_ids, train_ids
    train_samples = tuple(sample for hand_id in train_ids for sample in by_hand[hand_id])
    if config.train_pair_limit is not None:
        train_samples = train_samples[: config.train_pair_limit]
    return FixedHandMarginSplit(
        train_samples=train_samples,
        validation_samples=tuple(
            sample for hand_id in validation_ids for sample in by_hand[hand_id]
        ),
        final_samples=tuple(sample for hand_id in final_ids for sample in by_hand[hand_id]),
        train_fixed_hand_ids=tuple(train_ids),
        validation_fixed_hand_ids=tuple(validation_ids),
        final_fixed_hand_ids=tuple(final_ids),
    )


def train_fixed_hand_margin_model(
    dataset: FixedHandMarginDataset,
    config: FixedHandMarginTrainConfig,
) -> FixedHandMarginTrainResult:
    if config.variant not in {"M1", "M2"}:
        raise ValueError("variant must be M1 or M2.")
    device = resolve_torch_device(config.device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    split = create_fixed_hand_margin_split(dataset, config)
    if not split.train_samples or not split.validation_samples or not split.final_samples:
        raise ValueError("train/validation/final splits must all contain samples.")
    standardization = _standardization(split.train_samples, enabled=config.target_standardization)
    constant_sigma = float(np.mean([sample.empirical_margin_std for sample in split.train_samples]))
    model = create_seeded_bidding_margin_model(
        BiddingMarginHeteroscedasticModelConfig(
            input_dim=BIDDING_MODEL_INPUT_FEATURE_COUNT,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
            log_variance_min=math.log(max(config.min_sigma, 1e-6) ** 2),
            log_variance_max=math.log(20.0**2),
        ),
        seed=config.seed,
    ).to(device.torch_device)
    if config.init_checkpoint_path is not None:
        init_model, init_raw = load_fixed_hand_margin_checkpoint(config.init_checkpoint_path)
        init_config = init_raw.get("modelConfig")
        if init_config != model.config.to_dict():
            raise FixedHandMarginCheckpointError(
                "init checkpoint modelConfig does not match requested architecture."
            )
        model.load_state_dict(cpu_state_dict(init_model))
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = DataLoader(
        _FixedHandDataset(split.train_samples, standardization=standardization),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    best_monitor = math.inf
    best_epoch = 0
    best_state = copy.deepcopy(cpu_state_dict(model))
    epochs_without_improvement = 0
    epoch_reports: list[dict[str, object]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            batch = FixedHandBatch(
                model_input=batch.model_input.to(device.torch_device),
                action_index=batch.action_index.to(device.torch_device),
                mean=batch.mean.to(device.torch_device),
                std=batch.std.to(device.torch_device),
            )
            optimizer.zero_grad(set_to_none=True)
            mean, log_variance = model(batch.model_input)
            loss = fixed_hand_margin_loss(
                mean=mean,
                log_variance=log_variance,
                action_index=batch.action_index,
                target_mean=batch.mean,
                target_std=batch.std,
                variant=config.variant,
                std_loss_weight=config.std_loss_weight,
                target_scale=standardization.std if standardization.enabled else 1.0,
            )
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        validation_report = evaluate_fixed_hand_margin_model(
            model,
            split.validation_samples,
            device=device,
            standardization=standardization,
            variant=config.variant,
            constant_sigma=constant_sigma,
        )
        validation_mean = cast(dict[str, object], validation_report["mean"])
        monitor_raw = validation_mean["rmse"]
        if not isinstance(monitor_raw, int | float):
            raise ValueError("validation mean RMSE is unavailable.")
        monitor = float(monitor_raw)
        report: dict[str, object] = {
            "epoch": epoch,
            "trainLoss": float(np.mean(losses)) if losses else None,
            "validationMeanRmse": monitor,
        }
        epoch_reports.append(report)
        if monitor < best_monitor - config.min_delta:
            best_monitor = monitor
            best_epoch = epoch
            best_state = copy.deepcopy(cpu_state_dict(model))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break
    model.load_state_dict(best_state)
    return FixedHandMarginTrainResult(
        model=model,
        split=split,
        config=config,
        device=device,
        target_standardization=standardization,
        constant_sigma=constant_sigma,
        epoch_reports=epoch_reports,
        best_epoch=best_epoch,
        best_monitor_value=best_monitor,
        train_report=evaluate_fixed_hand_margin_model(
            model,
            split.train_samples,
            device=device,
            standardization=standardization,
            variant=config.variant,
            constant_sigma=constant_sigma,
        ),
        validation_report=evaluate_fixed_hand_margin_model(
            model,
            split.validation_samples,
            device=device,
            standardization=standardization,
            variant=config.variant,
            constant_sigma=constant_sigma,
        ),
        final_report=evaluate_fixed_hand_margin_model(
            model,
            split.final_samples,
            device=device,
            standardization=standardization,
            variant=config.variant,
            constant_sigma=constant_sigma,
        ),
    )


def fixed_hand_margin_loss(
    *,
    mean: Tensor,
    log_variance: Tensor,
    action_index: Tensor,
    target_mean: Tensor,
    target_std: Tensor,
    variant: Literal["M1", "M2"],
    std_loss_weight: float,
    target_scale: float,
) -> Tensor:
    if mean.shape != log_variance.shape or mean.ndim != 2 or mean.shape[1] != BIDDING_ACTION_COUNT:
        raise ValueError("mean/log_variance must have shape (batch, action_count).")
    batch_index = torch.arange(action_index.shape[0], device=action_index.device)
    selected_mean = mean[batch_index, action_index]
    mean_loss = torch.mean((selected_mean - target_mean) ** 2)
    if variant == "M1":
        return mean_loss
    selected_log_variance = log_variance[batch_index, action_index]
    valid_std = target_std > 1e-6
    if not bool(valid_std.any().item()):
        return mean_loss
    target_log_variance = torch.log(
        torch.clamp((target_std / max(target_scale, 1e-12)) ** 2, min=1e-12)
    )
    std_loss = torch.mean((selected_log_variance[valid_std] - target_log_variance[valid_std]) ** 2)
    return mean_loss + std_loss_weight * std_loss


def predict_fixed_hand_margin_samples(
    model: BiddingMarginHeteroscedasticModel,
    samples: Sequence[FixedHandMarginSample],
    *,
    device: ResolvedTorchDevice,
    standardization: Standardization,
    variant: Literal["M1", "M2"],
    constant_sigma: float,
) -> dict[str, np.ndarray]:
    model.eval()
    means: list[np.ndarray] = []
    sigmas: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(samples), 512):
            batch = samples[start : start + 512]
            model_input = torch.as_tensor(
                np.stack([sample.model_input for sample in batch]),
                dtype=torch.float32,
                device=device.torch_device,
            )
            mean, log_variance = model(model_input)
            means.append(mean.detach().cpu().numpy())
            sigma = np.exp(0.5 * log_variance.detach().cpu().numpy())
            if standardization.enabled:
                sigma = sigma * standardization.std
            sigmas.append(sigma)
    mean_array = standardization.decode_array(np.concatenate(means, axis=0))
    sigma_array = np.concatenate(sigmas, axis=0)
    if variant == "M1":
        sigma_array = np.full_like(mean_array, constant_sigma)
    return {"mean": mean_array, "sigma": sigma_array}


def evaluate_fixed_hand_margin_model(
    model: BiddingMarginHeteroscedasticModel,
    samples: Sequence[FixedHandMarginSample],
    *,
    device: ResolvedTorchDevice,
    standardization: Standardization,
    variant: Literal["M1", "M2"],
    constant_sigma: float,
) -> dict[str, object]:
    predictions = predict_fixed_hand_margin_samples(
        model,
        samples,
        device=device,
        standardization=standardization,
        variant=variant,
        constant_sigma=constant_sigma,
    )
    return fixed_hand_margin_evaluation_report(
        samples,
        mean=predictions["mean"],
        sigma=predictions["sigma"],
    )


def fixed_hand_margin_evaluation_report(
    samples: Sequence[FixedHandMarginSample],
    *,
    mean: np.ndarray,
    sigma: np.ndarray,
) -> dict[str, object]:
    selected_mean, selected_sigma = selected_prediction_arrays(samples, mean, sigma)
    truth = np.asarray([sample.empirical_margin_mean for sample in samples], dtype=np.float64)
    truth_std = np.asarray([sample.empirical_margin_std for sample in samples], dtype=np.float64)
    truth_win_rate = np.asarray([sample.empirical_win_rate for sample in samples], dtype=np.float64)
    p_win = gaussian_success_probability(selected_mean, selected_sigma)
    return {
        "sampleCount": len(samples),
        "fixedHandCount": len({sample.fixed_hand_id for sample in samples}),
        "rolloutCount": int(sum(sample.rollout_count for sample in samples)),
        "mean": {
            **regression_metrics(selected_mean, truth),
            "bias": float(np.mean(selected_mean - truth)) if truth.size else None,
        },
        "std": regression_metrics(selected_sigma, truth_std),
        "winProbability": {
            **regression_metrics(p_win, truth_win_rate),
            "brier": float(np.mean((p_win - truth_win_rate) ** 2)) if truth_win_rate.size else None,
            "calibration": _calibration_bins(truth_win_rate, p_win),
            "rocAuc": _binary_auc((truth_win_rate >= 0.5).astype(np.float64), p_win),
            "prAuc": _pr_auc((truth_win_rate >= 0.5).astype(np.float64), p_win),
        },
        "sameHandRanking": same_hand_ranking(samples, score=selected_mean, teacher=truth),
        "sameHandWinRanking": same_hand_ranking(samples, score=p_win, teacher=truth_win_rate),
        "napoleonEv400": ev400_summary(samples, p_win),
    }


def selected_prediction_arrays(
    samples: Sequence[FixedHandMarginSample],
    mean: np.ndarray,
    sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indexes = np.asarray([sample.forced_action_index for sample in samples], dtype=np.int64)
    rows = np.arange(len(samples))
    return mean[rows, indexes], sigma[rows, indexes]


def same_hand_ranking(
    samples: Sequence[FixedHandMarginSample],
    *,
    score: np.ndarray,
    teacher: np.ndarray,
) -> dict[str, object]:
    by_hand: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        by_hand[sample.state_key].append(index)
    pair_count = 0
    correct = 0
    ties = 0
    ranking_states = 0
    top_matches = 0
    for indexes in by_hand.values():
        if len(indexes) < 2:
            continue
        ranking_states += 1
        best_teacher = max(indexes, key=lambda index: teacher[index])
        best_score = max(indexes, key=lambda index: score[index])
        if best_teacher == best_score:
            top_matches += 1
        for left_pos, left in enumerate(indexes):
            for right in indexes[left_pos + 1 :]:
                teacher_diff = teacher[left] - teacher[right]
                if teacher_diff == 0:
                    ties += 1
                    continue
                pair_count += 1
                score_diff = score[left] - score[right]
                if score_diff == 0 or math.copysign(1.0, score_diff) == math.copysign(
                    1.0,
                    teacher_diff,
                ):
                    correct += 1
    return {
        "rankingStateCount": ranking_states,
        "diffPairCount": pair_count,
        "tiePairCount": ties,
        "pairwiseAccuracy": correct / pair_count if pair_count else None,
        "topActionAccuracy": top_matches / ranking_states if ranking_states else None,
    }


def ev400_summary(samples: Sequence[FixedHandMarginSample], p_win: np.ndarray) -> dict[str, object]:
    ev = np.asarray(
        [
            2.0 * sample.forced_target_point_cards * probability - 5.0 * (1.0 - probability)
            for sample, probability in zip(samples, p_win, strict=True)
        ],
        dtype=np.float64,
    )
    thresholds = np.asarray(
        [5.0 / (2.0 * sample.forced_target_point_cards + 5.0) for sample in samples],
        dtype=np.float64,
    )
    return {
        "meanEv": float(np.mean(ev)) if ev.size else None,
        "positiveEvRate": float(np.mean(ev > 0.0)) if ev.size else None,
        "thresholdMean": float(np.mean(thresholds)) if thresholds.size else None,
    }


def save_fixed_hand_margin_artifact(
    output_directory: Path | str,
    *,
    result: FixedHandMarginTrainResult,
    dataset: FixedHandMarginDataset,
) -> dict[str, object]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    report_path = output / "report.json"
    checkpoint = {
        "checkpointSchemaVersion": FIXED_HAND_MARGIN_CHECKPOINT_SCHEMA_VERSION,
        "modelType": FIXED_HAND_MARGIN_MODEL_TYPE,
        "variant": result.config.variant,
        "modelConfig": result.model.config.to_dict(),
        "modelState": cpu_state_dict(result.model),
        "targetStandardization": result.target_standardization.to_dict(),
        "constantSigma": result.constant_sigma,
        "trainingConfig": result.config.to_dict(),
        "dataset": {
            "path": str(dataset.directory),
            "sampleCount": len(dataset.samples),
            "manifest": dataset.manifest,
        },
    }
    torch.save(checkpoint, checkpoint_path)
    report = {
        "bestEpoch": result.best_epoch,
        "bestMonitorValue": result.best_monitor_value,
        "split": {
            "trainHands": len(result.split.train_fixed_hand_ids),
            "validationHands": len(result.split.validation_fixed_hand_ids),
            "finalHands": len(result.split.final_fixed_hand_ids),
            "trainSamples": len(result.split.train_samples),
            "validationSamples": len(result.split.validation_samples),
            "finalSamples": len(result.split.final_samples),
        },
        "epochs": result.epoch_reports,
        "train": result.train_report,
        "validation": result.validation_report,
        "final": result.final_report,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"checkpointPath": str(checkpoint_path), "reportPath": str(report_path)}


def load_fixed_hand_margin_checkpoint(
    path: Path | str,
) -> tuple[BiddingMarginHeteroscedasticModel, dict[str, object]]:
    try:
        raw = torch.load(Path(path), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, pickle.UnpicklingError) as error:
        raise FixedHandMarginCheckpointError(f"checkpoint cannot be loaded: {error}") from error
    if not isinstance(raw, dict):
        raise FixedHandMarginCheckpointError("checkpoint must be a dictionary.")
    if raw.get("checkpointSchemaVersion") != FIXED_HAND_MARGIN_CHECKPOINT_SCHEMA_VERSION:
        raise FixedHandMarginCheckpointError("checkpoint schema version mismatch.")
    if raw.get("modelType") != FIXED_HAND_MARGIN_MODEL_TYPE:
        raise FixedHandMarginCheckpointError("checkpoint modelType mismatch.")
    config_raw = raw.get("modelConfig")
    if not isinstance(config_raw, dict):
        raise FixedHandMarginCheckpointError("checkpoint modelConfig must be an object.")
    model = BiddingMarginHeteroscedasticModel(
        BiddingMarginHeteroscedasticModelConfig.from_dict(config_raw)
    )
    state = raw.get("modelState")
    if not isinstance(state, dict):
        raise FixedHandMarginCheckpointError("checkpoint modelState must be a state dict.")
    model.load_state_dict(state)
    return model, raw


def regression_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, object]:
    if truth.size == 0:
        return {"mae": None, "rmse": None, "pearson": None, "bias": None}
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


def _standardization(
    samples: Sequence[FixedHandMarginSample],
    *,
    enabled: bool,
) -> Standardization:
    values = np.asarray([sample.empirical_margin_mean for sample in samples], dtype=np.float64)
    std = float(np.std(values))
    return Standardization(
        enabled=enabled and std > 1e-12,
        mean=float(np.mean(values)) if values.size else 0.0,
        std=std if std > 1e-12 else 1.0,
    )


def _parse_fixed_hand_sample(raw: object) -> FixedHandMarginSample:
    if not isinstance(raw, dict):
        raise FixedHandMarginDatasetError("sample must be an object.")
    model_input = np.asarray(_list(raw, "modelInput"), dtype=np.float32)
    if model_input.shape != (BIDDING_MODEL_INPUT_FEATURE_COUNT,):
        raise FixedHandMarginDatasetError("sample modelInput has unexpected feature count.")
    return FixedHandMarginSample(
        fixed_hand_id=_string(raw, "fixedHandId"),
        hand_ids=tuple(str(value) for value in _list(raw, "handIds")),
        forced_action_index=_int(raw, "forcedActionIndex"),
        forced_target_point_cards=_int(raw, "forcedTargetPointCards"),
        forced_suit=_string(raw, "forcedSuit"),
        model_input=model_input,
        rollout_count=_int(raw, "rolloutCount"),
        empirical_margin_mean=_float(raw, "empiricalMarginMean"),
        empirical_margin_std=_float(raw, "empiricalMarginStd"),
        empirical_win_rate=_float(raw, "empiricalWinRate"),
        split_hint=_nullable_string(raw.get("splitHint")),
        source_nn_mu=_nullable_float(raw.get("sourceNnMu")),
        source_nn_sigma=_nullable_float(raw.get("sourceNnSigma")),
        source_nn_p_win=_nullable_float(raw.get("sourceNnPWin")),
        source_state_key=_nullable_string(raw.get("sourceStateKey")),
        deal_seed=_nullable_int(raw.get("dealSeed")),
    )


def _string(raw: dict[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str):
        raise FixedHandMarginDatasetError(f"{key} must be a string.")
    return value


def _int(raw: dict[str, object], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int):
        raise FixedHandMarginDatasetError(f"{key} must be an integer.")
    return value


def _float(raw: dict[str, object], key: str) -> float:
    value = raw.get(key)
    if not isinstance(value, int | float):
        raise FixedHandMarginDatasetError(f"{key} must be a number.")
    return float(value)


def _list(raw: dict[str, object], key: str) -> list[object]:
    value = raw.get(key)
    if not isinstance(value, list):
        raise FixedHandMarginDatasetError(f"{key} must be a list.")
    return value


def _nullable_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise FixedHandMarginDatasetError("nullable string field has invalid type.")
    return value


def _nullable_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float):
        raise FixedHandMarginDatasetError("nullable float field has invalid type.")
    return float(value)


def _nullable_int(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise FixedHandMarginDatasetError("nullable integer field has invalid type.")
    return value
