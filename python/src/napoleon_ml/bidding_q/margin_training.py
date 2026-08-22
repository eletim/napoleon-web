"""Training and diagnostics for Issue #380 margin uncertainty models."""

from __future__ import annotations

import copy
import json
import math
import pickle
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple, cast

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

from .dataset import BiddingQDataset, BiddingQRawSample
from .margin_model import (
    BIDDING_MARGIN_HETEROSCEDASTIC_ARCHITECTURE_ID,
    BiddingMarginHeteroscedasticModel,
    BiddingMarginHeteroscedasticModelConfig,
    create_seeded_bidding_margin_model,
)
from .multi_head_training import (
    BiddingMultiHeadQSplit,
    BiddingMultiHeadQTrainConfig,
    Standardization,
    _binary_auc,
    _calibration_bins,
    _contract_margin,
    _contract_mask,
    _grouped_margin,
    _pr_auc,
    _ranking_for_score,
    _regression_metrics,
    _safe_div,
    create_multi_head_split,
)

BIDDING_MARGIN_HETEROSCEDASTIC_CHECKPOINT_SCHEMA_VERSION = 1
BIDDING_MARGIN_HETEROSCEDASTIC_MODEL_TYPE = "bidding-margin-heteroscedastic"
BIDDING_MARGIN_ONNX_METADATA_SCHEMA_VERSION = 1
BIDDING_MARGIN_ONNX_OPSET_VERSION = 18
BIDDING_MARGIN_ONNX_INPUT_NAME = "model_input"
BIDDING_MARGIN_ONNX_MEAN_OUTPUT_NAME = "mean"
BIDDING_MARGIN_ONNX_LOG_VARIANCE_OUTPUT_NAME = "log_variance"


class BiddingMarginCheckpointError(ValueError):
    """Raised when a margin checkpoint is incompatible."""


@dataclass(frozen=True)
class BiddingMarginTrainConfig:
    seed: int = 380
    epochs: int = 80
    batch_size: int = 128
    learning_rate: float = 1e-3
    hidden_dims: tuple[int, ...] = (512, 512, 256, 256)
    dropout: float = 0.0
    train_state_count: int | None = 20000
    validation_state_keys_path: str | None = None
    train_ratio: float = 20000 / 22000
    target_standardization: bool = True
    log_variance_min: float = -6.0
    log_variance_max: float = 5.0
    weight_decay: float = 1e-4
    patience: int = 10
    min_delta: float = 0.0
    device: RequestedTorchDevice = "cpu"
    optimizer: str = "AdamW"

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["hidden_dims"] = list(self.hidden_dims)
        data["algorithm"] = "contract-margin-heteroscedastic-gaussian-nll-v1"
        return data


@dataclass(frozen=True)
class BiddingMarginTrainResult:
    model: BiddingMarginHeteroscedasticModel
    split: BiddingMultiHeadQSplit
    config: BiddingMarginTrainConfig
    device: ResolvedTorchDevice
    target_standardization: Standardization
    epoch_reports: list[dict[str, object]]
    best_epoch: int
    best_monitor_value: float
    train_report: dict[str, object]
    validation_report: dict[str, object]


class BiddingMarginBatch(NamedTuple):
    model_input: Tensor
    action_index: Tensor
    margin: Tensor
    contract_mask: Tensor


class _BiddingMarginDataset(Dataset[BiddingMarginBatch]):
    def __init__(
        self,
        samples: Iterable[BiddingQRawSample],
        *,
        standardization: Standardization,
    ) -> None:
        self.samples = tuple(samples)
        self.standardization = standardization

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> BiddingMarginBatch:
        sample = self.samples[index]
        has_contract = _contract_mask(sample)
        margin = _contract_margin(sample) if has_contract else 0.0
        margin_tensor = self.standardization.encode_tensor(
            torch.tensor(margin, dtype=torch.float32)
        )
        return BiddingMarginBatch(
            model_input=torch.as_tensor(sample.model_input, dtype=torch.float32),
            action_index=torch.tensor(sample.forced_action_index, dtype=torch.long),
            margin=margin_tensor,
            contract_mask=torch.tensor(has_contract, dtype=torch.bool),
        )


def train_bidding_margin_model(
    dataset: BiddingQDataset,
    config: BiddingMarginTrainConfig,
) -> BiddingMarginTrainResult:
    _validate_train_config(config)
    _configure_reproducibility(config.seed)
    split = create_margin_split(dataset, config)
    device = resolve_torch_device(config.device, flag_name="--device")
    standardization = _standardization(
        [_contract_margin(sample) for sample in split.train_samples if _contract_mask(sample)],
        enabled=config.target_standardization,
    )
    model_config = BiddingMarginHeteroscedasticModelConfig(
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        log_variance_min=config.log_variance_min,
        log_variance_max=config.log_variance_max,
    )
    model = create_seeded_bidding_margin_model(model_config, seed=config.seed).to(
        device.torch_device
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_loader = _create_loader(
        split.train_samples,
        standardization=standardization,
        batch_size=config.batch_size,
        seed=config.seed,
        shuffle=True,
    )
    validation_loader = _create_loader(
        split.validation_samples,
        standardization=standardization,
        batch_size=config.batch_size,
        seed=config.seed,
        shuffle=False,
    )
    epoch_reports: list[dict[str, object]] = []
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    best_monitor = math.inf
    epochs_without_improvement = 0
    for epoch in range(1, config.epochs + 1):
        train_nll = _run_nll_epoch(
            model=model,
            dataloader=train_loader,
            device=device,
            optimizer=optimizer,
        )
        validation_nll = _run_nll_epoch(
            model=model,
            dataloader=validation_loader,
            device=device,
            optimizer=None,
        )
        improved = validation_nll < best_monitor - config.min_delta
        if improved:
            best_monitor = validation_nll
            best_epoch = epoch
            best_state = copy.deepcopy(cpu_state_dict(model))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        epoch_reports.append(
            {
                "epoch": epoch,
                "trainNll": train_nll,
                "validationNll": validation_nll,
                "best": improved,
            }
        )
        if epochs_without_improvement >= config.patience:
            break
    if best_state is None:
        raise AssertionError("best checkpoint state was not captured.")
    model.load_state_dict(best_state)
    train_report = evaluate_bidding_margin_model(
        model,
        split.train_samples,
        train_samples=split.train_samples,
        device=device,
        standardization=standardization,
    )
    validation_report = evaluate_bidding_margin_model(
        model,
        split.validation_samples,
        train_samples=split.train_samples,
        device=device,
        standardization=standardization,
    )
    return BiddingMarginTrainResult(
        model=model,
        split=split,
        config=config,
        device=device,
        target_standardization=standardization,
        epoch_reports=epoch_reports,
        best_epoch=best_epoch,
        best_monitor_value=best_monitor,
        train_report=train_report,
        validation_report=validation_report,
    )


def create_margin_split(
    dataset: BiddingQDataset,
    config: BiddingMarginTrainConfig,
) -> BiddingMultiHeadQSplit:
    split_config = BiddingMultiHeadQTrainConfig(
        seed=config.seed,
        train_state_count=config.train_state_count,
        validation_state_keys_path=config.validation_state_keys_path,
        train_ratio=config.train_ratio,
    )
    return create_multi_head_split(dataset, split_config)


def gaussian_margin_nll(
    mean: Tensor,
    log_variance: Tensor,
    action_index: Tensor,
    target: Tensor,
    contract_mask: Tensor,
) -> Tensor:
    if mean.shape != log_variance.shape:
        raise ValueError("mean and log_variance must have the same shape.")
    if mean.ndim != 2 or mean.shape[1] != BIDDING_ACTION_COUNT:
        raise ValueError("mean/log_variance must have shape (batch, 29).")
    if action_index.ndim != 1 or target.ndim != 1 or contract_mask.ndim != 1:
        raise ValueError("action_index, target, and mask must be one-dimensional.")
    if mean.shape[0] != action_index.shape[0]:
        raise ValueError("batch dimension mismatch.")
    batch_index = torch.arange(action_index.shape[0], device=action_index.device)
    selected_mean = mean[batch_index, action_index]
    selected_log_variance = log_variance[batch_index, action_index]
    if not bool(contract_mask.any().item()):
        return selected_mean.sum() * 0.0
    residual = target[contract_mask] - selected_mean[contract_mask]
    selected_log_variance = selected_log_variance[contract_mask]
    inverse_variance = torch.exp(-selected_log_variance)
    return torch.mean(0.5 * (selected_log_variance + residual.square() * inverse_variance))


def evaluate_bidding_margin_model(
    model: BiddingMarginHeteroscedasticModel,
    samples: Iterable[BiddingQRawSample],
    *,
    train_samples: Iterable[BiddingQRawSample],
    device: ResolvedTorchDevice,
    standardization: Standardization,
) -> dict[str, object]:
    sample_tuple = tuple(samples)
    train_tuple = tuple(train_samples)
    predictions = predict_margin_samples(
        model,
        sample_tuple,
        device=device,
        standardization=standardization,
    )
    return margin_evaluation_report(
        sample_tuple,
        train_samples=train_tuple,
        mean=predictions["mean"],
        sigma=predictions["sigma"],
    )


def predict_margin_samples(
    model: BiddingMarginHeteroscedasticModel,
    samples: tuple[BiddingQRawSample, ...],
    *,
    device: ResolvedTorchDevice,
    standardization: Standardization,
) -> dict[str, np.ndarray]:
    model.eval()
    mean_batches: list[np.ndarray] = []
    log_variance_batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(samples), 512):
            batch = samples[start : start + 512]
            model_input = torch.as_tensor(
                np.stack([sample.model_input for sample in batch]),
                dtype=torch.float32,
                device=device.torch_device,
            )
            mean, log_variance = model(model_input)
            mean_batches.append(mean.detach().cpu().numpy())
            log_variance_batches.append(log_variance.detach().cpu().numpy())
    mean_standard = np.concatenate(mean_batches, axis=0)
    log_variance = np.concatenate(log_variance_batches, axis=0)
    sigma_standard = np.exp(0.5 * log_variance)
    return {
        "mean": standardization.decode_array(mean_standard),
        "sigma": sigma_standard * standardization.std,
        "logVariance": log_variance,
    }


def margin_evaluation_report(
    samples: tuple[BiddingQRawSample, ...],
    *,
    train_samples: tuple[BiddingQRawSample, ...],
    mean: np.ndarray,
    sigma: np.ndarray,
) -> dict[str, object]:
    truth, selected_mean, selected_sigma, selected_samples = _selected_margin_arrays(
        samples, mean, sigma
    )
    success_probability = gaussian_success_probability(selected_mean, selected_sigma)
    return {
        "rawSampleCount": len(samples),
        "stateCount": len({sample.state_key for sample in samples}),
        "contractSampleCount": int(truth.size),
        "mean": {
            "overall": {
                **_regression_metrics(selected_mean, truth),
                "signAccuracy": _sign_accuracy(selected_mean, truth),
            },
            "bySuit": _grouped_margin(samples, mean, "suit"),
            "byTarget": _grouped_margin(samples, mean, "target"),
            "pairwiseRanking": _ranking_for_score(
                samples,
                mean,
                teacher=_contract_margin,
                teacher_name="contractMargin",
            ),
        },
        "sigma": sigma_diagnostics(
            selected_samples,
            truth=truth,
            mean=selected_mean,
            sigma=selected_sigma,
        ),
        "baselines": sigma_baselines(train_samples, selected_samples, selected_mean),
        "riskAwareRanking": risk_aware_ranking(samples, mean=mean, sigma=sigma),
        "gaussianSuccess": gaussian_success_metrics(
            selected_samples,
            probability=success_probability,
            score_by_sample=success_probability,
        ),
        "coverage": _coverage(samples),
    }


def sigma_diagnostics(
    samples: tuple[BiddingQRawSample, ...],
    *,
    truth: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
) -> dict[str, object]:
    residual = truth - mean
    absolute_residual = np.abs(residual)
    z = residual / np.maximum(sigma, 1e-12)
    return {
        "sharpness": {
            "overall": _numeric_distribution(sigma),
            "byActionIndex": _group_numeric(
                samples, sigma, lambda sample: str(sample.forced_action_index)
            ),
            "bySuit": _group_numeric(samples, sigma, lambda sample: str(sample.forced_suit)),
            "byTarget": _group_numeric(
                samples, sigma, lambda sample: str(sample.forced_target_point_cards)
            ),
            "byHandStrength": _hand_strength_buckets(samples, sigma),
        },
        "residualCorrelation": {
            "pearson": _pearson(sigma, absolute_residual),
            "spearman": _spearman(sigma, absolute_residual),
        },
        "sigmaBuckets": _sigma_buckets(truth=truth, mean=mean, sigma=sigma, bucket_count=3),
        "standardizedResidual": {
            "mean": float(np.mean(z)),
            "std": float(np.std(z)),
            "median": float(np.median(z)),
            "p5": float(np.percentile(z, 5)),
            "p50": float(np.percentile(z, 50)),
            "p95": float(np.percentile(z, 95)),
        },
        "intervalCoverage": interval_coverage(truth=truth, mean=mean, sigma=sigma),
        "nll": gaussian_nll_numpy(truth=truth, mean=mean, sigma=sigma),
    }


def sigma_baselines(
    train_samples: tuple[BiddingQRawSample, ...],
    validation_samples: tuple[BiddingQRawSample, ...],
    validation_mean: np.ndarray,
) -> dict[str, object]:
    train_contract = tuple(sample for sample in train_samples if _contract_mask(sample))
    truth = np.asarray([_contract_margin(sample) for sample in validation_samples])
    baselines = {
        "globalResidualStd": _baseline_sigma(
            train_contract,
            validation_samples,
            validation_mean,
            key_fn=lambda _sample: "global",
        ),
        "actionIndexResidualStd": _baseline_sigma(
            train_contract,
            validation_samples,
            validation_mean,
            key_fn=lambda sample: str(sample.forced_action_index),
        ),
        "suitTargetResidualStd": _baseline_sigma(
            train_contract,
            validation_samples,
            validation_mean,
            key_fn=lambda sample: f"{sample.forced_suit}:{sample.forced_target_point_cards}",
        ),
    }
    for baseline in baselines.values():
        baseline_sigma = np.asarray(cast(dict[str, object], baseline)["sigma"], dtype=np.float64)
        cast(dict[str, object], baseline).update(
            {
                "nll": gaussian_nll_numpy(truth=truth, mean=validation_mean, sigma=baseline_sigma),
                "intervalCoverage": interval_coverage(
                    truth=truth, mean=validation_mean, sigma=baseline_sigma
                ),
                "sigmaBuckets": _sigma_buckets(
                    truth=truth, mean=validation_mean, sigma=baseline_sigma, bucket_count=3
                ),
            }
        )
        cast(dict[str, object], baseline).pop("sigma", None)
    return baselines


def risk_aware_ranking(
    samples: tuple[BiddingQRawSample, ...],
    *,
    mean: np.ndarray,
    sigma: np.ndarray,
) -> dict[str, object]:
    result = {}
    for penalty in (0.0, 0.25, 0.5, 1.0):
        score = mean - penalty * sigma
        result[str(penalty)] = _ranking_for_score(
            samples,
            score,
            teacher=_contract_margin,
            teacher_name="contractMargin",
        )
    return result


def gaussian_success_probability(mean: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    z = mean / np.maximum(sigma, 1e-12)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def gaussian_success_metrics(
    samples: tuple[BiddingQRawSample, ...],
    *,
    probability: np.ndarray,
    score_by_sample: np.ndarray,
) -> dict[str, object]:
    truth = np.asarray([1.0 if sample.contract_success else 0.0 for sample in samples])
    pairwise = _success_pairwise(samples, score_by_sample)
    return {
        "rocAuc": _binary_auc(truth, probability),
        "prAuc": _pr_auc(truth, probability),
        "calibration": _calibration_bins(truth, probability),
        "brier": float(np.mean((probability - truth) ** 2)) if truth.size else None,
        "pairwiseRankingAccuracy": pairwise,
        "positiveRate": float(truth.mean()) if truth.size else None,
    }


def gaussian_nll_numpy(*, truth: np.ndarray, mean: np.ndarray, sigma: np.ndarray) -> float:
    variance = np.maximum(sigma, 1e-12) ** 2
    return float(np.mean(0.5 * (np.log(variance) + ((truth - mean) ** 2) / variance)))


def interval_coverage(
    *,
    truth: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
) -> dict[str, object]:
    absolute_error = np.abs(truth - mean)
    return {
        "oneSigma": float(np.mean(absolute_error <= sigma)) if truth.size else None,
        "twoSigma": float(np.mean(absolute_error <= 2.0 * sigma)) if truth.size else None,
    }


def save_bidding_margin_artifact(
    output_directory: Path | str,
    *,
    result: BiddingMarginTrainResult,
    dataset: BiddingQDataset,
) -> dict[str, object]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    metadata_path = output / "metadata.json"
    report_path = output / "report.json"
    onnx_path = output / "model.onnx"
    onnx_metadata_path = output / "model.json"
    checkpoint = _checkpoint_dict(result=result, dataset=dataset)
    torch.save(checkpoint, checkpoint_path)
    metadata = _metadata_dict(result=result, dataset=dataset, checkpoint_path=checkpoint_path)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "epochs": result.epoch_reports,
        "bestEpoch": result.best_epoch,
        "bestMonitorValue": result.best_monitor_value,
        "train": result.train_report,
        "validation": result.validation_report,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    parity = export_bidding_margin_onnx(
        model=result.model,
        metadata=metadata,
        onnx_path=onnx_path,
        metadata_path=onnx_metadata_path,
        sample_model_input=dataset.raw_samples[0].model_input,
    )
    return {
        "checkpointPath": str(checkpoint_path),
        "metadataPath": str(metadata_path),
        "reportPath": str(report_path),
        "onnxPath": str(onnx_path),
        "onnxMetadataPath": str(onnx_metadata_path),
        "onnxParity": parity,
    }


def load_bidding_margin_checkpoint(
    path: Path | str,
) -> tuple[BiddingMarginHeteroscedasticModel, dict[str, object]]:
    try:
        raw = torch.load(Path(path), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, pickle.UnpicklingError) as error:
        raise BiddingMarginCheckpointError(f"checkpoint cannot be loaded: {error}") from error
    if not isinstance(raw, dict):
        raise BiddingMarginCheckpointError("checkpoint must be a dictionary.")
    if (
        raw.get("checkpointSchemaVersion")
        != BIDDING_MARGIN_HETEROSCEDASTIC_CHECKPOINT_SCHEMA_VERSION
    ):
        raise BiddingMarginCheckpointError("checkpoint schema version mismatch.")
    if raw.get("modelType") != BIDDING_MARGIN_HETEROSCEDASTIC_MODEL_TYPE:
        raise BiddingMarginCheckpointError("checkpoint modelType mismatch.")
    config_raw = raw.get("modelConfig")
    if not isinstance(config_raw, dict):
        raise BiddingMarginCheckpointError("checkpoint modelConfig must be an object.")
    model = BiddingMarginHeteroscedasticModel(
        BiddingMarginHeteroscedasticModelConfig.from_dict(config_raw)
    )
    state = raw.get("modelState")
    if not isinstance(state, dict):
        raise BiddingMarginCheckpointError("checkpoint modelState must be a state dict.")
    model.load_state_dict(state)
    return model, raw


def export_bidding_margin_onnx(
    *,
    model: BiddingMarginHeteroscedasticModel,
    metadata: dict[str, object],
    onnx_path: Path | str,
    metadata_path: Path | str,
    sample_model_input: np.ndarray,
) -> dict[str, object]:
    output = Path(onnx_path)
    metadata_output = Path(metadata_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model_cpu = BiddingMarginHeteroscedasticModel(model.config)
    model_cpu.load_state_dict(cpu_state_dict(model))
    model_cpu.eval()
    dummy = torch.zeros((2, BIDDING_MODEL_INPUT_FEATURE_COUNT), dtype=torch.float32)
    torch.onnx.export(
        model_cpu,
        (dummy,),
        output,
        input_names=[BIDDING_MARGIN_ONNX_INPUT_NAME],
        output_names=[
            BIDDING_MARGIN_ONNX_MEAN_OUTPUT_NAME,
            BIDDING_MARGIN_ONNX_LOG_VARIANCE_OUTPUT_NAME,
        ],
        dynamic_axes={
            BIDDING_MARGIN_ONNX_INPUT_NAME: {0: "batch"},
            BIDDING_MARGIN_ONNX_MEAN_OUTPUT_NAME: {0: "batch"},
            BIDDING_MARGIN_ONNX_LOG_VARIANCE_OUTPUT_NAME: {0: "batch"},
        },
        opset_version=BIDDING_MARGIN_ONNX_OPSET_VERSION,
    )
    onnx_metadata = dict(metadata)
    onnx_metadata.update(
        {
            "metadataSchemaVersion": BIDDING_MARGIN_ONNX_METADATA_SCHEMA_VERSION,
            "artifactType": "napoleon-bidding-margin-heteroscedastic-onnx",
            "modelType": BIDDING_MARGIN_HETEROSCEDASTIC_MODEL_TYPE,
            "inputName": BIDDING_MARGIN_ONNX_INPUT_NAME,
            "outputNames": [
                BIDDING_MARGIN_ONNX_MEAN_OUTPUT_NAME,
                BIDDING_MARGIN_ONNX_LOG_VARIANCE_OUTPUT_NAME,
            ],
            "outputValueType": "standardized-margin-mean-and-log-variance",
            "onnx": {
                "opsetVersion": BIDDING_MARGIN_ONNX_OPSET_VERSION,
                "inputs": [
                    {
                        "name": BIDDING_MARGIN_ONNX_INPUT_NAME,
                        "dtype": "float32",
                        "shape": ["batch", BIDDING_MODEL_INPUT_FEATURE_COUNT],
                    }
                ],
                "outputs": [
                    {
                        "name": BIDDING_MARGIN_ONNX_MEAN_OUTPUT_NAME,
                        "dtype": "float32",
                        "shape": ["batch", BIDDING_ACTION_COUNT],
                    },
                    {
                        "name": BIDDING_MARGIN_ONNX_LOG_VARIANCE_OUTPUT_NAME,
                        "dtype": "float32",
                        "shape": ["batch", BIDDING_ACTION_COUNT],
                    },
                ],
            },
        }
    )
    metadata_output.write_text(
        json.dumps(onnx_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return _check_onnx_parity(
        model_cpu=model_cpu,
        onnx_path=output,
        sample_model_input=sample_model_input,
    )


def _run_nll_epoch(
    *,
    model: BiddingMarginHeteroscedasticModel,
    dataloader: DataLoader[BiddingMarginBatch],
    device: ResolvedTorchDevice,
    optimizer: optim.Optimizer | None,
) -> float:
    model.train(optimizer is not None)
    total_loss = 0.0
    total_samples = 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for batch in dataloader:
            moved = _move_batch(batch, device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            mean, log_variance = model(moved.model_input)
            loss = gaussian_margin_nll(
                mean,
                log_variance,
                moved.action_index,
                moved.margin,
                moved.contract_mask,
            )
            if optimizer is not None:
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
            contract_count = int(moved.contract_mask.sum().item())
            total_loss += float(loss.item()) * contract_count
            total_samples += contract_count
    if total_samples == 0:
        raise ValueError("split contains no contract samples.")
    return total_loss / total_samples


def _move_batch(batch: BiddingMarginBatch, device: ResolvedTorchDevice) -> BiddingMarginBatch:
    return BiddingMarginBatch(
        model_input=batch.model_input.to(device=device.torch_device, dtype=torch.float32),
        action_index=batch.action_index.to(device=device.torch_device, dtype=torch.long),
        margin=batch.margin.to(device=device.torch_device, dtype=torch.float32),
        contract_mask=batch.contract_mask.to(device=device.torch_device, dtype=torch.bool),
    )


def _create_loader(
    samples: Iterable[BiddingQRawSample],
    *,
    standardization: Standardization,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> DataLoader[BiddingMarginBatch]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        _BiddingMarginDataset(samples, standardization=standardization),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def _selected_margin_arrays(
    samples: tuple[BiddingQRawSample, ...],
    mean: np.ndarray,
    sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[BiddingQRawSample, ...]]:
    truth = []
    selected_mean = []
    selected_sigma = []
    selected_samples = []
    for index, sample in enumerate(samples):
        if not _contract_mask(sample):
            continue
        truth.append(_contract_margin(sample))
        selected_mean.append(float(mean[index, sample.forced_action_index]))
        selected_sigma.append(float(sigma[index, sample.forced_action_index]))
        selected_samples.append(sample)
    return (
        np.asarray(truth, dtype=np.float64),
        np.asarray(selected_mean, dtype=np.float64),
        np.asarray(selected_sigma, dtype=np.float64),
        tuple(selected_samples),
    )


def _sigma_buckets(
    *,
    truth: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
    bucket_count: int,
) -> list[dict[str, object]]:
    order = np.argsort(sigma)
    buckets = []
    names = (
        ("low", "mid", "high") if bucket_count == 3 else tuple(str(i) for i in range(bucket_count))
    )
    for bucket_index, indices in enumerate(np.array_split(order, bucket_count)):
        bucket_truth = truth[indices]
        bucket_mean = mean[indices]
        bucket_sigma = sigma[indices]
        regression = _regression_metrics(bucket_mean, bucket_truth)
        buckets.append(
            {
                "bucket": names[bucket_index],
                "count": int(indices.size),
                "meanSigma": float(np.mean(bucket_sigma)) if indices.size else None,
                "mae": regression["mae"],
                "rmse": regression["rmse"],
            }
        )
    return buckets


def _baseline_sigma(
    train_samples: tuple[BiddingQRawSample, ...],
    validation_samples: tuple[BiddingQRawSample, ...],
    validation_mean: np.ndarray,
    *,
    key_fn: Callable[[BiddingQRawSample], str],
) -> dict[str, object]:
    grouped_train: dict[str, list[float]] = defaultdict(list)
    for sample in train_samples:
        grouped_train[key_fn(sample)].append(_contract_margin(sample))
    global_std = _std_floor([_contract_margin(sample) for sample in train_samples])
    std_by_key = {key: _std_floor(values) for key, values in grouped_train.items()}
    sigma = np.asarray(
        [std_by_key.get(key_fn(sample), global_std) for sample in validation_samples],
        dtype=np.float64,
    )
    return {"meanSigma": float(np.mean(sigma)), "stdSigma": float(np.std(sigma)), "sigma": sigma}


def _std_floor(values: Iterable[float]) -> float:
    std = float(np.std(list(values)))
    return max(std, 1e-3)


def _numeric_distribution(values: np.ndarray) -> dict[str, object]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }


def _group_numeric(
    samples: tuple[BiddingQRawSample, ...],
    values: np.ndarray,
    key_fn: Callable[[BiddingQRawSample], str],
) -> dict[str, object]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for sample, value in zip(samples, values, strict=True):
        buckets[key_fn(sample)].append(float(value))
    return {
        key: _numeric_distribution(np.asarray(bucket_values, dtype=np.float64))
        for key, bucket_values in sorted(buckets.items())
    }


def _hand_strength_buckets(
    samples: tuple[BiddingQRawSample, ...], values: np.ndarray
) -> dict[str, object]:
    scores = np.asarray([sample.strongest_suit_score for sample in samples], dtype=np.float64)
    low, high = np.percentile(scores, [33.333, 66.667])

    def key(sample: BiddingQRawSample) -> str:
        if sample.strongest_suit_score <= low:
            return "low"
        if sample.strongest_suit_score <= high:
            return "mid"
        return "high"

    return _group_numeric(samples, values, key)


def _success_pairwise(samples: tuple[BiddingQRawSample, ...], score: np.ndarray) -> float | None:
    by_state: dict[str, list[tuple[BiddingQRawSample, float]]] = defaultdict(list)
    for sample, sample_score in zip(samples, score, strict=True):
        by_state[sample.state_key].append((sample, float(sample_score)))
    correct = 0.0
    total = 0
    for state_samples in by_state.values():
        for left in range(len(state_samples)):
            for right in range(left + 1, len(state_samples)):
                left_success = state_samples[left][0].contract_success
                right_success = state_samples[right][0].contract_success
                if left_success == right_success:
                    continue
                score_diff = state_samples[left][1] - state_samples[right][1]
                truth_diff = (1.0 if left_success else 0.0) - (1.0 if right_success else 0.0)
                total += 1
                if score_diff == 0.0:
                    correct += 0.5
                elif (score_diff > 0.0) == (truth_diff > 0.0):
                    correct += 1.0
    return _safe_div(correct, total)


def _sign_accuracy(prediction: np.ndarray, truth: np.ndarray) -> float | None:
    if truth.size == 0:
        return None
    return float(((prediction >= 0.0) == (truth >= 0.0)).mean())


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    return _pearson(_average_ranks(a), _average_ranks(b))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


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


def _check_onnx_parity(
    *,
    model_cpu: BiddingMarginHeteroscedasticModel,
    onnx_path: Path,
    sample_model_input: np.ndarray,
) -> dict[str, object]:
    try:
        import onnxruntime as ort
    except ImportError:
        return {"checked": False, "reason": "onnxruntime not installed"}
    inputs = np.stack(
        [
            sample_model_input.astype(np.float32),
            np.zeros_like(sample_model_input, dtype=np.float32),
        ]
    ).astype(np.float32)
    with torch.no_grad():
        torch_mean, torch_log_variance = model_cpu(torch.as_tensor(inputs, dtype=torch.float32))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_mean, onnx_log_variance = session.run(
        None,
        {BIDDING_MARGIN_ONNX_INPUT_NAME: inputs},
    )
    max_abs_diff = max(
        float(np.max(np.abs(torch_mean.numpy() - onnx_mean))),
        float(np.max(np.abs(torch_log_variance.numpy() - onnx_log_variance))),
    )
    return {"checked": True, "maxAbsDiff": max_abs_diff, "withinTolerance": max_abs_diff <= 1e-5}


def _checkpoint_dict(
    *,
    result: BiddingMarginTrainResult,
    dataset: BiddingQDataset,
) -> dict[str, object]:
    return {
        "checkpointSchemaVersion": BIDDING_MARGIN_HETEROSCEDASTIC_CHECKPOINT_SCHEMA_VERSION,
        "modelType": BIDDING_MARGIN_HETEROSCEDASTIC_MODEL_TYPE,
        "architectureId": BIDDING_MARGIN_HETEROSCEDASTIC_ARCHITECTURE_ID,
        "modelConfig": result.model.config.to_dict(),
        "modelState": cpu_state_dict(result.model),
        "trainingConfig": result.config.to_dict(),
        "targetStandardization": result.target_standardization.to_dict(),
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
    result: BiddingMarginTrainResult,
    dataset: BiddingQDataset,
    checkpoint_path: Path,
) -> dict[str, object]:
    checkpoint = _checkpoint_dict(result=result, dataset=dataset)
    checkpoint.pop("modelState", None)
    checkpoint["checkpointPath"] = str(checkpoint_path)
    return checkpoint


def _configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _validate_train_config(config: BiddingMarginTrainConfig) -> None:
    if config.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if config.patience <= 0:
        raise ValueError("patience must be positive.")
    if config.log_variance_min >= config.log_variance_max:
        raise ValueError("log variance clamp range is invalid.")
