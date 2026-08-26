"""Training, checkpointing, and ranking evaluation for exchange value models."""

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
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from napoleon_ml.bidding_q.multi_head_training import Standardization
from napoleon_ml.policy.device import (
    RequestedTorchDevice,
    ResolvedTorchDevice,
    cpu_state_dict,
    resolve_torch_device,
)

from .dataset import (
    EXCHANGE_COMPACT_VALUE_INPUT_FEATURE_COUNT,
    EXCHANGE_COUNTERFACTUAL_COMBINATION_COUNT,
    EXCHANGE_TACTICAL_VALUE_INPUT_FEATURE_COUNT,
    EXCHANGE_VALUE_INPUT_FEATURE_COUNT,
    EXCHANGE_VALUE_INPUT_VARIANTS,
    ExchangeCounterfactualDataset,
    ExchangeCounterfactualSample,
    ExchangeValueInputVariant,
    ExchangeValueSplit,
    create_exchange_value_split,
    dataset_provenance,
)
from .model import (
    ExchangeValueMlpConfig,
    ExchangeValueMlpModel,
    create_seeded_exchange_value_model,
)

EXCHANGE_VALUE_CHECKPOINT_SCHEMA_VERSION = 1
EXCHANGE_VALUE_MODEL_TYPE = "exchange-combination-value"


class ExchangeValueCheckpointError(ValueError):
    """Raised when an exchange value checkpoint is incompatible."""


@dataclass(frozen=True)
class ExchangeValueTrainConfig:
    seed: int = 436
    epochs: int = 60
    batch_size: int = 512
    learning_rate: float = 1e-3
    hidden_dims: tuple[int, ...] = (512, 512, 256, 256)
    dropout: float = 0.0
    input_variant: ExchangeValueInputVariant = "legacy2724"
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    final_ratio: float = 0.1
    train_state_count: int | None = None
    target_standardization: bool = True
    loss: str = "mse"
    huber_delta: float = 1.0
    pairwise_loss_weight: float = 0.0
    pointwise_loss_weight: float = 1.0
    listwise_loss_weight: float = 0.0
    listwise_temperature: float = 1.0
    pairwise_state_batch_size: int = 4
    weight_decay: float = 1e-4
    patience: int = 8
    min_delta: float = 0.0
    device: RequestedTorchDevice = "cpu"
    optimizer: str = "AdamW"
    warm_start_checkpoint: str | None = None

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["hidden_dims"] = list(self.hidden_dims)
        if self.listwise_loss_weight > 0.0:
            data["algorithm"] = "state-wise-listwise-exchange-reranker-v1"
        elif self.pairwise_loss_weight > 0.0 and self.pointwise_loss_weight == 0.0:
            data["algorithm"] = "pairwise-only-exchange-ranker-v1"
        elif self.pairwise_loss_weight > 0.0:
            data["algorithm"] = "pointwise-pairwise-exchange-ranker-v1"
        else:
            data["algorithm"] = "pointwise-contract-margin-regression-v1"
        if self.warm_start_checkpoint is not None:
            checkpoint_path = Path(self.warm_start_checkpoint)
            data["warmStartCheckpointSha256"] = hashlib.sha256(
                checkpoint_path.read_bytes()
            ).hexdigest()
        return data


@dataclass(frozen=True)
class ExchangeValueTrainResult:
    model: ExchangeValueMlpModel
    split: ExchangeValueSplit
    config: ExchangeValueTrainConfig
    device: ResolvedTorchDevice
    target_standardization: Standardization
    epoch_reports: list[dict[str, object]]
    best_epoch: int
    best_monitor_value: float
    train_report: dict[str, object]
    validation_report: dict[str, object]
    final_report: dict[str, object]


class ExchangeValueBatch(NamedTuple):
    value_input: Tensor
    contract_margin: Tensor


class ExchangeValueStateBatch(NamedTuple):
    value_input: Tensor
    contract_margin: Tensor


class _ExchangeValueDataset(Dataset[ExchangeValueBatch]):
    def __init__(
        self,
        samples: Iterable[ExchangeCounterfactualSample],
        *,
        standardization: Standardization,
        input_variant: ExchangeValueInputVariant,
    ) -> None:
        self.samples = tuple(samples)
        self.standardization = standardization
        self.input_variant = input_variant

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ExchangeValueBatch:
        sample = self.samples[index]
        margin = self.standardization.encode_tensor(
            torch.tensor(sample.contract_margin, dtype=torch.float32)
        )
        return ExchangeValueBatch(
            value_input=torch.as_tensor(
                sample.value_input_for_variant(self.input_variant),
                dtype=torch.float32,
            ),
            contract_margin=margin,
        )


class _ExchangeValueStateDataset(Dataset[ExchangeValueStateBatch]):
    def __init__(
        self,
        samples: Iterable[ExchangeCounterfactualSample],
        *,
        standardization: Standardization,
        input_variant: ExchangeValueInputVariant,
    ) -> None:
        by_state: dict[str, list[ExchangeCounterfactualSample]] = defaultdict(list)
        for sample in samples:
            by_state[sample.source_state_key].append(sample)
        self.groups = tuple(
            tuple(sorted(group, key=lambda sample: sample.candidate_index))
            for group in by_state.values()
        )
        self.standardization = standardization
        self.input_variant = input_variant

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> ExchangeValueStateBatch:
        group = self.groups[index]
        value_input = np.stack(
            [sample.value_input_for_variant(self.input_variant) for sample in group]
        )
        margin = np.asarray([sample.contract_margin for sample in group], dtype=np.float32)
        margin_tensor = self.standardization.encode_tensor(torch.as_tensor(margin))
        return ExchangeValueStateBatch(
            value_input=torch.as_tensor(value_input, dtype=torch.float32),
            contract_margin=margin_tensor,
        )


def train_exchange_value_model(
    dataset: ExchangeCounterfactualDataset,
    config: ExchangeValueTrainConfig,
) -> ExchangeValueTrainResult:
    _validate_train_config(config)
    _configure_reproducibility(config.seed)
    split = create_exchange_value_split(
        dataset,
        seed=config.seed,
        train_ratio=config.train_ratio,
        validation_ratio=config.validation_ratio,
        final_ratio=config.final_ratio,
        train_state_count=config.train_state_count,
    )
    device = resolve_torch_device(config.device, flag_name="--device")
    standardization = _standardization(
        [sample.contract_margin for sample in split.train_samples],
        enabled=config.target_standardization,
    )
    model_config = ExchangeValueMlpConfig(
        input_dim=_input_dim_for_variant(config.input_variant),
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    )
    model = create_seeded_exchange_value_model(model_config, seed=config.seed).to(
        device.torch_device
    )
    if config.warm_start_checkpoint is not None:
        _warm_start_exchange_model(model, Path(config.warm_start_checkpoint))
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    # The sample-wise fast path computes the unweighted pointwise objective. Use
    # the state-wise objective whenever its configurable weight must be applied.
    use_statewise = (
        config.pairwise_loss_weight > 0.0
        or config.listwise_loss_weight > 0.0
        or config.pointwise_loss_weight != 1.0
    )
    if use_statewise:
        train_state_loader = _create_state_loader(
            split.train_samples,
            standardization=standardization,
            input_variant=config.input_variant,
            batch_size=config.pairwise_state_batch_size,
            seed=config.seed,
            shuffle=True,
        )
        validation_state_loader = _create_state_loader(
            split.validation_samples,
            standardization=standardization,
            input_variant=config.input_variant,
            batch_size=config.pairwise_state_batch_size,
            seed=config.seed,
            shuffle=False,
        )
    else:
        train_sample_loader = _create_loader(
            split.train_samples,
            standardization=standardization,
            input_variant=config.input_variant,
            batch_size=config.batch_size,
            seed=config.seed,
            shuffle=True,
        )
        validation_sample_loader = _create_loader(
            split.validation_samples,
            standardization=standardization,
            input_variant=config.input_variant,
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
        if use_statewise:
            train_loss = _run_pairwise_epoch(
                model=model,
                dataloader=train_state_loader,
                config=config,
                device=device,
                optimizer=optimizer,
            )
            validation_loss = _run_pairwise_epoch(
                model=model,
                dataloader=validation_state_loader,
                config=config,
                device=device,
                optimizer=None,
            )
        else:
            train_loss = _run_epoch(
                model=model,
                dataloader=train_sample_loader,
                config=config,
                device=device,
                optimizer=optimizer,
            )
            validation_loss = _run_epoch(
                model=model,
                dataloader=validation_sample_loader,
                config=config,
                device=device,
                optimizer=None,
            )
        improved = validation_loss < best_monitor - config.min_delta
        if improved:
            best_monitor = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(cpu_state_dict(model))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        epoch_reports.append(
            {
                "epoch": epoch,
                "trainLoss": train_loss,
                "validationLoss": validation_loss,
                "best": improved,
            }
        )
        if epochs_without_improvement >= config.patience:
            break
    if best_state is None:
        raise AssertionError("best checkpoint state was not captured.")
    model.load_state_dict(best_state)
    train_report = evaluate_exchange_value_model(
        model,
        split.train_samples,
        split="train",
        device=device,
        standardization=standardization,
        input_variant=config.input_variant,
    )
    validation_report = evaluate_exchange_value_model(
        model,
        split.validation_samples,
        split="validation",
        device=device,
        standardization=standardization,
        input_variant=config.input_variant,
    )
    final_report = evaluate_exchange_value_model(
        model,
        split.final_samples,
        split="final",
        device=device,
        standardization=standardization,
        input_variant=config.input_variant,
    )
    return ExchangeValueTrainResult(
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
        final_report=final_report,
    )


def evaluate_exchange_value_model(
    model: ExchangeValueMlpModel,
    samples: Iterable[ExchangeCounterfactualSample],
    *,
    split: str,
    device: ResolvedTorchDevice,
    standardization: Standardization,
    input_variant: ExchangeValueInputVariant = "legacy2724",
) -> dict[str, object]:
    sample_tuple = tuple(samples)
    predictions = predict_exchange_value_samples(
        model,
        sample_tuple,
        device=device,
        standardization=standardization,
        input_variant=input_variant,
    )
    return exchange_value_evaluation_report(sample_tuple, predictions=predictions, split=split)


def predict_exchange_value_samples(
    model: ExchangeValueMlpModel,
    samples: tuple[ExchangeCounterfactualSample, ...],
    *,
    device: ResolvedTorchDevice,
    standardization: Standardization,
    input_variant: ExchangeValueInputVariant = "legacy2724",
) -> np.ndarray:
    model.eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(samples), 2048):
            batch = samples[start : start + 2048]
            value_input = torch.as_tensor(
                np.stack([sample.value_input_for_variant(input_variant) for sample in batch]),
                dtype=torch.float32,
                device=device.torch_device,
            )
            prediction = model(value_input)
            batches.append(prediction.detach().cpu().numpy())
    if not batches:
        return np.empty((0,), dtype=np.float32)
    return standardization.decode_array(np.concatenate(batches, axis=0))


def exchange_value_evaluation_report(
    samples: tuple[ExchangeCounterfactualSample, ...],
    *,
    predictions: np.ndarray,
    split: str,
) -> dict[str, object]:
    truth = np.asarray([sample.contract_margin for sample in samples], dtype=np.float64)
    ranking = _ranking_metrics(samples, predictions)
    return {
        "split": split,
        "sampleCount": len(samples),
        "stateCount": len({sample.source_state_key for sample in samples}),
        "fixedThirteenGroupCount": len({sample.fixed_thirteen_group_id for sample in samples}),
        "scalar": _regression_metrics(predictions.astype(np.float64), truth),
        "ranking": ranking,
        "sameThirteen": _same_thirteen_summary(samples, predictions),
        "ruleBased": _rule_based_summary(ranking),
        "buryContent": _bury_content_summary(samples, predictions),
    }


def save_exchange_value_artifact(
    output_directory: Path | str,
    *,
    result: ExchangeValueTrainResult,
    dataset: ExchangeCounterfactualDataset,
) -> dict[str, object]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    metadata_path = output / "metadata.json"
    report_path = output / "report.json"
    split_path = output / "split.json"
    checkpoint = _checkpoint_dict(result=result, dataset=dataset)
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    metadata = _metadata_dict(
        result=result,
        dataset=dataset,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "epochs": result.epoch_reports,
        "bestEpoch": result.best_epoch,
        "bestMonitorValue": result.best_monitor_value,
        "train": result.train_report,
        "validation": result.validation_report,
        "final": result.final_report,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    split_report = _split_report(result.split)
    split_path.write_text(
        json.dumps(split_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "checkpointPath": str(checkpoint_path),
        "metadataPath": str(metadata_path),
        "reportPath": str(report_path),
        "splitPath": str(split_path),
        "checkpointSha256": checkpoint_sha256,
    }


def load_exchange_value_checkpoint(
    path: Path | str,
) -> tuple[ExchangeValueMlpModel, dict[str, object]]:
    try:
        raw = torch.load(Path(path), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, pickle.UnpicklingError) as error:
        raise ExchangeValueCheckpointError(f"checkpoint cannot be loaded: {error}") from error
    if not isinstance(raw, dict):
        raise ExchangeValueCheckpointError("checkpoint must be a dictionary.")
    if raw.get("checkpointSchemaVersion") != EXCHANGE_VALUE_CHECKPOINT_SCHEMA_VERSION:
        raise ExchangeValueCheckpointError("checkpoint schema version mismatch.")
    if raw.get("modelType") != EXCHANGE_VALUE_MODEL_TYPE:
        raise ExchangeValueCheckpointError("checkpoint modelType mismatch.")
    config_raw = raw.get("modelConfig")
    if not isinstance(config_raw, dict):
        raise ExchangeValueCheckpointError("checkpoint modelConfig must be an object.")
    state = raw.get("modelState")
    if not isinstance(state, dict):
        raise ExchangeValueCheckpointError("checkpoint modelState must be a state dict.")
    model = ExchangeValueMlpModel(ExchangeValueMlpConfig.from_dict(config_raw))
    model.load_state_dict(state)
    return model, raw


def _run_epoch(
    *,
    model: ExchangeValueMlpModel,
    dataloader: DataLoader[ExchangeValueBatch],
    config: ExchangeValueTrainConfig,
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
            prediction = model(moved.value_input)
            loss = _pointwise_loss(prediction, moved.contract_margin, config=config)
            if optimizer is not None:
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
            batch_size = int(moved.contract_margin.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
    if total_samples == 0:
        raise ValueError("split contains no samples.")
    return total_loss / total_samples


def _pointwise_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    config: ExchangeValueTrainConfig,
) -> Tensor:
    if config.loss == "mse":
        return F.mse_loss(prediction, target)
    if config.loss == "huber":
        return F.huber_loss(prediction, target, delta=config.huber_delta)
    raise ValueError(f"unsupported loss: {config.loss}.")


def _run_pairwise_epoch(
    *,
    model: ExchangeValueMlpModel,
    dataloader: DataLoader[ExchangeValueStateBatch],
    config: ExchangeValueTrainConfig,
    device: ResolvedTorchDevice,
    optimizer: optim.Optimizer | None,
) -> float:
    model.train(optimizer is not None)
    total_loss = 0.0
    total_samples = 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for batch in dataloader:
            moved = _move_state_batch(batch, device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            state_count, candidate_count, input_dim = moved.value_input.shape
            flat_input = moved.value_input.reshape(state_count * candidate_count, input_dim)
            prediction = model(flat_input).reshape(state_count, candidate_count)
            pointwise = _pointwise_loss(
                prediction.reshape(-1),
                moved.contract_margin.reshape(-1),
                config=config,
            )
            pairwise = (
                _pairwise_ranking_loss(prediction, moved.contract_margin)
                if config.pairwise_loss_weight > 0.0
                else prediction.sum() * 0.0
            )
            listwise = (
                _listwise_ranking_loss(
                    prediction,
                    moved.contract_margin,
                    temperature=config.listwise_temperature,
                )
                if config.listwise_loss_weight > 0.0
                else prediction.sum() * 0.0
            )
            loss = (
                config.pointwise_loss_weight * pointwise
                + config.pairwise_loss_weight * pairwise
                + config.listwise_loss_weight * listwise
            )
            if optimizer is not None:
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
            sample_count = int(state_count * candidate_count)
            total_loss += float(loss.item()) * sample_count
            total_samples += sample_count
    if total_samples == 0:
        raise ValueError("split contains no samples.")
    return total_loss / total_samples


def _pairwise_ranking_loss(prediction: Tensor, target: Tensor) -> Tensor:
    target_delta = target.unsqueeze(2) - target.unsqueeze(1)
    mask = target_delta != 0.0
    if not bool(mask.any().item()):
        return prediction.sum() * 0.0
    sign = torch.sign(target_delta[mask])
    prediction_delta = (prediction.unsqueeze(2) - prediction.unsqueeze(1))[mask]
    return F.softplus(-sign * prediction_delta).mean()


def _listwise_ranking_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    temperature: float,
) -> Tensor:
    target_distribution = F.softmax(target / temperature, dim=1)
    log_prediction_distribution = F.log_softmax(prediction, dim=1)
    return -(target_distribution * log_prediction_distribution).sum(dim=1).mean()


def _warm_start_exchange_model(
    model: ExchangeValueMlpModel,
    checkpoint_path: Path,
) -> None:
    source_model, _checkpoint = load_exchange_value_checkpoint(checkpoint_path)
    source_state = source_model.state_dict()
    target_state = model.state_dict()
    if source_model.config.hidden_dims != model.config.hidden_dims:
        raise ValueError("warm-start hidden_dims must match target model.")
    source_input_dim = source_model.config.input_dim
    target_input_dim = model.config.input_dim
    if source_input_dim != target_input_dim and (
        source_input_dim,
        target_input_dim,
    ) != (
        EXCHANGE_COMPACT_VALUE_INPUT_FEATURE_COUNT,
        EXCHANGE_TACTICAL_VALUE_INPUT_FEATURE_COUNT,
    ):
        raise ValueError(
            "warm-start input layout must match, except compact396 may initialize compact406."
        )
    for key, target in target_state.items():
        source = source_state.get(key)
        if source is None:
            raise ValueError(f"warm-start checkpoint is missing {key}.")
        if source.shape == target.shape:
            target.copy_(source)
            continue
        if (
            key == "network.0.weight"
            and source.ndim == 2
            and target.ndim == 2
            and source.shape[0] == target.shape[0]
            and source.shape[1] < target.shape[1]
        ):
            target.zero_()
            target[:, : source.shape[1]].copy_(source)
            continue
        raise ValueError(
            f"warm-start tensor shape mismatch for {key}: "
            f"{tuple(source.shape)} != {tuple(target.shape)}."
        )
    model.load_state_dict(target_state)


def _move_batch(batch: ExchangeValueBatch, device: ResolvedTorchDevice) -> ExchangeValueBatch:
    return ExchangeValueBatch(
        value_input=batch.value_input.to(device=device.torch_device, dtype=torch.float32),
        contract_margin=batch.contract_margin.to(device=device.torch_device, dtype=torch.float32),
    )


def _move_state_batch(
    batch: ExchangeValueStateBatch,
    device: ResolvedTorchDevice,
) -> ExchangeValueStateBatch:
    return ExchangeValueStateBatch(
        value_input=batch.value_input.to(device=device.torch_device, dtype=torch.float32),
        contract_margin=batch.contract_margin.to(device=device.torch_device, dtype=torch.float32),
    )


def _create_loader(
    samples: Iterable[ExchangeCounterfactualSample],
    *,
    standardization: Standardization,
    input_variant: ExchangeValueInputVariant,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> DataLoader[ExchangeValueBatch]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        _ExchangeValueDataset(
            samples,
            standardization=standardization,
            input_variant=input_variant,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def _create_state_loader(
    samples: Iterable[ExchangeCounterfactualSample],
    *,
    standardization: Standardization,
    input_variant: ExchangeValueInputVariant,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> DataLoader[ExchangeValueStateBatch]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        _ExchangeValueStateDataset(
            samples,
            standardization=standardization,
            input_variant=input_variant,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def _ranking_metrics(
    samples: tuple[ExchangeCounterfactualSample, ...],
    predictions: np.ndarray,
) -> dict[str, object]:
    by_state: dict[str, list[tuple[int, ExchangeCounterfactualSample, float]]] = defaultdict(list)
    for index, sample in enumerate(samples):
        by_state[sample.source_state_key].append((index, sample, float(predictions[index])))
    rows: list[dict[str, object]] = []
    pairwise_correct = 0.0
    pairwise_total = 0
    for state_key, group in by_state.items():
        if len(group) != EXCHANGE_COUNTERFACTUAL_COMBINATION_COUNT:
            raise ValueError(f"{state_key}: expected 286 candidates in evaluation group.")
        ranked_teacher = sorted(
            (item[1] for item in group),
            key=lambda sample: (
                -sample.contract_margin,
                -sample.napoleon_relative_reward,
                sample.candidate_index,
            ),
        )
        teacher_rank_by_index = {
            sample.candidate_index: rank for rank, sample in enumerate(ranked_teacher, start=1)
        }
        best = ranked_teacher[0]
        model_selected = max(group, key=lambda item: (item[2], -item[1].candidate_index))[1]
        rule_based = next(sample for _i, sample, _p in group if sample.is_rule_based_action)
        for left_index in range(len(group) - 1):
            left = group[left_index]
            for right in group[left_index + 1 :]:
                truth_delta = left[1].contract_margin - right[1].contract_margin
                if truth_delta == 0.0:
                    continue
                pred_delta = left[2] - right[2]
                pairwise_total += 1
                if pred_delta == 0.0:
                    pairwise_correct += 0.5
                elif (pred_delta > 0.0) == (truth_delta > 0.0):
                    pairwise_correct += 1
        rows.append(
            {
                "sourceStateKey": state_key,
                "bestCandidateIndex": best.candidate_index,
                "modelCandidateIndex": model_selected.candidate_index,
                "ruleBasedCandidateIndex": rule_based.candidate_index,
                "modelTeacherRank": teacher_rank_by_index[model_selected.candidate_index],
                "ruleBasedTeacherRank": teacher_rank_by_index[rule_based.candidate_index],
                "modelMarginRegret": best.contract_margin - model_selected.contract_margin,
                "ruleBasedMarginRegret": best.contract_margin - rule_based.contract_margin,
                "modelRelativeRewardRegret": (
                    best.napoleon_relative_reward - model_selected.napoleon_relative_reward
                ),
                "ruleBasedRelativeRewardRegret": (
                    best.napoleon_relative_reward - rule_based.napoleon_relative_reward
                ),
            }
        )
    model_ranks = np.asarray([row["modelTeacherRank"] for row in rows], dtype=np.float64)
    rb_ranks = np.asarray([row["ruleBasedTeacherRank"] for row in rows], dtype=np.float64)
    model_margin = np.asarray([row["modelMarginRegret"] for row in rows], dtype=np.float64)
    rb_margin = np.asarray([row["ruleBasedMarginRegret"] for row in rows], dtype=np.float64)
    model_reward = np.asarray([row["modelRelativeRewardRegret"] for row in rows], dtype=np.float64)
    rb_reward = np.asarray([row["ruleBasedRelativeRewardRegret"] for row in rows], dtype=np.float64)
    n = float(EXCHANGE_COUNTERFACTUAL_COMBINATION_COUNT)
    return {
        "stateCount": len(rows),
        "pairwiseAccuracy": _safe_div(pairwise_correct, pairwise_total),
        "pairwiseCount": pairwise_total,
        "exactTopAction": _safe_div(float(np.sum(model_ranks == 1)), len(rows)),
        "top3Hit": _safe_div(float(np.sum(model_ranks <= 3)), len(rows)),
        "top5Hit": _safe_div(float(np.sum(model_ranks <= 5)), len(rows)),
        "modelSelectedTeacherRankPercentile": _numeric_summary((n - model_ranks) / (n - 1.0)),
        "modelSelectedTeacherMarginRegret": _numeric_summary(model_margin),
        "modelSelectedRelativeRewardRegret": _numeric_summary(model_reward),
        "ruleBasedTeacherRankPercentile": _numeric_summary((n - rb_ranks) / (n - 1.0)),
        "ruleBasedTeacherMarginRegret": _numeric_summary(rb_margin),
        "ruleBasedRelativeRewardRegret": _numeric_summary(rb_reward),
        "modelVsRuleBasedMarginRegret": _win_tie_loss(model_margin, rb_margin),
    }


def _rule_based_summary(ranking: dict[str, object]) -> dict[str, object]:
    return {
        "teacherRankPercentile": ranking["ruleBasedTeacherRankPercentile"],
        "teacherMarginRegret": ranking["ruleBasedTeacherMarginRegret"],
        "relativeRewardRegret": ranking["ruleBasedRelativeRewardRegret"],
        "modelVsRuleBasedMarginRegret": ranking["modelVsRuleBasedMarginRegret"],
    }


def _same_thirteen_summary(
    samples: tuple[ExchangeCounterfactualSample, ...],
    predictions: np.ndarray,
) -> dict[str, object]:
    by_state: dict[str, list[tuple[ExchangeCounterfactualSample, float]]] = defaultdict(list)
    for index, sample in enumerate(samples):
        by_state[sample.source_state_key].append((sample, float(predictions[index])))

    by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    for state_key, group in by_state.items():
        if len(group) != EXCHANGE_COUNTERFACTUAL_COMBINATION_COUNT:
            raise ValueError(f"{state_key}: expected 286 candidates in same-13 summary.")
        best = max(
            (sample for sample, _prediction in group),
            key=lambda sample: (
                sample.contract_margin,
                sample.napoleon_relative_reward,
                -sample.candidate_index,
            ),
        )
        model_selected = max(group, key=lambda item: (item[1], -item[0].candidate_index))[0]
        prediction_by_candidate = {
            sample.candidate_index: prediction for sample, prediction in group
        }
        group_id = best.fixed_thirteen_group_id
        by_group[group_id].append(
            {
                "sourceStateKey": state_key,
                "modelMarginRegret": best.contract_margin - model_selected.contract_margin,
                "modelRelativeRewardRegret": best.napoleon_relative_reward
                - model_selected.napoleon_relative_reward,
                "modelDiscardKey": "|".join(model_selected.candidate_discard_card_ids),
                "teacherBestDiscardKey": "|".join(best.candidate_discard_card_ids),
                "modelSelectedPrediction": prediction_by_candidate[model_selected.candidate_index],
                "teacherBestPrediction": prediction_by_candidate[best.candidate_index],
                "stateMeanPrediction": float(
                    np.mean(
                        np.asarray(
                            list(prediction_by_candidate.values()),
                            dtype=np.float64,
                        )
                    )
                ),
            }
        )

    group_rows: list[dict[str, object]] = []
    for group_id, rows in by_group.items():
        margin = np.asarray([row["modelMarginRegret"] for row in rows], dtype=np.float64)
        reward = np.asarray(
            [row["modelRelativeRewardRegret"] for row in rows],
            dtype=np.float64,
        )
        group_rows.append(
            {
                "fixedThirteenGroupId": group_id,
                "dealCount": len(rows),
                "meanMarginRegret": float(np.mean(margin)),
                "meanRelativeRewardRegret": float(np.mean(reward)),
                "modelSelectedDiscardUniqueCount": len(
                    {str(row["modelDiscardKey"]) for row in rows}
                ),
                "teacherBestDiscardUniqueCount": len(
                    {str(row["teacherBestDiscardKey"]) for row in rows}
                ),
                "modelSelectedPredictedValueStdDev": float(
                    np.std(
                        np.asarray(
                            [row["modelSelectedPrediction"] for row in rows],
                            dtype=np.float64,
                        )
                    )
                ),
                "teacherBestPredictedValueStdDev": float(
                    np.std(
                        np.asarray(
                            [row["teacherBestPrediction"] for row in rows],
                            dtype=np.float64,
                        )
                    )
                ),
                "stateMeanPredictedValueStdDev": float(
                    np.std(
                        np.asarray(
                            [row["stateMeanPrediction"] for row in rows],
                            dtype=np.float64,
                        )
                    )
                ),
            }
        )

    return {
        "groupCount": len(group_rows),
        "dealCountPerGroup": _numeric_summary(
            np.asarray([row["dealCount"] for row in group_rows], dtype=np.float64)
        ),
        "groupMeanMarginRegret": _numeric_summary(
            np.asarray([row["meanMarginRegret"] for row in group_rows], dtype=np.float64)
        ),
        "groupMeanRelativeRewardRegret": _numeric_summary(
            np.asarray(
                [row["meanRelativeRewardRegret"] for row in group_rows],
                dtype=np.float64,
            )
        ),
        "modelSelectedDiscardUniqueCount": _numeric_summary(
            np.asarray(
                [row["modelSelectedDiscardUniqueCount"] for row in group_rows],
                dtype=np.float64,
            )
        ),
        "teacherBestDiscardUniqueCount": _numeric_summary(
            np.asarray(
                [row["teacherBestDiscardUniqueCount"] for row in group_rows],
                dtype=np.float64,
            )
        ),
        "modelSelectedPredictedValueStdDev": _numeric_summary(
            np.asarray(
                [row["modelSelectedPredictedValueStdDev"] for row in group_rows],
                dtype=np.float64,
            )
        ),
        "teacherBestPredictedValueStdDev": _numeric_summary(
            np.asarray(
                [row["teacherBestPredictedValueStdDev"] for row in group_rows],
                dtype=np.float64,
            )
        ),
        "stateMeanPredictedValueStdDev": _numeric_summary(
            np.asarray(
                [row["stateMeanPredictedValueStdDev"] for row in group_rows],
                dtype=np.float64,
            )
        ),
    }


def _bury_content_summary(
    samples: tuple[ExchangeCounterfactualSample, ...],
    predictions: np.ndarray,
) -> dict[str, object]:
    by_state: dict[str, list[tuple[ExchangeCounterfactualSample, float]]] = defaultdict(list)
    for index, sample in enumerate(samples):
        by_state[sample.source_state_key].append((sample, float(predictions[index])))
    selected: dict[str, list[ExchangeCounterfactualSample]] = {
        "modelSelected": [],
        "ruleBased": [],
        "teacherBest": [],
    }
    for group in by_state.values():
        selected["modelSelected"].append(max(group, key=lambda item: item[1])[0])
        selected["ruleBased"].append(
            next(sample for sample, _p in group if sample.is_rule_based_action)
        )
        selected["teacherBest"].append(
            max(
                group,
                key=lambda item: (item[0].contract_margin, item[0].napoleon_relative_reward),
            )[0]
        )
    return {name: _bury_distribution(tuple(rows)) for name, rows in selected.items()}


def _bury_distribution(samples: tuple[ExchangeCounterfactualSample, ...]) -> dict[str, object]:
    point_counts = Counter(sample.buried_point_card_count for sample in samples)
    special_keys = ("joker", "oruma", "yoromeki", "seiJack", "uraJack", "calledAdjutant")
    return {
        "stateCount": len(samples),
        "buriedPointCardCount": {str(key): point_counts.get(key, 0) for key in range(4)},
        "buriedTrumpCount": _numeric_summary(
            np.asarray([sample.buried_trump_count for sample in samples], dtype=np.float64)
        ),
        "specialCardBuryFrequency": {
            key: _safe_div(
                sum(1 for sample in samples if sample.buried_special_cards[key]),
                len(samples),
            )
            for key in special_keys
        }
        | {
            "any": _safe_div(
                sum(1 for sample in samples if any(sample.buried_special_cards.values())),
                len(samples),
            )
        },
    }


def _regression_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, object]:
    if truth.size == 0:
        return {"sampleCount": 0, "mae": None, "rmse": None, "pearson": None, "bias": None}
    error = prediction - truth
    return {
        "sampleCount": int(truth.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "pearson": _pearson(prediction, truth),
        "bias": float(np.mean(error)),
    }


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _numeric_summary(values: np.ndarray) -> dict[str, object]:
    if values.size == 0:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _win_tie_loss(model_regret: np.ndarray, rule_based_regret: np.ndarray) -> dict[str, object]:
    diff = rule_based_regret - model_regret
    return {
        "win": int(np.sum(diff > 0.0)),
        "tie": int(np.sum(diff == 0.0)),
        "loss": int(np.sum(diff < 0.0)),
        "winRate": _safe_div(float(np.sum(diff > 0.0)), int(diff.size)),
        "meanRegretImprovement": float(np.mean(diff)) if diff.size else None,
    }


def _safe_div(numerator: float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _standardization(values: Iterable[float], *, enabled: bool) -> Standardization:
    array = np.asarray(list(values), dtype=np.float64)
    if not enabled:
        return Standardization(enabled=False, mean=0.0, std=1.0)
    if array.size == 0:
        raise ValueError("cannot standardize empty target set.")
    std = float(np.std(array))
    if std <= 0.0:
        std = 1.0
    return Standardization(enabled=True, mean=float(np.mean(array)), std=std)


def _checkpoint_dict(
    *,
    result: ExchangeValueTrainResult,
    dataset: ExchangeCounterfactualDataset,
) -> dict[str, object]:
    return {
        "checkpointSchemaVersion": EXCHANGE_VALUE_CHECKPOINT_SCHEMA_VERSION,
        "modelType": EXCHANGE_VALUE_MODEL_TYPE,
        "modelConfig": result.model.config.to_dict(),
        "modelState": cpu_state_dict(result.model),
        "trainingConfig": result.config.to_dict(),
        "targetStandardization": result.target_standardization.to_dict(),
        "dataset": dataset_provenance(dataset),
        "split": _split_report(result.split),
        "bestEpoch": result.best_epoch,
        "bestMonitorValue": result.best_monitor_value,
    }


def _metadata_dict(
    *,
    result: ExchangeValueTrainResult,
    dataset: ExchangeCounterfactualDataset,
    checkpoint_path: Path,
    checkpoint_sha256: str,
) -> dict[str, object]:
    return {
        "artifactType": "napoleon-exchange-combination-value",
        "modelType": EXCHANGE_VALUE_MODEL_TYPE,
        "checkpointSchemaVersion": EXCHANGE_VALUE_CHECKPOINT_SCHEMA_VERSION,
        "checkpointPath": str(checkpoint_path),
        "checkpointSha256": checkpoint_sha256,
        "modelConfig": result.model.config.to_dict(),
        "trainingConfig": result.config.to_dict(),
        "targetStandardization": result.target_standardization.to_dict(),
        "dataset": dataset_provenance(dataset),
        "split": _split_report(result.split),
        "device": result.device.to_metadata(),
    }


def _split_report(split: ExchangeValueSplit) -> dict[str, object]:
    return {
        "trainStateCount": len(split.train_state_keys),
        "validationStateCount": len(split.validation_state_keys),
        "finalStateCount": len(split.final_state_keys),
        "trainSampleCount": len(split.train_samples),
        "validationSampleCount": len(split.validation_samples),
        "finalSampleCount": len(split.final_samples),
        "trainStateKeyHash": split.train_state_key_hash,
        "validationStateKeyHash": split.validation_state_key_hash,
        "finalStateKeyHash": split.final_state_key_hash,
        "leakageGuard": split.leakage_guard,
    }


def _validate_train_config(config: ExchangeValueTrainConfig) -> None:
    if config.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if config.input_variant not in EXCHANGE_VALUE_INPUT_VARIANTS:
        raise ValueError(
            f"input_variant must be one of {', '.join(EXCHANGE_VALUE_INPUT_VARIANTS)}."
        )
    if config.loss not in {"mse", "huber"}:
        raise ValueError("loss must be mse or huber.")
    if config.huber_delta <= 0.0:
        raise ValueError("huber_delta must be positive.")
    if config.pairwise_loss_weight < 0.0:
        raise ValueError("pairwise_loss_weight must be non-negative.")
    if config.pointwise_loss_weight < 0.0:
        raise ValueError("pointwise_loss_weight must be non-negative.")
    if config.listwise_loss_weight < 0.0:
        raise ValueError("listwise_loss_weight must be non-negative.")
    if (
        config.pointwise_loss_weight == 0.0
        and config.pairwise_loss_weight == 0.0
        and config.listwise_loss_weight == 0.0
    ):
        raise ValueError("at least one training loss weight must be positive.")
    if config.listwise_temperature <= 0.0:
        raise ValueError("listwise_temperature must be positive.")
    if config.pairwise_state_batch_size <= 0:
        raise ValueError("pairwise_state_batch_size must be positive.")
    if config.patience <= 0:
        raise ValueError("patience must be positive.")


def _input_dim_for_variant(input_variant: ExchangeValueInputVariant) -> int:
    if input_variant == "legacy2724":
        return EXCHANGE_VALUE_INPUT_FEATURE_COUNT
    if input_variant == "compact396":
        return EXCHANGE_COMPACT_VALUE_INPUT_FEATURE_COUNT
    if input_variant == "compact406":
        return EXCHANGE_TACTICAL_VALUE_INPUT_FEATURE_COUNT
    raise ValueError(f"unsupported exchange value input variant: {input_variant}.")


def _configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
