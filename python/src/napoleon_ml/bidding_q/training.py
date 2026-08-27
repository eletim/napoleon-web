"""Training, checkpointing, ONNX export, and diagnostics for bidding Q models."""

from __future__ import annotations

import json
import math
import pickle
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import numpy as np
import torch
from torch import Tensor, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from napoleon_ml.dataset.constants import BIDDING_ACTION_COUNT
from napoleon_ml.dataset.tensors import (
    BIDDING_MODEL_INPUT_FEATURE_COUNT,
    BIDDING_MODEL_INPUT_SCHEMA_VERSION,
)
from napoleon_ml.dataset.validation import calculate_card_ids_sha256
from napoleon_ml.policy.device import (
    RequestedTorchDevice,
    ResolvedTorchDevice,
    cpu_state_dict,
    resolve_torch_device,
)

from .dataset import (
    BIDDING_Q_ACTION_MAPPING_ID,
    BIDDING_Q_SUITS,
    BIDDING_Q_TARGETS,
    BiddingQAggregatedExample,
    BiddingQDataset,
    BiddingQDatasetManifest,
    BiddingQSplit,
    create_bidding_q_split,
)
from .model import (
    BIDDING_Q_MLP_ARCHITECTURE_ID,
    BiddingQModel,
    BiddingQModelConfig,
    create_seeded_bidding_q_model,
)

BIDDING_Q_CHECKPOINT_SCHEMA_VERSION = 1
BIDDING_Q_MODEL_TYPE = "bidding-q"
BIDDING_Q_SELECTION_SEMANTICS = "legal-q-argmax"
BIDDING_Q_ONNX_METADATA_SCHEMA_VERSION = 1
BIDDING_Q_ONNX_OPSET_VERSION = 18
BIDDING_Q_ONNX_INPUT_NAME = "model_input"
BIDDING_Q_ONNX_OUTPUT_NAME = "q_values"
SUPPORTED_BIDDING_Q_LOSSES = ("huber", "mse")


class BiddingQCheckpointError(ValueError):
    """Raised when a bidding Q checkpoint or artifact is incompatible."""


@dataclass(frozen=True)
class BiddingQTrainConfig:
    seed: int = 370
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    hidden_dims: tuple[int, ...] = (512, 512, 256, 256)
    dropout: float = 0.0
    loss_type: Literal["huber", "mse"] = "huber"
    train_ratio: float = 0.8
    device: RequestedTorchDevice = "cpu"
    optimizer: str = "AdamW"
    target_normalization: Literal["none", "standardize"] = "none"

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["hidden_dims"] = list(self.hidden_dims)
        data["algorithm"] = "monte-carlo-fitted-q-regression-v1"
        return data


@dataclass(frozen=True)
class BiddingQTargetNormalization:
    kind: Literal["none", "standardize"]
    mean: float
    std: float

    def encode(self, value: Tensor) -> Tensor:
        if self.kind == "none":
            return value
        return (value - self.mean) / self.std

    def decode(self, value: Tensor) -> Tensor:
        if self.kind == "none":
            return value
        return value * self.std + self.mean

    def to_dict(self) -> dict[str, object]:
        return {"type": self.kind, "mean": self.mean, "std": self.std}


@dataclass(frozen=True)
class BiddingQTrainResult:
    model: BiddingQModel
    split: BiddingQSplit
    config: BiddingQTrainConfig
    target_normalization: BiddingQTargetNormalization
    device: ResolvedTorchDevice
    epoch_reports: list[dict[str, object]]
    train_report: dict[str, object]
    validation_report: dict[str, object]
    baselines: dict[str, object]


class BiddingQBatch(NamedTuple):
    model_input: Tensor
    legal_bid_mask: Tensor
    action_index: Tensor
    target: Tensor


class _BiddingQExampleDataset(Dataset[BiddingQBatch]):
    def __init__(
        self,
        examples: Iterable[BiddingQAggregatedExample],
        normalization: BiddingQTargetNormalization,
    ) -> None:
        self.examples = tuple(examples)
        self.normalization = normalization

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> BiddingQBatch:
        example = self.examples[index]
        target = torch.tensor(example.target_mean, dtype=torch.float32)
        return BiddingQBatch(
            model_input=torch.as_tensor(example.model_input, dtype=torch.float32),
            legal_bid_mask=torch.as_tensor(example.legal_bid_mask, dtype=torch.bool),
            action_index=torch.tensor(example.action_index, dtype=torch.long),
            target=self.normalization.encode(target),
        )


def train_bidding_q_model(
    dataset: BiddingQDataset,
    config: BiddingQTrainConfig,
) -> BiddingQTrainResult:
    _validate_train_config(config)
    _configure_reproducibility(config.seed)
    split = create_bidding_q_split(
        dataset.examples,
        train_ratio=config.train_ratio,
        seed=config.seed,
    )
    device = resolve_torch_device(config.device, flag_name="--device")
    model_config = BiddingQModelConfig(hidden_dims=config.hidden_dims, dropout=config.dropout)
    model = create_seeded_bidding_q_model(model_config, seed=config.seed).to(device.torch_device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    normalization = _target_normalization(split.train_examples, config.target_normalization)
    train_loader = _create_loader(
        split.train_examples,
        normalization=normalization,
        batch_size=config.batch_size,
        seed=config.seed,
        shuffle=True,
    )
    epoch_reports: list[dict[str, object]] = []
    for epoch in range(1, config.epochs + 1):
        train_loss = _train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            normalization=normalization,
            loss_type=config.loss_type,
            device=device,
        )
        validation = evaluate_bidding_q_model(
            model,
            split.validation_examples,
            train_examples=split.train_examples,
            normalization=normalization,
            loss_type=config.loss_type,
            device=device,
        )
        regression = cast(dict[str, object], validation["regression"])
        ranking = cast(dict[str, object], validation["ranking"])
        epoch_reports.append(
            {
                "epoch": epoch,
                "trainLoss": train_loss,
                "validationMae": regression["mae"],
                "validationRmse": regression["rmse"],
                "validationPearson": regression["pearsonCorrelation"],
                "bestActionHitRate": ranking["bestActionHitRate"],
            }
        )
    train_report = evaluate_bidding_q_model(
        model,
        split.train_examples,
        train_examples=split.train_examples,
        normalization=normalization,
        loss_type=config.loss_type,
        device=device,
    )
    validation_report = evaluate_bidding_q_model(
        model,
        split.validation_examples,
        train_examples=split.train_examples,
        normalization=normalization,
        loss_type=config.loss_type,
        device=device,
    )
    return BiddingQTrainResult(
        model=model,
        split=split,
        config=config,
        target_normalization=normalization,
        device=device,
        epoch_reports=epoch_reports,
        train_report=train_report,
        validation_report=validation_report,
        baselines=cast(dict[str, object], validation_report["baselines"]),
    )


def bidding_q_selected_action_loss(
    q_values: Tensor,
    action_index: Tensor,
    target: Tensor,
    *,
    loss_type: Literal["huber", "mse"] = "huber",
) -> Tensor:
    if q_values.ndim != 2 or q_values.shape[1] != BIDDING_ACTION_COUNT:
        raise ValueError("q_values must have shape (batch, 29).")
    if action_index.ndim != 1 or target.ndim != 1:
        raise ValueError("action_index and target must be one-dimensional.")
    if q_values.shape[0] != action_index.shape[0] or q_values.shape[0] != target.shape[0]:
        raise ValueError("batch dimension mismatch.")
    prediction = q_values.gather(1, action_index.unsqueeze(1)).squeeze(1)
    if loss_type == "huber":
        return F.smooth_l1_loss(prediction, target)
    if loss_type == "mse":
        return F.mse_loss(prediction, target)
    raise ValueError(f"unsupported loss type: {loss_type}.")


def evaluate_bidding_q_model(
    model: BiddingQModel,
    examples: Iterable[BiddingQAggregatedExample],
    *,
    train_examples: Iterable[BiddingQAggregatedExample],
    normalization: BiddingQTargetNormalization,
    loss_type: Literal["huber", "mse"],
    device: ResolvedTorchDevice,
) -> dict[str, object]:
    example_tuple = tuple(examples)
    if not example_tuple:
        raise ValueError("evaluation examples must not be empty.")
    predictions = _predict_examples(
        model,
        example_tuple,
        normalization=normalization,
        device=device,
    )
    targets = np.asarray([example.target_mean for example in example_tuple], dtype=np.float64)
    selected_predictions = np.asarray(
        [predictions[index, example.action_index] for index, example in enumerate(example_tuple)],
        dtype=np.float64,
    )
    baselines = _baseline_reports(
        train_examples=tuple(train_examples),
        validation_examples=example_tuple,
    )
    return {
        "sampleCount": len(example_tuple),
        "stateCount": len({example.state_key for example in example_tuple}),
        "coverage": coverage_diagnostics(example_tuple),
        "regression": _regression_metrics(
            selected_predictions,
            targets,
            loss_type=loss_type,
        ),
        "baselines": baselines,
        "ranking": ranking_metrics(example_tuple, predictions),
        "qCollapseDiagnostics": q_collapse_diagnostics(example_tuple, predictions),
    }


def coverage_diagnostics(examples: Iterable[BiddingQAggregatedExample]) -> dict[str, object]:
    example_tuple = tuple(examples)
    action_counts = {str(index): 0 for index in range(BIDDING_ACTION_COUNT)}
    suit_counts = {suit: 0 for suit in BIDDING_Q_SUITS}
    target_counts = {str(target): 0 for target in BIDDING_Q_TARGETS}
    strongest_counts = {suit: 0 for suit in BIDDING_Q_SUITS}
    pass_count = 0
    bid_count = 0
    for example in example_tuple:
        action_counts[str(example.action_index)] += 1
        strongest_counts[example.strongest_suit] += 1
        if example.action_index == 0:
            pass_count += 1
            continue
        bid_count += 1
        if example.suit is not None:
            suit_counts[example.suit] += 1
        if example.target_point_cards is not None:
            target_counts[str(example.target_point_cards)] += 1
    return {
        "sampleCount": len(example_tuple),
        "passCount": pass_count,
        "bidCount": bid_count,
        "suitCounts": suit_counts,
        "targetCounts": target_counts,
        "actionIndexCounts": action_counts,
        "strongestSuitCounts": strongest_counts,
    }


def ranking_metrics(
    examples: Iterable[BiddingQAggregatedExample],
    q_values: np.ndarray,
) -> dict[str, object]:
    example_tuple = tuple(examples)
    by_state = _examples_by_state(example_tuple)
    q_by_example = {id(example): q_values[index] for index, example in enumerate(example_tuple)}
    best_hits = 0
    top3_hits = 0
    states_with_multiple = 0
    pairwise_correct = 0
    pairwise_total = 0
    top_gaps: list[float] = []
    pass_vs_bid_correct = 0
    pass_vs_bid_total = 0
    suit_best_hits = 0
    suit_best_total = 0
    target_best_hits = 0
    target_best_total = 0
    for state_examples in by_state.values():
        if len(state_examples) < 2:
            continue
        states_with_multiple += 1
        teacher_best = max(
            state_examples,
            key=lambda example: (example.target_mean, -example.action_index),
        )
        predicted_order = sorted(
            state_examples,
            key=lambda example: (
                float(q_by_example[id(example)][example.action_index]),
                -example.action_index,
            ),
            reverse=True,
        )
        predicted_best = predicted_order[0]
        if teacher_best.action_index == predicted_best.action_index:
            best_hits += 1
        if teacher_best.action_index in {example.action_index for example in predicted_order[:3]}:
            top3_hits += 1
        if len(predicted_order) >= 2:
            top_gaps.append(
                float(
                    q_by_example[id(predicted_order[0])][predicted_order[0].action_index]
                    - q_by_example[id(predicted_order[1])][predicted_order[1].action_index]
                )
            )
        for left_index, left in enumerate(state_examples):
            for right in state_examples[left_index + 1 :]:
                teacher_delta = left.target_mean - right.target_mean
                if teacher_delta == 0.0:
                    continue
                pred_delta = (
                    float(q_by_example[id(left)][left.action_index])
                    - float(q_by_example[id(right)][right.action_index])
                )
                pairwise_total += 1
                if teacher_delta * pred_delta > 0:
                    pairwise_correct += 1
        pass_examples = [example for example in state_examples if example.action_index == 0]
        bid_examples = [example for example in state_examples if example.action_index != 0]
        if pass_examples and bid_examples:
            pass_example = pass_examples[0]
            best_bid = max(bid_examples, key=lambda example: example.target_mean)
            teacher_prefers_pass = pass_example.target_mean >= best_bid.target_mean
            predicted_prefers_pass = (
                q_by_example[id(pass_example)][pass_example.action_index]
                >= q_by_example[id(best_bid)][best_bid.action_index]
            )
            pass_vs_bid_total += 1
            if bool(teacher_prefers_pass) == bool(predicted_prefers_pass):
                pass_vs_bid_correct += 1
        bid_teacher_best = max(
            [example for example in state_examples if example.suit is not None],
            key=lambda example: example.target_mean,
            default=None,
        )
        bid_predicted_best = max(
            [example for example in state_examples if example.suit is not None],
            key=lambda example: q_by_example[id(example)][example.action_index],
            default=None,
        )
        if bid_teacher_best is not None and bid_predicted_best is not None:
            suit_best_total += 1
            target_best_total += 1
            if bid_teacher_best.suit == bid_predicted_best.suit:
                suit_best_hits += 1
            if bid_teacher_best.target_point_cards == bid_predicted_best.target_point_cards:
                target_best_hits += 1
    random_best_rate = _mean(
        [
            1.0 / len(state_examples)
            for state_examples in by_state.values()
            if len(state_examples) >= 2
        ]
    )
    return {
        "stateCount": states_with_multiple,
        "randomBestActionHitRate": random_best_rate,
        "bestActionHitRate": _safe_rate(best_hits, states_with_multiple),
        "top3HitRate": _safe_rate(top3_hits, states_with_multiple),
        "pairwiseRankingAccuracy": _safe_rate(pairwise_correct, pairwise_total),
        "pairwiseComparisonCount": pairwise_total,
        "predictedTop1Top2QGapMean": _mean(top_gaps),
        "passVsBestBidAccuracy": _safe_rate(pass_vs_bid_correct, pass_vs_bid_total),
        "passVsBestBidStateCount": pass_vs_bid_total,
        "suitLevelBestHitRate": _safe_rate(suit_best_hits, suit_best_total),
        "targetLevelBestHitRate": _safe_rate(target_best_hits, target_best_total),
    }


def q_collapse_diagnostics(
    examples: Iterable[BiddingQAggregatedExample],
    q_values: np.ndarray,
) -> dict[str, object]:
    example_tuple = tuple(examples)
    by_state = _examples_by_state(example_tuple)
    q_by_example = {id(example): q_values[index] for index, example in enumerate(example_tuple)}
    pass_count = 0
    bid_count = 0
    suit_counts = {suit: 0 for suit in BIDDING_Q_SUITS}
    target_counts = {str(target): 0 for target in BIDDING_Q_TARGETS}
    strongest_match = 0
    strongest_match_total = 0
    pass_minus_bid_gaps: list[float] = []
    for state_examples in by_state.values():
        predicted_best = max(
            state_examples,
            key=lambda example: (
                float(q_by_example[id(example)][example.action_index]),
                -example.action_index,
            ),
        )
        if predicted_best.action_index == 0:
            pass_count += 1
        else:
            bid_count += 1
            if predicted_best.suit is not None:
                suit_counts[predicted_best.suit] += 1
                strongest_match_total += 1
                if predicted_best.suit == predicted_best.strongest_suit:
                    strongest_match += 1
            if predicted_best.target_point_cards is not None:
                target_counts[str(predicted_best.target_point_cards)] += 1
        pass_example = next(
            (example for example in state_examples if example.action_index == 0),
            None,
        )
        bid_examples = [example for example in state_examples if example.action_index != 0]
        if pass_example is not None and bid_examples:
            best_bid = max(
                bid_examples,
                key=lambda example: float(q_by_example[id(example)][example.action_index]),
            )
            pass_minus_bid_gaps.append(
                float(
                    q_by_example[id(pass_example)][pass_example.action_index]
                    - q_by_example[id(best_bid)][best_bid.action_index]
                )
            )
    state_count = len(by_state)
    return {
        "stateCount": state_count,
        "predictedPassCount": pass_count,
        "predictedBidCount": bid_count,
        "predictedPassRate": _safe_rate(pass_count, state_count),
        "predictedSuitCounts": suit_counts,
        "predictedTargetCounts": target_counts,
        "strongestSuitMatchRate": _safe_rate(strongest_match, strongest_match_total),
        "passMinusBestBidQGapMean": _mean(pass_minus_bid_gaps),
    }


def save_bidding_q_artifact(
    output_directory: Path | str,
    *,
    result: BiddingQTrainResult,
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
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "epochs": result.epoch_reports,
        "train": result.train_report,
        "validation": result.validation_report,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    parity = export_bidding_q_onnx(
        model=result.model,
        metadata=metadata,
        onnx_path=onnx_path,
        metadata_path=onnx_metadata_path,
        sample_model_input=dataset.examples[0].model_input,
    )
    return {
        "checkpointPath": str(checkpoint_path),
        "metadataPath": str(metadata_path),
        "reportPath": str(report_path),
        "onnxPath": str(onnx_path),
        "onnxMetadataPath": str(onnx_metadata_path),
        "onnxParity": parity,
    }


def load_bidding_q_checkpoint(path: Path | str) -> tuple[BiddingQModel, dict[str, object]]:
    try:
        raw = torch.load(Path(path), map_location="cpu", weights_only=True)
    except OSError as error:
        raise BiddingQCheckpointError(f"checkpoint cannot be read: {error}") from error
    except RuntimeError as error:
        raise BiddingQCheckpointError(f"checkpoint cannot be loaded: {error}") from error
    except pickle.UnpicklingError as error:
        raise BiddingQCheckpointError(f"checkpoint cannot be loaded: {error}") from error
    if not isinstance(raw, dict):
        raise BiddingQCheckpointError("checkpoint must be a dictionary.")
    if raw.get("checkpointSchemaVersion") != BIDDING_Q_CHECKPOINT_SCHEMA_VERSION:
        raise BiddingQCheckpointError("checkpoint schema version mismatch.")
    if raw.get("modelType") != BIDDING_Q_MODEL_TYPE:
        raise BiddingQCheckpointError("checkpoint modelType mismatch.")
    config_raw = raw.get("modelConfig")
    if not isinstance(config_raw, dict):
        raise BiddingQCheckpointError("checkpoint modelConfig must be an object.")
    model = BiddingQModel(BiddingQModelConfig.from_dict(config_raw))
    state = raw.get("modelState")
    if not isinstance(state, dict):
        raise BiddingQCheckpointError("checkpoint modelState must be a state dict.")
    model.load_state_dict(state)
    return model, raw


def export_bidding_q_onnx(
    *,
    model: BiddingQModel,
    metadata: dict[str, object],
    onnx_path: Path | str,
    metadata_path: Path | str,
    sample_model_input: np.ndarray,
) -> dict[str, object]:
    output = Path(onnx_path)
    metadata_output = Path(metadata_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    model_cpu = BiddingQModel(model.config)
    model_cpu.load_state_dict(cpu_state_dict(model))
    model_cpu.eval()
    dummy = torch.zeros((2, BIDDING_MODEL_INPUT_FEATURE_COUNT), dtype=torch.float32)
    torch.onnx.export(
        model_cpu,
        (dummy,),
        output,
        input_names=[BIDDING_Q_ONNX_INPUT_NAME],
        output_names=[BIDDING_Q_ONNX_OUTPUT_NAME],
        dynamic_axes={
            BIDDING_Q_ONNX_INPUT_NAME: {0: "batch"},
            BIDDING_Q_ONNX_OUTPUT_NAME: {0: "batch"},
        },
        opset_version=BIDDING_Q_ONNX_OPSET_VERSION,
    )
    onnx_metadata = dict(metadata)
    onnx_metadata.update(
        {
            "metadataSchemaVersion": BIDDING_Q_ONNX_METADATA_SCHEMA_VERSION,
            "artifactType": "napoleon-bidding-q-onnx",
            "modelType": BIDDING_Q_MODEL_TYPE,
            "policyType": BIDDING_Q_MODEL_TYPE,
            "selectionSemantics": BIDDING_Q_SELECTION_SEMANTICS,
            "inputName": BIDDING_Q_ONNX_INPUT_NAME,
            "outputName": BIDDING_Q_ONNX_OUTPUT_NAME,
            "outputValueType": "q-values",
            "onnx": {
                "opsetVersion": BIDDING_Q_ONNX_OPSET_VERSION,
                "inputs": [
                    {
                        "name": BIDDING_Q_ONNX_INPUT_NAME,
                        "dtype": "float32",
                        "shape": ["batch", BIDDING_MODEL_INPUT_FEATURE_COUNT],
                    }
                ],
                "outputs": [
                    {
                        "name": BIDDING_Q_ONNX_OUTPUT_NAME,
                        "dtype": "float32",
                        "shape": ["batch", BIDDING_ACTION_COUNT],
                    }
                ],
            },
        }
    )
    metadata_output.write_text(
        json.dumps(onnx_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    parity = _check_onnx_parity(
        model_cpu=model_cpu,
        onnx_path=output,
        sample_model_input=sample_model_input,
    )
    return parity


def _train_one_epoch(
    *,
    model: BiddingQModel,
    dataloader: DataLoader[BiddingQBatch],
    optimizer: optim.Optimizer,
    normalization: BiddingQTargetNormalization,
    loss_type: Literal["huber", "mse"],
    device: ResolvedTorchDevice,
) -> float:
    del normalization
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in dataloader:
        model_input = batch.model_input.to(device=device.torch_device, dtype=torch.float32)
        action_index = batch.action_index.to(device=device.torch_device, dtype=torch.long)
        target = batch.target.to(device=device.torch_device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        q_values = model(model_input)
        loss = bidding_q_selected_action_loss(q_values, action_index, target, loss_type=loss_type)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        batch_size = int(action_index.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
    if total_samples == 0:
        raise ValueError("train split contains no samples.")
    return total_loss / total_samples


def _create_loader(
    examples: Iterable[BiddingQAggregatedExample],
    *,
    normalization: BiddingQTargetNormalization,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> DataLoader[BiddingQBatch]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        _BiddingQExampleDataset(examples, normalization),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def _predict_examples(
    model: BiddingQModel,
    examples: tuple[BiddingQAggregatedExample, ...],
    *,
    normalization: BiddingQTargetNormalization,
    device: ResolvedTorchDevice,
) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(examples), 512):
            batch = examples[start : start + 512]
            model_input = torch.as_tensor(
                np.stack([example.model_input for example in batch]),
                dtype=torch.float32,
                device=device.torch_device,
            )
            normalized_q = model(model_input)
            q_values = normalization.decode(normalized_q).detach().cpu().numpy()
            values.append(q_values)
    return np.concatenate(values, axis=0)


def _regression_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    loss_type: Literal["huber", "mse"],
) -> dict[str, float | None]:
    diff = prediction - target
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    huber = float(np.mean(np.where(np.abs(diff) < 1.0, 0.5 * diff**2, np.abs(diff) - 0.5)))
    return {
        "lossType": loss_type,  # type: ignore[dict-item]
        "huber": huber,
        "mse": mse,
        "mae": mae,
        "rmse": math.sqrt(mse),
        "predictedQMean": float(prediction.mean()),
        "predictedQStd": float(prediction.std()),
        "targetMean": float(target.mean()),
        "targetStd": float(target.std()),
        "pearsonCorrelation": _pearson(prediction, target),
    }


def _baseline_reports(
    *,
    train_examples: tuple[BiddingQAggregatedExample, ...],
    validation_examples: tuple[BiddingQAggregatedExample, ...],
) -> dict[str, object]:
    targets = np.asarray([example.target_mean for example in validation_examples], dtype=np.float64)
    global_mean = float(np.mean([example.target_mean for example in train_examples]))
    action_means = _mean_by_key(train_examples, key=lambda example: str(example.action_index))
    suit_target_means = _mean_by_key(
        train_examples,
        key=lambda example: _suit_target_key(example),
    )

    def values_for(mapping: dict[str, float], key_fn: Any) -> np.ndarray:
        return np.asarray(
            [mapping.get(key_fn(example), global_mean) for example in validation_examples],
            dtype=np.float64,
        )

    return {
        "globalMean": _baseline_metric(np.full_like(targets, global_mean), targets),
        "actionIndexMean": _baseline_metric(
            values_for(action_means, lambda example: str(example.action_index)),
            targets,
        ),
        "suitTargetBucketMean": _baseline_metric(
            values_for(suit_target_means, _suit_target_key),
            targets,
        ),
    }


def _baseline_metric(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | None]:
    diff = prediction - target
    mse = float(np.mean(diff**2))
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": math.sqrt(mse),
        "mse": mse,
        "pearsonCorrelation": _pearson(prediction, target),
    }


def _mean_by_key(
    examples: tuple[BiddingQAggregatedExample, ...],
    *,
    key: Any,
) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for example in examples:
        values[str(key(example))].append(example.target_mean)
    return {name: float(np.mean(items)) for name, items in values.items()}


def _suit_target_key(example: BiddingQAggregatedExample) -> str:
    if example.action_index == 0:
        return "PASS"
    return f"{example.target_point_cards}:{example.suit}"


def _examples_by_state(
    examples: tuple[BiddingQAggregatedExample, ...],
) -> dict[str, list[BiddingQAggregatedExample]]:
    by_state: dict[str, list[BiddingQAggregatedExample]] = defaultdict(list)
    for example in examples:
        by_state[example.state_key].append(example)
    return by_state


def _checkpoint_dict(
    *,
    result: BiddingQTrainResult,
    dataset: BiddingQDataset,
) -> dict[str, object]:
    return {
        "checkpointSchemaVersion": BIDDING_Q_CHECKPOINT_SCHEMA_VERSION,
        "modelType": BIDDING_Q_MODEL_TYPE,
        "architectureId": BIDDING_Q_MLP_ARCHITECTURE_ID,
        "selectionSemantics": BIDDING_Q_SELECTION_SEMANTICS,
        "modelState": cpu_state_dict(result.model),
        "modelConfig": result.model.config.to_dict(),
        "trainingConfig": result.config.to_dict(),
        "targetNormalization": result.target_normalization.to_dict(),
        "dataset": _dataset_identity(dataset.manifest),
        "reward": {
            "id": dataset.manifest.reward_id,
            "version": dataset.manifest.reward_version,
        },
        "compactObservation": {
            "phase": "bidding",
            "modelInputSchemaVersion": BIDDING_MODEL_INPUT_SCHEMA_VERSION,
            "modelInputFeatureCount": BIDDING_MODEL_INPUT_FEATURE_COUNT,
        },
        "actionMapping": {
            "id": BIDDING_Q_ACTION_MAPPING_ID,
            "actionCount": BIDDING_ACTION_COUNT,
        },
        "split": _split_metadata(result.split),
        "metrics": {
            "epochs": result.epoch_reports,
            "train": result.train_report,
            "validation": result.validation_report,
        },
        "cardIdsSha256": calculate_card_ids_sha256(),
    }


def _metadata_dict(
    *,
    result: BiddingQTrainResult,
    dataset: BiddingQDataset,
    checkpoint_path: Path,
) -> dict[str, object]:
    return {
        "metadataSchemaVersion": 1,
        "modelType": BIDDING_Q_MODEL_TYPE,
        "policyType": BIDDING_Q_MODEL_TYPE,
        "selectionSemantics": BIDDING_Q_SELECTION_SEMANTICS,
        "architectureId": BIDDING_Q_MLP_ARCHITECTURE_ID,
        "checkpointFileName": checkpoint_path.name,
        "modelConfig": result.model.config.to_dict(),
        "trainingConfig": result.config.to_dict(),
        "targetNormalization": result.target_normalization.to_dict(),
        "dataset": _dataset_identity(dataset.manifest),
        "reward": {
            "id": dataset.manifest.reward_id,
            "version": dataset.manifest.reward_version,
        },
        "compactObservation": {
            "phase": "bidding",
            "modelInputSchemaVersion": BIDDING_MODEL_INPUT_SCHEMA_VERSION,
            "modelInputFeatureCount": BIDDING_MODEL_INPUT_FEATURE_COUNT,
        },
        "actionMapping": {
            "id": BIDDING_Q_ACTION_MAPPING_ID,
            "actionCount": BIDDING_ACTION_COUNT,
            "passActionIndex": 0,
        },
        "split": _split_metadata(result.split),
        "device": result.device.to_metadata(),
        "finalValidation": result.validation_report,
        "cardIdsSha256": calculate_card_ids_sha256(),
    }


def _dataset_identity(manifest: BiddingQDatasetManifest) -> dict[str, object]:
    return {
        "datasetDirectory": str(manifest.dataset_directory),
        "datasetSchemaVersion": manifest.dataset_schema_version,
        "sampleType": manifest.sample_type,
        "sampleCount": manifest.sample_count,
        "sourceStates": manifest.source_states,
        "forcedStateActionPairs": manifest.forced_state_action_pairs,
        "manifestSha256": manifest.manifest_sha256,
    }


def _split_metadata(split: BiddingQSplit) -> dict[str, object]:
    return {
        "seed": split.seed,
        "trainRatio": split.train_ratio,
        "validationRatio": split.validation_ratio,
        "trainStateCount": len(split.train_state_keys),
        "validationStateCount": len(split.validation_state_keys),
        "trainSampleCount": len(split.train_examples),
        "validationSampleCount": len(split.validation_examples),
    }


def _target_normalization(
    examples: tuple[BiddingQAggregatedExample, ...],
    kind: Literal["none", "standardize"],
) -> BiddingQTargetNormalization:
    if kind == "none":
        return BiddingQTargetNormalization(kind="none", mean=0.0, std=1.0)
    targets = np.asarray([example.target_mean for example in examples], dtype=np.float64)
    std = float(targets.std())
    if std < 1e-8:
        std = 1.0
    return BiddingQTargetNormalization(kind="standardize", mean=float(targets.mean()), std=std)


def _validate_train_config(config: BiddingQTrainConfig) -> None:
    if config.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if config.loss_type not in SUPPORTED_BIDDING_Q_LOSSES:
        raise ValueError(f"loss_type must be one of {SUPPORTED_BIDDING_Q_LOSSES}.")
    if config.target_normalization not in ("none", "standardize"):
        raise ValueError("target_normalization must be none or standardize.")


def _configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _check_onnx_parity(
    *,
    model_cpu: BiddingQModel,
    onnx_path: Path,
    sample_model_input: np.ndarray,
) -> dict[str, object]:
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]
    except ImportError as error:
        raise BiddingQCheckpointError("onnxruntime is required for ONNX parity.") from error
    sample = torch.as_tensor(sample_model_input.reshape(1, -1), dtype=torch.float32)
    with torch.no_grad():
        pytorch = model_cpu(sample).detach().cpu().numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx = session.run([BIDDING_Q_ONNX_OUTPUT_NAME], {BIDDING_Q_ONNX_INPUT_NAME: sample.numpy()})[0]
    max_abs_diff = float(np.max(np.abs(pytorch - onnx)))
    return {
        "maxAbsQDiff": max_abs_diff,
        "pytorchShape": list(pytorch.shape),
        "onnxShape": list(onnx.shape),
        "withinTolerance": max_abs_diff <= 1e-5,
    }
