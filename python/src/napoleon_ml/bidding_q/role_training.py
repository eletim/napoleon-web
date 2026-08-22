"""Training and diagnostics for role-decomposed bidding Q models."""

from __future__ import annotations

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
    BiddingQRawSample,
    BiddingQSplit,
    create_bidding_q_split,
)
from .role_model import (
    BIDDING_ROLE_CLASSES,
    BIDDING_ROLE_COUNT,
    BIDDING_ROLE_Q_ARCHITECTURE_ID,
    BiddingRoleQModel,
    BiddingRoleQModelConfig,
    compose_role_q,
    create_seeded_bidding_role_q_model,
)
from .training import (
    q_collapse_diagnostics,
    ranking_metrics,
)

BIDDING_ROLE_Q_CHECKPOINT_SCHEMA_VERSION = 1
BIDDING_ROLE_Q_MODEL_TYPE = "bidding-role-decomposed-q"
BIDDING_ROLE_Q_SELECTION_SEMANTICS = "legal-role-decomposed-q-argmax"
BIDDING_ROLE_Q_ONNX_METADATA_SCHEMA_VERSION = 1
BIDDING_ROLE_Q_ONNX_OPSET_VERSION = 18
BIDDING_ROLE_Q_ONNX_INPUT_NAME = "model_input"
BIDDING_ROLE_Q_ONNX_LOGITS_OUTPUT_NAME = "role_logits"
BIDDING_ROLE_Q_ONNX_VALUES_OUTPUT_NAME = "role_values"
SUPPORTED_BIDDING_ROLE_Q_VALUE_LOSSES = ("huber", "mse")
SUPPORTED_BIDDING_ROLE_Q_VALUE_TARGETS = ("terminal-reward", "team-point-cards")


class BiddingRoleQCheckpointError(ValueError):
    """Raised when a role-decomposed Q artifact is incompatible."""


@dataclass(frozen=True)
class BiddingRoleQTrainConfig:
    seed: int = 373
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    hidden_dims: tuple[int, ...] = (512, 512, 256, 256)
    dropout: float = 0.0
    value_loss_type: Literal["huber", "mse"] = "huber"
    value_loss_coefficient: float = 1.0
    role_loss_coefficient: float = 1.0
    value_target: Literal["terminal-reward", "team-point-cards"] = "terminal-reward"
    train_ratio: float = 0.8
    device: RequestedTorchDevice = "cpu"
    optimizer: str = "AdamW"

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["hidden_dims"] = list(self.hidden_dims)
        data["algorithm"] = "role-decomposed-monte-carlo-q-v1"
        return data


@dataclass(frozen=True)
class BiddingRoleQTrainResult:
    model: BiddingRoleQModel
    split: BiddingQSplit
    train_samples: tuple[BiddingQRawSample, ...]
    validation_samples: tuple[BiddingQRawSample, ...]
    config: BiddingRoleQTrainConfig
    device: ResolvedTorchDevice
    epoch_reports: list[dict[str, object]]
    train_report: dict[str, object]
    validation_report: dict[str, object]
    dataset_diagnostics: dict[str, object]


class BiddingRoleQBatch(NamedTuple):
    model_input: Tensor
    legal_bid_mask: Tensor
    action_index: Tensor
    role_index: Tensor
    target_reward: Tensor
    value_mask: Tensor


class _BiddingRoleQSampleDataset(Dataset[BiddingRoleQBatch]):
    def __init__(
        self,
        samples: Iterable[BiddingQRawSample],
        *,
        value_target: Literal["terminal-reward", "team-point-cards"],
    ) -> None:
        self.samples = tuple(samples)
        self.value_target = value_target

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> BiddingRoleQBatch:
        sample = self.samples[index]
        value, mask = value_target_for_sample(sample, self.value_target)
        return BiddingRoleQBatch(
            model_input=torch.as_tensor(sample.model_input, dtype=torch.float32),
            legal_bid_mask=torch.as_tensor(sample.legal_bid_mask, dtype=torch.bool),
            action_index=torch.tensor(sample.forced_action_index, dtype=torch.long),
            role_index=torch.tensor(
                role_index_for_terminal_role(sample.terminal_role),
                dtype=torch.long,
            ),
            target_reward=torch.tensor(value, dtype=torch.float32),
            value_mask=torch.tensor(mask, dtype=torch.bool),
        )


def canonical_bidding_role(terminal_role: str) -> str:
    if terminal_role in ("napoleon", "napoleon-adjutant", "adjutant", "citizen", "no-contract"):
        return terminal_role
    if terminal_role in ("all-pass-starter", "all-pass-other"):
        return "no-contract"
    raise ValueError(f"unsupported terminalRole: {terminal_role!r}.")


def role_index_for_terminal_role(terminal_role: str) -> int:
    return BIDDING_ROLE_CLASSES.index(canonical_bidding_role(terminal_role))


def value_target_for_sample(
    sample: BiddingQRawSample,
    value_target: Literal["terminal-reward", "team-point-cards"],
) -> tuple[float, bool]:
    if value_target == "terminal-reward":
        return sample.terminal_reward, True
    if value_target == "team-point-cards":
        if not sample.team_point_cards_regression_mask:
            return 0.0, False
        if sample.candidate_team_point_cards is None:
            raise ValueError(f"{sample.state_key}:{sample.forced_action_index} missing team cards.")
        return float(sample.candidate_team_point_cards), True
    raise ValueError(f"unsupported value target: {value_target}.")


def bidding_role_q_losses(
    role_logits: Tensor,
    role_values: Tensor,
    action_index: Tensor,
    role_index: Tensor,
    target_reward: Tensor,
    *,
    value_loss_type: Literal["huber", "mse"] = "huber",
    value_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    if role_logits.shape != role_values.shape:
        raise ValueError("role_logits and role_values must have the same shape.")
    if role_logits.ndim != 3:
        raise ValueError("role outputs must have shape (batch, 29, roleCount).")
    if role_logits.shape[1:] != (BIDDING_ACTION_COUNT, BIDDING_ROLE_COUNT):
        raise ValueError("role outputs must have shape (batch, 29, roleCount).")
    if action_index.ndim != 1 or role_index.ndim != 1 or target_reward.ndim != 1:
        raise ValueError("action_index, role_index, and target_reward must be one-dimensional.")
    if (
        role_logits.shape[0] != action_index.shape[0]
        or action_index.shape[0] != role_index.shape[0]
        or role_index.shape[0] != target_reward.shape[0]
    ):
        raise ValueError("batch dimension mismatch.")
    batch_index = torch.arange(action_index.shape[0], device=action_index.device)
    selected_logits = role_logits[batch_index, action_index]
    role_loss = F.cross_entropy(selected_logits, role_index)
    selected_values = role_values[batch_index, action_index, role_index]
    if value_mask is not None:
        if value_mask.ndim != 1 or value_mask.shape[0] != selected_values.shape[0]:
            raise ValueError("value_mask must have shape (batch,).")
        selected_values = selected_values[value_mask]
        target_reward = target_reward[value_mask]
    if selected_values.numel() == 0:
        return role_loss, role_values.sum() * 0.0
    if value_loss_type == "huber":
        value_loss = F.smooth_l1_loss(selected_values, target_reward)
    elif value_loss_type == "mse":
        value_loss = F.mse_loss(selected_values, target_reward)
    else:
        raise ValueError(f"unsupported value loss type: {value_loss_type}.")
    return role_loss, value_loss


def train_bidding_role_q_model(
    dataset: BiddingQDataset,
    config: BiddingRoleQTrainConfig,
) -> BiddingRoleQTrainResult:
    _validate_train_config(config)
    _configure_reproducibility(config.seed)
    split = create_bidding_q_split(
        dataset.examples,
        train_ratio=config.train_ratio,
        seed=config.seed,
    )
    train_samples = tuple(
        sample for sample in dataset.raw_samples if sample.state_key in split.train_state_keys
    )
    validation_samples = tuple(
        sample for sample in dataset.raw_samples if sample.state_key in split.validation_state_keys
    )
    if not train_samples or not validation_samples:
        raise ValueError("train and validation raw splits must both contain samples.")
    device = resolve_torch_device(config.device, flag_name="--device")
    model_config = BiddingRoleQModelConfig(hidden_dims=config.hidden_dims, dropout=config.dropout)
    model = create_seeded_bidding_role_q_model(model_config, seed=config.seed).to(
        device.torch_device
    )
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    train_loader = _create_loader(
        train_samples,
        batch_size=config.batch_size,
        seed=config.seed,
        shuffle=True,
        value_target=config.value_target,
    )
    epoch_reports: list[dict[str, object]] = []
    for epoch in range(1, config.epochs + 1):
        losses = _train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            config=config,
            device=device,
        )
        validation = evaluate_bidding_role_q_model(
            model,
            validation_samples,
            validation_examples=split.validation_examples,
            train_samples=train_samples,
            train_examples=split.train_examples,
            value_loss_type=config.value_loss_type,
            value_target=config.value_target,
            device=device,
        )
        classifier = cast(dict[str, object], validation["roleClassifier"])
        ranking = cast(
            dict[str, object],
            validation["teamPointCardsActionSignal"]
            if config.value_target == "team-point-cards"
            else validation["ranking"],
        )
        epoch_reports.append(
            {
                "epoch": epoch,
                "trainLoss": losses["loss"],
                "trainRoleLoss": losses["roleLoss"],
                "trainValueLoss": losses["valueLoss"],
                "validationRoleAccuracy": classifier["accuracy"],
                "validationRoleCrossEntropy": classifier["crossEntropy"],
                "bestActionHitRate": ranking["bestActionHitRate"],
            }
        )
    train_report = evaluate_bidding_role_q_model(
        model,
        train_samples,
        validation_examples=split.train_examples,
        train_samples=train_samples,
        train_examples=split.train_examples,
        value_loss_type=config.value_loss_type,
        value_target=config.value_target,
        device=device,
    )
    validation_report = evaluate_bidding_role_q_model(
        model,
        validation_samples,
        validation_examples=split.validation_examples,
        train_samples=train_samples,
        train_examples=split.train_examples,
        value_loss_type=config.value_loss_type,
        value_target=config.value_target,
        device=device,
    )
    return BiddingRoleQTrainResult(
        model=model,
        split=split,
        train_samples=train_samples,
        validation_samples=validation_samples,
        config=config,
        device=device,
        epoch_reports=epoch_reports,
        train_report=train_report,
        validation_report=validation_report,
        dataset_diagnostics=role_dataset_diagnostics(dataset.raw_samples),
    )


def evaluate_bidding_role_q_model(
    model: BiddingRoleQModel,
    samples: Iterable[BiddingQRawSample],
    *,
    validation_examples: Iterable[BiddingQAggregatedExample],
    train_samples: Iterable[BiddingQRawSample],
    train_examples: Iterable[BiddingQAggregatedExample],
    value_loss_type: Literal["huber", "mse"],
    value_target: Literal["terminal-reward", "team-point-cards"],
    device: ResolvedTorchDevice,
) -> dict[str, object]:
    sample_tuple = tuple(samples)
    example_tuple = tuple(validation_examples)
    if not sample_tuple or not example_tuple:
        raise ValueError("evaluation samples and examples must not be empty.")
    raw_predictions = _predict_samples(model, sample_tuple, device=device)
    example_q_values = _predict_composed_q_for_examples(model, example_tuple, device=device)
    role_report = _role_classifier_metrics(
        samples=sample_tuple,
        role_probabilities=raw_predictions["roleProbabilities"],
        train_samples=tuple(train_samples),
    )
    value_report = _role_value_metrics(
        samples=sample_tuple,
        role_values=raw_predictions["roleValues"],
        value_loss_type=value_loss_type,
        value_target=value_target,
    )
    if value_target == "terminal-reward":
        composed_selected = np.asarray(
            [
                example_q_values[index, example.action_index]
                for index, example in enumerate(example_tuple)
            ],
            dtype=np.float64,
        )
        targets = np.asarray([example.target_mean for example in example_tuple], dtype=np.float64)
        composed_regression: dict[str, object] = dict(
            _regression_metrics(composed_selected, targets)
        )
        ranking: dict[str, object] = ranking_metrics(example_tuple, example_q_values)
        collapse: dict[str, object] = q_collapse_diagnostics(example_tuple, example_q_values)
    else:
        composed_regression = {
            "skipped": True,
            "reason": "team-point-cards target uses raw masked samples",
        }
        ranking = {"skipped": True, "reason": "use teamPointCardsActionSignal"}
        collapse = {
            "skipped": True,
            "reason": "team-point-cards target is not a final action score",
        }
    baselines = _role_baselines(
        train_samples=tuple(train_samples),
        validation_samples=sample_tuple,
        value_target=value_target,
    )
    return {
        "rawSampleCount": len(sample_tuple),
        "aggregatedSampleCount": len(example_tuple),
        "stateCount": len({sample.state_key for sample in sample_tuple}),
        "roleClassifier": role_report,
        "roleValue": value_report,
        "composedQRegression": composed_regression,
        "baselines": baselines,
        "ranking": ranking,
        "qCollapseDiagnostics": collapse,
        "teamPointCardsActionSignal": (
            _team_point_card_action_signal(
                samples=sample_tuple,
                role_probabilities=raw_predictions["roleProbabilities"],
                role_values=raw_predictions["roleValues"],
            )
            if value_target == "team-point-cards"
            else {"skipped": True}
        ),
        "coverage": _raw_coverage_diagnostics(sample_tuple),
        "stateConditionedRoleSignal": _state_conditioned_role_signal(
            role_report,
            baselines,
        ),
    }


def role_dataset_diagnostics(samples: Iterable[BiddingQRawSample]) -> dict[str, object]:
    sample_tuple = tuple(samples)
    if not sample_tuple:
        raise ValueError("samples must not be empty.")
    role_counts = _empty_role_counts()
    action_role_counts = {
        str(index): _empty_role_counts() for index in range(BIDDING_ACTION_COUNT)
    }
    pass_role_counts = _empty_role_counts()
    bid_role_counts = _empty_role_counts()
    target_role_counts = {str(target): _empty_role_counts() for target in BIDDING_Q_TARGETS}
    suit_role_counts = {suit: _empty_role_counts() for suit in BIDDING_Q_SUITS}
    opponent_role_counts: dict[str, dict[str, int]] = defaultdict(_empty_role_counts)
    lowest_suit_role_counts = {suit: _empty_role_counts() for suit in BIDDING_Q_SUITS}
    by_state: dict[str, list[BiddingQRawSample]] = defaultdict(list)
    team_point_cards: list[float] = []
    contract_margins: list[float] = []
    for sample in sample_tuple:
        role = canonical_bidding_role(sample.terminal_role)
        role_counts[role] += 1
        action_role_counts[str(sample.forced_action_index)][role] += 1
        by_state[sample.state_key].append(sample)
        if sample.opponent_configuration_key is not None:
            opponent_role_counts[sample.opponent_configuration_key][role] += 1
        if sample.team_point_cards_regression_mask:
            if sample.candidate_team_point_cards is not None:
                team_point_cards.append(float(sample.candidate_team_point_cards))
            if sample.contract_margin is not None:
                contract_margins.append(float(sample.contract_margin))
        if sample.forced_action_index == 0:
            pass_role_counts[role] += 1
        else:
            bid_role_counts[role] += 1
            if sample.forced_suit is not None:
                suit_role_counts[sample.forced_suit][role] += 1
            if sample.forced_target_point_cards is not None:
                target_role_counts[str(sample.forced_target_point_cards)][role] += 1
    state_tv_distances: dict[str, list[float]] = {suit: [] for suit in BIDDING_Q_SUITS}
    napoleon_rate_deltas: dict[str, list[float]] = {suit: [] for suit in BIDDING_Q_SUITS}
    for state_samples in by_state.values():
        pass_distribution = _role_distribution_for_samples(
            sample for sample in state_samples if sample.forced_action_index == 0
        )
        if pass_distribution is None:
            continue
        lowest_by_suit = _lowest_legal_bid_by_suit(state_samples)
        for suit, suit_samples in lowest_by_suit.items():
            for sample in suit_samples:
                lowest_suit_role_counts[suit][canonical_bidding_role(sample.terminal_role)] += 1
            suit_distribution = _role_distribution_for_samples(suit_samples)
            if suit_distribution is None:
                continue
            state_tv_distances[suit].append(
                _total_variation_distance(pass_distribution, suit_distribution)
            )
            napoleon_rate_deltas[suit].append(
                suit_distribution["napoleon"] + suit_distribution["napoleon-adjutant"]
                - pass_distribution["napoleon"]
                - pass_distribution["napoleon-adjutant"]
            )
    return {
        "sampleCount": len(sample_tuple),
        "stateCount": len(by_state),
        "roleDistribution": _counts_with_rates(role_counts),
        "actionRoleDistribution": {
            action: _counts_with_rates(counts) for action, counts in action_role_counts.items()
        },
        "passRoleDistribution": _counts_with_rates(pass_role_counts),
        "bidRoleDistribution": _counts_with_rates(bid_role_counts),
        "suitRoleDistribution": {
            suit: _counts_with_rates(counts) for suit, counts in suit_role_counts.items()
        },
        "lowestLegalSuitRoleDistribution": {
            suit: _counts_with_rates(counts) for suit, counts in lowest_suit_role_counts.items()
        },
        "targetRoleDistribution": {
            target: _counts_with_rates(counts) for target, counts in target_role_counts.items()
        },
        "opponentConfigurationRoleDistribution": {
            key: _counts_with_rates(counts) for key, counts in sorted(opponent_role_counts.items())
        },
        "teamPointCardsDistribution": _numeric_summary(team_point_cards),
        "contractMarginDistribution": _numeric_summary(contract_margins),
        "sameTargetSuitSignal": _same_target_suit_signal(sample_tuple),
        "targetSignal": _target_signal(sample_tuple),
        "passVsLowestLegalSuitRoleChange": {
            suit: {
                "meanTotalVariationDistance": _mean(values),
                "stateCount": len(values),
                "meanNapoleonProbabilityDelta": _mean(napoleon_rate_deltas[suit]),
            }
            for suit, values in state_tv_distances.items()
        },
    }


def save_bidding_role_q_artifact(
    output_directory: Path | str,
    *,
    result: BiddingRoleQTrainResult,
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
        "datasetDiagnostics": result.dataset_diagnostics,
        "epochs": result.epoch_reports,
        "train": result.train_report,
        "validation": result.validation_report,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    parity = export_bidding_role_q_onnx(
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


def load_bidding_role_q_checkpoint(
    path: Path | str,
) -> tuple[BiddingRoleQModel, dict[str, object]]:
    try:
        raw = torch.load(Path(path), map_location="cpu", weights_only=True)
    except OSError as error:
        raise BiddingRoleQCheckpointError(f"checkpoint cannot be read: {error}") from error
    except RuntimeError as error:
        raise BiddingRoleQCheckpointError(f"checkpoint cannot be loaded: {error}") from error
    except pickle.UnpicklingError as error:
        raise BiddingRoleQCheckpointError(f"checkpoint cannot be loaded: {error}") from error
    if not isinstance(raw, dict):
        raise BiddingRoleQCheckpointError("checkpoint must be a dictionary.")
    if raw.get("checkpointSchemaVersion") != BIDDING_ROLE_Q_CHECKPOINT_SCHEMA_VERSION:
        raise BiddingRoleQCheckpointError("checkpoint schema version mismatch.")
    if raw.get("modelType") != BIDDING_ROLE_Q_MODEL_TYPE:
        raise BiddingRoleQCheckpointError("checkpoint modelType mismatch.")
    config_raw = raw.get("modelConfig")
    if not isinstance(config_raw, dict):
        raise BiddingRoleQCheckpointError("checkpoint modelConfig must be an object.")
    model = BiddingRoleQModel(BiddingRoleQModelConfig.from_dict(config_raw))
    state = raw.get("modelState")
    if not isinstance(state, dict):
        raise BiddingRoleQCheckpointError("checkpoint modelState must be a state dict.")
    model.load_state_dict(state)
    return model, raw


def export_bidding_role_q_onnx(
    *,
    model: BiddingRoleQModel,
    metadata: dict[str, object],
    onnx_path: Path | str,
    metadata_path: Path | str,
    sample_model_input: np.ndarray,
) -> dict[str, object]:
    output = Path(onnx_path)
    metadata_output = Path(metadata_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    model_cpu = BiddingRoleQModel(model.config)
    model_cpu.load_state_dict(cpu_state_dict(model))
    model_cpu.eval()
    dummy = torch.zeros((2, BIDDING_MODEL_INPUT_FEATURE_COUNT), dtype=torch.float32)
    torch.onnx.export(
        model_cpu,
        (dummy,),
        output,
        input_names=[BIDDING_ROLE_Q_ONNX_INPUT_NAME],
        output_names=[
            BIDDING_ROLE_Q_ONNX_LOGITS_OUTPUT_NAME,
            BIDDING_ROLE_Q_ONNX_VALUES_OUTPUT_NAME,
        ],
        dynamic_axes={
            BIDDING_ROLE_Q_ONNX_INPUT_NAME: {0: "batch"},
            BIDDING_ROLE_Q_ONNX_LOGITS_OUTPUT_NAME: {0: "batch"},
            BIDDING_ROLE_Q_ONNX_VALUES_OUTPUT_NAME: {0: "batch"},
        },
        opset_version=BIDDING_ROLE_Q_ONNX_OPSET_VERSION,
    )
    onnx_metadata = dict(metadata)
    onnx_metadata.update(
        {
            "metadataSchemaVersion": BIDDING_ROLE_Q_ONNX_METADATA_SCHEMA_VERSION,
            "artifactType": "napoleon-bidding-role-decomposed-q-onnx",
            "modelType": BIDDING_ROLE_Q_MODEL_TYPE,
            "policyType": BIDDING_ROLE_Q_MODEL_TYPE,
            "selectionSemantics": BIDDING_ROLE_Q_SELECTION_SEMANTICS,
            "inputName": BIDDING_ROLE_Q_ONNX_INPUT_NAME,
            "outputNames": [
                BIDDING_ROLE_Q_ONNX_LOGITS_OUTPUT_NAME,
                BIDDING_ROLE_Q_ONNX_VALUES_OUTPUT_NAME,
            ],
            "outputValueType": "role-logits-and-role-conditioned-values",
            "onnx": {
                "opsetVersion": BIDDING_ROLE_Q_ONNX_OPSET_VERSION,
                "inputs": [
                    {
                        "name": BIDDING_ROLE_Q_ONNX_INPUT_NAME,
                        "dtype": "float32",
                        "shape": ["batch", BIDDING_MODEL_INPUT_FEATURE_COUNT],
                    }
                ],
                "outputs": [
                    {
                        "name": BIDDING_ROLE_Q_ONNX_LOGITS_OUTPUT_NAME,
                        "dtype": "float32",
                        "shape": ["batch", BIDDING_ACTION_COUNT, BIDDING_ROLE_COUNT],
                    },
                    {
                        "name": BIDDING_ROLE_Q_ONNX_VALUES_OUTPUT_NAME,
                        "dtype": "float32",
                        "shape": ["batch", BIDDING_ACTION_COUNT, BIDDING_ROLE_COUNT],
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


def _train_one_epoch(
    *,
    model: BiddingRoleQModel,
    dataloader: DataLoader[BiddingRoleQBatch],
    optimizer: optim.Optimizer,
    config: BiddingRoleQTrainConfig,
    device: ResolvedTorchDevice,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_role_loss = 0.0
    total_value_loss = 0.0
    total_samples = 0
    for batch in dataloader:
        model_input = batch.model_input.to(device=device.torch_device, dtype=torch.float32)
        action_index = batch.action_index.to(device=device.torch_device, dtype=torch.long)
        role_index = batch.role_index.to(device=device.torch_device, dtype=torch.long)
        target_reward = batch.target_reward.to(device=device.torch_device, dtype=torch.float32)
        value_mask = batch.value_mask.to(device=device.torch_device, dtype=torch.bool)
        optimizer.zero_grad(set_to_none=True)
        role_logits, role_values = model(model_input)
        role_loss, value_loss = bidding_role_q_losses(
            role_logits,
            role_values,
            action_index,
            role_index,
            target_reward,
            value_loss_type=config.value_loss_type,
            value_mask=value_mask,
        )
        loss = (
            config.role_loss_coefficient * role_loss
            + config.value_loss_coefficient * value_loss
        )
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        batch_size = int(action_index.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_role_loss += float(role_loss.item()) * batch_size
        total_value_loss += float(value_loss.item()) * batch_size
        total_samples += batch_size
    if total_samples == 0:
        raise ValueError("train split contains no samples.")
    return {
        "loss": total_loss / total_samples,
        "roleLoss": total_role_loss / total_samples,
        "valueLoss": total_value_loss / total_samples,
    }


def _create_loader(
    samples: Iterable[BiddingQRawSample],
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
    value_target: Literal["terminal-reward", "team-point-cards"],
) -> DataLoader[BiddingRoleQBatch]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        _BiddingRoleQSampleDataset(samples, value_target=value_target),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def _predict_samples(
    model: BiddingRoleQModel,
    samples: tuple[BiddingQRawSample, ...],
    *,
    device: ResolvedTorchDevice,
) -> dict[str, np.ndarray]:
    model.eval()
    probability_batches: list[np.ndarray] = []
    value_batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(samples), 512):
            batch = samples[start : start + 512]
            model_input = torch.as_tensor(
                np.stack([sample.model_input for sample in batch]),
                dtype=torch.float32,
                device=device.torch_device,
            )
            role_logits, role_values = model(model_input)
            probabilities = torch.softmax(role_logits, dim=2)
            probability_batches.append(probabilities.detach().cpu().numpy())
            value_batches.append(role_values.detach().cpu().numpy())
    return {
        "roleProbabilities": np.concatenate(probability_batches, axis=0),
        "roleValues": np.concatenate(value_batches, axis=0),
    }


def _predict_composed_q_for_examples(
    model: BiddingRoleQModel,
    examples: tuple[BiddingQAggregatedExample, ...],
    *,
    device: ResolvedTorchDevice,
) -> np.ndarray:
    model.eval()
    q_batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(examples), 512):
            batch = examples[start : start + 512]
            model_input = torch.as_tensor(
                np.stack([example.model_input for example in batch]),
                dtype=torch.float32,
                device=device.torch_device,
            )
            role_logits, role_values = model(model_input)
            q_values = compose_role_q(role_logits, role_values)
            q_batches.append(q_values.detach().cpu().numpy())
    return np.concatenate(q_batches, axis=0)


def _role_classifier_metrics(
    *,
    samples: tuple[BiddingQRawSample, ...],
    role_probabilities: np.ndarray,
    train_samples: tuple[BiddingQRawSample, ...],
) -> dict[str, object]:
    true_roles = np.asarray(
        [role_index_for_terminal_role(sample.terminal_role) for sample in samples]
    )
    selected_probabilities = np.asarray(
        [
            role_probabilities[index, sample.forced_action_index]
            for index, sample in enumerate(samples)
        ],
        dtype=np.float64,
    )
    predictions = np.argmax(selected_probabilities, axis=1)
    confusion = np.zeros((BIDDING_ROLE_COUNT, BIDDING_ROLE_COUNT), dtype=np.int64)
    for truth, prediction in zip(true_roles, predictions, strict=True):
        confusion[int(truth), int(prediction)] += 1
    precision: dict[str, float | None] = {}
    recall: dict[str, float | None] = {}
    f1_scores: list[float] = []
    for index, role in enumerate(BIDDING_ROLE_CLASSES):
        tp = int(confusion[index, index])
        predicted_total = int(confusion[:, index].sum())
        true_total = int(confusion[index, :].sum())
        role_precision = _safe_rate(tp, predicted_total)
        role_recall = _safe_rate(tp, true_total)
        precision[role] = role_precision
        recall[role] = role_recall
        if role_precision is not None and role_recall is not None and role_precision + role_recall:
            f1_scores.append(2 * role_precision * role_recall / (role_precision + role_recall))
    clipped = np.clip(selected_probabilities[np.arange(len(true_roles)), true_roles], 1e-12, 1.0)
    train_role_counts = Counter(
        canonical_bidding_role(sample.terminal_role) for sample in train_samples
    )
    return {
        "sampleCount": len(samples),
        "accuracy": float(np.mean(predictions == true_roles)),
        "crossEntropy": float(-np.mean(np.log(clipped))),
        "macroF1": _mean(f1_scores),
        "classDistribution": _counts_with_rates(
            Counter(canonical_bidding_role(sample.terminal_role) for sample in samples)
        ),
        "trainClassDistribution": _counts_with_rates(train_role_counts),
        "perRolePrecision": precision,
        "perRoleRecall": recall,
        "confusionMatrix": _confusion_to_dict(confusion),
        "calibration": _confidence_calibration(selected_probabilities, true_roles),
    }


def _role_value_metrics(
    *,
    samples: tuple[BiddingQRawSample, ...],
    role_values: np.ndarray,
    value_loss_type: Literal["huber", "mse"],
    value_target: Literal["terminal-reward", "team-point-cards"],
) -> dict[str, object]:
    rows: list[tuple[int, BiddingQRawSample, float]] = []
    for index, sample in enumerate(samples):
        target, mask = value_target_for_sample(sample, value_target)
        if mask:
            rows.append((index, sample, target))
    if not rows:
        return {
            "sampleCount": 0,
            "maskedSampleCount": len(samples),
            "lossType": value_loss_type,
            "valueTarget": value_target,
            "overall": {"mae": None, "rmse": None, "mse": None, "pearsonCorrelation": None},
            "perRole": {role: {"sampleCount": 0} for role in BIDDING_ROLE_CLASSES},
            "perSuit": {},
            "perTarget": {},
            "perHandStrengthBin": {},
        }
    predictions = np.asarray(
        [
            role_values[
                index,
                sample.forced_action_index,
                role_index_for_terminal_role(sample.terminal_role),
            ]
            for index, sample, _target in rows
        ],
        dtype=np.float64,
    )
    targets = np.asarray([target for _index, _sample, target in rows], dtype=np.float64)
    by_role: dict[str, dict[str, object]] = {}
    for role in BIDDING_ROLE_CLASSES:
        indices = [
            index
            for index, (_row_index, sample, _target) in enumerate(rows)
            if canonical_bidding_role(sample.terminal_role) == role
        ]
        if not indices:
            by_role[role] = {"sampleCount": 0}
            continue
        by_role[role] = {
            "sampleCount": len(indices),
            **_regression_metrics(predictions[indices], targets[indices]),
        }
    by_suit = _grouped_value_metrics(
        rows=rows,
        predictions=predictions,
        targets=targets,
        key_fn=lambda sample: sample.forced_suit if sample.forced_suit is not None else "PASS",
    )
    by_target = _grouped_value_metrics(
        rows=rows,
        predictions=predictions,
        targets=targets,
        key_fn=lambda sample: (
            str(sample.forced_target_point_cards)
            if sample.forced_target_point_cards is not None
            else "PASS"
        ),
    )
    by_hand_strength = _grouped_value_metrics(
        rows=rows,
        predictions=predictions,
        targets=targets,
        key_fn=lambda sample: _hand_strength_bin(sample.strongest_suit_score),
    )
    return {
        "sampleCount": len(rows),
        "maskedSampleCount": len(samples) - len(rows),
        "lossType": value_loss_type,
        "valueTarget": value_target,
        "overall": _regression_metrics(predictions, targets),
        "perRole": by_role,
        "perSuit": by_suit,
        "perTarget": by_target,
        "perHandStrengthBin": by_hand_strength,
    }


def _role_baselines(
    *,
    train_samples: tuple[BiddingQRawSample, ...],
    validation_samples: tuple[BiddingQRawSample, ...],
    value_target: Literal["terminal-reward", "team-point-cards"],
) -> dict[str, object]:
    global_counts = Counter(
        canonical_bidding_role(sample.terminal_role) for sample in train_samples
    )
    global_distribution = _smoothed_role_distribution(global_counts)
    majority_role = max(BIDDING_ROLE_CLASSES, key=lambda role: (global_counts[role], role))
    action_counts = {index: _empty_role_counts() for index in range(BIDDING_ACTION_COUNT)}
    for sample in train_samples:
        action_counts[sample.forced_action_index][canonical_bidding_role(sample.terminal_role)] += 1
    action_distributions = {
        action: _smoothed_role_distribution(counts) for action, counts in action_counts.items()
    }
    truth = [canonical_bidding_role(sample.terminal_role) for sample in validation_samples]
    majority_predictions = [majority_role] * len(validation_samples)
    action_predictions = [
        max(
            BIDDING_ROLE_CLASSES,
            key=lambda role: (
                action_counts[sample.forced_action_index][role],
                global_counts[role],
                role,
            ),
        )
        for sample in validation_samples
    ]
    majority_probabilities = [global_distribution[role] for role in truth]
    action_probabilities = [
        action_distributions[sample.forced_action_index][role]
        for sample, role in zip(validation_samples, truth, strict=True)
    ]
    return {
        "majorityRole": {
            "role": majority_role,
            "accuracy": _accuracy(majority_predictions, truth),
            "crossEntropy": _cross_entropy_from_probabilities(majority_probabilities),
        },
        "actionIndexRoleFrequency": {
            "accuracy": _accuracy(action_predictions, truth),
            "crossEntropy": _cross_entropy_from_probabilities(action_probabilities),
        },
        "value": _value_baselines(
            train_samples=train_samples,
            validation_samples=validation_samples,
            value_target=value_target,
        ),
    }


def _state_conditioned_role_signal(
    role_report: dict[str, object],
    baselines: dict[str, object],
) -> dict[str, object]:
    action_baseline = cast(dict[str, object], baselines["actionIndexRoleFrequency"])
    model_accuracy = cast(float, role_report["accuracy"])
    baseline_accuracy = cast(float, action_baseline["accuracy"])
    model_ce = cast(float, role_report["crossEntropy"])
    baseline_ce = cast(float, action_baseline["crossEntropy"])
    return {
        "beatsActionIndexAccuracyBaseline": model_accuracy > baseline_accuracy,
        "beatsActionIndexCrossEntropyBaseline": model_ce < baseline_ce,
        "accuracyDeltaVsActionIndexBaseline": model_accuracy - baseline_accuracy,
        "crossEntropyDeltaVsActionIndexBaseline": model_ce - baseline_ce,
    }


def _raw_coverage_diagnostics(samples: tuple[BiddingQRawSample, ...]) -> dict[str, object]:
    pass_count = sum(1 for sample in samples if sample.forced_action_index == 0)
    bid_count = len(samples) - pass_count
    suit_counts = {suit: 0 for suit in BIDDING_Q_SUITS}
    target_counts = {str(target): 0 for target in BIDDING_Q_TARGETS}
    action_counts = {str(index): 0 for index in range(BIDDING_ACTION_COUNT)}
    strongest_counts = {suit: 0 for suit in BIDDING_Q_SUITS}
    for sample in samples:
        action_counts[str(sample.forced_action_index)] += 1
        strongest_counts[sample.strongest_suit] += 1
        if sample.forced_suit is not None:
            suit_counts[sample.forced_suit] += 1
        if sample.forced_target_point_cards is not None:
            target_counts[str(sample.forced_target_point_cards)] += 1
    return {
        "sampleCount": len(samples),
        "passCount": pass_count,
        "bidCount": bid_count,
        "suitCounts": suit_counts,
        "targetCounts": target_counts,
        "actionIndexCounts": action_counts,
        "strongestSuitCounts": strongest_counts,
    }


def _team_point_card_action_signal(
    *,
    samples: tuple[BiddingQRawSample, ...],
    role_probabilities: np.ndarray,
    role_values: np.ndarray,
) -> dict[str, object]:
    q_rows = np.sum(role_probabilities * role_values, axis=2)
    state_q: dict[str, np.ndarray] = {}
    action_semantics: dict[tuple[str, int], BiddingQRawSample] = {}
    target_values: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for index, sample in enumerate(samples):
        _target, mask = value_target_for_sample(sample, "team-point-cards")
        if not mask:
            continue
        if sample.candidate_team_point_cards is None:
            continue
        state_q.setdefault(sample.state_key, q_rows[index])
        action_semantics[(sample.state_key, sample.forced_action_index)] = sample
        target_values[sample.state_key][sample.forced_action_index].append(
            float(sample.candidate_team_point_cards)
        )

    best_hits = 0
    top3_hits = 0
    states_with_ranking = 0
    pair_correct = 0
    pair_total = 0
    predicted_action_counts = {str(index): 0 for index in range(BIDDING_ACTION_COUNT)}
    predicted_suit_counts = {suit: 0 for suit in BIDDING_Q_SUITS}
    predicted_target_counts = {str(target): 0 for target in BIDDING_Q_TARGETS}
    strongest_match = 0
    strongest_match_total = 0
    for state_key, action_rewards in target_values.items():
        if len(action_rewards) < 2 or state_key not in state_q:
            continue
        action_means = {
            action: float(np.mean(np.asarray(rewards, dtype=np.float64)))
            for action, rewards in action_rewards.items()
        }
        covered_actions = sorted(action_means)
        predicted_scores = state_q[state_key]
        empirical_best_value = max(action_means.values())
        empirical_best_actions = {
            action for action, value in action_means.items() if value == empirical_best_value
        }
        predicted_order = sorted(
            covered_actions,
            key=lambda action: (float(predicted_scores[action]), -action),
            reverse=True,
        )
        predicted_best = predicted_order[0]
        if predicted_best in empirical_best_actions:
            best_hits += 1
        if empirical_best_actions.intersection(predicted_order[:3]):
            top3_hits += 1
        states_with_ranking += 1
        predicted_action_counts[str(predicted_best)] += 1
        semantic = action_semantics[(state_key, predicted_best)]
        if semantic.forced_suit is not None:
            predicted_suit_counts[semantic.forced_suit] += 1
            strongest_match_total += 1
            if semantic.forced_suit == semantic.strongest_suit:
                strongest_match += 1
        if semantic.forced_target_point_cards is not None:
            predicted_target_counts[str(semantic.forced_target_point_cards)] += 1
        for left_index, left_action in enumerate(covered_actions):
            for right_action in covered_actions[left_index + 1 :]:
                target_delta = action_means[left_action] - action_means[right_action]
                if target_delta == 0.0:
                    continue
                predicted_delta = float(
                    predicted_scores[left_action] - predicted_scores[right_action]
                )
                if predicted_delta == 0.0:
                    continue
                pair_total += 1
                if (target_delta > 0.0) == (predicted_delta > 0.0):
                    pair_correct += 1
    return {
        "stateCount": states_with_ranking,
        "bestActionHitRate": _safe_rate(best_hits, states_with_ranking),
        "top3HitRate": _safe_rate(top3_hits, states_with_ranking),
        "pairwiseRankingAccuracy": _safe_rate(pair_correct, pair_total),
        "pairwiseComparisonCount": pair_total,
        "predictedActionCounts": predicted_action_counts,
        "predictedSuitCounts": predicted_suit_counts,
        "predictedTargetCounts": predicted_target_counts,
        "strongestSuitMatchRate": _safe_rate(strongest_match, strongest_match_total),
        "sameTargetSuit": _same_target_suit_signal(samples),
        "sameSuitTarget": _target_signal(samples),
    }


def _same_target_suit_signal(samples: tuple[BiddingQRawSample, ...]) -> dict[str, object]:
    by_state_target: dict[tuple[str, int], dict[str, list[BiddingQRawSample]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for sample in samples:
        if (
            not sample.team_point_cards_regression_mask
            or sample.candidate_team_point_cards is None
            or sample.forced_suit is None
            or sample.forced_target_point_cards is None
        ):
            continue
        by_state_target[(sample.state_key, sample.forced_target_point_cards)][
            sample.forced_suit
        ].append(sample)
    gaps: list[float] = []
    exact_ties = 0
    strongest_best = 0
    strongest_best_total = 0
    best_suit_counts = {suit: 0 for suit in BIDDING_Q_SUITS}
    margin_by_suit: dict[str, list[float]] = {suit: [] for suit in BIDDING_Q_SUITS}
    target13: dict[str, object] = {"comparisonCount": 0}
    target13_gaps: list[float] = []
    for (_state_key, target), by_suit in by_state_target.items():
        if any(suit not in by_suit for suit in BIDDING_Q_SUITS):
            continue
        suit_means = {
            suit: float(
                np.mean(
                    np.asarray(
                        [sample.candidate_team_point_cards for sample in by_suit[suit]],
                        dtype=np.float64,
                    )
                )
            )
            for suit in BIDDING_Q_SUITS
        }
        best_value = max(suit_means.values())
        worst_value = min(suit_means.values())
        gap = best_value - worst_value
        gaps.append(gap)
        if target == 13:
            target13_gaps.append(gap)
        if gap == 0.0:
            exact_ties += 1
        best_suits = {suit for suit, value in suit_means.items() if value == best_value}
        for suit in best_suits:
            best_suit_counts[suit] += 1
        strongest = by_suit[BIDDING_Q_SUITS[0]][0].strongest_suit
        strongest_best_total += 1
        if strongest in best_suits:
            strongest_best += 1
        for suit, suit_samples in by_suit.items():
            margins = [
                float(sample.contract_margin)
                for sample in suit_samples
                if sample.contract_margin is not None
            ]
            margin_by_suit[suit].extend(margins)
    if target13_gaps:
        target13 = {
            "comparisonCount": len(target13_gaps),
            "bestWorstGap": _numeric_summary(target13_gaps),
        }
    return {
        "comparisonCount": len(gaps),
        "bestWorstGap": _numeric_summary(gaps),
        "exactTieCount": exact_ties,
        "exactTieRate": _safe_rate(exact_ties, len(gaps)),
        "strongestSuitTeacherBestRate": _safe_rate(strongest_best, strongest_best_total),
        "teacherBestSuitCounts": best_suit_counts,
        "contractMarginBySuit": {
            suit: _numeric_summary(values) for suit, values in margin_by_suit.items()
        },
        "target13": target13,
    }


def _target_signal(samples: tuple[BiddingQRawSample, ...]) -> dict[str, object]:
    by_state_suit: dict[tuple[str, str], dict[int, list[BiddingQRawSample]]] = defaultdict(
        lambda: defaultdict(list)
    )
    target_values: dict[str, list[float]] = {str(target): [] for target in BIDDING_Q_TARGETS}
    margin_by_target: dict[str, list[float]] = {str(target): [] for target in BIDDING_Q_TARGETS}
    for sample in samples:
        if (
            not sample.team_point_cards_regression_mask
            or sample.candidate_team_point_cards is None
            or sample.forced_suit is None
            or sample.forced_target_point_cards is None
        ):
            continue
        by_state_suit[(sample.state_key, sample.forced_suit)][
            sample.forced_target_point_cards
        ].append(sample)
        target_values[str(sample.forced_target_point_cards)].append(
            float(sample.candidate_team_point_cards)
        )
        if sample.contract_margin is not None:
            margin_by_target[str(sample.forced_target_point_cards)].append(
                float(sample.contract_margin)
            )
    adjacent_diffs: list[float] = []
    best_target_counts = {str(target): 0 for target in BIDDING_Q_TARGETS}
    comparison_count = 0
    for by_target in by_state_suit.values():
        if len(by_target) < 2:
            continue
        means = {
            target: float(
                np.mean(
                    np.asarray(
                        [sample.candidate_team_point_cards for sample in target_samples],
                        dtype=np.float64,
                    )
                )
            )
            for target, target_samples in by_target.items()
        }
        sorted_targets = sorted(means)
        comparison_count += 1
        best_value = max(means.values())
        for target, value in means.items():
            if value == best_value:
                best_target_counts[str(target)] += 1
        for left, right in zip(sorted_targets, sorted_targets[1:], strict=False):
            if right == left + 1:
                adjacent_diffs.append(means[right] - means[left])
    return {
        "comparisonCount": comparison_count,
        "adjacentTargetDifference": _numeric_summary(adjacent_diffs),
        "teacherBestTargetCounts": best_target_counts,
        "targetMeanTeamPointCards": {
            target: _numeric_summary(values) for target, values in target_values.items()
        },
        "contractMarginByTarget": {
            target: _numeric_summary(values) for target, values in margin_by_target.items()
        },
    }


def _grouped_value_metrics(
    *,
    rows: list[tuple[int, BiddingQRawSample, float]],
    predictions: np.ndarray,
    targets: np.ndarray,
    key_fn: object,
) -> dict[str, dict[str, object]]:
    typed_key_fn = cast(Callable[[BiddingQRawSample], str], key_fn)
    indices_by_key: dict[str, list[int]] = defaultdict(list)
    for index, (_row_index, sample, _target) in enumerate(rows):
        indices_by_key[typed_key_fn(sample)].append(index)
    return {
        key: {
            "sampleCount": len(indices),
            **_regression_metrics(predictions[indices], targets[indices]),
        }
        for key, indices in sorted(indices_by_key.items())
    }


def _value_baselines(
    *,
    train_samples: tuple[BiddingQRawSample, ...],
    validation_samples: tuple[BiddingQRawSample, ...],
    value_target: Literal["terminal-reward", "team-point-cards"],
) -> dict[str, object]:
    train_rows = [
        (sample, target)
        for sample in train_samples
        for target, mask in (value_target_for_sample(sample, value_target),)
        if mask
    ]
    validation_rows = [
        (sample, target)
        for sample in validation_samples
        for target, mask in (value_target_for_sample(sample, value_target),)
        if mask
    ]
    if not train_rows or not validation_rows:
        return {"sampleCount": len(validation_rows), "skipped": True}
    global_mean = float(np.mean(np.asarray([target for _sample, target in train_rows])))
    role_mean = _mean_lookup(
        train_rows,
        key_fn=lambda sample: canonical_bidding_role(sample.terminal_role),
        fallback=global_mean,
    )
    action_mean = _mean_lookup(
        train_rows,
        key_fn=lambda sample: str(sample.forced_action_index),
        fallback=global_mean,
    )
    role_action_mean = _mean_lookup(
        train_rows,
        key_fn=lambda sample: (
            f"{canonical_bidding_role(sample.terminal_role)}:{sample.forced_action_index}"
        ),
        fallback=global_mean,
    )
    targets = np.asarray([target for _sample, target in validation_rows], dtype=np.float64)

    def evaluate_baseline(name: str, predictions: list[float]) -> dict[str, object]:
        return {
            "name": name,
            "sampleCount": len(predictions),
            **_regression_metrics(np.asarray(predictions, dtype=np.float64), targets),
        }

    return {
        "sampleCount": len(validation_rows),
        "valueTarget": value_target,
        "globalMean": evaluate_baseline(
            "globalMean",
            [global_mean for _sample, _target in validation_rows],
        ),
        "roleMean": evaluate_baseline(
            "roleMean",
            [
                role_mean[canonical_bidding_role(sample.terminal_role)]
                for sample, _target in validation_rows
            ],
        ),
        "actionIndexMean": evaluate_baseline(
            "actionIndexMean",
            [action_mean[str(sample.forced_action_index)] for sample, _target in validation_rows],
        ),
        "roleActionIndexMean": evaluate_baseline(
            "roleActionIndexMean",
            [
                role_action_mean[
                    f"{canonical_bidding_role(sample.terminal_role)}:{sample.forced_action_index}"
                ]
                for sample, _target in validation_rows
            ],
        ),
    }


def _mean_lookup(
    rows: list[tuple[BiddingQRawSample, float]],
    *,
    key_fn: object,
    fallback: float,
) -> defaultdict[str, float]:
    typed_key_fn = cast(Callable[[BiddingQRawSample], str], key_fn)
    values: dict[str, list[float]] = defaultdict(list)
    for sample, target in rows:
        values[typed_key_fn(sample)].append(target)
    means: defaultdict[str, float] = defaultdict(lambda: fallback)
    for key, key_values in values.items():
        means[key] = float(np.mean(np.asarray(key_values, dtype=np.float64)))
    return means


def _checkpoint_dict(
    *,
    result: BiddingRoleQTrainResult,
    dataset: BiddingQDataset,
) -> dict[str, object]:
    return {
        "checkpointSchemaVersion": BIDDING_ROLE_Q_CHECKPOINT_SCHEMA_VERSION,
        "modelType": BIDDING_ROLE_Q_MODEL_TYPE,
        "architectureId": BIDDING_ROLE_Q_ARCHITECTURE_ID,
        "selectionSemantics": BIDDING_ROLE_Q_SELECTION_SEMANTICS,
        "roleClasses": list(BIDDING_ROLE_CLASSES),
        "modelState": cpu_state_dict(result.model),
        "modelConfig": result.model.config.to_dict(),
        "trainingConfig": result.config.to_dict(),
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
        "split": _split_metadata(result.split, result),
        "metrics": {
            "epochs": result.epoch_reports,
            "train": result.train_report,
            "validation": result.validation_report,
        },
        "cardIdsSha256": calculate_card_ids_sha256(),
    }


def _metadata_dict(
    *,
    result: BiddingRoleQTrainResult,
    dataset: BiddingQDataset,
    checkpoint_path: Path,
) -> dict[str, object]:
    return {
        "metadataSchemaVersion": 1,
        "modelType": BIDDING_ROLE_Q_MODEL_TYPE,
        "policyType": BIDDING_ROLE_Q_MODEL_TYPE,
        "selectionSemantics": BIDDING_ROLE_Q_SELECTION_SEMANTICS,
        "architectureId": BIDDING_ROLE_Q_ARCHITECTURE_ID,
        "roleClasses": list(BIDDING_ROLE_CLASSES),
        "checkpointFileName": checkpoint_path.name,
        "modelConfig": result.model.config.to_dict(),
        "trainingConfig": result.config.to_dict(),
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
        "split": _split_metadata(result.split, result),
        "device": result.device.to_metadata(),
        "datasetDiagnostics": result.dataset_diagnostics,
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


def _split_metadata(
    split: BiddingQSplit,
    result: BiddingRoleQTrainResult,
) -> dict[str, object]:
    return {
        "seed": split.seed,
        "trainRatio": split.train_ratio,
        "validationRatio": split.validation_ratio,
        "trainStateCount": len(split.train_state_keys),
        "validationStateCount": len(split.validation_state_keys),
        "trainAggregatedSampleCount": len(split.train_examples),
        "validationAggregatedSampleCount": len(split.validation_examples),
        "trainRawSampleCount": len(result.train_samples),
        "validationRawSampleCount": len(result.validation_samples),
    }


def _validate_train_config(config: BiddingRoleQTrainConfig) -> None:
    if config.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if config.value_loss_type not in SUPPORTED_BIDDING_ROLE_Q_VALUE_LOSSES:
        raise ValueError(
            f"value_loss_type must be one of {SUPPORTED_BIDDING_ROLE_Q_VALUE_LOSSES}."
        )
    if config.value_target not in SUPPORTED_BIDDING_ROLE_Q_VALUE_TARGETS:
        raise ValueError(
            f"value_target must be one of {SUPPORTED_BIDDING_ROLE_Q_VALUE_TARGETS}."
        )
    if config.value_loss_coefficient < 0.0 or config.role_loss_coefficient < 0.0:
        raise ValueError("loss coefficients must be non-negative.")


def _configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | None]:
    diff = prediction - target
    mse = float(np.mean(diff**2))
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": math.sqrt(mse),
        "mse": mse,
        "predictedMean": float(prediction.mean()),
        "predictedStd": float(prediction.std()),
        "targetMean": float(target.mean()),
        "targetStd": float(target.std()),
        "pearsonCorrelation": _pearson(prediction, target),
    }


def _empty_role_counts() -> dict[str, int]:
    return {role: 0 for role in BIDDING_ROLE_CLASSES}


def _counts_with_rates(counts: Counter[str] | dict[str, int]) -> dict[str, object]:
    normalized_counts = {role: int(counts.get(role, 0)) for role in BIDDING_ROLE_CLASSES}
    total = sum(normalized_counts.values())
    return {
        "count": normalized_counts,
        "rate": {
            role: _safe_rate(count, total) for role, count in normalized_counts.items()
        },
        "total": total,
    }


def _confusion_to_dict(confusion: np.ndarray) -> dict[str, dict[str, int]]:
    return {
        truth: {
            prediction: int(confusion[truth_index, prediction_index])
            for prediction_index, prediction in enumerate(BIDDING_ROLE_CLASSES)
        }
        for truth_index, truth in enumerate(BIDDING_ROLE_CLASSES)
    }


def _confidence_calibration(
    selected_probabilities: np.ndarray,
    true_roles: np.ndarray,
) -> list[dict[str, float | int | None]]:
    predictions = np.argmax(selected_probabilities, axis=1)
    confidence = np.max(selected_probabilities, axis=1)
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0000001)]
    report: list[dict[str, float | int | None]] = []
    for lower, upper in bins:
        mask = (confidence >= lower) & (confidence < upper)
        count = int(mask.sum())
        if count == 0:
            report.append(
                {
                    "lower": lower,
                    "upper": min(upper, 1.0),
                    "sampleCount": 0,
                    "accuracy": None,
                    "meanConfidence": None,
                }
            )
            continue
        report.append(
            {
                "lower": lower,
                "upper": min(upper, 1.0),
                "sampleCount": count,
                "accuracy": float(np.mean(predictions[mask] == true_roles[mask])),
                "meanConfidence": float(np.mean(confidence[mask])),
            }
        )
    return report


def _smoothed_role_distribution(counts: Counter[str] | dict[str, int]) -> dict[str, float]:
    total = sum(int(counts.get(role, 0)) for role in BIDDING_ROLE_CLASSES)
    denominator = total + BIDDING_ROLE_COUNT
    return {
        role: (int(counts.get(role, 0)) + 1.0) / denominator
        for role in BIDDING_ROLE_CLASSES
    }


def _role_distribution_for_samples(
    samples: Iterable[BiddingQRawSample],
) -> dict[str, float] | None:
    counts = _empty_role_counts()
    total = 0
    for sample in samples:
        counts[canonical_bidding_role(sample.terminal_role)] += 1
        total += 1
    if total == 0:
        return None
    return {role: counts[role] / total for role in BIDDING_ROLE_CLASSES}


def _lowest_legal_bid_by_suit(
    samples: Iterable[BiddingQRawSample],
) -> dict[str, list[BiddingQRawSample]]:
    result: dict[str, list[BiddingQRawSample]] = {}
    by_suit: dict[str, list[BiddingQRawSample]] = defaultdict(list)
    for sample in samples:
        if sample.forced_suit is not None:
            by_suit[sample.forced_suit].append(sample)
    for suit, suit_samples in by_suit.items():
        minimum_target = min(
            sample.forced_target_point_cards
            for sample in suit_samples
            if sample.forced_target_point_cards is not None
        )
        result[suit] = [
            sample
            for sample in suit_samples
            if sample.forced_target_point_cards == minimum_target
        ]
    return result


def _total_variation_distance(
    left: dict[str, float],
    right: dict[str, float],
) -> float:
    return 0.5 * sum(abs(left[role] - right[role]) for role in BIDDING_ROLE_CLASSES)


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _mean(values: Iterable[float]) -> float | None:
    value_tuple = tuple(values)
    if not value_tuple:
        return None
    return float(np.mean(np.asarray(value_tuple, dtype=np.float64)))


def _numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    value_array = np.asarray(tuple(values), dtype=np.float64)
    if value_array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "median": None,
            "p90": None,
            "max": None,
        }
    return {
        "count": int(value_array.size),
        "mean": float(value_array.mean()),
        "std": float(value_array.std()),
        "min": float(value_array.min()),
        "median": float(np.median(value_array)),
        "p90": float(np.percentile(value_array, 90)),
        "max": float(value_array.max()),
    }


def _hand_strength_bin(score: float) -> str:
    if score < 200:
        return "<200"
    if score < 250:
        return "200-249"
    if score < 300:
        return "250-299"
    if score < 350:
        return "300-349"
    return ">=350"


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _accuracy(predictions: list[str], truth: list[str]) -> float:
    if len(predictions) != len(truth):
        raise ValueError("prediction/truth length mismatch.")
    return sum(pred == actual for pred, actual in zip(predictions, truth, strict=True)) / len(truth)


def _cross_entropy_from_probabilities(probabilities: list[float]) -> float:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
    return float(-np.mean(np.log(clipped)))


def _check_onnx_parity(
    *,
    model_cpu: BiddingRoleQModel,
    onnx_path: Path,
    sample_model_input: np.ndarray,
) -> dict[str, object]:
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]
    except ImportError as error:
        raise BiddingRoleQCheckpointError("onnxruntime is required for ONNX parity.") from error
    sample = torch.as_tensor(sample_model_input.reshape(1, -1), dtype=torch.float32)
    with torch.no_grad():
        pytorch_logits, pytorch_values = model_cpu(sample)
        pytorch_q = compose_role_q(pytorch_logits, pytorch_values).detach().cpu().numpy()
        pytorch_logits_np = pytorch_logits.detach().cpu().numpy()
        pytorch_values_np = pytorch_values.detach().cpu().numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits, onnx_values = session.run(
        [BIDDING_ROLE_Q_ONNX_LOGITS_OUTPUT_NAME, BIDDING_ROLE_Q_ONNX_VALUES_OUTPUT_NAME],
        {BIDDING_ROLE_Q_ONNX_INPUT_NAME: sample.numpy()},
    )
    onnx_q = compose_role_q(
        torch.as_tensor(onnx_logits, dtype=torch.float32),
        torch.as_tensor(onnx_values, dtype=torch.float32),
    ).detach().cpu().numpy()
    max_logits_diff = float(np.max(np.abs(pytorch_logits_np - onnx_logits)))
    max_values_diff = float(np.max(np.abs(pytorch_values_np - onnx_values)))
    max_q_diff = float(np.max(np.abs(pytorch_q - onnx_q)))
    max_abs_diff = max(max_logits_diff, max_values_diff, max_q_diff)
    return {
        "maxAbsDiff": max_abs_diff,
        "maxAbsRoleLogitsDiff": max_logits_diff,
        "maxAbsRoleValuesDiff": max_values_diff,
        "maxAbsComposedQDiff": max_q_diff,
        "pytorchRoleLogitsShape": list(pytorch_logits_np.shape),
        "onnxRoleLogitsShape": list(onnx_logits.shape),
        "pytorchRoleValuesShape": list(pytorch_values_np.shape),
        "onnxRoleValuesShape": list(onnx_values.shape),
        "withinTolerance": max_abs_diff <= 1e-5,
    }
