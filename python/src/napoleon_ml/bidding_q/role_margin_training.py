"""Training for Issue #400 Citizen/Adjutant heteroscedastic margin models."""

from __future__ import annotations

import copy
import json
import math
import pickle
import random
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import torch
from torch import Tensor, optim
from torch.utils.data import DataLoader, Dataset

from napoleon_ml.policy.device import (
    RequestedTorchDevice,
    ResolvedTorchDevice,
    cpu_state_dict,
    resolve_torch_device,
)

from .dataset import BiddingQDataset, BiddingQRawSample
from .margin_training import (
    gaussian_margin_nll,
    margin_evaluation_report,
    predict_margin_samples,
)
from .multi_head_training import (
    BiddingMultiHeadQSplit,
    BiddingMultiHeadQTrainConfig,
    Standardization,
    _contract_margin,
    _contract_mask,
    create_multi_head_split,
)
from .role_margin_model import (
    BiddingRoleMarginHeteroscedasticModel,
    BiddingRoleMarginHeteroscedasticModelConfig,
    BiddingRoleMarginRole,
    create_seeded_bidding_role_margin_model,
    role_margin_architecture_id,
)
from .role_training import canonical_bidding_role

BIDDING_ROLE_MARGIN_HETEROSCEDASTIC_CHECKPOINT_SCHEMA_VERSION = 1
BIDDING_ROLE_MARGIN_HETEROSCEDASTIC_MODEL_TYPE = "bidding-role-margin-heteroscedastic"


class BiddingRoleMarginCheckpointError(ValueError):
    """Raised when a role margin checkpoint is incompatible."""


@dataclass(frozen=True)
class BiddingRoleMarginTrainConfig:
    role: BiddingRoleMarginRole
    seed: int = 400
    epochs: int = 80
    batch_size: int = 128
    learning_rate: float = 1e-3
    hidden_dims: tuple[int, ...] = (512, 512, 256, 256)
    dropout: float = 0.0
    train_state_count: int | None = None
    validation_state_keys_path: str | None = None
    train_ratio: float = 0.75
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
        data["algorithm"] = "role-contract-margin-heteroscedastic-gaussian-nll-v1"
        data["teacher"] = "napoleon-side-contract-margin"
        return data


@dataclass(frozen=True)
class BiddingRoleMarginTrainResult:
    model: BiddingRoleMarginHeteroscedasticModel
    split: BiddingMultiHeadQSplit
    config: BiddingRoleMarginTrainConfig
    device: ResolvedTorchDevice
    target_standardization: Standardization
    epoch_reports: list[dict[str, object]]
    best_epoch: int
    best_monitor_value: float
    train_report: dict[str, object]
    validation_report: dict[str, object]
    coverage: dict[str, object]


class BiddingRoleMarginBatch(NamedTuple):
    model_input: Tensor
    action_index: Tensor
    margin: Tensor
    role_contract_mask: Tensor


class _BiddingRoleMarginDataset(Dataset[BiddingRoleMarginBatch]):
    def __init__(
        self,
        samples: Iterable[BiddingQRawSample],
        *,
        role: BiddingRoleMarginRole,
        standardization: Standardization,
    ) -> None:
        self.samples = tuple(samples)
        self.role = role
        self.standardization = standardization

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> BiddingRoleMarginBatch:
        sample = self.samples[index]
        has_teacher = role_margin_teacher(sample, self.role) is not None
        margin = role_margin_teacher(sample, self.role) if has_teacher else 0.0
        return BiddingRoleMarginBatch(
            model_input=torch.as_tensor(sample.model_input, dtype=torch.float32),
            action_index=torch.tensor(sample.forced_action_index, dtype=torch.long),
            margin=self.standardization.encode_tensor(torch.tensor(margin, dtype=torch.float32)),
            role_contract_mask=torch.tensor(has_teacher, dtype=torch.bool),
        )


def role_margin_teacher(sample: BiddingQRawSample, role: BiddingRoleMarginRole) -> float | None:
    if canonical_bidding_role(sample.final_role) != role:
        return None
    if not _contract_mask(sample):
        return None
    return float(_contract_margin(sample))


def role_margin_coverage(samples: Iterable[BiddingQRawSample]) -> dict[str, object]:
    sample_tuple = tuple(samples)
    by_role = Counter(canonical_bidding_role(sample.final_role) for sample in sample_tuple)
    result: dict[str, object] = {
        "rawSampleCount": len(sample_tuple),
        "stateCount": len({sample.state_key for sample in sample_tuple}),
        "roleSampleCounts": dict(sorted(by_role.items())),
    }
    for role in ("citizen", "adjutant"):
        role_samples = tuple(
            sample for sample in sample_tuple if role_margin_teacher(sample, role) is not None
        )
        result[role] = {
            "teacherSampleCount": len(role_samples),
            "teacherStateCount": len({sample.state_key for sample in role_samples}),
            "actionCounts": dict(
                sorted(Counter(str(sample.forced_action_index) for sample in role_samples).items())
            ),
        }
    return result


def train_bidding_role_margin_model(
    dataset: BiddingQDataset,
    config: BiddingRoleMarginTrainConfig,
) -> BiddingRoleMarginTrainResult:
    _validate_train_config(config)
    _configure_reproducibility(config.seed)
    split = create_role_margin_split(dataset, config)
    train_teacher_samples = _role_teacher_samples(split.train_samples, config.role)
    validation_teacher_samples = _role_teacher_samples(split.validation_samples, config.role)
    if not train_teacher_samples:
        raise ValueError(f"train split has no {config.role} contract-margin teacher samples.")
    if not validation_teacher_samples:
        raise ValueError(f"validation split has no {config.role} contract-margin teacher samples.")
    device = resolve_torch_device(config.device, flag_name="--device")
    standardization = _standardization(
        [float(_contract_margin(sample)) for sample in train_teacher_samples],
        enabled=config.target_standardization,
    )
    model = create_seeded_bidding_role_margin_model(
        BiddingRoleMarginHeteroscedasticModelConfig(
            role=config.role,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
            log_variance_min=config.log_variance_min,
            log_variance_max=config.log_variance_max,
        ),
        seed=config.seed,
    ).to(device.torch_device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_loader = _create_loader(
        split.train_samples,
        role=config.role,
        standardization=standardization,
        batch_size=config.batch_size,
        seed=config.seed,
        shuffle=True,
    )
    validation_loader = _create_loader(
        split.validation_samples,
        role=config.role,
        standardization=standardization,
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
        train_nll = _run_nll_epoch(
            model=model, dataloader=train_loader, device=device, optimizer=optimizer
        )
        validation_nll = _run_nll_epoch(
            model=model, dataloader=validation_loader, device=device, optimizer=None
        )
        improved = validation_nll < best_monitor - config.min_delta
        if improved:
            best_state = copy.deepcopy(cpu_state_dict(model))
            best_epoch = epoch
            best_monitor = validation_nll
            misses = 0
        else:
            misses += 1
        epoch_reports.append(
            {
                "epoch": epoch,
                "trainNll": train_nll,
                "validationNll": validation_nll,
                "best": improved,
            }
        )
        if misses >= config.patience:
            break
    if best_state is None:
        raise AssertionError("best checkpoint state was not captured.")
    model.load_state_dict(best_state)
    train_report = evaluate_bidding_role_margin_model(
        model,
        split.train_samples,
        train_samples=split.train_samples,
        role=config.role,
        device=device,
        standardization=standardization,
    )
    validation_report = evaluate_bidding_role_margin_model(
        model,
        split.validation_samples,
        train_samples=split.train_samples,
        role=config.role,
        device=device,
        standardization=standardization,
    )
    return BiddingRoleMarginTrainResult(
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
        coverage=role_margin_coverage(dataset.raw_samples),
    )


def create_role_margin_split(
    dataset: BiddingQDataset, config: BiddingRoleMarginTrainConfig
) -> BiddingMultiHeadQSplit:
    return create_multi_head_split(
        dataset,
        BiddingMultiHeadQTrainConfig(
            seed=config.seed,
            train_state_count=config.train_state_count,
            validation_state_keys_path=config.validation_state_keys_path,
            train_ratio=config.train_ratio,
        ),
    )


def evaluate_bidding_role_margin_model(
    model: BiddingRoleMarginHeteroscedasticModel,
    samples: Iterable[BiddingQRawSample],
    *,
    train_samples: Iterable[BiddingQRawSample],
    role: BiddingRoleMarginRole,
    device: ResolvedTorchDevice,
    standardization: Standardization,
) -> dict[str, object]:
    sample_tuple = tuple(samples)
    train_tuple = tuple(train_samples)
    role_samples = _role_teacher_samples(sample_tuple, role)
    role_train_samples = _role_teacher_samples(train_tuple, role)
    predictions = predict_role_margin_samples(
        model,
        role_samples,
        device=device,
        standardization=standardization,
    )
    report = margin_evaluation_report(
        role_samples,
        train_samples=role_train_samples,
        mean=predictions["mean"],
        sigma=predictions["sigma"],
    )
    report["role"] = role
    report["roleCoverage"] = {
        "rawSampleCount": len(sample_tuple),
        "teacherSampleCount": len(role_samples),
        "teacherStateCount": len({sample.state_key for sample in role_samples}),
    }
    return report


def predict_role_margin_samples(
    model: BiddingRoleMarginHeteroscedasticModel,
    samples: tuple[BiddingQRawSample, ...],
    *,
    device: ResolvedTorchDevice,
    standardization: Standardization,
) -> dict[str, np.ndarray]:
    return predict_margin_samples(
        cast(Any, model), samples, device=device, standardization=standardization
    )


def save_bidding_role_margin_artifact(
    output_directory: Path | str,
    *,
    result: BiddingRoleMarginTrainResult,
    dataset: BiddingQDataset,
) -> dict[str, str]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    metadata_path = output / "metadata.json"
    report_path = output / "report.json"
    checkpoint = _checkpoint_dict(result=result, dataset=dataset)
    torch.save(checkpoint, checkpoint_path)
    metadata = dict(checkpoint)
    metadata.pop("modelState", None)
    metadata["checkpointPath"] = str(checkpoint_path)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "coverage": result.coverage,
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


def load_bidding_role_margin_checkpoint(
    path: Path | str,
) -> tuple[BiddingRoleMarginHeteroscedasticModel, dict[str, Any]]:
    try:
        raw = torch.load(Path(path), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, pickle.UnpicklingError) as error:
        raise BiddingRoleMarginCheckpointError(
            f"checkpoint cannot be loaded: {error}"
        ) from error
    if not isinstance(raw, dict):
        raise BiddingRoleMarginCheckpointError("checkpoint must be a dictionary.")
    if (
        raw.get("checkpointSchemaVersion")
        != BIDDING_ROLE_MARGIN_HETEROSCEDASTIC_CHECKPOINT_SCHEMA_VERSION
    ):
        raise BiddingRoleMarginCheckpointError("checkpoint schema version mismatch.")
    if raw.get("modelType") != BIDDING_ROLE_MARGIN_HETEROSCEDASTIC_MODEL_TYPE:
        raise BiddingRoleMarginCheckpointError("checkpoint modelType mismatch.")
    config_raw = raw.get("modelConfig")
    if not isinstance(config_raw, dict):
        raise BiddingRoleMarginCheckpointError("checkpoint modelConfig must be an object.")
    model = BiddingRoleMarginHeteroscedasticModel(
        BiddingRoleMarginHeteroscedasticModelConfig.from_dict(config_raw)
    )
    state = raw.get("modelState")
    if not isinstance(state, dict):
        raise BiddingRoleMarginCheckpointError("checkpoint modelState must be a state dict.")
    model.load_state_dict(state)
    return model, raw


def _run_nll_epoch(
    *,
    model: BiddingRoleMarginHeteroscedasticModel,
    dataloader: DataLoader[BiddingRoleMarginBatch],
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
                moved.role_contract_mask,
            )
            if optimizer is not None:
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
            teacher_count = int(moved.role_contract_mask.sum().item())
            total_loss += float(loss.item()) * teacher_count
            total_samples += teacher_count
    if total_samples == 0:
        raise ValueError("split contains no role contract-margin teacher samples.")
    return total_loss / total_samples


def _move_batch(
    batch: BiddingRoleMarginBatch, device: ResolvedTorchDevice
) -> BiddingRoleMarginBatch:
    return BiddingRoleMarginBatch(
        model_input=batch.model_input.to(device=device.torch_device, dtype=torch.float32),
        action_index=batch.action_index.to(device=device.torch_device, dtype=torch.long),
        margin=batch.margin.to(device=device.torch_device, dtype=torch.float32),
        role_contract_mask=batch.role_contract_mask.to(
            device=device.torch_device, dtype=torch.bool
        ),
    )


def _create_loader(
    samples: Iterable[BiddingQRawSample],
    *,
    role: BiddingRoleMarginRole,
    standardization: Standardization,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> DataLoader[BiddingRoleMarginBatch]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        _BiddingRoleMarginDataset(samples, role=role, standardization=standardization),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def _role_teacher_samples(
    samples: Iterable[BiddingQRawSample], role: BiddingRoleMarginRole
) -> tuple[BiddingQRawSample, ...]:
    return tuple(sample for sample in samples if role_margin_teacher(sample, role) is not None)


def _standardization(values: list[float], *, enabled: bool) -> Standardization:
    if not values:
        raise ValueError("cannot standardize empty target values.")
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std <= 0.0:
        std = 1.0
    return Standardization(enabled=enabled, mean=mean, std=std)


def _checkpoint_dict(
    *,
    result: BiddingRoleMarginTrainResult,
    dataset: BiddingQDataset,
) -> dict[str, object]:
    return {
        "checkpointSchemaVersion": BIDDING_ROLE_MARGIN_HETEROSCEDASTIC_CHECKPOINT_SCHEMA_VERSION,
        "modelType": BIDDING_ROLE_MARGIN_HETEROSCEDASTIC_MODEL_TYPE,
        "architectureId": role_margin_architecture_id(result.config.role),
        "role": result.config.role,
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
            "stateKeyLeakage": False,
        },
    }


def _configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _validate_train_config(config: BiddingRoleMarginTrainConfig) -> None:
    if config.role not in ("citizen", "adjutant"):
        raise ValueError("role must be 'citizen' or 'adjutant'.")
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
