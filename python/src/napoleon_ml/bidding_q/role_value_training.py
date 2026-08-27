"""Training and coverage diagnostics for Issue #383 role-specific bidding values."""

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

from .dataset import BiddingQDataset, BiddingQRawSample
from .multi_head_training import (
    BiddingMultiHeadQSplit,
    BiddingMultiHeadQTrainConfig,
    Standardization,
    _regression_metrics,
    _safe_div,
    create_multi_head_split,
    state_key_hash,
)
from .role_training import canonical_bidding_role
from .role_value_model import (
    BiddingRoleValueModel,
    BiddingRoleValueModelConfig,
    BiddingRoleValueRole,
    architecture_id_for_role,
    create_seeded_bidding_role_value_model,
)

BIDDING_ROLE_VALUE_CHECKPOINT_SCHEMA_VERSION = 1
BIDDING_ROLE_VALUE_MODEL_TYPE = "bidding-role-value"
CitizenTeacher = Literal[
    "coalition-side-point-cards",
    "negative-contract-margin",
    "citizen-margin",
    "contract-failure-binary",
]
AdjutantTeacher = Literal[
    "contract-margin",
    "napoleon-side-point-cards",
    "contract-success",
]
BiddingRoleValueTeacher = CitizenTeacher | AdjutantTeacher


class BiddingRoleValueCheckpointError(ValueError):
    """Raised when a role-value checkpoint is incompatible."""


@dataclass(frozen=True)
class BiddingRoleValueTrainConfig:
    role: BiddingRoleValueRole
    teacher: BiddingRoleValueTeacher | None = None
    seed: int = 383
    epochs: int = 80
    batch_size: int = 128
    learning_rate: float = 1e-3
    hidden_dims: tuple[int, ...] = (512, 512, 256, 256)
    dropout: float = 0.0
    train_state_count: int | None = 20000
    validation_state_keys_path: str | None = None
    validation_state_count: int | None = None
    role_stratified_validation: bool = False
    train_ratio: float = 20000 / 22000
    target_standardization: bool = True
    weight_decay: float = 1e-4
    patience: int = 10
    min_delta: float = 0.0
    minimum_validation_diff_pairs: int = 200
    minimum_validation_ranking_states: int = 100
    device: RequestedTorchDevice = "cpu"
    optimizer: str = "AdamW"

    def resolved_teacher(self) -> BiddingRoleValueTeacher:
        if self.teacher is not None:
            return self.teacher
        return default_teacher_for_role(self.role)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["hidden_dims"] = list(self.hidden_dims)
        data["teacher"] = self.resolved_teacher()
        data["algorithm"] = "role-specific-bidding-value-regression-v1"
        return data


@dataclass(frozen=True)
class BiddingRoleValueTrainResult:
    model: BiddingRoleValueModel
    split: BiddingMultiHeadQSplit
    config: BiddingRoleValueTrainConfig
    device: ResolvedTorchDevice
    standardization: Standardization
    epoch_reports: list[dict[str, object]]
    best_epoch: int
    best_monitor_value: float
    train_report: dict[str, object]
    validation_report: dict[str, object]
    coverage: dict[str, object]


class BiddingRoleValueBatch(NamedTuple):
    model_input: Tensor
    action_index: Tensor
    target: Tensor
    role_mask: Tensor


class _BiddingRoleValueDataset(Dataset[BiddingRoleValueBatch]):
    def __init__(
        self,
        samples: Iterable[BiddingQRawSample],
        *,
        role: BiddingRoleValueRole,
        teacher: BiddingRoleValueTeacher,
        standardization: Standardization,
    ) -> None:
        self.samples = tuple(samples)
        self.role = role
        self.teacher = teacher
        self.standardization = standardization

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> BiddingRoleValueBatch:
        sample = self.samples[index]
        value = role_value_teacher(sample, role=self.role, teacher=self.teacher)
        return BiddingRoleValueBatch(
            model_input=torch.as_tensor(sample.model_input, dtype=torch.float32),
            action_index=torch.tensor(sample.forced_action_index, dtype=torch.long),
            target=self.standardization.encode_tensor(
                torch.tensor(0.0 if value is None else value, dtype=torch.float32)
            ),
            role_mask=torch.tensor(value is not None, dtype=torch.bool),
        )


def train_bidding_role_value_model(
    dataset: BiddingQDataset,
    config: BiddingRoleValueTrainConfig,
) -> BiddingRoleValueTrainResult:
    _validate_config(config)
    _configure_reproducibility(config.seed)
    teacher = config.resolved_teacher()
    split = create_role_value_split(dataset, config)
    device = resolve_torch_device(config.device, flag_name="--device")
    train_values = [
        cast(float, role_value_teacher(sample, role=config.role, teacher=teacher))
        for sample in split.train_samples
        if role_value_teacher(sample, role=config.role, teacher=teacher) is not None
    ]
    if not train_values:
        raise ValueError(f"train split has no {config.role} samples for teacher {teacher}.")
    standardization = _standardization(train_values, enabled=config.target_standardization)
    model = create_seeded_bidding_role_value_model(
        BiddingRoleValueModelConfig(
            role=config.role,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
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
        teacher=teacher,
        standardization=standardization,
        batch_size=config.batch_size,
        seed=config.seed,
        shuffle=True,
    )
    validation_loader = _create_loader(
        split.validation_samples,
        role=config.role,
        teacher=teacher,
        standardization=standardization,
        batch_size=config.batch_size,
        seed=config.seed,
        shuffle=False,
    )
    epoch_reports: list[dict[str, object]] = []
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    best_monitor = math.inf
    misses = 0
    for epoch in range(1, config.epochs + 1):
        train_loss = _run_epoch(model, train_loader, device=device, optimizer=optimizer)
        validation_loss = _run_epoch(model, validation_loader, device=device, optimizer=None)
        improved = validation_loss < best_monitor - config.min_delta
        if improved:
            best_monitor = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(cpu_state_dict(model))
            misses = 0
        else:
            misses += 1
        epoch_reports.append(
            {
                "epoch": epoch,
                "trainMse": train_loss,
                "validationMse": validation_loss,
                "best": improved,
            }
        )
        if misses >= config.patience:
            break
    if best_state is None:
        raise AssertionError("best checkpoint was not captured.")
    model.load_state_dict(best_state)
    coverage = role_value_coverage(dataset.raw_samples)
    train_report = evaluate_bidding_role_value_model(
        model,
        split.train_samples,
        train_samples=split.train_samples,
        role=config.role,
        teacher=teacher,
        device=device,
        standardization=standardization,
    )
    validation_report = evaluate_bidding_role_value_model(
        model,
        split.validation_samples,
        train_samples=split.train_samples,
        role=config.role,
        teacher=teacher,
        device=device,
        standardization=standardization,
    )
    return BiddingRoleValueTrainResult(
        model=model,
        split=split,
        config=config,
        device=device,
        standardization=standardization,
        epoch_reports=epoch_reports,
        best_epoch=best_epoch,
        best_monitor_value=best_monitor,
        train_report=train_report,
        validation_report=validation_report,
        coverage=coverage,
    )


def create_role_value_split(
    dataset: BiddingQDataset,
    config: BiddingRoleValueTrainConfig,
) -> BiddingMultiHeadQSplit:
    if config.validation_state_keys_path is not None:
        return create_multi_head_split(
            dataset,
            BiddingMultiHeadQTrainConfig(
                seed=config.seed,
                train_state_count=config.train_state_count,
                validation_state_keys_path=config.validation_state_keys_path,
                train_ratio=config.train_ratio,
            ),
        )
    if config.role_stratified_validation:
        validation_keys = select_role_stratified_validation_state_keys(
            dataset.raw_samples,
            role=config.role,
            teacher=config.resolved_teacher(),
            seed=config.seed,
            validation_state_count=_resolved_validation_state_count(dataset, config),
        )
        return _create_split_from_validation_keys(dataset, config, validation_keys)
    return create_multi_head_split(
        dataset,
        BiddingMultiHeadQTrainConfig(
            seed=config.seed,
            train_state_count=config.train_state_count,
            train_ratio=config.train_ratio,
        ),
    )


def select_role_stratified_validation_state_keys(
    samples: Iterable[BiddingQRawSample],
    *,
    role: BiddingRoleValueRole,
    teacher: BiddingRoleValueTeacher,
    seed: int,
    validation_state_count: int,
) -> tuple[str, ...]:
    sample_tuple = tuple(samples)
    all_state_keys = _state_keys_in_source_order(sample_tuple)
    if validation_state_count <= 0 or validation_state_count >= len(all_state_keys):
        raise ValueError(
            f"validation_state_count must be in [1,{len(all_state_keys) - 1}], "
            f"got {validation_state_count}."
        )
    state_stats = _role_value_state_stats(sample_tuple, role=role, teacher=teacher)

    def stable_hash(value: str, bucket: str) -> str:
        return hashlib.sha256(f"{seed}:{bucket}:{value}".encode()).hexdigest()

    ranked = sorted(
        all_state_keys,
        key=lambda key: (
            _role_stratified_state_bucket(state_stats[key]),
            stable_hash(key, "role-stratified-validation"),
        ),
    )
    selected = tuple(sorted(ranked[:validation_state_count]))
    selected_stats = _teacher_pair_coverage(
        tuple(sample for sample in sample_tuple if sample.state_key in selected),
        role=role,
        teacher=teacher,
    )
    if selected_stats["sampleCount"] == 0:
        raise ValueError(f"role-stratified validation selected no {role} teacher samples.")
    return selected


def _role_stratified_state_bucket(stats: dict[str, int]) -> int:
    if stats["differentPairCount"] > 0:
        return 0
    if stats["pairCount"] > 0:
        return 1
    if stats["sampleCount"] > 0:
        return 2
    return 3


def role_value_loss(
    value: Tensor,
    action_index: Tensor,
    target: Tensor,
    role_mask: Tensor,
) -> Tensor:
    if value.ndim != 2 or value.shape[1] != BIDDING_ACTION_COUNT:
        raise ValueError("value must have shape (batch, 29).")
    batch_index = torch.arange(action_index.shape[0], device=action_index.device)
    selected = value[batch_index, action_index]
    if not bool(role_mask.any().item()):
        return selected.sum() * 0.0
    return F.mse_loss(selected[role_mask], target[role_mask])


def role_value_teacher(
    sample: BiddingQRawSample,
    *,
    role: BiddingRoleValueRole,
    teacher: BiddingRoleValueTeacher,
) -> float | None:
    if canonical_bidding_role(sample.final_role) != role:
        return None
    if (
        sample.result_type != "standard"
        or sample.final_declared_target is None
        or sample.napoleon_side_point_cards is None
        or sample.coalition_side_point_cards is None
    ):
        return None
    if role == "citizen":
        if teacher == "coalition-side-point-cards":
            return float(sample.coalition_side_point_cards)
        if teacher == "negative-contract-margin":
            return float(sample.final_declared_target - sample.napoleon_side_point_cards)
        if teacher == "citizen-margin":
            return float(sample.coalition_side_point_cards - sample.napoleon_side_point_cards)
        if teacher == "contract-failure-binary":
            return 1.0 if not sample.contract_success else 0.0
    if role == "adjutant":
        margin = (
            sample.contract_margin
            if sample.contract_margin is not None
            else sample.napoleon_side_point_cards - sample.final_declared_target
        )
        if teacher == "contract-margin":
            return float(margin)
        if teacher == "napoleon-side-point-cards":
            return float(sample.napoleon_side_point_cards)
        if teacher == "contract-success":
            return 1.0 if sample.contract_success else 0.0
    raise ValueError(f"teacher {teacher!r} is not valid for role {role!r}.")


def default_teacher_for_role(role: BiddingRoleValueRole) -> BiddingRoleValueTeacher:
    return "negative-contract-margin" if role == "citizen" else "contract-margin"


def evaluate_bidding_role_value_model(
    model: BiddingRoleValueModel,
    samples: Iterable[BiddingQRawSample],
    *,
    train_samples: Iterable[BiddingQRawSample],
    role: BiddingRoleValueRole,
    teacher: BiddingRoleValueTeacher,
    device: ResolvedTorchDevice,
    standardization: Standardization,
) -> dict[str, object]:
    sample_tuple = tuple(samples)
    train_tuple = tuple(train_samples)
    prediction = predict_role_value_samples(
        model,
        sample_tuple,
        device=device,
        standardization=standardization,
    )
    truth, selected, selected_samples = _selected_arrays(sample_tuple, prediction, role, teacher)
    return {
        "role": role,
        "teacher": teacher,
        "rawSampleCount": len(sample_tuple),
        "teacherSampleCount": int(truth.size),
        "teacherStateCount": len({sample.state_key for sample in selected_samples}),
        "coverage": role_value_coverage(sample_tuple),
        "regression": {
            **_regression_metrics(selected, truth),
            "sampleCount": int(truth.size),
        },
        "byActionType": _grouped_regression(sample_tuple, prediction, role, teacher, "actionType"),
        "bySuit": _grouped_regression(sample_tuple, prediction, role, teacher, "suit"),
        "byTarget": _grouped_regression(sample_tuple, prediction, role, teacher, "target"),
        "ranking": role_value_ranking(sample_tuple, prediction, role=role, teacher=teacher),
        "baselines": role_value_baselines(
            train_tuple,
            selected_samples,
            truth=truth,
            role=role,
            teacher=teacher,
        ),
    }


def role_value_learning_assessment(
    validation_report: dict[str, object],
    *,
    minimum_diff_pairs: int,
    minimum_ranking_states: int,
) -> dict[str, object]:
    ranking = cast(dict[str, object], validation_report["ranking"])
    regression = cast(dict[str, object], validation_report["regression"])
    baselines = cast(dict[str, object], validation_report["baselines"])
    raw_diff_pairs = ranking["differentPairCount"]
    raw_ranking_states = ranking["rankingStateCount"]
    if not isinstance(raw_diff_pairs, int) or not isinstance(raw_ranking_states, int):
        raise ValueError("ranking report must include integer coverage counts.")
    diff_pairs = raw_diff_pairs
    ranking_states = raw_ranking_states
    coverage_sufficient = (
        diff_pairs >= minimum_diff_pairs and ranking_states >= minimum_ranking_states
    )
    baseline_names = ("globalMean", "actionIndexMean", "suitTargetMean")
    mae = regression["mae"]
    rmse = regression["rmse"]
    pearson = regression["pearsonCorrelation"]
    pairwise = ranking["pairwiseAccuracy"]
    beats_baselines = _beats_regression_baselines(
        mae=mae,
        rmse=rmse,
        baselines=baselines,
        names=baseline_names,
    )
    ranking_signal = (
        isinstance(pairwise, float)
        and pairwise > 0.5
        and isinstance(pearson, float)
        and pearson > 0.0
    )
    established = coverage_sufficient and beats_baselines and ranking_signal
    reasons = []
    if not coverage_sufficient:
        reasons.append(
            "insufficient validation ranking coverage: "
            f"diff_pairs={diff_pairs}/{minimum_diff_pairs}, "
            f"ranking_states={ranking_states}/{minimum_ranking_states}"
        )
    if coverage_sufficient and not beats_baselines:
        reasons.append("NN regression did not beat all required baselines on MAE/RMSE")
    if coverage_sufficient and not ranking_signal:
        reasons.append("state-conditioned ranking signal was not clearly positive")
    return {
        "established": established,
        "coverageSufficient": coverage_sufficient,
        "minimumDifferentPairs": minimum_diff_pairs,
        "minimumRankingStates": minimum_ranking_states,
        "differentPairCount": diff_pairs,
        "rankingStateCount": ranking_states,
        "beatsBaselines": beats_baselines,
        "rankingSignal": ranking_signal,
        "reason": "; ".join(reasons) if reasons else "coverage and validation metrics passed",
    }


def predict_role_value_samples(
    model: BiddingRoleValueModel,
    samples: tuple[BiddingQRawSample, ...],
    *,
    device: ResolvedTorchDevice,
    standardization: Standardization,
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
            batches.append(model(model_input).detach().cpu().numpy())
    return standardization.decode_array(np.concatenate(batches, axis=0))


def role_value_coverage(samples: Iterable[BiddingQRawSample]) -> dict[str, object]:
    sample_tuple = tuple(samples)
    roles: tuple[BiddingRoleValueRole, ...] = ("citizen", "adjutant")
    role_counts: dict[str, object] = {}
    role_state_counts: dict[str, object] = {}
    pass_bid: dict[str, object] = {}
    action_counts: dict[str, object] = {}
    action_coverage_counts: dict[str, object] = {}
    suit_counts: dict[str, object] = {}
    target_counts: dict[str, object] = {}
    role_matching_action_distribution: dict[str, object] = {}
    teacher_coverage: dict[str, object] = {}
    for role in roles:
        role_samples = [
            sample for sample in sample_tuple if canonical_bidding_role(sample.final_role) == role
        ]
        role_counts[role] = len(role_samples)
        role_state_counts[role] = len({sample.state_key for sample in role_samples})
        pass_bid[role] = dict(Counter(sample.forced_action_type for sample in role_samples))
        action_counts[role] = dict(
            Counter(str(sample.forced_action_index) for sample in role_samples)
        )
        action_coverage_counts[role] = len(
            {sample.forced_action_index for sample in role_samples}
        )
        suit_counts[role] = dict(Counter(str(sample.forced_suit) for sample in role_samples))
        target_counts[role] = dict(
            Counter(str(sample.forced_target_point_cards) for sample in role_samples)
        )
        by_state = Counter(sample.state_key for sample in role_samples)
        role_matching_action_distribution[role] = dict(
            Counter(str(count) for count in by_state.values())
        )
        teachers: tuple[BiddingRoleValueTeacher, ...] = (
            (
                "coalition-side-point-cards",
                "negative-contract-margin",
                "citizen-margin",
                "contract-failure-binary",
            )
            if role == "citizen"
            else ("contract-margin", "napoleon-side-point-cards", "contract-success")
        )
        teacher_coverage[role] = {
            teacher: _teacher_pair_coverage(sample_tuple, role=role, teacher=teacher)
            for teacher in teachers
        }
    return {
        "sampleCount": len(sample_tuple),
        "stateCount": len({sample.state_key for sample in sample_tuple}),
        "roleSampleCounts": role_counts,
        "roleStateCounts": role_state_counts,
        "passBidRoleDistribution": pass_bid,
        "actionCoverage": action_counts,
        "actionIndexCoverageCounts": action_coverage_counts,
        "suitCoverage": suit_counts,
        "targetCoverage": target_counts,
        "roleMatchingActionCountDistribution": role_matching_action_distribution,
        "teacherCoverage": teacher_coverage,
        "forcedActionRoleDistribution": _forced_action_role_distribution(sample_tuple),
    }


def role_value_ranking(
    samples: tuple[BiddingQRawSample, ...],
    score: np.ndarray,
    *,
    role: BiddingRoleValueRole,
    teacher: BiddingRoleValueTeacher,
) -> dict[str, object]:
    by_state: dict[str, list[tuple[int, float, float, BiddingQRawSample]]] = defaultdict(list)
    for index, sample in enumerate(samples):
        truth = role_value_teacher(sample, role=role, teacher=teacher)
        if truth is None:
            continue
        by_state[sample.state_key].append(
            (
                sample.forced_action_index,
                float(score[index, sample.forced_action_index]),
                truth,
                sample,
            )
        )
    pair_total = 0
    pair_different = 0
    tie_pairs = 0
    correct = 0.0
    best_hit = 0
    top3_hit = 0
    evaluated_states = 0
    strongest_match = 0
    strongest_total = 0
    for state_items in by_state.values():
        if len(state_items) < 2:
            continue
        evaluated_states += 1
        for left in range(len(state_items)):
            for right in range(left + 1, len(state_items)):
                pair_total += 1
                truth_diff = state_items[left][2] - state_items[right][2]
                if truth_diff == 0.0:
                    tie_pairs += 1
                    continue
                pair_different += 1
                score_diff = state_items[left][1] - state_items[right][1]
                if score_diff == 0.0:
                    correct += 0.5
                elif (score_diff > 0.0) == (truth_diff > 0.0):
                    correct += 1.0
        best_truth = max(item[2] for item in state_items)
        teacher_best = {item[0] for item in state_items if item[2] == best_truth}
        ranked = sorted(state_items, key=lambda item: (-item[1], item[0]))
        if ranked[0][0] in teacher_best:
            best_hit += 1
        if any(item[0] in teacher_best for item in ranked[:3]):
            top3_hit += 1
        best_sample = ranked[0][3]
        if best_sample.forced_suit is not None:
            strongest_total += 1
            if best_sample.forced_suit == best_sample.strongest_suit:
                strongest_match += 1
    return {
        "rankingStateCount": evaluated_states,
        "pairCount": pair_total,
        "differentPairCount": pair_different,
        "teacherTiePairCount": tie_pairs,
        "teacherTieRate": _safe_div(tie_pairs, pair_total),
        "pairwiseAccuracy": _safe_div(correct, pair_different),
        "bestActionHitRate": _safe_div(best_hit, evaluated_states),
        "top3HitRate": _safe_div(top3_hit, evaluated_states),
        "strongestSuitMatchRate": _safe_div(strongest_match, strongest_total),
    }


def role_value_baselines(
    train_samples: tuple[BiddingQRawSample, ...],
    validation_samples: tuple[BiddingQRawSample, ...],
    *,
    truth: np.ndarray,
    role: BiddingRoleValueRole,
    teacher: BiddingRoleValueTeacher,
) -> dict[str, object]:
    return {
        "globalMean": _baseline(
            train_samples,
            validation_samples,
            truth,
            role,
            teacher,
            lambda _s: "global",
        ),
        "actionIndexMean": _baseline(
            train_samples,
            validation_samples,
            truth,
            role,
            teacher,
            lambda s: str(s.forced_action_index),
        ),
        "suitTargetMean": _baseline(
            train_samples,
            validation_samples,
            truth,
            role,
            teacher,
            lambda s: f"{s.forced_suit}:{s.forced_target_point_cards}",
        ),
        "roleActionIndexMean": _baseline(
            train_samples,
            validation_samples,
            truth,
            role,
            teacher,
            lambda s: f"{canonical_bidding_role(s.final_role)}:{s.forced_action_index}",
        ),
    }


def save_bidding_role_value_artifact(
    output_directory: Path | str,
    *,
    result: BiddingRoleValueTrainResult,
    dataset: BiddingQDataset,
) -> dict[str, object]:
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
    assessment = role_value_learning_assessment(
        result.validation_report,
        minimum_diff_pairs=result.config.minimum_validation_diff_pairs,
        minimum_ranking_states=result.config.minimum_validation_ranking_states,
    )
    metadata["learningAssessment"] = assessment
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "epochs": result.epoch_reports,
        "bestEpoch": result.best_epoch,
        "bestMonitorValue": result.best_monitor_value,
        "coverage": result.coverage,
        "train": result.train_report,
        "validation": result.validation_report,
        "learningAssessment": assessment,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "checkpointPath": str(checkpoint_path),
        "metadataPath": str(metadata_path),
        "reportPath": str(report_path),
    }


def load_bidding_role_value_checkpoint(
    path: Path | str,
) -> tuple[BiddingRoleValueModel, dict[str, object]]:
    try:
        raw = torch.load(Path(path), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, pickle.UnpicklingError) as error:
        raise BiddingRoleValueCheckpointError(f"checkpoint cannot be loaded: {error}") from error
    if not isinstance(raw, dict):
        raise BiddingRoleValueCheckpointError("checkpoint must be a dictionary.")
    if raw.get("checkpointSchemaVersion") != BIDDING_ROLE_VALUE_CHECKPOINT_SCHEMA_VERSION:
        raise BiddingRoleValueCheckpointError("checkpoint schema version mismatch.")
    if raw.get("modelType") != BIDDING_ROLE_VALUE_MODEL_TYPE:
        raise BiddingRoleValueCheckpointError("checkpoint modelType mismatch.")
    config_raw = raw.get("modelConfig")
    if not isinstance(config_raw, dict):
        raise BiddingRoleValueCheckpointError("checkpoint modelConfig must be an object.")
    model = BiddingRoleValueModel(BiddingRoleValueModelConfig.from_dict(config_raw))
    state = raw.get("modelState")
    if not isinstance(state, dict):
        raise BiddingRoleValueCheckpointError("checkpoint modelState must be a state dict.")
    model.load_state_dict(state)
    return model, raw


def _run_epoch(
    model: BiddingRoleValueModel,
    dataloader: DataLoader[BiddingRoleValueBatch],
    *,
    device: ResolvedTorchDevice,
    optimizer: optim.Optimizer | None,
) -> float:
    model.train(optimizer is not None)
    total_loss = 0.0
    total = 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for batch in dataloader:
            moved = _move_batch(batch, device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss = role_value_loss(
                model(moved.model_input),
                moved.action_index,
                moved.target,
                moved.role_mask,
            )
            if optimizer is not None:
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
            count = int(moved.role_mask.sum().item())
            total_loss += float(loss.item()) * count
            total += count
    if total == 0:
        raise ValueError("split contains no teacher samples.")
    return total_loss / total


def _create_loader(
    samples: Iterable[BiddingQRawSample],
    *,
    role: BiddingRoleValueRole,
    teacher: BiddingRoleValueTeacher,
    standardization: Standardization,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> DataLoader[BiddingRoleValueBatch]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        _BiddingRoleValueDataset(
            samples,
            role=role,
            teacher=teacher,
            standardization=standardization,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def _move_batch(batch: BiddingRoleValueBatch, device: ResolvedTorchDevice) -> BiddingRoleValueBatch:
    return BiddingRoleValueBatch(
        model_input=batch.model_input.to(device=device.torch_device, dtype=torch.float32),
        action_index=batch.action_index.to(device=device.torch_device, dtype=torch.long),
        target=batch.target.to(device=device.torch_device, dtype=torch.float32),
        role_mask=batch.role_mask.to(device=device.torch_device, dtype=torch.bool),
    )


def _selected_arrays(
    samples: tuple[BiddingQRawSample, ...],
    prediction: np.ndarray,
    role: BiddingRoleValueRole,
    teacher: BiddingRoleValueTeacher,
) -> tuple[np.ndarray, np.ndarray, tuple[BiddingQRawSample, ...]]:
    truth = []
    selected = []
    selected_samples = []
    for index, sample in enumerate(samples):
        value = role_value_teacher(sample, role=role, teacher=teacher)
        if value is None:
            continue
        truth.append(value)
        selected.append(float(prediction[index, sample.forced_action_index]))
        selected_samples.append(sample)
    return (
        np.asarray(truth, dtype=np.float64),
        np.asarray(selected, dtype=np.float64),
        tuple(selected_samples),
    )


def _grouped_regression(
    samples: tuple[BiddingQRawSample, ...],
    prediction: np.ndarray,
    role: BiddingRoleValueRole,
    teacher: BiddingRoleValueTeacher,
    group: Literal["actionType", "suit", "target"],
) -> dict[str, object]:
    buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for index, sample in enumerate(samples):
        truth = role_value_teacher(sample, role=role, teacher=teacher)
        if truth is None:
            continue
        key: str
        if group == "actionType":
            key = sample.forced_action_type
        elif group == "suit":
            key = str(sample.forced_suit)
        else:
            key = str(sample.forced_target_point_cards)
        buckets[key].append((float(prediction[index, sample.forced_action_index]), truth))
    return {
        key: {
            **_regression_metrics(
                np.asarray([item[0] for item in values], dtype=np.float64),
                np.asarray([item[1] for item in values], dtype=np.float64),
            ),
            "sampleCount": len(values),
        }
        for key, values in sorted(buckets.items())
    }


def _teacher_pair_coverage(
    samples: tuple[BiddingQRawSample, ...],
    *,
    role: BiddingRoleValueRole,
    teacher: BiddingRoleValueTeacher,
) -> dict[str, object]:
    by_state: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        value = role_value_teacher(sample, role=role, teacher=teacher)
        if value is not None:
            by_state[sample.state_key].append(value)
    pairs = 0
    diff = 0
    ties = 0
    states_with_pair = 0
    for values in by_state.values():
        if len(values) >= 2:
            states_with_pair += 1
        for left in range(len(values)):
            for right in range(left + 1, len(values)):
                pairs += 1
                if values[left] == values[right]:
                    ties += 1
                else:
                    diff += 1
    return {
        "sampleCount": sum(len(values) for values in by_state.values()),
        "stateCount": len(by_state),
        "rankingStateCount": states_with_pair,
        "pairCount": pairs,
        "differentPairCount": diff,
        "teacherTiePairCount": ties,
        "teacherTieRate": _safe_div(ties, pairs),
    }


def _forced_action_role_distribution(
    samples: tuple[BiddingQRawSample, ...],
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        counts[str(sample.forced_action_index)][canonical_bidding_role(sample.final_role)] += 1
    return {
        key: dict(value)
        for key, value in sorted(counts.items(), key=lambda item: int(item[0]))
    }


def _create_split_from_validation_keys(
    dataset: BiddingQDataset,
    config: BiddingRoleValueTrainConfig,
    validation_keys: tuple[str, ...],
) -> BiddingMultiHeadQSplit:
    all_state_keys = _state_keys_in_source_order(dataset.raw_samples)
    all_state_key_set = set(all_state_keys)
    missing = set(validation_keys) - all_state_key_set
    if missing:
        raise ValueError(f"validation state keys missing from dataset: {len(missing)}.")
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


def _resolved_validation_state_count(
    dataset: BiddingQDataset,
    config: BiddingRoleValueTrainConfig,
) -> int:
    all_state_count = len(_state_keys_in_source_order(dataset.raw_samples))
    if config.validation_state_count is not None:
        return config.validation_state_count
    return all_state_count - int(round(all_state_count * config.train_ratio))


def _state_keys_in_source_order(samples: Iterable[BiddingQRawSample]) -> tuple[str, ...]:
    keys: list[str] = []
    seen: set[str] = set()
    for sample in samples:
        if sample.state_key in seen:
            continue
        seen.add(sample.state_key)
        keys.append(sample.state_key)
    return tuple(keys)


def _role_value_state_stats(
    samples: tuple[BiddingQRawSample, ...],
    *,
    role: BiddingRoleValueRole,
    teacher: BiddingRoleValueTeacher,
) -> dict[str, dict[str, int]]:
    values_by_state: dict[str, list[float]] = defaultdict(list)
    all_state_keys = _state_keys_in_source_order(samples)
    for sample in samples:
        value = role_value_teacher(sample, role=role, teacher=teacher)
        if value is not None:
            values_by_state[sample.state_key].append(value)
    stats: dict[str, dict[str, int]] = {}
    for state_key in all_state_keys:
        values = values_by_state[state_key]
        pair_count = 0
        different_pair_count = 0
        for left in range(len(values)):
            for right in range(left + 1, len(values)):
                pair_count += 1
                if values[left] != values[right]:
                    different_pair_count += 1
        stats[state_key] = {
            "sampleCount": len(values),
            "pairCount": pair_count,
            "differentPairCount": different_pair_count,
        }
    return stats


def _beats_regression_baselines(
    *,
    mae: object,
    rmse: object,
    baselines: dict[str, object],
    names: tuple[str, ...],
) -> bool:
    if not isinstance(mae, float) or not isinstance(rmse, float):
        return False
    for name in names:
        baseline = baselines.get(name)
        if not isinstance(baseline, dict):
            return False
        baseline_mae = baseline.get("mae")
        baseline_rmse = baseline.get("rmse")
        if not isinstance(baseline_mae, float) or not isinstance(baseline_rmse, float):
            return False
        if mae >= baseline_mae or rmse >= baseline_rmse:
            return False
    return True


def _baseline(
    train_samples: tuple[BiddingQRawSample, ...],
    validation_samples: tuple[BiddingQRawSample, ...],
    truth: np.ndarray,
    role: BiddingRoleValueRole,
    teacher: BiddingRoleValueTeacher,
    key_fn: Callable[[BiddingQRawSample], str],
) -> dict[str, object]:
    grouped: dict[str, list[float]] = defaultdict(list)
    all_train = []
    for sample in train_samples:
        value = role_value_teacher(sample, role=role, teacher=teacher)
        if value is None:
            continue
        grouped[key_fn(sample)].append(value)
        all_train.append(value)
    if not all_train:
        return {"skipped": True}
    global_mean = float(np.mean(all_train))
    means = {key: float(np.mean(values)) for key, values in grouped.items()}
    prediction = np.asarray(
        [means.get(key_fn(sample), global_mean) for sample in validation_samples],
        dtype=np.float64,
    )
    return {
        **_regression_metrics(prediction, truth),
        "sampleCount": int(truth.size),
    }


def _standardization(values: list[float], *, enabled: bool) -> Standardization:
    std = float(np.std(values))
    if std <= 0.0:
        std = 1.0
    return Standardization(enabled=enabled, mean=float(np.mean(values)), std=std)


def _checkpoint_dict(
    *,
    result: BiddingRoleValueTrainResult,
    dataset: BiddingQDataset,
) -> dict[str, object]:
    return {
        "checkpointSchemaVersion": BIDDING_ROLE_VALUE_CHECKPOINT_SCHEMA_VERSION,
        "modelType": BIDDING_ROLE_VALUE_MODEL_TYPE,
        "architectureId": architecture_id_for_role(result.config.role),
        "modelConfig": result.model.config.to_dict(),
        "modelState": cpu_state_dict(result.model),
        "trainingConfig": result.config.to_dict(),
        "targetStandardization": result.standardization.to_dict(),
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
            "validationMode": (
                "role-stratified" if result.config.role_stratified_validation else "default"
            ),
            "stateKeyLeakage": False,
        },
        "learningAssessment": role_value_learning_assessment(
            result.validation_report,
            minimum_diff_pairs=result.config.minimum_validation_diff_pairs,
            minimum_ranking_states=result.config.minimum_validation_ranking_states,
        ),
    }


def _validate_config(config: BiddingRoleValueTrainConfig) -> None:
    if config.role not in ("citizen", "adjutant"):
        raise ValueError("role must be citizen or adjutant.")
    teacher = config.resolved_teacher()
    if config.role == "citizen" and teacher not in (
        "coalition-side-point-cards",
        "negative-contract-margin",
        "citizen-margin",
        "contract-failure-binary",
    ):
        raise ValueError("invalid Citizen teacher.")
    if config.role == "adjutant" and teacher not in (
        "contract-margin",
        "napoleon-side-point-cards",
        "contract-success",
    ):
        raise ValueError("invalid Adjutant teacher.")
    if config.epochs <= 0 or config.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive.")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if config.patience <= 0:
        raise ValueError("patience must be positive.")
    if config.validation_state_count is not None and config.validation_state_count <= 0:
        raise ValueError("validation_state_count must be positive.")
    if config.minimum_validation_diff_pairs < 0:
        raise ValueError("minimum_validation_diff_pairs must be non-negative.")
    if config.minimum_validation_ranking_states < 0:
        raise ValueError("minimum_validation_ranking_states must be non-negative.")


def _configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
