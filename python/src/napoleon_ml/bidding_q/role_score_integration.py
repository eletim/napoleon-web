"""Offline role-probability x role-value bidding score evaluation for Issue #390."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from napoleon_ml.dataset.constants import BIDDING_ACTION_COUNT
from napoleon_ml.policy.device import RequestedTorchDevice, resolve_torch_device

from .dataset import (
    BiddingQDataset,
    BiddingQRawSample,
    SuitName,
    action_suit,
    action_target,
    load_bidding_q_dataset,
)
from .margin_training import load_bidding_margin_checkpoint, predict_margin_samples
from .multi_head_training import (
    Standardization,
    load_bidding_multi_head_q_checkpoint,
    predict_multi_head_samples,
)
from .role_model import BIDDING_ROLE_CLASSES
from .role_training import canonical_bidding_role, role_index_for_terminal_role
from .role_value_training import load_bidding_role_value_checkpoint, predict_role_value_samples

ScoreVariant = Literal["S0", "S1", "S2", "S3"]
BaselineName = Literal[
    "behavior",
    "rule-based-offline",
    "conservative-offline",
    "random-legal",
    "pass-only",
]

_NAPOLEON_INDEX = BIDDING_ROLE_CLASSES.index("napoleon")
_NAPOLEON_ADJUTANT_INDEX = BIDDING_ROLE_CLASSES.index("napoleon-adjutant")
_ADJUTANT_INDEX = BIDDING_ROLE_CLASSES.index("adjutant")
_CITIZEN_INDEX = BIDDING_ROLE_CLASSES.index("citizen")


@dataclass(frozen=True)
class AffineCalibration:
    role: str
    slope: float
    intercept: float
    sample_count: int

    def apply(self, values: np.ndarray) -> np.ndarray:
        return self.slope * values + self.intercept

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "slope": self.slope,
            "intercept": self.intercept,
            "sampleCount": self.sample_count,
        }


@dataclass(frozen=True)
class StateKeySplit:
    calibration_state_keys: frozenset[str]
    validation_state_keys: frozenset[str]
    final_state_keys: frozenset[str]
    seed: int

    def select(
        self, samples: tuple[BiddingQRawSample, ...], split: str
    ) -> tuple[BiddingQRawSample, ...]:
        if split == "calibration":
            keys = self.calibration_state_keys
        elif split == "validation":
            keys = self.validation_state_keys
        elif split == "final":
            keys = self.final_state_keys
        else:
            raise ValueError(f"unknown split: {split}")
        return tuple(sample for sample in samples if sample.state_key in keys)

    def to_dict(self, samples: tuple[BiddingQRawSample, ...]) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "calibrationStates": len(self.calibration_state_keys),
            "validationStates": len(self.validation_state_keys),
            "finalStates": len(self.final_state_keys),
            "calibrationSamples": len(self.select(samples, "calibration")),
            "validationSamples": len(self.select(samples, "validation")),
            "finalSamples": len(self.select(samples, "final")),
            "calibrationStateKeyHash": _state_key_hash(self.calibration_state_keys),
            "validationStateKeyHash": _state_key_hash(self.validation_state_keys),
            "finalStateKeyHash": _state_key_hash(self.final_state_keys),
            "stateKeyLeakage": False,
        }


@dataclass(frozen=True)
class RoleScorePredictions:
    role_probabilities: np.ndarray
    napoleon_value: np.ndarray
    napoleon_sigma: np.ndarray
    adjutant_value: np.ndarray
    citizen_value: np.ndarray


@dataclass(frozen=True)
class ScoreMatrices:
    variants: dict[ScoreVariant, np.ndarray]
    components: dict[ScoreVariant, dict[str, np.ndarray]]
    calibrations: dict[str, AffineCalibration]


def create_role_score_state_split(
    samples: tuple[BiddingQRawSample, ...],
    *,
    calibration_ratio: float = 0.25,
    validation_ratio: float = 0.25,
    seed: int = 390,
) -> StateKeySplit:
    if calibration_ratio <= 0.0 or validation_ratio <= 0.0:
        raise ValueError("calibration_ratio and validation_ratio must be positive.")
    if calibration_ratio + validation_ratio >= 1.0:
        raise ValueError("calibration_ratio + validation_ratio must be less than 1.")
    state_keys = sorted({sample.state_key for sample in samples})
    if len(state_keys) < 3:
        raise ValueError(
            "at least three states are required for calibration/validation/final split."
        )
    ordered = sorted(state_keys, key=lambda key: _stable_split_key(seed, key))
    calibration_count = max(1, int(round(len(ordered) * calibration_ratio)))
    validation_count = max(1, int(round(len(ordered) * validation_ratio)))
    if calibration_count + validation_count >= len(ordered):
        validation_count = max(1, len(ordered) - calibration_count - 1)
    calibration = frozenset(ordered[:calibration_count])
    validation = frozenset(ordered[calibration_count : calibration_count + validation_count])
    final = frozenset(ordered[calibration_count + validation_count :])
    if not final:
        raise ValueError("final split is empty.")
    if calibration & validation or calibration & final or validation & final:
        raise AssertionError("stateKey split leaked between partitions.")
    return StateKeySplit(
        calibration_state_keys=calibration,
        validation_state_keys=validation,
        final_state_keys=final,
        seed=seed,
    )


def teacher_utility(sample: BiddingQRawSample) -> float:
    role = canonical_bidding_role(sample.terminal_role)
    if role == "no-contract":
        return 0.0
    margin = _teacher_contract_margin(sample)
    if margin is None:
        return 0.0
    if role in ("napoleon", "napoleon-adjutant", "adjutant"):
        return float(margin)
    if role == "citizen":
        return float(-margin)
    raise ValueError(f"unsupported terminal role: {role}")


def fit_affine_calibration(
    role: str, predicted: np.ndarray, target: np.ndarray
) -> AffineCalibration:
    if predicted.shape != target.shape:
        raise ValueError("predicted and target must have the same shape.")
    if predicted.ndim != 1:
        raise ValueError("calibration arrays must be one-dimensional.")
    if predicted.size < 2:
        return AffineCalibration(
            role=role, slope=1.0, intercept=0.0, sample_count=int(predicted.size)
        )
    x = predicted.astype(np.float64)
    y = target.astype(np.float64)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("calibration arrays must be finite.")
    design = np.column_stack([x, np.ones_like(x)])
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    if not math.isfinite(float(slope)) or not math.isfinite(float(intercept)):
        return AffineCalibration(
            role=role, slope=1.0, intercept=0.0, sample_count=int(predicted.size)
        )
    return AffineCalibration(
        role=role,
        slope=float(slope),
        intercept=float(intercept),
        sample_count=int(predicted.size),
    )


def fit_role_calibrations(
    samples: tuple[BiddingQRawSample, ...],
    predictions: RoleScorePredictions,
) -> dict[str, AffineCalibration]:
    return {
        "napoleon": fit_affine_calibration(
            "napoleon",
            *_calibration_arrays(
                samples, predictions.napoleon_value, roles=("napoleon", "napoleon-adjutant")
            ),
        ),
        "adjutant": fit_affine_calibration(
            "adjutant",
            *_calibration_arrays(samples, predictions.adjutant_value, roles=("adjutant",)),
        ),
        "citizen": fit_affine_calibration(
            "citizen",
            *_calibration_arrays(samples, predictions.citizen_value, roles=("citizen",)),
        ),
    }


def compose_score_matrices(
    predictions: RoleScorePredictions,
    *,
    calibrations: dict[str, AffineCalibration],
    risk_lambda: float = 0.25,
) -> ScoreMatrices:
    role_probabilities = predictions.role_probabilities.astype(np.float64)
    validate_role_probabilities(role_probabilities)
    p_napoleon = (
        role_probabilities[:, :, _NAPOLEON_INDEX]
        + role_probabilities[:, :, _NAPOLEON_ADJUTANT_INDEX]
    )
    p_adjutant = role_probabilities[:, :, _ADJUTANT_INDEX]
    p_citizen = role_probabilities[:, :, _CITIZEN_INDEX]

    raw_napoleon = predictions.napoleon_value.astype(np.float64)
    raw_adjutant = predictions.adjutant_value.astype(np.float64)
    raw_citizen = predictions.citizen_value.astype(np.float64)
    calibrated_napoleon = calibrations["napoleon"].apply(raw_napoleon)
    calibrated_adjutant = calibrations["adjutant"].apply(raw_adjutant)
    calibrated_citizen = calibrations["citizen"].apply(raw_citizen)
    risk_napoleon = calibrations["napoleon"].apply(
        raw_napoleon - risk_lambda * predictions.napoleon_sigma.astype(np.float64)
    )

    s1_components = {
        "napoleon": p_napoleon * raw_napoleon,
        "adjutant": p_adjutant * raw_adjutant,
        "citizen": p_citizen * raw_citizen,
    }
    s2_components = {
        "napoleon": p_napoleon * calibrated_napoleon,
        "adjutant": p_adjutant * calibrated_adjutant,
        "citizen": p_citizen * calibrated_citizen,
    }
    s3_components = {
        "napoleon": p_napoleon * risk_napoleon,
        "adjutant": p_adjutant * calibrated_adjutant,
        "citizen": p_citizen * calibrated_citizen,
    }
    variants: dict[ScoreVariant, np.ndarray] = {
        "S0": raw_napoleon,
        "S1": _sum_components(s1_components),
        "S2": _sum_components(s2_components),
        "S3": _sum_components(s3_components),
    }
    return ScoreMatrices(
        variants=variants,
        components={
            "S0": {"napoleon": raw_napoleon},
            "S1": s1_components,
            "S2": s2_components,
            "S3": s3_components,
        },
        calibrations=calibrations,
    )


def validate_role_probabilities(role_probabilities: np.ndarray, *, atol: float = 1e-5) -> None:
    if role_probabilities.ndim != 3:
        raise ValueError("role probabilities must have shape (sample, action, role).")
    if role_probabilities.shape[1] != BIDDING_ACTION_COUNT:
        raise ValueError("role probabilities action dimension mismatch.")
    if role_probabilities.shape[2] != len(BIDDING_ROLE_CLASSES):
        raise ValueError("role probabilities role dimension mismatch.")
    if not np.all(np.isfinite(role_probabilities)):
        raise ValueError("role probabilities must be finite.")
    if np.any(role_probabilities < -atol):
        raise ValueError("role probabilities must be non-negative.")
    totals = role_probabilities.sum(axis=2)
    if not np.allclose(totals, 1.0, atol=atol):
        raise ValueError("role probabilities must sum to 1 across roles.")


def evaluate_score_matrices(
    samples: tuple[BiddingQRawSample, ...],
    score_matrices: dict[str, np.ndarray],
) -> dict[str, Any]:
    grouped = _group_state_actions(samples)
    results: dict[str, Any] = {}
    for name, scores in score_matrices.items():
        results[name] = _evaluate_named_score(grouped, scores)
    return results


def evaluate_policy_baselines(samples: tuple[BiddingQRawSample, ...]) -> dict[str, Any]:
    grouped = _group_state_actions(samples)
    baselines: dict[str, np.ndarray] = {}
    for name in (
        "behavior",
        "rule-based-offline",
        "conservative-offline",
        "random-legal",
        "pass-only",
    ):
        scores = np.zeros((len(samples), BIDDING_ACTION_COUNT), dtype=np.float64)
        for state in grouped.values():
            selected = _baseline_action_index(name, state)
            if selected not in state["action_indices"]:
                selected = 0 if 0 in state["action_indices"] else state["action_indices"][0]
            scores[state["first_sample_index"], selected] = 1.0
        baselines[name] = scores
    return {name: _evaluate_named_score(grouped, scores) for name, scores in baselines.items()}


def role_probability_metrics(
    samples: tuple[BiddingQRawSample, ...],
    role_probabilities: np.ndarray,
) -> dict[str, Any]:
    validate_role_probabilities(role_probabilities)
    selected = np.asarray(
        [
            role_probabilities[index, sample.forced_action_index]
            for index, sample in enumerate(samples)
        ],
        dtype=np.float64,
    )
    truth = np.asarray([role_index_for_terminal_role(sample.terminal_role) for sample in samples])
    predicted = selected.argmax(axis=1)
    confusion = np.zeros((len(BIDDING_ROLE_CLASSES), len(BIDDING_ROLE_CLASSES)), dtype=np.int64)
    for true_index, predicted_index in zip(truth, predicted, strict=True):
        confusion[int(true_index), int(predicted_index)] += 1
    per_role: dict[str, Any] = {}
    f1_values = []
    for index, role in enumerate(BIDDING_ROLE_CLASSES):
        tp = float(confusion[index, index])
        fp = float(confusion[:, index].sum() - confusion[index, index])
        fn = float(confusion[index, :].sum() - confusion[index, index])
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _f1(precision, recall)
        f1_values.append(0.0 if f1 is None else f1)
        per_role[role] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(confusion[index, :].sum()),
            "meanProbability": float(selected[:, index].mean()) if len(selected) else None,
        }
    one_hot = np.zeros_like(selected)
    one_hot[np.arange(len(truth)), truth] = 1.0
    return {
        "sampleCount": len(samples),
        "stateCount": len({sample.state_key for sample in samples}),
        "accuracy": float((predicted == truth).mean()) if len(truth) else None,
        "macroF1": float(np.mean(f1_values)) if f1_values else None,
        "brier": float(np.mean(np.sum((selected - one_hot) ** 2, axis=1))) if len(truth) else None,
        "perRole": per_role,
        "confusionMatrix": confusion.tolist(),
        "meanSelectedProbabilities": {
            role: float(selected[:, index].mean()) if len(selected) else None
            for index, role in enumerate(BIDDING_ROLE_CLASSES)
        },
    }


def evaluate_role_score_offline(
    *,
    dataset_directory: Path | str,
    role_checkpoint_path: Path | str,
    margin_checkpoint_path: Path | str,
    citizen_checkpoint_path: Path | str,
    adjutant_checkpoint_path: Path | str,
    device: RequestedTorchDevice = "cpu",
    split_seed: int = 390,
    verify_integrity: bool = True,
) -> dict[str, Any]:
    dataset = load_bidding_q_dataset(dataset_directory, verify_integrity=verify_integrity)
    resolved_device = resolve_torch_device(device, flag_name="--device")
    samples = dataset.raw_samples

    role_model, role_raw = load_bidding_multi_head_q_checkpoint(role_checkpoint_path)
    margin_model, margin_raw = load_bidding_margin_checkpoint(margin_checkpoint_path)
    citizen_model, citizen_raw = load_bidding_role_value_checkpoint(citizen_checkpoint_path)
    adjutant_model, adjutant_raw = load_bidding_role_value_checkpoint(adjutant_checkpoint_path)
    role_model.to(resolved_device.torch_device)
    margin_model.to(resolved_device.torch_device)
    citizen_model.to(resolved_device.torch_device)
    adjutant_model.to(resolved_device.torch_device)

    role_predictions = predict_multi_head_samples(
        role_model,
        samples,
        device=resolved_device,
        napoleon_standardization=_standardization_from_checkpoint(
            role_raw, "napoleonStandardization"
        ),
        margin_standardization=_standardization_from_checkpoint(role_raw, "marginStandardization"),
    )
    margin_predictions = predict_margin_samples(
        margin_model,
        samples,
        device=resolved_device,
        standardization=_standardization_from_checkpoint(margin_raw, "targetStandardization"),
    )
    citizen_values = predict_role_value_samples(
        citizen_model,
        samples,
        device=resolved_device,
        standardization=_standardization_from_checkpoint(citizen_raw, "targetStandardization"),
    )
    adjutant_values = predict_role_value_samples(
        adjutant_model,
        samples,
        device=resolved_device,
        standardization=_standardization_from_checkpoint(adjutant_raw, "targetStandardization"),
    )
    predictions = RoleScorePredictions(
        role_probabilities=role_predictions["roleProbabilities"],
        napoleon_value=margin_predictions["mean"],
        napoleon_sigma=margin_predictions["sigma"],
        adjutant_value=adjutant_values,
        citizen_value=citizen_values,
    )
    split = create_role_score_state_split(samples, seed=split_seed)
    calibration_indices = [
        index
        for index, sample in enumerate(samples)
        if sample.state_key in split.calibration_state_keys
    ]
    calibrations = fit_role_calibrations(
        tuple(samples[index] for index in calibration_indices),
        RoleScorePredictions(
            role_probabilities=predictions.role_probabilities[calibration_indices],
            napoleon_value=predictions.napoleon_value[calibration_indices],
            napoleon_sigma=predictions.napoleon_sigma[calibration_indices],
            adjutant_value=predictions.adjutant_value[calibration_indices],
            citizen_value=predictions.citizen_value[calibration_indices],
        ),
    )
    scores = compose_score_matrices(predictions, calibrations=calibrations)
    report: dict[str, Any] = {
        "dataset": _dataset_identity(dataset),
        "artifacts": {
            "roleProbabilityCheckpoint": str(role_checkpoint_path),
            "marginCheckpoint": str(margin_checkpoint_path),
            "citizenValueCheckpoint": str(citizen_checkpoint_path),
            "adjutantValueCheckpoint": str(adjutant_checkpoint_path),
        },
        "device": resolved_device.to_metadata(),
        "split": split.to_dict(samples),
        "calibration": {role: calibration.to_dict() for role, calibration in calibrations.items()},
        "scoreVariants": _variant_descriptions(),
        "splits": {},
    }
    split_reports: dict[str, Any] = {}
    for split_name in ("calibration", "validation", "final"):
        subset = split.select(samples, split_name)
        subset_indices = [
            index
            for index, sample in enumerate(samples)
            if sample.state_key in {s.state_key for s in subset}
        ]
        split_scores: dict[str, np.ndarray] = {
            name: matrix[subset_indices] for name, matrix in scores.variants.items()
        }
        split_role_probabilities = predictions.role_probabilities[subset_indices]
        split_reports[split_name] = {
            "sampleCount": len(subset),
            "stateCount": len({sample.state_key for sample in subset}),
            "roleProbability": role_probability_metrics(subset, split_role_probabilities),
            "variants": evaluate_score_matrices(subset, split_scores),
            "baselines": evaluate_policy_baselines(subset),
            "collapse": {
                name: _collapse_summary(metrics)
                for name, metrics in evaluate_score_matrices(subset, split_scores).items()
            },
            "decompositionExamples": _decomposition_examples(
                subset, split_scores, scores.components, subset_indices
            ),
        }
    report["splits"] = split_reports
    report["learningAssessment"] = _learning_assessment(split_reports["final"])  # type: ignore[arg-type]
    return report


def _calibration_arrays(
    samples: tuple[BiddingQRawSample, ...],
    values: np.ndarray,
    *,
    roles: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    predicted: list[float] = []
    target: list[float] = []
    for index, sample in enumerate(samples):
        if sample.terminal_role not in roles:
            continue
        predicted.append(float(values[index, sample.forced_action_index]))
        target.append(teacher_utility(sample))
    return np.asarray(predicted, dtype=np.float64), np.asarray(target, dtype=np.float64)


def _teacher_contract_margin(sample: BiddingQRawSample) -> int | None:
    if sample.contract_margin is not None:
        return sample.contract_margin
    if sample.final_declared_target is None or sample.napoleon_side_point_cards is None:
        return None
    return sample.napoleon_side_point_cards - sample.final_declared_target


def _sum_components(components: dict[str, np.ndarray]) -> np.ndarray:
    total = np.zeros_like(next(iter(components.values())), dtype=np.float64)
    for value in components.values():
        total = total + value
    return total


def _group_state_actions(samples: tuple[BiddingQRawSample, ...]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    utility_accumulator: dict[tuple[str, int], list[float]] = defaultdict(list)
    index_accumulator: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        utility_accumulator[(sample.state_key, sample.forced_action_index)].append(
            teacher_utility(sample)
        )
        index_accumulator[(sample.state_key, sample.forced_action_index)].append(index)
        if sample.state_key not in grouped:
            legal_actions = [
                action_index
                for action_index, is_legal in enumerate(sample.legal_bid_mask)
                if is_legal == 1.0
            ]
            grouped[sample.state_key] = {
                "state_key": sample.state_key,
                "first_sample": sample,
                "first_sample_index": index,
                "action_indices": legal_actions,
                "utilities": {},
            }
    for (state_key, action_index), values in utility_accumulator.items():
        grouped[state_key]["utilities"][action_index] = float(np.mean(values))  # type: ignore[index]
    for state in grouped.values():
        present = set(state["utilities"].keys())  # type: ignore[union-attr]
        state["action_indices"] = [
            action_index
            for action_index in state["action_indices"]  # type: ignore[union-attr]
            if action_index in present
        ]
    return grouped


def _evaluate_named_score(grouped: dict[str, dict[str, Any]], scores: np.ndarray) -> dict[str, Any]:
    pair_count = 0
    different_pair_count = 0
    tie_pair_count = 0
    correct = 0.0
    best_hits = 0
    top3_hits = 0
    selected_utilities: list[float] = []
    regrets: list[float] = []
    selected_actions: list[int] = []
    by_strength: dict[str, Counter[str]] = defaultdict(Counter)
    for state in grouped.values():
        first_index = int(state["first_sample_index"])
        sample = state["first_sample"]
        actions = list(state["action_indices"])  # type: ignore[arg-type]
        utilities = state["utilities"]  # type: ignore[assignment]
        action_scores = {action: float(scores[first_index, action]) for action in actions}
        selected = max(actions, key=lambda action: (action_scores[action], -action))
        selected_actions.append(selected)
        utility_values = {action: float(utilities[action]) for action in actions}
        best_utility = max(utility_values.values())
        teacher_best = {action for action, value in utility_values.items() if value == best_utility}
        selected_utility = utility_values[selected]
        selected_utilities.append(selected_utility)
        regrets.append(best_utility - selected_utility)
        if selected in teacher_best:
            best_hits += 1
        top3 = sorted(actions, key=lambda action: (action_scores[action], -action), reverse=True)[
            :3
        ]
        if teacher_best & set(top3):
            top3_hits += 1
        for left_index, left in enumerate(actions):
            for right in actions[left_index + 1 :]:
                utility_cmp = _compare(utility_values[left], utility_values[right])
                score_cmp = _compare(action_scores[left], action_scores[right])
                pair_count += 1
                if utility_cmp == 0:
                    tie_pair_count += 1
                    continue
                different_pair_count += 1
                if score_cmp == 0:
                    correct += 0.5
                elif score_cmp == utility_cmp:
                    correct += 1.0
        bucket = _strength_bucket(float(sample.strongest_suit_score))
        by_strength[bucket]["total"] += 1
        by_strength[bucket]["pass" if selected == 0 else "bid"] += 1
    state_count = len(grouped)
    action_diagnostics = _selected_action_diagnostics(
        [state["first_sample"] for state in grouped.values()],  # type: ignore[list-item]
        selected_actions,
        by_strength,
    )
    return {
        "stateCount": state_count,
        "pairCount": pair_count,
        "differentPairCount": different_pair_count,
        "teacherTiePairCount": tie_pair_count,
        "teacherTieRate": _safe_div(tie_pair_count, pair_count),
        "pairwiseAccuracy": _safe_div(correct, different_pair_count),
        "bestActionAccuracy": _safe_div(best_hits, state_count),
        "top3ContainsBestRate": _safe_div(top3_hits, state_count),
        "meanSelectedTeacherUtility": float(np.mean(selected_utilities))
        if selected_utilities
        else None,
        "meanRegret": float(np.mean(regrets)) if regrets else None,
        "medianRegret": float(np.median(regrets)) if regrets else None,
        "regretP90": float(np.quantile(regrets, 0.9)) if regrets else None,
        "selectedActions": action_diagnostics,
    }


def _selected_action_diagnostics(
    samples: list[BiddingQRawSample],
    selected_actions: list[int],
    by_strength: dict[str, Counter[str]],
) -> dict[str, Any]:
    total = len(selected_actions)
    pass_count = sum(1 for action in selected_actions if action == 0)
    bid_actions = [action for action in selected_actions if action != 0]
    suit_counts = Counter(action_suit(action) for action in bid_actions)
    target_counts = Counter(str(action_target(action)) for action in bid_actions)
    strongest_matches = 0
    for sample, action in zip(samples, selected_actions, strict=True):
        if action != 0 and action_suit(action) == sample.strongest_suit:
            strongest_matches += 1
    return {
        "passCount": pass_count,
        "bidCount": total - pass_count,
        "passRate": _safe_div(pass_count, total),
        "bidRate": _safe_div(total - pass_count, total),
        "suitDistribution": {str(key): value for key, value in sorted(suit_counts.items())},
        "targetDistribution": dict(sorted(target_counts.items())),
        "strongestSuitBidMatchRate": _safe_div(strongest_matches, len(bid_actions)),
        "handStrengthPassBid": {
            bucket: {
                "stateCount": int(counter["total"]),
                "passRate": _safe_div(counter["pass"], counter["total"]),
                "bidRate": _safe_div(counter["bid"], counter["total"]),
            }
            for bucket, counter in sorted(by_strength.items())
        },
    }


def _baseline_action_index(name: str, state: dict[str, Any]) -> int:
    sample = state["first_sample"]
    assert isinstance(sample, BiddingQRawSample)
    actions = list(state["action_indices"])  # type: ignore[arg-type]
    if name == "behavior":
        return (
            sample.source_selected_action_index
            if sample.source_selected_action_index is not None
            else 0
        )
    if name == "random-legal":
        return actions[int(_stable_int(sample.state_key) % len(actions))]
    if name == "pass-only":
        return 0
    if name == "rule-based-offline":
        bid_limit = _bid_limit_for_score(
            float(sample.strongest_suit_score), threshold=200.0, step=30.0
        )
        return _threshold_bid_or_pass(actions, sample.strongest_suit, bid_limit)
    if name == "conservative-offline":
        bid_limit = _bid_limit_for_score(
            float(sample.strongest_suit_score), threshold=280.0, step=55.0
        )
        needed_target = _needed_target(actions)
        current_target = None if needed_target == 13 else needed_target - 1
        premium = 0.0 if current_target is None else 35.0 + max(0, current_target - 13) * 20.0
        effective = _bid_limit_for_score(
            float(sample.strongest_suit_score) - premium,
            threshold=280.0,
            step=55.0,
        )
        limit = None if bid_limit is None or effective is None else min(bid_limit, effective)
        return _threshold_bid_or_pass(actions, sample.strongest_suit, limit)
    raise ValueError(f"unknown baseline: {name}")


def _threshold_bid_or_pass(
    legal_actions: list[int],
    strongest_suit: SuitName,
    bid_limit: int | None,
) -> int:
    needed_target = _needed_target(legal_actions)
    if bid_limit is None or needed_target > bid_limit:
        return 0
    for action in legal_actions:
        if action_suit(action) == strongest_suit and action_target(action) == needed_target:
            return action
    return 0


def _needed_target(legal_actions: list[int]) -> int:
    targets = [target for action in legal_actions if (target := action_target(action)) is not None]
    return min(targets) if targets else 20


def _bid_limit_for_score(score: float, *, threshold: float, step: float) -> int | None:
    if score < threshold:
        return None
    return min(19, 13 + int(math.floor((score - threshold) / step)))


def _collapse_summary(metrics: Any) -> dict[str, Any]:
    selected = metrics["selectedActions"]  # type: ignore[index]
    pass_rate = selected["passRate"]  # type: ignore[index]
    suit_distribution = selected["suitDistribution"]  # type: ignore[index]
    target_distribution = selected["targetDistribution"]  # type: ignore[index]
    bid_count = selected["bidCount"]  # type: ignore[index]
    top_suit_rate = _top_distribution_rate(suit_distribution, bid_count)
    top_target_rate = _top_distribution_rate(target_distribution, bid_count)
    flags = []
    if pass_rate is not None and (pass_rate < 0.05 or pass_rate > 0.95):
        flags.append("pass_bid_collapse")
    if top_suit_rate is not None and top_suit_rate > 0.8:
        flags.append("suit_collapse")
    if top_target_rate is not None and top_target_rate > 0.8:
        flags.append("target_collapse")
    return {
        "collapsed": bool(flags),
        "flags": flags,
        "passRate": pass_rate,
        "topSuitRateAmongBids": top_suit_rate,
        "topTargetRateAmongBids": top_target_rate,
    }


def _top_distribution_rate(distribution: Any, total: Any) -> float | None:
    if not isinstance(distribution, dict) or not isinstance(total, int) or total <= 0:
        return None
    values = [int(value) for value in distribution.values()]
    return max(values) / total if values else None


def _decomposition_examples(
    samples: tuple[BiddingQRawSample, ...],
    split_scores: dict[str, np.ndarray],
    all_components: dict[ScoreVariant, dict[str, np.ndarray]],
    subset_indices: list[int],
) -> list[dict[str, Any]]:
    if not samples:
        return []
    grouped = _group_state_actions(samples)
    examples: list[dict[str, Any]] = []
    for state in list(grouped.values())[:5]:
        first_index = int(state["first_sample_index"])
        global_index = subset_indices[first_index]
        sample = state["first_sample"]
        actions = list(state["action_indices"])  # type: ignore[arg-type]
        s2_scores = split_scores["S2"]
        selected = max(actions, key=lambda action: (float(s2_scores[first_index, action]), -action))
        components = {
            key: float(value[global_index, selected]) for key, value in all_components["S2"].items()
        }
        examples.append(
            {
                "stateKey": sample.state_key,  # type: ignore[union-attr]
                "selectedActionIndex": selected,
                "selectedAction": _action_label(selected),
                "score": float(s2_scores[first_index, selected]),
                "componentSum": float(sum(components.values())),
                "components": components,
            }
        )
    return examples


def _learning_assessment(final_report: dict[str, Any]) -> dict[str, Any]:
    variants = final_report["variants"]  # type: ignore[index]
    baselines = final_report["baselines"]  # type: ignore[index]
    collapse = final_report["collapse"]  # type: ignore[index]
    s0 = variants["S0"]  # type: ignore[index]
    pass_only = baselines["pass-only"]  # type: ignore[index]
    improved = {}
    for variant_name in ("S1", "S2", "S3"):
        variant = variants[variant_name]  # type: ignore[index]
        improves_napoleon_only = (
            variant["meanSelectedTeacherUtility"] is not None
            and s0["meanSelectedTeacherUtility"] is not None
            and variant["meanRegret"] is not None
            and s0["meanRegret"] is not None
            and variant["meanSelectedTeacherUtility"] >= s0["meanSelectedTeacherUtility"]
            and variant["meanRegret"] <= s0["meanRegret"]
        )
        beats_pass_only = (
            variant["meanSelectedTeacherUtility"] is not None
            and pass_only["meanSelectedTeacherUtility"] is not None
            and variant["meanRegret"] is not None
            and pass_only["meanRegret"] is not None
            and variant["meanSelectedTeacherUtility"] >= pass_only["meanSelectedTeacherUtility"]
            and variant["meanRegret"] <= pass_only["meanRegret"]
        )
        improved[variant_name] = {
            "selectedUtilityDeltaVsS0": _maybe_delta(
                variant["meanSelectedTeacherUtility"],  # type: ignore[index]
                s0["meanSelectedTeacherUtility"],  # type: ignore[index]
            ),
            "regretDeltaVsS0": _maybe_delta(
                variant["meanRegret"],  # type: ignore[index]
                s0["meanRegret"],  # type: ignore[index]
            ),
            "selectedUtilityDeltaVsPassOnly": _maybe_delta(
                variant["meanSelectedTeacherUtility"],  # type: ignore[index]
                pass_only["meanSelectedTeacherUtility"],  # type: ignore[index]
            ),
            "regretDeltaVsPassOnly": _maybe_delta(
                variant["meanRegret"],  # type: ignore[index]
                pass_only["meanRegret"],  # type: ignore[index]
            ),
            "improvesNapoleonOnly": improves_napoleon_only,
            "beatsPassOnly": beats_pass_only,
            "collapsed": bool(collapse[variant_name]["collapsed"]),  # type: ignore[index]
        }
    runtime_ready = any(
        bool(value["improvesNapoleonOnly"])
        and bool(value["beatsPassOnly"])
        and not bool(value["collapsed"])
        for value in improved.values()
    )
    return {
        "primaryBaseline": "S0 Napoleon-only",
        "guardrailBaseline": "pass-only",
        "variantComparison": improved,
        "runtimeConnectionReady": runtime_ready,
        "runtimeConnectionPolicy": "do not connect runtime bidding agent in Issue #390",
    }


def _standardization_from_checkpoint(raw: dict[str, Any], key: str) -> Standardization:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint missing {key}.")
    return Standardization(
        enabled=bool(value.get("enabled")),
        mean=float(value.get("mean", 0.0)),
        std=float(value.get("std", 1.0)),
    )


def _dataset_identity(dataset: BiddingQDataset) -> dict[str, Any]:
    return {
        "path": str(dataset.manifest.dataset_directory),
        "manifestSha256": dataset.manifest.manifest_sha256,
        "sampleType": dataset.manifest.sample_type,
        "sampleSchemaVersion": dataset.manifest.sample_schema_version,
        "sampleCount": dataset.manifest.sample_count,
        "sourceStates": dataset.manifest.source_states,
        "forcedStateActionPairs": dataset.manifest.forced_state_action_pairs,
        "actionMappingId": dataset.manifest.action_mapping_id,
        "rewardId": dataset.manifest.reward_id,
        "modelInputSchemaVersion": dataset.manifest.model_input_schema_version,
        "modelInputFeatureCount": dataset.manifest.model_input_feature_count,
    }


def _variant_descriptions() -> dict[str, str]:
    return {
        "S0": "Napoleon-only margin score.",
        "S1": "Raw P(role | state, action) x role-specific value integration.",
        "S2": "S1 with role-wise affine calibration fitted only on calibration split.",
        "S3": "S2 with Napoleon value replaced by calibrated mu - 0.25 sigma.",
    }


def _action_label(action_index: int) -> str:
    if action_index == 0:
        return "PASS"
    return f"{action_suit(action_index)}-{action_target(action_index)}"


def _stable_split_key(seed: int, state_key: str) -> str:
    return hashlib.sha256(f"{seed}:{state_key}".encode()).hexdigest()


def _stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def _state_key_hash(keys: frozenset[str]) -> str:
    return hashlib.sha256("\n".join(sorted(keys)).encode("utf-8")).hexdigest()


def _compare(left: float, right: float) -> int:
    if left > right:
        return 1
    if left < right:
        return -1
    return 0


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0.0:
        return None
    return 2.0 * precision * recall / (precision + recall)


def _strength_bucket(score: float) -> str:
    if score < 200:
        return "low:<200"
    if score < 280:
        return "medium:200-279"
    if score < 330:
        return "strong:280-329"
    return "very-strong:330+"


def _maybe_delta(value: Any, baseline: Any) -> float | None:
    if isinstance(value, int | float) and isinstance(baseline, int | float):
        return float(value) - float(baseline)
    return None
