"""Offline PASS-binary bidding score evaluation for Issue #395."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from napoleon_ml.dataset.constants import BIDDING_ACTION_COUNT
from napoleon_ml.policy.device import RequestedTorchDevice, resolve_torch_device

from .dataset import BiddingQRawSample, load_bidding_q_dataset
from .margin_training import load_bidding_margin_checkpoint, predict_margin_samples
from .pass_role_binary_training import (
    BiddingPassRoleBinarySplit,
    audit_pass_binary_dataset,
    create_pass_role_binary_split,
    evaluate_pass_role_binary_model,
    load_bidding_pass_role_binary_checkpoint,
    pass_binary_label,
    predict_pass_adjutant_probability,
)
from .role_score_integration import (
    AffineCalibration,
    _collapse_summary,
    _dataset_identity,
    _maybe_delta,
    _standardization_from_checkpoint,
    create_role_score_state_split,
    evaluate_policy_baselines,
    evaluate_score_matrices,
    fit_affine_calibration,
    teacher_utility,
)
from .role_value_training import load_bidding_role_value_checkpoint, predict_role_value_samples

PassSimpleScoreVariant = Literal["T0", "T1", "T2", "T3"]


@dataclass(frozen=True)
class PassSimplePredictions:
    pass_adjutant_probability: np.ndarray
    napoleon_value: np.ndarray
    napoleon_sigma: np.ndarray
    adjutant_value: np.ndarray
    citizen_value: np.ndarray


@dataclass(frozen=True)
class PassSimpleScoreMatrices:
    variants: dict[PassSimpleScoreVariant, np.ndarray]
    components: dict[PassSimpleScoreVariant, dict[str, np.ndarray]]
    calibrations: dict[str, AffineCalibration]


def fit_pass_value_calibrations(
    samples: tuple[BiddingQRawSample, ...],
    *,
    adjutant_value: np.ndarray,
    citizen_value: np.ndarray,
) -> dict[str, AffineCalibration]:
    adj_predicted: list[float] = []
    adj_target: list[float] = []
    citizen_predicted: list[float] = []
    citizen_target: list[float] = []
    for index, sample in enumerate(samples):
        label = pass_binary_label(sample)
        if label == 1:
            adj_predicted.append(float(adjutant_value[index, 0]))
            adj_target.append(teacher_utility(sample))
        elif label == 0:
            citizen_predicted.append(float(citizen_value[index, 0]))
            citizen_target.append(teacher_utility(sample))
    return {
        "adjutant": fit_affine_calibration(
            "pass-adjutant",
            np.asarray(adj_predicted, dtype=np.float64),
            np.asarray(adj_target, dtype=np.float64),
        ),
        "citizen": fit_affine_calibration(
            "pass-citizen",
            np.asarray(citizen_predicted, dtype=np.float64),
            np.asarray(citizen_target, dtype=np.float64),
        ),
    }


def compose_pass_simple_score_matrices(
    predictions: PassSimplePredictions,
    *,
    calibrations: dict[str, AffineCalibration],
    risk_lambda: float = 0.25,
) -> PassSimpleScoreMatrices:
    _validate_predictions(predictions)
    p_adjutant = predictions.pass_adjutant_probability.astype(np.float64)
    p_citizen = 1.0 - p_adjutant
    raw_napoleon = predictions.napoleon_value.astype(np.float64)
    risk_napoleon = raw_napoleon - risk_lambda * predictions.napoleon_sigma.astype(np.float64)
    raw_pass = (
        p_adjutant * predictions.adjutant_value[:, 0].astype(np.float64)
        + p_citizen * predictions.citizen_value[:, 0].astype(np.float64)
    )
    calibrated_pass = (
        p_adjutant * calibrations["adjutant"].apply(predictions.adjutant_value[:, 0])
        + p_citizen * calibrations["citizen"].apply(predictions.citizen_value[:, 0])
    )

    t0 = raw_napoleon.copy()
    t0[:, 0] = 0.0
    t1 = raw_napoleon.copy()
    t1[:, 0] = raw_pass
    t2 = raw_napoleon.copy()
    t2[:, 0] = calibrated_pass
    t3 = risk_napoleon.copy()
    t3[:, 0] = calibrated_pass
    return PassSimpleScoreMatrices(
        variants={"T0": t0, "T1": t1, "T2": t2, "T3": t3},
        components={
            "T0": {"bidNapoleonValue": raw_napoleon, "passConstant": np.zeros_like(raw_pass)},
            "T1": {
                "bidNapoleonValue": raw_napoleon,
                "passAdjutantProbability": p_adjutant,
                "passRawExpectedRoleValue": raw_pass,
            },
            "T2": {
                "bidNapoleonValue": raw_napoleon,
                "passAdjutantProbability": p_adjutant,
                "passCalibratedExpectedRoleValue": calibrated_pass,
            },
            "T3": {
                "bidRiskAwareNapoleonValue": risk_napoleon,
                "passAdjutantProbability": p_adjutant,
                "passCalibratedExpectedRoleValue": calibrated_pass,
            },
        },
        calibrations=calibrations,
    )


def evaluate_pass_simple_score_offline(
    *,
    dataset_directory: Path | str,
    pass_checkpoint_path: Path | str,
    margin_checkpoint_path: Path | str,
    citizen_checkpoint_path: Path | str,
    adjutant_checkpoint_path: Path | str,
    calibration_dataset_directory: Path | str,
    issue390_report_path: Path | str | None = None,
    device: RequestedTorchDevice = "cpu",
    split_seed: int | None = 390,
    verify_integrity: bool = True,
) -> dict[str, Any]:
    dataset = load_bidding_q_dataset(dataset_directory, verify_integrity=verify_integrity)
    calibration_dataset = load_bidding_q_dataset(
        calibration_dataset_directory, verify_integrity=verify_integrity
    )
    resolved_device = resolve_torch_device(device, flag_name="--device")
    all_samples = dataset.raw_samples
    evaluation_split = (
        create_role_score_state_split(all_samples, seed=split_seed)
        if split_seed is not None
        else None
    )
    samples = (
        evaluation_split.select(all_samples, "final")
        if evaluation_split is not None
        else all_samples
    )

    pass_model, pass_raw = load_bidding_pass_role_binary_checkpoint(pass_checkpoint_path)
    margin_model, margin_raw = load_bidding_margin_checkpoint(margin_checkpoint_path)
    citizen_model, citizen_raw = load_bidding_role_value_checkpoint(citizen_checkpoint_path)
    adjutant_model, adjutant_raw = load_bidding_role_value_checkpoint(adjutant_checkpoint_path)
    pass_model.to(resolved_device.torch_device)
    margin_model.to(resolved_device.torch_device)
    citizen_model.to(resolved_device.torch_device)
    adjutant_model.to(resolved_device.torch_device)

    pass_probability = predict_pass_adjutant_probability(
        pass_model, samples, device=resolved_device
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

    calibration_split = _calibration_split_from_checkpoint(
        pass_raw, calibration_dataset.raw_samples
    )
    calibration_samples = calibration_split.calibration_samples
    calibration_citizen_values = predict_role_value_samples(
        citizen_model,
        calibration_samples,
        device=resolved_device,
        standardization=_standardization_from_checkpoint(citizen_raw, "targetStandardization"),
    )
    calibration_adjutant_values = predict_role_value_samples(
        adjutant_model,
        calibration_samples,
        device=resolved_device,
        standardization=_standardization_from_checkpoint(adjutant_raw, "targetStandardization"),
    )
    calibrations = fit_pass_value_calibrations(
        calibration_samples,
        adjutant_value=calibration_adjutant_values,
        citizen_value=calibration_citizen_values,
    )
    predictions = PassSimplePredictions(
        pass_adjutant_probability=pass_probability,
        napoleon_value=margin_predictions["mean"],
        napoleon_sigma=margin_predictions["sigma"],
        adjutant_value=adjutant_values,
        citizen_value=citizen_values,
    )
    scores = compose_pass_simple_score_matrices(predictions, calibrations=calibrations)
    variant_scores: dict[str, np.ndarray] = {
        name: matrix for name, matrix in scores.variants.items()
    }
    variants = evaluate_score_matrices(samples, variant_scores)
    baselines = evaluate_policy_baselines(samples)
    collapse = {name: _collapse_summary(metrics) for name, metrics in variants.items()}
    issue390_comparison = _issue390_comparison(issue390_report_path)
    report: dict[str, Any] = {
        "dataset": _dataset_identity(dataset),
        "calibrationDataset": _dataset_identity(calibration_dataset),
        "artifacts": {
            "passBinaryCheckpoint": str(pass_checkpoint_path),
            "marginCheckpoint": str(margin_checkpoint_path),
            "citizenValueCheckpoint": str(citizen_checkpoint_path),
            "adjutantValueCheckpoint": str(adjutant_checkpoint_path),
        },
        "device": resolved_device.to_metadata(),
        "evaluation": {
            "mode": "role-score-final-split" if evaluation_split is not None else "all-states",
            "sampleCount": len(samples),
            "stateCount": len({sample.state_key for sample in samples}),
            "split": evaluation_split.to_dict(all_samples)
            if evaluation_split is not None
            else None,
        },
        "passBinary": {
            "auditTrainingDataset": pass_raw.get("trainingDatasetAudit"),
            "auditFinalDataset": audit_pass_binary_dataset(samples),
            "finalMetrics": evaluate_pass_role_binary_model(
                pass_model, samples, device=resolved_device
            ),
            "noContractConclusion": _no_contract_conclusion(audit_pass_binary_dataset(samples)),
            "calibrationSplit": {
                "trainStates": len(calibration_split.train_state_keys),
                "calibrationStates": len(calibration_split.calibration_state_keys),
                "trainSamples": len(calibration_split.train_samples),
                "calibrationSamples": len(calibration_split.calibration_samples),
                "calibrationStateKeyHash": calibration_split.state_key_hash,
                "stateKeyLeakage": False,
            },
        },
        "calibration": {role: calibration.to_dict() for role, calibration in calibrations.items()},
        "scoreVariants": _variant_descriptions(),
        "variants": variants,
        "baselines": baselines,
        "collapse": collapse,
        "issue390Comparison": issue390_comparison,
        "decompositionExamples": _decomposition_examples(samples, scores),
    }
    report["learningAssessment"] = _learning_assessment(report)
    return report


def _calibration_split_from_checkpoint(
    checkpoint: dict[str, Any], samples: tuple[BiddingQRawSample, ...]
) -> BiddingPassRoleBinarySplit:
    training_config = checkpoint.get("trainingConfig")
    if not isinstance(training_config, dict):
        raise ValueError("PASS checkpoint missing trainingConfig.")
    return create_pass_role_binary_split(
        samples,
        train_ratio=float(training_config.get("train_ratio", 0.75)),
        seed=int(training_config.get("seed", 395)),
    )


def _validate_predictions(predictions: PassSimplePredictions) -> None:
    sample_count = predictions.napoleon_value.shape[0]
    for name, matrix in (
        ("napoleon_value", predictions.napoleon_value),
        ("napoleon_sigma", predictions.napoleon_sigma),
        ("adjutant_value", predictions.adjutant_value),
        ("citizen_value", predictions.citizen_value),
    ):
        if matrix.shape != (sample_count, BIDDING_ACTION_COUNT):
            raise ValueError(f"{name} must have shape (sample, {BIDDING_ACTION_COUNT}).")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} must be finite.")
    probability = predictions.pass_adjutant_probability
    if probability.shape != (sample_count,):
        raise ValueError("pass_adjutant_probability must have shape (sample,).")
    if not np.all(np.isfinite(probability)):
        raise ValueError("pass_adjutant_probability must be finite.")
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise ValueError("pass_adjutant_probability must be in [0, 1].")


def _variant_descriptions() -> dict[str, str]:
    return {
        "T0": "BID actions use raw V_Napoleon; PASS score is 0.",
        "T1": "BID actions use raw V_Napoleon; PASS uses P(Adj)*V_Adj + P(Citizen)*V_Citizen.",
        "T2": (
            "T1 with PASS role values affine-calibrated on the PASS binary calibration split only."
        ),
        "T3": "T2 PASS with risk-aware BID score V_Napoleon mean - 0.25 sigma.",
    }


def _no_contract_conclusion(audit: dict[str, Any]) -> dict[str, Any]:
    rate = audit.get("noContractRate")
    requires = bool(audit.get("noContractRequiresSeparateHead"))
    return {
        "maskedFromBinaryTeacher": True,
        "rate": rate,
        "requiresSeparateHead": requires,
        "conclusion": (
            "NoContract is not negligible; a separate PASS NoContract head is needed."
            if requires
            else "NoContract frequency is low enough for Issue #395 binary mask treatment."
        ),
    }


def _issue390_comparison(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    content = json.loads(Path(path).read_text(encoding="utf-8"))
    final = content.get("splits", {}).get("final", {})
    variants = final.get("variants", {})
    if not isinstance(variants, dict) or not variants:
        return {"path": str(path), "available": False}
    integrated = {
        name: value for name, value in variants.items() if name in {"S1", "S2", "S3"}
    }
    if not integrated:
        return {"path": str(path), "available": False}
    best_name = max(
        integrated,
        key=lambda name: (
            integrated[name].get("meanSelectedTeacherUtility")
            if integrated[name].get("meanSelectedTeacherUtility") is not None
            else float("-inf"),
            -(
                integrated[name].get("meanRegret")
                if integrated[name].get("meanRegret") is not None
                else float("inf")
            ),
        ),
    )
    return {
        "path": str(path),
        "available": True,
        "s0": _compact_policy_metrics(variants.get("S0", {})),
        "bestIntegratedName": best_name,
        "bestIntegrated": _compact_policy_metrics(integrated[best_name]),
    }


def _learning_assessment(report: dict[str, Any]) -> dict[str, Any]:
    variants = report["variants"]
    collapse = report["collapse"]
    issue390 = report.get("issue390Comparison")
    t0 = variants["T0"]
    best_name = max(
        ("T1", "T2", "T3"),
        key=lambda name: (
            variants[name]["meanSelectedTeacherUtility"]
            if variants[name]["meanSelectedTeacherUtility"] is not None
            else float("-inf"),
            -(
                variants[name]["meanRegret"]
                if variants[name]["meanRegret"] is not None
                else float("inf")
            ),
        ),
    )
    best = variants[best_name]
    best_vs_t0 = {
        "selectedUtilityDelta": _maybe_delta(
            best["meanSelectedTeacherUtility"], t0["meanSelectedTeacherUtility"]
        ),
        "regretDelta": _maybe_delta(best["meanRegret"], t0["meanRegret"]),
    }
    best_vs_issue390 = None
    if isinstance(issue390, dict) and issue390.get("available"):
        prior = issue390["bestIntegrated"]
        best_vs_issue390 = {
            "selectedUtilityDelta": _maybe_delta(
                best["meanSelectedTeacherUtility"], prior["meanSelectedTeacherUtility"]
            ),
            "regretDelta": _maybe_delta(best["meanRegret"], prior["meanRegret"]),
            "baseline": issue390["bestIntegratedName"],
        }
    t0_very_strong = _very_strong_bid_rate(t0)
    best_very_strong = _very_strong_bid_rate(best)
    very_strong_ok = (
        best_very_strong is not None
        and t0_very_strong is not None
        and best_very_strong >= max(0.8, t0_very_strong - 0.05)
    )
    improves_t0 = (
        best_vs_t0["selectedUtilityDelta"] is not None
        and best_vs_t0["selectedUtilityDelta"] >= 0.0
        and best_vs_t0["regretDelta"] is not None
        and best_vs_t0["regretDelta"] <= 0.0
    )
    improves_issue390 = (
        best_vs_issue390 is not None
        and best_vs_issue390["selectedUtilityDelta"] is not None
        and best_vs_issue390["selectedUtilityDelta"] >= 0.0
        and best_vs_issue390["regretDelta"] is not None
        and best_vs_issue390["regretDelta"] <= 0.0
    )
    runtime_ready = (
        improves_t0
        and (best_vs_issue390 is None or improves_issue390)
        and very_strong_ok
        and not bool(collapse[best_name]["collapsed"])
    )
    return {
        "bestVariant": best_name,
        "bestVsT0": best_vs_t0,
        "bestVsIssue390": best_vs_issue390,
        "veryStrongHandBidRate": best_very_strong,
        "veryStrongHandBidGuardrailPassed": very_strong_ok,
        "collapseFree": not bool(collapse[best_name]["collapsed"]),
        "runtimeConnectionReady": runtime_ready,
        "runtimeConnectionPolicy": "do not connect runtime bidding agent in Issue #395",
    }


def _very_strong_bid_rate(metrics: dict[str, Any]) -> float | None:
    hand_strength = metrics["selectedActions"]["handStrengthPassBid"]
    very_strong = hand_strength.get("very-strong:330+")
    if not isinstance(very_strong, dict):
        return None
    value = very_strong.get("bidRate")
    return float(value) if isinstance(value, int | float) else None


def _compact_policy_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    selected = metrics.get("selectedActions", {})
    return {
        "pairwiseAccuracy": metrics.get("pairwiseAccuracy"),
        "bestActionAccuracy": metrics.get("bestActionAccuracy"),
        "top3ContainsBestRate": metrics.get("top3ContainsBestRate"),
        "differentPairCount": metrics.get("differentPairCount"),
        "meanSelectedTeacherUtility": metrics.get("meanSelectedTeacherUtility"),
        "meanRegret": metrics.get("meanRegret"),
        "passRate": selected.get("passRate") if isinstance(selected, dict) else None,
        "bidRate": selected.get("bidRate") if isinstance(selected, dict) else None,
        "handStrengthPassBid": selected.get("handStrengthPassBid")
        if isinstance(selected, dict)
        else None,
    }


def _decomposition_examples(
    samples: tuple[BiddingQRawSample, ...],
    scores: PassSimpleScoreMatrices,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, sample in enumerate(samples):
        if sample.state_key in seen:
            continue
        seen.add(sample.state_key)
        legal_actions = [
            action_index
            for action_index, is_legal in enumerate(sample.legal_bid_mask)
            if is_legal == 1.0
        ]
        selected = max(
            legal_actions,
            key=lambda action: (float(scores.variants["T2"][index, action]), -action),
        )
        examples.append(
            {
                "stateKey": sample.state_key,
                "selectedActionIndex": selected,
                "selectedScoreT2": float(scores.variants["T2"][index, selected]),
                "passScoreT2": float(scores.variants["T2"][index, 0]),
                "passPAdjutant": float(scores.components["T2"]["passAdjutantProbability"][index]),
                "passCalibratedExpectedRoleValue": float(
                    scores.components["T2"]["passCalibratedExpectedRoleValue"][index]
                ),
                "bestBidScoreT2": max(
                    float(scores.variants["T2"][index, action])
                    for action in legal_actions
                    if action != 0
                )
                if any(action != 0 for action in legal_actions)
                else None,
            }
        )
        if len(examples) >= 5:
            break
    return examples
