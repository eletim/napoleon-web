"""Reward-expectation bidding score evaluation for Issue #400."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from napoleon_ml.dataset.constants import BIDDING_ACTION_COUNT
from napoleon_ml.policy.device import RequestedTorchDevice, resolve_torch_device

from .dataset import (
    BiddingQRawSample,
    action_suit,
    action_target,
    load_bidding_q_dataset,
)
from .margin_training import (
    gaussian_success_probability,
    load_bidding_margin_checkpoint,
    predict_margin_samples,
)
from .multi_head_training import (
    _binary_auc,
    _calibration_bins,
    _pr_auc,
    _safe_div,
    load_bidding_multi_head_q_checkpoint,
    predict_multi_head_samples,
)
from .pass_role_binary_training import (
    audit_pass_binary_dataset,
    evaluate_pass_role_binary_model,
    load_bidding_pass_role_binary_checkpoint,
    predict_pass_adjutant_probability,
)
from .role_margin_training import (
    evaluate_bidding_role_margin_model,
    load_bidding_role_margin_checkpoint,
    predict_role_margin_samples,
)
from .role_score_integration import (
    _collapse_summary,
    _dataset_identity,
    _standardization_from_checkpoint,
    create_role_score_state_split,
)
from .role_training import canonical_bidding_role
from .role_value_training import load_bidding_role_value_checkpoint, predict_role_value_samples

RewardScoreVariant = Literal["R0", "R1", "R2"]


@dataclass(frozen=True)
class RoleWinPredictions:
    napoleon_win_probability: np.ndarray
    adjutant_win_probability: np.ndarray
    citizen_napoleon_win_probability: np.ndarray
    pass_adjutant_probability: np.ndarray


def reward_d_for_action(sample: BiddingQRawSample, action_index: int) -> int:
    if action_index == 0:
        if sample.final_declared_target is None:
            return 0
        return int(sample.final_declared_target)
    target = action_target(action_index)
    if target is None:
        raise ValueError(f"action {action_index} has no target.")
    return int(target)


def napoleon_positive_ev_threshold(d: int) -> float | None:
    if d <= 0:
        return None
    return 5.0 / (2.0 * d + 5.0)


def role_expected_reward(
    *,
    role: str,
    napoleon_side_win_probability: np.ndarray | float,
    d: np.ndarray | float,
) -> np.ndarray | float:
    p = napoleon_side_win_probability
    if role == "napoleon":
        return np.asarray(p) * (2.0 * np.asarray(d)) + (1.0 - np.asarray(p)) * -5.0
    if role == "napoleon-adjutant":
        return np.asarray(p) * (3.0 * np.asarray(d)) + (1.0 - np.asarray(p)) * -5.0
    if role == "adjutant":
        return np.asarray(p) * np.asarray(d)
    if role == "citizen":
        return (1.0 - np.asarray(p)) * np.asarray(d)
    if role == "no-contract":
        return np.zeros_like(np.asarray(p), dtype=np.float64)
    raise ValueError(f"unsupported reward role: {role}")


def realized_role_reward(sample: BiddingQRawSample) -> float:
    role = canonical_bidding_role(sample.final_role)
    if role == "no-contract" or sample.final_declared_target is None:
        return 0.0
    d = float(sample.final_declared_target)
    napoleon_won = bool(sample.contract_success)
    if role == "napoleon":
        return 2.0 * d if napoleon_won else -5.0
    if role == "napoleon-adjutant":
        return 3.0 * d if napoleon_won else -5.0
    if role == "adjutant":
        return d if napoleon_won else 0.0
    if role == "citizen":
        return 0.0 if napoleon_won else d
    return 0.0


def compose_reward_score_matrices(
    samples: tuple[BiddingQRawSample, ...],
    *,
    predictions: RoleWinPredictions,
    r0_score: np.ndarray | None,
    explicit_napoleon_win_probability: np.ndarray | None = None,
    pass_terminal_role_prior: dict[str, dict[str, float]] | None = None,
) -> dict[str, np.ndarray]:
    d_matrix = _difficulty_matrix(samples)
    pass_context = _expand_pass_terminal_role_prior(
        pass_terminal_role_prior or _default_pass_terminal_role_prior(), len(samples)
    )
    r1 = np.asarray(
        role_expected_reward(
            role="napoleon",
            napoleon_side_win_probability=predictions.napoleon_win_probability,
            d=d_matrix,
        ),
        dtype=np.float64,
    )
    conditional_citizen_adjutant_ev = (
        predictions.pass_adjutant_probability
        * role_expected_reward(
            role="adjutant",
            napoleon_side_win_probability=predictions.adjutant_win_probability[:, 0],
            d=pass_context["d"]["adjutant"],
        )
        + (1.0 - predictions.pass_adjutant_probability)
        * role_expected_reward(
            role="citizen",
            napoleon_side_win_probability=predictions.citizen_napoleon_win_probability[:, 0],
            d=pass_context["d"]["citizen"],
        )
    )
    pass_ev = (
        (pass_context["masses"]["adjutant"] + pass_context["masses"]["citizen"])
        * conditional_citizen_adjutant_ev
        + pass_context["masses"]["napoleon"]
        * role_expected_reward(
            role="napoleon",
            napoleon_side_win_probability=predictions.napoleon_win_probability[:, 0],
            d=pass_context["d"]["napoleon"],
        )
        + pass_context["masses"]["napoleon-adjutant"]
        * role_expected_reward(
            role="napoleon-adjutant",
            napoleon_side_win_probability=predictions.napoleon_win_probability[:, 0],
            d=pass_context["d"]["napoleon-adjutant"],
        )
    )
    r1[:, 0] = pass_ev
    scores: dict[str, np.ndarray] = {"R1": r1.astype(np.float64)}
    if r0_score is not None:
        scores["R0"] = r0_score.astype(np.float64)
    if explicit_napoleon_win_probability is not None:
        r2 = np.asarray(
            role_expected_reward(
                role="napoleon",
                napoleon_side_win_probability=explicit_napoleon_win_probability,
                d=d_matrix,
            ),
            dtype=np.float64,
        )
        r2_pass_ev = (
            (pass_context["masses"]["adjutant"] + pass_context["masses"]["citizen"])
            * conditional_citizen_adjutant_ev
            + pass_context["masses"]["napoleon"]
            * role_expected_reward(
                role="napoleon",
                napoleon_side_win_probability=explicit_napoleon_win_probability[:, 0],
                d=pass_context["d"]["napoleon"],
            )
            + pass_context["masses"]["napoleon-adjutant"]
            * role_expected_reward(
                role="napoleon-adjutant",
                napoleon_side_win_probability=explicit_napoleon_win_probability[:, 0],
                d=pass_context["d"]["napoleon-adjutant"],
            )
        )
        r2[:, 0] = r2_pass_ev
        scores["R2"] = r2.astype(np.float64)
    return scores


def evaluate_reward_expectation_offline(
    *,
    dataset_directory: Path | str,
    pass_checkpoint_path: Path | str,
    napoleon_margin_checkpoint_path: Path | str,
    citizen_margin_checkpoint_path: Path | str,
    adjutant_margin_checkpoint_path: Path | str,
    citizen_value_checkpoint_path: Path | str | None = None,
    adjutant_value_checkpoint_path: Path | str | None = None,
    explicit_success_checkpoint_path: Path | str | None = None,
    issue395_report_path: Path | str | None = None,
    device: RequestedTorchDevice = "cpu",
    split_seed: int | None = 390,
    verify_integrity: bool = True,
) -> dict[str, Any]:
    dataset = load_bidding_q_dataset(dataset_directory, verify_integrity=verify_integrity)
    resolved_device = resolve_torch_device(device, flag_name="--device")
    all_samples = dataset.raw_samples
    split = (
        create_role_score_state_split(all_samples, seed=split_seed)
        if split_seed is not None
        else None
    )
    samples = split.select(all_samples, "final") if split is not None else all_samples

    pass_model, _pass_raw = load_bidding_pass_role_binary_checkpoint(pass_checkpoint_path)
    napoleon_model, napoleon_raw = load_bidding_margin_checkpoint(napoleon_margin_checkpoint_path)
    citizen_model, citizen_raw = load_bidding_role_margin_checkpoint(citizen_margin_checkpoint_path)
    adjutant_model, adjutant_raw = load_bidding_role_margin_checkpoint(
        adjutant_margin_checkpoint_path
    )
    _validate_role_margin_checkpoint_role(
        citizen_raw, expected_role="citizen", path=citizen_margin_checkpoint_path
    )
    _validate_role_margin_checkpoint_role(
        adjutant_raw, expected_role="adjutant", path=adjutant_margin_checkpoint_path
    )
    pass_model.to(resolved_device.torch_device)
    napoleon_model.to(resolved_device.torch_device)
    citizen_model.to(resolved_device.torch_device)
    adjutant_model.to(resolved_device.torch_device)

    pass_q = predict_pass_adjutant_probability(pass_model, samples, device=resolved_device)
    napoleon_pred = predict_margin_samples(
        napoleon_model,
        samples,
        device=resolved_device,
        standardization=_standardization_from_checkpoint(napoleon_raw, "targetStandardization"),
    )
    citizen_pred = predict_role_margin_samples(
        citizen_model,
        samples,
        device=resolved_device,
        standardization=_standardization_from_checkpoint(citizen_raw, "targetStandardization"),
    )
    adjutant_pred = predict_role_margin_samples(
        adjutant_model,
        samples,
        device=resolved_device,
        standardization=_standardization_from_checkpoint(adjutant_raw, "targetStandardization"),
    )
    role_win = RoleWinPredictions(
        napoleon_win_probability=gaussian_success_probability(
            napoleon_pred["mean"], napoleon_pred["sigma"]
        ),
        adjutant_win_probability=gaussian_success_probability(
            adjutant_pred["mean"], adjutant_pred["sigma"]
        ),
        citizen_napoleon_win_probability=gaussian_success_probability(
            citizen_pred["mean"], citizen_pred["sigma"]
        ),
        pass_adjutant_probability=pass_q,
    )
    explicit_success = _explicit_success_probabilities(
        explicit_success_checkpoint_path, samples, resolved_device
    )
    pass_prior_samples = split.select(all_samples, "calibration") if split is not None else ()
    pass_terminal_role_prior = (
        _pass_terminal_role_prior(pass_prior_samples)
        if pass_prior_samples
        else _default_pass_terminal_role_prior()
    )
    r0 = _r0_value_score(
        samples,
        pass_q=pass_q,
        napoleon_margin_mean=napoleon_pred["mean"],
        citizen_value_checkpoint_path=citizen_value_checkpoint_path,
        adjutant_value_checkpoint_path=adjutant_value_checkpoint_path,
        device=resolved_device,
    )
    scores = compose_reward_score_matrices(
        samples,
        predictions=role_win,
        r0_score=r0,
        explicit_napoleon_win_probability=explicit_success,
        pass_terminal_role_prior=pass_terminal_role_prior,
    )
    variants = evaluate_reward_score_matrices(samples, scores)
    baselines = evaluate_reward_policy_baselines(samples)
    collapse = {name: _collapse_summary(metrics) for name, metrics in variants.items()}
    return {
        "dataset": _dataset_identity(dataset),
        "evaluation": {
            "mode": "role-score-final-split" if split is not None else "all-states",
            "sampleCount": len(samples),
            "stateCount": len({sample.state_key for sample in samples}),
            "split": split.to_dict(all_samples) if split is not None else None,
        },
        "artifacts": {
            "passBinaryCheckpoint": str(pass_checkpoint_path),
            "napoleonMarginCheckpoint": str(napoleon_margin_checkpoint_path),
            "citizenMarginCheckpoint": str(citizen_margin_checkpoint_path),
            "adjutantMarginCheckpoint": str(adjutant_margin_checkpoint_path),
            "citizenValueCheckpoint": str(citizen_value_checkpoint_path)
            if citizen_value_checkpoint_path is not None
            else None,
            "adjutantValueCheckpoint": str(adjutant_value_checkpoint_path)
            if adjutant_value_checkpoint_path is not None
            else None,
            "explicitSuccessCheckpoint": str(explicit_success_checkpoint_path)
            if explicit_success_checkpoint_path is not None
            else None,
        },
        "device": resolved_device.to_metadata(),
        "rewardSemantics": reward_semantics_report(samples),
        "passTerminalRolePrior": pass_terminal_role_prior,
        "passBinary": {
            "audit": audit_pass_binary_dataset(samples),
            "metrics": evaluate_pass_role_binary_model(pass_model, samples, device=resolved_device),
        },
        "roleMargin": {
            "citizen": _role_margin_eval_from_checkpoint(
                citizen_model, citizen_raw, samples, resolved_device
            ),
            "adjutant": _role_margin_eval_from_checkpoint(
                adjutant_model, adjutant_raw, samples, resolved_device
            ),
        },
        "roleWinProbability": {
            "napoleon": win_probability_metrics_for_roles(
                samples,
                roles=("napoleon", "napoleon-adjutant"),
                probability_matrix=role_win.napoleon_win_probability,
            ),
            "citizenNapoleonSide": win_probability_metrics_for_roles(
                samples,
                roles=("citizen",),
                probability_matrix=role_win.citizen_napoleon_win_probability,
            ),
            "adjutant": win_probability_metrics_for_roles(
                samples,
                roles=("adjutant",),
                probability_matrix=role_win.adjutant_win_probability,
            ),
            "explicitNapoleonAvailable": explicit_success is not None,
        },
        "scoreVariants": _variant_descriptions(explicit_success is not None),
        "variants": variants,
        "baselines": baselines,
        "collapse": collapse,
        "rewardExpectationDiagnostics": reward_expectation_diagnostics(
            samples, scores, role_win
        ),
        "issue395Comparison": _issue395_comparison(issue395_report_path),
        "learningAssessment": _learning_assessment(variants, baselines, collapse),
    }


def evaluate_reward_score_matrices(
    samples: tuple[BiddingQRawSample, ...],
    score_matrices: dict[str, np.ndarray],
) -> dict[str, Any]:
    grouped = _group_state_actions(samples)
    return {
        name: _evaluate_named_reward_score(grouped, scores)
        for name, scores in score_matrices.items()
    }


def evaluate_reward_policy_baselines(samples: tuple[BiddingQRawSample, ...]) -> dict[str, Any]:
    grouped = _group_state_actions(samples)
    result = {}
    for name in ("behavior", "pass-only", "conservative-offline", "rule-based-offline"):
        scores = np.zeros((len(samples), BIDDING_ACTION_COUNT), dtype=np.float64)
        for state in grouped.values():
            selected = _baseline_action_index(name, state)
            scores[int(state["first_sample_index"]), selected] = 1.0
        result[name] = _evaluate_named_reward_score(grouped, scores)
    return result


def role_win_probability_metrics(
    samples: tuple[BiddingQRawSample, ...],
    *,
    role: Literal["citizen", "adjutant"],
    probability_matrix: np.ndarray,
) -> dict[str, Any]:
    return win_probability_metrics_for_roles(
        samples, roles=(role,), probability_matrix=probability_matrix
    )


def win_probability_metrics_for_roles(
    samples: tuple[BiddingQRawSample, ...],
    *,
    roles: tuple[str, ...],
    probability_matrix: np.ndarray,
) -> dict[str, Any]:
    rows: list[tuple[BiddingQRawSample, float]] = []
    for index, sample in enumerate(samples):
        if (
            canonical_bidding_role(sample.final_role) not in roles
            or sample.result_type != "standard"
        ):
            continue
        rows.append((sample, float(probability_matrix[index, sample.forced_action_index])))
    truth = np.asarray([1.0 if sample.contract_success else 0.0 for sample, _ in rows])
    prob = np.asarray([probability for _, probability in rows], dtype=np.float64)
    return {
        "sampleCount": int(truth.size),
        "stateCount": len({sample.state_key for sample, _ in rows}),
        "positiveRate": float(np.mean(truth)) if truth.size else None,
        "rocAuc": _binary_auc(truth, prob),
        "prAuc": _pr_auc(truth, prob),
        "brier": float(np.mean((prob - truth) ** 2)) if truth.size else None,
        "logLoss": _log_loss(truth, prob),
        "calibration": _calibration_bins(truth, prob),
        "pairwiseSuccessRanking": _success_pairwise(tuple(sample for sample, _ in rows), prob),
    }


def reward_semantics_report(samples: tuple[BiddingQRawSample, ...]) -> dict[str, Any]:
    ds = [reward_d_for_action(sample, sample.forced_action_index) for sample in samples]
    thresholds = {
        str(d): napoleon_positive_ev_threshold(d)
        for d in sorted({value for value in ds if value > 0})
    }
    return {
        "dSemantics": (
            "d is targetPointCards. BID uses the bid action target; PASS/teacher uses the "
            "terminal finalDeclaredTarget when a contract exists, otherwise 0 for NoContract."
        ),
        "rewardTable": {
            "napoleon": "win=+2d, loss=-5",
            "adjutant": "win=+d, loss=0",
            "citizen": "Napoleon win=0, Napoleon loss=+d",
            "noContract": 0,
        },
        "napoleonPositiveEvWinRateThresholdByD": thresholds,
    }


def reward_expectation_diagnostics(
    samples: tuple[BiddingQRawSample, ...],
    scores: dict[str, np.ndarray],
    role_win: RoleWinPredictions,
) -> dict[str, Any]:
    r1 = scores["R1"]
    legal_bid_scores: list[float] = []
    pass_scores: list[float] = []
    p_bid: list[float] = []
    for index, sample in enumerate(samples):
        if sample.forced_action_index == 0:
            pass_scores.append(float(r1[index, 0]))
        for action_index, is_legal in enumerate(sample.legal_bid_mask):
            if action_index != 0 and is_legal == 1.0:
                legal_bid_scores.append(float(r1[index, action_index]))
                p_bid.append(float(role_win.napoleon_win_probability[index, action_index]))
    return {
        "R1PassEv": _distribution(np.asarray(pass_scores, dtype=np.float64)),
        "R1BidEv": _distribution(np.asarray(legal_bid_scores, dtype=np.float64)),
        "napoleonBidWinProbability": _distribution(np.asarray(p_bid, dtype=np.float64)),
    }


def _evaluate_named_reward_score(
    grouped: dict[str, dict[str, Any]], scores: np.ndarray
) -> dict[str, Any]:
    pair_count = 0
    different_pair_count = 0
    tie_pair_count = 0
    correct = 0.0
    best_hits = 0
    top3_hits = 0
    selected_rewards: list[float] = []
    regrets: list[float] = []
    selected_actions: list[int] = []
    selected_positive_expected = 0
    by_strength: dict[str, Counter[str]] = defaultdict(Counter)
    by_current_bid: dict[str, Counter[str]] = defaultdict(Counter)
    for state in grouped.values():
        first_index = int(state["first_sample_index"])
        sample = state["first_sample"]
        actions = list(state["action_indices"])
        rewards = state["rewards"]
        action_scores = {action: float(scores[first_index, action]) for action in actions}
        selected = max(actions, key=lambda action: (action_scores[action], -action))
        selected_actions.append(selected)
        if action_scores[selected] > 0.0:
            selected_positive_expected += 1
        best_reward = max(float(rewards[action]) for action in actions)
        teacher_best = {action for action in actions if float(rewards[action]) == best_reward}
        selected_reward = float(rewards[selected])
        selected_rewards.append(selected_reward)
        regrets.append(best_reward - selected_reward)
        if selected in teacher_best:
            best_hits += 1
        top3 = sorted(actions, key=lambda action: (action_scores[action], -action), reverse=True)[
            :3
        ]
        if teacher_best & set(top3):
            top3_hits += 1
        for left_index, left in enumerate(actions):
            for right in actions[left_index + 1 :]:
                reward_cmp = _compare(float(rewards[left]), float(rewards[right]))
                score_cmp = _compare(action_scores[left], action_scores[right])
                pair_count += 1
                if reward_cmp == 0:
                    tie_pair_count += 1
                    continue
                different_pair_count += 1
                if score_cmp == 0:
                    correct += 0.5
                elif score_cmp == reward_cmp:
                    correct += 1.0
        strength = _strength_bucket(float(sample.strongest_suit_score))
        by_strength[strength]["total"] += 1
        by_strength[strength]["pass" if selected == 0 else "bid"] += 1
        current_bid = _current_bid_level(sample)
        by_current_bid[current_bid]["total"] += 1
        by_current_bid[current_bid]["pass" if selected == 0 else "bid"] += 1
    state_count = len(grouped)
    return {
        "stateCount": state_count,
        "pairCount": pair_count,
        "differentPairCount": different_pair_count,
        "teacherTiePairCount": tie_pair_count,
        "teacherTieRate": _safe_div(tie_pair_count, pair_count),
        "pairwiseAccuracy": _safe_div(correct, different_pair_count),
        "bestActionAccuracy": _safe_div(best_hits, state_count),
        "top3ContainsBestRate": _safe_div(top3_hits, state_count),
        "meanSelectedRealizedReward": float(np.mean(selected_rewards))
        if selected_rewards
        else None,
        "meanRegret": float(np.mean(regrets)) if regrets else None,
        "positiveExpectedRewardSelectionRate": _safe_div(selected_positive_expected, state_count),
        "selectedActions": _selected_action_diagnostics(
            [state["first_sample"] for state in grouped.values()],
            selected_actions,
            by_strength,
            by_current_bid,
        ),
    }


def _selected_action_diagnostics(
    samples: list[BiddingQRawSample],
    selected_actions: list[int],
    by_strength: dict[str, Counter[str]],
    by_current_bid: dict[str, Counter[str]],
) -> dict[str, Any]:
    total = len(selected_actions)
    pass_count = sum(1 for action in selected_actions if action == 0)
    bid_actions = [action for action in selected_actions if action != 0]
    suit_counts = Counter(action_suit(action) for action in bid_actions)
    target_counts = Counter(str(action_target(action)) for action in bid_actions)
    strongest_matches = sum(
        1
        for sample, action in zip(samples, selected_actions, strict=True)
        if action != 0 and action_suit(action) == sample.strongest_suit
    )
    targets = [int(action_target(action) or 0) for action in bid_actions]
    return {
        "passCount": pass_count,
        "bidCount": total - pass_count,
        "passRate": _safe_div(pass_count, total),
        "bidRate": _safe_div(total - pass_count, total),
        "suitDistribution": {str(key): value for key, value in sorted(suit_counts.items())},
        "targetDistribution": dict(sorted(target_counts.items())),
        "averageTarget": float(np.mean(targets)) if targets else None,
        "strongestSuitBidMatchRate": _safe_div(strongest_matches, len(bid_actions)),
        "handStrengthBidRate": _bid_rate_table(by_strength),
        "currentBidLevelBidRate": _bid_rate_table(by_current_bid),
    }


def _group_state_actions(samples: tuple[BiddingQRawSample, ...]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    reward_accumulator: dict[tuple[str, int], list[float]] = defaultdict(list)
    for index, sample in enumerate(samples):
        reward_accumulator[(sample.state_key, sample.forced_action_index)].append(
            realized_role_reward(sample)
        )
        if sample.state_key not in grouped:
            legal_actions = [
                action_index
                for action_index, is_legal in enumerate(sample.legal_bid_mask)
                if is_legal == 1.0
            ]
            grouped[sample.state_key] = {
                "first_sample": sample,
                "first_sample_index": index,
                "action_indices": legal_actions,
                "rewards": {},
            }
    for (state_key, action_index), rewards in reward_accumulator.items():
        grouped[state_key]["rewards"][action_index] = float(np.mean(rewards))
    for state in grouped.values():
        present = set(state["rewards"].keys())
        state["action_indices"] = [
            action for action in state["action_indices"] if action in present
        ]
    return grouped


def _difficulty_matrix(samples: tuple[BiddingQRawSample, ...]) -> np.ndarray:
    matrix = np.zeros((len(samples), BIDDING_ACTION_COUNT), dtype=np.float64)
    pass_d_accumulator: dict[str, list[int]] = defaultdict(list)
    for sample in samples:
        if sample.forced_action_index == 0:
            pass_d_accumulator[sample.state_key].append(reward_d_for_action(sample, 0))
    pass_d_by_state = {
        state_key: float(np.mean(values))
        for state_key, values in pass_d_accumulator.items()
    }
    for index, sample in enumerate(samples):
        for action_index in range(BIDDING_ACTION_COUNT):
            matrix[index, action_index] = (
                pass_d_by_state.get(sample.state_key, 0)
                if action_index == 0
                else reward_d_for_action(sample, action_index)
            )
    return matrix


def _pass_terminal_role_context(
    samples: tuple[BiddingQRawSample, ...],
) -> dict[str, dict[str, np.ndarray]]:
    return _expand_pass_terminal_role_prior(_pass_terminal_role_prior(samples), len(samples))


def _default_pass_terminal_role_prior() -> dict[str, dict[str, float]]:
    return {
        "masses": {
            "adjutant": 1.0,
            "citizen": 0.0,
            "napoleon": 0.0,
            "napoleon-adjutant": 0.0,
            "no-contract": 0.0,
        },
        "d": {
            "adjutant": 13.0,
            "citizen": 13.0,
            "napoleon": 13.0,
            "napoleon-adjutant": 13.0,
            "no-contract": 0.0,
        },
    }


def _pass_terminal_role_prior(
    samples: tuple[BiddingQRawSample, ...],
) -> dict[str, dict[str, float]]:
    role_counts: Counter[str] = Counter()
    role_d: dict[str, list[int]] = defaultdict(list)
    for sample in samples:
        if sample.forced_action_index == 0:
            role = canonical_bidding_role(sample.final_role)
            role_counts[role] += 1
            role_d[role].append(reward_d_for_action(sample, 0))
    roles = ("adjutant", "citizen", "napoleon", "napoleon-adjutant", "no-contract")
    total = sum(role_counts.values())
    if total <= 0:
        return _default_pass_terminal_role_prior()
    all_d = [value for values in role_d.values() for value in values]
    all_contract_d = [value for value in all_d if value > 0]
    citizen_adjutant_d = role_d["adjutant"] + role_d["citizen"]
    ca_fallback = _mean_or_zero(citizen_adjutant_d or all_contract_d or all_d)
    contract_fallback = _mean_or_zero(all_contract_d or all_d)
    d_values = {}
    for role in roles:
        if role in ("adjutant", "citizen"):
            d_values[role] = _mean_or_default(role_d[role], ca_fallback)
        elif role == "no-contract":
            d_values[role] = 0.0
        else:
            d_values[role] = _mean_or_default(role_d[role], contract_fallback)
    return {
        "masses": {role: role_counts[role] / total for role in roles},
        "d": d_values,
    }


def _expand_pass_terminal_role_prior(
    prior: dict[str, dict[str, float]], sample_count: int
) -> dict[str, dict[str, np.ndarray]]:
    roles = ("adjutant", "citizen", "napoleon", "napoleon-adjutant", "no-contract")
    return {
        "masses": {
            role: np.full((sample_count,), prior["masses"].get(role, 0.0), dtype=np.float64)
            for role in roles
        },
        "d": {
            role: np.full((sample_count,), prior["d"].get(role, 0.0), dtype=np.float64)
            for role in roles
        },
    }


def _mean_or_zero(values: list[int]) -> float:
    return float(np.mean(values)) if values else 0.0


def _mean_or_default(values: list[int], default: float) -> float:
    return float(np.mean(values)) if values else default


def _validate_role_checkpoint_role(
    raw: dict[str, Any],
    *,
    expected_role: Literal["citizen", "adjutant"],
    path: Path | str,
    checkpoint_kind: str,
) -> None:
    checkpoint_role = raw.get("role")
    config_raw = raw.get("modelConfig")
    config_role = config_raw.get("role") if isinstance(config_raw, dict) else None
    mismatches = [
        str(role)
        for role in (checkpoint_role, config_role)
        if role is not None and role != expected_role
    ]
    if mismatches:
        raise ValueError(
            f"{path} is a {mismatches[0]} {checkpoint_kind} checkpoint, "
            f"but {expected_role} was expected."
        )


def _validate_role_margin_checkpoint_role(
    raw: dict[str, Any],
    *,
    expected_role: Literal["citizen", "adjutant"],
    path: Path | str,
) -> None:
    _validate_role_checkpoint_role(
        raw, expected_role=expected_role, path=path, checkpoint_kind="role-margin"
    )


def _validate_role_value_checkpoint_role(
    raw: dict[str, Any],
    *,
    expected_role: Literal["citizen", "adjutant"],
    path: Path | str,
) -> None:
    _validate_role_checkpoint_role(
        raw, expected_role=expected_role, path=path, checkpoint_kind="role-value"
    )


def _r0_value_score(
    samples: tuple[BiddingQRawSample, ...],
    *,
    pass_q: np.ndarray,
    napoleon_margin_mean: np.ndarray,
    citizen_value_checkpoint_path: Path | str | None,
    adjutant_value_checkpoint_path: Path | str | None,
    device: Any,
) -> np.ndarray | None:
    if citizen_value_checkpoint_path is None or adjutant_value_checkpoint_path is None:
        return None
    citizen_model, citizen_raw = load_bidding_role_value_checkpoint(citizen_value_checkpoint_path)
    adjutant_model, adjutant_raw = load_bidding_role_value_checkpoint(
        adjutant_value_checkpoint_path
    )
    _validate_role_value_checkpoint_role(
        citizen_raw, expected_role="citizen", path=citizen_value_checkpoint_path
    )
    _validate_role_value_checkpoint_role(
        adjutant_raw, expected_role="adjutant", path=adjutant_value_checkpoint_path
    )
    citizen_model.to(device.torch_device)
    adjutant_model.to(device.torch_device)
    citizen_value = predict_role_value_samples(
        citizen_model,
        samples,
        device=device,
        standardization=_standardization_from_checkpoint(citizen_raw, "targetStandardization"),
    )
    adjutant_value = predict_role_value_samples(
        adjutant_model,
        samples,
        device=device,
        standardization=_standardization_from_checkpoint(adjutant_raw, "targetStandardization"),
    )
    score = napoleon_margin_mean.copy()
    score[:, 0] = pass_q * adjutant_value[:, 0] + (1.0 - pass_q) * citizen_value[:, 0]
    return score


def _explicit_success_probabilities(
    checkpoint_path: Path | str | None,
    samples: tuple[BiddingQRawSample, ...],
    device: Any,
) -> np.ndarray | None:
    if checkpoint_path is None:
        return None
    model, raw = load_bidding_multi_head_q_checkpoint(checkpoint_path)
    model.to(device.torch_device)
    predictions = predict_multi_head_samples(
        model,
        samples,
        device=device,
        napoleon_standardization=_standardization_from_checkpoint(raw, "napoleonStandardization"),
        margin_standardization=_standardization_from_checkpoint(raw, "marginStandardization"),
    )
    return predictions["successProbabilities"]


def _role_margin_eval_from_checkpoint(
    model: Any,
    raw: dict[str, Any],
    samples: tuple[BiddingQRawSample, ...],
    device: Any,
) -> dict[str, Any]:
    role = raw.get("role")
    if role not in ("citizen", "adjutant"):
        raise ValueError("role margin checkpoint role mismatch.")
    return evaluate_bidding_role_margin_model(
        model,
        samples,
        train_samples=samples,
        role=role,
        device=device,
        standardization=_standardization_from_checkpoint(raw, "targetStandardization"),
    )


def _success_pairwise(
    samples: tuple[BiddingQRawSample, ...], probability: np.ndarray
) -> float | None:
    by_state: dict[str, list[tuple[BiddingQRawSample, float]]] = defaultdict(list)
    for sample, p in zip(samples, probability, strict=True):
        by_state[sample.state_key].append((sample, float(p)))
    correct = 0.0
    total = 0
    for rows in by_state.values():
        for left_index, left in enumerate(rows):
            for right in rows[left_index + 1 :]:
                left_success = left[0].contract_success
                right_success = right[0].contract_success
                if left_success == right_success:
                    continue
                total += 1
                score_cmp = _compare(left[1], right[1])
                truth_cmp = _compare(1.0 if left_success else 0.0, 1.0 if right_success else 0.0)
                if score_cmp == 0:
                    correct += 0.5
                elif score_cmp == truth_cmp:
                    correct += 1.0
    return _safe_div(correct, total)


def _baseline_action_index(name: str, state: dict[str, Any]) -> int:
    sample = state["first_sample"]
    actions = list(state["action_indices"])
    if name == "behavior":
        selected = sample.source_selected_action_index
        return selected if selected in actions else 0
    if name == "pass-only":
        return 0 if 0 in actions else actions[0]
    bid_limit = _bid_limit_for_score(
        float(sample.strongest_suit_score),
        threshold=280.0 if name == "conservative-offline" else 200.0,
        step=55.0 if name == "conservative-offline" else 30.0,
    )
    needed = _needed_target(actions)
    if bid_limit is None or needed > bid_limit:
        return 0
    for action in actions:
        if action_suit(action) == sample.strongest_suit and action_target(action) == needed:
            return action
    return 0


def _needed_target(actions: list[int]) -> int:
    targets = [target for action in actions if (target := action_target(action)) is not None]
    return min(targets) if targets else 20


def _bid_limit_for_score(score: float, *, threshold: float, step: float) -> int | None:
    if score < threshold:
        return None
    return min(19, 13 + int(math.floor((score - threshold) / step)))


def _current_bid_level(sample: BiddingQRawSample) -> str:
    targets = [
        target
        for action_index, is_legal in enumerate(sample.legal_bid_mask)
        if is_legal == 1.0 and (target := action_target(action_index)) is not None
    ]
    if not targets:
        return "no-legal-bid"
    needed = min(targets)
    return "open:13" if needed == 13 else f"raise-to:{needed}"


def _bid_rate_table(counters: dict[str, Counter[str]]) -> dict[str, Any]:
    return {
        key: {
            "stateCount": int(counter["total"]),
            "passRate": _safe_div(counter["pass"], counter["total"]),
            "bidRate": _safe_div(counter["bid"], counter["total"]),
        }
        for key, counter in sorted(counters.items())
    }


def _variant_descriptions(has_r2: bool) -> dict[str, str]:
    descriptions = {
        "R0": "#395 value-based simple score, evaluated against Issue #400 realized reward.",
        "R1": "Reward expectation from Gaussian-derived role win probabilities.",
    }
    if has_r2:
        descriptions["R2"] = (
            "R1 but Napoleon BID win probability comes from the #378 explicit success head."
        )
    return descriptions


def _issue395_comparison(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        content = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"path": str(path), "available": False}
    variants = content.get("variants", {})
    if not isinstance(variants, dict) or not variants:
        return {"path": str(path), "available": False}
    candidates = {name: value for name, value in variants.items() if name in {"T1", "T2", "T3"}}
    if not candidates:
        return {"path": str(path), "available": False}
    best_name = str(content.get("learningAssessment", {}).get("bestVariant") or "")
    if best_name not in candidates:
        best_name = max(
            candidates,
            key=lambda name: (
                candidates[name].get("meanSelectedTeacherUtility")
                if candidates[name].get("meanSelectedTeacherUtility") is not None
                else float("-inf"),
                -(
                    candidates[name].get("meanRegret")
                    if candidates[name].get("meanRegret") is not None
                    else float("inf")
                ),
            ),
        )
    return {
        "path": str(path),
        "available": True,
        "t0": _compact_issue395_metrics(variants.get("T0", {})),
        "bestVariantName": best_name,
        "bestVariant": _compact_issue395_metrics(candidates[best_name]),
    }


def _compact_issue395_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
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
        "handStrengthBidRate": selected.get("handStrengthBidRate")
        if isinstance(selected, dict)
        else None,
    }


def _learning_assessment(
    variants: dict[str, Any],
    baselines: dict[str, Any],
    collapse: dict[str, Any],
) -> dict[str, Any]:
    r1 = variants["R1"]
    pass_only = baselines["pass-only"]
    conservative = baselines["conservative-offline"]
    pass_rate = r1["selectedActions"]["passRate"]
    strong = r1["selectedActions"]["handStrengthBidRate"].get("strong:280-329", {})
    very_strong = r1["selectedActions"]["handStrengthBidRate"].get("very-strong:330+", {})
    runtime_ready = (
        r1["meanSelectedRealizedReward"] is not None
        and pass_only["meanSelectedRealizedReward"] is not None
        and r1["meanSelectedRealizedReward"] >= pass_only["meanSelectedRealizedReward"]
        and r1["meanRegret"] <= conservative["meanRegret"]
        and pass_rate is not None
        and pass_rate < 0.95
        and not collapse["R1"]["collapsed"]
    )
    return {
        "primaryVariant": "R1",
        "r1VsPassOnlyRewardDelta": _delta(
            r1["meanSelectedRealizedReward"], pass_only["meanSelectedRealizedReward"]
        ),
        "r1VsConservativeRewardDelta": _delta(
            r1["meanSelectedRealizedReward"], conservative["meanSelectedRealizedReward"]
        ),
        "strongHandBidRate": strong.get("bidRate"),
        "veryStrongHandBidRate": very_strong.get("bidRate"),
        "collapseFree": not collapse["R1"]["collapsed"],
        "runtimeConnectionReady": bool(runtime_ready),
        "runtimeConnectionPolicy": "do not connect runtime bidding agent in Issue #400",
    }


def _distribution(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def _strength_bucket(score: float) -> str:
    if score < 200:
        return "low:<200"
    if score < 280:
        return "medium:200-279"
    if score < 330:
        return "strong:280-329"
    return "very-strong:330+"


def _log_loss(truth: np.ndarray, probability: np.ndarray) -> float | None:
    if truth.size == 0:
        return None
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(truth * np.log(clipped) + (1.0 - truth) * np.log(1.0 - clipped)))


def _delta(value: Any, baseline: Any) -> float | None:
    if isinstance(value, int | float) and isinstance(baseline, int | float):
        return float(value) - float(baseline)
    return None


def _compare(left: float, right: float) -> int:
    if left > right:
        return 1
    if left < right:
        return -1
    return 0
