from __future__ import annotations

import numpy as np
import pytest

from napoleon_ml.bidding_q import (
    AffineCalibration,
    RoleScorePredictions,
    compose_score_matrices,
    create_role_score_state_split,
    evaluate_score_matrices,
    fit_affine_calibration,
    fit_role_calibrations,
    role_probability_metrics,
    teacher_utility,
    validate_role_probabilities,
)
from napoleon_ml.bidding_q.dataset import BiddingQRawSample
from napoleon_ml.bidding_q.role_model import BIDDING_ROLE_CLASSES
from napoleon_ml.dataset.constants import BIDDING_ACTION_COUNT
from napoleon_ml.dataset.tensors import BIDDING_MODEL_INPUT_FEATURE_COUNT


def test_teacher_utility_uses_role_sign_and_no_contract_zero() -> None:
    assert teacher_utility(_sample("s", 1, "napoleon", 4)) == 4.0
    assert teacher_utility(_sample("s", 1, "napoleon-adjutant", 4)) == 4.0
    assert teacher_utility(_sample("s", 1, "adjutant", 4)) == 4.0
    assert teacher_utility(_sample("s", 1, "citizen", 4)) == -4.0
    assert (
        teacher_utility(
            _sample(
                "s",
                1,
                "citizen",
                None,
                final_declared_target=15,
                napoleon_side_point_cards=11,
            )
        )
        == 4.0
    )
    assert teacher_utility(_sample("s", 0, "no-contract", None)) == 0.0


def test_validate_role_probabilities_rejects_unnormalized_values() -> None:
    probabilities = np.zeros((2, BIDDING_ACTION_COUNT, len(BIDDING_ROLE_CLASSES)))
    probabilities[:, :, 0] = 1.0
    validate_role_probabilities(probabilities)
    probabilities[0, 0, 0] = 0.5
    with pytest.raises(ValueError, match="sum to 1"):
        validate_role_probabilities(probabilities)


def test_raw_role_score_keeps_no_contract_mass_at_zero() -> None:
    predictions = _predictions(sample_count=1)
    predictions.role_probabilities[0, :, :] = 0.0
    predictions.role_probabilities[0, :, _role("napoleon")] = 0.2
    predictions.role_probabilities[0, :, _role("adjutant")] = 0.1
    predictions.role_probabilities[0, :, _role("citizen")] = 0.1
    predictions.role_probabilities[0, :, _role("no-contract")] = 0.6
    predictions.napoleon_value[0, 1] = 10.0
    predictions.adjutant_value[0, 1] = 20.0
    predictions.citizen_value[0, 1] = -5.0
    scores = compose_score_matrices(predictions, calibrations=_identity_calibrations())
    assert scores.variants["S1"][0, 1] == pytest.approx(0.2 * 10.0 + 0.1 * 20.0 - 0.1 * 5.0)


def test_affine_calibration_is_fitted_from_supplied_split_only() -> None:
    calibration = fit_affine_calibration(
        "citizen",
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([3.0, 5.0, 7.0]),
    )
    assert calibration.slope == pytest.approx(2.0)
    assert calibration.intercept == pytest.approx(1.0)
    assert calibration.apply(np.asarray([10.0]))[0] == pytest.approx(21.0)


def test_fit_role_calibrations_uses_only_matching_terminal_roles() -> None:
    samples = (
        _sample("n", 1, "napoleon", 3),
        _sample("a", 1, "adjutant", 4),
        _sample("c", 1, "citizen", -5),
        _sample("x", 1, "no-contract", None),
    )
    predictions = _predictions(sample_count=len(samples))
    predictions.napoleon_value[:, 1] = [2.0, 200.0, 200.0, 200.0]
    predictions.adjutant_value[:, 1] = [300.0, 4.0, 300.0, 300.0]
    predictions.citizen_value[:, 1] = [400.0, 400.0, 5.0, 400.0]
    calibrations = fit_role_calibrations(samples, predictions)
    assert calibrations["napoleon"].sample_count == 1
    assert calibrations["adjutant"].sample_count == 1
    assert calibrations["citizen"].sample_count == 1


def test_state_split_has_no_leakage() -> None:
    samples = tuple(_sample(f"s{i}", 0, "no-contract", None) for i in range(12))
    split = create_role_score_state_split(samples, calibration_ratio=0.25, validation_ratio=0.25)
    assert split.calibration_state_keys.isdisjoint(split.validation_state_keys)
    assert split.calibration_state_keys.isdisjoint(split.final_state_keys)
    assert split.validation_state_keys.isdisjoint(split.final_state_keys)
    assert len(split.final_state_keys) > 0


def test_ranking_reports_selected_utility_and_regret() -> None:
    samples = (
        _sample("s0", 0, "no-contract", None),
        _sample("s0", 1, "napoleon", 5),
        _sample("s0", 2, "citizen", -10),
    )
    scores = np.zeros((len(samples), BIDDING_ACTION_COUNT), dtype=np.float64)
    scores[:, 1] = 2.0
    scores[:, 2] = 1.0
    report = evaluate_score_matrices(samples, {"model": scores})["model"]
    assert report["meanSelectedTeacherUtility"] == pytest.approx(5.0)
    assert report["meanRegret"] == pytest.approx(5.0)
    assert report["bestActionAccuracy"] == pytest.approx(0.0)
    assert report["selectedActions"]["bidRate"] == pytest.approx(1.0)


def test_role_probability_metrics_reports_brier_and_per_role_support() -> None:
    samples = (
        _sample("s0", 0, "no-contract", None),
        _sample("s1", 1, "citizen", 3),
    )
    probabilities = np.zeros((2, BIDDING_ACTION_COUNT, len(BIDDING_ROLE_CLASSES)))
    probabilities[:, :, _role("no-contract")] = 1.0
    probabilities[1, 1, _role("no-contract")] = 0.25
    probabilities[1, 1, _role("citizen")] = 0.75
    metrics = role_probability_metrics(samples, probabilities)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["brier"] is not None
    assert metrics["perRole"]["citizen"]["support"] == 1


def _predictions(sample_count: int) -> RoleScorePredictions:
    probabilities = np.zeros((sample_count, BIDDING_ACTION_COUNT, len(BIDDING_ROLE_CLASSES)))
    probabilities[:, :, _role("no-contract")] = 1.0
    return RoleScorePredictions(
        role_probabilities=probabilities,
        napoleon_value=np.zeros((sample_count, BIDDING_ACTION_COUNT), dtype=np.float64),
        napoleon_sigma=np.ones((sample_count, BIDDING_ACTION_COUNT), dtype=np.float64),
        adjutant_value=np.zeros((sample_count, BIDDING_ACTION_COUNT), dtype=np.float64),
        citizen_value=np.zeros((sample_count, BIDDING_ACTION_COUNT), dtype=np.float64),
    )


def _identity_calibrations() -> dict[str, AffineCalibration]:
    return {
        "napoleon": AffineCalibration("napoleon", 1.0, 0.0, 10),
        "adjutant": AffineCalibration("adjutant", 1.0, 0.0, 10),
        "citizen": AffineCalibration("citizen", 1.0, 0.0, 10),
    }


def _role(name: str) -> int:
    return BIDDING_ROLE_CLASSES.index(name)


def _sample(
    state_key: str,
    action_index: int,
    terminal_role: str,
    contract_margin: int | None,
    *,
    final_declared_target: int | None = None,
    napoleon_side_point_cards: int | None = None,
) -> BiddingQRawSample:
    legal = np.zeros((BIDDING_ACTION_COUNT,), dtype=np.float32)
    legal[0] = 1.0
    legal[1] = 1.0
    legal[2] = 1.0
    return BiddingQRawSample(
        state_key=state_key,
        model_input=np.zeros((BIDDING_MODEL_INPUT_FEATURE_COUNT,), dtype=np.float32),
        legal_bid_mask=legal,
        forced_action_index=action_index,
        terminal_reward=0.0,
        raw_terminal_reward=0.0,
        repeat_index=0,
        rollout_seed=1,
        source_seed=1,
        source_game_seed=1,
        candidate_seat_index=0,
        acting_player_index=0,
        bidding_step=0,
        strongest_suit="spades",
        strongest_suit_score=300.0,
        forced_action_type="pass" if action_index == 0 else "bid",
        forced_target_point_cards=None if action_index == 0 else 13,
        forced_suit=None if action_index == 0 else ("spades" if action_index == 1 else "hearts"),
        terminal_role=terminal_role,
        contract_success=contract_margin is not None and contract_margin >= 0,
        result_type="all-pass" if terminal_role == "no-contract" else "contract",
        final_role=terminal_role,
        candidate_final_team=None,
        napoleon_side_point_cards=napoleon_side_point_cards,
        coalition_side_point_cards=None,
        candidate_team_point_cards=None,
        team_point_cards_regression_mask=False,
        final_declared_target=final_declared_target
        if final_declared_target is not None
        else (None if contract_margin is None else 13),
        final_declared_suit=None if contract_margin is None else "spades",
        contract_margin=contract_margin,
        opponent_configuration_key=None,
        source_selected_action_index=1,
    )
