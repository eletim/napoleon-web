from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from test_bidding_role_score_integration import _sample

from napoleon_ml.bidding_q import (
    BIDDING_ADJUTANT_MARGIN_HETEROSCEDASTIC_ARCHITECTURE_ID,
    BIDDING_CITIZEN_MARGIN_HETEROSCEDASTIC_ARCHITECTURE_ID,
    BiddingRoleMarginHeteroscedasticModel,
    BiddingRoleMarginHeteroscedasticModelConfig,
    RoleWinPredictions,
    compose_reward_score_matrices,
    evaluate_reward_score_matrices,
    napoleon_positive_ev_threshold,
    realized_role_reward,
    reward_d_for_action,
    role_expected_reward,
    role_margin_architecture_id,
    role_margin_teacher,
)
from napoleon_ml.bidding_q.margin_training import gaussian_margin_nll, gaussian_success_probability
from napoleon_ml.bidding_q.reward_expectation_score import (
    _difficulty_matrix,
    _issue395_comparison,
    _pass_terminal_role_prior,
    _validate_role_margin_checkpoint_role,
    _validate_role_value_checkpoint_role,
)
from napoleon_ml.dataset.constants import BIDDING_ACTION_COUNT
from napoleon_ml.dataset.tensors import BIDDING_MODEL_INPUT_FEATURE_COUNT


def test_role_margin_model_shape_and_architecture_id() -> None:
    citizen = BiddingRoleMarginHeteroscedasticModel(
        BiddingRoleMarginHeteroscedasticModelConfig(role="citizen", hidden_dims=(8,))
    )
    adjutant = BiddingRoleMarginHeteroscedasticModel(
        BiddingRoleMarginHeteroscedasticModelConfig(role="adjutant", hidden_dims=(8,))
    )
    model_input = torch.zeros((2, BIDDING_MODEL_INPUT_FEATURE_COUNT), dtype=torch.float32)

    citizen_mean, citizen_logvar = citizen(model_input)
    adjutant_mean, adjutant_logvar = adjutant(model_input)

    assert citizen_mean.shape == (2, BIDDING_ACTION_COUNT)
    assert citizen_logvar.shape == (2, BIDDING_ACTION_COUNT)
    assert adjutant_mean.shape == (2, BIDDING_ACTION_COUNT)
    assert adjutant_logvar.shape == (2, BIDDING_ACTION_COUNT)
    assert (
        role_margin_architecture_id("citizen")
        == BIDDING_CITIZEN_MARGIN_HETEROSCEDASTIC_ARCHITECTURE_ID
    )
    assert (
        role_margin_architecture_id("adjutant")
        == BIDDING_ADJUTANT_MARGIN_HETEROSCEDASTIC_ARCHITECTURE_ID
    )


def test_role_margin_teacher_is_contract_margin_and_role_masked() -> None:
    assert role_margin_teacher(_standard_sample("c", 1, "citizen", 4), "citizen") == 4.0
    assert role_margin_teacher(_standard_sample("a", 1, "adjutant", -2), "adjutant") == -2.0
    assert role_margin_teacher(_standard_sample("a", 1, "adjutant", -2), "citizen") is None
    assert role_margin_teacher(_sample("n", 0, "no-contract", None), "citizen") is None


def test_selected_action_nll_uses_role_mask_only() -> None:
    mean = torch.zeros((2, BIDDING_ACTION_COUNT), dtype=torch.float32, requires_grad=True)
    log_variance = torch.zeros((2, BIDDING_ACTION_COUNT), dtype=torch.float32)
    action_index = torch.tensor([3, 4], dtype=torch.long)
    target = torch.tensor([2.0, 100.0], dtype=torch.float32)
    mask = torch.tensor([True, False], dtype=torch.bool)

    loss = gaussian_margin_nll(mean, log_variance, action_index, target, mask)
    assert loss.item() == pytest.approx(2.0)
    loss.backward()  # type: ignore[no-untyped-call]

    assert mean.grad is not None
    assert mean.grad.nonzero().tolist() == [[0, 3]]


def test_gaussian_win_probability_fixture() -> None:
    probability = gaussian_success_probability(
        np.asarray([0.0, 1.0, -1.0]),
        np.asarray([1.0, 1.0, 1.0]),
    )
    assert probability[0] == pytest.approx(0.5)
    assert probability[1] > 0.84
    assert probability[2] < 0.16


def test_reward_table_and_positive_ev_threshold() -> None:
    p = np.asarray([0.0, 0.5, 1.0])
    d = np.asarray([13.0, 13.0, 13.0])

    assert role_expected_reward(
        role="napoleon", napoleon_side_win_probability=p, d=d
    ).tolist() == pytest.approx([-5.0, 10.5, 26.0])
    assert role_expected_reward(
        role="napoleon-adjutant", napoleon_side_win_probability=p, d=d
    ).tolist() == pytest.approx([-5.0, 17.0, 39.0])
    assert role_expected_reward(
        role="adjutant", napoleon_side_win_probability=p, d=d
    ).tolist() == pytest.approx([0.0, 6.5, 13.0])
    assert role_expected_reward(
        role="citizen", napoleon_side_win_probability=p, d=d
    ).tolist() == pytest.approx([13.0, 6.5, 0.0])
    assert napoleon_positive_ev_threshold(13) == pytest.approx(5.0 / 31.0)


def test_realized_reward_and_d_handling() -> None:
    assert reward_d_for_action(_sample("b", 1, "napoleon", 3), 1) == 13
    assert reward_d_for_action(_sample("p", 0, "citizen", -3), 0) == 13
    assert realized_role_reward(_sample("nwin", 1, "napoleon", 1)) == 26.0
    assert realized_role_reward(_sample("nawin", 1, "napoleon-adjutant", 1)) == 39.0
    assert realized_role_reward(_sample("nlose", 1, "napoleon", -1)) == -5.0
    assert realized_role_reward(_sample("awin", 1, "adjutant", 1)) == 13.0
    assert realized_role_reward(_sample("closs", 1, "citizen", -1)) == 13.0
    assert realized_role_reward(_sample("nc", 0, "no-contract", None)) == 0.0


def test_reward_score_ranking_and_regret_use_realized_reward() -> None:
    samples = (
        _sample("s", 0, "citizen", -1),
        _sample("s", 1, "napoleon", 1),
        _sample("s", 2, "napoleon", -1),
    )
    scores = np.zeros((len(samples), BIDDING_ACTION_COUNT), dtype=np.float64)
    scores[:, 1] = 2.0
    scores[:, 0] = 1.0

    metrics = evaluate_reward_score_matrices(samples, {"model": scores})["model"]

    assert metrics["meanSelectedRealizedReward"] == pytest.approx(26.0)
    assert metrics["meanRegret"] == pytest.approx(0.0)
    assert metrics["pairwiseAccuracy"] is not None


def test_pass_expected_reward_composes_binary_probability_and_role_ev() -> None:
    samples = (
        _sample("s", 0, "adjutant", 1),
        _sample("s", 1, "napoleon", 1),
    )
    predictions = RoleWinPredictions(
        napoleon_win_probability=np.ones((2, BIDDING_ACTION_COUNT), dtype=np.float64),
        adjutant_win_probability=np.full((2, BIDDING_ACTION_COUNT), 0.8, dtype=np.float64),
        citizen_napoleon_win_probability=np.full(
            (2, BIDDING_ACTION_COUNT), 0.25, dtype=np.float64
        ),
        pass_adjutant_probability=np.asarray([0.25, 0.25], dtype=np.float64),
    )
    scores = compose_reward_score_matrices(
        samples,
        predictions=predictions,
        r0_score=None,
        pass_terminal_role_prior=_pass_terminal_role_prior(samples),
    )

    expected_pass = 0.25 * (0.8 * 13.0) + 0.75 * ((1.0 - 0.25) * 13.0)
    assert scores["R1"][0, 0] == pytest.approx(expected_pass)
    assert scores["R1"][0, 1] == pytest.approx(26.0)


def test_pass_expected_reward_accounts_for_omitted_pass_roles() -> None:
    samples = (
        _sample("s", 0, "adjutant", 1),
        _sample("s", 0, "no-contract", None),
        _sample("s", 1, "napoleon", 1),
    )
    predictions = RoleWinPredictions(
        napoleon_win_probability=np.ones((3, BIDDING_ACTION_COUNT), dtype=np.float64),
        adjutant_win_probability=np.ones((3, BIDDING_ACTION_COUNT), dtype=np.float64),
        citizen_napoleon_win_probability=np.zeros(
            (3, BIDDING_ACTION_COUNT), dtype=np.float64
        ),
        pass_adjutant_probability=np.ones((3,), dtype=np.float64),
    )

    scores = compose_reward_score_matrices(
        samples,
        predictions=predictions,
        r0_score=None,
        pass_terminal_role_prior=_pass_terminal_role_prior(samples),
    )

    assert scores["R1"][0, 0] == pytest.approx(6.5)


def test_pass_expected_reward_accounts_for_napoleon_adjutant_pass_role() -> None:
    samples = (
        _sample("s", 0, "napoleon-adjutant", 1),
        _sample("s", 1, "napoleon", 1),
    )
    predictions = RoleWinPredictions(
        napoleon_win_probability=np.ones((2, BIDDING_ACTION_COUNT), dtype=np.float64),
        adjutant_win_probability=np.zeros((2, BIDDING_ACTION_COUNT), dtype=np.float64),
        citizen_napoleon_win_probability=np.ones((2, BIDDING_ACTION_COUNT), dtype=np.float64),
        pass_adjutant_probability=np.zeros((2,), dtype=np.float64),
    )

    scores = compose_reward_score_matrices(
        samples,
        predictions=predictions,
        r0_score=None,
        pass_terminal_role_prior=_pass_terminal_role_prior(samples),
    )

    assert scores["R1"][0, 0] == pytest.approx(39.0)


def test_pass_difficulty_averages_repeated_rollout_outcomes() -> None:
    samples = (
        _sample("s", 0, "citizen", -1, final_declared_target=13),
        _sample("s", 0, "citizen", -1, final_declared_target=19),
        _sample("s", 1, "napoleon", 1),
    )

    difficulty = _difficulty_matrix(samples)

    assert difficulty[0, 0] == pytest.approx(16.0)
    assert difficulty[1, 0] == pytest.approx(16.0)
    assert difficulty[2, 0] == pytest.approx(16.0)
    assert difficulty[0, 1] == pytest.approx(13.0)


def test_role_margin_checkpoint_role_validation_rejects_swapped_inputs() -> None:
    _validate_role_margin_checkpoint_role(
        {"role": "citizen", "modelConfig": {"role": "citizen"}},
        expected_role="citizen",
        path="citizen.pt",
    )

    with pytest.raises(ValueError, match="adjutant role-margin checkpoint"):
        _validate_role_margin_checkpoint_role(
            {"role": "adjutant", "modelConfig": {"role": "adjutant"}},
            expected_role="citizen",
            path="citizen.pt",
        )
    with pytest.raises(ValueError, match="adjutant role-value checkpoint"):
        _validate_role_value_checkpoint_role(
            {"modelConfig": {"role": "adjutant"}},
            expected_role="citizen",
            path="citizen-value.pt",
        )


def test_issue395_comparison_reads_and_validates_report(tmp_path) -> None:
    report_path = tmp_path / "issue395.json"
    report_path.write_text(
        """
{
  "learningAssessment": {"bestVariant": "T2"},
  "variants": {
    "T0": {"pairwiseAccuracy": 0.1, "selectedActions": {"passRate": 1.0}},
    "T2": {
      "pairwiseAccuracy": 0.2,
      "meanSelectedTeacherUtility": 7.0,
      "meanRegret": 3.0,
      "selectedActions": {"passRate": 0.5, "bidRate": 0.5}
    }
  }
}
""",
        encoding="utf-8",
    )

    comparison = _issue395_comparison(report_path)
    missing = _issue395_comparison(tmp_path / "missing.json")

    assert comparison is not None
    assert comparison["available"] is True
    assert comparison["bestVariantName"] == "T2"
    assert comparison["bestVariant"]["meanSelectedTeacherUtility"] == pytest.approx(7.0)
    assert missing == {"path": str(tmp_path / "missing.json"), "available": False}


def _standard_sample(state_key: str, action_index: int, role: str, margin: int):
    return replace(
        _sample(
            state_key,
            action_index,
            role,
            margin,
            final_declared_target=13,
            napoleon_side_point_cards=13 + margin,
        ),
        result_type="standard",
    )
