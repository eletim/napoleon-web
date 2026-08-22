from __future__ import annotations

import numpy as np
import pytest
import torch
from test_bidding_role_score_integration import _sample

from napoleon_ml.bidding_q import (
    BIDDING_PASS_ROLE_BINARY_ARCHITECTURE_ID,
    AffineCalibration,
    BiddingPassRoleBinaryModel,
    BiddingPassRoleBinaryModelConfig,
    PassSimplePredictions,
    audit_pass_binary_dataset,
    compose_pass_simple_score_matrices,
    create_pass_role_binary_split,
    fit_pass_value_calibrations,
    pass_binary_label,
    pass_binary_metrics,
)
from napoleon_ml.dataset.constants import BIDDING_ACTION_COUNT
from napoleon_ml.dataset.tensors import BIDDING_MODEL_INPUT_FEATURE_COUNT


def test_pass_binary_model_shape_and_architecture_id() -> None:
    model = BiddingPassRoleBinaryModel(BiddingPassRoleBinaryModelConfig(hidden_dims=(8,)))
    model_input = torch.zeros((3, BIDDING_MODEL_INPUT_FEATURE_COUNT), dtype=torch.float32)

    assert model(model_input).shape == (3,)
    assert model.config.to_dict()["architectureId"] == BIDDING_PASS_ROLE_BINARY_ARCHITECTURE_ID


def test_pass_binary_label_masks_no_contract_and_non_pass_actions() -> None:
    assert pass_binary_label(_sample("a", 0, "adjutant", 2)) == 1
    assert pass_binary_label(_sample("c", 0, "citizen", 2)) == 0
    assert pass_binary_label(_sample("n", 0, "no-contract", None)) is None
    assert pass_binary_label(_sample("b", 1, "adjutant", 2)) is None


def test_pass_audit_reports_no_contract_and_separate_head_need() -> None:
    audit = audit_pass_binary_dataset(
        (
            _sample("c", 0, "citizen", 2),
            _sample("a", 0, "adjutant", 2),
            _sample("n0", 0, "no-contract", None),
            _sample("n1", 0, "no-contract", None),
            _sample("b", 1, "citizen", 2),
        )
    )

    assert audit["passSamplesTotal"] == 4
    assert audit["citizenCount"] == 1
    assert audit["adjutantCount"] == 1
    assert audit["noContractCount"] == 2
    assert audit["noContractRate"] == pytest.approx(0.5)
    assert audit["noContractRequiresSeparateHead"] is True


def test_pass_split_has_no_state_key_leakage() -> None:
    samples = tuple(
        _sample(f"s{i}", 0, "adjutant" if i % 2 else "citizen", i) for i in range(8)
    )
    split = create_pass_role_binary_split(samples, train_ratio=0.5, seed=395)

    assert split.train_state_keys.isdisjoint(split.calibration_state_keys)
    assert len(split.train_samples) > 0
    assert len(split.calibration_samples) > 0


def test_pass_binary_metrics_reports_calibration_and_classification() -> None:
    samples = (
        _sample("c0", 0, "citizen", 2),
        _sample("a0", 0, "adjutant", 2),
        _sample("a1", 0, "adjutant", 3),
    )
    labels = np.asarray([0.0, 1.0, 1.0])
    probabilities = np.asarray([0.1, 0.8, 0.9])

    metrics = pass_binary_metrics(samples, labels, probabilities)

    assert metrics["rocAuc"] == pytest.approx(1.0)
    assert metrics["adjutantF1"] == pytest.approx(1.0)
    assert metrics["brier"] is not None
    assert metrics["reliability"]


def test_pass_simple_score_uses_napoleon_for_bids_and_binary_pass_expectation() -> None:
    predictions = _predictions(sample_count=2, p_adjutant=np.asarray([0.25, 0.75]))
    predictions.napoleon_value[:, 1] = [4.0, 5.0]
    predictions.napoleon_sigma[:, 1] = [2.0, 4.0]
    predictions.adjutant_value[:, 0] = [10.0, 20.0]
    predictions.citizen_value[:, 0] = [-2.0, -6.0]

    scores = compose_pass_simple_score_matrices(
        predictions,
        calibrations={
            "adjutant": AffineCalibration("adjutant", 1.0, 0.0, 2),
            "citizen": AffineCalibration("citizen", 1.0, 0.0, 2),
        },
    )

    assert scores.variants["T0"][0, 0] == pytest.approx(0.0)
    assert scores.variants["T1"][0, 0] == pytest.approx(0.25 * 10.0 + 0.75 * -2.0)
    assert scores.variants["T1"][1, 0] == pytest.approx(0.75 * 20.0 + 0.25 * -6.0)
    assert scores.variants["T1"][1, 1] == pytest.approx(5.0)
    assert scores.variants["T3"][1, 1] == pytest.approx(5.0 - 0.25 * 4.0)


def test_pass_value_calibration_fits_only_supplied_pass_labeled_samples() -> None:
    samples = (
        _sample("a", 0, "adjutant", 4),
        _sample("c", 0, "citizen", -6),
        _sample("n", 0, "no-contract", None),
        _sample("b", 1, "adjutant", 100),
    )
    adjutant_value = np.zeros((len(samples), BIDDING_ACTION_COUNT), dtype=np.float64)
    citizen_value = np.zeros((len(samples), BIDDING_ACTION_COUNT), dtype=np.float64)
    adjutant_value[:, 0] = [2.0, 100.0, 100.0, 100.0]
    citizen_value[:, 0] = [100.0, 3.0, 100.0, 100.0]

    calibrations = fit_pass_value_calibrations(
        samples, adjutant_value=adjutant_value, citizen_value=citizen_value
    )

    assert calibrations["adjutant"].sample_count == 1
    assert calibrations["citizen"].sample_count == 1


def _predictions(sample_count: int, p_adjutant: np.ndarray) -> PassSimplePredictions:
    return PassSimplePredictions(
        pass_adjutant_probability=p_adjutant,
        napoleon_value=np.zeros((sample_count, BIDDING_ACTION_COUNT), dtype=np.float64),
        napoleon_sigma=np.ones((sample_count, BIDDING_ACTION_COUNT), dtype=np.float64),
        adjutant_value=np.zeros((sample_count, BIDDING_ACTION_COUNT), dtype=np.float64),
        citizen_value=np.zeros((sample_count, BIDDING_ACTION_COUNT), dtype=np.float64),
    )
