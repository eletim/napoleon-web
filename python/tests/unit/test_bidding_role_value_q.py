from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
import torch
from test_bidding_role_q import _write_role_q_dataset

from napoleon_ml.bidding_q import (
    BIDDING_ADJUTANT_VALUE_ARCHITECTURE_ID,
    BIDDING_CITIZEN_VALUE_ARCHITECTURE_ID,
    BiddingRoleValueModel,
    BiddingRoleValueModelConfig,
    BiddingRoleValueTrainConfig,
    create_role_value_split,
    load_bidding_q_dataset,
    load_bidding_role_value_checkpoint,
    role_value_coverage,
    role_value_learning_assessment,
    role_value_loss,
    role_value_ranking,
    role_value_teacher,
    save_bidding_role_value_artifact,
    state_key_hash,
    train_bidding_role_value_model,
)
from napoleon_ml.dataset.constants import BIDDING_ACTION_COUNT
from napoleon_ml.dataset.tensors import BIDDING_MODEL_INPUT_FEATURE_COUNT


def test_role_value_model_shape_and_architecture_id() -> None:
    citizen = BiddingRoleValueModel(BiddingRoleValueModelConfig(role="citizen", hidden_dims=(8,)))
    adjutant = BiddingRoleValueModel(
        BiddingRoleValueModelConfig(role="adjutant", hidden_dims=(8,))
    )
    model_input = torch.zeros((2, BIDDING_MODEL_INPUT_FEATURE_COUNT), dtype=torch.float32)

    assert citizen(model_input).shape == (2, BIDDING_ACTION_COUNT)
    assert adjutant(model_input).shape == (2, BIDDING_ACTION_COUNT)
    assert citizen.config.to_dict()["architectureId"] == BIDDING_CITIZEN_VALUE_ARCHITECTURE_ID
    assert adjutant.config.to_dict()["architectureId"] == BIDDING_ADJUTANT_VALUE_ARCHITECTURE_ID


def test_role_value_loss_is_selected_action_only_and_role_masked() -> None:
    value = torch.zeros((2, BIDDING_ACTION_COUNT), dtype=torch.float32, requires_grad=True)
    action_index = torch.tensor([3, 4], dtype=torch.long)
    target = torch.tensor([2.0, 100.0], dtype=torch.float32)
    mask = torch.tensor([True, False], dtype=torch.bool)

    loss = role_value_loss(value, action_index, target, mask)
    assert loss.item() == pytest.approx(4.0)
    loss.backward()  # type: ignore[no-untyped-call]

    assert value.grad is not None
    assert value.grad.nonzero().tolist() == [[0, 3]]

    assert role_value_loss(
        value.detach(),
        action_index,
        target,
        torch.tensor([False, False], dtype=torch.bool),
    ).item() == pytest.approx(0.0)


def test_role_value_teachers_and_coverage(tmp_path: Path) -> None:
    _write_role_q_dataset(tmp_path / "dataset", repeats=1, state_count=8)
    dataset = load_bidding_q_dataset(tmp_path / "dataset")
    citizen = next(sample for sample in dataset.raw_samples if sample.final_role == "citizen")
    adjutant = next(sample for sample in dataset.raw_samples if sample.final_role == "adjutant")

    assert citizen.final_declared_target is not None
    assert citizen.napoleon_side_point_cards is not None
    assert role_value_teacher(
        citizen, role="citizen", teacher="coalition-side-point-cards"
    ) == pytest.approx(citizen.coalition_side_point_cards)
    assert role_value_teacher(
        citizen, role="citizen", teacher="negative-contract-margin"
    ) == pytest.approx(citizen.final_declared_target - citizen.napoleon_side_point_cards)
    assert role_value_teacher(
        adjutant, role="adjutant", teacher="contract-margin"
    ) == pytest.approx(adjutant.contract_margin)
    assert role_value_teacher(adjutant, role="citizen", teacher="negative-contract-margin") is None

    coverage = role_value_coverage(dataset.raw_samples)
    role_counts = cast(dict[str, object], coverage["roleSampleCounts"])
    assert cast(int, role_counts["citizen"]) > 0
    teacher_coverage = cast(dict[str, object], coverage["teacherCoverage"])
    citizen_coverage = cast(dict[str, object], teacher_coverage["citizen"])
    assert "negative-contract-margin" in citizen_coverage


def test_role_value_split_has_no_state_key_leakage(tmp_path: Path) -> None:
    _write_role_q_dataset(tmp_path / "dataset", repeats=1, state_count=12)
    dataset = load_bidding_q_dataset(tmp_path / "dataset")
    validation_keys = sorted([f"state-{index}" for index in range(3)])
    validation_file = tmp_path / "validation-state-keys.json"
    validation_file.write_text(json.dumps({"stateKeys": validation_keys}), encoding="utf-8")

    split = create_role_value_split(
        dataset,
        BiddingRoleValueTrainConfig(
            role="citizen",
            hidden_dims=(8,),
            train_state_count=5,
            validation_state_keys_path=str(validation_file),
        ),
    )

    assert split.validation_state_key_hash == state_key_hash(validation_keys)
    assert split.train_state_keys.isdisjoint(split.validation_state_keys)


def test_role_value_split_can_use_role_stratified_validation(tmp_path: Path) -> None:
    _write_role_q_dataset(tmp_path / "dataset", repeats=1, state_count=18)
    dataset = load_bidding_q_dataset(tmp_path / "dataset")

    split = create_role_value_split(
        dataset,
        BiddingRoleValueTrainConfig(
            role="adjutant",
            hidden_dims=(8,),
            train_state_count=8,
            validation_state_count=4,
            role_stratified_validation=True,
        ),
    )

    validation_coverage = role_value_coverage(split.validation_samples)
    teacher_coverage = cast(dict[str, object], validation_coverage["teacherCoverage"])
    adjutant_coverage = cast(dict[str, object], teacher_coverage["adjutant"])
    contract_margin = cast(dict[str, object], adjutant_coverage["contract-margin"])
    assert split.train_state_keys.isdisjoint(split.validation_state_keys)
    assert contract_margin["rankingStateCount"] == 4
    assert contract_margin["pairCount"] == 4


def test_role_value_ranking_reports_ties(tmp_path: Path) -> None:
    _write_role_q_dataset(tmp_path / "dataset", repeats=1, state_count=8)
    dataset = load_bidding_q_dataset(tmp_path / "dataset")
    score = torch.zeros(
        (len(dataset.raw_samples), BIDDING_ACTION_COUNT), dtype=torch.float32
    ).numpy()
    for index, sample in enumerate(dataset.raw_samples):
        value = role_value_teacher(sample, role="citizen", teacher="negative-contract-margin")
        if value is not None:
            score[index, sample.forced_action_index] = value

    ranking = role_value_ranking(
        dataset.raw_samples,
        score,
        role="citizen",
        teacher="negative-contract-margin",
    )
    assert ranking["pairCount"] is not None
    assert ranking["teacherTieRate"] is not None


def test_role_value_learning_assessment_rejects_low_pair_coverage() -> None:
    report = {
        "regression": {"mae": 1.0, "rmse": 2.0, "pearsonCorrelation": 0.5},
        "ranking": {
            "differentPairCount": 3,
            "rankingStateCount": 2,
            "pairwiseAccuracy": 1.0,
        },
        "baselines": {
            "globalMean": {"mae": 2.0, "rmse": 3.0},
            "actionIndexMean": {"mae": 2.0, "rmse": 3.0},
            "suitTargetMean": {"mae": 2.0, "rmse": 3.0},
        },
    }

    assessment = role_value_learning_assessment(
        report,
        minimum_diff_pairs=200,
        minimum_ranking_states=100,
    )

    assert assessment["coverageSufficient"] is False
    assert assessment["established"] is False
    assert "insufficient validation ranking coverage" in str(assessment["reason"])


def test_role_value_training_checkpoint_and_low_sample_handling(tmp_path: Path) -> None:
    _write_role_q_dataset(tmp_path / "dataset", repeats=1, state_count=16)
    dataset = load_bidding_q_dataset(tmp_path / "dataset")
    validation_keys = sorted([f"state-{index}" for index in range(4)])
    validation_file = tmp_path / "validation-state-keys.json"
    validation_file.write_text(json.dumps({"stateKeys": validation_keys}), encoding="utf-8")
    config = BiddingRoleValueTrainConfig(
        role="citizen",
        seed=383,
        epochs=4,
        batch_size=8,
        learning_rate=5e-3,
        hidden_dims=(16, 8),
        train_state_count=8,
        validation_state_keys_path=str(validation_file),
        patience=2,
        device="cpu",
    )

    first = train_bidding_role_value_model(dataset, config)
    second = train_bidding_role_value_model(dataset, config)
    assert first.epoch_reports == second.epoch_reports
    validation = first.validation_report
    assert cast(dict[str, object], validation["regression"])["mae"] is not None
    assert cast(dict[str, object], validation["ranking"])["pairCount"] is not None
    artifact = save_bidding_role_value_artifact(
        tmp_path / "artifact", result=first, dataset=dataset
    )
    loaded_model, checkpoint = load_bidding_role_value_checkpoint(
        Path(str(artifact["checkpointPath"]))
    )
    assert checkpoint["modelType"] == "bidding-role-value"
    assert loaded_model(
        torch.zeros((1, BIDDING_MODEL_INPUT_FEATURE_COUNT), dtype=torch.float32)
    ).shape == (1, BIDDING_ACTION_COUNT)

    with pytest.raises(ValueError, match="no adjutant samples|no teacher samples"):
        train_bidding_role_value_model(
            dataset,
            BiddingRoleValueTrainConfig(
                role="adjutant",
                seed=383,
                train_state_count=1,
                validation_state_keys_path=str(_validation_file_without_state(tmp_path, 5, 16)),
                hidden_dims=(8,),
                epochs=1,
                device="cpu",
            ),
        )


def _validation_file_without_state(
    tmp_path: Path, train_state_index: int, state_count: int
) -> Path:
    path = tmp_path / f"validation-without-state-{train_state_index}.json"
    keys = [f"state-{index}" for index in range(state_count) if index != train_state_index]
    path.write_text(json.dumps({"stateKeys": keys}), encoding="utf-8")
    return path
