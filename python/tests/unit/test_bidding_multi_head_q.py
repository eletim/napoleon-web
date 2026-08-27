from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from test_bidding_role_q import _write_role_q_dataset

from napoleon_ml.bidding_q import (
    BIDDING_MULTI_HEAD_Q_ARCHITECTURE_ID,
    BiddingMultiHeadQModel,
    BiddingMultiHeadQModelConfig,
    BiddingMultiHeadQTrainConfig,
    Standardization,
    bidding_multi_head_q_losses,
    create_multi_head_split,
    load_bidding_multi_head_q_checkpoint,
    load_bidding_q_dataset,
    load_validation_state_keys,
    save_bidding_multi_head_q_artifact,
    state_key_hash,
    teacher_signal_analysis,
    train_bidding_multi_head_q_model,
)
from napoleon_ml.bidding_q.multi_head_training import (
    BiddingMultiHeadQBatch,
    baseline_metrics,
    offline_score_metrics,
)
from napoleon_ml.dataset.constants import BIDDING_ACTION_COUNT
from napoleon_ml.dataset.tensors import BIDDING_MODEL_INPUT_FEATURE_COUNT


def test_multi_head_model_shape_and_no_future_label_input() -> None:
    model = BiddingMultiHeadQModel(BiddingMultiHeadQModelConfig(hidden_dims=(8,)))
    model_input = torch.zeros((2, BIDDING_MODEL_INPUT_FEATURE_COUNT), dtype=torch.float32)
    output = model(model_input)

    assert output.role_logits.shape == (2, BIDDING_ACTION_COUNT, 5)
    assert output.napoleon_point_cards.shape == (2, BIDDING_ACTION_COUNT)
    assert output.success_logits.shape == (2, BIDDING_ACTION_COUNT)
    assert output.contract_margin.shape == (2, BIDDING_ACTION_COUNT)
    config_text = json.dumps(model.config.to_dict())
    assert model.config.to_dict()["architectureId"] == BIDDING_MULTI_HEAD_Q_ARCHITECTURE_ID
    for forbidden in (
        "finalRole",
        "finalNapoleon",
        "finalAdjutant",
        "candidateTeamPointCards",
        "contractSuccess",
        "contractMargin",
        "finalDeclaredTarget",
        "finalDeclaredSuit",
        "stateKey",
    ):
        assert forbidden not in config_text


def test_multi_head_losses_selected_action_masks_no_contract() -> None:
    model = BiddingMultiHeadQModel(BiddingMultiHeadQModelConfig(hidden_dims=(8,)))
    output = model(torch.zeros((2, BIDDING_MODEL_INPUT_FEATURE_COUNT), dtype=torch.float32))
    batch = BiddingMultiHeadQBatch(
        model_input=torch.zeros((2, BIDDING_MODEL_INPUT_FEATURE_COUNT), dtype=torch.float32),
        action_index=torch.tensor([1, 0], dtype=torch.long),
        role_index=torch.tensor([0, 4], dtype=torch.long),
        napoleon_point_cards=torch.tensor([16.0, 0.0], dtype=torch.float32),
        coalition_point_cards=torch.tensor([4.0, 0.0], dtype=torch.float32),
        contract_success=torch.tensor([1.0, 0.0], dtype=torch.float32),
        contract_margin=torch.tensor([3.0, 0.0], dtype=torch.float32),
        contract_mask=torch.tensor([True, False], dtype=torch.bool),
    )
    losses = bidding_multi_head_q_losses(
        output,
        batch,
        config=BiddingMultiHeadQTrainConfig(hidden_dims=(8,)),
    )
    assert losses.role_loss.item() > 0.0
    assert losses.napoleon_point_loss.item() > 0.0
    assert losses.success_loss.item() > 0.0
    assert losses.margin_loss.item() > 0.0

    no_contract_batch = batch._replace(contract_mask=torch.tensor([False, False]))
    no_contract_losses = bidding_multi_head_q_losses(
        output,
        no_contract_batch,
        config=BiddingMultiHeadQTrainConfig(hidden_dims=(8,)),
    )
    assert no_contract_losses.napoleon_point_loss.item() == pytest.approx(0.0)
    assert no_contract_losses.success_loss.item() == pytest.approx(0.0)
    assert no_contract_losses.margin_loss.item() == pytest.approx(0.0)


def test_fixed_validation_hash_and_dataset_compatibility(tmp_path: Path) -> None:
    _write_role_q_dataset(tmp_path / "dataset", repeats=1, state_count=12)
    dataset = load_bidding_q_dataset(tmp_path / "dataset")
    validation_keys = sorted([f"state-{index}" for index in range(2)])
    validation_file = tmp_path / "validation-state-keys.json"
    validation_file.write_text(
        json.dumps({"stateKeys": validation_keys, "stateKeyHash": state_key_hash(validation_keys)}),
        encoding="utf-8",
    )

    assert load_validation_state_keys(validation_file) == tuple(validation_keys)
    split = create_multi_head_split(
        dataset,
        BiddingMultiHeadQTrainConfig(
            hidden_dims=(8,),
            train_state_count=5,
            validation_state_keys_path=str(validation_file),
        ),
    )

    assert split.validation_state_key_hash == state_key_hash(validation_keys)
    assert split.train_state_keys.isdisjoint(split.validation_state_keys)
    assert len(split.train_state_keys) == 5
    assert dataset.manifest.dataset_schema_version == 2
    assert dataset.manifest.sample_schema_version == 2


def test_training_early_stopping_metrics_ranking_and_checkpoint(tmp_path: Path) -> None:
    _write_role_q_dataset(tmp_path / "dataset", repeats=1, state_count=16)
    dataset = load_bidding_q_dataset(tmp_path / "dataset")
    validation_keys = sorted([f"state-{index}" for index in range(4)])
    validation_file = tmp_path / "validation-state-keys.json"
    validation_file.write_text(json.dumps({"stateKeys": validation_keys}), encoding="utf-8")
    config = BiddingMultiHeadQTrainConfig(
        seed=378,
        epochs=8,
        batch_size=8,
        learning_rate=5e-3,
        hidden_dims=(16, 8),
        train_state_count=8,
        validation_state_keys_path=str(validation_file),
        patience=3,
        device="cpu",
    )

    first = train_bidding_multi_head_q_model(dataset, config)
    second = train_bidding_multi_head_q_model(dataset, config)

    assert first.epoch_reports == second.epoch_reports
    assert 1 <= first.best_epoch <= len(first.epoch_reports) <= config.epochs
    role = cast(dict[str, object], first.validation_report["role"])
    napoleon = cast(dict[str, object], first.validation_report["napoleonSidePointCards"])
    coalition = cast(dict[str, object], first.validation_report["coalitionSidePointCards"])
    success = cast(dict[str, object], first.validation_report["contractSuccess"])
    margin = cast(dict[str, object], first.validation_report["contractMargin"])
    ranking = cast(dict[str, object], first.validation_report["ranking"])
    baselines = cast(dict[str, object], first.validation_report["baselines"])
    assert role["accuracy"] is not None
    assert cast(dict[str, object], napoleon["overall"])["mae"] is not None
    assert cast(dict[str, object], coalition["overall"])["mae"] is not None
    assert success["rocAuc"] is not None
    assert cast(dict[str, object], margin["overall"])["signAccuracy"] is not None
    assert (
        cast(dict[str, object], ranking["napoleonSideCards"])["pairwiseRankingAccuracy"] is not None
    )
    assert cast(dict[str, object], baselines["contractSuccess"])["globalSuccessRate"] is not None

    teacher = teacher_signal_analysis(dataset.raw_samples)
    derive = cast(dict[str, object], teacher["coalitionDerive"])
    assert derive["derivedAsTotalPointCardsMinusNapoleon"] is True

    artifact = save_bidding_multi_head_q_artifact(
        tmp_path / "artifact", result=first, dataset=dataset
    )
    loaded_model, checkpoint = load_bidding_multi_head_q_checkpoint(
        Path(str(artifact["checkpointPath"]))
    )
    assert checkpoint["modelType"] == "bidding-multi-head-q"
    loaded_output = loaded_model(torch.zeros((1, BIDDING_MODEL_INPUT_FEATURE_COUNT)))
    assert loaded_output.role_logits.shape == (1, BIDDING_ACTION_COUNT, 5)


def test_baselines_and_offline_score_fixture(tmp_path: Path) -> None:
    _write_role_q_dataset(tmp_path / "dataset", repeats=1, state_count=8)
    dataset = load_bidding_q_dataset(tmp_path / "dataset")
    train = dataset.raw_samples[:12]
    validation = dataset.raw_samples[12:]
    baselines = baseline_metrics(train, validation)
    assert cast(dict[str, object], baselines["napoleonSidePointCards"])["globalMean"] is not None
    global_success = cast(dict[str, object], baselines["contractSuccess"])["globalSuccessRate"]
    assert cast(dict[str, object], global_success)["rocAuc"] == pytest.approx(0.5)

    sample_count = len(validation)
    predictions = {
        "roleProbabilities": np.full((sample_count, BIDDING_ACTION_COUNT, 5), 0.2),
        "napoleonPointCards": np.zeros((sample_count, BIDDING_ACTION_COUNT)),
        "successProbabilities": np.zeros((sample_count, BIDDING_ACTION_COUNT)),
        "contractMargin": np.zeros((sample_count, BIDDING_ACTION_COUNT)),
    }
    for index, sample in enumerate(validation):
        predictions["napoleonPointCards"][index, sample.forced_action_index] = (
            sample.napoleon_side_point_cards or 0
        )
        predictions["successProbabilities"][index, sample.forced_action_index] = (
            1.0 if sample.contract_success else 0.0
        )
        if (
            sample.final_declared_target is not None
            and sample.napoleon_side_point_cards is not None
        ):
            predictions["contractMargin"][index, sample.forced_action_index] = (
                sample.napoleon_side_point_cards - sample.final_declared_target
            )
    scores = offline_score_metrics(validation, predictions)
    assert cast(dict[str, object], scores["napoleonSideCards"])["bestActionHitRate"] is not None


def test_standardization_round_trip() -> None:
    standardization = Standardization(enabled=True, mean=10.0, std=2.0)
    value = torch.tensor([8.0, 10.0, 12.0])
    encoded = standardization.encode_tensor(value)
    decoded = standardization.decode_array(encoded.numpy())
    assert decoded.tolist() == pytest.approx([8.0, 10.0, 12.0])
