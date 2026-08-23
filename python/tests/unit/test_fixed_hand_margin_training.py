from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import torch

from napoleon_ml.bidding_q.fixed_hand_margin_training import (
    FixedHandMarginDataset,
    FixedHandMarginSample,
    FixedHandMarginTrainConfig,
    create_fixed_hand_margin_split,
    fixed_hand_margin_evaluation_report,
    fixed_hand_margin_loss,
    load_fixed_hand_margin_checkpoint,
    load_fixed_hand_margin_dataset,
    same_hand_ranking,
    save_fixed_hand_margin_artifact,
    train_fixed_hand_margin_model,
)
from napoleon_ml.dataset.tensors import BIDDING_MODEL_INPUT_FEATURE_COUNT


def test_fixed_hand_split_has_no_hand_leakage(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path, hand_count=8, actions_per_hand=2)
    split = create_fixed_hand_margin_split(
        dataset,
        FixedHandMarginTrainConfig(seed=411, validation_ratio=0.25, final_ratio=0.25),
    )
    train = set(split.train_fixed_hand_ids)
    validation = set(split.validation_fixed_hand_ids)
    final = set(split.final_fixed_hand_ids)
    assert train.isdisjoint(validation)
    assert train.isdisjoint(final)
    assert validation.isdisjoint(final)
    assert "hand-0" in final


def test_napoleon_fixed_sample_type_loads_with_hand_level_split(tmp_path: Path) -> None:
    dataset = write_dataset(
        tmp_path,
        hand_count=5,
        actions_per_hand=3,
        sample_type="napoleon-fixed-contract-margin-sample",
    )
    assert dataset.manifest["sampleType"] == "napoleon-fixed-contract-margin-sample"
    assert len(dataset.samples) == 15
    split = create_fixed_hand_margin_split(
        dataset,
        FixedHandMarginTrainConfig(seed=423, validation_ratio=0.2, final_ratio=0.2),
    )
    train = set(split.train_fixed_hand_ids)
    validation = set(split.validation_fixed_hand_ids)
    final = set(split.final_fixed_hand_ids)
    assert train.isdisjoint(validation)
    assert train.isdisjoint(final)
    assert validation.isdisjoint(final)
    assert {sample.fixed_hand_id for sample in split.train_samples} == train
    assert {sample.fixed_hand_id for sample in split.validation_samples} == validation
    assert {sample.fixed_hand_id for sample in split.final_samples} == final


def test_history_consistent_raise_sample_type_loads_source_identity(tmp_path: Path) -> None:
    dataset = write_dataset(
        tmp_path,
        hand_count=4,
        actions_per_hand=2,
        sample_type="history-consistent-raise-margin-sample",
    )
    assert dataset.manifest["sampleType"] == "history-consistent-raise-margin-sample"
    first = dataset.samples[0]
    assert first.source_state_key == "state-0-1"
    assert first.deal_seed == 427001
    assert first.state_key == "state-0-1"
    ranking = same_hand_ranking(
        dataset.samples[:2],
        score=np.asarray([0.0, 1.0]),
        teacher=np.asarray([0.0, 1.0]),
    )
    assert ranking["rankingStateCount"] == 0


def test_mixed_fixed_hand_margin_sample_type_loads(tmp_path: Path) -> None:
    dataset = write_dataset(
        tmp_path,
        hand_count=3,
        actions_per_hand=2,
        sample_type="mixed-fixed-hand-margin-sample",
    )
    assert len(dataset.samples) == 6
    assert dataset.manifest["sampleType"] == "mixed-fixed-hand-margin-sample"


def test_selected_action_only_mean_loss_fixture() -> None:
    mean = torch.zeros((2, 29), dtype=torch.float32)
    log_variance = torch.zeros((2, 29), dtype=torch.float32)
    action_index = torch.tensor([1, 5], dtype=torch.long)
    target = torch.tensor([2.0, -1.0], dtype=torch.float32)
    std = torch.tensor([3.0, 4.0], dtype=torch.float32)
    loss = fixed_hand_margin_loss(
        mean=mean,
        log_variance=log_variance,
        action_index=action_index,
        target_mean=target,
        target_std=std,
        variant="M1",
        std_loss_weight=0.25,
        target_scale=1.0,
    )
    assert float(loss) == 2.5
    mean[:, 0] = 100.0
    unchanged = fixed_hand_margin_loss(
        mean=mean,
        log_variance=log_variance,
        action_index=action_index,
        target_mean=target,
        target_std=std,
        variant="M1",
        std_loss_weight=0.25,
        target_scale=1.0,
    )
    assert float(unchanged) == 2.5


def test_m2_std_loss_uses_log_variance_target() -> None:
    mean = torch.zeros((1, 29), dtype=torch.float32)
    target_std = torch.tensor([2.0], dtype=torch.float32)
    perfect = torch.zeros((1, 29), dtype=torch.float32)
    perfect[0, 1] = np.log(4.0)
    loss = fixed_hand_margin_loss(
        mean=mean,
        log_variance=perfect,
        action_index=torch.tensor([1]),
        target_mean=torch.tensor([0.0]),
        target_std=target_std,
        variant="M2",
        std_loss_weight=1.0,
        target_scale=1.0,
    )
    assert float(loss) == 0.0


def test_m2_std_loss_ignores_single_rollout_zero_std_targets() -> None:
    mean = torch.zeros((1, 29), dtype=torch.float32)
    mean[0, 1] = 2.0
    loss = fixed_hand_margin_loss(
        mean=mean,
        log_variance=torch.full((1, 29), 100.0, dtype=torch.float32),
        action_index=torch.tensor([1]),
        target_mean=torch.tensor([1.0]),
        target_std=torch.tensor([0.0]),
        variant="M2",
        std_loss_weight=100.0,
        target_scale=1.0,
    )
    assert float(loss) == 1.0


def test_checkpoint_save_load_roundtrip(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "data", hand_count=6, actions_per_hand=2)
    result = train_fixed_hand_margin_model(
        dataset,
        FixedHandMarginTrainConfig(
            variant="M1",
            seed=7,
            epochs=2,
            batch_size=4,
            hidden_dims=(16,),
            validation_ratio=0.2,
            final_ratio=0.2,
            patience=2,
        ),
    )
    artifact = save_fixed_hand_margin_artifact(
        tmp_path / "artifact",
        result=result,
        dataset=dataset,
    )
    checkpoint_path = artifact["checkpointPath"]
    assert isinstance(checkpoint_path, str)
    model, raw = load_fixed_hand_margin_checkpoint(checkpoint_path)
    assert raw["modelType"] == "fixed-hand-bidding-margin"
    assert raw["variant"] == "M1"
    assert model.mean_head.out_features == 29


def test_empirical_probability_metric_fixture() -> None:
    samples = tuple(sample("hand-a", 1, 1.0, 2.0, 0.75) for _ in range(2))
    mean = np.zeros((2, 29), dtype=np.float64)
    sigma = np.ones((2, 29), dtype=np.float64)
    mean[:, 1] = 0.0
    sigma[:, 1] = 1.0
    report = fixed_hand_margin_evaluation_report(samples, mean=mean, sigma=sigma)
    assert report["sampleCount"] == 2
    win_probability = report["winProbability"]
    assert isinstance(win_probability, dict)
    assert win_probability["mae"] == 0.25
    assert win_probability["brier"] == 0.0625


def test_selected_contract_loss_fixture_reports_negative_sign() -> None:
    samples = (sample("losing-hand", 4, -3.0, 1.0, 0.0),)
    mean = np.zeros((1, 29), dtype=np.float64)
    sigma = np.ones((1, 29), dtype=np.float64)
    mean[0, 4] = -2.0
    report = fixed_hand_margin_evaluation_report(samples, mean=mean, sigma=sigma)
    assert cast(dict[str, object], report["mean"])["bias"] == 1.0
    assert cast(dict[str, object], report["winProbability"])["mae"] < 0.05


def test_same_hand_ranking_fixture() -> None:
    samples = (
        sample("hand-a", 1, 1.0, 2.0, 0.6),
        sample("hand-a", 2, 3.0, 2.0, 0.7),
        sample("hand-b", 1, 0.0, 2.0, 0.4),
    )
    score = np.asarray([0.5, 2.0, 0.0])
    teacher = np.asarray([1.0, 3.0, 0.0])
    ranking = same_hand_ranking(samples, score=score, teacher=teacher)
    assert ranking["rankingStateCount"] == 1
    assert ranking["diffPairCount"] == 1
    assert ranking["pairwiseAccuracy"] == 1.0
    assert ranking["topActionAccuracy"] == 1.0


def write_dataset(
    directory: Path,
    *,
    hand_count: int,
    actions_per_hand: int,
    sample_type: str = "fixed-hand-bidding-margin-sample",
) -> FixedHandMarginDataset:
    directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for hand_index in range(hand_count):
        for action_offset in range(actions_per_hand):
            action_index = 1 + action_offset
            rows.append(sample_dict(hand_index, action_index, sample_type=sample_type))
    body = "".join(f"{json.dumps(row)}\n" for row in rows)
    (directory / "shard-00000.jsonl").write_text(body, encoding="utf-8")
    manifest = {
        "format": "jsonl-shards-v1",
        "sampleType": sample_type,
        "pairCount": len(rows),
        "rolloutCount": sum(cast(int, row["rolloutCount"]) for row in rows),
        "shards": [{"file": "shard-00000.jsonl", "sampleCount": len(rows)}],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return load_fixed_hand_margin_dataset(directory)


def sample_dict(hand_index: int, action_index: int, *, sample_type: str) -> dict[str, object]:
    row = {
        "sampleType": sample_type,
        "schemaVersion": 1,
        "fixedHandId": f"hand-{hand_index}",
        "handIds": [f"card-{index}" for index in range(10)],
        "forcedActionIndex": action_index,
        "forcedTargetPointCards": 13 + action_index,
        "forcedSuit": "spades",
        "modelInput": [
            float((hand_index + action_index + index) % 7)
            for index in range(BIDDING_MODEL_INPUT_FEATURE_COUNT)
        ],
        "rolloutCount": 50,
        "empiricalMarginMean": float(hand_index - action_index),
        "empiricalMarginStd": 2.0 + action_index / 10.0,
        "empiricalWinRate": 0.2 + 0.05 * action_index,
        "splitHint": "final-diagnostic" if hand_index == 0 else None,
    }
    if sample_type == "history-consistent-raise-margin-sample":
        row.update({
            "sourceStateKey": f"state-{hand_index}-{action_index}",
            "dealSeed": 427000 + hand_index * 10 + action_index,
        })
    return row


def sample(
    hand_id: str,
    action_index: int,
    empirical_mean: float,
    empirical_std: float,
    empirical_win_rate: float,
) -> FixedHandMarginSample:
    return FixedHandMarginSample(
        fixed_hand_id=hand_id,
        hand_ids=tuple(f"card-{index}" for index in range(10)),
        forced_action_index=action_index,
        forced_target_point_cards=13,
        forced_suit="spades",
        model_input=np.zeros(BIDDING_MODEL_INPUT_FEATURE_COUNT, dtype=np.float32),
        rollout_count=50,
        empirical_margin_mean=empirical_mean,
        empirical_margin_std=empirical_std,
        empirical_win_rate=empirical_win_rate,
    )
