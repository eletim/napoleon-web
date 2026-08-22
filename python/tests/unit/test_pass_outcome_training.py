from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from napoleon_ml.bidding_q.pass_outcome_training import (
    EmpiricalQTrainConfig,
    PassOutcomeDataset,
    _pass_score_ev,
    _pass_teacher_reward,
    create_pass_outcome_split,
    load_pass_outcome_dataset,
    pass_role_margin_dataset,
    train_empirical_q_model,
)
from napoleon_ml.dataset.tensors import BIDDING_MODEL_INPUT_FEATURE_COUNT


def test_pass_outcome_loader_preserves_soft_q_teacher(tmp_path: Path) -> None:
    dataset = write_pass_dataset(tmp_path, hand_count=4)
    sample = dataset.samples[0]
    assert sample.q_teacher == 0.25
    assert sample.q_denominator == 40
    assert sample.p_no_contract == 0.2
    assert sample.citizen.mean == -1.5
    assert sample.adjutant.target_mean == 14.0


def test_pass_outcome_split_uses_fixed_hand_ids_without_leakage(tmp_path: Path) -> None:
    dataset = write_pass_dataset(tmp_path, hand_count=10)
    split = create_pass_outcome_split(
        dataset,
        seed=414,
        validation_ratio=0.2,
        final_ratio=0.2,
    )
    train = set(split.train_fixed_hand_ids)
    validation = set(split.validation_fixed_hand_ids)
    final = set(split.final_fixed_hand_ids)
    assert train.isdisjoint(validation)
    assert train.isdisjoint(final)
    assert validation.isdisjoint(final)
    assert "hand-0" in final


def test_role_margin_dataset_masks_low_count_role_teacher(tmp_path: Path) -> None:
    dataset = write_pass_dataset(tmp_path, hand_count=3)
    citizen = pass_role_margin_dataset(dataset, role="citizen", min_role_count=2)
    adjutant = pass_role_margin_dataset(dataset, role="adjutant", min_role_count=2)
    assert len(citizen.samples) == 3
    assert len(adjutant.samples) == 0
    assert citizen.samples[0].forced_action_index == 0
    assert citizen.samples[0].empirical_margin_mean == -1.5


def test_role_margin_dataset_can_force_pass_final_ids(tmp_path: Path) -> None:
    dataset = write_pass_dataset(tmp_path, hand_count=3)
    citizen = pass_role_margin_dataset(
        dataset,
        role="citizen",
        min_role_count=2,
        final_fixed_hand_ids=("hand-1",),
    )
    by_hand = {sample.fixed_hand_id: sample for sample in citizen.samples}
    assert by_hand["hand-1"].split_hint == "final-diagnostic"
    assert by_hand["hand-2"].split_hint is None


def test_pass_score_does_not_read_empirical_teacher_fields(tmp_path: Path) -> None:
    dataset = write_pass_dataset(tmp_path, hand_count=2)
    left, right = dataset.samples
    pred = {
        "q": np.asarray([0.25, 0.25]),
        "citizenPWin": np.asarray([0.2, 0.2]),
        "adjutantPWin": np.asarray([0.6, 0.6]),
    }
    assert _pass_score_ev(left, pred, 0, reward_d=13.0) == _pass_score_ev(
        right,
        pred,
        1,
        reward_d=13.0,
    )


def test_pass_teacher_reward_uses_citizen_adjutant_mass(tmp_path: Path) -> None:
    dataset = write_pass_dataset(tmp_path, hand_count=1)
    sample = dataset.samples[0]
    assert _pass_teacher_reward(sample) == pytest.approx(0.8 * (
        0.25 * (0.75 * 14.0) + 0.75 * ((1.0 - 0.25) * 13.0)
    ))


def test_empirical_q_training_accepts_soft_targets(tmp_path: Path) -> None:
    dataset = write_pass_dataset(tmp_path, hand_count=8)
    result = train_empirical_q_model(
        dataset,
        EmpiricalQTrainConfig(
            seed=7,
            epochs=2,
            batch_size=4,
            hidden_dims=(16,),
            validation_ratio=0.25,
            final_ratio=0.25,
            patience=2,
        ),
    )
    assert result.final_report["sampleCount"] == 2
    assert result.final_report["rmse"] is not None


def write_pass_dataset(directory: Path, *, hand_count: int) -> PassOutcomeDataset:
    directory.mkdir(parents=True, exist_ok=True)
    rows = [pass_sample_dict(index) for index in range(hand_count)]
    (directory / "shard-00000.jsonl").write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "format": "jsonl-shards-v1",
        "sampleType": "fixed-hand-pass-outcome-sample",
        "sampleCount": len(rows),
        "fixedHandCount": len(rows),
        "rolloutCount": sum(cast(int, row["rolloutCount"]) for row in rows),
        "teacher": "fixed-hand-pass-hidden-deal-role-probability-role-margin-v1",
        "shards": [{"file": "shard-00000.jsonl", "sampleCount": len(rows)}],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return load_pass_outcome_dataset(directory)


def pass_sample_dict(index: int) -> dict[str, object]:
    return {
        "sampleType": "fixed-hand-pass-outcome-sample",
        "schemaVersion": 1,
        "fixedHandId": f"hand-{index}",
        "handIds": [f"card-{card_index}" for card_index in range(10)],
        "candidateSeatIndex": 0,
        "modelInput": [
            float((index + feature_index) % 5)
            for feature_index in range(BIDDING_MODEL_INPUT_FEATURE_COUNT)
        ],
        "legalBidMask": [1, 1, 0],
        "rolloutCount": 50,
        "nCitizen": 30,
        "nAdjutant": 10,
        "nNoContract": 10,
        "nNapoleonAfterPass": 0,
        "pCitizenEmpirical": 0.6,
        "pAdjutantEmpirical": 0.2,
        "pNoContractEmpirical": 0.2,
        "qTeacher": 0.25,
        "qTeacherDenominator": 40,
        "citizenMargin": {
            "count": 30,
            "empiricalMarginMean": -1.5,
            "empiricalMarginStd": 2.5,
            "empiricalWinRate": 0.25,
            "empiricalTargetMean": 13.0,
            "marginMin": -6,
            "marginMax": 3,
        },
        "adjutantMargin": {
            "count": 1,
            "empiricalMarginMean": 2.0,
            "empiricalMarginStd": 1.0,
            "empiricalWinRate": 0.75,
            "empiricalTargetMean": 14.0,
            "marginMin": 1,
            "marginMax": 3,
        },
        "resultTypeCounts": {},
        "finalRoleCounts": {},
        "strongestSuit": "spades",
        "strongestSuitScore": 4.0 + index,
        "splitHint": "final-diagnostic" if index == 0 else None,
    }
