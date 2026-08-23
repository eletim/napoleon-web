from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from napoleon_ml.dataset.constants import CARD_COUNT, EXPECTED_CARD_IDS
from napoleon_ml.dataset.tensors import EXCHANGE_MODEL_INPUT_FEATURE_COUNT
from napoleon_ml.exchange_value import (
    EXCHANGE_COMPACT_STATE_FEATURE_COUNT,
    EXCHANGE_COMPACT_VALUE_INPUT_FEATURE_COUNT,
    EXCHANGE_VALUE_INPUT_FEATURE_COUNT,
    ExchangeValueMlpConfig,
    ExchangeValueMlpModel,
    ExchangeValueTrainConfig,
    create_exchange_value_split,
    load_exchange_counterfactual_dataset,
    load_exchange_value_checkpoint,
    save_exchange_value_artifact,
    train_exchange_value_model,
)
from napoleon_ml.exchange_value.training import exchange_value_evaluation_report


def _mask(indices: list[int], *, length: int = CARD_COUNT) -> list[int]:
    values = [0] * length
    for index in indices:
        values[index] = 1
    return values


def _compact_state(original: list[int], kitty: list[int], *, target_offset: int = 0) -> list[float]:
    table = [0.0] * (28 * 6)
    for position in range(28):
        table[position * 6] = 1.0
    values = [
        *_mask(original),
        *_mask(kitty),
        *_mask([0]),
        1.0, 0.0, 0.0, 0.0,
        *([1.0 if index == target_offset else 0.0 for index in range(7)]),
        1.0, 0.0, 0.0, 0.0, 0.0,
        *table,
    ]
    assert len(values) == EXCHANGE_COMPACT_STATE_FEATURE_COUNT
    return values


def _sample(*, state: int, candidate: int) -> dict[str, Any]:
    base = candidate % 13
    discard = sorted({base, (base + 1) % 13, (base + 2) % 13})
    while len(discard) < 3:
        discard.append(len(discard))
    margin = 10.0 - abs(candidate - 7)
    if candidate == 0:
        margin = 20.0
    buried_special = {
        "joker": 52 in discard,
        "oruma": 0 in discard,
        "yoromeki": 1 in discard,
        "seiJack": 2 in discard,
        "uraJack": 3 in discard,
        "calledAdjutant": 4 in discard,
    }
    original_indices = list(range(state, state + 10))
    kitty_indices = list(range(state + 10, state + 13))
    return {
        "sampleType": "exchange-counterfactual-value-v1",
        "schemaVersion": 1,
        "sourceStateKey": f"state-{state}",
        "fixedHandId": f"fixed-{state}",
        "dealSeed": 436000 + state,
        "sourceIndex": state,
        "candidateIndex": candidate,
        "candidateKey": f"state-{state}-candidate-{candidate}",
        "napoleonPlayerId": "player-0",
        "napoleonSeatIndex": 0,
        "contractTargetPointCards": 13,
        "contractSuit": "spades",
        "calledAdjutantCardId": "spades-A",
        "originalHandCardIds": [EXPECTED_CARD_IDS[index] for index in original_indices],
        "kittyPickupCardIds": [EXPECTED_CARD_IDS[index] for index in kitty_indices],
        "pickupHandCardIds": [
            EXPECTED_CARD_IDS[index] for index in original_indices + kitty_indices
        ],
        "modelInput": [float(state % 3)] * EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
        "compactExchangeStateInput": _compact_state(original_indices, kitty_indices),
        "legalDiscardCardMask": _mask(list(range(13))),
        "candidateDiscardCardIds": [EXPECTED_CARD_IDS[index] for index in discard],
        "candidateDiscardMask": _mask(discard),
        "buriedPointCardCount": min(candidate % 4, 3),
        "buriedTrumpCount": candidate % 3,
        "buriedSpecialCards": buried_special,
        "contractMargin": margin,
        "contractSuccess": margin >= 0,
        "napoleonSidePointCards": 13 + margin,
        "napoleonRawReward": 26 if margin >= 0 else 0,
        "napoleonRelativeReward": margin * 2.0,
        "ruleBasedDiscardCardIds": [EXPECTED_CARD_IDS[index] for index in [1, 2, 3]],
        "isRuleBasedAction": candidate == 1,
        "ruleBasedCandidateIndex": 1,
        "hiddenDealChecksum": f"hidden-{state}",
        "biddingHistoryHash": f"history-{state}",
        "biddingHistoryActionCount": 5,
        "invariantChecks": {},
    }


def _write_dataset(directory: Path, *, states: int = 12) -> None:
    rows = [
        _sample(state=state, candidate=candidate)
        for state in range(states)
        for candidate in range(286)
    ]
    shard_bytes = b"".join(
        (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8") for row in rows
    )
    (directory / "shard-00000.jsonl").write_bytes(shard_bytes)
    manifest = {
        "datasetSchemaVersion": 1,
        "generatorVersion": 1,
        "format": "jsonl",
        "sampleType": "exchange-counterfactual-value-v1",
        "sampleSchemaVersion": 1,
        "teacherId": "exchange-discard-combination-playing-rollout-v1",
        "discardActionSpaceId": "exchange-unordered-13c3-discard-combinations-v1",
        "discardCombinationCount": 286,
        "sourceFlow": [],
        "modelInput": {
            "observation": "existing-exchange-observation",
            "featureCount": EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
            "hiddenOpponentHandsIncluded": False,
        },
        "compactExchangeValueInput": {
            "stateFeatureCount": EXCHANGE_COMPACT_STATE_FEATURE_COUNT,
            "candidateDiscardMaskFeatureCount": CARD_COUNT,
            "featureCount": EXCHANGE_COMPACT_VALUE_INPUT_FEATURE_COUNT,
            "layout": [],
            "biddingHistorySemantics": "compact278-bid-owner-table",
        },
        "teacherUsesCompleteHiddenState": True,
        "permutationActionsIncluded": False,
        "startSeed": 436000,
        "endSeed": 436000 + states - 1,
        "sourceStateCount": states,
        "requestedSourceStateCount": states,
        "sampleCount": len(rows),
        "rolloutCount": len(rows),
        "statesPerShard": states,
        "dealAttemptCount": states,
        "sourceCommit": "fixture",
        "playerCount": 5,
        "cardCount": CARD_COUNT,
        "cardIds": list(EXPECTED_CARD_IDS),
        "cardIdsSha256": "fixture",
        "biddingPolicy": {"id": "frozen-raise-v1"},
        "adjutantPolicy": {"id": "rule-based-adjutant-v1"},
        "playingPolicy": {"id": "ppo-separated-v1000"},
        "reward": {},
        "summary": {},
        "shardCount": 1,
        "shards": [
            {
                "file": "shard-00000.jsonl",
                "startSeed": 436000,
                "endSeed": 436000 + states - 1,
                "gameCount": states,
                "sampleCount": len(rows),
                "byteLength": len(shard_bytes),
                "sha256": hashlib.sha256(shard_bytes).hexdigest(),
            }
        ],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_loader_roundtrip_and_input_dimensions(tmp_path: Path) -> None:
    _write_dataset(tmp_path, states=3)

    dataset = load_exchange_counterfactual_dataset(tmp_path)
    sample = dataset.raw_samples[0]

    assert dataset.source_state_count == 3
    assert dataset.sample_count == 858
    assert sample.model_input.shape == (EXCHANGE_MODEL_INPUT_FEATURE_COUNT,)
    assert sample.compact_exchange_state_input is not None
    assert sample.compact_exchange_state_input.shape == (EXCHANGE_COMPACT_STATE_FEATURE_COUNT,)
    assert sample.candidate_discard_mask.shape == (CARD_COUNT,)
    assert sample.value_input.shape == (EXCHANGE_VALUE_INPUT_FEATURE_COUNT,)
    assert sample.compact_value_input.shape == (EXCHANGE_COMPACT_VALUE_INPUT_FEATURE_COUNT,)
    assert int(sample.candidate_discard_mask.sum()) == 3


def test_group_split_guards_state_and_identity_leakage(tmp_path: Path) -> None:
    _write_dataset(tmp_path, states=12)
    dataset = load_exchange_counterfactual_dataset(tmp_path)

    split = create_exchange_value_split(dataset, seed=436, train_state_count=4)

    assert split.train_state_keys.isdisjoint(split.validation_state_keys)
    assert split.train_state_keys.isdisjoint(split.final_state_keys)
    assert split.leakage_guard["status"] == "passed"
    assert {sample.source_state_key for sample in split.train_samples} == split.train_state_keys


def test_model_uses_2671_plus_53_input_dimensions() -> None:
    model = ExchangeValueMlpModel(ExchangeValueMlpConfig(hidden_dims=(8,)))
    output = model(torch.zeros((2, EXCHANGE_VALUE_INPUT_FEATURE_COUNT), dtype=torch.float32))

    assert output.shape == (2,)


def test_model_uses_compact396_input_dimensions() -> None:
    model = ExchangeValueMlpModel(
        ExchangeValueMlpConfig(
            input_dim=EXCHANGE_COMPACT_VALUE_INPUT_FEATURE_COUNT,
            hidden_dims=(8,),
        )
    )
    output = model(
        torch.zeros((2, EXCHANGE_COMPACT_VALUE_INPUT_FEATURE_COUNT), dtype=torch.float32)
    )

    assert output.shape == (2,)


def test_ranking_metrics_and_rule_based_fixture(tmp_path: Path) -> None:
    _write_dataset(tmp_path, states=2)
    dataset = load_exchange_counterfactual_dataset(tmp_path)
    samples = dataset.raw_samples
    predictions = np.asarray(
        [
            100.0 if sample.candidate_index == 0 else -float(sample.candidate_index)
            for sample in samples
        ],
        dtype=np.float32,
    )

    report = exchange_value_evaluation_report(samples, predictions=predictions, split="fixture")

    scalar = cast(dict[str, object], report["scalar"])
    assert scalar["sampleCount"] == 572
    ranking = cast(dict[str, object], report["ranking"])
    assert ranking["exactTopAction"] == 1.0
    assert ranking["top3Hit"] == 1.0
    assert ranking["top5Hit"] == 1.0
    model_regret = cast(dict[str, object], ranking["modelSelectedTeacherMarginRegret"])
    rule_based_regret = cast(dict[str, object], ranking["ruleBasedTeacherMarginRegret"])
    assert model_regret["mean"] == 0.0
    assert cast(float, rule_based_regret["mean"]) > 0.0
    bury_content = cast(dict[str, object], report["buryContent"])
    model_selected = cast(dict[str, object], bury_content["modelSelected"])
    assert model_selected["stateCount"] == 2


def test_training_smoke_checkpoint_save_load_and_deterministic_eval(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_dataset(dataset_dir, states=12)
    dataset = load_exchange_counterfactual_dataset(dataset_dir)

    result = train_exchange_value_model(
        dataset,
        ExchangeValueTrainConfig(
            seed=123,
            epochs=1,
            batch_size=256,
            hidden_dims=(16,),
            train_state_count=4,
            device="cpu",
        ),
    )
    artifact = save_exchange_value_artifact(tmp_path / "artifact", result=result, dataset=dataset)
    loaded_model, checkpoint = load_exchange_value_checkpoint(cast(str, artifact["checkpointPath"]))

    assert checkpoint["modelType"] == "exchange-combination-value"
    assert loaded_model.config.input_dim == EXCHANGE_VALUE_INPUT_FEATURE_COUNT
    assert result.validation_report == result.validation_report
    assert artifact["checkpointSha256"]


def test_compact_training_smoke_checkpoint_save_load(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_dataset(dataset_dir, states=12)
    dataset = load_exchange_counterfactual_dataset(dataset_dir)

    result = train_exchange_value_model(
        dataset,
        ExchangeValueTrainConfig(
            seed=123,
            epochs=1,
            batch_size=256,
            hidden_dims=(16,),
            input_variant="compact396",
            train_state_count=4,
            device="cpu",
        ),
    )
    artifact = save_exchange_value_artifact(tmp_path / "artifact", result=result, dataset=dataset)
    loaded_model, checkpoint = load_exchange_value_checkpoint(cast(str, artifact["checkpointPath"]))

    assert loaded_model.config.input_dim == EXCHANGE_COMPACT_VALUE_INPUT_FEATURE_COUNT
    assert checkpoint["trainingConfig"]["input_variant"] == "compact396"


def test_compact_variant_rejects_legacy_dataset_without_compact_state(tmp_path: Path) -> None:
    _write_dataset(tmp_path, states=12)
    shard = tmp_path / "shard-00000.jsonl"
    rows = [json.loads(line) for line in shard.read_text().splitlines()]
    for row in rows:
        row.pop("compactExchangeStateInput")
    shard_bytes = b"".join(
        (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8") for row in rows
    )
    shard.write_bytes(shard_bytes)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest.pop("compactExchangeValueInput")
    manifest["shards"][0]["byteLength"] = len(shard_bytes)
    manifest["shards"][0]["sha256"] = hashlib.sha256(shard_bytes).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    dataset = load_exchange_counterfactual_dataset(tmp_path)

    with pytest.raises(ValueError, match="compact396 input is unavailable"):
        train_exchange_value_model(
            dataset,
            ExchangeValueTrainConfig(
                seed=123,
                epochs=1,
                batch_size=256,
                hidden_dims=(16,),
                input_variant="compact396",
                train_state_count=4,
                device="cpu",
            ),
        )


def test_loader_rejects_non_three_bit_candidate_mask(tmp_path: Path) -> None:
    _write_dataset(tmp_path, states=1)
    shard = tmp_path / "shard-00000.jsonl"
    rows = [json.loads(line) for line in shard.read_text().splitlines()]
    rows[0]["candidateDiscardMask"] = _mask([0, 1])
    shard_bytes = b"".join(
        (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8") for row in rows
    )
    shard.write_bytes(shard_bytes)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["shards"][0]["byteLength"] = len(shard_bytes)
    manifest["shards"][0]["sha256"] = hashlib.sha256(shard_bytes).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly 3"):
        load_exchange_counterfactual_dataset(tmp_path)
