from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from napoleon_ml.dataset.constants import EXPECTED_CARD_IDS
from napoleon_ml.dataset.errors import SampleValidationError
from napoleon_ml.dataset.reader import iter_samples, iter_tensorized_samples
from napoleon_ml.dataset.sample import parse_sample
from napoleon_ml.dataset.tensors import (
    ADJUTANT_MODEL_INPUT_FEATURE_COUNT,
    BIDDING_MODEL_INPUT_FEATURE_COUNT,
    EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
    tensorize_sample,
)
from napoleon_ml.dataset.validation import calculate_card_ids_sha256, validate_sample

_NONPLAYING_MODEL_INPUT_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "nonplaying_model_input_samples.json"
)
_NONPLAYING_MODEL_INPUT_SHA256 = {
    "bidding": "e4f86e8b9dd5661301e701c71ad6fc167123acedb8e118192e8ccfe6bc6df877",
    "exchange": "f48558692dbd4bf825b0130e9940db4271c2f9653057e5f2186cb36d6d551233",
    "adjutant": "2f0c47b5a113059ed7d06a9966db1ee553163104215981928d6111917d603d52",
}


def _empty_bidding_history() -> dict[str, list[int]]:
    return {
        "actionTypeIndices": [-1] * 117,
        "playerIndices": [-1] * 117,
        "suitIndices": [-1] * 117,
        "targetPointCards": [0] * 117,
        "actionMask": [0] * 117,
    }


def _mask(indices: list[int], *, length: int = 53) -> list[int]:
    values = [0] * length

    for index in indices:
        values[index] = 1

    return values


def _common_sample(
    sample_type: str, observation: dict[str, Any], actor_target: Any
) -> dict[str, Any]:
    return {
        "sampleType": sample_type,
        "schemaVersion": 1,
        "seed": 0,
        "step": 1,
        "actingPlayerId": "player-0",
        "relativePlayerIds": ["player-0", "player-1", "player-2", "player-3", "player-4"],
        "observation": observation,
        "actorTarget": actor_target,
    }


def _bidding_sample() -> dict[str, Any]:
    legal_bid_mask = _mask([0, 5], length=29)
    observation = {
        "schemaVersion": 1,
        "relativePlayerIds": ["player-0", "player-1", "player-2", "player-3", "player-4"],
        "selfHandMask": _mask(list(range(10))),
        "legalBidMask": legal_bid_mask,
        "starterPlayerIndex": 0,
        "highestBidPresent": 0,
        "highestBidPlayerIndex": -1,
        "highestBidSuitIndex": -1,
        "highestBidTargetPointCards": 0,
        "consecutivePassCount": 0,
        "biddingHistory": _empty_bidding_history(),
    }
    return _common_sample("bidding-training-sample", observation, 0)


def _exchange_sample() -> dict[str, Any]:
    self_hand = _mask(list(range(13)))
    observation = {
        "schemaVersion": 1,
        "relativePlayerIds": ["player-0", "player-1", "player-2", "player-3", "player-4"],
        "contractTargetPointCards": 12,
        "trumpSuitOneHot": [1, 0, 0, 0],
        "calledAdjutantCardMask": _mask([20]),
        "selfHandMask": self_hand,
        "legalDiscardCardMask": list(self_hand),
        "handCountByPlayer": [13, 10, 10, 10, 10],
        "specialCardIndices": {"oruma": 0, "yoromeki": 15, "seiJack": 29, "uraJack": 16},
        "biddingHistory": _empty_bidding_history(),
    }
    return _common_sample(
        "exchange-training-sample", observation, {"discardTargetMask": _mask([0, 1, 2])}
    )


def _adjutant_sample() -> dict[str, Any]:
    legal_adjutant_mask = _mask([20, 21])
    observation = {
        "schemaVersion": 1,
        "relativePlayerIds": ["player-0", "player-1", "player-2", "player-3", "player-4"],
        "trumpSuitOneHot": [1, 0, 0, 0],
        "contractTargetPointCards": 12,
        "selfHandMask": _mask(list(range(10))),
        "legalAdjutantMask": legal_adjutant_mask,
        "specialCardIndices": {"oruma": 0, "yoromeki": 15, "seiJack": 29, "uraJack": 16},
        "biddingHistory": _empty_bidding_history(),
    }
    return _common_sample("adjutant-training-sample", observation, 20)


@pytest.mark.parametrize(
    ("sample_type", "factory", "feature_count"),
    [
        ("bidding-training-sample", _bidding_sample, BIDDING_MODEL_INPUT_FEATURE_COUNT),
        ("exchange-training-sample", _exchange_sample, EXCHANGE_MODEL_INPUT_FEATURE_COUNT),
        ("adjutant-training-sample", _adjutant_sample, ADJUTANT_MODEL_INPUT_FEATURE_COUNT),
    ],
)
def test_multiphase_sample_parse_validate_tensorize_smoke(
    sample_type: str, factory: Any, feature_count: int
) -> None:
    raw = factory()
    sample = parse_sample(raw, sample_type=sample_type)
    validate_sample(sample)
    tensorized = tensorize_sample(sample)

    assert tensorized.model_input.shape == (feature_count,)
    assert tensorized.model_input.dtype == np.float32
    assert (
        tensorized.model_input.tobytes()
        == tensorize_sample(parse_sample(raw)).model_input.tobytes()
    )


def test_nonplaying_model_input_shared_fixture_sha256_parity() -> None:
    fixture = json.loads(_NONPLAYING_MODEL_INPUT_FIXTURE_PATH.read_text(encoding="utf-8"))

    for phase, expected_sha256 in _NONPLAYING_MODEL_INPUT_SHA256.items():
        sample = parse_sample(fixture[phase])
        validate_sample(sample)
        model_input = tensorize_sample(sample).model_input

        assert hashlib.sha256(model_input.tobytes()).hexdigest() == expected_sha256


def test_bidding_rejects_illegal_target_and_bad_highest_bid_sentinel() -> None:
    raw = _bidding_sample()
    raw["actorTarget"] = 1

    with pytest.raises(SampleValidationError, match="legalBidMask"):
        validate_sample(parse_sample(raw))

    raw = _bidding_sample()
    raw["observation"]["highestBidPlayerIndex"] = 0

    with pytest.raises(SampleValidationError, match="highest bid"):
        validate_sample(parse_sample(raw))


def test_bidding_rejects_wrong_self_hand_count() -> None:
    raw = _bidding_sample()
    raw["observation"]["selfHandMask"] = _mask(list(range(9)))

    with pytest.raises(SampleValidationError, match="selfHandMask"):
        validate_sample(parse_sample(raw))


def test_exchange_rejects_wrong_masks_and_target_subset() -> None:
    raw = _exchange_sample()
    raw["observation"]["selfHandMask"] = _mask(list(range(12)))

    with pytest.raises(SampleValidationError, match="selfHandMask"):
        validate_sample(parse_sample(raw))

    raw = _exchange_sample()
    raw["actorTarget"]["discardTargetMask"] = _mask([0, 1, 30])

    with pytest.raises(SampleValidationError, match="legalDiscardCardMask"):
        validate_sample(parse_sample(raw))


def test_exchange_rejects_empty_special_card_sentinel() -> None:
    raw = _exchange_sample()
    raw["observation"]["specialCardIndices"]["seiJack"] = -1

    with pytest.raises(SampleValidationError, match="specialCardIndices.seiJack"):
        validate_sample(parse_sample(raw))


def test_adjutant_rejects_wrong_mask_shape_and_illegal_target() -> None:
    raw = _adjutant_sample()
    raw["observation"]["legalAdjutantMask"] = raw["observation"]["legalAdjutantMask"][:-1]

    with pytest.raises(SampleValidationError, match="legalAdjutantMask"):
        parse_sample(raw)

    raw = _adjutant_sample()
    raw["actorTarget"] = 22

    with pytest.raises(SampleValidationError, match="legalAdjutantMask"):
        validate_sample(parse_sample(raw))


def test_adjutant_rejects_empty_special_card_sentinel() -> None:
    raw = _adjutant_sample()
    raw["observation"]["specialCardIndices"]["seiJack"] = -1

    with pytest.raises(SampleValidationError, match="specialCardIndices.seiJack"):
        validate_sample(parse_sample(raw))


@pytest.mark.parametrize(
    ("factory", "first_target", "second_target"),
    [
        (_bidding_sample, 0, 5),
        (
            _exchange_sample,
            {"discardTargetMask": _mask([0, 1, 2])},
            {"discardTargetMask": _mask([3, 4, 5])},
        ),
        (_adjutant_sample, 20, 21),
    ],
)
def test_teacher_target_is_not_in_model_input(
    factory: Any, first_target: Any, second_target: Any
) -> None:
    first_raw = factory()
    second_raw = factory()
    first_raw["actorTarget"] = first_target
    second_raw["actorTarget"] = second_target

    first = tensorize_sample(parse_sample(first_raw)).model_input
    second = tensorize_sample(parse_sample(second_raw)).model_input

    assert first.tobytes() == second.tobytes()


def test_manifest_sample_type_mismatch_is_rejected(tmp_path: Path) -> None:
    row = _bidding_sample()
    row["sampleType"] = "adjutant-training-sample"
    shard_bytes = (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8")
    (tmp_path / "shard-00000.jsonl").write_bytes(shard_bytes)
    manifest = _manifest(
        "bidding-training-sample", len(shard_bytes), hashlib.sha256(shard_bytes).hexdigest()
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SampleValidationError, match="sampleType mismatch"):
        list(iter_samples(tmp_path))


def test_multiphase_reader_tensorizes_by_manifest_sample_type(tmp_path: Path) -> None:
    row = _exchange_sample()
    shard_bytes = (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8")
    (tmp_path / "shard-00000.jsonl").write_bytes(shard_bytes)
    manifest = _manifest(
        "exchange-training-sample", len(shard_bytes), hashlib.sha256(shard_bytes).hexdigest()
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    [tensorized] = list(iter_tensorized_samples(tmp_path))

    assert tensorized.model_input.shape == (EXCHANGE_MODEL_INPUT_FEATURE_COUNT,)


def _manifest(sample_type: str, byte_length: int, sha256: str) -> dict[str, Any]:
    return {
        "datasetSchemaVersion": 2,
        "generatorVersion": 2,
        "encoderSchemaVersion": 1,
        "format": "jsonl",
        "sampleType": sample_type,
        "agent": {"type": "rule-based", "version": 1},
        "startSeed": 0,
        "endSeed": 0,
        "gameCount": 1,
        "sampleCount": 1,
        "gamesPerShard": 1,
        "shardCount": 1,
        "playerCount": 5,
        "cardCount": 53,
        "cardIds": list(EXPECTED_CARD_IDS),
        "cardIdsSha256": calculate_card_ids_sha256(),
        "shards": [
            {
                "file": "shard-00000.jsonl",
                "startSeed": 0,
                "endSeed": 0,
                "gameCount": 1,
                "sampleCount": 1,
                "byteLength": byte_length,
                "sha256": sha256,
            }
        ],
    }
