from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import napoleon_ml.dataset.pytorch as pytorch_module
from napoleon_ml.dataset.constants import CARD_COUNT, EXPECTED_CARD_IDS
from napoleon_ml.dataset.errors import DatasetError
from napoleon_ml.dataset.pytorch import (
    BiddingIterableDataset,
    PlayingIterableDataset,
    create_bidding_dataloader,
    create_playing_dataloader,
    create_training_dataloader,
)
from napoleon_ml.dataset.sample import parse_sample
from napoleon_ml.dataset.split import DatasetSplit, SplitConfig
from napoleon_ml.dataset.tensors import (
    ADJUTANT_MODEL_INPUT_FEATURE_COUNT,
    BIDDING_MODEL_INPUT_FEATURE_COUNT,
    EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
    MODEL_INPUT_FEATURE_COUNT,
    TensorizedBiddingSample,
    TensorizedPlayingSample,
    tensorize_sample,
)
from napoleon_ml.dataset.validation import calculate_card_ids_sha256

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "valid_sample.json"


def _load_valid_sample() -> dict[str, Any]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _one_line(sample: dict[str, Any]) -> bytes:
    return (json.dumps(sample) + "\n").encode("utf-8")


def _samples_by_seed(*, seeds: tuple[int, ...], steps_per_seed: int) -> Iterator[dict[str, Any]]:
    base = _load_valid_sample()

    for seed in seeds:
        for step in range(1, steps_per_seed + 1):
            yield dict(base, seed=seed, step=step)


def _write_dataset(directory: Path, samples: list[dict[str, Any]]) -> None:
    shard_bytes = b"".join(_one_line(sample) for sample in samples)
    shard_path = directory / "shard-00000.jsonl"
    shard_path.write_bytes(shard_bytes)

    seeds = [sample["seed"] for sample in samples]
    unique_seed_runs = 0
    previous_seed: int | None = None

    for seed in seeds:
        if seed != previous_seed:
            unique_seed_runs += 1
            previous_seed = seed

    manifest = {
        "datasetSchemaVersion": 1,
        "generatorVersion": 1,
        "playingEncoderSchemaVersion": 1,
        "format": "jsonl",
        "sampleType": "playing-training-sample",
        "agent": {"type": "rule-based", "version": 1},
        "startSeed": seeds[0],
        "endSeed": seeds[-1],
        "gameCount": unique_seed_runs,
        "sampleCount": len(samples),
        "gamesPerShard": unique_seed_runs,
        "shardCount": 1,
        "playerCount": 5,
        "cardCount": CARD_COUNT,
        "cardIds": list(EXPECTED_CARD_IDS),
        "cardIdsSha256": calculate_card_ids_sha256(),
        "shards": [
            {
                "file": "shard-00000.jsonl",
                "startSeed": seeds[0],
                "endSeed": seeds[-1],
                "gameCount": unique_seed_runs,
                "sampleCount": len(samples),
                "byteLength": len(shard_bytes),
                "sha256": hashlib.sha256(shard_bytes).hexdigest(),
            }
        ],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _empty_bidding_history() -> dict[str, list[int]]:
    return {
        "actionTypeIndices": [-1] * 117,
        "playerIndices": [-1] * 117,
        "suitIndices": [-1] * 117,
        "targetPointCards": [0] * 117,
        "actionMask": [0] * 117,
    }


def _mask(indices: list[int], *, length: int = CARD_COUNT) -> list[int]:
    values = [0] * length

    for index in indices:
        values[index] = 1

    return values


def _common_multiphase_sample(
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
    observation = {
        "schemaVersion": 1,
        "relativePlayerIds": ["player-0", "player-1", "player-2", "player-3", "player-4"],
        "selfHandMask": _mask(list(range(10))),
        "legalBidMask": _mask([0, 5], length=29),
        "starterPlayerIndex": 0,
        "highestBidPresent": 0,
        "highestBidPlayerIndex": -1,
        "highestBidSuitIndex": -1,
        "highestBidTargetPointCards": 0,
        "consecutivePassCount": 0,
        "biddingHistory": _empty_bidding_history(),
    }
    return _common_multiphase_sample("bidding-training-sample", observation, 0)


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
    return _common_multiphase_sample(
        "exchange-training-sample", observation, {"discardTargetMask": _mask([0, 1, 2])}
    )


def _adjutant_sample() -> dict[str, Any]:
    observation = {
        "schemaVersion": 1,
        "relativePlayerIds": ["player-0", "player-1", "player-2", "player-3", "player-4"],
        "trumpSuitOneHot": [1, 0, 0, 0],
        "contractTargetPointCards": 12,
        "selfHandMask": _mask(list(range(10))),
        "legalAdjutantMask": _mask([20, 21]),
        "specialCardIndices": {"oruma": 0, "yoromeki": 15, "seiJack": 29, "uraJack": 16},
        "biddingHistory": _empty_bidding_history(),
    }
    return _common_multiphase_sample("adjutant-training-sample", observation, 20)


def _write_multiphase_dataset(directory: Path, sample: dict[str, Any]) -> None:
    shard_bytes = _one_line(sample)
    (directory / "shard-00000.jsonl").write_bytes(shard_bytes)
    manifest = {
        "datasetSchemaVersion": 2,
        "generatorVersion": 2,
        "encoderSchemaVersion": 1,
        "format": "jsonl",
        "sampleType": sample["sampleType"],
        "agent": {"type": "rule-based", "version": 1},
        "startSeed": sample["seed"],
        "endSeed": sample["seed"],
        "gameCount": 1,
        "sampleCount": 1,
        "gamesPerShard": 1,
        "shardCount": 1,
        "playerCount": 5,
        "cardCount": CARD_COUNT,
        "cardIds": list(EXPECTED_CARD_IDS),
        "cardIdsSha256": calculate_card_ids_sha256(),
        "shards": [
            {
                "file": "shard-00000.jsonl",
                "startSeed": sample["seed"],
                "endSeed": sample["seed"],
                "gameCount": 1,
                "sampleCount": 1,
                "byteLength": len(shard_bytes),
                "sha256": hashlib.sha256(shard_bytes).hexdigest(),
            }
        ],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_dataloader_filters_split_and_batches_fixed_shapes(tmp_path: Path) -> None:
    _write_dataset(tmp_path, list(_samples_by_seed(seeds=(0, 1, 2), steps_per_seed=2)))
    split_config = SplitConfig(train=1, validation=1, test=98)

    loader = create_playing_dataloader(
        tmp_path,
        split=DatasetSplit.VALIDATION,
        split_config=split_config,
        batch_size=2,
    )
    batch = next(iter(loader))

    assert set(batch) == {
        "model_input",
        "actor_target",
        "legal_play_mask",
        "belief_target",
        "belief_hidden_ownership_loss_mask",
        "seed",
        "step",
    }
    assert batch["model_input"].shape == (2, MODEL_INPUT_FEATURE_COUNT)
    assert batch["model_input"].dtype == torch.float32
    assert batch["actor_target"].shape == (2,)
    assert batch["actor_target"].dtype == torch.int64
    assert batch["legal_play_mask"].shape == (2, CARD_COUNT)
    assert batch["legal_play_mask"].dtype == torch.bool
    assert batch["legal_play_mask"][torch.arange(2), batch["actor_target"]].all()
    assert batch["belief_target"].shape == (2, CARD_COUNT)
    assert batch["belief_target"].dtype == torch.int64
    assert batch["belief_hidden_ownership_loss_mask"].shape == (2, CARD_COUNT)
    assert batch["belief_hidden_ownership_loss_mask"].dtype == torch.bool
    assert batch["seed"].dtype == torch.int64
    assert batch["step"].dtype == torch.int64
    assert batch["seed"].tolist() == [1, 1]
    assert batch["step"].tolist() == [1, 2]


def test_iterable_dataset_restarts_from_front_each_epoch(tmp_path: Path) -> None:
    _write_dataset(tmp_path, list(_samples_by_seed(seeds=(0, 1), steps_per_seed=2)))
    dataset = PlayingIterableDataset(tmp_path, split=DatasetSplit.TRAIN)

    first_epoch = [(int(sample["seed"]), int(sample["step"])) for sample in dataset]
    second_epoch = [(int(sample["seed"]), int(sample["step"])) for sample in dataset]

    assert first_epoch == [(0, 1), (0, 2), (1, 1), (1, 2)]
    assert second_epoch == first_epoch


def test_same_configuration_produces_same_batch_order(tmp_path: Path) -> None:
    _write_dataset(tmp_path, list(_samples_by_seed(seeds=(0, 1, 2), steps_per_seed=2)))
    split_config = SplitConfig(train=1, validation=1, test=98)

    first = list(
        create_playing_dataloader(
            tmp_path,
            split="test",
            split_config=split_config,
            batch_size=1,
        )
    )
    second = list(
        create_playing_dataloader(
            tmp_path,
            split="test",
            split_config=split_config,
            batch_size=1,
        )
    )

    assert [batch["seed"].tolist() for batch in first] == [[2], [2]]
    assert [batch["step"].tolist() for batch in first] == [[1], [2]]
    assert [batch["seed"].tolist() for batch in second] == [[2], [2]]
    assert [batch["step"].tolist() for batch in second] == [[1], [2]]
    assert first[0]["model_input"].equal(second[0]["model_input"])


def test_empty_split_produces_no_batches(tmp_path: Path) -> None:
    _write_dataset(tmp_path, list(_samples_by_seed(seeds=(0, 1), steps_per_seed=1)))

    loader = create_playing_dataloader(
        tmp_path,
        split=DatasetSplit.VALIDATION,
        split_config=SplitConfig(train=100, validation=0, test=0),
        batch_size=2,
    )

    assert list(loader) == []


def test_mask_dtype_can_be_uint8(tmp_path: Path) -> None:
    _write_dataset(tmp_path, list(_samples_by_seed(seeds=(0,), steps_per_seed=1)))

    loader = create_playing_dataloader(
        tmp_path,
        split=DatasetSplit.TRAIN,
        batch_size=1,
        mask_dtype=torch.uint8,
    )
    batch = next(iter(loader))

    assert batch["belief_hidden_ownership_loss_mask"].dtype == torch.uint8
    assert batch["legal_play_mask"].dtype == torch.bool


@pytest.mark.parametrize(
    ("sample_factory", "expected_fields", "model_input_width", "mask_field", "target_field"),
    [
        (
            _bidding_sample,
            {"model_input", "legal_bid_mask", "actor_target", "seed", "step"},
            BIDDING_MODEL_INPUT_FEATURE_COUNT,
            "legal_bid_mask",
            "actor_target",
        ),
        (
            _exchange_sample,
            {"model_input", "legal_discard_card_mask", "discard_target_mask", "seed", "step"},
            EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
            "legal_discard_card_mask",
            "discard_target_mask",
        ),
        (
            _adjutant_sample,
            {"model_input", "legal_adjutant_mask", "actor_target", "seed", "step"},
            ADJUTANT_MODEL_INPUT_FEATURE_COUNT,
            "legal_adjutant_mask",
            "actor_target",
        ),
    ],
)
def test_training_dataloader_selects_multiphase_batch_from_manifest(
    tmp_path: Path,
    sample_factory: Any,
    expected_fields: set[str],
    model_input_width: int,
    mask_field: str,
    target_field: str,
) -> None:
    _write_multiphase_dataset(tmp_path, sample_factory())

    loader = create_training_dataloader(tmp_path, split=DatasetSplit.TRAIN, batch_size=1)
    batch = next(iter(loader))

    assert set(batch) == expected_fields
    assert batch["model_input"].shape == (1, model_input_width)
    assert batch["model_input"].dtype == torch.float32
    assert batch[mask_field].shape == (1, CARD_COUNT if mask_field != "legal_bid_mask" else 29)
    assert batch[mask_field].dtype == torch.bool
    assert batch["seed"].shape == (1,)
    assert batch["seed"].dtype == torch.int64
    assert batch["step"].shape == (1,)
    assert batch["step"].dtype == torch.int64

    if target_field == "discard_target_mask":
        assert batch[target_field].shape == (1, CARD_COUNT)
        assert batch[target_field].dtype == torch.bool
        assert batch[mask_field].sum(dim=1).tolist() == [13]
        assert batch[target_field].sum(dim=1).tolist() == [3]
        assert torch.logical_or(torch.logical_not(batch[target_field]), batch[mask_field]).all()
    else:
        assert batch[target_field].shape == (1,)
        assert batch[target_field].dtype == torch.int64
        assert batch[mask_field][torch.arange(1), batch[target_field]].all()


def test_specific_multiphase_dataloader_rejects_different_manifest_sample_type(
    tmp_path: Path,
) -> None:
    _write_multiphase_dataset(tmp_path, _exchange_sample())

    loader = create_bidding_dataloader(tmp_path, split=DatasetSplit.TRAIN, batch_size=1)

    with pytest.raises(DatasetError, match="requires a bidding-training-sample dataset"):
        next(iter(loader))


def test_training_dataloader_rejects_non_playing_mask_dtype_override(tmp_path: Path) -> None:
    _write_multiphase_dataset(tmp_path, _bidding_sample())

    with pytest.raises(DatasetError, match="non-playing DataLoader masks are always torch.bool"):
        create_training_dataloader(
            tmp_path,
            split=DatasetSplit.TRAIN,
            batch_size=1,
            mask_dtype=torch.uint8,
        )


def test_bidding_dataloader_rejects_in_range_actor_target_missing_from_legal_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_multiphase_dataset(tmp_path, _bidding_sample())
    tensorized = tensorize_sample(parse_sample(_bidding_sample()))
    assert isinstance(tensorized, TensorizedBiddingSample)
    legal_bid_mask = tensorized.legal_bid_mask.copy()
    legal_bid_mask[int(tensorized.actor_target)] = 0
    bad = dataclasses.replace(tensorized, legal_bid_mask=legal_bid_mask)

    def _fake_iter_tensorized_samples(
        *args: object, **kwargs: object
    ) -> Iterator[TensorizedBiddingSample]:
        yield bad

    monkeypatch.setattr(pytorch_module, "iter_tensorized_samples", _fake_iter_tensorized_samples)

    dataset = BiddingIterableDataset(tmp_path, split=DatasetSplit.TRAIN)

    with pytest.raises(DatasetError, match="actor_target must be legal"):
        next(iter(dataset))


def test_dataloader_rejects_actor_target_outside_legal_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_dataset(tmp_path, [_load_valid_sample()])
    tensorized = tensorize_sample(parse_sample(_load_valid_sample()))
    bad = dataclasses.replace(tensorized, actor_target=np.int64(-1))

    def _fake_iter_tensorized_samples(
        *args: object, **kwargs: object
    ) -> Iterator[TensorizedPlayingSample]:
        yield bad

    monkeypatch.setattr(pytorch_module, "iter_tensorized_samples", _fake_iter_tensorized_samples)

    dataset = PlayingIterableDataset(tmp_path, split=DatasetSplit.TRAIN)

    with pytest.raises(DatasetError, match="actor_target must be between"):
        next(iter(dataset))


def test_dataloader_rejects_in_range_actor_target_missing_from_legal_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_dataset(tmp_path, [_load_valid_sample()])
    tensorized = tensorize_sample(parse_sample(_load_valid_sample()))
    actor_target = int(tensorized.actor_target)
    legal_play_mask = tensorized.legal_play_mask.copy()
    legal_play_mask[actor_target] = 0
    bad = dataclasses.replace(tensorized, legal_play_mask=legal_play_mask)

    def _fake_iter_tensorized_samples(
        *args: object, **kwargs: object
    ) -> Iterator[TensorizedPlayingSample]:
        yield bad

    monkeypatch.setattr(pytorch_module, "iter_tensorized_samples", _fake_iter_tensorized_samples)

    dataset = PlayingIterableDataset(tmp_path, split=DatasetSplit.TRAIN)

    with pytest.raises(DatasetError, match="actor_target must be legal"):
        next(iter(dataset))


def test_invalid_pytorch_loader_configuration_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="split must be one of"):
        PlayingIterableDataset(tmp_path, split="dev")

    with pytest.raises(DatasetError, match="split_config"):
        PlayingIterableDataset(
            tmp_path,
            split=DatasetSplit.TRAIN,
            split_config=object(),  # type: ignore[arg-type]
        )

    with pytest.raises(DatasetError, match="batch_size"):
        create_playing_dataloader(tmp_path, split=DatasetSplit.TRAIN, batch_size=0)

    with pytest.raises(DatasetError, match="num_workers=0"):
        create_playing_dataloader(
            tmp_path,
            split=DatasetSplit.TRAIN,
            batch_size=1,
            num_workers=1,
        )

    with pytest.raises(DatasetError, match="mask_dtype"):
        PlayingIterableDataset(tmp_path, split=DatasetSplit.TRAIN, mask_dtype=torch.float32)
