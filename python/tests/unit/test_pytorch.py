from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

from napoleon_ml.dataset.constants import CARD_COUNT, EXPECTED_CARD_IDS
from napoleon_ml.dataset.errors import DatasetError
from napoleon_ml.dataset.pytorch import PlayingIterableDataset, create_playing_dataloader
from napoleon_ml.dataset.split import DatasetSplit, SplitConfig
from napoleon_ml.dataset.tensors import MODEL_INPUT_FEATURE_COUNT
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
        "belief_target",
        "belief_hidden_ownership_loss_mask",
        "seed",
        "step",
    }
    assert batch["model_input"].shape == (2, MODEL_INPUT_FEATURE_COUNT)
    assert batch["model_input"].dtype == torch.float32
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
