from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from napoleon_ml.bidding.checkpoint import (
    BiddingCheckpointCompatibilityError,
    load_bidding_checkpoint,
)
from napoleon_ml.bidding.metrics import (
    bidding_action_suit,
    bidding_action_target_point_cards,
    evaluate_bidding_model,
    mask_illegal_bidding_logits,
    masked_bidding_accuracy,
    masked_bidding_cross_entropy,
    select_bidding_action,
)
from napoleon_ml.bidding.model import (
    BiddingMlpConfig,
    BiddingMlpModel,
    create_seeded_bidding_model,
)
from napoleon_ml.cli.evaluate_bidding_mlp import main as evaluate_main
from napoleon_ml.cli.train_bidding_mlp import main as train_main
from napoleon_ml.dataset.constants import (
    BIDDING_ACTION_COUNT,
    BIDDING_DATASET_SAMPLE_TYPE,
    CARD_COUNT,
    EXPECTED_CARD_IDS,
    MAX_BIDDING_ACTION_COUNT,
)
from napoleon_ml.dataset.pytorch import create_bidding_dataloader
from napoleon_ml.dataset.reader import load_manifest
from napoleon_ml.dataset.split import DatasetSplit, SplitConfig, split_for_seed
from napoleon_ml.dataset.tensors import BIDDING_MODEL_INPUT_FEATURE_COUNT
from napoleon_ml.dataset.validation import calculate_card_ids_sha256


def _empty_bidding_history() -> dict[str, list[int]]:
    return {
        "actionTypeIndices": [-1] * MAX_BIDDING_ACTION_COUNT,
        "playerIndices": [-1] * MAX_BIDDING_ACTION_COUNT,
        "suitIndices": [-1] * MAX_BIDDING_ACTION_COUNT,
        "targetPointCards": [0] * MAX_BIDDING_ACTION_COUNT,
        "actionMask": [0] * MAX_BIDDING_ACTION_COUNT,
    }


def _mask(indices: list[int], *, length: int) -> list[int]:
    values = [0] * length
    for index in indices:
        values[index] = 1
    return values


def _bidding_sample(*, seed: int, actor_target: int, legal_actions: list[int]) -> dict[str, Any]:
    relative_player_ids = [f"player-{index}" for index in range(5)]
    return {
        "sampleType": BIDDING_DATASET_SAMPLE_TYPE,
        "schemaVersion": 1,
        "seed": seed,
        "step": 1,
        "actingPlayerId": "player-0",
        "relativePlayerIds": relative_player_ids,
        "observation": {
            "schemaVersion": 1,
            "relativePlayerIds": relative_player_ids,
            "selfHandMask": _mask(list(range(10)), length=CARD_COUNT),
            "legalBidMask": _mask(legal_actions, length=BIDDING_ACTION_COUNT),
            "starterPlayerIndex": 0,
            "highestBidPresent": 0,
            "highestBidPlayerIndex": -1,
            "highestBidSuitIndex": -1,
            "highestBidTargetPointCards": 0,
            "consecutivePassCount": 0,
            "biddingHistory": _empty_bidding_history(),
        },
        "actorTarget": actor_target,
    }


def _one_line(sample: dict[str, Any]) -> bytes:
    return (json.dumps(sample) + "\n").encode("utf-8")


def _write_bidding_dataset(
    directory: Path,
    *,
    seeds: tuple[int, ...],
    legal_actions_by_sample: tuple[list[int], ...] | None = None,
) -> None:
    actor_targets = (0, 1, 5, 2, 9)
    legal_actions = (
        legal_actions_by_sample
        if legal_actions_by_sample is not None
        else tuple([0, 1, 2, 5, 9] for _ in seeds)
    )
    if len(legal_actions) != len(seeds):
        raise ValueError("legal_actions_by_sample must match seeds length.")

    samples = [
        _bidding_sample(
            seed=seed,
            actor_target=actor_targets[index % len(actor_targets)],
            legal_actions=legal_actions[index],
        )
        for index, seed in enumerate(seeds)
    ]
    shard_bytes = b"".join(_one_line(sample) for sample in samples)
    (directory / "shard-00000.jsonl").write_bytes(shard_bytes)

    manifest = {
        "datasetSchemaVersion": 2,
        "generatorVersion": 2,
        "encoderSchemaVersion": 1,
        "format": "jsonl",
        "sampleType": BIDDING_DATASET_SAMPLE_TYPE,
        "agent": {"type": "rule-based", "version": 1},
        "startSeed": seeds[0],
        "endSeed": seeds[-1],
        "gameCount": len(seeds),
        "sampleCount": len(samples),
        "gamesPerShard": len(seeds),
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
                "gameCount": len(seeds),
                "sampleCount": len(samples),
                "byteLength": len(shard_bytes),
                "sha256": hashlib.sha256(shard_bytes).hexdigest(),
            }
        ],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_bidding_mlp_outputs_action_logits() -> None:
    model = BiddingMlpModel(BiddingMlpConfig(hidden_dim=8, hidden_layers=1))
    logits = model(torch.zeros((2, BIDDING_MODEL_INPUT_FEATURE_COUNT), dtype=torch.float32))

    assert logits.shape == (2, BIDDING_ACTION_COUNT)


def test_seeded_bidding_model_initialization_is_reproducible() -> None:
    config = BiddingMlpConfig(hidden_dim=8, hidden_layers=1)
    first = create_seeded_bidding_model(config, seed=123)
    second = create_seeded_bidding_model(config, seed=123)

    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters(), strict=True
    ):
        assert first_parameter.equal(second_parameter)


def test_bidding_masking_loss_accuracy_and_inference_exclude_illegal_actions() -> None:
    logits = torch.zeros((2, BIDDING_ACTION_COUNT), dtype=torch.float32)
    target = torch.tensor([1, 5], dtype=torch.long)
    legal_mask = torch.zeros((2, BIDDING_ACTION_COUNT), dtype=torch.bool)
    legal_mask[0, [1, 2]] = True
    legal_mask[1, [3, 5, 7]] = True

    logits[0, 0] = 100.0
    logits[0, 1] = 5.0
    logits[1, 28] = 100.0
    logits[1, 3] = 7.0

    masked = mask_illegal_bidding_logits(logits, legal_mask)
    actions = select_bidding_action(logits, legal_mask)
    loss = masked_bidding_cross_entropy(logits, target, legal_mask)

    assert torch.isfinite(loss)
    assert masked[0, 0] == torch.finfo(torch.float32).min
    assert actions.tolist() == [1, 3]
    assert bool(legal_mask.gather(1, actions.unsqueeze(1)).all().item())
    assert masked_bidding_accuracy(logits, target, legal_mask) == 0.5


def test_bidding_loss_rejects_illegal_target() -> None:
    logits = torch.zeros((1, BIDDING_ACTION_COUNT), dtype=torch.float32)
    target = torch.tensor([2], dtype=torch.long)
    legal_mask = torch.zeros((1, BIDDING_ACTION_COUNT), dtype=torch.bool)
    legal_mask[0, 1] = True

    with pytest.raises(ValueError, match="target must be legal"):
        masked_bidding_cross_entropy(logits, target, legal_mask)


@pytest.mark.parametrize(
    "operation",
    (mask_illegal_bidding_logits, select_bidding_action),
)
def test_bidding_masking_rejects_rows_without_legal_actions(operation: Any) -> None:
    logits = torch.zeros((1, BIDDING_ACTION_COUNT), dtype=torch.float32)
    legal_mask = torch.zeros((1, BIDDING_ACTION_COUNT), dtype=torch.bool)

    with pytest.raises(ValueError, match="at least one legal action"):
        operation(logits, legal_mask)


@pytest.mark.parametrize(
    "config_factory",
    (
        lambda: BiddingMlpConfig(input_dim=0),
        lambda: BiddingMlpConfig(hidden_dim=0),
        lambda: BiddingMlpConfig(hidden_layers=0),
        lambda: BiddingMlpConfig(dropout=1.0),
    ),
)
def test_bidding_config_rejects_invalid_values(
    config_factory: Callable[[], BiddingMlpConfig],
) -> None:
    with pytest.raises(ValueError):
        config_factory()


def test_bidding_action_breakdown_decoding() -> None:
    assert bidding_action_target_point_cards(0) is None
    assert bidding_action_suit(0) is None
    assert bidding_action_target_point_cards(1) == 13
    assert bidding_action_suit(1) == "spades"
    assert bidding_action_target_point_cards(5) == 14
    assert bidding_action_suit(5) == "spades"
    assert bidding_action_target_point_cards(28) == 19
    assert bidding_action_suit(28) == "clubs"


def test_bidding_evaluation_reports_required_breakdowns_and_uniform_baseline(
    tmp_path: Path,
) -> None:
    _write_bidding_dataset(tmp_path, seeds=(0, 1, 2))
    loader = create_bidding_dataloader(
        tmp_path,
        split=DatasetSplit.TRAIN,
        split_config=SplitConfig(train=100, validation=0, test=0),
        batch_size=3,
    )
    model = BiddingMlpModel(BiddingMlpConfig(hidden_dim=8, hidden_layers=1))

    report = evaluate_bidding_model(model, loader, split=DatasetSplit.TRAIN.value)

    assert report.sample_count == 3
    assert report.illegal_prediction_count == 0
    assert report.top1.count == 3
    assert report.top1.legal_uniform_accuracy == pytest.approx(0.2)
    assert report.pass_actions.count == 1
    assert report.bid_actions.count == 2
    assert report.target_point_cards["13"].count == 1
    assert report.target_point_cards["14"].count == 1
    assert report.suits["spades"].count == 2


def test_bidding_evaluation_uses_per_row_legal_counts_for_uniform_baseline(
    tmp_path: Path,
) -> None:
    _write_bidding_dataset(
        tmp_path,
        seeds=(0, 1, 2),
        legal_actions_by_sample=([0, 1], [0, 1, 2, 5, 9], [0, 5, 9]),
    )
    loader = create_bidding_dataloader(
        tmp_path,
        split=DatasetSplit.TRAIN,
        split_config=SplitConfig(train=100, validation=0, test=0),
        batch_size=3,
    )
    model = BiddingMlpModel(BiddingMlpConfig(hidden_dim=8, hidden_layers=1))

    report = evaluate_bidding_model(model, loader, split=DatasetSplit.TRAIN.value)

    assert report.masked_loss is not None
    assert report.masked_loss > 0.0
    assert report.illegal_prediction_count == 0
    assert report.top1.legal_uniform_accuracy == pytest.approx((1 / 2 + 1 / 5 + 1 / 3) / 3)


def test_train_cli_saves_checkpoint_and_evaluate_cli_loads_test_split(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_bidding_dataset(tmp_path, seeds=(0, 1, 2))
    checkpoint_path = tmp_path.parent / f"{tmp_path.name}-bidding.pt"
    split_config = SplitConfig(train=1, validation=1, test=98)

    assert split_for_seed(0, split_config) == DatasetSplit.TRAIN
    assert split_for_seed(1, split_config) == DatasetSplit.VALIDATION
    assert split_for_seed(2, split_config) == DatasetSplit.TEST

    train_exit = train_main(
        [
            str(tmp_path),
            "--output",
            str(checkpoint_path),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--hidden-dim",
            "8",
            "--hidden-layers",
            "1",
            "--train-ratio",
            "1",
            "--validation-ratio",
            "1",
            "--test-ratio",
            "98",
            "--json",
        ]
    )

    assert train_exit == 0
    assert checkpoint_path.is_file()
    train_report = json.loads(capsys.readouterr().out)
    assert train_report["finalValidation"]["model"]["top1"]["count"] == 1
    assert train_report["finalValidation"]["baselineLegalUniform"]["top1"][
        "accuracy"
    ] == pytest.approx(0.2)
    assert train_report["finalValidation"]["model"]["illegalPredictionCount"] == 0

    evaluate_exit = evaluate_main(
        [
            str(tmp_path),
            "--checkpoint",
            str(checkpoint_path),
            "--split",
            "test",
            "--batch-size",
            "1",
            "--train-ratio",
            "1",
            "--validation-ratio",
            "1",
            "--test-ratio",
            "98",
            "--json",
        ]
    )

    assert evaluate_exit == 0
    evaluate_report = json.loads(capsys.readouterr().out)
    assert evaluate_report["split"] == "test"
    assert evaluate_report["model"]["top1"]["count"] == 1
    assert evaluate_report["model"]["illegalPredictionCount"] == 0
    assert evaluate_report["model"]["targetPointCards"]["14"]["count"] == 1


def test_evaluate_cli_loads_bidding_checkpoint_in_new_process(tmp_path: Path) -> None:
    _write_bidding_dataset(tmp_path, seeds=(0, 1, 2))
    checkpoint_path = tmp_path.parent / f"{tmp_path.name}-bidding.pt"

    assert (
        train_main(
            [
                str(tmp_path),
                "--output",
                str(checkpoint_path),
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--hidden-dim",
                "8",
                "--hidden-layers",
                "1",
                "--train-ratio",
                "1",
                "--validation-ratio",
                "1",
                "--test-ratio",
                "98",
                "--json",
            ]
        )
        == 0
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "napoleon_ml.cli.evaluate_bidding_mlp",
            str(tmp_path),
            "--checkpoint",
            str(checkpoint_path),
            "--split",
            "test",
            "--batch-size",
            "1",
            "--train-ratio",
            "1",
            "--validation-ratio",
            "1",
            "--test-ratio",
            "98",
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["split"] == "test"
    assert report["model"]["illegalPredictionCount"] == 0


@pytest.mark.parametrize(
    ("metadata_key", "bad_value"),
    (
        ("checkpoint_schema_version", 999),
        ("dataset_schema_version", 999),
        ("sample_type", "playing-training-sample"),
        ("bidding_encoder_schema_version", 999),
        ("model_input_schema_version", 999),
        ("action_count", 53),
        ("card_ids_sha256", "0" * 64),
        ("seed", True),
    ),
)
def test_bidding_checkpoint_rejects_incompatible_metadata(
    tmp_path: Path,
    metadata_key: str,
    bad_value: int | str | bool,
) -> None:
    _write_bidding_dataset(tmp_path, seeds=(0, 1, 2))
    checkpoint_path = tmp_path.parent / f"{tmp_path.name}-bidding.pt"
    assert (
        train_main(
            [
                str(tmp_path),
                "--output",
                str(checkpoint_path),
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--hidden-dim",
                "8",
                "--hidden-layers",
                "1",
                "--train-ratio",
                "1",
                "--validation-ratio",
                "1",
                "--test-ratio",
                "98",
                "--json",
            ]
        )
        == 0
    )

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    raw[metadata_key] = bad_value
    torch.save(raw, checkpoint_path)

    with pytest.raises(BiddingCheckpointCompatibilityError, match=metadata_key):
        load_bidding_checkpoint(checkpoint_path, manifest=load_manifest(tmp_path))


def test_bidding_checkpoint_rejects_invalid_model_config_as_compatibility_error(
    tmp_path: Path,
) -> None:
    _write_bidding_dataset(tmp_path, seeds=(0, 1, 2))
    checkpoint_path = tmp_path.parent / f"{tmp_path.name}-bad-config-bidding.pt"
    assert (
        train_main(
            [
                str(tmp_path),
                "--output",
                str(checkpoint_path),
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--hidden-dim",
                "8",
                "--hidden-layers",
                "1",
                "--train-ratio",
                "1",
                "--validation-ratio",
                "1",
                "--test-ratio",
                "98",
                "--json",
            ]
        )
        == 0
    )

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    raw["model_config"] = {"input_dim": BIDDING_MODEL_INPUT_FEATURE_COUNT}
    torch.save(raw, checkpoint_path)

    with pytest.raises(BiddingCheckpointCompatibilityError, match="model_config"):
        load_bidding_checkpoint(checkpoint_path, manifest=load_manifest(tmp_path))


def test_evaluate_cli_reports_missing_bidding_checkpoint_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_bidding_dataset(tmp_path, seeds=(0, 1, 2))

    exit_code = evaluate_main(
        [
            str(tmp_path),
            "--checkpoint",
            str(tmp_path.parent / "missing-bidding.pt"),
            "--train-ratio",
            "1",
            "--validation-ratio",
            "1",
            "--test-ratio",
            "98",
        ]
    )

    assert exit_code == 1
    assert "checkpoint cannot be read" in capsys.readouterr().err


def test_evaluate_cli_rejects_bidding_split_ratios_that_differ_from_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_bidding_dataset(tmp_path, seeds=(0, 1, 2))
    checkpoint_path = tmp_path.parent / f"{tmp_path.name}-bidding.pt"
    assert (
        train_main(
            [
                str(tmp_path),
                "--output",
                str(checkpoint_path),
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--hidden-dim",
                "8",
                "--hidden-layers",
                "1",
                "--train-ratio",
                "1",
                "--validation-ratio",
                "1",
                "--test-ratio",
                "98",
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()

    exit_code = evaluate_main(
        [
            str(tmp_path),
            "--checkpoint",
            str(checkpoint_path),
            "--split",
            "test",
        ]
    )

    assert exit_code == 1
    assert "split ratios do not match" in capsys.readouterr().err
