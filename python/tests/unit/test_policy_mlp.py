from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from napoleon_ml.cli.evaluate_policy_mlp import main as evaluate_main
from napoleon_ml.cli.train_policy_mlp import main as train_main
from napoleon_ml.dataset.constants import CARD_COUNT, EXPECTED_CARD_IDS
from napoleon_ml.dataset.pytorch import create_playing_dataloader
from napoleon_ml.dataset.reader import load_manifest
from napoleon_ml.dataset.split import DatasetSplit, SplitConfig, split_for_seed
from napoleon_ml.dataset.tensors import MODEL_INPUT_FEATURE_COUNT
from napoleon_ml.dataset.validation import calculate_card_ids_sha256
from napoleon_ml.policy.checkpoint import (
    PolicyCheckpointCompatibilityError,
    load_policy_checkpoint,
)
from napoleon_ml.policy.metrics import (
    evaluate_policy_model,
    mask_illegal_policy_logits,
    masked_policy_accuracy,
    masked_policy_cross_entropy,
    select_policy_action,
)
from napoleon_ml.policy.model import (
    PolicyMlpConfig,
    PolicyMlpModel,
    create_seeded_policy_model,
)

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "valid_sample.json"


def _load_valid_sample() -> dict[str, Any]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _one_line(sample: dict[str, Any]) -> bytes:
    return (json.dumps(sample) + "\n").encode("utf-8")


def _force_selected_card_only(sample: dict[str, Any]) -> dict[str, Any]:
    forced = json.loads(json.dumps(sample))
    selected = int(forced["actorTarget"]["selectedCardIndex"])
    forced["observation"]["legalPlayMask"] = [
        1 if card_index == selected else 0 for card_index in range(CARD_COUNT)
    ]
    return forced  # type: ignore[no-any-return]


def _write_dataset(
    directory: Path, *, seeds: tuple[int, ...], include_forced: bool = False
) -> None:
    base = _load_valid_sample()
    samples: list[dict[str, Any]] = []
    for seed in seeds:
        sample = dict(base, seed=seed, step=1)
        samples.append(
            _force_selected_card_only(sample) if include_forced and seed == 0 else sample
        )

    shard_bytes = b"".join(_one_line(sample) for sample in samples)
    (directory / "shard-00000.jsonl").write_bytes(shard_bytes)

    manifest = {
        "datasetSchemaVersion": 1,
        "generatorVersion": 1,
        "playingEncoderSchemaVersion": 1,
        "format": "jsonl",
        "sampleType": "playing-training-sample",
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


def test_policy_mlp_outputs_card_logits() -> None:
    model = PolicyMlpModel(PolicyMlpConfig(hidden_dim=8, hidden_layers=1))
    logits = model(torch.zeros((2, MODEL_INPUT_FEATURE_COUNT), dtype=torch.float32))

    assert logits.shape == (2, CARD_COUNT)


def test_seeded_policy_model_initialization_is_reproducible() -> None:
    config = PolicyMlpConfig(hidden_dim=8, hidden_layers=1)
    first = create_seeded_policy_model(config, seed=123)
    second = create_seeded_policy_model(config, seed=123)

    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters(), strict=True
    ):
        assert first_parameter.equal(second_parameter)


def test_policy_masking_loss_accuracy_and_inference_exclude_illegal_cards() -> None:
    logits = torch.zeros((2, CARD_COUNT), dtype=torch.float32)
    target = torch.tensor([1, 4], dtype=torch.long)
    legal_mask = torch.zeros((2, CARD_COUNT), dtype=torch.bool)
    legal_mask[0, [1, 2]] = True
    legal_mask[1, [3, 4, 5]] = True

    logits[0, 0] = 100.0
    logits[0, 1] = 5.0
    logits[1, 52] = 100.0
    logits[1, 3] = 7.0

    masked = mask_illegal_policy_logits(logits, legal_mask)
    actions = select_policy_action(logits, legal_mask)
    loss = masked_policy_cross_entropy(logits, target, legal_mask)

    assert torch.isfinite(loss)
    assert masked[0, 0] == torch.finfo(torch.float32).min
    assert actions.tolist() == [1, 3]
    assert bool(legal_mask.gather(1, actions.unsqueeze(1)).all().item())
    assert masked_policy_accuracy(logits, target, legal_mask) == 0.5


def test_policy_loss_rejects_illegal_target() -> None:
    logits = torch.zeros((1, CARD_COUNT), dtype=torch.float32)
    target = torch.tensor([2], dtype=torch.long)
    legal_mask = torch.zeros((1, CARD_COUNT), dtype=torch.bool)
    legal_mask[0, 1] = True

    with pytest.raises(ValueError, match="target must be legal"):
        masked_policy_cross_entropy(logits, target, legal_mask)


def test_policy_evaluation_reports_forced_non_forced_and_uniform_baseline(tmp_path: Path) -> None:
    _write_dataset(tmp_path, seeds=(0, 1), include_forced=True)
    loader = create_playing_dataloader(
        tmp_path,
        split=DatasetSplit.TRAIN,
        split_config=SplitConfig(train=100, validation=0, test=0),
        batch_size=2,
    )
    model = PolicyMlpModel(PolicyMlpConfig(hidden_dim=8, hidden_layers=1))

    report = evaluate_policy_model(model, loader, split=DatasetSplit.TRAIN.value)

    assert report.sample_count == 2
    assert report.illegal_prediction_count == 0
    assert report.all_positions.count == 2
    assert report.forced_positions.count == 1
    assert report.forced_positions.legal_uniform_accuracy == 1.0
    assert report.non_forced_positions.count == 1
    assert report.non_forced_positions.legal_uniform_accuracy == pytest.approx(0.1)


def test_train_cli_saves_checkpoint_and_evaluate_cli_loads_test_split(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_dataset(tmp_path, seeds=(0, 1, 2))
    checkpoint_path = tmp_path.parent / f"{tmp_path.name}-policy.pt"
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
    assert train_report["finalValidation"]["model"]["allPositions"]["count"] == 1
    assert train_report["finalValidation"]["baselineLegalUniform"]["allPositions"][
        "accuracy"
    ] == pytest.approx(0.1)
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
    assert evaluate_report["model"]["allPositions"]["count"] == 1
    assert evaluate_report["model"]["illegalPredictionCount"] == 0
    assert evaluate_report["baselineLegalUniform"]["nonForcedPositions"][
        "accuracy"
    ] == pytest.approx(0.1)


def test_evaluate_cli_loads_checkpoint_in_new_process(tmp_path: Path) -> None:
    _write_dataset(tmp_path, seeds=(0, 1, 2))
    checkpoint_path = tmp_path.parent / f"{tmp_path.name}-policy.pt"

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
            "napoleon_ml.cli.evaluate_policy_mlp",
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
        ("playing_encoder_schema_version", 999),
        ("model_input_schema_version", 999),
        ("card_ids_sha256", "0" * 64),
    ),
)
def test_policy_checkpoint_rejects_incompatible_metadata(
    tmp_path: Path,
    metadata_key: str,
    bad_value: int | str,
) -> None:
    _write_dataset(tmp_path, seeds=(0, 1, 2))
    checkpoint_path = tmp_path.parent / f"{tmp_path.name}-policy.pt"
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

    with pytest.raises(PolicyCheckpointCompatibilityError, match=metadata_key):
        load_policy_checkpoint(checkpoint_path, manifest=load_manifest(tmp_path))


def test_evaluate_cli_reports_missing_checkpoint_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_dataset(tmp_path, seeds=(0, 1, 2))

    exit_code = evaluate_main(
        [
            str(tmp_path),
            "--checkpoint",
            str(tmp_path.parent / "missing-policy.pt"),
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


def test_evaluate_cli_rejects_split_ratios_that_differ_from_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_dataset(tmp_path, seeds=(0, 1, 2))
    checkpoint_path = tmp_path.parent / f"{tmp_path.name}-policy.pt"
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
