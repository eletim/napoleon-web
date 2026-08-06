from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from napoleon_ml.cli.evaluate_ownership_mlp import main as evaluate_main
from napoleon_ml.cli.train_ownership_mlp import main as train_main
from napoleon_ml.dataset.constants import CARD_COUNT, EXPECTED_CARD_IDS
from napoleon_ml.dataset.reader import load_manifest
from napoleon_ml.dataset.tensors import MODEL_INPUT_FEATURE_COUNT
from napoleon_ml.dataset.validation import calculate_card_ids_sha256
from napoleon_ml.ownership.checkpoint import (
    CheckpointCompatibilityError,
    load_ownership_checkpoint,
)
from napoleon_ml.ownership.metrics import (
    masked_ownership_accuracy,
    masked_ownership_cross_entropy,
)
from napoleon_ml.ownership.model import (
    OwnershipMlpConfig,
    OwnershipMlpModel,
    create_seeded_ownership_model,
)

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "valid_sample.json"


def _load_valid_sample() -> dict[str, Any]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _one_line(sample: dict[str, Any]) -> bytes:
    return (json.dumps(sample) + "\n").encode("utf-8")


def _write_dataset(directory: Path, *, seeds: tuple[int, ...]) -> None:
    base = _load_valid_sample()
    samples = [dict(base, seed=seed, step=1) for seed in seeds]
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


def test_mlp_outputs_card_owner_logits() -> None:
    model = OwnershipMlpModel(OwnershipMlpConfig(hidden_dim=8, hidden_layers=1))
    logits = model(torch.zeros((2, MODEL_INPUT_FEATURE_COUNT), dtype=torch.float32))

    assert logits.shape == (2, CARD_COUNT, 6)


def test_seeded_model_initialization_is_reproducible() -> None:
    config = OwnershipMlpConfig(hidden_dim=8, hidden_layers=1)
    first = create_seeded_ownership_model(config, seed=123)
    second = create_seeded_ownership_model(config, seed=123)

    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters(), strict=True
    ):
        assert first_parameter.equal(second_parameter)


def test_masked_loss_and_accuracy_ignore_unmasked_cards() -> None:
    logits = torch.zeros((1, CARD_COUNT, 6), dtype=torch.float32)
    target = torch.zeros((1, CARD_COUNT), dtype=torch.long)
    mask = torch.zeros((1, CARD_COUNT), dtype=torch.bool)

    target[0, 0] = 1
    target[0, 1] = 2
    target[0, 2] = 3
    mask[0, 0] = True
    mask[0, 1] = True

    logits[0, 0, 1] = 5.0
    logits[0, 1, 3] = 5.0
    logits[0, 2, 0] = 5.0

    loss = masked_ownership_cross_entropy(logits, target, mask)

    assert loss.item() > 0.0
    assert masked_ownership_accuracy(logits, target, mask) == 0.5


def test_train_cli_saves_checkpoint_and_evaluate_cli_loads_test_split(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_dataset(tmp_path, seeds=(0, 1, 2))
    checkpoint_path = tmp_path.parent / f"{tmp_path.name}-ownership.pt"

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
    assert train_report["finalValidation"]["model"]["maskedLoss"] is not None
    assert train_report["finalValidation"]["baselineAlwaysNotInHand"]["maskedAccuracy"] is not None

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
    assert evaluate_report["model"]["maskedLoss"] is not None
    assert evaluate_report["baselineAlwaysNotInHand"]["maskedAccuracy"] is not None


def test_checkpoint_rejects_incompatible_model_input_schema(tmp_path: Path) -> None:
    _write_dataset(tmp_path, seeds=(0, 1, 2))
    checkpoint_path = tmp_path.parent / f"{tmp_path.name}-ownership.pt"
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

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw["model_input_schema_version"] = 999
    torch.save(raw, checkpoint_path)

    with pytest.raises(CheckpointCompatibilityError, match="model_input_schema_version"):
        load_ownership_checkpoint(checkpoint_path, manifest=load_manifest(tmp_path))
