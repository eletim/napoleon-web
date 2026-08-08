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

from napoleon_ml.cli.evaluate_exchange_mlp import main as evaluate_main
from napoleon_ml.cli.export_policy_onnx import main as export_main
from napoleon_ml.cli.train_exchange_mlp import main as train_main
from napoleon_ml.dataset.constants import CARD_COUNT, EXPECTED_CARD_IDS
from napoleon_ml.dataset.pytorch import create_exchange_dataloader
from napoleon_ml.dataset.reader import iter_tensorized_samples, load_manifest
from napoleon_ml.dataset.split import DatasetSplit, SplitConfig, split_for_seed
from napoleon_ml.dataset.tensors import (
    EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
    TensorizedExchangeSample,
)
from napoleon_ml.dataset.validation import calculate_card_ids_sha256
from napoleon_ml.exchange.checkpoint import (
    ExchangeCheckpointCompatibilityError,
    load_exchange_checkpoint,
)
from napoleon_ml.exchange.metrics import (
    DISCARD_COUNT,
    evaluate_exchange_model,
    masked_exchange_bce_loss,
    select_exchange_discards,
    selected_indices_to_mask,
)
from napoleon_ml.exchange.model import (
    ExchangeMlpConfig,
    ExchangeMlpModel,
    create_seeded_exchange_model,
)
from napoleon_ml.nonplaying_onnx_export import (
    ONNX_INPUT_NAME,
    ONNX_OUTPUT_NAME,
    validate_nonplaying_onnx_metadata,
)


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


def _exchange_sample(*, seed: int, step: int, target_indices: list[int]) -> dict[str, Any]:
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
    return {
        "sampleType": "exchange-training-sample",
        "schemaVersion": 1,
        "seed": seed,
        "step": step,
        "actingPlayerId": "player-0",
        "relativePlayerIds": ["player-0", "player-1", "player-2", "player-3", "player-4"],
        "observation": observation,
        "actorTarget": {"discardTargetMask": _mask(target_indices)},
    }


def _one_line(sample: dict[str, Any]) -> bytes:
    return (json.dumps(sample, separators=(",", ":")) + "\n").encode("utf-8")


def _write_dataset(directory: Path, *, seeds: tuple[int, ...]) -> None:
    samples = [
        _exchange_sample(
            seed=seed,
            step=1,
            target_indices=[0, 1, 2] if seed == 0 else [2, 3, 4],
        )
        for seed in seeds
    ]
    shard_bytes = b"".join(_one_line(sample) for sample in samples)
    (directory / "shard-00000.jsonl").write_bytes(shard_bytes)

    manifest = {
        "datasetSchemaVersion": 2,
        "generatorVersion": 2,
        "encoderSchemaVersion": 1,
        "format": "jsonl",
        "sampleType": "exchange-training-sample",
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


def test_exchange_mlp_outputs_card_logits() -> None:
    model = ExchangeMlpModel(ExchangeMlpConfig(hidden_dim=8, hidden_layers=1))
    logits = model(torch.zeros((2, EXCHANGE_MODEL_INPUT_FEATURE_COUNT), dtype=torch.float32))

    assert logits.shape == (2, CARD_COUNT)


def test_seeded_exchange_model_initialization_is_reproducible() -> None:
    config = ExchangeMlpConfig(hidden_dim=8, hidden_layers=1)
    first = create_seeded_exchange_model(config, seed=123)
    second = create_seeded_exchange_model(config, seed=123)

    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters(), strict=True
    ):
        assert first_parameter.equal(second_parameter)


def test_exchange_loss_only_trains_legal_candidates() -> None:
    logits = torch.zeros((1, CARD_COUNT), dtype=torch.float32)
    target = torch.zeros((1, CARD_COUNT), dtype=torch.bool)
    legal = torch.zeros((1, CARD_COUNT), dtype=torch.bool)
    target[0, [0, 1, 2]] = True
    legal[0, list(range(13))] = True

    baseline_loss = masked_exchange_bce_loss(logits, target, legal)
    logits[0, 52] = 1000.0
    ignored_illegal_loss = masked_exchange_bce_loss(logits, target, legal)

    assert ignored_illegal_loss == baseline_loss


def test_exchange_inference_excludes_illegal_cards_and_breaks_ties_by_index() -> None:
    logits = torch.zeros((2, CARD_COUNT), dtype=torch.float32)
    legal = torch.zeros((2, CARD_COUNT), dtype=torch.bool)
    legal[0, [5, 6, 7, 8]] = True
    legal[1, [0, 1, 2, 3]] = True
    logits[0, 0] = 100.0
    logits[0, 8] = 2.0
    logits[0, 7] = 2.0
    logits[0, 6] = 2.0
    logits[0, 5] = 2.0
    logits[1, 52] = 100.0

    selected = select_exchange_discards(logits, legal)

    assert selected.tolist() == [[5, 6, 7], [0, 1, 2]]
    assert bool(legal.gather(1, selected).all().item())


def test_exchange_evaluation_reports_set_metrics_and_random_baseline(tmp_path: Path) -> None:
    _write_dataset(tmp_path, seeds=(0, 1))
    loader = create_exchange_dataloader(
        tmp_path,
        split=DatasetSplit.TRAIN,
        split_config=SplitConfig(train=100, validation=0, test=0),
        batch_size=2,
    )
    model = ExchangeMlpModel(ExchangeMlpConfig(hidden_dim=8, hidden_layers=1))
    for parameter in model.parameters():
        parameter.data.zero_()

    report = evaluate_exchange_model(model, loader, split=DatasetSplit.TRAIN.value)

    assert report.sample_count == 2
    assert report.illegal_selected_card_count == 0
    assert report.exact_set_accuracy == 0.5
    assert report.average_overlap_count == 2.0
    assert report.selected_teacher_match_rate == pytest.approx(2.0 / DISCARD_COUNT)
    assert report.baseline_random_legal_exact_set_accuracy == pytest.approx(1 / 286)
    assert report.baseline_random_legal_average_overlap_count == pytest.approx(9 / 13)
    assert report.baseline_random_legal_selected_teacher_match_rate == pytest.approx(3 / 13)


def test_selected_indices_to_mask_rejects_non_three_card_indices() -> None:
    with pytest.raises(ValueError, match="shape"):
        selected_indices_to_mask(torch.tensor([[0, 1]], dtype=torch.long))


@pytest.mark.parametrize(
    "config_factory",
    (
        lambda: ExchangeMlpConfig(input_dim=0),
        lambda: ExchangeMlpConfig(hidden_dim=0),
        lambda: ExchangeMlpConfig(hidden_layers=0),
        lambda: ExchangeMlpConfig(dropout=1.0),
    ),
)
def test_exchange_config_rejects_invalid_values(
    config_factory: Callable[[], ExchangeMlpConfig],
) -> None:
    with pytest.raises(ValueError):
        config_factory()


def test_train_cli_saves_checkpoint_and_evaluate_cli_loads_test_split(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_dataset(tmp_path, seeds=(0, 1, 2))
    checkpoint_path = tmp_path.parent / f"{tmp_path.name}-exchange.pt"
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
    assert train_report["finalValidation"]["model"]["exactSetAccuracy"] is not None
    assert train_report["finalValidation"]["model"]["illegalSelectedCardCount"] == 0
    assert train_report["finalValidation"]["baselineRandomLegal"][
        "averageOverlapCount"
    ] == pytest.approx(9 / 13)

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
    assert evaluate_report["model"]["illegalSelectedCardCount"] == 0


def test_evaluate_cli_loads_checkpoint_in_new_process(tmp_path: Path) -> None:
    _write_dataset(tmp_path, seeds=(0, 1, 2))
    checkpoint_path = tmp_path.parent / f"{tmp_path.name}-exchange.pt"

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
            "napoleon_ml.cli.evaluate_exchange_mlp",
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
    assert report["model"]["illegalSelectedCardCount"] == 0


@pytest.mark.parametrize(
    ("metadata_key", "bad_value"),
    (
        ("checkpoint_schema_version", 999),
        ("dataset_schema_version", 999),
        ("sample_type", "playing-training-sample"),
        ("encoder_schema_version", 999),
        ("exchange_model_input_schema_version", 999),
        ("action_count", 52),
        ("card_ids_sha256", "0" * 64),
        ("discard_count", 2),
        ("seed", "0"),
    ),
)
def test_exchange_checkpoint_rejects_incompatible_metadata(
    tmp_path: Path,
    metadata_key: str,
    bad_value: int | str,
) -> None:
    _write_dataset(tmp_path, seeds=(0, 1, 2))
    checkpoint_path = tmp_path.parent / f"{tmp_path.name}-exchange.pt"
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

    with pytest.raises(ExchangeCheckpointCompatibilityError, match=metadata_key):
        load_exchange_checkpoint(checkpoint_path, manifest=load_manifest(tmp_path))


def test_exchange_onnx_export_cli_writes_metadata_and_checks_top3_set_parity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    onnx = pytest.importorskip("onnx")
    onnxruntime = pytest.importorskip("onnxruntime")
    _write_dataset(tmp_path, seeds=(0, 1, 2))
    checkpoint_path = tmp_path.parent / f"{tmp_path.name}-exchange.pt"
    onnx_path = tmp_path.parent / f"{tmp_path.name}-exchange.onnx"
    metadata_path = tmp_path.parent / f"{tmp_path.name}-exchange.json"
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

    exit_code = export_main(
        [
            str(tmp_path),
            "--policy-type",
            "exchange",
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(onnx_path),
            "--metadata-output",
            str(metadata_path),
            "--json",
        ]
    )

    assert exit_code == 0
    assert onnx_path.is_file()
    onnx_model = onnx.load_model(onnx_path, load_external_data=False)
    assert all(len(initializer.external_data) == 0 for initializer in onnx_model.graph.initializer)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_nonplaying_onnx_metadata(metadata)
    assert metadata["artifactType"] == "napoleon-exchange-policy-onnx"
    assert metadata["policyType"] == "exchange"
    assert metadata["modelInputFeatureCount"] == EXCHANGE_MODEL_INPUT_FEATURE_COUNT
    assert metadata["outputLogitCount"] == CARD_COUNT
    assert metadata["discardCount"] == DISCARD_COUNT
    assert metadata["inputShape"] == ["batch", EXCHANGE_MODEL_INPUT_FEATURE_COUNT]
    assert metadata["outputShape"] == ["batch", CARD_COUNT]
    assert metadata["inputDtype"] == "float32"
    assert metadata["outputDtype"] == "float32"

    report = json.loads(capsys.readouterr().out)
    pytorch_top3 = [int(index) for index in report["paritySample"]["pytorchSelection"]]
    onnx_top3 = [int(index) for index in report["paritySample"]["onnxSelection"]]
    assert set(pytorch_top3) == set(onnx_top3)
    assert report["paritySample"]["selectionParity"] is True
    assert report["paritySample"]["maxAbsLogitDiff"] <= 1e-4
    assert len(onnx_top3) == DISCARD_COUNT
    assert len(set(onnx_top3)) == DISCARD_COUNT

    first_sample = next(iter_tensorized_samples(tmp_path, verify_integrity=True))
    assert isinstance(first_sample, TensorizedExchangeSample)
    assert all(bool(first_sample.legal_discard_card_mask[index]) for index in onnx_top3)

    session = onnxruntime.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    assert session.get_inputs()[0].name == ONNX_INPUT_NAME
    assert session.get_outputs()[0].name == ONNX_OUTPUT_NAME
    assert session.get_inputs()[0].shape[1] == EXCHANGE_MODEL_INPUT_FEATURE_COUNT
    assert session.get_outputs()[0].shape[1] == CARD_COUNT
    assert session.get_inputs()[0].type == "tensor(float)"
    assert session.get_outputs()[0].type == "tensor(float)"
