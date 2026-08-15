from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor, nn

from napoleon_ml.adjutant.checkpoint import (
    AdjutantCheckpointCompatibilityError,
    load_adjutant_checkpoint,
)
from napoleon_ml.adjutant.metrics import (
    ADJUTANT_CATEGORY_LABELS,
    classify_adjutant_targets,
    evaluate_adjutant_model,
    mask_illegal_adjutant_logits,
    masked_adjutant_accuracy,
    masked_adjutant_cross_entropy,
    select_adjutant_action,
)
from napoleon_ml.adjutant.model import (
    AdjutantMlpConfig,
    AdjutantMlpModel,
    create_seeded_adjutant_model,
)
from napoleon_ml.cli.evaluate_adjutant_mlp import main as evaluate_main
from napoleon_ml.cli.export_policy_onnx import main as export_main
from napoleon_ml.cli.train_adjutant_mlp import main as train_main
from napoleon_ml.dataset.constants import CARD_COUNT, EXPECTED_CARD_IDS
from napoleon_ml.dataset.pytorch import AdjutantTorchSample, create_adjutant_dataloader
from napoleon_ml.dataset.reader import iter_tensorized_samples, load_manifest
from napoleon_ml.dataset.split import DatasetSplit, SplitConfig, split_for_seed
from napoleon_ml.dataset.tensors import (
    ADJUTANT_MODEL_INPUT_FEATURE_COUNT,
    ADJUTANT_MODEL_INPUT_LAYOUT,
    TensorizedAdjutantSample,
)
from napoleon_ml.dataset.validation import calculate_card_ids_sha256
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


def _adjutant_sample(*, seed: int, actor_target: int = 20) -> dict[str, Any]:
    return {
        "sampleType": "adjutant-training-sample",
        "schemaVersion": 1,
        "seed": seed,
        "step": 1,
        "actingPlayerId": "player-0",
        "relativePlayerIds": ["player-0", "player-1", "player-2", "player-3", "player-4"],
        "observation": {
            "schemaVersion": 1,
            "relativePlayerIds": ["player-0", "player-1", "player-2", "player-3", "player-4"],
            "trumpSuitOneHot": [1, 0, 0, 0],
            "contractTargetPointCards": 12,
            "selfHandMask": _mask(list(range(10))),
            "legalAdjutantMask": _mask(list(range(CARD_COUNT))),
            "specialCardIndices": {"oruma": 0, "yoromeki": 15, "seiJack": 3, "uraJack": 42},
            "biddingHistory": _empty_bidding_history(),
        },
        "actorTarget": actor_target,
    }


def _one_line(sample: dict[str, Any]) -> bytes:
    return (json.dumps(sample, separators=(",", ":")) + "\n").encode("utf-8")


def _write_dataset(directory: Path, *, seeds: tuple[int, ...]) -> None:
    samples = [_adjutant_sample(seed=seed) for seed in seeds]
    shard_bytes = b"".join(_one_line(sample) for sample in samples)
    (directory / "shard-00000.jsonl").write_bytes(shard_bytes)
    manifest = {
        "datasetSchemaVersion": 2,
        "generatorVersion": 2,
        "encoderSchemaVersion": 1,
        "format": "jsonl",
        "sampleType": "adjutant-training-sample",
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


def _feature(model_input: Tensor, name: str) -> Tensor:
    for feature in ADJUTANT_MODEL_INPUT_LAYOUT:
        if feature.name == name:
            return model_input[:, feature.start : feature.stop]

    raise AssertionError(f"Unknown feature: {name}")


def _model_input_for_categories(targets: list[int]) -> Tensor:
    model_input = torch.zeros(
        (len(targets), ADJUTANT_MODEL_INPUT_FEATURE_COUNT), dtype=torch.float32
    )
    _feature(model_input, "trumpSuitOneHot")[:, 0] = 1.0
    special = _feature(model_input, "specialCardIndicesOneHot").reshape(
        len(targets), 4, CARD_COUNT
    )
    special[:, 0, 0] = 1.0
    special[:, 1, 15] = 1.0
    special[:, 2, 3] = 1.0
    special[:, 3, 42] = 1.0
    return model_input


class _FixedLogitModel(nn.Module):
    def __init__(self, logits: Tensor) -> None:
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(self, model_input: Tensor) -> Tensor:
        return cast(Tensor, self.logits)[: model_input.shape[0]]


def test_adjutant_mlp_outputs_card_logits() -> None:
    model = AdjutantMlpModel(AdjutantMlpConfig(hidden_dim=8, hidden_layers=1))
    logits = model(torch.zeros((2, ADJUTANT_MODEL_INPUT_FEATURE_COUNT), dtype=torch.float32))

    assert logits.shape == (2, CARD_COUNT)


def test_seeded_adjutant_model_initialization_is_reproducible() -> None:
    config = AdjutantMlpConfig(hidden_dim=8, hidden_layers=1)
    first = create_seeded_adjutant_model(config, seed=123)
    second = create_seeded_adjutant_model(config, seed=123)

    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters(), strict=True
    ):
        assert first_parameter.equal(second_parameter)


def test_adjutant_masking_loss_accuracy_and_inference_exclude_illegal_cards() -> None:
    logits = torch.zeros((2, CARD_COUNT), dtype=torch.float32)
    target = torch.tensor([1, 4], dtype=torch.long)
    legal_mask = torch.zeros((2, CARD_COUNT), dtype=torch.bool)
    legal_mask[0, [1, 2]] = True
    legal_mask[1, [3, 4, 5]] = True

    logits[0, 0] = 100.0
    logits[0, 1] = 5.0
    logits[1, 52] = 100.0
    logits[1, 3] = 7.0

    masked = mask_illegal_adjutant_logits(logits, legal_mask)
    actions = select_adjutant_action(logits, legal_mask)
    loss = masked_adjutant_cross_entropy(logits, target, legal_mask)

    assert torch.isfinite(loss)
    assert masked[0, 0] == torch.finfo(torch.float32).min
    assert actions.tolist() == [1, 3]
    assert bool(legal_mask.gather(1, actions.unsqueeze(1)).all().item())
    assert masked_adjutant_accuracy(logits, target, legal_mask) == 0.5


def test_adjutant_loss_rejects_illegal_target() -> None:
    logits = torch.zeros((1, CARD_COUNT), dtype=torch.float32)
    target = torch.tensor([2], dtype=torch.long)
    legal_mask = torch.zeros((1, CARD_COUNT), dtype=torch.bool)
    legal_mask[0, 1] = True

    with pytest.raises(ValueError, match="target must be legal"):
        masked_adjutant_cross_entropy(logits, target, legal_mask)


@pytest.mark.parametrize(
    "operation",
    (mask_illegal_adjutant_logits, select_adjutant_action),
)
def test_adjutant_masking_rejects_rows_without_legal_cards(operation: Any) -> None:
    logits = torch.zeros((1, CARD_COUNT), dtype=torch.float32)
    legal_mask = torch.zeros((1, CARD_COUNT), dtype=torch.bool)

    with pytest.raises(ValueError, match="at least one legal card"):
        operation(logits, legal_mask)


@pytest.mark.parametrize(
    "config_factory",
    (
        lambda: AdjutantMlpConfig(input_dim=10),
        lambda: AdjutantMlpConfig(hidden_dim=0),
        lambda: AdjutantMlpConfig(hidden_layers=0),
        lambda: AdjutantMlpConfig(dropout=1.0),
    ),
)
def test_adjutant_config_rejects_invalid_values(
    config_factory: Callable[[], AdjutantMlpConfig],
) -> None:
    with pytest.raises(ValueError):
        config_factory()


def test_adjutant_evaluation_reports_required_categories_and_uniform_baseline() -> None:
    targets = torch.tensor([0, 3, 42, 1, 14, 52], dtype=torch.long)
    model_input = _model_input_for_categories(targets.tolist())
    legal_mask = torch.ones((len(targets), CARD_COUNT), dtype=torch.bool)
    logits = torch.zeros((len(targets), CARD_COUNT), dtype=torch.float32)
    logits[torch.arange(len(targets)), targets] = 10.0
    batch: AdjutantTorchSample = {
        "model_input": model_input,
        "legal_adjutant_mask": legal_mask,
        "actor_target": targets,
        "seed": torch.arange(len(targets), dtype=torch.int64),
        "step": torch.ones(len(targets), dtype=torch.int64),
    }

    categories = classify_adjutant_targets(model_input, targets)
    report = evaluate_adjutant_model(_FixedLogitModel(logits), [batch], split="test")

    assert sorted(report.by_category) == sorted(ADJUTANT_CATEGORY_LABELS)
    assert categories.tolist() == list(range(len(ADJUTANT_CATEGORY_LABELS)))
    assert report.sample_count == len(targets)
    assert report.illegal_prediction_count == 0
    assert report.top1.accuracy == 1.0
    assert report.top1.legal_uniform_accuracy == pytest.approx(1.0 / CARD_COUNT)
    for metric in report.by_category.values():
        assert metric.count == 1
        assert metric.accuracy == 1.0
        assert metric.legal_uniform_accuracy == pytest.approx(1.0 / CARD_COUNT)


def test_train_cli_saves_checkpoint_and_evaluate_cli_loads_test_split(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_dataset(tmp_path, seeds=(0, 1, 2))
    checkpoint_path = tmp_path.parent / f"{tmp_path.name}-adjutant.pt"
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
    ] == pytest.approx(1.0 / CARD_COUNT)
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
    assert evaluate_report["baselineLegalUniform"]["top1"]["accuracy"] == pytest.approx(
        1.0 / CARD_COUNT
    )


def test_evaluate_cli_loads_adjutant_checkpoint_in_new_process(tmp_path: Path) -> None:
    _write_dataset(tmp_path, seeds=(0, 1, 2))
    checkpoint_path = tmp_path.parent / f"{tmp_path.name}-adjutant.pt"

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
            "napoleon_ml.cli.evaluate_adjutant_mlp",
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
        ("adjutant_encoder_schema_version", 999),
        ("adjutant_model_input_schema_version", 999),
        ("action_count", 52),
        ("card_ids_sha256", "0" * 64),
        ("seed", "0"),
    ),
)
def test_adjutant_checkpoint_rejects_incompatible_metadata(
    tmp_path: Path,
    metadata_key: str,
    bad_value: int | str,
) -> None:
    _write_dataset(tmp_path, seeds=(0, 1, 2))
    checkpoint_path = tmp_path.parent / f"{tmp_path.name}-adjutant.pt"
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

    with pytest.raises(AdjutantCheckpointCompatibilityError, match=metadata_key):
        load_adjutant_checkpoint(checkpoint_path, manifest=load_manifest(tmp_path))


def test_adjutant_checkpoint_rejects_seed_mismatch_in_training_config(tmp_path: Path) -> None:
    _write_dataset(tmp_path, seeds=(0, 1, 2))
    checkpoint_path = tmp_path.parent / f"{tmp_path.name}-adjutant.pt"
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
    raw["training_config"] = dict(raw["training_config"], seed=999)
    torch.save(raw, checkpoint_path)

    with pytest.raises(AdjutantCheckpointCompatibilityError, match="training_config.seed"):
        load_adjutant_checkpoint(checkpoint_path, manifest=load_manifest(tmp_path))


def test_evaluate_cli_rejects_split_ratios_that_differ_from_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_dataset(tmp_path, seeds=(0, 1, 2))
    checkpoint_path = tmp_path.parent / f"{tmp_path.name}-adjutant.pt"
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


def test_adjutant_dataloader_uses_fixed_contract(tmp_path: Path) -> None:
    _write_dataset(tmp_path, seeds=(0,))
    [batch] = list(
        create_adjutant_dataloader(
            tmp_path,
            split=DatasetSplit.TRAIN,
            split_config=SplitConfig(train=100, validation=0, test=0),
            batch_size=1,
        )
    )

    assert batch["model_input"].shape == (1, ADJUTANT_MODEL_INPUT_FEATURE_COUNT)
    assert batch["legal_adjutant_mask"].shape == (1, CARD_COUNT)
    assert int(batch["actor_target"][0].item()) == 20


def test_adjutant_onnx_export_cli_writes_metadata_and_checks_parity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    onnx = pytest.importorskip("onnx")
    onnxruntime = pytest.importorskip("onnxruntime")
    _write_dataset(tmp_path, seeds=(0, 1, 2))
    checkpoint_path = tmp_path.parent / f"{tmp_path.name}-adjutant.pt"
    onnx_path = tmp_path.parent / f"{tmp_path.name}-adjutant.onnx"
    metadata_path = tmp_path.parent / f"{tmp_path.name}-adjutant.json"
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
            "adjutant",
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
    assert metadata["artifactType"] == "napoleon-adjutant-policy-onnx"
    assert metadata["policyType"] == "adjutant"
    assert metadata["modelInputFeatureCount"] == ADJUTANT_MODEL_INPUT_FEATURE_COUNT
    assert metadata["outputLogitCount"] == CARD_COUNT
    assert metadata["inputShape"] == ["batch", ADJUTANT_MODEL_INPUT_FEATURE_COUNT]
    assert metadata["outputShape"] == ["batch", CARD_COUNT]
    assert metadata["inputDtype"] == "float32"
    assert metadata["outputDtype"] == "float32"

    report = json.loads(capsys.readouterr().out)
    assert report["policyType"] == "adjutant"
    assert report["paritySample"]["pytorchSelection"] == report["paritySample"][
        "onnxSelection"
    ]
    assert report["paritySample"]["selectionParity"] is True
    assert report["paritySample"]["maxAbsLogitDiff"] <= 1e-4

    first_sample = next(iter_tensorized_samples(tmp_path, verify_integrity=True))
    assert isinstance(first_sample, TensorizedAdjutantSample)
    selected = int(report["paritySample"]["onnxSelection"])
    assert bool(first_sample.legal_adjutant_mask[selected])

    session = onnxruntime.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    assert session.get_inputs()[0].name == ONNX_INPUT_NAME
    assert session.get_outputs()[0].name == ONNX_OUTPUT_NAME
    assert session.get_inputs()[0].shape[1] == ADJUTANT_MODEL_INPUT_FEATURE_COUNT
    assert session.get_outputs()[0].shape[1] == CARD_COUNT
    assert session.get_inputs()[0].type == "tensor(float)"
    assert session.get_outputs()[0].type == "tensor(float)"


def test_adjutant_onnx_export_cli_rejects_overlapping_output_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_dataset(tmp_path, seeds=(0, 1, 2))
    checkpoint_path = tmp_path.parent / f"{tmp_path.name}-adjutant.pt"
    output_path = tmp_path.parent / f"{tmp_path.name}-adjutant.json"
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
            "adjutant",
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 1
    assert "must be different paths" in capsys.readouterr().err
    assert not output_path.exists()
