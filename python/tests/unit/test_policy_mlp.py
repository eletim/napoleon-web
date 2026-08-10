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

import napoleon_ml.policy.onnx_export as onnx_export_module
from napoleon_ml.cli.evaluate_policy_mlp import main as evaluate_main
from napoleon_ml.cli.export_policy_onnx import main as export_main
from napoleon_ml.cli.migrate_policy_architecture import main as migrate_architecture_main
from napoleon_ml.cli.train_policy_mlp import main as train_main
from napoleon_ml.dataset.constants import CARD_COUNT, EXPECTED_CARD_IDS
from napoleon_ml.dataset.pytorch import create_playing_dataloader
from napoleon_ml.dataset.reader import iter_tensorized_samples, load_manifest
from napoleon_ml.dataset.split import DatasetSplit, SplitConfig, split_for_seed
from napoleon_ml.dataset.tensors import MODEL_INPUT_FEATURE_COUNT
from napoleon_ml.dataset.validation import calculate_card_ids_sha256
from napoleon_ml.policy.checkpoint import (
    PolicyCheckpointCompatibilityError,
    load_policy_checkpoint,
    migrate_policy_checkpoint_to_hidden_dims,
    migrate_policy_checkpoint_v1_to_v2,
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
from napoleon_ml.policy.onnx_export import (
    ONNX_INPUT_NAME,
    ONNX_OPSET_VERSION,
    ONNX_OUTPUT_NAME,
    build_policy_onnx_metadata,
    export_policy_checkpoint_to_onnx,
    validate_policy_onnx_metadata,
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
        "playingEncoderSchemaVersion": 2,
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


def test_policy_config_loads_legacy_and_roundtrips_hidden_dims() -> None:
    legacy = PolicyMlpConfig.from_dict(
        {
            "input_dim": MODEL_INPUT_FEATURE_COUNT,
            "hidden_dim": 8,
            "hidden_layers": 2,
            "dropout": 0.0,
        }
    )
    assert legacy.hidden_widths == (8, 8)
    assert legacy.to_dict()["hidden_dims"] == [8, 8]

    large = PolicyMlpConfig(hidden_dims=(512, 512, 256, 256))
    assert large.hidden_dim == 512
    assert large.hidden_layers == 4
    assert PolicyMlpConfig.from_dict(large.to_dict()).hidden_widths == (
        512,
        512,
        256,
        256,
    )
    assert PolicyMlpConfig.from_dict(
        {
            "input_dim": MODEL_INPUT_FEATURE_COUNT,
            "hidden_dims": [512, 512, 256, 256],
            "dropout": 0.0,
        }
    ).hidden_widths == (
        512,
        512,
        256,
        256,
    )
    with pytest.raises(ValueError, match="non-empty"):
        PolicyMlpConfig.from_dict(
            {"input_dim": MODEL_INPUT_FEATURE_COUNT, "hidden_dims": [], "dropout": 0.0}
        )
    with pytest.raises(ValueError, match="hidden_layers"):
        PolicyMlpConfig.from_dict({"hidden_dim": 512, "hidden_layers": 2, "hidden_dims": [512]})


def test_large_policy_mlp_outputs_card_logits() -> None:
    model = PolicyMlpModel(PolicyMlpConfig(hidden_dims=(512, 512, 256, 256)))
    logits = model(torch.zeros((2, MODEL_INPUT_FEATURE_COUNT), dtype=torch.float32))

    assert logits.shape == (2, CARD_COUNT)


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


@pytest.mark.parametrize(
    "operation",
    (mask_illegal_policy_logits, select_policy_action),
)
def test_policy_masking_rejects_rows_without_legal_cards(operation: Any) -> None:
    logits = torch.zeros((1, CARD_COUNT), dtype=torch.float32)
    legal_mask = torch.zeros((1, CARD_COUNT), dtype=torch.bool)

    with pytest.raises(ValueError, match="at least one legal card"):
        operation(logits, legal_mask)


@pytest.mark.parametrize(
    "config_factory",
    (
        lambda: PolicyMlpConfig(input_dim=0),
        lambda: PolicyMlpConfig(hidden_dim=0),
        lambda: PolicyMlpConfig(hidden_layers=0),
        lambda: PolicyMlpConfig(hidden_dims=()),
        lambda: PolicyMlpConfig(hidden_dims=(8, 0)),
        lambda: PolicyMlpConfig(hidden_dims=(8, -1)),
        lambda: PolicyMlpConfig(dropout=1.0),
    ),
)
def test_policy_config_rejects_invalid_values(
    config_factory: Callable[[], PolicyMlpConfig],
) -> None:
    with pytest.raises(ValueError):
        config_factory()


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


def test_train_cli_hidden_dims_records_resolved_training_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
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
                "--hidden-dims",
                "16,12",
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

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert raw["model_config"]["hidden_dims"] == [16, 12]
    assert raw["training_config"]["hidden_dim"] == 16
    assert raw["training_config"]["hidden_layers"] == 2
    assert raw["training_config"]["hidden_dims"] == [16, 12]


@pytest.mark.parametrize(
    ("invalid_args", "expected_error"),
    (
        (["--epochs", "0"], "epochs must be positive"),
        (["--learning-rate", "0"], "learning-rate must be positive"),
        (["--hidden-dims", "16,,12"], "must not contain empty segments"),
    ),
)
def test_train_cli_reports_invalid_training_settings_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    invalid_args: list[str],
    expected_error: str,
) -> None:
    _write_dataset(tmp_path, seeds=(0, 1, 2))
    checkpoint_path = tmp_path.parent / f"{tmp_path.name}-invalid-policy.pt"

    exit_code = train_main(
        [
            str(tmp_path),
            "--output",
            str(checkpoint_path),
            *invalid_args,
        ]
    )

    assert exit_code == 1
    assert expected_error in capsys.readouterr().err
    assert not checkpoint_path.exists()


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


def test_migrate_policy_checkpoint_v1_to_v2_preserves_logits_and_zeroes_new_role_weights(
    tmp_path: Path,
) -> None:
    _write_dataset(tmp_path, seeds=(0,))
    source_path = tmp_path.parent / f"{tmp_path.name}-old-policy.pt"
    migrated_path = tmp_path.parent / f"{tmp_path.name}-migrated-policy.pt"
    old_model = create_seeded_policy_model(
        PolicyMlpConfig(input_dim=6242, hidden_dim=8, hidden_layers=1),
        seed=123,
    )
    torch.save(
        {
            "checkpoint_schema_version": 1,
            "model_state": old_model.state_dict(),
            "model_config": old_model.config.to_dict(),
            "training_config": {},
            "dataset_schema_version": 1,
            "playing_encoder_schema_version": 1,
            "model_input_schema_version": 1,
            "card_ids_sha256": calculate_card_ids_sha256(),
        },
        source_path,
    )
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()

    migrated = migrate_policy_checkpoint_v1_to_v2(source_path, migrated_path)
    migrated_model, checkpoint = load_policy_checkpoint(
        migrated_path, manifest=load_manifest(tmp_path)
    )

    first_weight = cast(dict[str, torch.Tensor], migrated["model_state"])["network.0.weight"]
    assert torch.equal(first_weight[:, :6242], old_model.state_dict()["network.0.weight"])
    assert torch.count_nonzero(first_weight[:, 6242:]).item() == 0
    provenance = cast(dict[str, object], checkpoint["migration_provenance"])
    assert provenance["sourceCheckpointSha256"] == source_sha256
    assert provenance["sourceModelInputSchemaVersion"] == 1
    assert provenance["targetModelInputSchemaVersion"] == 2

    old_model.eval()
    migrated_model.eval()
    old_input = torch.randn((3, 6242), generator=torch.Generator().manual_seed(456))
    role_features = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    with torch.no_grad():
        old_logits = old_model(old_input)
        migrated_logits = migrated_model(torch.cat([old_input, role_features], dim=1))

    torch.testing.assert_close(migrated_logits, old_logits, rtol=0, atol=1e-7)


def test_migrate_policy_checkpoint_to_hidden_dims_preserves_logits_and_reloads(
    tmp_path: Path,
) -> None:
    _write_dataset(tmp_path, seeds=(0,))
    source_path = tmp_path.parent / f"{tmp_path.name}-small-policy.pt"
    migrated_path = tmp_path.parent / f"{tmp_path.name}-large-policy.pt"
    source_model = create_seeded_policy_model(
        PolicyMlpConfig(hidden_dim=8, hidden_layers=2),
        seed=123,
    )
    _write_current_policy_checkpoint(source_path, source_model)
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()

    migrated = migrate_policy_checkpoint_to_hidden_dims(
        source_path,
        migrated_path,
        target_hidden_dims=(16, 16, 12, 12),
        seed=456,
    )
    migrated_model, checkpoint = load_policy_checkpoint(
        migrated_path,
        manifest=load_manifest(tmp_path),
    )
    provenance = cast(dict[str, object], migrated["architecture_migration_provenance"])
    assert provenance["sourceCheckpointSha256"] == source_sha256
    assert provenance["sourceModelConfig"] == source_model.config.to_dict()
    assert cast(dict[str, object], provenance["targetModelConfig"])["hidden_dims"] == [
        16,
        16,
        12,
        12,
    ]
    assert checkpoint["architecture_migration_provenance"] == provenance
    assert migrated_model.config.hidden_widths == (16, 16, 12, 12)
    training_config = cast(dict[str, object], checkpoint["training_config"])
    assert training_config["hidden_dim"] == 16
    assert training_config["hidden_layers"] == 4
    assert training_config["hidden_dims"] == [16, 16, 12, 12]

    model_input = torch.randn(
        (5, MODEL_INPUT_FEATURE_COUNT),
        generator=torch.Generator().manual_seed(789),
    )
    source_model.eval()
    migrated_model.eval()
    with torch.no_grad():
        source_logits = source_model(model_input)
        migrated_logits = migrated_model(model_input)
    torch.testing.assert_close(migrated_logits, source_logits, rtol=0, atol=1e-6)


def test_migrate_policy_checkpoint_to_hidden_dims_rejects_existing_output(
    tmp_path: Path,
) -> None:
    _write_dataset(tmp_path, seeds=(0,))
    source_path = tmp_path.parent / f"{tmp_path.name}-small-policy.pt"
    migrated_path = tmp_path.parent / f"{tmp_path.name}-large-policy.pt"
    source_model = create_seeded_policy_model(
        PolicyMlpConfig(hidden_dim=8, hidden_layers=2),
        seed=123,
    )
    _write_current_policy_checkpoint(source_path, source_model)
    migrated_path.write_bytes(b"existing")

    with pytest.raises(PolicyCheckpointCompatibilityError, match="already exists"):
        migrate_policy_checkpoint_to_hidden_dims(
            source_path,
            migrated_path,
            target_hidden_dims=(16, 16, 12, 12),
        )

    assert migrated_path.read_bytes() == b"existing"


def test_migrated_policy_extra_capacity_receives_optimizer_update(tmp_path: Path) -> None:
    _write_dataset(tmp_path, seeds=(0,))
    source_path = tmp_path.parent / f"{tmp_path.name}-small-policy.pt"
    migrated_path = tmp_path.parent / f"{tmp_path.name}-large-policy.pt"
    source_model = create_seeded_policy_model(
        PolicyMlpConfig(hidden_dim=8, hidden_layers=2),
        seed=123,
    )
    _write_current_policy_checkpoint(source_path, source_model)
    migrate_policy_checkpoint_to_hidden_dims(
        source_path,
        migrated_path,
        target_hidden_dims=(16, 16, 12, 12),
        seed=456,
    )
    model, _ = load_policy_checkpoint(migrated_path, manifest=load_manifest(tmp_path))
    final_weight = cast(torch.nn.Linear, list(model.network.children())[-1]).weight
    before_extra = final_weight[:, 8:].detach().clone()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model.train()
    model_input = torch.randn(
        (32, MODEL_INPUT_FEATURE_COUNT),
        generator=torch.Generator().manual_seed(789),
    )
    loss = model(model_input).pow(2).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert final_weight.grad is not None
    assert torch.count_nonzero(final_weight.grad[:, 8:]).item() > 0
    optimizer.step()

    assert torch.count_nonzero(final_weight[:, 8:].detach() - before_extra).item() > 0


def test_migrated_policy_checkpoint_exports_to_onnx_with_source_logit_parity(
    tmp_path: Path,
) -> None:
    pytest.importorskip("onnxruntime")
    _write_dataset(tmp_path, seeds=(0, 1))
    source_path = tmp_path.parent / f"{tmp_path.name}-small-policy.pt"
    migrated_path = tmp_path.parent / f"{tmp_path.name}-large-policy.pt"
    onnx_path = tmp_path.parent / f"{tmp_path.name}-large-policy.onnx"
    metadata_path = tmp_path.parent / f"{tmp_path.name}-large-policy.json"
    source_model = create_seeded_policy_model(
        PolicyMlpConfig(hidden_dim=8, hidden_layers=2),
        seed=123,
    )
    _write_current_policy_checkpoint(source_path, source_model)
    assert (
        migrate_architecture_main(
            [
                "--input-checkpoint",
                str(source_path),
                "--output",
                str(migrated_path),
                "--target-hidden-dims",
                "16,16,12,12",
            ]
        )
        == 0
    )
    migrated_model, _ = load_policy_checkpoint(migrated_path, manifest=load_manifest(tmp_path))
    report = export_policy_checkpoint_to_onnx(
        dataset_directory=tmp_path,
        checkpoint_path=migrated_path,
        onnx_path=onnx_path,
        metadata_path=metadata_path,
        manifest=load_manifest(tmp_path),
    )
    assert report.max_abs_logit_diff <= 1e-4

    sample = next(iter_tensorized_samples(tmp_path, verify_integrity=True))
    model_input = torch.from_numpy(sample.model_input.copy().reshape(1, -1))
    with torch.no_grad():
        source_logits = source_model(model_input)
        migrated_logits = migrated_model(model_input)
    torch.testing.assert_close(migrated_logits, source_logits, rtol=0, atol=1e-6)


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


def test_policy_onnx_metadata_records_runtime_contract(tmp_path: Path) -> None:
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

    model, checkpoint = load_policy_checkpoint(checkpoint_path, manifest=load_manifest(tmp_path))
    metadata = build_policy_onnx_metadata(model=model, checkpoint=checkpoint)

    validate_policy_onnx_metadata(metadata)
    assert metadata["datasetSchemaVersion"] == 1
    assert metadata["playingEncoderSchemaVersion"] == 2
    assert metadata["modelInputSchemaVersion"] == 2
    assert metadata["cardIdsSha256"] == calculate_card_ids_sha256()
    assert metadata["policyModel"] == {
        "input_dim": MODEL_INPUT_FEATURE_COUNT,
        "hidden_dim": 8,
        "hidden_layers": 1,
        "hidden_dims": [8],
        "dropout": 0.0,
    }
    onnx_metadata = cast(dict[str, object], metadata["onnx"])
    assert onnx_metadata["inputs"] == [
        {
            "name": ONNX_INPUT_NAME,
            "shape": ["batch", MODEL_INPUT_FEATURE_COUNT],
            "dtype": "float32",
        }
    ]
    assert onnx_metadata["outputs"] == [
        {"name": ONNX_OUTPUT_NAME, "shape": ["batch", CARD_COUNT], "dtype": "float32"}
    ]
    assert onnx_metadata["opsetVersion"] == ONNX_OPSET_VERSION

    invalid_metadata = dict(metadata)
    invalid_metadata["onnx"] = dict(onnx_metadata, opsetVersion=ONNX_OPSET_VERSION + 1)
    with pytest.raises(PolicyCheckpointCompatibilityError, match="opsetVersion"):
        validate_policy_onnx_metadata(invalid_metadata)


def test_policy_onnx_export_rejects_checkpoint_with_wrong_input_shape_before_output(
    tmp_path: Path,
) -> None:
    _write_dataset(tmp_path, seeds=(0,))
    checkpoint_path = tmp_path.parent / f"{tmp_path.name}-bad-policy.pt"
    bad_model = PolicyMlpModel(PolicyMlpConfig(input_dim=10, hidden_dim=8, hidden_layers=1))
    torch.save(
        {
            "checkpoint_schema_version": 1,
            "model_state": bad_model.state_dict(),
            "model_config": bad_model.config.to_dict(),
            "training_config": {},
            "dataset_schema_version": 1,
            "playing_encoder_schema_version": 2,
            "model_input_schema_version": 2,
            "card_ids_sha256": calculate_card_ids_sha256(),
        },
        checkpoint_path,
    )
    onnx_path = tmp_path.parent / f"{tmp_path.name}-policy.onnx"
    metadata_path = tmp_path.parent / f"{tmp_path.name}-policy.json"

    with pytest.raises(PolicyCheckpointCompatibilityError, match="input_dim"):
        export_policy_checkpoint_to_onnx(
            dataset_directory=tmp_path,
            checkpoint_path=checkpoint_path,
            onnx_path=onnx_path,
            metadata_path=metadata_path,
            manifest=load_manifest(tmp_path),
        )

    assert not onnx_path.exists()
    assert not metadata_path.exists()


def test_policy_onnx_export_cli_rejects_overlapping_output_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_dataset(tmp_path, seeds=(0, 1, 2))
    checkpoint_path = tmp_path.parent / f"{tmp_path.name}-policy.pt"
    output_path = tmp_path.parent / f"{tmp_path.name}-policy.json"
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
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 1
    assert "must be different paths" in capsys.readouterr().err
    assert not output_path.exists()


@pytest.mark.parametrize("metadata_nested_under_output", (True, False))
def test_policy_onnx_export_cli_rejects_nested_artifact_paths_before_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    metadata_nested_under_output: bool,
) -> None:
    _write_dataset(tmp_path, seeds=(0, 1, 2))
    checkpoint_path = tmp_path.parent / f"{tmp_path.name}-policy.pt"
    artifact_root = tmp_path.parent / f"{tmp_path.name}-artifact"
    if metadata_nested_under_output:
        onnx_path = artifact_root
        metadata_path = artifact_root / "policy.json"
    else:
        onnx_path = artifact_root / "policy.onnx"
        metadata_path = artifact_root
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
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(onnx_path),
            "--metadata-output",
            str(metadata_path),
        ]
    )

    assert exit_code == 1
    assert "must not be nested under each other" in capsys.readouterr().err
    assert not artifact_root.exists()


def test_policy_onnx_export_cli_rejects_checkpoint_output_path_overlap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_dataset(tmp_path, seeds=(0, 1, 2))
    checkpoint_path = tmp_path.parent / f"{tmp_path.name}-policy.pt"
    metadata_path = tmp_path.parent / f"{tmp_path.name}-policy.json"
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
    original_checkpoint = checkpoint_path.read_bytes()

    exit_code = export_main(
        [
            str(tmp_path),
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(checkpoint_path),
            "--metadata-output",
            str(metadata_path),
        ]
    )

    assert exit_code == 1
    assert "checkpoint input must be different paths" in capsys.readouterr().err
    assert checkpoint_path.read_bytes() == original_checkpoint
    assert not metadata_path.exists()


def test_policy_onnx_export_cli_rejects_corrupted_later_dataset_bytes_before_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_dataset(tmp_path, seeds=(0, 1, 2))
    checkpoint_path = tmp_path.parent / f"{tmp_path.name}-policy.pt"
    onnx_path = tmp_path.parent / f"{tmp_path.name}-policy.onnx"
    metadata_path = tmp_path.parent / f"{tmp_path.name}-policy.json"
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
    with (tmp_path / "shard-00000.jsonl").open("ab") as shard:
        shard.write(b"corrupt")

    exit_code = export_main(
        [
            str(tmp_path),
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(onnx_path),
            "--metadata-output",
            str(metadata_path),
        ]
    )

    assert exit_code == 1
    assert "final line is missing a trailing newline" in capsys.readouterr().err
    assert not onnx_path.exists()
    assert not metadata_path.exists()


def test_policy_onnx_export_keeps_existing_artifact_when_staged_validation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_dataset(tmp_path, seeds=(0, 1, 2))
    checkpoint_path = tmp_path.parent / f"{tmp_path.name}-policy.pt"
    onnx_path = tmp_path.parent / f"{tmp_path.name}-policy.onnx"
    metadata_path = tmp_path.parent / f"{tmp_path.name}-policy.json"
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
    onnx_path.write_bytes(b"previous onnx")

    def write_staged_onnx(
        *, model: PolicyMlpModel, dummy_input: torch.Tensor, output: Path
    ) -> None:
        output.write_bytes(b"new onnx")

    def fail_parity(**kwargs: object) -> object:
        raise PolicyCheckpointCompatibilityError("parity failed")

    monkeypatch.setattr(onnx_export_module, "_export_onnx", write_staged_onnx)
    monkeypatch.setattr(onnx_export_module, "_check_onnx_runtime_parity", fail_parity)

    with pytest.raises(PolicyCheckpointCompatibilityError, match="parity failed"):
        export_policy_checkpoint_to_onnx(
            dataset_directory=tmp_path,
            checkpoint_path=checkpoint_path,
            onnx_path=onnx_path,
            metadata_path=metadata_path,
            manifest=load_manifest(tmp_path),
        )

    assert onnx_path.read_bytes() == b"previous onnx"
    assert not metadata_path.exists()
    assert not list(tmp_path.parent.glob(f".{tmp_path.name}-policy.*.tmp"))


def test_policy_onnx_export_rejects_metadata_directory_before_replacing_existing_onnx(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_dataset(tmp_path, seeds=(0, 1, 2))
    checkpoint_path = tmp_path.parent / f"{tmp_path.name}-policy.pt"
    onnx_path = tmp_path.parent / f"{tmp_path.name}-policy.onnx"
    metadata_path = tmp_path.parent / f"{tmp_path.name}-policy.json"
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
    onnx_path.write_bytes(b"previous onnx")
    metadata_path.mkdir()

    exit_code = export_main(
        [
            str(tmp_path),
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(onnx_path),
            "--metadata-output",
            str(metadata_path),
        ]
    )

    assert exit_code == 1
    assert "existing non-file path" in capsys.readouterr().err
    assert onnx_path.read_bytes() == b"previous onnx"
    assert metadata_path.is_dir()


def test_policy_onnx_export_cli_writes_model_metadata_and_checks_runtime_parity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    onnx = pytest.importorskip("onnx")
    onnxruntime = pytest.importorskip("onnxruntime")
    _write_dataset(tmp_path, seeds=(0, 1, 2))
    checkpoint_path = tmp_path.parent / f"{tmp_path.name}-policy.pt"
    onnx_path = tmp_path.parent / f"{tmp_path.name}-policy.onnx"
    metadata_path = tmp_path.parent / f"{tmp_path.name}-policy.json"
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
    validate_policy_onnx_metadata(metadata)
    report = json.loads(capsys.readouterr().out)
    assert report["onnxPath"] == str(onnx_path)
    assert report["metadataPath"] == str(metadata_path)
    assert report["paritySample"]["pytorchSelectedCardIndex"] == report["paritySample"][
        "onnxSelectedCardIndex"
    ]
    assert report["paritySample"]["maxAbsLogitDiff"] <= 1e-4

    samples = list(iter_tensorized_samples(tmp_path, verify_integrity=True))[:2]
    batch_input = torch.stack(
        [torch.from_numpy(sample.model_input.copy()) for sample in samples]
    ).numpy()
    session = onnxruntime.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    batch_logits = session.run([ONNX_OUTPUT_NAME], {ONNX_INPUT_NAME: batch_input})[0]
    assert batch_logits.shape == (2, CARD_COUNT)


def _write_current_policy_checkpoint(path: Path, model: PolicyMlpModel) -> None:
    torch.save(
        {
            "checkpoint_schema_version": 1,
            "model_state": model.state_dict(),
            "model_config": model.config.to_dict(),
            "training_config": {
                "hidden_dim": model.config.hidden_dim,
                "hidden_layers": model.config.hidden_layers,
                "hidden_dims": list(model.config.hidden_widths),
            },
            "dataset_schema_version": 1,
            "playing_encoder_schema_version": 2,
            "model_input_schema_version": 2,
            "card_ids_sha256": calculate_card_ids_sha256(),
        },
        path,
    )
