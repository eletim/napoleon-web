from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch

import napoleon_ml.policy.reinforce as reinforce_module
from napoleon_ml.cli.train_policy_reinforce import main as reinforce_main
from napoleon_ml.dataset.constants import CARD_COUNT, EXPECTED_CARD_IDS
from napoleon_ml.dataset.pytorch import create_playing_self_play_dataloader
from napoleon_ml.dataset.reader import load_manifest
from napoleon_ml.dataset.sample import PlayingSelfPlaySample
from napoleon_ml.dataset.split import DatasetSplit, SplitConfig
from napoleon_ml.dataset.validation import calculate_card_ids_sha256
from napoleon_ml.policy.checkpoint import (
    PolicyCheckpointCompatibilityError,
    load_policy_checkpoint,
)
from napoleon_ml.policy.model import PolicyMlpConfig, PolicyMlpModel
from napoleon_ml.policy.onnx_export import build_policy_onnx_metadata
from napoleon_ml.policy.reinforce import (
    ReinforceTrainSettings,
    masked_selected_log_probability,
    reinforce_policy_loss,
    train_policy_reinforce,
)

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "valid_sample.json"


def _load_valid_sample() -> dict[str, Any]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def test_reinforce_loss_sign_mask_temperature_and_forced_gradient() -> None:
    logits = torch.tensor([[0.0, 1.0, 100.0], [2.0, -5.0, 99.0]], requires_grad=True)
    padded_logits = torch.zeros((2, CARD_COUNT), dtype=torch.float32, requires_grad=True)
    padded_logits.data[:, :3] = logits.data
    selected = torch.tensor([1, 2])
    legal_mask = torch.zeros((2, CARD_COUNT), dtype=torch.bool)
    legal_mask[0, [0, 1]] = True
    legal_mask[1, 2] = True

    selected_logp = masked_selected_log_probability(
        padded_logits,
        selected,
        legal_mask,
        temperature=2.0,
    )
    expected_first = torch.log_softmax(torch.tensor([0.0 / 2.0, 1.0 / 2.0]), dim=0)[1]

    assert float(selected_logp[0].detach().item()) == pytest.approx(float(expected_first))
    assert selected_logp[1].item() == 0.0

    loss = reinforce_policy_loss(selected_logp, torch.tensor([1.0, -1.0]))
    expected_loss = -float(expected_first) / 2.0
    assert loss.item() == pytest.approx(expected_loss)
    loss.backward()  # type: ignore[no-untyped-call]

    assert padded_logits.grad is not None
    assert padded_logits.grad[0, 2].item() == 0.0
    assert torch.count_nonzero(padded_logits.grad[1]).item() == 0


def test_reinforce_rejects_illegal_selected_card() -> None:
    logits = torch.zeros((1, CARD_COUNT), dtype=torch.float32)
    selected = torch.tensor([2])
    legal_mask = torch.zeros((1, CARD_COUNT), dtype=torch.bool)
    legal_mask[0, 1] = True

    with pytest.raises(ValueError, match="selected_card_index must be legal"):
        masked_selected_log_probability(logits, selected, legal_mask, temperature=1.0)


def test_reinforce_cli_updates_checkpoint_and_is_reproducible(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    supervised_dataset = tmp_path / "supervised"
    self_play_dataset = tmp_path / "self-play"
    supervised_dataset.mkdir()
    self_play_dataset.mkdir()
    model = PolicyMlpModel(PolicyMlpConfig(hidden_dim=8, hidden_layers=1))
    checkpoint_path = tmp_path / "input.pt"
    output_a = tmp_path / "output-a.pt"
    output_b = tmp_path / "output-b.pt"
    _write_supervised_dataset(supervised_dataset, seeds=(0, 1))
    checkpoint = _write_checkpoint(checkpoint_path, model)
    _write_self_play_dataset(
        self_play_dataset,
        model=model,
        checkpoint=checkpoint,
        rewards=(1, -1),
    )

    first_exit = reinforce_main(
        [
            str(self_play_dataset),
            "--input-checkpoint",
            str(checkpoint_path),
            "--output",
            str(output_a),
            "--learning-rate",
            "0.01",
            "--epochs",
            "2",
            "--batch-size",
            "1",
            "--device",
            "cpu",
            "--seed",
            "123",
            "--json",
        ]
    )
    assert first_exit == 0
    first_report = json.loads(capsys.readouterr().out)
    assert first_report["optimizerStepCount"] == 4
    assert first_report["sampleCount"] == 2
    assert first_report["forcedSampleCount"] == 1
    assert first_report["nonForcedSampleCount"] == 1
    assert first_report["maxBehaviorLogProbabilityParityError"] == pytest.approx(0.0)
    assert first_report["parameterDeltaNorm"] > 0
    assert first_report["changedParameterCount"] > 0
    assert first_report["requestedDevice"] == "cpu"
    assert first_report["resolvedDevice"] == "cpu"
    assert first_report["cudaDeviceName"] is None
    for key in (
        "preEvalElapsedSeconds",
        "optimizerTrainingElapsedSeconds",
        "postEvalElapsedSeconds",
        "totalElapsedSeconds",
    ):
        value = first_report[key]
        assert isinstance(value, int | float)
        assert float(value) >= 0.0

    second_exit = reinforce_main(
        [
            str(self_play_dataset),
            "--input-checkpoint",
            str(checkpoint_path),
            "--output",
            str(output_b),
            "--learning-rate",
            "0.01",
            "--epochs",
            "2",
            "--batch-size",
            "1",
            "--seed",
            "123",
            "--json",
        ]
    )
    assert second_exit == 0
    capsys.readouterr()

    first_raw = torch.load(output_a, map_location="cpu", weights_only=True)
    second_raw = torch.load(output_b, map_location="cpu", weights_only=True)
    _assert_state_dicts_equal(first_raw["model_state"], second_raw["model_state"])
    first_provenance = cast(dict[str, object], first_raw["rl_provenance"])
    assert first_provenance["algorithm"] == "reinforce-v1"
    assert first_provenance["optimizerSteps"] == 4
    assert first_provenance["requestedDevice"] == "cpu"
    assert first_provenance["resolvedDevice"] == "cpu"
    assert first_provenance["cudaDeviceName"] is None

    loaded_model, loaded_checkpoint = load_policy_checkpoint(
        output_a,
        manifest=load_manifest(supervised_dataset),
    )
    assert isinstance(loaded_model, PolicyMlpModel)
    loaded_provenance = cast(dict[str, object], loaded_checkpoint["rl_provenance"])
    assert loaded_provenance["sampleCount"] == 2


def test_reinforce_wrong_checkpoint_fails_before_optimizer_step(tmp_path: Path) -> None:
    self_play_dataset = tmp_path / "self-play"
    self_play_dataset.mkdir()
    behavior_model = PolicyMlpModel(PolicyMlpConfig(hidden_dim=8, hidden_layers=1))
    checkpoint_path = tmp_path / "wrong.pt"
    checkpoint = _write_checkpoint(checkpoint_path, behavior_model)
    _write_self_play_dataset(
        self_play_dataset,
        model=behavior_model,
        checkpoint=checkpoint,
        rewards=(1, -1),
    )

    wrong_raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = cast(dict[str, torch.Tensor], wrong_raw["model_state"])
    final_bias_key = "network.2.bias"
    state[final_bias_key] = state[final_bias_key].clone()
    state[final_bias_key][16] += 5.0
    torch.save(wrong_raw, checkpoint_path)

    loader = create_playing_self_play_dataloader(
        self_play_dataset,
        split=DatasetSplit.TRAIN,
        split_config=SplitConfig(train=100, validation=0, test=0),
        batch_size=2,
    )
    with pytest.raises(PolicyCheckpointCompatibilityError, match="別policy"):
        train_policy_reinforce(
            input_checkpoint=checkpoint_path,
            self_play_dataset_directory=self_play_dataset,
            output_checkpoint=tmp_path / "should-not-exist.pt",
            manifest=load_manifest(self_play_dataset),
            dataloader=loader,
            settings=ReinforceTrainSettings(
                seed=0,
                epochs=1,
                batch_size=2,
                learning_rate=0.01,
                verify_integrity=True,
            ),
        )

    assert not (tmp_path / "should-not-exist.pt").exists()


def test_reinforce_does_not_save_when_report_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    self_play_dataset = tmp_path / "self-play"
    self_play_dataset.mkdir()
    model = PolicyMlpModel(PolicyMlpConfig(hidden_dim=8, hidden_layers=1))
    checkpoint_path = tmp_path / "input.pt"
    output_path = tmp_path / "should-not-exist.pt"
    checkpoint = _write_checkpoint(checkpoint_path, model)
    _write_self_play_dataset(
        self_play_dataset,
        model=model,
        checkpoint=checkpoint,
        rewards=(1, -1),
    )
    save_called = False

    def fail_report_validation(report: object) -> None:
        del report
        raise ValueError("synthetic nonfinite report")

    def record_save(*args: object, **kwargs: object) -> None:
        nonlocal save_called
        del args, kwargs
        save_called = True

    monkeypatch.setattr(reinforce_module, "_assert_report_finite", fail_report_validation)
    monkeypatch.setattr(reinforce_module, "_save_reinforce_checkpoint", record_save)

    loader = create_playing_self_play_dataloader(
        self_play_dataset,
        split=DatasetSplit.TRAIN,
        split_config=SplitConfig(train=100, validation=0, test=0),
        batch_size=2,
    )
    with pytest.raises(ValueError, match="synthetic nonfinite report"):
        train_policy_reinforce(
            input_checkpoint=checkpoint_path,
            self_play_dataset_directory=self_play_dataset,
            output_checkpoint=output_path,
            manifest=load_manifest(self_play_dataset),
            dataloader=loader,
            settings=ReinforceTrainSettings(
                seed=0,
                epochs=1,
                batch_size=2,
                learning_rate=0.01,
                verify_integrity=True,
            ),
        )

    assert not save_called
    assert not output_path.exists()


def _write_checkpoint(path: Path, model: PolicyMlpModel) -> dict[str, object]:
    checkpoint: dict[str, object] = {
        "checkpoint_schema_version": 1,
        "model_state": model.state_dict(),
        "model_config": model.config.to_dict(),
        "training_config": {"source": "unit-test"},
        "dataset_schema_version": 1,
        "playing_encoder_schema_version": 2,
        "model_input_schema_version": 2,
        "card_ids_sha256": calculate_card_ids_sha256(),
    }
    torch.save(checkpoint, path)
    return checkpoint


def _write_supervised_dataset(directory: Path, *, seeds: tuple[int, ...]) -> None:
    base = _load_valid_sample()
    samples = [dict(base, seed=seed, step=1) for seed in seeds]
    _write_dataset_manifest(
        directory,
        samples=samples,
        manifest_extra={
            "datasetSchemaVersion": 1,
            "generatorVersion": 1,
            "playingEncoderSchemaVersion": 2,
            "sampleType": "playing-training-sample",
            "agent": {"type": "rule-based", "version": 1},
        },
    )


def _write_self_play_dataset(
    directory: Path,
    *,
    model: PolicyMlpModel,
    checkpoint: dict[str, object],
    rewards: tuple[int, int],
) -> None:
    base = _load_valid_sample()
    selected = int(base["actorTarget"]["selectedCardIndex"])
    metadata = build_policy_onnx_metadata(model=model, checkpoint=checkpoint)
    samples = []
    for index, reward in enumerate(rewards):
        sample = json.loads(json.dumps(base))
        del sample["actorTarget"]
        del sample["beliefTarget"]
        if index == 1:
            sample["observation"]["legalPlayMask"] = [
                1 if card_index == selected else 0 for card_index in range(CARD_COUNT)
            ]
        sample.update(
            {
                "sampleType": "playing-self-play-sample",
                "schemaVersion": 3,
                "seed": index,
                "step": 1,
                "actingSeatSource": "current-policy",
                "behaviorPolicyArtifactId": "unit-policy",
                "rolloutSeatSources": [
                    "current-policy",
                    "current-policy",
                    "current-policy",
                    "current-policy",
                    "current-policy",
                ],
                "selectedCardIndex": selected,
                "behaviorLogProbability": 0,
                "terminalReward": reward,
                "outcome": {
                    "winner": "napoleon-team" if reward == 1 else "alliance",
                    "napoleonPlayerId": "player-4",
                    "actingPlayerTeam": "napoleon-team",
                    "actingPlayerRole": "napoleon",
                },
            }
        )
        sample["behaviorLogProbability"] = _behavior_log_probability(model, sample, selected)
        samples.append(sample)

    _write_dataset_manifest(
        directory,
        samples=samples,
        manifest_extra={
            "datasetSchemaVersion": 3,
            "generatorVersion": 1,
            "sampleType": "playing-self-play-sample",
            "sampleSchemaVersion": 3,
            "playingEncoderSchemaVersion": 2,
            "playingModelInputSchemaVersion": 2,
            "behaviorPolicy": {
                "type": "playing-onnx",
                "artifactId": "unit-policy",
                "onnxFileName": "policy.onnx",
                "metadataFileName": "policy.json",
                "onnxSha256": "a" * 64,
                "metadataSha256": "b" * 64,
                "metadata": metadata,
            },
            "samplingAlgorithm": "masked-categorical",
            "temperature": 1.0,
            "reward": {"type": "terminal-team-win", "version": 1},
            "nonPlayingAgent": {"type": "rule-based", "version": 1},
            "rolloutRoster": {
                "assignment": "rotate-by-seed",
                "seats": [{"source": "current-policy"} for _ in range(5)],
            },
        },
    )


def _behavior_log_probability(
    model: PolicyMlpModel,
    raw_sample: dict[str, Any],
    selected: int,
) -> float:
    from napoleon_ml.dataset.sample import parse_sample
    from napoleon_ml.dataset.tensors import tensorize_playing_self_play_sample

    parsed = cast(PlayingSelfPlaySample, parse_sample(raw_sample))
    tensorized = tensorize_playing_self_play_sample(parsed)
    model_input = torch.from_numpy(tensorized.model_input.copy().reshape(1, -1))
    legal_mask = torch.from_numpy(tensorized.legal_play_mask.copy().reshape(1, -1)).to(
        dtype=torch.bool
    )
    selected_tensor = torch.tensor([selected], dtype=torch.long)
    with torch.no_grad():
        logits = model(model_input)
        logp = masked_selected_log_probability(
            logits,
            selected_tensor,
            legal_mask,
            temperature=1.0,
        )
    return float(logp.item())


def _write_dataset_manifest(
    directory: Path,
    *,
    samples: list[dict[str, Any]],
    manifest_extra: dict[str, Any],
) -> None:
    shard_bytes = b"".join((json.dumps(sample) + "\n").encode("utf-8") for sample in samples)
    (directory / "shard-00000.jsonl").write_bytes(shard_bytes)
    seeds = [int(sample["seed"]) for sample in samples]
    manifest = {
        **manifest_extra,
        "format": "jsonl",
        "startSeed": min(seeds),
        "endSeed": max(seeds),
        "gameCount": len(set(seeds)),
        "sampleCount": len(samples),
        "gamesPerShard": len(set(seeds)),
        "shardCount": 1,
        "playerCount": 5,
        "cardCount": CARD_COUNT,
        "cardIds": list(EXPECTED_CARD_IDS),
        "cardIdsSha256": calculate_card_ids_sha256(),
        "shards": [
            {
                "file": "shard-00000.jsonl",
                "startSeed": min(seeds),
                "endSeed": max(seeds),
                "gameCount": len(set(seeds)),
                "sampleCount": len(samples),
                "byteLength": len(shard_bytes),
                "sha256": hashlib.sha256(shard_bytes).hexdigest(),
            }
        ],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _assert_state_dicts_equal(left: object, right: object) -> None:
    left_state = cast(dict[str, torch.Tensor], left)
    right_state = cast(dict[str, torch.Tensor], right)
    assert left_state.keys() == right_state.keys()
    for key in left_state:
        assert torch.equal(left_state[key], right_state[key])
