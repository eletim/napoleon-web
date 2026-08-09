from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from napoleon_ml.dataset.constants import CARD_COUNT, EXPECTED_CARD_IDS
from napoleon_ml.dataset.pytorch import create_playing_self_play_dataloader
from napoleon_ml.dataset.reader import load_manifest
from napoleon_ml.dataset.sample import PlayingSelfPlaySample
from napoleon_ml.dataset.split import DatasetSplit, SplitConfig
from napoleon_ml.dataset.validation import calculate_card_ids_sha256
from napoleon_ml.policy.actor_critic import (
    ACTOR_CRITIC_ALGORITHM,
    ActorCriticTrainReport,
    ActorCriticTrainSettings,
    actor_critic_losses,
    load_checkpoint_for_actor_critic,
    train_policy_actor_critic,
)
from napoleon_ml.policy.checkpoint import (
    ACTOR_CRITIC_MODEL_ARCHITECTURE,
    PolicyCheckpointCompatibilityError,
)
from napoleon_ml.policy.model import PolicyActorCriticModel, PolicyMlpConfig, PolicyMlpModel
from napoleon_ml.policy.onnx_export import build_policy_onnx_metadata
from napoleon_ml.policy.reinforce import (
    ReinforceTrainSettings,
    masked_selected_log_probability,
    train_policy_reinforce,
)

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "valid_sample.json"


def test_actor_critic_loss_uses_reward_value_advantage_and_detaches_actor_value() -> None:
    selected_logp = torch.tensor([-0.25, -0.75], requires_grad=True)
    value = torch.tensor([0.2, -0.4], requires_grad=True)
    reward = torch.tensor([1.0, -1.0])

    actor_loss, value_loss, total_loss = actor_critic_losses(
        selected_logp,
        reward,
        value,
        value_loss_coefficient=0.5,
    )

    expected_advantage = reward - value.detach()
    assert actor_loss.item() == pytest.approx(
        float(-torch.mean(expected_advantage * selected_logp).item())
    )
    assert value_loss.item() == pytest.approx(float(torch.mean((reward - value) ** 2).item()))
    assert total_loss.item() == pytest.approx(actor_loss.item() + 0.5 * value_loss.item())

    actor_loss.backward(retain_graph=True)  # type: ignore[no-untyped-call]
    assert selected_logp.grad is not None
    assert value.grad is None

    total_loss.backward()  # type: ignore[no-untyped-call]
    assert value.grad is not None
    assert torch.count_nonzero(value.grad).item() > 0


def test_actor_critic_forced_action_has_no_actor_gradient_but_updates_critic() -> None:
    logits = torch.zeros((1, CARD_COUNT), dtype=torch.float32, requires_grad=True)
    value = torch.tensor([0.0], requires_grad=True)
    selected = torch.tensor([3])
    legal_mask = torch.zeros((1, CARD_COUNT), dtype=torch.bool)
    legal_mask[0, 3] = True

    selected_logp = masked_selected_log_probability(
        logits,
        selected,
        legal_mask,
        temperature=1.0,
    )
    actor_loss, value_loss, total_loss = actor_critic_losses(
        selected_logp,
        torch.tensor([1.0]),
        value,
        value_loss_coefficient=0.5,
    )

    assert selected_logp.item() == 0.0
    assert actor_loss.item() == pytest.approx(0.0)
    assert value_loss.item() == pytest.approx(1.0)
    total_loss.backward()  # type: ignore[no-untyped-call]
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad).item() == 0
    assert value.grad is not None
    assert value.grad.item() != 0.0


def test_actor_critic_training_migrates_policy_logits_and_saves_checkpoint(
    tmp_path: Path,
) -> None:
    self_play_dataset = tmp_path / "self-play"
    self_play_dataset.mkdir()
    policy_model = PolicyMlpModel(PolicyMlpConfig(hidden_dim=8, hidden_layers=1))
    checkpoint_path = tmp_path / "input.pt"
    output_path = tmp_path / "output.pt"
    checkpoint = _write_checkpoint(checkpoint_path, policy_model)
    _write_self_play_dataset(
        self_play_dataset,
        model=policy_model,
        checkpoint=checkpoint,
        rewards=(1, -1),
    )

    loaded = load_checkpoint_for_actor_critic(
        checkpoint_path,
        manifest=load_manifest(self_play_dataset),
        value_head_seed=123,
    )
    assert loaded.migrated_from_policy
    assert isinstance(loaded.training_model, PolicyActorCriticModel)
    model_input = _first_model_input(self_play_dataset)
    with torch.no_grad():
        source_logits = policy_model(model_input)
        migrated_logits = loaded.training_model(model_input)
    torch.testing.assert_close(migrated_logits, source_logits)
    other_seed = load_checkpoint_for_actor_critic(
        checkpoint_path,
        manifest=load_manifest(self_play_dataset),
        value_head_seed=456,
    )
    assert not torch.equal(
        loaded.training_model.value_head.weight,
        other_seed.training_model.value_head.weight,
    )

    report = _run_actor_critic(
        input_checkpoint=checkpoint_path,
        self_play_dataset=self_play_dataset,
        output_checkpoint=output_path,
    )
    assert report.sample_count == 2
    assert report.optimizer_step_count == 1
    assert report.max_behavior_log_probability_parity_error == pytest.approx(0.0)
    assert report.actor_parameter_delta_norm > 0
    assert report.critic_parameter_delta_norm > 0
    assert report.changed_actor_parameter_count > 0
    assert report.changed_critic_parameter_count > 0
    assert report.role_stats_before["napoleon"]["sampleCount"] == 2
    assert report.positive_reward_count == 1
    assert report.negative_reward_count == 1

    raw = torch.load(output_path, map_location="cpu", weights_only=True)
    assert raw["model_architecture"] == ACTOR_CRITIC_MODEL_ARCHITECTURE
    provenance = cast(dict[str, object], raw["rl_provenance"])
    assert provenance["algorithm"] == ACTOR_CRITIC_ALGORITHM
    assert provenance["migratedFromPolicyCheckpoint"] is True
    assert provenance["valueHeadInitializationSeed"] == 0
    migration = cast(dict[str, object], raw["actor_critic_migration_provenance"])
    assert migration["policyLogitsPreserved"] is True

    reloaded = load_checkpoint_for_actor_critic(
        output_path,
        manifest=load_manifest(self_play_dataset),
    )
    assert not reloaded.migrated_from_policy
    assert isinstance(reloaded.training_model, PolicyActorCriticModel)

    loader = create_playing_self_play_dataloader(
        self_play_dataset,
        split=DatasetSplit.TRAIN,
        split_config=SplitConfig(train=100, validation=0, test=0),
        batch_size=2,
    )
    with pytest.raises(PolicyCheckpointCompatibilityError, match="Actor-Critic"):
        train_policy_reinforce(
            input_checkpoint=output_path,
            self_play_dataset_directory=self_play_dataset,
            output_checkpoint=tmp_path / "reinforce-should-not-exist.pt",
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


def test_actor_critic_wrong_behavior_checkpoint_fails_before_save(tmp_path: Path) -> None:
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

    with pytest.raises(PolicyCheckpointCompatibilityError, match="別policy"):
        _run_actor_critic(
            input_checkpoint=checkpoint_path,
            self_play_dataset=self_play_dataset,
            output_checkpoint=tmp_path / "should-not-exist.pt",
        )

    assert not (tmp_path / "should-not-exist.pt").exists()


def _run_actor_critic(
    *,
    input_checkpoint: Path,
    self_play_dataset: Path,
    output_checkpoint: Path,
) -> ActorCriticTrainReport:
    settings = ActorCriticTrainSettings(
        seed=0,
        epochs=1,
        batch_size=2,
        learning_rate=0.01,
        value_loss_coefficient=0.5,
        verify_integrity=True,
    )
    loader = create_playing_self_play_dataloader(
        self_play_dataset,
        split=DatasetSplit.TRAIN,
        split_config=SplitConfig(train=100, validation=0, test=0),
        batch_size=settings.batch_size,
    )
    return train_policy_actor_critic(
        input_checkpoint=input_checkpoint,
        self_play_dataset_directory=self_play_dataset,
        output_checkpoint=output_checkpoint,
        manifest=load_manifest(self_play_dataset),
        dataloader=loader,
        settings=settings,
    )


def _load_valid_sample() -> dict[str, Any]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


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


def _first_model_input(directory: Path) -> torch.Tensor:
    from napoleon_ml.dataset.reader import iter_tensorized_samples

    sample = next(iter_tensorized_samples(directory))
    return torch.from_numpy(sample.model_input.copy().reshape(1, -1))


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
