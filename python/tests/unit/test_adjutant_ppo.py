from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from napoleon_ml.adjutant.model import AdjutantActorCriticModel, AdjutantMlpConfig, AdjutantMlpModel
from napoleon_ml.adjutant.ppo import (
    ADJUTANT_ACTOR_CRITIC_MODEL_ARCHITECTURE,
    ADJUTANT_PPO_ALGORITHM,
    NON_PLAYING_RL_ALL_PASS_RULE_ID,
    NON_PLAYING_RL_REWARD_ID,
    NON_PLAYING_RL_TERMINAL_REWARD_TRANSFORM_ID,
    AdjutantPpoCompatibilityError,
    AdjutantPpoTrainSettings,
    adjutant_ppo_loss,
    initialize_actor_from_checkpoint,
    iter_non_playing_adjutant_rl_samples,
    load_adjutant_ppo_checkpoint,
    load_non_playing_adjutant_rl_manifest,
    masked_selected_log_probability,
    train_adjutant_ppo,
)
from napoleon_ml.cli.export_policy_onnx import main as export_policy_onnx_main
from napoleon_ml.cli.train_adjutant_ppo import main as train_adjutant_ppo_main
from napoleon_ml.dataset.constants import CARD_COUNT
from napoleon_ml.dataset.tensors import ADJUTANT_MODEL_INPUT_FEATURE_COUNT
from napoleon_ml.dataset.validation import calculate_card_ids_sha256
from napoleon_ml.nonplaying_onnx_export import export_adjutant_rl_checkpoint_to_onnx


def test_masked_selected_log_probability_handles_forced_and_illegal_actions() -> None:
    logits = torch.tensor([[5.0, -2.0, 1.0], [0.0, 2.0, 4.0]], dtype=torch.float32)
    logits = torch.nn.functional.pad(logits, (0, CARD_COUNT - 3))
    legal_mask = torch.zeros((2, CARD_COUNT), dtype=torch.bool)
    legal_mask[0, 0] = True
    legal_mask[1, 1] = True
    legal_mask[1, 2] = True

    log_probability = masked_selected_log_probability(
        logits,
        torch.tensor([0, 2], dtype=torch.int64),
        legal_mask,
    )

    assert log_probability[0].item() == 0.0
    assert log_probability[1].item() == pytest.approx(
        torch.log_softmax(torch.tensor([2.0, 4.0]), dim=0)[1].item()
    )

    with pytest.raises(ValueError, match="illegal action"):
        masked_selected_log_probability(
            logits,
            torch.tensor([1, 2], dtype=torch.int64),
            legal_mask,
        )


def test_adjutant_ppo_loss_clips_ratio_and_accepts_arbitrary_float_reward() -> None:
    logits = torch.zeros((2, CARD_COUNT), dtype=torch.float32)
    logits[0, 1] = math.log(0.9 / 0.1)
    logits[1, 1] = math.log(0.9 / 0.1)
    legal_mask = torch.zeros((2, CARD_COUNT), dtype=torch.bool)
    legal_mask[:, 0] = True
    legal_mask[:, 1] = True

    loss = adjutant_ppo_loss(
        logits=logits,
        value_prediction=torch.tensor([1.0, -2.0], dtype=torch.float32),
        selected_action_index=torch.tensor([1, 0], dtype=torch.int64),
        legal_adjutant_mask=legal_mask,
        behavior_log_probability=torch.log(torch.tensor([0.5, 0.5], dtype=torch.float32)),
        terminal_reward=torch.tensor([18.0, -3.0], dtype=torch.float32),
        clip_epsilon=0.2,
        value_loss_coefficient=0.5,
    )

    assert loss.probability_ratio.tolist() == pytest.approx([1.8, 0.2])
    assert loss.clipped_sample_mask.tolist() == [True, True]
    assert loss.value_loss.item() == pytest.approx(((17.0**2) + ((-1.0) ** 2)) / 2)
    assert torch.isfinite(loss.total_loss)


def test_adjutant_ppo_train_checkpoint_and_export_smoke(tmp_path: Path) -> None:
    dataset = _write_rl_dataset(tmp_path / "dataset")
    parent = tmp_path / "parent.pt"
    config = AdjutantMlpConfig(hidden_dim=8, hidden_layers=1, dropout=0.0)
    parent_model = AdjutantMlpModel(config)
    torch.save(
        {
            "model_state": parent_model.state_dict(),
            "model_config": config.to_dict(),
            "action_count": CARD_COUNT,
        },
        parent,
    )
    output = tmp_path / "adjutant-ppo.pt"
    report = train_adjutant_ppo(
        dataset_directory=dataset,
        output_checkpoint_path=output,
        settings=AdjutantPpoTrainSettings(
            seed=123,
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            parent_actor_checkpoint=str(parent),
        ),
        model_config=config,
    )

    assert report.sample_count == 4
    model, checkpoint = load_adjutant_ppo_checkpoint(output)
    assert isinstance(model, AdjutantActorCriticModel)
    assert checkpoint["algorithm"] == ADJUTANT_PPO_ALGORITHM
    assert checkpoint["model_architecture"] == ADJUTANT_ACTOR_CRITIC_MODEL_ARCHITECTURE
    reward = checkpoint["reward"]
    fixed_playing_policy = checkpoint["fixed_playing_policy"]
    assert isinstance(reward, dict)
    assert isinstance(fixed_playing_policy, dict)
    assert reward["id"] == NON_PLAYING_RL_REWARD_ID
    assert checkpoint["terminal_reward_transform"]["id"] == (
        NON_PLAYING_RL_TERMINAL_REWARD_TRANSFORM_ID
    )
    assert checkpoint["parent_actor_checkpoint_sha256"] == _sha256(parent)
    assert fixed_playing_policy["onnxSha256"] == "0" * 64

    onnx = tmp_path / "adjutant.onnx"
    metadata = tmp_path / "adjutant.json"
    export_report = export_adjutant_rl_checkpoint_to_onnx(
        dataset_directory=dataset,
        checkpoint_path=output,
        onnx_path=onnx,
        metadata_path=metadata,
    )
    metadata_json = json.loads(metadata.read_text(encoding="utf-8"))
    assert export_report.pytorch_selection == export_report.onnx_selection
    assert metadata_json["artifactType"] == "napoleon-adjutant-policy-onnx"
    assert metadata_json["policyType"] == "adjutant"
    assert metadata_json["outputLogitCount"] == CARD_COUNT


def test_adjutant_ppo_train_and_export_cli_smoke(tmp_path: Path) -> None:
    dataset = _write_rl_dataset(tmp_path / "dataset")
    checkpoint = tmp_path / "cli-adjutant-ppo.pt"
    onnx = tmp_path / "cli-adjutant.onnx"
    metadata = tmp_path / "cli-adjutant.json"

    assert train_adjutant_ppo_main(
        [
            str(dataset),
            "--output",
            str(checkpoint),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--hidden-dim",
            "8",
            "--hidden-layers",
            "1",
            "--json",
        ]
    ) == 0
    assert export_policy_onnx_main(
        [
            str(dataset),
            "--policy-type",
            "adjutant",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(onnx),
            "--metadata-output",
            str(metadata),
            "--json",
        ]
    ) == 0

    metadata_json = json.loads(metadata.read_text(encoding="utf-8"))
    assert metadata_json["policyType"] == "adjutant"
    assert metadata_json["checkpointCompatibilityMetadata"]["sampleType"] == (
        "non-playing-adjutant-rl-sample"
    )


def test_initialize_actor_from_actor_critic_checkpoint_leaves_critic_independent(
    tmp_path: Path,
) -> None:
    config = AdjutantMlpConfig(hidden_dim=8, hidden_layers=1, dropout=0.0)
    source = AdjutantActorCriticModel(config)
    target = AdjutantActorCriticModel(config)
    checkpoint = tmp_path / "actor-critic.pt"
    with torch.no_grad():
        for parameter in source.actor.parameters():
            parameter.fill_(0.25)
        for parameter in source.critic.parameters():
            parameter.fill_(0.75)
        for parameter in target.critic.parameters():
            parameter.fill_(0.0)
    torch.save(
        {
            "model_state": source.state_dict(),
            "model_config": config.to_dict(),
            "model_architecture": ADJUTANT_ACTOR_CRITIC_MODEL_ARCHITECTURE,
        },
        checkpoint,
    )

    initialize_actor_from_checkpoint(target, checkpoint)

    assert all(
        torch.equal(left, right)
        for left, right in zip(
            source.actor.parameters(), target.actor.parameters(), strict=True
        )
    )
    assert all(
        torch.equal(parameter, torch.zeros_like(parameter))
        for parameter in target.critic.parameters()
    )


def test_adjutant_ppo_loader_skips_zero_sample_shards(tmp_path: Path) -> None:
    dataset = _write_rl_dataset(tmp_path / "dataset")
    _append_empty_shard(dataset)

    samples = list(iter_non_playing_adjutant_rl_samples(dataset))

    assert len(samples) == 4


def test_adjutant_ppo_loader_rejects_missing_all_pass_rule(tmp_path: Path) -> None:
    dataset = _write_rl_dataset(tmp_path / "dataset")
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["allPassRule"]
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(AdjutantPpoCompatibilityError, match="allPassRule"):
        load_non_playing_adjutant_rl_manifest(dataset)


def test_adjutant_ppo_loader_rejects_missing_terminal_reward_transform(
    tmp_path: Path,
) -> None:
    dataset = _write_rl_dataset(tmp_path / "dataset")
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["terminalRewardTransform"]
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(AdjutantPpoCompatibilityError, match="terminalRewardTransform"):
        load_non_playing_adjutant_rl_manifest(dataset)


def _write_rl_dataset(directory: Path) -> Path:
    directory.mkdir(parents=True)
    samples = [_sample(seed=7, step=index + 1, selected=index % 2) for index in range(4)]
    shard = "".join(json.dumps(sample, separators=(",", ":")) + "\n" for sample in samples)
    (directory / "shard-00000.jsonl").write_text(shard, encoding="utf-8")
    manifest = {
        "datasetSchemaVersion": 4,
        "generatorVersion": 5,
        "format": "jsonl",
        "sampleType": "non-playing-adjutant-rl-sample",
        "sampleSchemaVersion": 4,
        "phaseScope": "adjutant-only",
        "learnedPhases": ["choosing-adjutant"],
        "ruleBasedPhases": ["bidding", "exchanging"],
        "fixedPhases": ["playing"],
        "startSeed": 7,
        "endSeed": 7,
        "gameCount": 1,
        "sampleCount": len(samples),
        "gamesPerShard": 1,
        "shardCount": 1,
        "playerCount": 5,
        "cardCount": 53,
        "cardIds": [],
        "cardIdsSha256": calculate_card_ids_sha256(),
        "adjutantEncoderSchemaVersion": 1,
        "adjutantModelInputSchemaVersion": 1,
        "adjutantModelInputFeatureCount": ADJUTANT_MODEL_INPUT_FEATURE_COUNT,
        "playingModelInputSchemaVersion": 2,
        "playingModelInputFeatureCount": 6246,
        "actionCount": CARD_COUNT,
        "behaviorPolicy": _policy("adjutant-onnx"),
        "fixedPlayingPolicy": _policy("playing-onnx"),
        "samplingAlgorithm": "masked-categorical",
        "temperature": 1.0,
        "reward": {
            "type": "non-playing-terminal-role-reward",
            "version": 3,
            "id": NON_PLAYING_RL_REWARD_ID,
        },
        "terminalRewardTransform": _terminal_reward_transform(),
        "allPassRule": {
            "id": NON_PLAYING_RL_ALL_PASS_RULE_ID,
            "starterPayoff": 0,
            "otherPayoff": 0,
        },
        "nonLearningAgents": {
            "choosingAdjutant": {"type": "rule-based", "version": 1},
            "exchanging": {"type": "rule-based", "version": 1},
        },
        "shards": [
            {
                "file": "shard-00000.jsonl",
                "startSeed": 7,
                "endSeed": 7,
                "gameCount": 1,
                "sampleCount": len(samples),
                "byteLength": len(shard.encode("utf-8")),
                "sha256": hashlib.sha256(shard.encode("utf-8")).hexdigest(),
            }
        ],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return directory


def _append_empty_shard(directory: Path) -> None:
    empty = b""
    empty_file = "shard-00001.jsonl"
    (directory / empty_file).write_bytes(empty)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["shardCount"] = 2
    manifest["shards"].append(
        {
            "file": empty_file,
            "startSeed": 8,
            "endSeed": 8,
            "gameCount": 1,
            "sampleCount": 0,
            "byteLength": 0,
            "sha256": hashlib.sha256(empty).hexdigest(),
        }
    )
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")


def _sample(*, seed: int, step: int, selected: int) -> dict[str, object]:
    legal = [0] * CARD_COUNT
    legal[0] = 1
    legal[1] = 1
    return {
        "sampleType": "non-playing-adjutant-rl-sample",
        "schemaVersion": 3,
        "seed": seed,
        "step": step,
        "phase": "choosing-adjutant",
        "actingPlayerId": "player-0",
        "actingPlayerIndex": 0,
        "relativePlayerIds": ["player-0", "player-1", "player-2", "player-3", "player-4"],
        "modelInput": [0.0] * ADJUTANT_MODEL_INPUT_FEATURE_COUNT,
        "legalAdjutantMask": legal,
        "selectedActionIndex": selected,
        "behaviorLogProbability": -0.6931471805599453,
        "terminalReward": 18.5 if step % 2 == 0 else -3.0,
        "outcome": {
            "winner": "napoleon-team",
            "targetPointCards": 18,
            "napoleonPlayerId": "player-0",
            "actingPlayerRole": "napoleon",
        },
    }


def _policy(policy_type: str) -> dict[str, object]:
    return {
        "type": policy_type,
        "artifactId": f"test-{policy_type}",
        "onnxFileName": f"{policy_type}.onnx",
        "metadataFileName": f"{policy_type}.json",
        "onnxSha256": "0" * 64,
        "metadataSha256": "1" * 64,
        "metadata": {},
    }


def _terminal_reward_transform() -> dict[str, object]:
    return {
        "type": "raw-reward-minus-game-player-mean",
        "version": 1,
        "id": NON_PLAYING_RL_TERMINAL_REWARD_TRANSFORM_ID,
        "sourceRewardId": NON_PLAYING_RL_REWARD_ID,
        "baseline": "meanRawRewardAllPlayers",
        "formula": "relative_reward_i = raw_reward_i - mean(raw_reward_all_players)",
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
