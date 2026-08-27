from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from napoleon_ml.dataset.constants import CARD_COUNT, EXCHANGE_ENCODER_SCHEMA_VERSION
from napoleon_ml.dataset.tensors import (
    EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
    EXCHANGE_MODEL_INPUT_SCHEMA_VERSION,
)
from napoleon_ml.dataset.validation import calculate_card_ids_sha256
from napoleon_ml.exchange.model import ExchangeActorCriticModel, ExchangeMlpConfig
from napoleon_ml.exchange.ppo import (
    EXCHANGE_ACTOR_CRITIC_MODEL_ARCHITECTURE,
    EXCHANGE_PPO_ALGORITHM,
    NON_PLAYING_RL_ALL_PASS_RULE_ID,
    NON_PLAYING_RL_REWARD_ID,
    NON_PLAYING_RL_TERMINAL_REWARD_TRANSFORM_ID,
    ExchangePpoCompatibilityError,
    ExchangePpoTrainSettings,
    iter_non_playing_exchange_rl_samples,
    load_exchange_ppo_checkpoint,
    load_non_playing_exchange_rl_manifest,
    train_exchange_ppo,
)


def test_exchange_ppo_loader_accepts_zero_sample_shards(tmp_path: Path) -> None:
    dataset = _write_zero_sample_dataset(tmp_path / "dataset")

    manifest = load_non_playing_exchange_rl_manifest(dataset)
    samples = list(iter_non_playing_exchange_rl_samples(dataset))

    assert manifest.sample_count == 0
    assert samples == []


def test_exchange_ppo_loader_rejects_missing_all_pass_rule(tmp_path: Path) -> None:
    dataset = _write_zero_sample_dataset(tmp_path / "dataset")
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["allPassRule"]
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ExchangePpoCompatibilityError, match="allPassRule"):
        load_non_playing_exchange_rl_manifest(dataset)


def test_exchange_ppo_loader_rejects_missing_terminal_reward_transform(tmp_path: Path) -> None:
    dataset = _write_zero_sample_dataset(tmp_path / "dataset")
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["terminalRewardTransform"]
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ExchangePpoCompatibilityError, match="terminalRewardTransform"):
        load_non_playing_exchange_rl_manifest(dataset)


def test_exchange_ppo_train_checkpoint_smoke(tmp_path: Path) -> None:
    dataset = _write_rl_dataset(tmp_path / "dataset")
    output = tmp_path / "exchange-ppo.pt"

    report = train_exchange_ppo(
        dataset_directory=dataset,
        output_checkpoint_path=output,
        settings=ExchangePpoTrainSettings(
            seed=123,
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            training_device="cpu",
        ),
        model_config=ExchangeMlpConfig(hidden_dim=8, hidden_layers=1, dropout=0.0),
    )

    assert report.sample_count == 4
    assert report.requested_training_device == "cpu"
    assert report.resolved_training_device == "cpu"
    model, checkpoint = load_exchange_ppo_checkpoint(output)
    assert isinstance(model, ExchangeActorCriticModel)
    assert checkpoint["algorithm"] == EXCHANGE_PPO_ALGORITHM
    assert checkpoint["model_architecture"] == EXCHANGE_ACTOR_CRITIC_MODEL_ARCHITECTURE
    training_config = checkpoint["training_config"]
    assert isinstance(training_config, dict)
    assert training_config["trainingDevice"] == "cpu"
    assert report.to_dict()["resolvedTrainingDevice"] == "cpu"
    model_state = checkpoint["model_state"]
    assert isinstance(model_state, dict)
    assert {tensor.device.type for tensor in model_state.values()} == {"cpu"}


def _write_zero_sample_dataset(directory: Path) -> Path:
    directory.mkdir(parents=True)
    shard = b""
    (directory / "shard-00000.jsonl").write_bytes(shard)
    manifest = {
        "datasetSchemaVersion": 4,
        "generatorVersion": 5,
        "format": "jsonl",
        "sampleType": "non-playing-exchange-rl-sample",
        "sampleSchemaVersion": 4,
        "phaseScope": "exchange-only",
        "learnedPhases": ["exchanging"],
        "ruleBasedPhases": ["bidding", "choosing-adjutant"],
        "fixedPhases": ["playing"],
        "startSeed": 0,
        "endSeed": 0,
        "gameCount": 1,
        "sampleCount": 0,
        "gamesPerShard": 1,
        "shardCount": 1,
        "playerCount": 5,
        "cardCount": 53,
        "cardIds": [],
        "cardIdsSha256": calculate_card_ids_sha256(),
        "exchangeEncoderSchemaVersion": EXCHANGE_ENCODER_SCHEMA_VERSION,
        "exchangeModelInputSchemaVersion": EXCHANGE_MODEL_INPUT_SCHEMA_VERSION,
        "exchangeModelInputFeatureCount": EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
        "playingModelInputSchemaVersion": 2,
        "playingModelInputFeatureCount": 6246,
        "actionCount": CARD_COUNT,
        "decisionMode": "sequential-card-v1",
        "behaviorPolicy": _policy("exchange-onnx"),
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
            "bidding": {"type": "conservative-bidding", "id": "conservative-bidding-v1"},
            "choosingAdjutant": {"type": "rule-based", "version": 1},
        },
        "shards": [
            {
                "file": "shard-00000.jsonl",
                "startSeed": 0,
                "endSeed": 0,
                "gameCount": 1,
                "sampleCount": 0,
                "byteLength": 0,
                "sha256": hashlib.sha256(shard).hexdigest(),
            }
        ],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return directory


def _write_rl_dataset(directory: Path) -> Path:
    directory.mkdir(parents=True)
    samples = [
        _sample(seed=7, step=index + 1, selected=index % 2, terminal_reward=reward)
        for index, reward in enumerate([-3.0, 18.5, -3.0, 18.5])
    ]
    shard = "".join(json.dumps(sample, separators=(",", ":")) + "\n" for sample in samples)
    (directory / "shard-00000.jsonl").write_text(shard, encoding="utf-8")
    manifest = {
        "datasetSchemaVersion": 4,
        "generatorVersion": 5,
        "format": "jsonl",
        "sampleType": "non-playing-exchange-rl-sample",
        "sampleSchemaVersion": 4,
        "phaseScope": "exchange-only",
        "learnedPhases": ["exchanging"],
        "ruleBasedPhases": ["bidding", "choosing-adjutant"],
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
        "exchangeEncoderSchemaVersion": EXCHANGE_ENCODER_SCHEMA_VERSION,
        "exchangeModelInputSchemaVersion": EXCHANGE_MODEL_INPUT_SCHEMA_VERSION,
        "exchangeModelInputFeatureCount": EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
        "playingModelInputSchemaVersion": 2,
        "playingModelInputFeatureCount": 6246,
        "actionCount": CARD_COUNT,
        "decisionMode": "sequential-card-v1",
        "behaviorPolicy": _policy("exchange-onnx"),
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
            "bidding": {"type": "conservative-bidding", "id": "conservative-bidding-v1"},
            "choosingAdjutant": {"type": "rule-based", "version": 1},
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


def _sample(
    *,
    seed: int,
    step: int,
    selected: int,
    terminal_reward: float,
) -> dict[str, object]:
    legal = [0] * CARD_COUNT
    legal[0] = 1
    legal[1] = 1
    return {
        "sampleType": "non-playing-exchange-rl-sample",
        "schemaVersion": 4,
        "seed": seed,
        "step": step,
        "phase": "exchanging",
        "actingPlayerId": "player-0",
        "actingPlayerIndex": 0,
        "relativePlayerIds": ["player-0", "player-1", "player-2", "player-3", "player-4"],
        "modelInput": [0.0] * EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
        "legalDiscardCardMask": legal,
        "selectedActionIndex": selected,
        "behaviorLogProbability": math.log(0.5),
        "terminalReward": terminal_reward,
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
