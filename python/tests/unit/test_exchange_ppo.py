from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from napoleon_ml.dataset.constants import CARD_COUNT, EXCHANGE_ENCODER_SCHEMA_VERSION
from napoleon_ml.dataset.tensors import (
    EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
    EXCHANGE_MODEL_INPUT_SCHEMA_VERSION,
)
from napoleon_ml.dataset.validation import calculate_card_ids_sha256
from napoleon_ml.exchange.ppo import (
    NON_PLAYING_RL_ALL_PASS_RULE_ID,
    NON_PLAYING_RL_REWARD_ID,
    NON_PLAYING_RL_TERMINAL_REWARD_TRANSFORM_ID,
    ExchangePpoCompatibilityError,
    iter_non_playing_exchange_rl_samples,
    load_non_playing_exchange_rl_manifest,
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


def _write_zero_sample_dataset(directory: Path) -> Path:
    directory.mkdir(parents=True)
    shard = b""
    (directory / "shard-00000.jsonl").write_bytes(shard)
    manifest = {
        "datasetSchemaVersion": 3,
        "generatorVersion": 1,
        "format": "jsonl",
        "sampleType": "non-playing-exchange-rl-sample",
        "sampleSchemaVersion": 3,
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
            "starterPayoff": 1,
            "otherPayoff": -1,
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
