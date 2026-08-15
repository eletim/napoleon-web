from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from napoleon_ml.bidding.model import BiddingActorCriticModel, BiddingMlpConfig, BiddingMlpModel
from napoleon_ml.bidding.ppo import (
    BIDDING_ACTOR_CRITIC_MODEL_ARCHITECTURE,
    BIDDING_PPO_ALGORITHM,
    NON_PLAYING_RL_ALL_PASS_RULE_ID,
    NON_PLAYING_RL_REWARD_ID,
    BiddingPpoTrainSettings,
    bidding_ppo_loss,
    initialize_actor_from_checkpoint,
    load_bidding_ppo_checkpoint,
    masked_selected_log_probability,
    train_bidding_ppo,
)
from napoleon_ml.cli.export_policy_onnx import main as export_policy_onnx_main
from napoleon_ml.cli.train_bidding_ppo import main as train_bidding_ppo_main
from napoleon_ml.dataset.constants import BIDDING_ACTION_COUNT
from napoleon_ml.dataset.tensors import BIDDING_MODEL_INPUT_FEATURE_COUNT
from napoleon_ml.dataset.validation import calculate_card_ids_sha256
from napoleon_ml.nonplaying_onnx_export import export_bidding_rl_checkpoint_to_onnx


def test_masked_selected_log_probability_handles_forced_and_illegal_actions() -> None:
    logits = torch.tensor([[5.0, -2.0, 1.0], [0.0, 2.0, 4.0]], dtype=torch.float32)
    logits = torch.nn.functional.pad(logits, (0, BIDDING_ACTION_COUNT - 3))
    legal_mask = torch.zeros((2, BIDDING_ACTION_COUNT), dtype=torch.bool)
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


def test_bidding_ppo_loss_clips_ratio_and_accepts_arbitrary_float_reward() -> None:
    logits = torch.zeros((2, BIDDING_ACTION_COUNT), dtype=torch.float32)
    logits[0, 1] = math.log(0.9 / 0.1)
    logits[1, 1] = math.log(0.9 / 0.1)
    legal_mask = torch.zeros((2, BIDDING_ACTION_COUNT), dtype=torch.bool)
    legal_mask[:, 0] = True
    legal_mask[:, 1] = True

    loss = bidding_ppo_loss(
        logits=logits,
        value_prediction=torch.tensor([1.0, -2.0], dtype=torch.float32),
        selected_action_index=torch.tensor([1, 0], dtype=torch.int64),
        legal_bid_mask=legal_mask,
        behavior_log_probability=torch.log(torch.tensor([0.5, 0.5], dtype=torch.float32)),
        terminal_reward=torch.tensor([18.0, -3.0], dtype=torch.float32),
        clip_epsilon=0.2,
        value_loss_coefficient=0.5,
    )

    assert loss.probability_ratio.tolist() == pytest.approx([1.8, 0.2])
    assert loss.clipped_sample_mask.tolist() == [True, True]
    assert loss.value_loss.item() == pytest.approx(((17.0**2) + ((-1.0) ** 2)) / 2)
    assert torch.isfinite(loss.total_loss)


def test_bidding_ppo_train_checkpoint_and_export_smoke(tmp_path: Path) -> None:
    dataset = _write_rl_dataset(tmp_path / "dataset")
    parent = tmp_path / "parent.pt"
    config = BiddingMlpConfig(hidden_dim=8, hidden_layers=1, dropout=0.0)
    parent_model = BiddingMlpModel(config)
    torch.save(
        {
            "model_state": parent_model.state_dict(),
            "model_config": config.to_dict(),
            "action_count": BIDDING_ACTION_COUNT,
        },
        parent,
    )
    output = tmp_path / "bidding-ppo.pt"
    report = train_bidding_ppo(
        dataset_directory=dataset,
        output_checkpoint_path=output,
        settings=BiddingPpoTrainSettings(
            seed=123,
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            parent_actor_checkpoint=str(parent),
        ),
        model_config=config,
    )

    assert report.sample_count == 4
    model, checkpoint = load_bidding_ppo_checkpoint(output)
    assert isinstance(model, BiddingActorCriticModel)
    assert checkpoint["algorithm"] == BIDDING_PPO_ALGORITHM
    assert checkpoint["model_architecture"] == BIDDING_ACTOR_CRITIC_MODEL_ARCHITECTURE
    reward = checkpoint["reward"]
    fixed_playing_policy = checkpoint["fixed_playing_policy"]
    assert isinstance(reward, dict)
    assert isinstance(fixed_playing_policy, dict)
    assert reward["id"] == NON_PLAYING_RL_REWARD_ID
    assert checkpoint["parent_actor_checkpoint_sha256"] == _sha256(parent)
    assert fixed_playing_policy["onnxSha256"] == "0" * 64

    onnx = tmp_path / "bidding.onnx"
    metadata = tmp_path / "bidding.json"
    export_report = export_bidding_rl_checkpoint_to_onnx(
        dataset_directory=dataset,
        checkpoint_path=output,
        onnx_path=onnx,
        metadata_path=metadata,
    )
    metadata_json = json.loads(metadata.read_text(encoding="utf-8"))
    assert export_report.pytorch_selection == export_report.onnx_selection
    assert metadata_json["artifactType"] == "napoleon-bidding-policy-onnx"
    assert metadata_json["policyType"] == "bidding"
    assert metadata_json["outputLogitCount"] == BIDDING_ACTION_COUNT


def test_bidding_ppo_train_and_export_cli_smoke(tmp_path: Path) -> None:
    dataset = _write_rl_dataset(tmp_path / "dataset")
    checkpoint = tmp_path / "cli-bidding-ppo.pt"
    onnx = tmp_path / "cli-bidding.onnx"
    metadata = tmp_path / "cli-bidding.json"

    assert train_bidding_ppo_main(
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
            "bidding",
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
    assert metadata_json["policyType"] == "bidding"
    assert metadata_json["checkpointCompatibilityMetadata"]["sampleType"] == (
        "non-playing-bidding-rl-sample"
    )


def test_initialize_actor_from_actor_critic_checkpoint_leaves_critic_independent(
    tmp_path: Path,
) -> None:
    config = BiddingMlpConfig(hidden_dim=8, hidden_layers=1, dropout=0.0)
    source = BiddingActorCriticModel(config)
    target = BiddingActorCriticModel(config)
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
            "model_architecture": BIDDING_ACTOR_CRITIC_MODEL_ARCHITECTURE,
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


def _write_rl_dataset(directory: Path) -> Path:
    directory.mkdir(parents=True)
    samples = [_sample(seed=7, step=index + 1, selected=index % 2) for index in range(4)]
    shard = "".join(json.dumps(sample, separators=(",", ":")) + "\n" for sample in samples)
    (directory / "shard-00000.jsonl").write_text(shard, encoding="utf-8")
    manifest = {
        "datasetSchemaVersion": 1,
        "generatorVersion": 1,
        "format": "jsonl",
        "sampleType": "non-playing-bidding-rl-sample",
        "sampleSchemaVersion": 1,
        "phaseScope": "bidding-only",
        "learnedPhases": ["bidding"],
        "ruleBasedPhases": ["choosing-adjutant", "exchanging"],
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
        "biddingEncoderSchemaVersion": 1,
        "biddingModelInputSchemaVersion": 1,
        "biddingModelInputFeatureCount": BIDDING_MODEL_INPUT_FEATURE_COUNT,
        "playingModelInputSchemaVersion": 2,
        "playingModelInputFeatureCount": 6246,
        "actionCount": BIDDING_ACTION_COUNT,
        "behaviorPolicy": _policy("bidding-onnx"),
        "fixedPlayingPolicy": _policy("playing-onnx"),
        "samplingAlgorithm": "masked-categorical",
        "temperature": 1.0,
        "reward": {
            "type": "non-playing-terminal-role-reward",
            "version": 3,
            "id": NON_PLAYING_RL_REWARD_ID,
        },
        "allPassRule": {
            "id": NON_PLAYING_RL_ALL_PASS_RULE_ID,
            "starterPayoff": 1,
            "otherPayoff": -1,
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


def _sample(*, seed: int, step: int, selected: int) -> dict[str, object]:
    legal = [0] * BIDDING_ACTION_COUNT
    legal[0] = 1
    legal[1] = 1
    return {
        "sampleType": "non-playing-bidding-rl-sample",
        "schemaVersion": 1,
        "seed": seed,
        "step": step,
        "phase": "bidding",
        "actingPlayerId": "player-0",
        "actingPlayerIndex": 0,
        "relativePlayerIds": ["player-0", "player-1", "player-2", "player-3", "player-4"],
        "modelInput": [0.0] * BIDDING_MODEL_INPUT_FEATURE_COUNT,
        "legalBidMask": legal,
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
