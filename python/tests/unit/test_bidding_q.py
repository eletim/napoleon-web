from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from napoleon_ml.bidding_q import (
    BIDDING_Q_ACTION_MAPPING_ID,
    BIDDING_Q_DATASET_SCHEMA_VERSION,
    BIDDING_Q_MLP_ARCHITECTURE_ID,
    BIDDING_Q_REWARD_ID,
    BiddingQModel,
    BiddingQModelConfig,
    BiddingQTrainConfig,
    bidding_q_selected_action_loss,
    coverage_diagnostics,
    create_bidding_q_split,
    evaluate_bidding_q_model,
    load_bidding_q_checkpoint,
    load_bidding_q_dataset,
    ranking_metrics,
    save_bidding_q_artifact,
    train_bidding_q_model,
)
from napoleon_ml.bidding_q.dataset import BIDDING_Q_SUITS, BiddingQDatasetError
from napoleon_ml.bidding_q.training import BiddingQTargetNormalization
from napoleon_ml.dataset.constants import BIDDING_ACTION_COUNT
from napoleon_ml.dataset.tensors import BIDDING_MODEL_INPUT_FEATURE_COUNT
from napoleon_ml.dataset.validation import calculate_card_ids_sha256
from napoleon_ml.policy.device import resolve_torch_device


def test_bidding_q_dataset_loader_split_and_repeat_aggregation(tmp_path: Path) -> None:
    _write_q_dataset(tmp_path, repeats=2, state_count=6)

    dataset = load_bidding_q_dataset(tmp_path)
    assert dataset.manifest.dataset_schema_version in (1, BIDDING_Q_DATASET_SCHEMA_VERSION)
    assert dataset.manifest.reward_id == BIDDING_Q_REWARD_ID
    assert dataset.manifest.action_mapping_id == BIDDING_Q_ACTION_MAPPING_ID
    assert len(dataset.raw_samples) == 6 * 3 * 2
    assert len(dataset.examples) == 6 * 3
    assert all(example.repeat_count == 2 for example in dataset.examples)

    first = next(
        example
        for example in dataset.examples
        if example.state_key == "state-0" and example.action_index == 1
    )
    assert first.target_mean == pytest.approx(3.75)
    assert first.target_variance == pytest.approx(0.25)

    split = create_bidding_q_split(dataset.examples, train_ratio=0.5, seed=123)
    assert split.train_state_keys.isdisjoint(split.validation_state_keys)
    assert len(split.train_state_keys) == 3
    assert {example.state_key for example in split.train_examples} <= split.train_state_keys
    assert (
        {example.state_key for example in split.validation_examples}
        <= split.validation_state_keys
    )


def test_bidding_q_dataset_rejects_bad_schema_and_illegal_forced_action(tmp_path: Path) -> None:
    _write_q_dataset(tmp_path, repeats=1, state_count=2)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["reward"]["id"] = "wrong"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BiddingQDatasetError, match="reward"):
        load_bidding_q_dataset(tmp_path, verify_integrity=False)

    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    _write_q_dataset(bad_dir, repeats=1, state_count=2, illegal_action=True)
    with pytest.raises(BiddingQDatasetError, match="forced action is illegal"):
        load_bidding_q_dataset(bad_dir, verify_integrity=False)


def test_bidding_q_model_shape_and_selected_action_only_loss() -> None:
    model = BiddingQModel(BiddingQModelConfig(hidden_dims=(8,)))
    q_values = model(torch.zeros((4, BIDDING_MODEL_INPUT_FEATURE_COUNT), dtype=torch.float32))
    assert q_values.shape == (4, BIDDING_ACTION_COUNT)
    assert model.config.to_dict()["architectureId"] == BIDDING_Q_MLP_ARCHITECTURE_ID

    selected_q = torch.zeros((2, BIDDING_ACTION_COUNT), dtype=torch.float32, requires_grad=True)
    selected_q.data[0, 1] = 2.0
    selected_q.data[0, 2] = 100.0
    selected_q.data[1, 0] = -1.0
    loss = bidding_q_selected_action_loss(
        selected_q,
        torch.tensor([1, 0], dtype=torch.long),
        torch.tensor([4.0, 1.0], dtype=torch.float32),
        loss_type="huber",
    )
    assert loss.item() == pytest.approx(1.5)
    loss.backward()  # type: ignore[no-untyped-call]
    assert selected_q.grad is not None
    nonzero = selected_q.grad.nonzero().tolist()
    assert nonzero == [[0, 1], [1, 0]]


def test_bidding_q_ranking_and_coverage_metrics(tmp_path: Path) -> None:
    _write_q_dataset(tmp_path, repeats=1, state_count=4)
    dataset = load_bidding_q_dataset(tmp_path)
    examples = dataset.examples
    q_values = np.zeros((len(examples), BIDDING_ACTION_COUNT), dtype=np.float32)
    for row, example in enumerate(examples):
        q_values[row, example.action_index] = example.target_mean

    ranking = ranking_metrics(examples, q_values)
    assert ranking["bestActionHitRate"] == pytest.approx(1.0)
    assert ranking["top3HitRate"] == pytest.approx(1.0)
    assert ranking["pairwiseRankingAccuracy"] == pytest.approx(1.0)
    assert ranking["passVsBestBidAccuracy"] == pytest.approx(1.0)

    coverage = coverage_diagnostics(examples)
    assert coverage["passCount"] == 4
    assert coverage["bidCount"] == 8
    suit_counts = cast(dict[str, int], coverage["suitCounts"])
    target_counts = cast(dict[str, int], coverage["targetCounts"])
    action_counts = cast(dict[str, int], coverage["actionIndexCounts"])
    assert suit_counts["spades"] > 0
    assert target_counts["13"] > 0
    assert sum(action_counts.values()) == len(examples)


def test_bidding_q_training_checkpoint_and_onnx_parity(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    _write_q_dataset(tmp_path / "dataset", repeats=2, state_count=12)
    dataset = load_bidding_q_dataset(tmp_path / "dataset")
    config = BiddingQTrainConfig(
        seed=370,
        epochs=4,
        batch_size=8,
        learning_rate=5e-3,
        hidden_dims=(16, 8),
        train_ratio=0.75,
        device="cpu",
    )
    first = train_bidding_q_model(dataset, config)
    second = train_bidding_q_model(dataset, config)
    assert first.epoch_reports == second.epoch_reports
    assert cast(float, first.epoch_reports[-1]["trainLoss"]) <= cast(
        float,
        first.epoch_reports[0]["trainLoss"],
    )

    artifact = save_bidding_q_artifact(tmp_path / "artifact", result=first, dataset=dataset)
    assert Path(str(artifact["checkpointPath"])).is_file()
    assert Path(str(artifact["metadataPath"])).is_file()
    assert Path(str(artifact["onnxPath"])).is_file()
    parity = cast(dict[str, object], artifact["onnxParity"])
    assert parity["withinTolerance"] is True
    assert cast(float, parity["maxAbsQDiff"]) <= 1e-5

    loaded_model, checkpoint = load_bidding_q_checkpoint(Path(str(artifact["checkpointPath"])))
    assert checkpoint["modelType"] == "bidding-q"
    loaded_q = loaded_model(
        torch.zeros((1, BIDDING_MODEL_INPUT_FEATURE_COUNT), dtype=torch.float32)
    )
    assert loaded_q.shape == (1, BIDDING_ACTION_COUNT)

    metadata = json.loads(Path(str(artifact["metadataPath"])).read_text(encoding="utf-8"))
    assert metadata["modelType"] == "bidding-q"
    assert metadata["selectionSemantics"] == "legal-q-argmax"
    assert metadata["dataset"]["manifestSha256"] == dataset.manifest.manifest_sha256
    assert metadata["reward"]["id"] == BIDDING_Q_REWARD_ID


def test_bidding_q_evaluation_compares_baselines(tmp_path: Path) -> None:
    _write_q_dataset(tmp_path, repeats=1, state_count=10)
    dataset = load_bidding_q_dataset(tmp_path)
    split = create_bidding_q_split(dataset.examples, train_ratio=0.8, seed=7)
    model = BiddingQModel(BiddingQModelConfig(hidden_dims=(8,)))
    report = evaluate_bidding_q_model(
        model,
        split.validation_examples,
        train_examples=split.train_examples,
        normalization=BiddingQTargetNormalization(kind="none", mean=0.0, std=1.0),
        loss_type="huber",
        device=resolve_torch_device("cpu"),
    )
    regression = cast(dict[str, object], report["regression"])
    baselines = cast(dict[str, dict[str, object]], report["baselines"])
    collapse = cast(dict[str, object], report["qCollapseDiagnostics"])
    assert regression["mae"] is not None
    assert baselines["globalMean"]["mae"] is not None
    assert baselines["actionIndexMean"]["mae"] is not None
    assert collapse["predictedPassRate"] is not None


def _write_q_dataset(
    directory: Path,
    *,
    repeats: int,
    state_count: int,
    illegal_action: bool = False,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    for state_index in range(state_count):
        for action_index in (0, 1 + (state_index % 4), 5 + (state_index % 4)):
            for repeat_index in range(repeats):
                reward = _reward_for(
                    state_index=state_index,
                    action_index=action_index,
                    repeat_index=repeat_index,
                )
                legal_mask = [1] * BIDDING_ACTION_COUNT
                if illegal_action and state_index == 0 and action_index != 0:
                    legal_mask[action_index] = 0
                samples.append(
                    _sample(
                        state_index=state_index,
                        action_index=action_index,
                        repeat_index=repeat_index,
                        reward=reward,
                        legal_mask=legal_mask,
                    )
                )
    shard_bytes = b"".join(
        (json.dumps(sample, sort_keys=True) + "\n").encode("utf-8")
        for sample in samples
    )
    (directory / "shard-00000.jsonl").write_bytes(shard_bytes)
    (directory / "summary.json").write_text("{}\n", encoding="utf-8")
    manifest = {
        "datasetSchemaVersion": 1,
        "generatorVersion": 1,
        "format": "jsonl",
        "sampleType": "bidding-q-monte-carlo-counterfactual-sample",
        "sampleSchemaVersion": 1,
        "compactObservation": {
            "phase": "bidding",
            "encoderSchemaVersion": 1,
            "modelInputSchemaVersion": 2,
            "modelInputFeatureCount": BIDDING_MODEL_INPUT_FEATURE_COUNT,
        },
        "actionMapping": {
            "id": BIDDING_Q_ACTION_MAPPING_ID,
            "actionCount": BIDDING_ACTION_COUNT,
            "passActionIndex": 0,
            "bidTargets": [13, 14, 15, 16, 17, 18, 19],
            "suitOrder": list(BIDDING_Q_SUITS),
        },
        "reward": {
            "id": BIDDING_Q_REWARD_ID,
            "type": "bidding-q-contract-result-terminal-reward",
            "version": 1,
            "contractLossReward": -1,
            "nonContractReward": 0,
        },
        "terminalRewardTransform": {"type": "identity"},
        "actionPlan": {"id": "fixture", "version": 1, "randomLegalBidCount": 0},
        "repeats": repeats,
        "sourceStates": state_count,
        "forcedStateActionPairs": state_count * 3,
        "sampleCount": len(samples),
        "startSeed": 1000,
        "endSeed": 1000 + state_count - 1,
        "logicalSeedCount": state_count,
        "actualSourceGameCount": state_count,
        "candidateSeatRotation": [0, 1, 2, 3, 4],
        "gamesPerShard": len(samples),
        "shardCount": 1,
        "playerCount": 5,
        "cardCount": 53,
        "cardIdsSha256": calculate_card_ids_sha256(),
        "simulation": {"backend": "typescript", "inferenceDevice": "cpu"},
        "opponentMix": {},
        "behaviorPolicy": {},
        "fixedPlayingPolicy": {},
        "sourceCommit": "fixture",
        "summary": {},
        "shards": [
            {
                "file": "shard-00000.jsonl",
                "startSeed": 1000,
                "endSeed": 1000 + state_count - 1,
                "gameCount": state_count,
                "sampleCount": len(samples),
                "byteLength": len(shard_bytes),
                "sha256": hashlib.sha256(shard_bytes).hexdigest(),
            }
        ],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def _sample(
    *,
    state_index: int,
    action_index: int,
    repeat_index: int,
    reward: float,
    legal_mask: list[int],
) -> dict[str, Any]:
    model_input = [0.0] * BIDDING_MODEL_INPUT_FEATURE_COUNT
    model_input[state_index % BIDDING_MODEL_INPUT_FEATURE_COUNT] = 1.0
    forced_action = _forced_action(action_index)
    return {
        "sampleType": "bidding-q-monte-carlo-counterfactual-sample",
        "schemaVersion": 1,
        "stateKey": f"state-{state_index}",
        "sourceSeed": 1000 + state_index,
        "sourceGameSeed": 1000 + state_index,
        "candidateSeatIndex": state_index % 5,
        "actingPlayerId": f"player-{state_index % 5}",
        "actingPlayerIndex": state_index % 5,
        "biddingStep": 1,
        "sourceSelectedActionIndex": 0,
        "sourceSelectedAction": {"type": "pass"},
        "modelInput": model_input,
        "legalBidMask": legal_mask,
        "forcedActionIndex": action_index,
        "forcedAction": forced_action,
        "strongestSuit": BIDDING_Q_SUITS[state_index % 4],
        "strongestSuitScore": 200 + state_index,
        "actionPlanId": "fixture",
        "repeatIndex": repeat_index,
        "rolloutSeed": 9000 + state_index * 100 + action_index * 10 + repeat_index,
        "terminalReward": reward,
        "rawTerminalReward": reward,
        "terminalRole": "napoleon" if action_index != 0 else "all-pass-starter",
        "contractSuccess": action_index != 0 and reward > 0,
        "resultType": "standard" if action_index != 0 else "all-pass",
        "result": {},
        "provenance": {
            "sourceStateKey": f"state-{state_index}",
            "sourceSeed": 1000 + state_index,
            "sourceGameSeed": 1000 + state_index,
            "sourceBiddingStep": 1,
            "replayMatchedModelInput": True,
            "replayMatchedLegalBidMask": True,
            "forcedOnce": True,
        },
    }


def _forced_action(action_index: int) -> dict[str, Any]:
    if action_index == 0:
        return {"type": "pass"}
    offset = action_index - 1
    return {
        "type": "bid",
        "targetPointCards": 13 + offset // 4,
        "suit": BIDDING_Q_SUITS[offset % 4],
    }


def _reward_for(*, state_index: int, action_index: int, repeat_index: int) -> float:
    if action_index == 0:
        return float(state_index % 2)
    return float(3 + state_index + action_index * 0.25 + repeat_index)
