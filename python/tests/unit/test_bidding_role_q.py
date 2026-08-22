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
    BIDDING_Q_REWARD_ID,
    BIDDING_ROLE_CLASSES,
    BIDDING_ROLE_COUNT,
    BIDDING_ROLE_Q_ARCHITECTURE_ID,
    BIDDING_ROLE_Q_MODEL_TYPE,
    BiddingRoleQModel,
    BiddingRoleQModelConfig,
    BiddingRoleQTrainConfig,
    bidding_role_q_losses,
    canonical_bidding_role,
    compose_role_q,
    create_bidding_q_split,
    load_bidding_q_dataset,
    load_bidding_role_q_checkpoint,
    role_dataset_diagnostics,
    role_index_for_terminal_role,
    save_bidding_role_q_artifact,
    train_bidding_role_q_model,
)
from napoleon_ml.bidding_q.dataset import BIDDING_Q_SUITS
from napoleon_ml.dataset.constants import BIDDING_ACTION_COUNT
from napoleon_ml.dataset.tensors import BIDDING_MODEL_INPUT_FEATURE_COUNT
from napoleon_ml.dataset.validation import calculate_card_ids_sha256


def test_role_label_extraction_and_exclusivity() -> None:
    assert canonical_bidding_role("napoleon") == "napoleon"
    assert canonical_bidding_role("napoleon-adjutant") == "napoleon-adjutant"
    assert canonical_bidding_role("adjutant") == "adjutant"
    assert canonical_bidding_role("citizen") == "citizen"
    assert canonical_bidding_role("all-pass-starter") == "no-contract"
    assert canonical_bidding_role("all-pass-other") == "no-contract"
    assert len(set(BIDDING_ROLE_CLASSES)) == BIDDING_ROLE_COUNT
    decoded_roles = [
        BIDDING_ROLE_CLASSES[role_index_for_terminal_role(role)] for role in BIDDING_ROLE_CLASSES
    ]
    assert decoded_roles == list(BIDDING_ROLE_CLASSES)
    with pytest.raises(ValueError, match="unsupported terminalRole"):
        canonical_bidding_role("unknown")


def test_role_model_shape_and_no_final_role_input() -> None:
    model = BiddingRoleQModel(BiddingRoleQModelConfig(hidden_dims=(8,)))
    first_input = torch.zeros((1, BIDDING_MODEL_INPUT_FEATURE_COUNT), dtype=torch.float32)
    second_input = first_input.clone()
    first_logits, first_values = model(first_input)
    second_logits, second_values = model(second_input)
    assert first_logits.shape == (1, BIDDING_ACTION_COUNT, BIDDING_ROLE_COUNT)
    assert first_values.shape == (1, BIDDING_ACTION_COUNT, BIDDING_ROLE_COUNT)
    assert torch.equal(first_logits, second_logits)
    assert torch.equal(first_values, second_values)
    assert model.config.to_dict()["architectureId"] == BIDDING_ROLE_Q_ARCHITECTURE_ID
    assert "terminalRole" not in json.dumps(model.config.to_dict())


def test_role_losses_are_selected_action_and_selected_cell_only() -> None:
    role_logits = torch.zeros(
        (2, BIDDING_ACTION_COUNT, BIDDING_ROLE_COUNT),
        dtype=torch.float32,
        requires_grad=True,
    )
    role_values = torch.zeros(
        (2, BIDDING_ACTION_COUNT, BIDDING_ROLE_COUNT),
        dtype=torch.float32,
        requires_grad=True,
    )
    action_index = torch.tensor([1, 0], dtype=torch.long)
    role_index = torch.tensor([0, 4], dtype=torch.long)
    target = torch.tensor([2.0, -1.0], dtype=torch.float32)
    role_loss, value_loss = bidding_role_q_losses(
        role_logits,
        role_values,
        action_index,
        role_index,
        target,
        value_loss_type="huber",
    )
    assert role_loss.item() == pytest.approx(np.log(BIDDING_ROLE_COUNT))
    assert value_loss.item() == pytest.approx(1.0)
    role_loss.backward(retain_graph=True)  # type: ignore[no-untyped-call]
    assert role_logits.grad is not None
    assert {tuple(index)[:2] for index in role_logits.grad.nonzero().tolist()} == {(0, 1), (1, 0)}
    value_loss.backward()  # type: ignore[no-untyped-call]
    assert role_values.grad is not None
    assert role_values.grad.nonzero().tolist() == [[0, 1, 0], [1, 0, 4]]

    role_values_masked = torch.zeros(
        (2, BIDDING_ACTION_COUNT, BIDDING_ROLE_COUNT),
        dtype=torch.float32,
        requires_grad=True,
    )
    _role_loss, masked_value_loss = bidding_role_q_losses(
        role_logits.detach(),
        role_values_masked,
        action_index,
        role_index,
        torch.tensor([2.0, 100.0], dtype=torch.float32),
        value_loss_type="huber",
        value_mask=torch.tensor([True, False], dtype=torch.bool),
    )
    assert masked_value_loss.item() == pytest.approx(1.5)
    masked_value_loss.backward()  # type: ignore[no-untyped-call]
    assert role_values_masked.grad is not None
    assert role_values_masked.grad.nonzero().tolist() == [[0, 1, 0]]


def test_role_q_composition() -> None:
    logits = torch.zeros((1, BIDDING_ACTION_COUNT, BIDDING_ROLE_COUNT), dtype=torch.float32)
    values = torch.zeros((1, BIDDING_ACTION_COUNT, BIDDING_ROLE_COUNT), dtype=torch.float32)
    logits[0, 0, 0] = 6.0
    logits[0, 0, 1] = -6.0
    values[0, 0, 0] = 10.0
    values[0, 0, 1] = -10.0
    values[0, 1, 0] = 2.0
    values[0, 1, 1] = 6.0
    q_values = compose_role_q(logits, values)
    assert q_values.shape == (1, BIDDING_ACTION_COUNT)
    assert q_values[0, 0].item() == pytest.approx(9.926, abs=1e-3)
    assert q_values[0, 1].item() == pytest.approx(1.6)


def test_role_q_split_diagnostics_and_training(tmp_path: Path) -> None:
    _write_role_q_dataset(tmp_path / "dataset", repeats=1, state_count=12)
    dataset = load_bidding_q_dataset(tmp_path / "dataset")
    split = create_bidding_q_split(dataset.examples, train_ratio=0.75, seed=373)
    assert split.train_state_keys.isdisjoint(split.validation_state_keys)

    diagnostics = role_dataset_diagnostics(dataset.raw_samples)
    role_distribution = cast(dict[str, object], diagnostics["roleDistribution"])
    assert role_distribution["total"] == len(dataset.raw_samples)
    pass_distribution = cast(dict[str, object], diagnostics["passRoleDistribution"])
    assert pass_distribution["total"] == 12
    suit_distribution = cast(dict[str, object], diagnostics["lowestLegalSuitRoleDistribution"])
    spades_distribution = cast(dict[str, object], suit_distribution["spades"])
    assert cast(int, spades_distribution["total"]) > 0

    config = BiddingRoleQTrainConfig(
        seed=373,
        epochs=3,
        batch_size=8,
        learning_rate=5e-3,
        hidden_dims=(16, 8),
        train_ratio=0.75,
        device="cpu",
    )
    first = train_bidding_role_q_model(dataset, config)
    second = train_bidding_role_q_model(dataset, config)
    assert first.epoch_reports == second.epoch_reports
    assert cast(float, first.epoch_reports[-1]["trainLoss"]) <= cast(
        float,
        first.epoch_reports[0]["trainLoss"],
    )
    validation = first.validation_report
    classifier = cast(dict[str, object], validation["roleClassifier"])
    baselines = cast(dict[str, dict[str, object]], validation["baselines"])
    ranking = cast(dict[str, object], validation["ranking"])
    assert classifier["accuracy"] is not None
    assert baselines["majorityRole"]["accuracy"] is not None
    assert baselines["actionIndexRoleFrequency"]["accuracy"] is not None
    assert ranking["bestActionHitRate"] is not None

    team_config = BiddingRoleQTrainConfig(
        seed=375,
        epochs=2,
        batch_size=8,
        learning_rate=5e-3,
        hidden_dims=(16, 8),
        train_ratio=0.75,
        value_target="team-point-cards",
        device="cpu",
    )
    team_result = train_bidding_role_q_model(dataset, team_config)
    team_validation = team_result.validation_report
    team_value = cast(dict[str, object], team_validation["roleValue"])
    assert team_value["valueTarget"] == "team-point-cards"
    assert cast(int, team_value["maskedSampleCount"]) > 0
    assert cast(dict[str, object], team_validation["ranking"])["skipped"] is True
    team_signal = cast(dict[str, object], team_validation["teamPointCardsActionSignal"])
    assert team_signal["bestActionHitRate"] is not None
    value_baselines = cast(dict[str, object], team_validation["baselines"])["value"]
    assert cast(dict[str, object], value_baselines)["roleActionIndexMean"] is not None


def test_role_q_checkpoint_and_onnx_parity(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    _write_role_q_dataset(tmp_path / "dataset", repeats=1, state_count=12)
    dataset = load_bidding_q_dataset(tmp_path / "dataset")
    config = BiddingRoleQTrainConfig(
        seed=373,
        epochs=2,
        batch_size=8,
        learning_rate=5e-3,
        hidden_dims=(16, 8),
        train_ratio=0.75,
        device="cpu",
    )
    result = train_bidding_role_q_model(dataset, config)
    artifact = save_bidding_role_q_artifact(tmp_path / "artifact", result=result, dataset=dataset)
    parity = cast(dict[str, object], artifact["onnxParity"])
    assert parity["withinTolerance"] is True
    assert cast(float, parity["maxAbsDiff"]) <= 1e-5

    loaded_model, checkpoint = load_bidding_role_q_checkpoint(Path(str(artifact["checkpointPath"])))
    assert checkpoint["modelType"] == BIDDING_ROLE_Q_MODEL_TYPE
    logits, values = loaded_model(
        torch.zeros((1, BIDDING_MODEL_INPUT_FEATURE_COUNT), dtype=torch.float32)
    )
    assert logits.shape == (1, BIDDING_ACTION_COUNT, BIDDING_ROLE_COUNT)
    assert values.shape == (1, BIDDING_ACTION_COUNT, BIDDING_ROLE_COUNT)

    metadata = json.loads(Path(str(artifact["metadataPath"])).read_text(encoding="utf-8"))
    assert metadata["modelType"] == BIDDING_ROLE_Q_MODEL_TYPE
    assert metadata["selectionSemantics"] == "legal-role-decomposed-q-argmax"
    assert metadata["dataset"]["manifestSha256"] == dataset.manifest.manifest_sha256
    assert metadata["reward"]["id"] == BIDDING_Q_REWARD_ID


def _write_role_q_dataset(
    directory: Path,
    *,
    repeats: int,
    state_count: int,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    for state_index in range(state_count):
        for action_index in (0, 1 + (state_index % 4), 5 + (state_index % 4)):
            for repeat_index in range(repeats):
                reward, role = _reward_and_role_for(
                    state_index=state_index,
                    action_index=action_index,
                )
                samples.append(
                    _sample(
                        state_index=state_index,
                        action_index=action_index,
                        repeat_index=repeat_index,
                        reward=reward,
                        role=role,
                    )
                )
    shard_bytes = b"".join(
        (json.dumps(sample, sort_keys=True) + "\n").encode("utf-8") for sample in samples
    )
    (directory / "shard-00000.jsonl").write_bytes(shard_bytes)
    (directory / "summary.json").write_text("{}\n", encoding="utf-8")
    manifest = {
        "datasetSchemaVersion": 2,
        "generatorVersion": 2,
        "format": "jsonl",
        "sampleType": "bidding-q-monte-carlo-counterfactual-sample",
        "sampleSchemaVersion": 2,
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
        "predictionTarget": {
            "id": "bidding-q-final-role-team-point-cards-v1",
            "version": 1,
            "roleLabel": "finalRole",
            "valueLabel": "candidate-team-point-card-count",
            "noContractHandling": "masked-null",
            "candidateTeamDefinition": {
                "napoleon": "napoleon-side",
                "napoleon-adjutant": "napoleon-side",
                "adjutant": "napoleon-side",
                "citizen": "coalition-side",
                "noContract": "masked",
            },
        },
        "opponentMix": {
            "id": "rule-based-conservative-50-50-per-seat-v1",
            "topology": "candidate-x1-frozen-x4",
            "selectionScope": "frozen-seat",
            "selectionDeterminism": "seeded",
            "mixingRuleVersion": "per-seat-seeded-rule-based-conservative-50-50-v1",
            "selectionSeedSource": "sourceSeed,candidateSeatIndex,playerIndex",
            "frozenSeatPolicies": [
                {"id": "rule-based-bidding-v1", "type": "rule-based", "weight": 0.5},
                {"id": "conservative-bidding-v1", "type": "conservative-bidding", "weight": 0.5},
            ],
        },
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
    role: str,
) -> dict[str, Any]:
    model_input = [0.0] * BIDDING_MODEL_INPUT_FEATURE_COUNT
    model_input[state_index % BIDDING_MODEL_INPUT_FEATURE_COUNT] = 1.0
    forced_action = _forced_action(action_index)
    return {
        "sampleType": "bidding-q-monte-carlo-counterfactual-sample",
        "schemaVersion": 2,
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
        "legalBidMask": [1] * BIDDING_ACTION_COUNT,
        "forcedActionIndex": action_index,
        "forcedAction": forced_action,
        "strongestSuit": BIDDING_Q_SUITS[state_index % 4],
        "strongestSuitScore": 200 + state_index,
        "actionPlanId": "fixture",
        "repeatIndex": repeat_index,
        "rolloutSeed": 9000 + state_index * 100 + action_index * 10 + repeat_index,
        "terminalReward": reward,
        "rawTerminalReward": reward,
        "terminalRole": role,
        **_team_point_card_fields(role=role, forced_action=forced_action),
        "contractSuccess": role in ("napoleon", "napoleon-adjutant") and reward > 0,
        "resultType": "all-pass" if role.startswith("all-pass") else "standard",
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


def _team_point_card_fields(*, role: str, forced_action: dict[str, Any]) -> dict[str, Any]:
    if role.startswith("all-pass"):
        return {
            "finalRole": role,
            "candidateFinalTeam": "no-contract",
            "napoleonSidePointCards": None,
            "coalitionSidePointCards": None,
            "candidateTeamPointCards": None,
            "teamPointCardsRegressionMask": False,
            "finalDeclaredTarget": None,
            "finalDeclaredSuit": None,
            "contractMargin": None,
            "frozenBiddingOpponentCounts": {"ruleBased": 2, "conservative": 2},
            "opponentConfigurationKey": "ruleBased=2,conservative=2",
        }
    napoleon_cards = 16 if role in ("napoleon", "napoleon-adjutant", "adjutant") else 8
    coalition_cards = 20 - napoleon_cards
    candidate_team = "alliance" if role == "citizen" else "napoleon-team"
    candidate_cards = coalition_cards if candidate_team == "alliance" else napoleon_cards
    return {
        "finalRole": role,
        "candidateFinalTeam": candidate_team,
        "napoleonSidePointCards": napoleon_cards,
        "coalitionSidePointCards": coalition_cards,
        "candidateTeamPointCards": candidate_cards,
        "teamPointCardsRegressionMask": True,
        "finalDeclaredTarget": 13,
        "finalDeclaredSuit": forced_action.get("suit", "spades"),
        "contractMargin": napoleon_cards - 13 if candidate_team == "napoleon-team" else None,
        "frozenBiddingOpponentCounts": {"ruleBased": 2, "conservative": 2},
        "opponentConfigurationKey": "ruleBased=2,conservative=2",
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


def _reward_and_role_for(*, state_index: int, action_index: int) -> tuple[float, str]:
    if action_index == 0:
        return 0.0, "all-pass-starter" if state_index % 2 == 0 else "citizen"
    if state_index % 5 == 0:
        return 39.0, "napoleon-adjutant"
    if state_index % 3 == 0:
        return 26.0, "napoleon"
    if state_index % 3 == 1:
        return 0.0, "adjutant"
    return 0.0, "citizen"
