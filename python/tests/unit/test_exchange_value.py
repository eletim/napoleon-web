from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from napoleon_ml.dataset.constants import CARD_COUNT, EXPECTED_CARD_IDS
from napoleon_ml.dataset.tensors import EXCHANGE_MODEL_INPUT_FEATURE_COUNT
from napoleon_ml.exchange_value import (
    EXCHANGE_COMPACT_STATE_FEATURE_COUNT,
    EXCHANGE_COMPACT_VALUE_INPUT_FEATURE_COUNT,
    EXCHANGE_ORACLE_LOCATION_INPUT_FEATURE_COUNT,
    EXCHANGE_TACTICAL_VALUE_INPUT_FEATURE_COUNT,
    EXCHANGE_VALUE_INPUT_FEATURE_COUNT,
    ExchangeValueMlpConfig,
    ExchangeValueMlpModel,
    ExchangeValueTrainConfig,
    Issue442Layout,
    collect_issue442_layout_summary,
    combine_exchange_counterfactual_datasets,
    create_exchange_value_split,
    load_exchange_counterfactual_dataset,
    load_exchange_value_checkpoint,
    render_issue442_markdown,
    save_exchange_value_artifact,
    train_exchange_value_model,
)
from napoleon_ml.exchange_value.dataset import dataset_provenance, state_key_hash
from napoleon_ml.exchange_value.full_gold_audit import (
    ExchangeFullGoldAudit,
    audit_training_leakage_report,
    exchange_full_gold_report,
    exclude_audit_overlaps,
    load_exchange_full_gold_audit,
)
from napoleon_ml.exchange_value.oracle_location import (
    load_full_gold_location_overlay,
    location_one_hot,
    relative_adjutant_location_class,
)
from napoleon_ml.exchange_value.tactical import compact396_tactical_features
from napoleon_ml.exchange_value.training import exchange_value_evaluation_report


def _mask(indices: list[int], *, length: int = CARD_COUNT) -> list[int]:
    values = [0] * length
    for index in indices:
        values[index] = 1
    return values


def _compact_state(original: list[int], kitty: list[int], *, target_offset: int = 0) -> list[float]:
    table = [0.0] * (28 * 6)
    for position in range(28):
        table[position * 6] = 1.0
    values = [
        *_mask(original),
        *_mask(kitty),
        *_mask([0]),
        1.0,
        0.0,
        0.0,
        0.0,
        *([1.0 if index == target_offset else 0.0 for index in range(7)]),
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        *table,
    ]
    assert len(values) == EXCHANGE_COMPACT_STATE_FEATURE_COUNT
    return values


def _sample(*, state: int, candidate: int) -> dict[str, Any]:
    base = candidate % 13
    discard = sorted({base, (base + 1) % 13, (base + 2) % 13})
    while len(discard) < 3:
        discard.append(len(discard))
    margin = 10.0 - abs(candidate - 7)
    if candidate == 0:
        margin = 20.0
    buried_special = {
        "joker": 52 in discard,
        "oruma": 0 in discard,
        "yoromeki": 1 in discard,
        "seiJack": 2 in discard,
        "uraJack": 3 in discard,
        "calledAdjutant": 4 in discard,
    }
    original_indices = list(range(state, state + 10))
    kitty_indices = list(range(state + 10, state + 13))
    return {
        "sampleType": "exchange-counterfactual-value-v1",
        "schemaVersion": 1,
        "sourceStateKey": f"state-{state}",
        "fixedHandId": f"fixed-{state}",
        "fixedThirteenGroupId": f"group-{state // 2}",
        "dealSeed": 436000 + state,
        "sourceIndex": state,
        "repeatIndex": state % 2,
        "candidateIndex": candidate,
        "candidateKey": f"state-{state}-candidate-{candidate}",
        "napoleonPlayerId": "player-0",
        "napoleonSeatIndex": 0,
        "contractTargetPointCards": 13,
        "contractSuit": "spades",
        "calledAdjutantCardId": "spades-A",
        "originalHandCardIds": [EXPECTED_CARD_IDS[index] for index in original_indices],
        "kittyPickupCardIds": [EXPECTED_CARD_IDS[index] for index in kitty_indices],
        "opponentPolicyIds": [
            "frozen-raise-v1",
            "strong-rule-based-bidding-v1",
            "conservative-bidding-v1",
            "frozen-raise-v1",
        ],
        "pickupHandCardIds": [
            EXPECTED_CARD_IDS[index] for index in original_indices + kitty_indices
        ],
        "modelInput": [float(state % 3)] * EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
        "compactExchangeStateInput": _compact_state(original_indices, kitty_indices),
        "legalDiscardCardMask": _mask(list(range(13))),
        "candidateDiscardCardIds": [EXPECTED_CARD_IDS[index] for index in discard],
        "candidateDiscardMask": _mask(discard),
        "buriedPointCardCount": min(candidate % 4, 3),
        "buriedTrumpCount": candidate % 3,
        "buriedSpecialCards": buried_special,
        "contractMargin": margin,
        "contractSuccess": margin >= 0,
        "napoleonSidePointCards": 13 + margin,
        "napoleonRawReward": 26 if margin >= 0 else 0,
        "napoleonRelativeReward": margin * 2.0,
        "ruleBasedDiscardCardIds": [EXPECTED_CARD_IDS[index] for index in [1, 2, 3]],
        "isRuleBasedAction": candidate == 1,
        "ruleBasedCandidateIndex": 1,
        "hiddenDealChecksum": f"hidden-{state}",
        "biddingHistoryHash": f"history-{state}",
        "biddingHistoryActionCount": 5,
        "invariantChecks": {},
    }


def _write_dataset(directory: Path, *, states: int = 12) -> None:
    rows = [
        _sample(state=state, candidate=candidate)
        for state in range(states)
        for candidate in range(286)
    ]
    shard_bytes = b"".join(
        (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8") for row in rows
    )
    (directory / "shard-00000.jsonl").write_bytes(shard_bytes)
    manifest = {
        "datasetSchemaVersion": 1,
        "generatorVersion": 1,
        "format": "jsonl",
        "sampleType": "exchange-counterfactual-value-v1",
        "sampleSchemaVersion": 1,
        "teacherId": "exchange-discard-combination-playing-rollout-v1",
        "discardActionSpaceId": "exchange-unordered-13c3-discard-combinations-v1",
        "discardCombinationCount": 286,
        "sourceFlow": [],
        "modelInput": {
            "observation": "existing-exchange-observation",
            "featureCount": EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
            "hiddenOpponentHandsIncluded": False,
        },
        "compactExchangeValueInput": {
            "stateFeatureCount": EXCHANGE_COMPACT_STATE_FEATURE_COUNT,
            "candidateDiscardMaskFeatureCount": CARD_COUNT,
            "featureCount": EXCHANGE_COMPACT_VALUE_INPUT_FEATURE_COUNT,
            "layout": [],
            "biddingHistorySemantics": "compact278-bid-owner-table",
        },
        "teacherUsesCompleteHiddenState": True,
        "permutationActionsIncluded": False,
        "startSeed": 436000,
        "endSeed": 436000 + states - 1,
        "sourceStateCount": states,
        "requestedSourceStateCount": states,
        "sampleCount": len(rows),
        "rolloutCount": len(rows),
        "statesPerShard": states,
        "dealAttemptCount": states,
        "sourceCommit": "fixture",
        "playerCount": 5,
        "cardCount": CARD_COUNT,
        "cardIds": list(EXPECTED_CARD_IDS),
        "cardIdsSha256": "fixture",
        "biddingPolicy": {"id": "frozen-raise-v1"},
        "adjutantPolicy": {"id": "rule-based-adjutant-v1"},
        "playingPolicy": {"id": "ppo-separated-v1000"},
        "reward": {},
        "summary": {},
        "shardCount": 1,
        "shards": [
            {
                "file": "shard-00000.jsonl",
                "startSeed": 436000,
                "endSeed": 436000 + states - 1,
                "gameCount": states,
                "sampleCount": len(rows),
                "byteLength": len(shard_bytes),
                "sha256": hashlib.sha256(shard_bytes).hexdigest(),
            }
        ],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_oracle_overlay(path: Path, *, states: int) -> None:
    class_names = [
        "opponentSeat1",
        "opponentSeat2",
        "opponentSeat3",
        "opponentSeat4",
        "selfKittySolo",
    ]
    entries = {
        f"state-{state}": {
            "classIndex": state % 5,
            "className": class_names[state % 5],
            "calledCardOrigin": "originalHand" if state % 5 == 4 else "opponentHand",
            "dealSeed": 436000 + state,
            "napoleonSeatIndex": 0,
            "calledAdjutantCardId": "spades-A",
        }
        for state in range(states)
    }
    path.write_text(json.dumps({
        "artifactType": "issue450-exchange-training-location-overlay-v1",
        "classNames": class_names,
        "sourceStateCount": states,
        "datasetManifests": [],
        "entries": entries,
    }), encoding="utf-8")


def test_loader_roundtrip_and_input_dimensions(tmp_path: Path) -> None:
    _write_dataset(tmp_path, states=3)

    dataset = load_exchange_counterfactual_dataset(tmp_path)
    sample = dataset.raw_samples[0]

    assert dataset.source_state_count == 3
    assert dataset.sample_count == 858
    assert sample.model_input.shape == (EXCHANGE_MODEL_INPUT_FEATURE_COUNT,)
    assert sample.compact_exchange_state_input is not None
    assert sample.compact_exchange_state_input.shape == (EXCHANGE_COMPACT_STATE_FEATURE_COUNT,)
    assert sample.candidate_discard_mask.shape == (CARD_COUNT,)
    assert sample.value_input.shape == (EXCHANGE_VALUE_INPUT_FEATURE_COUNT,)
    assert sample.compact_value_input.shape == (EXCHANGE_COMPACT_VALUE_INPUT_FEATURE_COUNT,)
    assert int(sample.candidate_discard_mask.sum()) == 3


def test_compact_only_loader_skips_legacy_tensor(tmp_path: Path) -> None:
    _write_dataset(tmp_path, states=1)

    sample = load_exchange_counterfactual_dataset(
        tmp_path, load_legacy_model_input=False
    ).raw_samples[0]

    assert sample.model_input is None
    assert sample.compact_value_input.shape == (EXCHANGE_COMPACT_VALUE_INPUT_FEATURE_COUNT,)
    with pytest.raises(ValueError, match="legacy2724 input was not loaded"):
        _ = sample.value_input


def test_oracle_location_relative_seat_and_compact401_fixture(tmp_path: Path) -> None:
    classes = [
        relative_adjutant_location_class(owner, 3) for owner in (4, 0, 1, 2, 3, None)
    ]
    assert classes == [0, 1, 2, 3, 4, 4]
    assert location_one_hot(2).tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]
    _write_dataset(tmp_path, states=2)
    overlay = tmp_path / "oracle.json"
    _write_oracle_overlay(overlay, states=2)
    sample = load_exchange_counterfactual_dataset(
        tmp_path, load_legacy_model_input=False, oracle_location_overlay=overlay
    ).raw_samples[0]
    oracle_input = sample.value_input_for_variant("compact401-oracle-location")
    assert oracle_input.shape == (EXCHANGE_ORACLE_LOCATION_INPUT_FEATURE_COUNT,)
    np.testing.assert_array_equal(oracle_input[:396], sample.compact_value_input)
    assert oracle_input[396:].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]


def test_full_gold_oracle_overlay_binds_source_seed_order(tmp_path: Path) -> None:
    overlay = tmp_path / "full-gold-oracle.json"
    overlay.write_text(
        json.dumps(
            {
                "artifactType": "issue450-fixed-full-gold-location-overlay-v1",
                "fixedHoldoutManifestSha256": "fixed-hash",
                "classNames": [
                    "opponentSeat1",
                    "opponentSeat2",
                    "opponentSeat3",
                    "opponentSeat4",
                    "selfKittySolo",
                ],
                "sourceSeeds": [11, 22],
                "classIndices": [[4] * 53, [0] * 53],
            }
        ),
        encoding="utf-8",
    )
    loaded = load_full_gold_location_overlay(
        overlay, manifest_sha256="fixed-hash", source_seeds=(11, 22)
    )
    assert loaded["classIndicesArray"].shape == (2, 53)
    with pytest.raises(ValueError, match="source seed order mismatch"):
        load_full_gold_location_overlay(
            overlay, manifest_sha256="fixed-hash", source_seeds=(22, 11)
        )


def test_combined_dataset_rejects_source_overlap(tmp_path: Path) -> None:
    _write_dataset(tmp_path, states=3)
    dataset = load_exchange_counterfactual_dataset(tmp_path)

    with pytest.raises(ValueError, match="overlap by sourceStateKey"):
        combine_exchange_counterfactual_datasets((dataset, dataset))


def test_group_split_guards_state_and_identity_leakage(tmp_path: Path) -> None:
    _write_dataset(tmp_path, states=12)
    dataset = load_exchange_counterfactual_dataset(tmp_path)

    split = create_exchange_value_split(dataset, seed=436, train_state_count=4)

    assert split.train_state_keys.isdisjoint(split.validation_state_keys)
    assert split.train_state_keys.isdisjoint(split.final_state_keys)
    assert split.leakage_guard["status"] == "passed"
    assert split.leakage_guard["fixedThirteenGroupId"]["crossSplitLeakageCount"] == 0
    assert {sample.source_state_key for sample in split.train_samples} == split.train_state_keys
    for left, right in (
        (split.train_samples, split.validation_samples),
        (split.train_samples, split.final_samples),
        (split.validation_samples, split.final_samples),
    ):
        assert {sample.fixed_thirteen_group_id for sample in left}.isdisjoint(
            {sample.fixed_thirteen_group_id for sample in right}
        )


def test_group_split_skips_absent_optional_hand_identities(tmp_path: Path) -> None:
    _write_dataset(tmp_path, states=12)
    dataset = load_exchange_counterfactual_dataset(tmp_path)
    legacy = replace(
        dataset,
        raw_samples=tuple(
            replace(sample, original_hand_card_ids=None, kitty_pickup_card_ids=None)
            for sample in dataset.raw_samples
        ),
    )

    split = create_exchange_value_split(legacy, seed=436, train_state_count=4)

    assert len(split.train_state_keys) == 4
    assert split.leakage_guard["originalHand"]["uniqueCount"] == 0
    assert split.leakage_guard["kittyPickup"]["uniqueCount"] == 0


def test_group_split_preserves_fixed_thirteen_layout_counts(tmp_path: Path) -> None:
    _write_dataset(tmp_path, states=20)
    dataset = load_exchange_counterfactual_dataset(tmp_path)

    split = create_exchange_value_split(dataset, seed=442)

    assert len(split.train_state_keys) == 16
    assert len(split.validation_state_keys) == 2
    assert len(split.final_state_keys) == 2
    assert len(split.train_samples) == 16 * 286
    assert len(split.validation_samples) == 2 * 286
    assert len(split.final_samples) == 2 * 286
    for key in (
        "sourceStateKey",
        "fixedThirteenGroupId",
        "dealSeed",
        "hiddenDealChecksum",
        "pickupHand",
    ):
        assert split.leakage_guard[key]["crossSplitLeakageCount"] == 0


def test_model_uses_2671_plus_53_input_dimensions() -> None:
    model = ExchangeValueMlpModel(ExchangeValueMlpConfig(hidden_dims=(8,)))
    output = model(torch.zeros((2, EXCHANGE_VALUE_INPUT_FEATURE_COUNT), dtype=torch.float32))

    assert output.shape == (2,)


def test_model_uses_compact396_input_dimensions() -> None:
    model = ExchangeValueMlpModel(
        ExchangeValueMlpConfig(
            input_dim=EXCHANGE_COMPACT_VALUE_INPUT_FEATURE_COUNT,
            hidden_dims=(8,),
        )
    )
    output = model(
        torch.zeros((2, EXCHANGE_COMPACT_VALUE_INPUT_FEATURE_COUNT), dtype=torch.float32)
    )

    assert output.shape == (2,)


def test_compact406_tactical_features_use_visible_candidate_fields() -> None:
    compact = np.asarray(
        [
            *_compact_state(list(range(10)), [13, 14, 52]),
            *_mask([0, 3, 52]),
        ],
        dtype=np.float32,
    )

    tactical = compact396_tactical_features(compact)

    assert tactical.shape == (10,)
    assert tactical[0] == 2.0 / 3.0
    assert tactical[1] == 2.0 / 3.0
    assert tactical[4] == 1.0
    assert tactical[5] == 1.0
    assert tactical[6] == 1.0
    assert tactical[9] == 1.0


def test_statewise_listwise_compact406_training_smoke(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_dataset(dataset_dir, states=12)
    dataset = load_exchange_counterfactual_dataset(dataset_dir)
    baseline_result = train_exchange_value_model(
        dataset,
        ExchangeValueTrainConfig(
            seed=123,
            epochs=1,
            hidden_dims=(16,),
            input_variant="compact396",
            train_state_count=4,
            device="cpu",
        ),
    )
    baseline_artifact = save_exchange_value_artifact(
        tmp_path / "baseline-artifact", result=baseline_result, dataset=dataset
    )

    result = train_exchange_value_model(
        dataset,
        ExchangeValueTrainConfig(
            seed=123,
            epochs=1,
            hidden_dims=(16,),
            input_variant="compact406",
            train_state_count=4,
            pointwise_loss_weight=0.0,
            listwise_loss_weight=1.0,
            pairwise_state_batch_size=2,
            device="cpu",
            warm_start_checkpoint=cast(str, baseline_artifact["checkpointPath"]),
        ),
    )
    artifact = save_exchange_value_artifact(
        tmp_path / "listwise-artifact", result=result, dataset=dataset
    )
    loaded_model, checkpoint = load_exchange_value_checkpoint(cast(str, artifact["checkpointPath"]))

    assert result.model.config.input_dim == EXCHANGE_TACTICAL_VALUE_INPUT_FEATURE_COUNT
    assert loaded_model.config.input_dim == EXCHANGE_TACTICAL_VALUE_INPUT_FEATURE_COUNT
    assert checkpoint["trainingConfig"]["algorithm"] == ("state-wise-listwise-exchange-reranker-v1")
    assert checkpoint["trainingConfig"]["warmStartCheckpointSha256"]
    assert result.epoch_reports[0]["trainLoss"] > 0.0

    with pytest.raises(ValueError, match="warm-start input layout must match"):
        train_exchange_value_model(
            dataset,
            ExchangeValueTrainConfig(
                seed=123,
                epochs=1,
                hidden_dims=(16,),
                input_variant="legacy2724",
                train_state_count=4,
                device="cpu",
                warm_start_checkpoint=cast(str, baseline_artifact["checkpointPath"]),
            ),
        )


def test_non_default_pointwise_weight_uses_weighted_objective(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_dataset(dataset_dir, states=12)
    dataset = load_exchange_counterfactual_dataset(dataset_dir)
    common = dict(
        seed=123,
        epochs=1,
        hidden_dims=(16,),
        input_variant="compact396",
        train_state_count=4,
        pairwise_state_batch_size=2,
        learning_rate=1e-30,
        device="cpu",
    )

    unweighted = train_exchange_value_model(
        dataset, ExchangeValueTrainConfig(**common, pointwise_loss_weight=1.0)
    )
    half_weighted = train_exchange_value_model(
        dataset, ExchangeValueTrainConfig(**common, pointwise_loss_weight=0.5)
    )

    assert half_weighted.epoch_reports[0]["trainLoss"] == pytest.approx(
        float(unweighted.epoch_reports[0]["trainLoss"]) * 0.5
    )
    assert half_weighted.epoch_reports[0]["validationLoss"] == pytest.approx(
        float(unweighted.epoch_reports[0]["validationLoss"]) * 0.5
    )


def test_full_gold_containment_rank_and_regret_fixture(tmp_path: Path) -> None:
    state = np.asarray(_compact_state(list(range(10)), [10, 11, 12]), dtype="<f4")
    masks = np.zeros((1, 286, 53), dtype="u1")
    for candidate in range(286):
        masks[0, candidate, [candidate % 10, 10, 11]] = 1
    margins = np.zeros((1, 286), dtype="<f4")
    rewards = np.zeros((1, 286), dtype="<f4")
    margins[0, 3] = 5.0
    margins[0, 0] = 3.0
    scores = -np.arange(286, dtype=np.float32)[None, :]
    files = {
        "stateFeatures": ("exchange-state-features.f32", state[None, :]),
        "candidateMask": ("exchange-candidate-mask.u8", masks),
        "contractMargin": ("exchange-contract-margin.f32", margins),
        "relativeReward": ("exchange-relative-reward.f32", rewards),
        "ruleBasedCandidate": (
            "exchange-rule-based-candidate.u32",
            np.asarray([1], dtype="<u4"),
        ),
        "goldCandidate": (
            "exchange-gold-candidate.u32",
            np.asarray([3], dtype="<u4"),
        ),
    }
    manifest_files = {}
    for key, (name, values) in files.items():
        path = tmp_path / name
        values.tofile(path)
        manifest_files[key] = {
            "path": name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "artifactType": "issue446-fixed-exchange-full-gold-audit-v1",
                "fixedHoldout": {"manifestSha256": "fixture"},
                "groupCount": 1,
                "files": manifest_files,
                "sourceDiagnostics": [
                    {
                        "seed": 1,
                        "contractSuit": "spades",
                        "contractTarget": 13,
                        "shardSourceIndex": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = exchange_full_gold_report(
        load_exchange_full_gold_audit(tmp_path), scores, scorer_name="fixture",
        location_classes=np.asarray([4], dtype=np.int64),
    )

    assert report["containment"]["1"] == 0.0
    assert report["containment"]["4"] == 1.0
    assert report["goldBestRank"]["median"] == 4.0
    assert report["topKOracle"]["1"]["marginRegret"]["mean"] == 2.0
    location = report["failureAnalysis"]["adjutantLocationOracle"]
    assert location["classes"]["selfKittySolo"]["count"] == 1
    assert location["classes"]["selfKittySolo"]["top16Containment"] == 1.0
    assert report["failureAnalysis"]["metricsByCalledCardOrigin"]["originalHand"][
        "practicalK16MarginRegret"
    ]["mean"] == 0.0

    corrupted = bytearray((tmp_path / "exchange-contract-margin.f32").read_bytes())
    corrupted[0] ^= 1
    (tmp_path / "exchange-contract-margin.f32").write_bytes(corrupted)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_exchange_full_gold_audit(tmp_path)


def test_fixed_audit_overlap_filter_removes_complete_state_group(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "training"
    dataset_dir.mkdir()
    _write_dataset(dataset_dir, states=2)
    dataset = load_exchange_counterfactual_dataset(dataset_dir, load_legacy_model_input=False)
    state = np.asarray(_compact_state(list(range(10)), [10, 11, 12]), dtype="<f4")
    audit = ExchangeFullGoldAudit(
        directory=tmp_path,
        manifest={
            "sourceDiagnostics": [
                {
                    "seed": 436000,
                    "hiddenDealChecksum": "hidden-0",
                    "biddingHistoryHash": "history-0",
                }
            ]
        },
        state_features=np.asarray([state]),
        candidate_masks=np.zeros((1, 286, 53), dtype="u1"),
        contract_margins=np.zeros((1, 286), dtype="<f4"),
        relative_rewards=np.zeros((1, 286), dtype="<f4"),
        rule_based_candidates=np.zeros(1, dtype="<u4"),
        gold_candidates=np.zeros(1, dtype="<u4"),
    )

    filtered, exclusion = exclude_audit_overlaps(audit, dataset)

    assert exclusion["excludedSourceStateCount"] == 1
    assert filtered.source_state_count == 1
    assert filtered.sample_count == 286
    assert audit_training_leakage_report(audit, filtered)["status"] == "passed"

    legacy = replace(
        dataset,
        raw_samples=tuple(
            replace(sample, compact_exchange_state_input=None) for sample in dataset.raw_samples
        ),
    )
    with pytest.raises(ValueError, match="requires compactExchangeStateInput"):
        exclude_audit_overlaps(audit, legacy)
    with pytest.raises(ValueError, match="requires compactExchangeStateInput"):
        audit_training_leakage_report(audit, legacy)

    provenance = dataset_provenance(filtered)
    assert provenance["manifestScope"] == "source-before-in-memory-filtering"
    assert provenance["fixedAuditExclusion"] == exclusion
    assert provenance["effectiveManifestSha256"] != provenance["manifestSha256"]
    assert provenance["effectiveSourceStateKeyHash"] == state_key_hash(
        frozenset({"state-1"})
    )


def test_ranking_metrics_and_rule_based_fixture(tmp_path: Path) -> None:
    _write_dataset(tmp_path, states=2)
    dataset = load_exchange_counterfactual_dataset(tmp_path)
    samples = dataset.raw_samples
    predictions = np.asarray(
        [
            100.0 if sample.candidate_index == 0 else -float(sample.candidate_index)
            for sample in samples
        ],
        dtype=np.float32,
    )

    report = exchange_value_evaluation_report(samples, predictions=predictions, split="fixture")

    scalar = cast(dict[str, object], report["scalar"])
    assert scalar["sampleCount"] == 572
    ranking = cast(dict[str, object], report["ranking"])
    assert ranking["exactTopAction"] == 1.0
    assert ranking["top3Hit"] == 1.0
    assert ranking["top5Hit"] == 1.0
    model_regret = cast(dict[str, object], ranking["modelSelectedTeacherMarginRegret"])
    rule_based_regret = cast(dict[str, object], ranking["ruleBasedTeacherMarginRegret"])
    assert model_regret["mean"] == 0.0
    assert cast(float, rule_based_regret["mean"]) > 0.0
    bury_content = cast(dict[str, object], report["buryContent"])
    model_selected = cast(dict[str, object], bury_content["modelSelected"])
    assert model_selected["stateCount"] == 2
    same_thirteen = cast(dict[str, object], report["sameThirteen"])
    assert same_thirteen["groupCount"] == 1


def test_same_thirteen_aggregation_tracks_teacher_and_model_variation(tmp_path: Path) -> None:
    _write_dataset(tmp_path, states=4)
    shard = tmp_path / "shard-00000.jsonl"
    rows = [json.loads(line) for line in shard.read_text().splitlines()]
    for row in rows:
        if row["sourceIndex"] == 1 and row["candidateIndex"] == 2:
            row["contractMargin"] = 30.0
        if row["sourceIndex"] == 3 and row["candidateIndex"] == 3:
            row["contractMargin"] = 30.0
    shard_bytes = b"".join(
        (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8") for row in rows
    )
    shard.write_bytes(shard_bytes)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["shards"][0]["byteLength"] = len(shard_bytes)
    manifest["shards"][0]["sha256"] = hashlib.sha256(shard_bytes).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    dataset = load_exchange_counterfactual_dataset(tmp_path)
    predictions = np.asarray(
        [
            100.0 + float(sample.source_index)
            if sample.candidate_index == (sample.source_index % 2)
            else -float(sample.candidate_index)
            for sample in dataset.raw_samples
        ],
        dtype=np.float32,
    )

    report = exchange_value_evaluation_report(
        dataset.raw_samples,
        predictions=predictions,
        split="fixture",
    )

    same_thirteen = cast(dict[str, object], report["sameThirteen"])
    teacher_unique = cast(dict[str, object], same_thirteen["teacherBestDiscardUniqueCount"])
    model_unique = cast(dict[str, object], same_thirteen["modelSelectedDiscardUniqueCount"])
    model_consistency = cast(
        dict[str, object],
        same_thirteen["modelSelectedPredictedValueStdDev"],
    )
    assert same_thirteen["groupCount"] == 2
    assert teacher_unique["mean"] == 2.0
    assert model_unique["mean"] == 2.0
    assert model_consistency["count"] == 2
    assert cast(float, model_consistency["mean"]) > 0.0


def test_training_smoke_checkpoint_save_load_and_deterministic_eval(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_dataset(dataset_dir, states=12)
    dataset = load_exchange_counterfactual_dataset(dataset_dir)

    result = train_exchange_value_model(
        dataset,
        ExchangeValueTrainConfig(
            seed=123,
            epochs=1,
            batch_size=256,
            hidden_dims=(16,),
            train_state_count=4,
            device="cpu",
        ),
    )
    artifact = save_exchange_value_artifact(tmp_path / "artifact", result=result, dataset=dataset)
    loaded_model, checkpoint = load_exchange_value_checkpoint(cast(str, artifact["checkpointPath"]))

    assert checkpoint["modelType"] == "exchange-combination-value"
    assert loaded_model.config.input_dim == EXCHANGE_VALUE_INPUT_FEATURE_COUNT
    assert result.validation_report == result.validation_report
    assert artifact["checkpointSha256"]


def test_compact_training_smoke_checkpoint_save_load(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_dataset(dataset_dir, states=12)
    dataset = load_exchange_counterfactual_dataset(dataset_dir)

    result = train_exchange_value_model(
        dataset,
        ExchangeValueTrainConfig(
            seed=123,
            epochs=1,
            batch_size=256,
            hidden_dims=(16,),
            input_variant="compact396",
            train_state_count=4,
            device="cpu",
        ),
    )
    artifact = save_exchange_value_artifact(tmp_path / "artifact", result=result, dataset=dataset)
    loaded_model, checkpoint = load_exchange_value_checkpoint(cast(str, artifact["checkpointPath"]))

    assert loaded_model.config.input_dim == EXCHANGE_COMPACT_VALUE_INPUT_FEATURE_COUNT
    assert checkpoint["trainingConfig"]["input_variant"] == "compact396"


def test_issue442_report_validates_layout_counts_and_renders_baseline(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    artifact_dir = tmp_path / "artifact"
    dataset_dir.mkdir()
    _write_dataset(dataset_dir, states=12)
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    manifest["pseudoFixedThirteen"] = {
        "method": "pseudo-fixed-original10-kitty3-v1",
        "fixedThirteenGroupCount": 6,
        "acceptedDealsPerFixedThirteenGroup": 2,
        "acceptedDealCount": 12,
        "rejectedDealCount": 0,
        "acceptanceRate": 1.0,
        "rejectionReasons": {},
        "targetDistribution": {"13": 12},
        "suitDistribution": {"spades": 12},
        "biddingHistoryActionCount": {"count": 12, "mean": 5.0},
        "biddingHistoryUniqueHashCount": 12,
        "opponentPolicyCounts": {"frozen-raise-v1": 24},
        "opponentPolicyRatios": {"frozen-raise-v1": 1.0},
        "groupContractDiversity": {
            "targetMeanUniqueCount": 1.0,
            "suitMeanUniqueCount": 1.0,
            "targetSuitMeanUniqueCount": 1.0,
        },
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    dataset = load_exchange_counterfactual_dataset(dataset_dir)
    result = train_exchange_value_model(
        dataset,
        ExchangeValueTrainConfig(
            seed=123,
            epochs=1,
            batch_size=256,
            hidden_dims=(16,),
            input_variant="compact396",
            device="cpu",
        ),
    )
    save_exchange_value_artifact(artifact_dir, result=result, dataset=dataset)

    summary = collect_issue442_layout_summary(
        layout=Issue442Layout("fixture", 6, 2),
        dataset_directory=dataset_dir,
        artifact_directory=artifact_dir,
    )
    markdown = render_issue442_markdown(
        {
            "layouts": [summary],
            "conclusion": {
                "pseudoFixedRepeatsHelped": True,
                "bestLayout": "fixture",
                "adoptPseudoFixedTeacherCandidate": False,
                "moveToAdjutantKittyJointIfNoImprovement": True,
            },
        }
    )

    assert summary["checks"]["sampleCount"] is True
    assert summary["split"]["leakageGuard"]["fixedThirteenGroupId"]["crossSplitLeakageCount"] == 0
    assert "| #438 compact396 baseline | 0.409 | 0.592 | 4.60 | 12.30 | 4.59 |" in markdown


def test_compact_variant_rejects_legacy_dataset_without_compact_state(tmp_path: Path) -> None:
    _write_dataset(tmp_path, states=12)
    shard = tmp_path / "shard-00000.jsonl"
    rows = [json.loads(line) for line in shard.read_text().splitlines()]
    for row in rows:
        row.pop("compactExchangeStateInput")
    shard_bytes = b"".join(
        (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8") for row in rows
    )
    shard.write_bytes(shard_bytes)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest.pop("compactExchangeValueInput")
    manifest["shards"][0]["byteLength"] = len(shard_bytes)
    manifest["shards"][0]["sha256"] = hashlib.sha256(shard_bytes).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    dataset = load_exchange_counterfactual_dataset(tmp_path)

    with pytest.raises(ValueError, match="compact396 input is unavailable"):
        train_exchange_value_model(
            dataset,
            ExchangeValueTrainConfig(
                seed=123,
                epochs=1,
                batch_size=256,
                hidden_dims=(16,),
                input_variant="compact396",
                train_state_count=4,
                device="cpu",
            ),
        )


def test_loader_rejects_non_three_bit_candidate_mask(tmp_path: Path) -> None:
    _write_dataset(tmp_path, states=1)
    shard = tmp_path / "shard-00000.jsonl"
    rows = [json.loads(line) for line in shard.read_text().splitlines()]
    rows[0]["candidateDiscardMask"] = _mask([0, 1])
    shard_bytes = b"".join(
        (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8") for row in rows
    )
    shard.write_bytes(shard_bytes)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["shards"][0]["byteLength"] = len(shard_bytes)
    manifest["shards"][0]["sha256"] = hashlib.sha256(shard_bytes).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly 3"):
        load_exchange_counterfactual_dataset(tmp_path)
