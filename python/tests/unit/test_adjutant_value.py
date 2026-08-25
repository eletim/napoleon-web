from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from napoleon_ml.adjutant_value.training import (
    ADJUTANT_CANDIDATE_COUNT,
    ADJUTANT_VALUE_FEATURE_COUNT,
    AdjutantValueMlp,
    AdjutantValueMlpConfig,
    evaluate_full_gold,
    load_adjutant_value_dataset,
    split_by_state,
    train_adjutant_value_model,
)
from napoleon_ml.cli.merge_adjutant_value_datasets import merge_adjutant_value_datasets


def test_loader_validates_compact290_group_shape_and_split(tmp_path: Path) -> None:
    _write_dataset(tmp_path, source_state_count=6)

    dataset = load_adjutant_value_dataset(tmp_path)
    split = split_by_state(dataset, seed=446)

    assert dataset.sample_count == 6 * ADJUTANT_CANDIDATE_COUNT
    assert dataset.features.shape == (dataset.sample_count, ADJUTANT_VALUE_FEATURE_COUNT)
    assert all(indices.shape[0] % ADJUTANT_CANDIDATE_COUNT == 0 for indices in split.values())

    for state_index in range(dataset.source_state_count):
        start = state_index * ADJUTANT_CANDIDATE_COUNT
        stop = start + ADJUTANT_CANDIDATE_COUNT
        features = np.asarray(dataset.features[start:stop])
        assert np.all(features[:, :237] == features[0:1, :237])
        assert np.all(features[:, 237:].sum(axis=1) == 1.0)
        np.testing.assert_array_equal(np.argmax(features[:, 237:], axis=1), np.arange(53))


def test_full_gold_metrics_report_model_and_rule_based_regret(tmp_path: Path) -> None:
    _write_dataset(tmp_path, source_state_count=2)
    dataset = load_adjutant_value_dataset(tmp_path)
    model = AdjutantValueMlp(AdjutantValueMlpConfig(hidden_dims=()))
    with torch.no_grad():
        model.network[0].weight.zero_()
        model.network[0].bias.zero_()

    report = evaluate_full_gold(model, dataset, device=torch.device("cpu"))

    assert report["ranking"]["stateCount"] == 2
    assert report["ranking"]["exact"] == 0.5
    assert report["ranking"]["selectedRegret"]["mean"] == 0.5
    assert report["decomposition"]["ruleBasedAdjutantRegret"]["mean"] == 1.5
    assert report["decomposition"]["modelAdjOptimizedExchange"]["mean"] == 9.5
    assert report["proposal"]["candidateCount"] == 106
    assert report["proposal"]["topKContainment"]["top64"] == 8 / 106
    assert report["proposal"]["regretMean"]["fullProposalTop16RuleBasedDiversity"] == 12 / 106


def test_merge_preserves_manifest_provenance_and_reindexes_states(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    output = tmp_path / "merged"
    _write_dataset(left, source_state_count=1, start_seed=100)
    _write_dataset(right, source_state_count=1, start_seed=200)

    merge_adjutant_value_datasets([left, right], output)
    merged = load_adjutant_value_dataset(output)
    manifest = merged.manifest

    assert merged.source_state_count == 2
    assert merged.sample_count == 106
    assert manifest["mergedShardCount"] == 2
    assert len(manifest["mergedShards"]) == 2
    assert manifest["startSeed"] == 100
    assert manifest["endSeed"] == 200
    assert manifest["sourceDistribution"]["contractSuit"]["spades"] == 2
    assert np.asarray(merged.state_indices[:53]).tolist() == [0] * 53
    assert np.asarray(merged.state_indices[53:]).tolist() == [1] * 53
    assert manifest["sourceDiagnostics"][1]["sourceIndex"] == 1


def test_merge_rejects_output_that_is_an_input_shard(tmp_path: Path) -> None:
    shard = tmp_path / "shard"
    _write_dataset(shard, source_state_count=1)
    original_features = (shard / "features.f32").read_bytes()

    with pytest.raises(ValueError, match="output directory"):
        merge_adjutant_value_datasets([shard], shard)

    assert (shard / "features.f32").read_bytes() == original_features


def test_merge_rejects_duplicate_and_overlapping_shards(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    output = tmp_path / "merged"
    _write_dataset(left, source_state_count=1, start_seed=100)
    _write_dataset(right, source_state_count=1, start_seed=100)

    with pytest.raises(ValueError, match="input shard paths must be unique"):
        merge_adjutant_value_datasets([left, left], output)
    with pytest.raises(ValueError, match="overlapping source identity seed=100"):
        merge_adjutant_value_datasets([left, right], output)


def test_training_smoke_writes_checkpoint_metadata(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "model"
    _write_dataset(dataset_dir, source_state_count=5)

    result = train_adjutant_value_model(
        dataset_dir,
        output_directory=output_dir,
        seed=446,
        epochs=1,
        batch_size=64,
        hidden_dims=(16,),
        patience=1,
        device_name="cpu",
    )

    assert (output_dir / "checkpoint.pt").exists()
    assert (output_dir / "report.json").exists()
    assert (output_dir / "metadata.json").exists()
    assert result["metadata"]["checkpointSha256"]
    assert result["report"]["bestEpoch"] == 1


def _write_dataset(
    directory: Path,
    *,
    source_state_count: int,
    start_seed: int = 446000000,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    sample_count = source_state_count * ADJUTANT_CANDIDATE_COUNT
    features = np.zeros((sample_count, ADJUTANT_VALUE_FEATURE_COUNT), dtype="<f4")
    margins = np.zeros(sample_count, dtype="<f4")
    rewards = np.zeros(sample_count, dtype="<f4")
    state_indices = np.zeros(sample_count, dtype="<u4")
    candidate_cards = np.zeros(sample_count, dtype="u1")
    source_diagnostics = []

    for state_index in range(source_state_count):
        gold_index = state_index % 2
        rb_index = 2
        state_start = state_index * ADJUTANT_CANDIDATE_COUNT
        state_features = np.zeros(237, dtype=np.float32)
        state_features[0] = float(state_index + 1)
        state_features[53 + (state_index % 4)] = 1.0
        state_features[57 + (state_index % 7)] = 1.0
        state_features[64 + (state_index % 5)] = 1.0
        for candidate in range(ADJUTANT_CANDIDATE_COUNT):
            row = state_start + candidate
            features[row, :237] = state_features
            features[row, 237 + candidate] = 1.0
            margins[row] = 10.0 - abs(candidate - gold_index)
            rewards[row] = margins[row] * 2.0
            state_indices[row] = state_index
            candidate_cards[row] = candidate
        source_diagnostics.append(
            {
                "sourceIndex": state_index,
                "seed": start_seed + state_index,
                "napoleonSeatIndex": 0,
                "contractSuit": "spades",
                "contractTarget": 13,
                "ruleBasedAdjutantIndex": rb_index,
                "rbAdjRbExchangeMargin": -1.0,
                "proposalGoldContainment": {
                    "top4": 1,
                    "top8": 1,
                    "top16": 2,
                    "top32": 3,
                    "top64": 4,
                    "top16PlusRuleBased": 2,
                    "fullProposal": 5,
                    "ruleBasedExchange": 1,
                },
                "proposalBestRegretTop16Sum": 10.0,
                "proposalBestRegretTop16PlusRuleBasedSum": 8.0,
                "proposalBestRegretSum": 6.0,
            }
        )

    features.tofile(directory / "features.f32")
    margins.tofile(directory / "contract-margin.f32")
    rewards.tofile(directory / "relative-reward.f32")
    state_indices.tofile(directory / "state-index.u32")
    candidate_cards.tofile(directory / "candidate-card.u8")
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "datasetSchemaVersion": 1,
                "generatorVersion": 2,
                "sampleType": "adjutant-joint-value-v1",
                "teacherId": "issue446-compact396-proposal-joint-teacher-v1",
                "mode": "full-gold",
                "featureCount": ADJUTANT_VALUE_FEATURE_COUNT,
                "stateFeatureCount": 237,
                "candidateCountPerState": ADJUTANT_CANDIDATE_COUNT,
                "runtimeOrder": "adjutant -> kitty pickup -> exchange -> playing",
                "policyPath": {
                    "bidding": "frozen-raise-v1",
                    "playing": "ppo-separated-v1000",
                    "playingCritic": "ppo-separated-v1000/critic.onnx",
                },
                "proposal": {
                    "scorer": "compact396",
                    "topK": 16,
                    "diversityCount": 8,
                },
                "sourceStateCount": source_state_count,
                "requestedSourceStateCount": source_state_count,
                "sampleCount": sample_count,
                "terminalRolloutCount": sample_count,
                "startSeed": start_seed,
                "endSeed": start_seed + source_state_count - 1,
                "sourceDistribution": {
                    "contractSuit": {
                        "spades": source_state_count,
                        "hearts": 0,
                        "diamonds": 0,
                        "clubs": 0,
                    },
                    "contractTarget": {
                        str(target): source_state_count if target == 13 else 0
                        for target in range(13, 20)
                    },
                    "napoleonSeatIndex": {
                        str(seat): source_state_count if seat == 0 else 0
                        for seat in range(5)
                    },
                },
                "sourceDiagnostics": source_diagnostics,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
