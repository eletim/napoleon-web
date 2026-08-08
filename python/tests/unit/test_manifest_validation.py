from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from napoleon_ml.dataset._json import loads_strict
from napoleon_ml.dataset.constants import EXPECTED_CARD_IDS
from napoleon_ml.dataset.errors import ManifestValidationError, ShardIntegrityError
from napoleon_ml.dataset.manifest import parse_manifest
from napoleon_ml.dataset.reader import load_manifest
from napoleon_ml.dataset.validation import (
    calculate_card_ids_sha256,
    validate_dataset_directory,
    validate_manifest,
)


def _valid_manifest_dict() -> dict[str, Any]:
    return {
        "datasetSchemaVersion": 1,
        "generatorVersion": 1,
        "playingEncoderSchemaVersion": 1,
        "format": "jsonl",
        "sampleType": "playing-training-sample",
        "agent": {"type": "rule-based", "version": 1},
        "startSeed": 0,
        "endSeed": 2,
        "gameCount": 3,
        "sampleCount": 150,
        "gamesPerShard": 2,
        "shardCount": 2,
        "playerCount": 5,
        "cardCount": 53,
        "cardIds": list(EXPECTED_CARD_IDS),
        "cardIdsSha256": calculate_card_ids_sha256(),
        "shards": [
            {
                "file": "shard-00000.jsonl",
                "startSeed": 0,
                "endSeed": 1,
                "gameCount": 2,
                "sampleCount": 100,
                "byteLength": 1000,
                "sha256": "a" * 64,
            },
            {
                "file": "shard-00001.jsonl",
                "startSeed": 2,
                "endSeed": 2,
                "gameCount": 1,
                "sampleCount": 50,
                "byteLength": 500,
                "sha256": "b" * 64,
            },
        ],
    }


def _valid_self_play_manifest_dict() -> dict[str, Any]:
    raw = _valid_manifest_dict()
    del raw["agent"]
    raw.update(
        {
            "datasetSchemaVersion": 3,
            "generatorVersion": 1,
            "sampleType": "playing-self-play-sample",
            "sampleSchemaVersion": 1,
            "playingEncoderSchemaVersion": 1,
            "playingModelInputSchemaVersion": 1,
            "behaviorPolicy": {
                "type": "playing-onnx",
                "artifactId": "policy-fixture",
                "onnxFileName": "policy.onnx",
                "metadataFileName": "policy.json",
                "onnxSha256": "c" * 64,
                "metadataSha256": "d" * 64,
                "metadata": {"metadataSchemaVersion": 1},
            },
            "samplingAlgorithm": "masked-categorical",
            "temperature": 1.25,
            "reward": {"type": "terminal-team-win", "version": 1},
            "nonPlayingAgent": {"type": "rule-based", "version": 1},
        }
    )
    return raw


def _parse_and_validate(raw: dict[str, Any]) -> None:
    validate_manifest(parse_manifest(raw))


def test_valid_manifest_passes() -> None:
    _parse_and_validate(_valid_manifest_dict())


def test_schema_version_mismatch_rejected() -> None:
    raw = _valid_manifest_dict()
    raw["datasetSchemaVersion"] = 4

    with pytest.raises(ManifestValidationError, match="datasetSchemaVersion"):
        _parse_and_validate(raw)


def test_multiphase_manifest_passes() -> None:
    raw = _valid_manifest_dict()
    raw["datasetSchemaVersion"] = 2
    raw["generatorVersion"] = 2
    raw["encoderSchemaVersion"] = 1
    raw["sampleType"] = "bidding-training-sample"
    del raw["playingEncoderSchemaVersion"]

    manifest = parse_manifest(raw)
    validate_manifest(manifest)

    assert manifest.encoder_schema_version == 1
    assert manifest.playing_encoder_schema_version is None


def test_self_play_manifest_v3_passes() -> None:
    manifest = parse_manifest(_valid_self_play_manifest_dict())
    validate_manifest(manifest)

    assert manifest.dataset_schema_version == 3
    assert manifest.sample_type == "playing-self-play-sample"
    assert manifest.sample_schema_version == 1
    assert manifest.playing_encoder_schema_version == 1
    assert manifest.playing_model_input_schema_version == 1
    assert manifest.agent is None
    assert manifest.behavior_policy is not None
    assert manifest.behavior_policy.artifact_id == "policy-fixture"
    assert manifest.reward is not None
    assert manifest.reward.type == "terminal-team-win"
    assert manifest.non_playing_agent is not None
    assert manifest.non_playing_agent.type == "rule-based"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: raw.update({"sampleSchemaVersion": 2}), "sampleSchemaVersion"),
        (lambda raw: raw.update({"playingModelInputSchemaVersion": 2}), "ModelInput"),
        (lambda raw: raw.update({"samplingAlgorithm": "greedy"}), "samplingAlgorithm"),
        (lambda raw: raw.update({"temperature": 0}), "temperature"),
        (lambda raw: raw["behaviorPolicy"].update({"onnxSha256": "not-a-hash"}), "onnxSha256"),
        (lambda raw: raw["reward"].update({"type": "dense"}), "reward"),
        (lambda raw: raw["nonPlayingAgent"].update({"version": 2}), "nonPlayingAgent"),
    ],
)
def test_invalid_self_play_manifest_v3_rejected(
    mutate: Any, message: str
) -> None:
    raw = _valid_self_play_manifest_dict()
    mutate(raw)

    with pytest.raises(ManifestValidationError, match=message):
        _parse_and_validate(raw)


def test_self_play_manifest_unknown_behavior_policy_key_rejected() -> None:
    raw = _valid_self_play_manifest_dict()
    raw["behaviorPolicy"]["extraField"] = True

    with pytest.raises(ManifestValidationError, match="unknown key"):
        parse_manifest(raw)


def test_generator_version_mismatch_rejected() -> None:
    raw = _valid_manifest_dict()
    raw["generatorVersion"] = 2

    with pytest.raises(ManifestValidationError, match="generatorVersion"):
        _parse_and_validate(raw)


def test_encoder_version_mismatch_rejected() -> None:
    raw = _valid_manifest_dict()
    raw["playingEncoderSchemaVersion"] = 2

    with pytest.raises(ManifestValidationError, match="playingEncoderSchemaVersion"):
        _parse_and_validate(raw)


def test_card_ids_order_mismatch_rejected() -> None:
    raw = _valid_manifest_dict()
    card_ids = list(raw["cardIds"])
    card_ids[0], card_ids[1] = card_ids[1], card_ids[0]
    raw["cardIds"] = card_ids

    with pytest.raises(ManifestValidationError, match="cardIds"):
        _parse_and_validate(raw)


def test_card_ids_count_mismatch_rejected() -> None:
    raw = _valid_manifest_dict()
    raw["cardIds"] = list(EXPECTED_CARD_IDS)[:-1]

    with pytest.raises(ManifestValidationError, match="cardIds"):
        _parse_and_validate(raw)


def test_card_ids_hash_mismatch_rejected() -> None:
    raw = _valid_manifest_dict()
    raw["cardIdsSha256"] = "0" * 64

    with pytest.raises(ManifestValidationError, match="cardIdsSha256"):
        _parse_and_validate(raw)


def test_seed_range_mismatch_rejected() -> None:
    raw = _valid_manifest_dict()
    raw["endSeed"] = 5  # gameCount (3) no longer matches endSeed - startSeed + 1

    with pytest.raises(ManifestValidationError, match="gameCount"):
        _parse_and_validate(raw)


def test_game_count_mismatch_rejected() -> None:
    raw = _valid_manifest_dict()
    raw["gameCount"] = 4

    with pytest.raises(ManifestValidationError, match="gameCount"):
        _parse_and_validate(raw)


def test_sample_count_mismatch_rejected() -> None:
    raw = _valid_manifest_dict()
    raw["sampleCount"] = 151

    with pytest.raises(ManifestValidationError, match="sampleCount"):
        _parse_and_validate(raw)


def test_shard_count_mismatch_rejected() -> None:
    raw = _valid_manifest_dict()
    raw["shardCount"] = 3

    with pytest.raises(ManifestValidationError, match="shardCount"):
        _parse_and_validate(raw)


def test_shard_gap_rejected() -> None:
    raw = _valid_manifest_dict()
    raw["shards"][1]["startSeed"] = 3  # gap: shard 0 ends at seed 1, shard 1 should start at 2
    raw["shards"][1]["endSeed"] = 3
    raw["shards"][1]["gameCount"] = 1
    raw["endSeed"] = 3
    raw["gameCount"] = 4

    with pytest.raises(ManifestValidationError, match="gap or overlap"):
        _parse_and_validate(raw)


def test_shard_overlap_rejected() -> None:
    raw = _valid_manifest_dict()
    raw["shards"][1]["startSeed"] = 1  # overlap: shard 0 already claims seed 1
    raw["shards"][1]["gameCount"] = 2

    with pytest.raises(ManifestValidationError, match="gap or overlap"):
        _parse_and_validate(raw)


def test_invalid_shard_name_rejected() -> None:
    raw = _valid_manifest_dict()
    raw["shards"][0]["file"] = "shard-0.jsonl"

    with pytest.raises(ManifestValidationError, match="shard-00000.jsonl"):
        _parse_and_validate(raw)


def test_unknown_top_level_key_rejected() -> None:
    raw = _valid_manifest_dict()
    raw["extraField"] = "unexpected"

    with pytest.raises(ManifestValidationError, match="unknown key"):
        parse_manifest(raw)


def test_missing_top_level_key_rejected() -> None:
    raw = _valid_manifest_dict()
    del raw["cardIdsSha256"]

    with pytest.raises(ManifestValidationError, match="cardIdsSha256"):
        parse_manifest(raw)


def test_bool_rejected_where_integer_expected() -> None:
    raw = _valid_manifest_dict()
    raw["gameCount"] = True

    with pytest.raises(ManifestValidationError, match="integer"):
        parse_manifest(raw)


def test_duplicate_json_key_rejected() -> None:
    text = '{"a": 1, "a": 2}'

    with pytest.raises(ValueError, match="duplicate JSON key"):
        loads_strict(text)


def test_duplicate_json_key_in_manifest_file_rejected(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        '{"datasetSchemaVersion": 1, "datasetSchemaVersion": 2}', encoding="utf-8"
    )

    with pytest.raises(ManifestValidationError, match="not valid JSON"):
        load_manifest(tmp_path)


def test_load_manifest_round_trip(tmp_path: Path) -> None:
    manifest_dict = _valid_manifest_dict()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest_dict), encoding="utf-8")
    (tmp_path / "shard-00000.jsonl").write_bytes(b"")
    (tmp_path / "shard-00001.jsonl").write_bytes(b"")

    manifest = load_manifest(tmp_path)

    assert manifest.game_count == 3
    assert manifest.sample_count == 150
    assert manifest.shard_count == 2
    assert manifest.card_ids == EXPECTED_CARD_IDS


def test_missing_shard_file_rejected(tmp_path: Path) -> None:
    manifest_dict = _valid_manifest_dict()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest_dict), encoding="utf-8")
    (tmp_path / "shard-00000.jsonl").write_bytes(b"")
    # shard-00001.jsonl intentionally not created

    with pytest.raises(ShardIntegrityError, match="missing"):
        load_manifest(tmp_path)


def test_extra_shard_file_rejected(tmp_path: Path) -> None:
    manifest_dict = _valid_manifest_dict()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest_dict), encoding="utf-8")
    (tmp_path / "shard-00000.jsonl").write_bytes(b"")
    (tmp_path / "shard-00001.jsonl").write_bytes(b"")
    (tmp_path / "shard-00002.jsonl").write_bytes(b"")  # not listed in manifest

    with pytest.raises(ShardIntegrityError, match="unexpected entries"):
        load_manifest(tmp_path)


def test_absolute_shard_path_rejected(tmp_path: Path) -> None:
    manifest_dict = _valid_manifest_dict()
    manifest_dict["shards"][0]["file"] = "/etc/passwd"
    manifest = parse_manifest(manifest_dict)

    with pytest.raises(ShardIntegrityError, match="bare file name"):
        validate_dataset_directory(tmp_path, manifest)


def test_path_traversal_shard_name_rejected(tmp_path: Path) -> None:
    manifest_dict = _valid_manifest_dict()
    manifest_dict["shards"][0]["file"] = "../outside.jsonl"
    manifest = parse_manifest(manifest_dict)

    with pytest.raises(ShardIntegrityError, match="bare file name"):
        validate_dataset_directory(tmp_path, manifest)


def test_nested_shard_path_rejected(tmp_path: Path) -> None:
    manifest_dict = _valid_manifest_dict()
    manifest_dict["shards"][0]["file"] = "nested/shard-00000.jsonl"
    manifest = parse_manifest(manifest_dict)

    with pytest.raises(ShardIntegrityError, match="bare file name"):
        validate_dataset_directory(tmp_path, manifest)


def test_symlinked_shard_rejected(tmp_path: Path) -> None:
    outside_target = tmp_path.parent / "outside-shard.jsonl"
    outside_target.write_bytes(b"")

    manifest_dict = _valid_manifest_dict()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest_dict), encoding="utf-8")
    (tmp_path / "shard-00000.jsonl").symlink_to(outside_target)
    (tmp_path / "shard-00001.jsonl").write_bytes(b"")

    with pytest.raises(ShardIntegrityError, match="symlink"):
        load_manifest(tmp_path)


def test_manifest_dict_is_never_mutated_by_validation() -> None:
    raw = _valid_manifest_dict()
    snapshot = copy.deepcopy(raw)
    _parse_and_validate(raw)

    assert raw == snapshot
