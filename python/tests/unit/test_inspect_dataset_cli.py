from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from napoleon_ml.cli import inspect_dataset
from napoleon_ml.cli.inspect_dataset import _Aggregates, main
from napoleon_ml.dataset.constants import EXPECTED_CARD_IDS
from napoleon_ml.dataset.manifest import DatasetAgentInfo, DatasetManifest, DatasetShardManifest
from napoleon_ml.dataset.validation import calculate_card_ids_sha256

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "valid_sample.json"


def _load_valid_sample() -> dict[str, Any]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_single_shard_dataset(directory: Path) -> None:
    sample = _load_valid_sample()
    shard_bytes = (json.dumps(sample) + "\n").encode("utf-8")
    shard_path = directory / "shard-00000.jsonl"
    shard_path.write_bytes(shard_bytes)

    manifest = {
        "datasetSchemaVersion": 1,
        "generatorVersion": 1,
        "playingEncoderSchemaVersion": 2,
        "format": "jsonl",
        "sampleType": "playing-training-sample",
        "agent": {"type": "rule-based", "version": 1},
        "startSeed": 0,
        "endSeed": 0,
        "gameCount": 1,
        "sampleCount": 1,
        "gamesPerShard": 1,
        "shardCount": 1,
        "playerCount": 5,
        "cardCount": 53,
        "cardIds": list(EXPECTED_CARD_IDS),
        "cardIdsSha256": calculate_card_ids_sha256(),
        "shards": [
            {
                "file": "shard-00000.jsonl",
                "startSeed": 0,
                "endSeed": 0,
                "gameCount": 1,
                "sampleCount": 1,
                "byteLength": len(shard_bytes),
                "sha256": _sha256_hex(shard_bytes),
            }
        ],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _minimal_manifest(*, sample_count: int) -> DatasetManifest:
    return DatasetManifest(
        dataset_schema_version=1,
        generator_version=1,
        playing_encoder_schema_version=1,
        playing_model_input_schema_version=None,
        encoder_schema_version=None,
        format="jsonl",
        sample_type="playing-training-sample",
        agent=DatasetAgentInfo(type="rule-based", version=1),
        start_seed=0,
        end_seed=0,
        game_count=1,
        sample_count=sample_count,
        games_per_shard=1,
        shard_count=1,
        player_count=5,
        card_count=53,
        card_ids=EXPECTED_CARD_IDS,
        card_ids_sha256=calculate_card_ids_sha256(),
        shards=(
            DatasetShardManifest(
                file="shard-00000.jsonl",
                start_seed=0,
                end_seed=0,
                game_count=1,
                sample_count=sample_count,
                byte_length=1,
                sha256="a" * 64,
            ),
        ),
    )


def test_main_succeeds_on_a_valid_dataset(tmp_path: Path) -> None:
    _write_single_shard_dataset(tmp_path)

    assert main([str(tmp_path)]) == 0


def test_main_reports_json_ok_on_a_valid_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_single_shard_dataset(tmp_path)

    assert main([str(tmp_path), "--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["dataset"]["samples"] == 1
    assert report["integrity"]["samplesValid"] == 1


def test_main_fails_when_streamed_sample_count_does_not_match_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The reader itself already enforces per-shard sample counts, so this
    # mismatch should be unreachable through real dataset content. This test
    # exercises the CLI's own independent cross-check as a last line of
    # defense, by forcing `_collect` to return a manifest/aggregate pair
    # that disagree, as a future bug elsewhere in the pipeline might.
    manifest = _minimal_manifest(sample_count=2)
    aggregates = _Aggregates(sample_count=1)

    def _fake_collect(*args: object, **kwargs: object) -> tuple[DatasetManifest, _Aggregates]:
        return manifest, aggregates

    monkeypatch.setattr(inspect_dataset, "_collect", _fake_collect)

    exit_code = main([str(tmp_path)])

    assert exit_code == 1
    assert "sample count mismatch" in capsys.readouterr().err
