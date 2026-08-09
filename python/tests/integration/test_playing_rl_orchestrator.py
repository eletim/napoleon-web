from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch

from napoleon_ml.cli.train_policy_mlp import main as train_supervised_main
from napoleon_ml.dataset.constants import CARD_COUNT, EXPECTED_CARD_IDS
from napoleon_ml.dataset.validation import calculate_card_ids_sha256
from napoleon_ml.rl_orchestrator import (
    PlayingRlOrchestratorError,
    PlayingRlRunConfig,
    run_playing_rl_experiment,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FIXTURE_PATH = Path(__file__).parents[1] / "unit" / "fixtures" / "valid_sample.json"


@pytest.fixture(scope="session", autouse=True)
def _built_typescript_helpers() -> None:
    result = subprocess.run(
        ["pnpm", "--filter", "@napoleon/self-play-cli...", "build"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"pnpm build failed (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.integration
def test_playing_rl_orchestrator_two_iteration_resume_and_safety(tmp_path: Path) -> None:
    supervised_dataset = tmp_path / "supervised"
    initial_checkpoint = tmp_path / "initial-playing.pt"
    _create_initial_checkpoint(supervised_dataset, initial_checkpoint)

    run_directory = tmp_path / "run"
    config = _small_config(
        run_directory=run_directory,
        initial_checkpoint=initial_checkpoint,
        supervised_dataset=supervised_dataset,
    )
    run_playing_rl_experiment(config, resume=False)

    iter0 = _load_json(run_directory / "iterations" / "iter-000" / "iteration.json")
    iter1 = _load_json(run_directory / "iterations" / "iter-001" / "iteration.json")
    assert iter0["inputCheckpointSha256"] == _sha256_file(initial_checkpoint)
    assert iter0["outputCheckpointSha256"] == iter1["inputCheckpointSha256"]
    assert iter0["selfPlayManifestSha256"] != iter1["selfPlayManifestSha256"]
    assert iter0["behaviorOnnxSha256"] != iter1["behaviorOnnxSha256"]
    assert _required_int(iter0["optimizerStepCount"]) > 0
    assert _required_int(iter1["optimizerStepCount"]) > 0

    manifest0 = _load_json(
        run_directory / "iterations" / "iter-000" / "selfplay" / "manifest.json"
    )
    manifest1 = _load_json(
        run_directory / "iterations" / "iter-001" / "selfplay" / "manifest.json"
    )
    assert manifest0["startSeed"] == 101
    assert manifest1["startSeed"] == 103
    assert (
        _required_object(manifest0["behaviorPolicy"])["onnxSha256"]
        == iter0["behaviorOnnxSha256"]
    )
    assert (
        _required_object(manifest1["behaviorPolicy"])["onnxSha256"]
        == iter1["behaviorOnnxSha256"]
    )

    evaluations = [
        _load_json(run_directory / "evaluations" / f"policy-v{generation:03d}" / "summary.json")
        for generation in (0, 1, 2)
    ]
    assert [evaluation["scheduledGames"] for evaluation in evaluations] == [5, 5, 5]
    assert evaluations[1]["pairedComparisonVsV0"] is not None
    assert evaluations[2]["winRateDeltaVsV0"] is not None

    before_resume = _sha256_file(run_directory / "iterations" / "iter-000" / "iteration.json")
    run_playing_rl_experiment(
        config,
        resume=True,
        provided_config_keys={"iterations", "gamesPerIteration"},
    )
    assert (
        _sha256_file(run_directory / "iterations" / "iter-000" / "iteration.json")
        == before_resume
    )

    mismatch = replace(config, games_per_iteration=config.games_per_iteration + 1)
    with pytest.raises(PlayingRlOrchestratorError, match="resume config mismatch"):
        run_playing_rl_experiment(
            mismatch,
            resume=True,
            provided_config_keys={"gamesPerIteration"},
        )

    checkpoint = run_directory / "iterations" / "iter-001" / "output-checkpoint.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"corrupt")
    with pytest.raises(PlayingRlOrchestratorError, match="SHA mismatch"):
        run_playing_rl_experiment(
            config,
            resume=True,
            provided_config_keys={"iterations", "gamesPerIteration"},
        )


@pytest.mark.integration
def test_playing_rl_orchestrator_reproducibility(tmp_path: Path) -> None:
    supervised_dataset = tmp_path / "supervised"
    initial_checkpoint = tmp_path / "initial-playing.pt"
    _create_initial_checkpoint(supervised_dataset, initial_checkpoint)

    left = tmp_path / "left-run"
    right = tmp_path / "right-run"
    left_config = _small_config(
        run_directory=left,
        initial_checkpoint=initial_checkpoint,
        supervised_dataset=supervised_dataset,
    )
    right_config = _small_config(
        run_directory=right,
        initial_checkpoint=initial_checkpoint,
        supervised_dataset=supervised_dataset,
    )
    run_playing_rl_experiment(left_config, resume=False)
    run_playing_rl_experiment(right_config, resume=False)

    for iteration in (0, 1):
        left_iter = _load_json(left / "iterations" / f"iter-{iteration:03d}" / "iteration.json")
        right_iter = _load_json(right / "iterations" / f"iter-{iteration:03d}" / "iteration.json")
        assert left_iter["selfPlayManifestSha256"] == right_iter["selfPlayManifestSha256"]
        assert left_iter["behaviorOnnxSha256"] == right_iter["behaviorOnnxSha256"]
        assert left_iter["outputCheckpointSha256"] == right_iter["outputCheckpointSha256"]
        _assert_jsonl_equal_excluding_paths(
            left / "iterations" / f"iter-{iteration:03d}" / "selfplay",
            right / "iterations" / f"iter-{iteration:03d}" / "selfplay",
        )
        _assert_checkpoint_parameters_equal(
            left / "iterations" / f"iter-{iteration:03d}" / "output-checkpoint.pt",
            right / "iterations" / f"iter-{iteration:03d}" / "output-checkpoint.pt",
        )

    left_eval = _load_json(left / "evaluations" / "policy-v002" / "summary.json")
    right_eval = _load_json(right / "evaluations" / "policy-v002" / "summary.json")
    assert _stable_evaluation_summary(left_eval) == _stable_evaluation_summary(right_eval)


def _small_config(
    *,
    run_directory: Path,
    initial_checkpoint: Path,
    supervised_dataset: Path,
) -> PlayingRlRunConfig:
    return PlayingRlRunConfig(
        run_directory=run_directory,
        initial_checkpoint=initial_checkpoint,
        supervised_dataset=supervised_dataset,
        iterations=2,
        games_per_iteration=2,
        games_per_shard=1,
        self_play_seed_base=101,
        temperature=1.0,
        learning_rate=0.001,
        epochs=1,
        batch_size=16,
        training_seed_base=201,
        evaluation_interval=1,
        evaluation_start_seed=301,
        evaluation_seed_count=1,
        build_typescript=False,
    )


def _create_initial_checkpoint(supervised_dataset: Path, output_checkpoint: Path) -> None:
    supervised_dataset.mkdir()
    _write_supervised_dataset(supervised_dataset, seeds=(0, 1, 2))
    assert (
        train_supervised_main(
            [
                str(supervised_dataset),
                "--output",
                str(output_checkpoint),
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--hidden-dim",
                "8",
                "--hidden-layers",
                "1",
                "--train-ratio",
                "100",
                "--validation-ratio",
                "0",
                "--test-ratio",
                "0",
                "--seed",
                "7",
                "--json",
            ]
        )
        == 0
    )


def _write_supervised_dataset(directory: Path, *, seeds: tuple[int, ...]) -> None:
    base = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    samples = [dict(base, seed=seed, step=1) for seed in seeds]
    shard_bytes = b"".join((json.dumps(sample) + "\n").encode("utf-8") for sample in samples)
    (directory / "shard-00000.jsonl").write_bytes(shard_bytes)
    manifest = {
        "datasetSchemaVersion": 1,
        "generatorVersion": 1,
        "playingEncoderSchemaVersion": 1,
        "format": "jsonl",
        "sampleType": "playing-training-sample",
        "agent": {"type": "rule-based", "version": 1},
        "startSeed": min(seeds),
        "endSeed": max(seeds),
        "gameCount": len(seeds),
        "sampleCount": len(samples),
        "gamesPerShard": len(seeds),
        "shardCount": 1,
        "playerCount": 5,
        "cardCount": CARD_COUNT,
        "cardIds": list(EXPECTED_CARD_IDS),
        "cardIdsSha256": calculate_card_ids_sha256(),
        "shards": [
            {
                "file": "shard-00000.jsonl",
                "startSeed": min(seeds),
                "endSeed": max(seeds),
                "gameCount": len(seeds),
                "sampleCount": len(samples),
                "byteLength": len(shard_bytes),
                "sha256": hashlib.sha256(shard_bytes).hexdigest(),
            }
        ],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _assert_jsonl_equal_excluding_paths(left: Path, right: Path) -> None:
    left_manifest = _load_json(left / "manifest.json")
    right_manifest = _load_json(right / "manifest.json")
    assert left_manifest == right_manifest
    for shard in _required_list(left_manifest["shards"]):
        file_name = cast(dict[str, object], shard)["file"]
        assert isinstance(file_name, str)
        assert (left / file_name).read_bytes() == (right / file_name).read_bytes()


def _assert_checkpoint_parameters_equal(left_path: Path, right_path: Path) -> None:
    left = torch.load(left_path, map_location="cpu", weights_only=True)
    right = torch.load(right_path, map_location="cpu", weights_only=True)
    left_state = cast(dict[str, torch.Tensor], left["model_state"])
    right_state = cast(dict[str, torch.Tensor], right["model_state"])
    assert left_state.keys() == right_state.keys()
    for key in left_state:
        assert torch.equal(left_state[key], right_state[key])


def _stable_evaluation_summary(summary: dict[str, object]) -> dict[str, object]:
    ignored = {"checkpointSha256", "onnxSha256", "metadataSha256"}
    return {key: value for key, value in summary.items() if key not in ignored}


def _load_json(path: Path) -> dict[str, object]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return cast(dict[str, object], parsed)


def _required_object(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _required_list(value: object) -> list[object]:
    assert isinstance(value, list)
    return cast(list[object], value)


def _required_int(value: object) -> int:
    assert isinstance(value, int)
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
