from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch

from napoleon_ml.cli.run_playing_rl import _config_from_args, build_parser
from napoleon_ml.cli.train_policy_mlp import main as train_supervised_main
from napoleon_ml.dataset.constants import (
    CARD_COUNT,
    COMPLETE_INFO_COMPACT_PLAYING_OBSERVATION_VARIANT,
    EXPECTED_CARD_IDS,
)
from napoleon_ml.dataset.validation import calculate_card_ids_sha256
from napoleon_ml.policy.actor_critic import (
    ACTOR_CRITIC_ALGORITHM,
    DEFAULT_PPO_CLIP_EPSILON,
    PPO_SEPARATED_ACTOR_CRITIC_ALGORITHM,
    SEPARATED_ACTOR_CRITIC_ALGORITHM,
)
from napoleon_ml.policy.checkpoint import SEPARATED_ACTOR_CRITIC_MODEL_ARCHITECTURE
from napoleon_ml.policy.reinforce import REINFORCE_ALGORITHM
from napoleon_ml.rl_orchestrator import (
    DEFAULT_EVALUATION_SEED_COUNT,
    DEFAULT_FULL_DIAGNOSTICS_INTERVAL,
    PlayingRlOrchestratorError,
    PlayingRlRunConfig,
    _config_from_file_dict,
    _full_diagnostics_is_due,
    _validate_config,
    _validate_resume_config,
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

    stored_config = _load_json(run_directory / "config.json")
    assert stored_config["device"] == "cpu"
    assert stored_config["inferenceDevice"] == "cpu"
    assert stored_config["simulationBackend"] == "typescript"
    assert stored_config["rolloutConcurrency"] == 2
    assert stored_config["inferenceMaxBatchSize"] == 256
    assert stored_config["retainSelfPlayData"] is False
    assert stored_config["fullDiagnosticsInterval"] == 1
    iter0 = _load_json(run_directory / "iterations" / "iter-000" / "iteration.json")
    iter1 = _load_json(run_directory / "iterations" / "iter-001" / "iteration.json")
    assert iter0["inputCheckpointSha256"] == _sha256_file(initial_checkpoint)
    assert iter0["outputCheckpointSha256"] == iter1["inputCheckpointSha256"]
    assert iter0["selfPlayManifestSha256"] != iter1["selfPlayManifestSha256"]
    assert iter0["behaviorOnnxSha256"] != iter1["behaviorOnnxSha256"]
    assert iter0["rolloutWorkers"] == 2
    assert iter1["rolloutWorkers"] == 2
    assert iter0["rolloutConcurrency"] == 2
    assert iter0["inferenceMaxBatchSize"] == 256
    assert _required_int(iter0["inferenceRequestCount"]) > 0
    assert _required_int(iter0["inferenceSessionRunCount"]) > 0
    assert _required_float(iter0["inferenceMeanBatchSize"]) >= 1.0
    assert _required_int(iter0["inferenceMaxObservedBatchSize"]) >= 1
    rollout_timing = _load_json(
        run_directory / "iterations" / "iter-000" / "selfplay-rollout-timing.json"
    )
    assert rollout_timing["inferenceSessionRunCount"] == iter0["inferenceSessionRunCount"]
    assert rollout_timing["inferenceBatchSizeHistogram"] == iter0["inferenceBatchSizeHistogram"]
    assert iter0["requestedInferenceDevice"] == "cpu"
    assert iter0["resolvedInferenceDevice"] == "cpu"
    assert iter0["executionProvider"] == "cpu"
    assert iter0["selfPlayFormat"] == "playing-self-play-binary-v1"
    assert iter0["selfPlayTensorCompression"] == "none"
    assert _required_int(iter0["selfPlayShardCount"]) > 0
    assert _required_int(iter0["selfPlayShardByteLength"]) > 0
    assert iter0["selfPlayCacheRetained"] is False
    assert iter0["requestedDevice"] == "cpu"
    assert iter0["resolvedDevice"] == "cpu"
    assert iter0["cudaDeviceName"] is None
    assert iter0["diagnosticsPerformed"] is True
    assert _required_float(iter0["safetyValidationElapsedSeconds"]) >= 0.0
    assert _required_float(iter0["preEvalElapsedSeconds"]) >= 0.0
    assert _required_float(iter0["optimizerTrainingElapsedSeconds"]) >= 0.0
    assert _required_float(iter0["postEvalElapsedSeconds"]) >= 0.0
    assert _required_float(iter0["totalElapsedSeconds"]) >= 0.0
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
    assert _required_object(manifest0["behaviorPolicy"])["requestedInferenceDevice"] == "cpu"
    assert _required_object(manifest0["behaviorPolicy"])["resolvedInferenceDevice"] == "cpu"
    assert _required_object(manifest0["behaviorPolicy"])["executionProvider"] == "cpu"
    assert (
        _required_object(manifest1["behaviorPolicy"])["onnxSha256"]
        == iter1["behaviorOnnxSha256"]
    )
    assert _required_object(manifest0["rolloutRoster"])["assignment"] == "rotate-by-seed"
    assert [
        _required_object(seat)["source"]
        for seat in _required_list(_required_object(manifest0["rolloutRoster"])["seats"])
    ] == ["current-policy"] * 5
    for shard in _required_list(manifest0["shards"]):
        shard_file = _required_str(_required_object(shard)["file"])
        assert not (run_directory / "iterations" / "iter-000" / "selfplay" / shard_file).exists()

    for shard in _required_list(manifest0["shards"]):
        shard_file = _required_str(_required_object(shard)["file"])
        (run_directory / "iterations" / "iter-000" / "selfplay" / shard_file).write_bytes(
            b"orphaned-after-crash"
        )
    run_playing_rl_experiment(
        config,
        resume=True,
        provided_config_keys={"iterations", "gamesPerIteration"},
    )
    for shard in _required_list(manifest0["shards"]):
        shard_file = _required_str(_required_object(shard)["file"])
        assert not (run_directory / "iterations" / "iter-000" / "selfplay" / shard_file).exists()

    evaluations = [
        _load_json(run_directory / "evaluations" / f"policy-v{generation:03d}" / "summary.json")
        for generation in (0, 1, 2)
    ]
    assert [evaluation["scheduledGames"] for evaluation in evaluations] == [5, 5, 5]
    assert evaluations[0]["requestedInferenceDevice"] == "cpu"
    assert evaluations[0]["resolvedInferenceDevice"] == "cpu"
    assert evaluations[0]["executionProvider"] == "cpu"
    assert evaluations[1]["pairedComparisonVsV0"] is not None
    assert evaluations[2]["winRateDeltaVsV0"] is not None

    final_evaluation = run_directory / "evaluations" / "policy-v002"
    shutil.rmtree(final_evaluation)
    run_playing_rl_experiment(
        config,
        resume=True,
        provided_config_keys={"iterations", "gamesPerIteration"},
    )
    assert (final_evaluation / "summary.json").is_file()

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

    legacy_config = _load_json(run_directory / "config.json")
    legacy_config.pop("device")
    legacy_config.pop("inferenceDevice")
    legacy_config.pop("rolloutConcurrency")
    legacy_config.pop("inferenceMaxBatchSize")
    legacy_config.pop("fullDiagnosticsInterval")
    (run_directory / "config.json").write_text(
        json.dumps(legacy_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run_playing_rl_experiment(
        config,
        resume=True,
        provided_config_keys={"iterations", "gamesPerIteration"},
    )

    mismatch = replace(config, games_per_iteration=config.games_per_iteration + 1)
    with pytest.raises(PlayingRlOrchestratorError, match="resume config mismatch"):
        run_playing_rl_experiment(
            mismatch,
            resume=True,
            provided_config_keys={"gamesPerIteration"},
        )

    roster_mismatch = replace(
        config,
        rollout_roster="current-policy,rule-based,rule-based,rule-based,rule-based",
    )
    with pytest.raises(
        PlayingRlOrchestratorError,
        match="resume config mismatch for rolloutRoster",
    ):
        run_playing_rl_experiment(
            roster_mismatch,
            resume=True,
            provided_config_keys=set(),
        )

    worker_mismatch = replace(
        config,
        rollout_workers=config.rollout_workers + 1,
        rollout_concurrency=config.rollout_workers,
    )
    with pytest.raises(
        PlayingRlOrchestratorError,
        match="resume config mismatch for rolloutWorkers",
    ):
        run_playing_rl_experiment(
            worker_mismatch,
            resume=True,
            provided_config_keys={"rolloutWorkers"},
        )

    inference_mismatch = replace(config, inference_device="cuda")
    with pytest.raises(
        PlayingRlOrchestratorError,
        match="resume config mismatch for inferenceDevice",
    ):
        run_playing_rl_experiment(
            inference_mismatch,
            resume=True,
            provided_config_keys={"inferenceDevice"},
        )

    concurrency_mismatch = replace(config, rollout_concurrency=config.rollout_workers + 1)
    with pytest.raises(
        PlayingRlOrchestratorError,
        match="resume config mismatch for rolloutConcurrency",
    ):
        run_playing_rl_experiment(
            concurrency_mismatch,
            resume=True,
            provided_config_keys={"rolloutConcurrency"},
        )

    max_batch_mismatch = replace(config, inference_max_batch_size=128)
    with pytest.raises(
        PlayingRlOrchestratorError,
        match="resume config mismatch for inferenceMaxBatchSize",
    ):
        run_playing_rl_experiment(
            max_batch_mismatch,
            resume=True,
            provided_config_keys={"inferenceMaxBatchSize"},
        )

    device_mismatch = replace(config, device="auto")
    with pytest.raises(
        PlayingRlOrchestratorError,
        match="resume config mismatch for device",
    ):
        run_playing_rl_experiment(
            device_mismatch,
            resume=True,
            provided_config_keys={"device"},
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
def test_playing_rl_orchestrator_can_retain_self_play_cache(tmp_path: Path) -> None:
    supervised_dataset = tmp_path / "supervised"
    initial_checkpoint = tmp_path / "initial-playing.pt"
    _create_initial_checkpoint(supervised_dataset, initial_checkpoint)

    run_directory = tmp_path / "run"
    config = replace(
        _small_config(
            run_directory=run_directory,
            initial_checkpoint=initial_checkpoint,
            supervised_dataset=supervised_dataset,
        ),
        iterations=1,
        retain_self_play_data=True,
    )
    run_playing_rl_experiment(config, resume=False)

    iter0 = _load_json(run_directory / "iterations" / "iter-000" / "iteration.json")
    manifest0 = _load_json(
        run_directory / "iterations" / "iter-000" / "selfplay" / "manifest.json"
    )
    assert iter0["selfPlayCacheRetained"] is True
    for shard in _required_list(manifest0["shards"]):
        shard_file = _required_str(_required_object(shard)["file"])
        assert (run_directory / "iterations" / "iter-000" / "selfplay" / shard_file).is_file()


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
    left_config = replace(left_config, retain_self_play_data=True)
    right_config = replace(right_config, retain_self_play_data=True)
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


@pytest.mark.integration
def test_playing_rl_orchestrator_actor_critic_two_iteration_resume_and_safety(
    tmp_path: Path,
) -> None:
    supervised_dataset = tmp_path / "supervised"
    initial_checkpoint = tmp_path / "initial-playing.pt"
    _create_initial_checkpoint(supervised_dataset, initial_checkpoint)

    run_directory = tmp_path / "actor-critic-run"
    config = _small_config(
        run_directory=run_directory,
        initial_checkpoint=initial_checkpoint,
        supervised_dataset=supervised_dataset,
        algorithm=ACTOR_CRITIC_ALGORITHM,
    )
    run_playing_rl_experiment(config, resume=False)

    iter0 = _load_json(run_directory / "iterations" / "iter-000" / "iteration.json")
    iter1 = _load_json(run_directory / "iterations" / "iter-001" / "iteration.json")
    assert iter0["algorithm"] == ACTOR_CRITIC_ALGORITHM
    assert iter1["algorithm"] == ACTOR_CRITIC_ALGORITHM
    assert iter0["outputCheckpointSha256"] == iter1["inputCheckpointSha256"]
    assert iter0["selfPlayManifestSha256"] != iter1["selfPlayManifestSha256"]
    assert _required_int(iter0["optimizerStepCount"]) > 0
    assert _required_int(iter1["optimizerStepCount"]) > 0

    for iteration in (0, 1):
        report = _load_json(
            run_directory / "iterations" / f"iter-{iteration:03d}" / "train-report.json"
        )
        assert report["requestedDevice"] == "cpu"
        assert report["resolvedDevice"] == "cpu"
        assert report["diagnosticsPerformed"] is True
        assert _required_float(report["totalElapsedSeconds"]) >= 0.0
        assert report["actorLossBefore"] is not None
        assert report["valueLossBefore"] is not None
        assert report["totalLossAfter"] is not None
        assert _required_float(report["criticParameterDeltaNorm"]) > 0
        assert _required_int(report["changedCriticParameterCount"]) > 0
        checkpoint = torch.load(
            run_directory / "iterations" / f"iter-{iteration:03d}" / "output-checkpoint.pt",
            map_location="cpu",
            weights_only=True,
        )
        assert checkpoint["model_architecture"] == "playing-actor-critic-v1"
        provenance = cast(dict[str, object], checkpoint["rl_provenance"])
        assert provenance["algorithm"] == ACTOR_CRITIC_ALGORITHM
        assert provenance["requestedDevice"] == "cpu"
        assert provenance["resolvedDevice"] == "cpu"

    final_evaluation = run_directory / "evaluations" / "policy-v002"
    shutil.rmtree(final_evaluation)
    run_playing_rl_experiment(
        config,
        resume=True,
        provided_config_keys={"iterations", "gamesPerIteration", "algorithm"},
    )
    assert (final_evaluation / "summary.json").is_file()

    mismatch = replace(config, algorithm=REINFORCE_ALGORITHM)
    with pytest.raises(PlayingRlOrchestratorError, match="resume config mismatch for algorithm"):
        run_playing_rl_experiment(
            mismatch,
            resume=True,
            provided_config_keys={"algorithm"},
        )


@pytest.mark.integration
def test_playing_rl_orchestrator_separated_actor_critic_typescript_smoke(
    tmp_path: Path,
) -> None:
    supervised_dataset = tmp_path / "supervised"
    initial_checkpoint = tmp_path / "initial-playing.pt"
    _create_initial_checkpoint(supervised_dataset, initial_checkpoint)

    run_directory = tmp_path / "separated-actor-critic-run"
    config = replace(
        _small_config(
            run_directory=run_directory,
            initial_checkpoint=initial_checkpoint,
            supervised_dataset=supervised_dataset,
            algorithm=SEPARATED_ACTOR_CRITIC_ALGORITHM,
        ),
        iterations=1,
        games_per_iteration=1,
        games_per_shard=1,
        rollout_workers=1,
        rollout_concurrency=1,
        batch_size=8,
        evaluation_seed_count=1,
    )
    run_playing_rl_experiment(config, resume=False)

    iter0 = _load_json(run_directory / "iterations" / "iter-000" / "iteration.json")
    report = _load_json(run_directory / "iterations" / "iter-000" / "train-report.json")
    assert iter0["algorithm"] == SEPARATED_ACTOR_CRITIC_ALGORITHM
    assert _required_int(iter0["optimizerStepCount"]) > 0
    assert _required_float(report["actorParameterDeltaNorm"]) > 0
    assert _required_float(report["criticParameterDeltaNorm"]) > 0
    checkpoint = torch.load(
        run_directory / "iterations" / "iter-000" / "output-checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["model_architecture"] == SEPARATED_ACTOR_CRITIC_MODEL_ARCHITECTURE

    evaluation = _load_json(run_directory / "evaluations" / "policy-v001" / "summary.json")
    assert evaluation["completedGames"] == 5
    metadata = _load_json(run_directory / "evaluations" / "policy-v001" / "policy.json")
    assert metadata["modelArchitecture"] == SEPARATED_ACTOR_CRITIC_MODEL_ARCHITECTURE
    assert _required_object(metadata["onnx"])["outputs"] == [
        {"name": "logits", "shape": ["batch", CARD_COUNT], "dtype": "float32"}
    ]


@pytest.mark.integration
def test_playing_rl_orchestrator_separated_actor_critic_cpp_smoke(tmp_path: Path) -> None:
    _ensure_cpp_onnxruntime_enabled_or_skip(tmp_path)
    supervised_dataset = tmp_path / "supervised"
    initial_checkpoint = tmp_path / "initial-playing.pt"
    _create_initial_checkpoint(supervised_dataset, initial_checkpoint)

    run_directory = tmp_path / "separated-actor-critic-cpp-run"
    config = replace(
        _small_config(
            run_directory=run_directory,
            initial_checkpoint=initial_checkpoint,
            supervised_dataset=supervised_dataset,
            algorithm=SEPARATED_ACTOR_CRITIC_ALGORITHM,
        ),
        iterations=1,
        games_per_iteration=1,
        games_per_shard=1,
        rollout_workers=1,
        rollout_concurrency=1,
        batch_size=8,
        evaluation_seed_count=1,
        simulation_backend="cpp",
        frozen_policy_onnx=_REPO_ROOT / "benchmarks/playing-policies/rl-v740/policy.onnx",
        frozen_policy_metadata=_REPO_ROOT / "benchmarks/playing-policies/rl-v740/policy.json",
        build_cpp=True,
    )
    run_playing_rl_experiment(config, resume=False)

    iter0 = _load_json(run_directory / "iterations" / "iter-000" / "iteration.json")
    report = _load_json(run_directory / "iterations" / "iter-000" / "train-report.json")
    assert iter0["algorithm"] == SEPARATED_ACTOR_CRITIC_ALGORITHM
    assert iter0["simulationBackend"] == "cpp"
    assert _required_int(iter0["optimizerStepCount"]) > 0
    assert _required_float(report["actorParameterDeltaNorm"]) > 0
    assert _required_float(report["criticParameterDeltaNorm"]) > 0
    checkpoint = torch.load(
        run_directory / "iterations" / "iter-000" / "output-checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["model_architecture"] == SEPARATED_ACTOR_CRITIC_MODEL_ARCHITECTURE

    evaluation = _load_json(run_directory / "evaluations" / "policy-v001" / "summary.json")
    assert evaluation["simulationBackend"] == "cpp"
    assert evaluation["completedGames"] == 5
    metadata = _load_json(run_directory / "evaluations" / "policy-v001" / "policy.json")
    assert metadata["modelArchitecture"] == SEPARATED_ACTOR_CRITIC_MODEL_ARCHITECTURE
    assert _required_object(metadata["onnx"])["outputs"] == [
        {"name": "logits", "shape": ["batch", CARD_COUNT], "dtype": "float32"}
    ]


@pytest.mark.integration
def test_playing_rl_orchestrator_ppo_separated_cpp_smoke(tmp_path: Path) -> None:
    _ensure_cpp_onnxruntime_enabled_or_skip(tmp_path)
    supervised_dataset = tmp_path / "supervised"
    initial_checkpoint = tmp_path / "initial-playing.pt"
    _create_initial_checkpoint(supervised_dataset, initial_checkpoint)

    run_directory = tmp_path / "ppo-separated-cpp-run"
    config = replace(
        _small_config(
            run_directory=run_directory,
            initial_checkpoint=initial_checkpoint,
            supervised_dataset=supervised_dataset,
            algorithm=PPO_SEPARATED_ACTOR_CRITIC_ALGORITHM,
        ),
        iterations=1,
        games_per_iteration=1,
        games_per_shard=1,
        rollout_workers=1,
        rollout_concurrency=1,
        batch_size=8,
        evaluation_seed_count=1,
        simulation_backend="cpp",
        frozen_policy_onnx=_REPO_ROOT / "benchmarks/playing-policies/rl-v740/policy.onnx",
        frozen_policy_metadata=_REPO_ROOT / "benchmarks/playing-policies/rl-v740/policy.json",
        ppo_clip_epsilon=0.2,
        build_cpp=True,
    )
    run_playing_rl_experiment(config, resume=False)

    iter0 = _load_json(run_directory / "iterations" / "iter-000" / "iteration.json")
    report = _load_json(run_directory / "iterations" / "iter-000" / "train-report.json")
    assert iter0["algorithm"] == PPO_SEPARATED_ACTOR_CRITIC_ALGORITHM
    assert iter0["simulationBackend"] == "cpp"
    assert iter0["ppoClipEpsilon"] == pytest.approx(0.2)
    assert iter0["ppoMeanProbabilityRatio"] is not None
    assert iter0["ppoClippedFraction"] is not None
    assert iter0["ppoApproximateKl"] is not None
    assert _required_int(iter0["optimizerStepCount"]) > 0
    assert report["ppoClipEpsilon"] == pytest.approx(0.2)
    assert report["ppoMeanProbabilityRatio"] is not None
    assert report["ppoClippedFraction"] is not None
    assert report["ppoApproximateKl"] is not None
    checkpoint = torch.load(
        run_directory / "iterations" / "iter-000" / "output-checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["model_architecture"] == SEPARATED_ACTOR_CRITIC_MODEL_ARCHITECTURE
    provenance = cast(dict[str, object], checkpoint["rl_provenance"])
    assert provenance["algorithm"] == PPO_SEPARATED_ACTOR_CRITIC_ALGORITHM
    assert provenance["ppoClipEpsilon"] == pytest.approx(0.2)

    evaluation = _load_json(run_directory / "evaluations" / "policy-v001" / "summary.json")
    assert evaluation["simulationBackend"] == "cpp"
    assert evaluation["completedGames"] == 5
    metadata = _load_json(run_directory / "evaluations" / "policy-v001" / "policy.json")
    assert metadata["modelArchitecture"] == SEPARATED_ACTOR_CRITIC_MODEL_ARCHITECTURE
    assert _required_object(metadata["onnx"])["outputs"] == [
        {"name": "logits", "shape": ["batch", CARD_COUNT], "dtype": "float32"}
    ]


def _ensure_cpp_onnxruntime_enabled_or_skip(tmp_path: Path) -> None:
    native_dir = _REPO_ROOT / "packages/cpp-core/native"
    probe_build_dir = tmp_path / "cpp-onnxruntime-probe"
    probe = subprocess.run(
        [
            "cmake",
            "-S",
            str(native_dir),
            "-B",
            str(probe_build_dir),
            "-DNAPOLEON_ENABLE_ONNXRUNTIME=ON",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if probe.returncode != 0:
        pytest.skip(
            "C++ ONNX Runtime SDK is unavailable; "
            f"cmake stdout:\n{probe.stdout}\nstderr:\n{probe.stderr}"
        )

    configure = subprocess.run(
        [
            "cmake",
            "-S",
            str(native_dir),
            "-B",
            str(_REPO_ROOT / "packages/cpp-core/build"),
            "-DNAPOLEON_ENABLE_ONNXRUNTIME=ON",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert configure.returncode == 0, (
        "C++ ONNX Runtime configure failed after probe success:\n"
        f"stdout:\n{configure.stdout}\nstderr:\n{configure.stderr}"
    )


def test_playing_rl_orchestrator_explicit_cuda_fails_during_config_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = PlayingRlRunConfig(
        run_directory=tmp_path / "run",
        initial_checkpoint=tmp_path / "missing.pt",
        supervised_dataset=tmp_path / "missing-dataset",
        device="cuda",
    )

    with pytest.raises(PlayingRlOrchestratorError, match="explicitly requested"):
        _validate_config(config)


def test_complete_info_compact_cpp_backend_is_allowed_after_cpp_rollout_support(
    tmp_path: Path,
) -> None:
    frozen_onnx = tmp_path / "frozen.onnx"
    frozen_metadata = tmp_path / "frozen.json"
    frozen_onnx.write_bytes(b"frozen")
    frozen_metadata.write_text("{}", encoding="utf-8")

    config = PlayingRlRunConfig(
        run_directory=tmp_path / "run",
        initial_checkpoint=tmp_path / "missing.pt",
        supervised_dataset=tmp_path / "missing-dataset",
        algorithm=PPO_SEPARATED_ACTOR_CRITIC_ALGORITHM,
        simulation_backend="cpp",
        playing_observation_variant=COMPLETE_INFO_COMPACT_PLAYING_OBSERVATION_VARIANT,
        frozen_policy_onnx=frozen_onnx,
        frozen_policy_metadata=frozen_metadata,
    )

    _validate_config(config)

    invalid_algorithm = replace(config, algorithm=REINFORCE_ALGORITHM)
    with pytest.raises(
        PlayingRlOrchestratorError,
        match="complete-info compact playing observation variant requires algorithm",
    ):
        _validate_config(invalid_algorithm)


def test_playing_rl_full_diagnostics_interval_uses_output_generation(
    tmp_path: Path,
) -> None:
    config = PlayingRlRunConfig(
        run_directory=tmp_path / "run",
        initial_checkpoint=tmp_path / "missing.pt",
        supervised_dataset=tmp_path / "missing-dataset",
    )
    assert config.full_diagnostics_interval == DEFAULT_FULL_DIAGNOSTICS_INTERVAL
    assert not _full_diagnostics_is_due(config, generation=1)
    assert _full_diagnostics_is_due(config, generation=10)
    assert not _full_diagnostics_is_due(config, generation=19)
    assert _full_diagnostics_is_due(config, generation=20)

    twelve_iterations = replace(config, iterations=12)
    assert not _full_diagnostics_is_due(twelve_iterations, generation=12)

    every_iteration = replace(config, full_diagnostics_interval=1)
    assert _full_diagnostics_is_due(every_iteration, generation=1)
    assert _full_diagnostics_is_due(every_iteration, generation=2)

    invalid = replace(config, full_diagnostics_interval=0)
    with pytest.raises(PlayingRlOrchestratorError, match="full_diagnostics_interval"):
        _validate_config(invalid)


def test_new_playing_rl_run_default_evaluation_seed_count_is_400(
    tmp_path: Path,
) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--run-directory",
            str(tmp_path / "run"),
            "--initial-checkpoint",
            str(tmp_path / "initial.pt"),
            "--supervised-dataset",
            str(tmp_path / "supervised"),
        ]
    )
    config = _config_from_args(args, parser)

    assert DEFAULT_EVALUATION_SEED_COUNT == 400
    assert config.evaluation_seed_count == 400


def test_resume_keeps_stored_evaluation_seed_count_100_when_not_provided(
    tmp_path: Path,
) -> None:
    supervised_dataset = tmp_path / "supervised"
    supervised_dataset.mkdir()
    (supervised_dataset / "manifest.json").write_text("{}", encoding="utf-8")
    initial_checkpoint = tmp_path / "initial-playing.pt"
    initial_checkpoint.write_bytes(b"checkpoint")
    stored_run = PlayingRlRunConfig(
        run_directory=tmp_path / "run",
        initial_checkpoint=initial_checkpoint,
        supervised_dataset=supervised_dataset,
        evaluation_seed_count=100,
    ).normalized()
    requested_run = replace(stored_run, evaluation_seed_count=400)
    stored_config = stored_run.to_file_dict()
    requested_config = requested_run.to_file_dict()

    _validate_resume_config(
        stored_config=stored_config,
        requested_config=requested_config,
        provided_config_keys=set(),
    )
    loaded = _config_from_file_dict(stored_config, build_typescript=False, build_cpp=False)
    assert loaded.evaluation_seed_count == 100

    with pytest.raises(
        PlayingRlOrchestratorError,
        match="resume config mismatch for evaluationSeedCount",
    ):
        _validate_resume_config(
            stored_config=stored_config,
            requested_config=requested_config,
            provided_config_keys={"evaluationSeedCount"},
        )


def test_ppo_resume_fails_close_when_clip_epsilon_changes(tmp_path: Path) -> None:
    supervised_dataset = tmp_path / "supervised"
    supervised_dataset.mkdir()
    (supervised_dataset / "manifest.json").write_text("{}", encoding="utf-8")
    initial_checkpoint = tmp_path / "initial-playing.pt"
    initial_checkpoint.write_bytes(b"checkpoint")
    stored_run = PlayingRlRunConfig(
        run_directory=tmp_path / "run",
        initial_checkpoint=initial_checkpoint,
        supervised_dataset=supervised_dataset,
        algorithm=PPO_SEPARATED_ACTOR_CRITIC_ALGORITHM,
        ppo_clip_epsilon=DEFAULT_PPO_CLIP_EPSILON,
    ).normalized()
    requested_run = replace(stored_run, ppo_clip_epsilon=0.1)

    with pytest.raises(
        PlayingRlOrchestratorError,
        match="resume config mismatch for ppoClipEpsilon",
    ):
        _validate_resume_config(
            stored_config=stored_run.to_file_dict(),
            requested_config=requested_run.to_file_dict(),
            provided_config_keys=set(),
        )


def test_run_playing_rl_cli_parses_rollout_diagnostic_and_cache_options(
    tmp_path: Path,
) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--run-directory",
            str(tmp_path / "run"),
            "--initial-checkpoint",
            str(tmp_path / "initial.pt"),
            "--supervised-dataset",
            str(tmp_path / "supervised"),
            "--full-diagnostics-interval",
            "7",
            "--inference-device",
            "auto",
            "--rollout-concurrency",
            "64",
            "--inference-max-batch-size",
            "128",
            "--ppo-clip-epsilon",
            "0.15",
            "--retain-self-play-data",
            "--simulation-backend",
            "cpp",
            "--frozen-policy-onnx",
            str(tmp_path / "frozen.onnx"),
            "--frozen-policy-metadata",
            str(tmp_path / "frozen.json"),
            "--frozen-policy-artifact-id",
            "rl-v-test",
            "--no-cpp-build",
        ]
    )
    config = _config_from_args(args, parser)

    assert config.full_diagnostics_interval == 7
    assert config.inference_device == "auto"
    assert config.rollout_concurrency == 64
    assert config.inference_max_batch_size == 128
    assert config.ppo_clip_epsilon == pytest.approx(0.15)
    assert config.retain_self_play_data is True
    assert config.simulation_backend == "cpp"
    assert config.frozen_policy_onnx == tmp_path / "frozen.onnx"
    assert config.frozen_policy_metadata == tmp_path / "frozen.json"
    assert config.frozen_policy_artifact_id == "rl-v-test"
    assert config.build_cpp is False


def test_resume_config_reload_preserves_build_flags(tmp_path: Path) -> None:
    supervised_dataset = tmp_path / "supervised"
    supervised_dataset.mkdir()
    (supervised_dataset / "manifest.json").write_text("{}", encoding="utf-8")
    initial_checkpoint = tmp_path / "initial-playing.pt"
    initial_checkpoint.write_bytes(b"checkpoint")
    config = _small_config(
        run_directory=tmp_path / "run",
        initial_checkpoint=initial_checkpoint,
        supervised_dataset=supervised_dataset,
    )

    loaded = _config_from_file_dict(
        config.to_file_dict(),
        build_typescript=False,
        build_cpp=False,
    )

    assert loaded.build_typescript is False
    assert loaded.build_cpp is False


def _small_config(
    *,
    run_directory: Path,
    initial_checkpoint: Path,
    supervised_dataset: Path,
    algorithm: str = REINFORCE_ALGORITHM,
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
        rollout_workers=2,
        algorithm=algorithm,
        learning_rate=0.001,
        epochs=1,
        batch_size=16,
        full_diagnostics_interval=1,
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
        "playingEncoderSchemaVersion": 2,
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


def _required_str(value: object) -> str:
    assert isinstance(value, str)
    return value


def _required_int(value: object) -> int:
    assert isinstance(value, int)
    return value


def _required_float(value: object) -> float:
    assert isinstance(value, int | float)
    return float(value)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
