"""One-command non-playing RL rollout, PPO, ONNX export, and evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal, TypeVar, cast

from napoleon_ml.adjutant.model import AdjutantMlpConfig
from napoleon_ml.adjutant.ppo import AdjutantPpoTrainSettings, train_adjutant_ppo
from napoleon_ml.bidding.model import BiddingMlpConfig
from napoleon_ml.bidding.ppo import BiddingPpoTrainSettings, train_bidding_ppo
from napoleon_ml.exchange.model import ExchangeMlpConfig
from napoleon_ml.exchange.ppo import ExchangePpoTrainSettings, train_exchange_ppo
from napoleon_ml.nonplaying_onnx_export import (
    export_adjutant_rl_checkpoint_to_onnx,
    export_bidding_rl_checkpoint_to_onnx,
    export_exchange_rl_checkpoint_to_onnx,
    export_seeded_nonplaying_bootstrap_policy_to_onnx,
)

DEFAULT_GAMES = 20
DEFAULT_EVALUATION_GAMES = 5
DEFAULT_GAMES_PER_SHARD = 20
DEFAULT_ITERATIONS = 100
DEFAULT_GAMES_PER_ITERATION = 200
DEFAULT_ITERATIVE_EVALUATION_GAMES = 100
DEFAULT_EVALUATION_INTERVAL = 10
DEFAULT_EPOCHS = 1
DEFAULT_ITERATIVE_EPOCHS = 4
DEFAULT_BATCH_SIZE = 32
DEFAULT_ITERATIVE_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_ITERATIVE_LEARNING_RATE = 1e-4
DEFAULT_HIDDEN_DIM = 128
DEFAULT_HIDDEN_LAYERS = 2
DEFAULT_DROPOUT = 0.0
DEFAULT_PPO_CLIP_EPSILON = 0.2
DEFAULT_VALUE_LOSS_COEFFICIENT = 0.5
DEFAULT_TEMPERATURE = 1.0
DEFAULT_INFERENCE_DEVICE: Literal["cpu", "auto", "cuda"] = "cpu"
DEFAULT_INFERENCE_MAX_BATCH_SIZE = 256
DEFAULT_SEED = 202
ITERATIVE_RUN_CONFIG_SCHEMA_VERSION = 4
NONPLAYING_ROLLOUT_POLICY_TOPOLOGY = "candidate-x1-frozen-x4-v1"
NONPLAYING_GAME_COUNT_UNIT = "logical-seeds"
NONPLAYING_ROTATION_OFFSETS = [0, 1, 2, 3, 4]
NONPLAYING_REWARD_TYPE = "non-playing-terminal-role-reward"
NONPLAYING_REWARD_VERSION = 3
NONPLAYING_REWARD_ID = "non-playing-terminal-role-reward-v3"
FROZEN_BIDDING_OPPONENT_MIX_RULE_VERSION = (
    "per-seat-seeded-conservative-passive-50-50-v1"
)
CONSERVATIVE_BIDDING_BASELINE_ID = "conservative-bidding-v1"
PASSIVE_BIDDING_BASELINE_ID = "passive-bidding-v1"
ITERATION_SEED_STRIDE = 1_000_000
PHASE_SEED_STRIDE = 100_000
EVALUATION_SEED_OFFSET = 300_000
DEFAULT_PLAYING_POLICY_ARTIFACT_ID = "ppo-separated-v1000"
DEFAULT_PLAYING_POLICY_ONNX = Path("benchmarks/playing-policies/ppo-separated-v1000/policy.onnx")
DEFAULT_PLAYING_POLICY_METADATA = Path(
    "benchmarks/playing-policies/ppo-separated-v1000/policy.json"
)
SUPPORTED_INFERENCE_DEVICES = ("cpu", "auto", "cuda")

PhaseName = Literal["bidding", "adjutant", "exchange"]
PHASES: tuple[PhaseName, ...] = ("bidding", "adjutant", "exchange")
_T = TypeVar("_T")


class NonPlayingRlOrchestratorError(RuntimeError):
    """Raised when the non-playing RL pipeline cannot continue safely."""


@dataclass(frozen=True)
class NonPlayingRlRunConfig:
    output_dir: Path
    games: int = DEFAULT_GAMES
    evaluation_games: int = DEFAULT_EVALUATION_GAMES
    games_per_shard: int | None = None
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    hidden_layers: int = DEFAULT_HIDDEN_LAYERS
    dropout: float = DEFAULT_DROPOUT
    ppo_clip_epsilon: float = DEFAULT_PPO_CLIP_EPSILON
    value_loss_coefficient: float = DEFAULT_VALUE_LOSS_COEFFICIENT
    seed: int = DEFAULT_SEED
    temperature: float = DEFAULT_TEMPERATURE
    inference_device: Literal["cpu", "auto", "cuda"] = DEFAULT_INFERENCE_DEVICE
    inference_max_batch_size: int = DEFAULT_INFERENCE_MAX_BATCH_SIZE
    playing_policy_onnx: Path = DEFAULT_PLAYING_POLICY_ONNX
    playing_policy_metadata: Path = DEFAULT_PLAYING_POLICY_METADATA
    playing_policy_artifact_id: str = DEFAULT_PLAYING_POLICY_ARTIFACT_ID
    build_typescript: bool = True
    overwrite: bool = False

    def normalized(self) -> NonPlayingRlRunConfig:
        return NonPlayingRlRunConfig(
            output_dir=self.output_dir.expanduser().resolve(),
            games=self.games,
            evaluation_games=self.evaluation_games,
            games_per_shard=self.games_per_shard,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            ppo_clip_epsilon=self.ppo_clip_epsilon,
            value_loss_coefficient=self.value_loss_coefficient,
            seed=self.seed,
            temperature=self.temperature,
            inference_device=self.inference_device,
            inference_max_batch_size=self.inference_max_batch_size,
            playing_policy_onnx=_resolve_repo_path(self.playing_policy_onnx),
            playing_policy_metadata=_resolve_repo_path(self.playing_policy_metadata),
            playing_policy_artifact_id=self.playing_policy_artifact_id,
            build_typescript=self.build_typescript,
            overwrite=self.overwrite,
        )

    @property
    def effective_games_per_shard(self) -> int:
        return self.games_per_shard if self.games_per_shard is not None else self.games

    def settings_dict(self) -> dict[str, object]:
        return {
            "games": self.games,
            "evaluationGames": self.evaluation_games,
            "gamesPerShard": self.effective_games_per_shard,
            "epochs": self.epochs,
            "batchSize": self.batch_size,
            "learningRate": self.learning_rate,
            "hiddenDim": self.hidden_dim,
            "hiddenLayers": self.hidden_layers,
            "dropout": self.dropout,
            "ppoClipEpsilon": self.ppo_clip_epsilon,
            "valueLossCoefficient": self.value_loss_coefficient,
            "seed": self.seed,
            "temperature": self.temperature,
            "reward": {
                "type": NONPLAYING_REWARD_TYPE,
                "version": NONPLAYING_REWARD_VERSION,
                "id": NONPLAYING_REWARD_ID,
            },
            "inferenceDevice": self.inference_device,
            "inferenceMaxBatchSize": self.inference_max_batch_size,
            "playingPolicyOnnx": str(self.playing_policy_onnx),
            "playingPolicyMetadata": str(self.playing_policy_metadata),
            "playingPolicyArtifactId": self.playing_policy_artifact_id,
        }


@dataclass(frozen=True)
class NonPlayingIterativeRlRunConfig:
    output_dir: Path
    iterations: int = DEFAULT_ITERATIONS
    games_per_iteration: int = DEFAULT_GAMES_PER_ITERATION
    evaluation_interval: int = DEFAULT_EVALUATION_INTERVAL
    evaluation_games: int = DEFAULT_ITERATIVE_EVALUATION_GAMES
    games_per_shard: int | None = DEFAULT_GAMES_PER_SHARD
    epochs: int = DEFAULT_ITERATIVE_EPOCHS
    batch_size: int = DEFAULT_ITERATIVE_BATCH_SIZE
    learning_rate: float = DEFAULT_ITERATIVE_LEARNING_RATE
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    hidden_layers: int = DEFAULT_HIDDEN_LAYERS
    dropout: float = DEFAULT_DROPOUT
    ppo_clip_epsilon: float = DEFAULT_PPO_CLIP_EPSILON
    value_loss_coefficient: float = DEFAULT_VALUE_LOSS_COEFFICIENT
    seed: int = DEFAULT_SEED
    temperature: float = DEFAULT_TEMPERATURE
    inference_device: Literal["cpu", "auto", "cuda"] = DEFAULT_INFERENCE_DEVICE
    inference_max_batch_size: int = DEFAULT_INFERENCE_MAX_BATCH_SIZE
    playing_policy_onnx: Path = DEFAULT_PLAYING_POLICY_ONNX
    playing_policy_metadata: Path = DEFAULT_PLAYING_POLICY_METADATA
    playing_policy_artifact_id: str = DEFAULT_PLAYING_POLICY_ARTIFACT_ID
    build_typescript: bool = True
    overwrite: bool = False

    def normalized(self) -> NonPlayingIterativeRlRunConfig:
        return NonPlayingIterativeRlRunConfig(
            output_dir=self.output_dir.expanduser().resolve(),
            iterations=self.iterations,
            games_per_iteration=self.games_per_iteration,
            evaluation_interval=self.evaluation_interval,
            evaluation_games=self.evaluation_games,
            games_per_shard=self.games_per_shard,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            ppo_clip_epsilon=self.ppo_clip_epsilon,
            value_loss_coefficient=self.value_loss_coefficient,
            seed=self.seed,
            temperature=self.temperature,
            inference_device=self.inference_device,
            inference_max_batch_size=self.inference_max_batch_size,
            playing_policy_onnx=_resolve_repo_path(self.playing_policy_onnx),
            playing_policy_metadata=_resolve_repo_path(self.playing_policy_metadata),
            playing_policy_artifact_id=self.playing_policy_artifact_id,
            build_typescript=self.build_typescript,
            overwrite=self.overwrite,
        )

    @property
    def effective_games_per_shard(self) -> int:
        if self.games_per_shard is None:
            return self.games_per_iteration
        return self.games_per_shard

    def as_one_shot_config(self) -> NonPlayingRlRunConfig:
        return NonPlayingRlRunConfig(
            output_dir=self.output_dir,
            games=self.games_per_iteration,
            evaluation_games=self.evaluation_games,
            games_per_shard=self.games_per_shard,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            ppo_clip_epsilon=self.ppo_clip_epsilon,
            value_loss_coefficient=self.value_loss_coefficient,
            seed=self.seed,
            temperature=self.temperature,
            inference_device=self.inference_device,
            inference_max_batch_size=self.inference_max_batch_size,
            playing_policy_onnx=self.playing_policy_onnx,
            playing_policy_metadata=self.playing_policy_metadata,
            playing_policy_artifact_id=self.playing_policy_artifact_id,
            build_typescript=self.build_typescript,
            overwrite=self.overwrite,
        )

    def file_dict(self) -> dict[str, object]:
        return {
            "schemaVersion": ITERATIVE_RUN_CONFIG_SCHEMA_VERSION,
            "runType": "non-playing-iterative-ppo",
            "outputDir": str(self.output_dir),
            "iterations": self.iterations,
            "gamesPerIteration": self.games_per_iteration,
            "gamesPerIterationUnit": NONPLAYING_GAME_COUNT_UNIT,
            "actualGamesPerIteration": (
                self.games_per_iteration * len(NONPLAYING_ROTATION_OFFSETS)
            ),
            "rolloutPolicyTopology": NONPLAYING_ROLLOUT_POLICY_TOPOLOGY,
            "rotationOffsets": NONPLAYING_ROTATION_OFFSETS,
            "reward": {
                "type": NONPLAYING_REWARD_TYPE,
                "version": NONPLAYING_REWARD_VERSION,
                "id": NONPLAYING_REWARD_ID,
            },
            "biddingFrozenOpponentMixRuleVersion": FROZEN_BIDDING_OPPONENT_MIX_RULE_VERSION,
            "biddingFrozenOpponentPolicyIds": {
                "conservative": CONSERVATIVE_BIDDING_BASELINE_ID,
                "passive": PASSIVE_BIDDING_BASELINE_ID,
            },
            "gamesPerShard": self.effective_games_per_shard,
            "evaluationInterval": self.evaluation_interval,
            "evaluationGames": self.evaluation_games,
            "epochs": self.epochs,
            "batchSize": self.batch_size,
            "learningRate": self.learning_rate,
            "hiddenDim": self.hidden_dim,
            "hiddenLayers": self.hidden_layers,
            "dropout": self.dropout,
            "ppoClipEpsilon": self.ppo_clip_epsilon,
            "valueLossCoefficient": self.value_loss_coefficient,
            "seed": self.seed,
            "temperature": self.temperature,
            "inferenceDevice": self.inference_device,
            "inferenceMaxBatchSize": self.inference_max_batch_size,
            "playingPolicyOnnx": str(self.playing_policy_onnx),
            "playingPolicyOnnxSha256": _sha256_file(self.playing_policy_onnx),
            "playingPolicyMetadata": str(self.playing_policy_metadata),
            "playingPolicyMetadataSha256": _sha256_file(self.playing_policy_metadata),
            "playingPolicyArtifactId": self.playing_policy_artifact_id,
        }


def run_nonplaying_rl_pipeline(config: NonPlayingRlRunConfig) -> dict[str, object]:
    config = config.normalized()
    _validate_config(config)
    _prepare_output_dir(config)
    if config.build_typescript:
        _stage("typescript-build", _build_typescript_helpers)

    started = time.monotonic()
    summary: dict[str, object] = {
        "schemaVersion": 1,
        "runType": "non-playing-rl-smoke",
        "settings": config.settings_dict(),
        "artifacts": {
            "playing": {
                "onnxPath": str(config.playing_policy_onnx),
                "metadataPath": str(config.playing_policy_metadata),
            }
        },
        "phases": {},
    }
    final_artifacts: dict[PhaseName, dict[str, str]] = {}

    for offset, phase in enumerate(("bidding", "adjutant", "exchange")):
        phase_summary, artifact = _run_phase(config, cast(PhaseName, phase), offset)
        cast(dict[str, object], summary["phases"])[phase] = phase_summary
        final_artifacts[cast(PhaseName, phase)] = artifact

    evaluation_path = config.output_dir / "evaluation.json"
    evaluation_summary = _stage(
        "full-policy-evaluation",
        lambda: _run_full_policy_evaluation(config, final_artifacts, evaluation_path),
    )
    summary["evaluation"] = evaluation_summary
    summary["artifactPaths"] = {
        "bidding": final_artifacts["bidding"],
        "adjutant": final_artifacts["adjutant"],
        "exchange": final_artifacts["exchange"],
        "evaluation": str(evaluation_path),
        "runSummary": str(config.output_dir / "run-summary.json"),
    }
    summary["completedAtUnixSeconds"] = int(time.time())
    summary["elapsedSeconds"] = time.monotonic() - started
    _atomic_write_json(config.output_dir / "run-summary.json", summary)
    _print_completion(summary)
    return summary


def run_iterative_nonplaying_rl_pipeline(
    config: NonPlayingIterativeRlRunConfig,
    *,
    resume: bool = False,
    stop_after_iterations: int | None = None,
    provided_config_keys: Sequence[str] = (),
) -> dict[str, object]:
    config = config.normalized()
    _validate_iterative_config(config)
    file_config = config.file_dict()
    config_path = config.output_dir / "config.json"

    if resume:
        stored_config = _load_json_object(config_path)
        _validate_iterative_resume_config(
            stored_config,
            file_config,
            provided_config_keys=set(provided_config_keys),
        )
        config = _iterative_config_from_file_dict(
            stored_config,
            build_typescript=config.build_typescript,
        )
        file_config = stored_config
        _validate_iterative_config(config)
    else:
        _prepare_iterative_output_dir(config)
        _atomic_write_json(config_path, file_config)

    if config.build_typescript:
        _stage("typescript-build", _build_typescript_helpers)

    started = time.monotonic()
    bootstrap_artifacts = _ensure_bootstrap_artifacts(config)
    next_iteration = _next_nonplaying_iteration(config, bootstrap_artifacts)
    state = _load_iterative_state(config.output_dir)
    state = _write_iterative_state(
        config,
        {
            **state,
            "completedIterationCount": next_iteration,
            "latestCompletedIteration": next_iteration - 1,
        },
    )
    stop_at = config.iterations
    if stop_after_iterations is not None:
        stop_at = min(config.iterations, next_iteration + stop_after_iterations)

    for iteration in range(next_iteration, stop_at):
        iteration_record = _run_iterative_iteration(config, iteration, bootstrap_artifacts)
        state = _write_iterative_state(
            config,
            {
                **state,
                "completedIterationCount": iteration + 1,
                "latestCompletedIteration": iteration,
                "latestArtifacts": iteration_record["artifacts"],
                "completedEvaluations": _completed_evaluation_iterations(config),
            },
        )
        _append_jsonl(
            config.output_dir / "run-summary.jsonl",
            _iteration_summary_line(iteration_record),
        )

    completed_iteration_count = _next_nonplaying_iteration(config, bootstrap_artifacts)
    summary = _write_iterative_run_summary(
        config,
        file_config,
        completed_iteration_count=completed_iteration_count,
        elapsed_seconds=time.monotonic() - started,
        state=state,
    )
    _print_iterative_completion(summary)
    return summary


def _run_iterative_iteration(
    config: NonPlayingIterativeRlRunConfig,
    iteration: int,
    bootstrap_artifacts: dict[PhaseName, dict[str, str]],
) -> dict[str, object]:
    total = config.iterations
    iteration_dir = _iterative_iteration_dir(config, iteration)
    completed = _load_completed_nonplaying_iteration(iteration_dir)
    if completed is not None:
        _validate_nonplaying_iteration_artifacts(completed)
        return completed
    if iteration_dir.exists():
        _quarantine_incomplete_directory(iteration_dir, label="iteration")
    iteration_dir.mkdir(parents=True)

    phase_records: dict[str, object] = {}
    final_artifacts: dict[PhaseName, dict[str, str]] = {}
    started = time.monotonic()

    print(f"[iter {iteration + 1}/{total}] starting", flush=True)
    for offset, phase_name in enumerate(("bidding", "adjutant", "exchange")):
        phase = cast(PhaseName, phase_name)
        phase_dir = iteration_dir / phase
        dataset_dir = phase_dir / "dataset"
        input_checkpoint = _iterative_input_checkpoint(config, iteration, phase)
        behavior_artifact = _iterative_behavior_artifact(
            config,
            iteration,
            phase,
            bootstrap_artifacts,
        )
        checkpoint_path = phase_dir / "output-checkpoint.pt"
        onnx_path = phase_dir / "policy.onnx"
        metadata_path = phase_dir / "policy.json"
        rollout_seed = _iterative_rollout_seed(config, iteration, offset)
        training_seed = _iterative_training_seed(config, iteration, offset)

        print(
            f"[iter {iteration + 1}/{total}] {phase} rollout "
            f"logical_seeds={config.games_per_iteration} "
            f"actual_games={config.games_per_iteration * len(NONPLAYING_ROTATION_OFFSETS)} "
            f"start_seed={rollout_seed}",
            flush=True,
        )
        rollout_summary = _stage(
            f"iter-{iteration:06d}-{phase}-rollout",
            partial(
                _run_iterative_rollout_stage,
                config,
                phase=phase,
                behavior_artifact=behavior_artifact,
                dataset_dir=dataset_dir,
                rollout_seed=rollout_seed,
                progress_prefix=f"[iter {iteration + 1}/{total}] {phase} rollout ",
            ),
        )
        manifest_path = dataset_dir / "manifest.json"
        behavior_onnx_sha256 = _sha256_file(Path(behavior_artifact["onnxPath"]))
        behavior_metadata_sha256 = _sha256_file(Path(behavior_artifact["metadataPath"]))
        _validate_nonplaying_rollout_manifest(
            manifest_path,
            phase=phase,
            expected_start_seed=rollout_seed,
            expected_game_count=config.games_per_iteration,
            expected_behavior_onnx_sha256=behavior_onnx_sha256,
            expected_behavior_metadata_sha256=behavior_metadata_sha256,
            expected_playing_artifact_id=config.playing_policy_artifact_id,
        )

        print(f"[iter {iteration + 1}/{total}] {phase} train", flush=True)
        train_report = _stage(
            f"iter-{iteration:06d}-{phase}-train",
            partial(
                _train_iterative_phase,
                config,
                phase=phase,
                dataset_dir=dataset_dir,
                checkpoint_path=checkpoint_path,
                training_seed=training_seed,
                input_checkpoint=input_checkpoint,
            ),
        )
        print(f"[iter {iteration + 1}/{total}] {phase} export", flush=True)
        export_report = _stage(
            f"iter-{iteration:06d}-{phase}-export",
            partial(
                _export_iterative_phase,
                phase=phase,
                dataset_dir=dataset_dir,
                checkpoint_path=checkpoint_path,
                onnx_path=onnx_path,
                metadata_path=metadata_path,
            ),
        )

        phase_artifact = {
            "checkpointPath": str(checkpoint_path),
            "checkpointSha256": _sha256_file(checkpoint_path),
            "onnxPath": str(onnx_path),
            "onnxSha256": _sha256_file(onnx_path),
            "metadataPath": str(metadata_path),
            "metadataSha256": _sha256_file(metadata_path),
            "artifactId": f"{phase}-iter-{iteration + 1:06d}",
        }
        final_artifacts[phase] = {
            "checkpointPath": str(checkpoint_path),
            "onnxPath": str(onnx_path),
            "metadataPath": str(metadata_path),
        }
        phase_records[phase] = {
            "datasetDirectory": str(dataset_dir),
            "manifestPath": str(manifest_path),
            "manifestSha256": _sha256_file(manifest_path),
            "rollout": rollout_summary,
            "rolloutStartSeed": rollout_seed,
            "trainingSeed": training_seed,
            "inputCheckpointPath": str(input_checkpoint) if input_checkpoint is not None else None,
            "inputCheckpointSha256": (
                _sha256_file(input_checkpoint) if input_checkpoint is not None else None
            ),
            "behaviorArtifact": {
                **behavior_artifact,
                "onnxSha256": behavior_onnx_sha256,
                "metadataSha256": behavior_metadata_sha256,
            },
            "train": train_report,
            "export": export_report,
            "artifact": phase_artifact,
        }
        print(
            f"[iter {iteration + 1}/{total}] {phase} complete "
            f"samples={rollout_summary.get('sampleCount')} "
            f"mean_reward={train_report.get('meanReward')} "
            f"loss={train_report.get('meanTotalLoss')} "
            f"value_loss={train_report.get('meanValueLoss')} "
            f"clipped={train_report.get('clippedFraction')}",
            flush=True,
        )

    evaluation_summary: dict[str, object] | None = None
    evaluation_path: str | None = None
    if _evaluation_is_due(config, iteration):
        evaluation_file = iteration_dir / "evaluation.json"
        evaluation_path = str(evaluation_file)
        print(
            f"[iter {iteration + 1}/{total}] full-policy evaluation "
            f"games={config.evaluation_games}",
            flush=True,
        )
        evaluation_summary = _stage(
            f"iter-{iteration:06d}-full-policy-evaluation",
            lambda: _run_full_policy_evaluation(
                config.as_one_shot_config(),
                final_artifacts,
                evaluation_file,
                start_seed=_iterative_evaluation_seed(config, iteration),
            ),
        )
        print(
            f"[iter {iteration + 1}/{total}] evaluation complete "
            f"completed={evaluation_summary.get('completedGames')}/"
            f"{evaluation_summary.get('scheduledGames')} "
            f"fallback={evaluation_summary.get('fallbackCount')} "
            f"illegal={evaluation_summary.get('illegalActionCount')}",
            flush=True,
        )

    iteration_record: dict[str, object] = {
        "schemaVersion": 1,
        "completionState": "completed",
        "iteration": iteration,
        "iterationIndexWidth": 6,
        "phaseOrder": ["bidding", "adjutant", "exchange"],
        "gamesPerIteration": config.games_per_iteration,
        "gamesPerIterationUnit": NONPLAYING_GAME_COUNT_UNIT,
        "actualGamesPerIteration": config.games_per_iteration * len(NONPLAYING_ROTATION_OFFSETS),
        "rolloutPolicyTopology": NONPLAYING_ROLLOUT_POLICY_TOPOLOGY,
        "rotationOffsets": NONPLAYING_ROTATION_OFFSETS,
        "reward": {
            "type": NONPLAYING_REWARD_TYPE,
            "version": NONPLAYING_REWARD_VERSION,
            "id": NONPLAYING_REWARD_ID,
        },
        "evaluationDue": evaluation_summary is not None,
        "evaluation": evaluation_summary,
        "evaluationPath": evaluation_path,
        "phases": phase_records,
        "artifacts": final_artifacts,
        "completedAtUnixSeconds": int(time.time()),
        "elapsedSeconds": time.monotonic() - started,
    }
    _atomic_write_json(iteration_dir / "iteration.json", iteration_record)
    _validate_nonplaying_iteration_artifacts(iteration_record)
    _update_latest_links(config, iteration)
    return iteration_record


def _run_iterative_rollout_stage(
    config: NonPlayingIterativeRlRunConfig,
    *,
    phase: PhaseName,
    behavior_artifact: dict[str, str],
    dataset_dir: Path,
    rollout_seed: int,
    progress_prefix: str,
) -> dict[str, object]:
    return _run_nonplaying_rollout(
        config.as_one_shot_config(),
        phase=phase,
        policy_onnx=Path(behavior_artifact["onnxPath"]),
        policy_metadata=Path(behavior_artifact["metadataPath"]),
        dataset_dir=dataset_dir,
        start_seed=rollout_seed,
        artifact_id=behavior_artifact["artifactId"],
        progress_prefix=progress_prefix,
    )


def _train_iterative_phase(
    config: NonPlayingIterativeRlRunConfig,
    *,
    phase: PhaseName,
    dataset_dir: Path,
    checkpoint_path: Path,
    training_seed: int,
    input_checkpoint: Path | None,
) -> dict[str, object]:
    return _train_phase(
        config.as_one_shot_config(),
        phase,
        dataset_dir,
        checkpoint_path,
        training_seed,
        parent_checkpoint=input_checkpoint,
    )


def _export_iterative_phase(
    *,
    phase: PhaseName,
    dataset_dir: Path,
    checkpoint_path: Path,
    onnx_path: Path,
    metadata_path: Path,
) -> dict[str, object]:
    return _export_trained_phase(
        phase,
        dataset_dir,
        checkpoint_path,
        onnx_path,
        metadata_path,
    )


def _run_phase(
    config: NonPlayingRlRunConfig,
    phase: PhaseName,
    offset: int,
) -> tuple[dict[str, object], dict[str, str]]:
    phase_dir = config.output_dir / phase
    dataset_dir = phase_dir / "dataset"
    checkpoint_path = phase_dir / "checkpoint.pt"
    onnx_path = phase_dir / "policy.onnx"
    metadata_path = phase_dir / "policy.json"
    bootstrap_dir = phase_dir / "bootstrap"
    bootstrap_onnx = bootstrap_dir / "policy.onnx"
    bootstrap_metadata = bootstrap_dir / "policy.json"
    rollout_seed = config.seed + offset * 100_000
    training_seed = config.seed + offset * 100_000 + 1

    bootstrap_report = _stage(
        f"{phase}-bootstrap-export",
        lambda: export_seeded_nonplaying_bootstrap_policy_to_onnx(
            policy_type=phase,
            onnx_path=bootstrap_onnx,
            metadata_path=bootstrap_metadata,
            seed=training_seed,
            hidden_dim=config.hidden_dim,
            hidden_layers=config.hidden_layers,
            dropout=config.dropout,
        ),
    )
    rollout_summary = _stage(
        f"{phase}-rollout",
        lambda: _run_nonplaying_rollout(
            config,
            phase=phase,
            policy_onnx=bootstrap_onnx,
            policy_metadata=bootstrap_metadata,
            dataset_dir=dataset_dir,
            start_seed=rollout_seed,
        ),
    )
    train_report = _stage(
        f"{phase}-train",
        lambda: _train_phase(config, phase, dataset_dir, checkpoint_path, training_seed),
    )
    export_report = _stage(
        f"{phase}-export",
        lambda: _export_trained_phase(
            phase,
            dataset_dir,
            checkpoint_path,
            onnx_path,
            metadata_path,
        ),
    )

    artifact = {
        "checkpointPath": str(checkpoint_path),
        "onnxPath": str(onnx_path),
        "metadataPath": str(metadata_path),
    }
    return (
        {
            "bootstrap": bootstrap_report,
            "rollout": rollout_summary,
            "train": train_report,
            "export": export_report,
            "artifact": artifact,
        },
        artifact,
    )


def _run_nonplaying_rollout(
    config: NonPlayingRlRunConfig,
    *,
    phase: PhaseName,
    policy_onnx: Path,
    policy_metadata: Path,
    dataset_dir: Path,
    start_seed: int,
    artifact_id: str | None = None,
    progress_prefix: str | None = None,
) -> dict[str, object]:
    return _run_node_json(
        [
            "node",
            str(_repo_root() / "apps/self-play-cli/dist/index.js"),
            "non-playing-rollout",
            "--phase",
            phase,
            "--policy-onnx",
            str(policy_onnx),
            "--policy-metadata",
            str(policy_metadata),
            "--playing-onnx",
            str(config.playing_policy_onnx),
            "--playing-metadata",
            str(config.playing_policy_metadata),
            "--output",
            str(dataset_dir),
            "--start-seed",
            str(start_seed),
            "--games",
            str(config.games),
            "--games-per-shard",
            str(config.effective_games_per_shard),
            "--temperature",
            repr(config.temperature),
            "--inference-device",
            config.inference_device,
            "--inference-max-batch-size",
            str(config.inference_max_batch_size),
            "--artifact-id",
            artifact_id if artifact_id is not None else f"{phase}-bootstrap-seed-{start_seed}",
            "--playing-artifact-id",
            config.playing_policy_artifact_id,
            "--progress-prefix",
            progress_prefix if progress_prefix is not None else f"[{phase} rollout] ",
        ],
        cwd=_repo_root(),
    )


def _train_phase(
    config: NonPlayingRlRunConfig,
    phase: PhaseName,
    dataset_dir: Path,
    checkpoint_path: Path,
    training_seed: int,
    parent_checkpoint: Path | None = None,
) -> dict[str, object]:
    if phase == "bidding":
        return train_bidding_ppo(
            dataset_directory=dataset_dir,
            output_checkpoint_path=checkpoint_path,
            settings=BiddingPpoTrainSettings(
                seed=training_seed,
                epochs=config.epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                ppo_clip_epsilon=config.ppo_clip_epsilon,
                value_loss_coefficient=config.value_loss_coefficient,
                parent_actor_checkpoint=None,
                parent_checkpoint=str(parent_checkpoint) if parent_checkpoint is not None else None,
            ),
            model_config=BiddingMlpConfig(
                hidden_dim=config.hidden_dim,
                hidden_layers=config.hidden_layers,
                dropout=config.dropout,
            ),
        ).to_dict()
    if phase == "adjutant":
        return train_adjutant_ppo(
            dataset_directory=dataset_dir,
            output_checkpoint_path=checkpoint_path,
            settings=AdjutantPpoTrainSettings(
                seed=training_seed,
                epochs=config.epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                ppo_clip_epsilon=config.ppo_clip_epsilon,
                value_loss_coefficient=config.value_loss_coefficient,
                parent_actor_checkpoint=None,
                parent_checkpoint=str(parent_checkpoint) if parent_checkpoint is not None else None,
            ),
            model_config=AdjutantMlpConfig(
                hidden_dim=config.hidden_dim,
                hidden_layers=config.hidden_layers,
                dropout=config.dropout,
            ),
        ).to_dict()
    return train_exchange_ppo(
        dataset_directory=dataset_dir,
        output_checkpoint_path=checkpoint_path,
        settings=ExchangePpoTrainSettings(
            seed=training_seed,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            ppo_clip_epsilon=config.ppo_clip_epsilon,
            value_loss_coefficient=config.value_loss_coefficient,
            parent_actor_checkpoint=None,
            parent_checkpoint=str(parent_checkpoint) if parent_checkpoint is not None else None,
        ),
        model_config=ExchangeMlpConfig(
            hidden_dim=config.hidden_dim,
            hidden_layers=config.hidden_layers,
            dropout=config.dropout,
        ),
    ).to_dict()


def _export_trained_phase(
    phase: PhaseName,
    dataset_dir: Path,
    checkpoint_path: Path,
    onnx_path: Path,
    metadata_path: Path,
) -> dict[str, object]:
    if phase == "bidding":
        report = export_bidding_rl_checkpoint_to_onnx(
            dataset_directory=dataset_dir,
            checkpoint_path=checkpoint_path,
            onnx_path=onnx_path,
            metadata_path=metadata_path,
        )
    elif phase == "adjutant":
        report = export_adjutant_rl_checkpoint_to_onnx(
            dataset_directory=dataset_dir,
            checkpoint_path=checkpoint_path,
            onnx_path=onnx_path,
            metadata_path=metadata_path,
        )
    else:
        report = export_exchange_rl_checkpoint_to_onnx(
            dataset_directory=dataset_dir,
            checkpoint_path=checkpoint_path,
            onnx_path=onnx_path,
            metadata_path=metadata_path,
        )
    return report.to_dict()


def _run_full_policy_evaluation(
    config: NonPlayingRlRunConfig,
    artifacts: dict[PhaseName, dict[str, str]],
    evaluation_path: Path,
    start_seed: int | None = None,
) -> dict[str, object]:
    return _run_node_json(
        [
            "node",
            str(_repo_root() / "apps/self-play-cli/dist/index.js"),
            "full-policy-evaluate",
            "--playing-onnx",
            str(config.playing_policy_onnx),
            "--playing-metadata",
            str(config.playing_policy_metadata),
            "--bidding-onnx",
            artifacts["bidding"]["onnxPath"],
            "--bidding-metadata",
            artifacts["bidding"]["metadataPath"],
            "--adjutant-onnx",
            artifacts["adjutant"]["onnxPath"],
            "--adjutant-metadata",
            artifacts["adjutant"]["metadataPath"],
            "--exchange-onnx",
            artifacts["exchange"]["onnxPath"],
            "--exchange-metadata",
            artifacts["exchange"]["metadataPath"],
            "--output",
            str(evaluation_path),
            "--start-seed",
            str(start_seed if start_seed is not None else config.seed + EVALUATION_SEED_OFFSET),
            "--games",
            str(config.evaluation_games),
            "--inference-device",
            config.inference_device,
            "--inference-max-batch-size",
            str(config.inference_max_batch_size),
            "--progress-prefix",
            "[full-policy eval] ",
        ],
        cwd=_repo_root(),
    )


def _ensure_bootstrap_artifacts(
    config: NonPlayingIterativeRlRunConfig,
) -> dict[PhaseName, dict[str, str]]:
    bootstrap_artifacts: dict[PhaseName, dict[str, str]] = {}
    for offset, phase_name in enumerate(("bidding", "adjutant", "exchange")):
        phase = cast(PhaseName, phase_name)
        bootstrap_dir = config.output_dir / "bootstrap" / phase
        onnx_path = bootstrap_dir / "policy.onnx"
        metadata_path = bootstrap_dir / "policy.json"
        if not onnx_path.exists() or not metadata_path.exists():
            _stage(
                f"{phase}-bootstrap-export",
                partial(
                    _export_iterative_bootstrap,
                    config,
                    phase=phase,
                    onnx_path=onnx_path,
                    metadata_path=metadata_path,
                    phase_offset=offset,
                ),
            )
        _ensure_file(onnx_path, f"{phase} bootstrap ONNX")
        _ensure_file(metadata_path, f"{phase} bootstrap metadata")
        bootstrap_artifacts[phase] = {
            "onnxPath": str(onnx_path),
            "metadataPath": str(metadata_path),
            "artifactId": f"{phase}-bootstrap-seed-{config.seed}",
            "provenance": "bootstrap",
        }
    return bootstrap_artifacts


def _export_iterative_bootstrap(
    config: NonPlayingIterativeRlRunConfig,
    *,
    phase: PhaseName,
    onnx_path: Path,
    metadata_path: Path,
    phase_offset: int,
) -> dict[str, object]:
    return export_seeded_nonplaying_bootstrap_policy_to_onnx(
        policy_type=phase,
        onnx_path=onnx_path,
        metadata_path=metadata_path,
        seed=config.seed + phase_offset * PHASE_SEED_STRIDE + 1,
        hidden_dim=config.hidden_dim,
        hidden_layers=config.hidden_layers,
        dropout=config.dropout,
    )


def _next_nonplaying_iteration(
    config: NonPlayingIterativeRlRunConfig,
    bootstrap_artifacts: dict[PhaseName, dict[str, str]],
) -> int:
    next_index = 0
    expected_inputs: dict[PhaseName, str | None] = {
        "bidding": None,
        "adjutant": None,
        "exchange": None,
    }
    expected_behaviors: dict[PhaseName, str] = {
        phase: _sha256_file(Path(artifact["onnxPath"]))
        for phase, artifact in bootstrap_artifacts.items()
    }
    for iteration in range(config.iterations):
        iteration_dir = _iterative_iteration_dir(config, iteration)
        record = _load_completed_nonplaying_iteration(iteration_dir)
        if record is None:
            break
        _validate_nonplaying_iteration_artifacts(record)
        phases = _require_dict(record.get("phases"), f"{iteration_dir}/iteration.json.phases")
        for phase in PHASES:
            phase_record = _require_dict(phases.get(phase), f"{phase}.record")
            input_sha = phase_record.get("inputCheckpointSha256")
            if input_sha != expected_inputs[phase]:
                raise NonPlayingRlOrchestratorError(
                    f"{iteration_dir}: {phase} input checkpoint chain mismatch: "
                    f"{input_sha!r} != {expected_inputs[phase]!r}"
                )
            behavior = _require_dict(
                phase_record.get("behaviorArtifact"),
                f"{phase}.behaviorArtifact",
            )
            if behavior.get("onnxSha256") != expected_behaviors[phase]:
                raise NonPlayingRlOrchestratorError(
                    f"{iteration_dir}: {phase} behavior ONNX chain mismatch."
                )
            artifact = _require_dict(phase_record.get("artifact"), f"{phase}.artifact")
            expected_inputs[phase] = _required_str(artifact.get("checkpointSha256"))
            expected_behaviors[phase] = _required_str(artifact.get("onnxSha256"))
        next_index = iteration + 1
    return next_index


def _iterative_input_checkpoint(
    config: NonPlayingIterativeRlRunConfig,
    iteration: int,
    phase: PhaseName,
) -> Path | None:
    if iteration == 0:
        return None
    previous = _iterative_iteration_dir(config, iteration - 1) / phase / "output-checkpoint.pt"
    _ensure_file(previous, f"{phase} previous checkpoint")
    return previous


def _iterative_behavior_artifact(
    config: NonPlayingIterativeRlRunConfig,
    iteration: int,
    phase: PhaseName,
    bootstrap_artifacts: dict[PhaseName, dict[str, str]],
) -> dict[str, str]:
    if iteration == 0:
        return dict(bootstrap_artifacts[phase])
    previous_phase_dir = _iterative_iteration_dir(config, iteration - 1) / phase
    return {
        "onnxPath": str(previous_phase_dir / "policy.onnx"),
        "metadataPath": str(previous_phase_dir / "policy.json"),
        "artifactId": f"{phase}-iter-{iteration:06d}",
        "provenance": f"iteration-{iteration - 1:06d}",
    }


def _evaluation_is_due(config: NonPlayingIterativeRlRunConfig, iteration: int) -> bool:
    return (iteration + 1) % config.evaluation_interval == 0 or iteration + 1 == config.iterations


def _iterative_rollout_seed(
    config: NonPlayingIterativeRlRunConfig,
    iteration: int,
    phase_offset: int,
) -> int:
    return config.seed + iteration * ITERATION_SEED_STRIDE + phase_offset * PHASE_SEED_STRIDE


def _iterative_training_seed(
    config: NonPlayingIterativeRlRunConfig,
    iteration: int,
    phase_offset: int,
) -> int:
    return _iterative_rollout_seed(config, iteration, phase_offset) + 1


def _iterative_evaluation_seed(
    config: NonPlayingIterativeRlRunConfig,
    iteration: int,
) -> int:
    return config.seed + iteration * ITERATION_SEED_STRIDE + EVALUATION_SEED_OFFSET


def _iterative_iteration_dir(config: NonPlayingIterativeRlRunConfig, iteration: int) -> Path:
    return config.output_dir / "iterations" / f"iter-{iteration:06d}"


def _load_completed_nonplaying_iteration(iteration_dir: Path) -> dict[str, object] | None:
    iteration_path = iteration_dir / "iteration.json"
    if not iteration_path.exists():
        return None
    data = _load_json_object(iteration_path)
    if data.get("completionState") != "completed":
        return None
    return data


def _validate_nonplaying_iteration_artifacts(record: dict[str, object]) -> None:
    phases = _require_dict(record.get("phases"), "iteration.phases")
    for phase_name in ("bidding", "adjutant", "exchange"):
        phase = _require_dict(phases.get(phase_name), f"iteration.phases.{phase_name}")
        artifact = _require_dict(phase.get("artifact"), f"{phase_name}.artifact")
        for path_key, sha_key in (
            ("checkpointPath", "checkpointSha256"),
            ("onnxPath", "onnxSha256"),
            ("metadataPath", "metadataSha256"),
        ):
            path = Path(_required_str(artifact.get(path_key)))
            _ensure_file(path, f"{phase_name} {path_key}")
            expected_sha = _required_str(artifact.get(sha_key))
            actual_sha = _sha256_file(path)
            if actual_sha != expected_sha:
                raise NonPlayingRlOrchestratorError(
                    f"{phase_name} {path_key} SHA mismatch: {actual_sha} != {expected_sha}"
                )
        manifest_path = Path(_required_str(phase.get("manifestPath")))
        _ensure_file(manifest_path, f"{phase_name} manifest")
        if _sha256_file(manifest_path) != _required_str(phase.get("manifestSha256")):
            raise NonPlayingRlOrchestratorError(f"{phase_name} manifest SHA mismatch.")
    if record.get("evaluationDue") is True:
        evaluation_path = Path(_required_str(record.get("evaluationPath")))
        _ensure_file(evaluation_path, "iteration evaluation")


def _validate_nonplaying_rollout_manifest(
    manifest_path: Path,
    *,
    phase: PhaseName,
    expected_start_seed: int,
    expected_game_count: int,
    expected_behavior_onnx_sha256: str,
    expected_behavior_metadata_sha256: str,
    expected_playing_artifact_id: str,
) -> None:
    manifest = _load_json_object(manifest_path)
    if manifest.get("startSeed") != expected_start_seed:
        raise NonPlayingRlOrchestratorError("rollout manifest startSeed mismatch.")
    if manifest.get("gameCount") != expected_game_count:
        raise NonPlayingRlOrchestratorError("rollout manifest gameCount mismatch.")
    if manifest.get("gameCountUnit") != NONPLAYING_GAME_COUNT_UNIT:
        raise NonPlayingRlOrchestratorError("rollout manifest gameCountUnit mismatch.")
    if manifest.get("logicalSeedCount") != expected_game_count:
        raise NonPlayingRlOrchestratorError("rollout manifest logicalSeedCount mismatch.")
    if manifest.get("actualGameCount") != expected_game_count * len(NONPLAYING_ROTATION_OFFSETS):
        raise NonPlayingRlOrchestratorError("rollout manifest actualGameCount mismatch.")
    if manifest.get("rolloutPolicyTopology") != NONPLAYING_ROLLOUT_POLICY_TOPOLOGY:
        raise NonPlayingRlOrchestratorError("rollout manifest policy topology mismatch.")
    if manifest.get("rotationOffsets") != NONPLAYING_ROTATION_OFFSETS:
        raise NonPlayingRlOrchestratorError("rollout manifest rotationOffsets mismatch.")
    reward = _require_dict(manifest.get("reward"), "manifest.reward")
    if (
        reward.get("type") != NONPLAYING_REWARD_TYPE
        or reward.get("version") != NONPLAYING_REWARD_VERSION
        or reward.get("id") != NONPLAYING_REWARD_ID
    ):
        raise NonPlayingRlOrchestratorError("rollout manifest reward metadata mismatch.")
    behavior = _require_dict(manifest.get("behaviorPolicy"), "manifest.behaviorPolicy")
    if behavior.get("onnxSha256") != expected_behavior_onnx_sha256:
        raise NonPlayingRlOrchestratorError("rollout behavior ONNX SHA mismatch.")
    if behavior.get("metadataSha256") != expected_behavior_metadata_sha256:
        raise NonPlayingRlOrchestratorError("rollout behavior metadata SHA mismatch.")
    fixed_playing = _require_dict(manifest.get("fixedPlayingPolicy"), "manifest.fixedPlayingPolicy")
    if fixed_playing.get("artifactId") != expected_playing_artifact_id:
        raise NonPlayingRlOrchestratorError("rollout fixed playing artifact mismatch.")
    if phase == "bidding":
        _validate_frozen_bidding_opponent_mix(manifest, expected_game_count)


def _validate_frozen_bidding_opponent_mix(
    manifest: dict[str, object],
    expected_game_count: int,
) -> None:
    non_learning = _require_dict(manifest.get("nonLearningAgents"), "manifest.nonLearningAgents")
    bidding = _require_dict(non_learning.get("bidding"), "manifest.nonLearningAgents.bidding")
    if bidding.get("type") != "mixed-frozen-bidding":
        raise NonPlayingRlOrchestratorError("rollout frozen bidding mix metadata mismatch.")
    if bidding.get("mixingRuleVersion") != FROZEN_BIDDING_OPPONENT_MIX_RULE_VERSION:
        raise NonPlayingRlOrchestratorError("rollout frozen bidding mix rule mismatch.")
    policies = _require_dict(bidding.get("policies"), "manifest.nonLearningAgents.bidding.policies")
    conservative = _require_dict(policies.get("conservative"), "bidding.policies.conservative")
    passive = _require_dict(policies.get("passive"), "bidding.policies.passive")
    if conservative.get("id") != CONSERVATIVE_BIDDING_BASELINE_ID:
        raise NonPlayingRlOrchestratorError("rollout conservative bidding baseline mismatch.")
    if passive.get("id") != PASSIVE_BIDDING_BASELINE_ID:
        raise NonPlayingRlOrchestratorError("rollout passive bidding baseline mismatch.")

    diagnostics = _require_dict(manifest.get("diagnostics"), "manifest.diagnostics")
    mix = _require_dict(
        diagnostics.get("frozenBiddingOpponentMix"),
        "manifest.diagnostics.frozenBiddingOpponentMix",
    )
    if mix.get("mixingRuleVersion") != FROZEN_BIDDING_OPPONENT_MIX_RULE_VERSION:
        raise NonPlayingRlOrchestratorError("rollout frozen bidding diagnostics mix rule mismatch.")
    expected_assignment_count = expected_game_count * len(NONPLAYING_ROTATION_OFFSETS) * 4
    assignments = mix.get("seatAssignments")
    if not isinstance(assignments, list) or len(assignments) != expected_assignment_count:
        raise NonPlayingRlOrchestratorError("rollout frozen bidding assignment count mismatch.")
    conservative_count = _required_int(mix.get("conservativeSeatCount"))
    passive_count = _required_int(mix.get("passiveSeatCount"))
    if conservative_count + passive_count != expected_assignment_count:
        raise NonPlayingRlOrchestratorError("rollout frozen bidding mix counts mismatch.")


def _update_latest_links(config: NonPlayingIterativeRlRunConfig, iteration: int) -> None:
    latest = config.output_dir / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    for phase_name in ("bidding", "adjutant", "exchange"):
        link = latest / phase_name
        target = Path("..") / "iterations" / f"iter-{iteration:06d}" / phase_name
        if link.exists() or link.is_symlink():
            if link.is_dir() and not link.is_symlink():
                raise NonPlayingRlOrchestratorError(f"latest path is not a symlink: {link}")
            link.unlink()
        link.symlink_to(target)


def _completed_evaluation_iterations(config: NonPlayingIterativeRlRunConfig) -> list[int]:
    completed: list[int] = []
    for iteration in range(config.iterations):
        record = _load_completed_nonplaying_iteration(_iterative_iteration_dir(config, iteration))
        if record is None:
            break
        if record.get("evaluationDue") is True:
            completed.append(iteration)
    return completed


def _iteration_summary_line(record: dict[str, object]) -> dict[str, object]:
    phases = _require_dict(record["phases"], "iteration.phases")
    summary: dict[str, object] = {
        "iteration": record["iteration"],
        "evaluationDue": record["evaluationDue"],
        "elapsedSeconds": record["elapsedSeconds"],
    }
    for phase_name in ("bidding", "adjutant", "exchange"):
        phase = _require_dict(phases[phase_name], phase_name)
        train = _require_dict(phase["train"], f"{phase_name}.train")
        rollout = _require_dict(phase["rollout"], f"{phase_name}.rollout")
        summary[phase_name] = {
            "sampleCount": rollout.get("sampleCount"),
            "gameCountUnit": rollout.get("gameCountUnit"),
            "logicalSeedCount": rollout.get("logicalSeedCount"),
            "actualGameCount": rollout.get("actualGameCount"),
            "diagnostics": rollout.get("diagnostics"),
            "meanReward": train.get("meanReward"),
            "meanTotalLoss": train.get("meanTotalLoss"),
            "meanValueLoss": train.get("meanValueLoss"),
            "clippedFraction": train.get("clippedFraction"),
            "checkpointSha256": _require_dict(phase["artifact"], "artifact").get(
                "checkpointSha256"
            ),
        }
    evaluation = record.get("evaluation")
    if isinstance(evaluation, dict):
        summary["evaluation"] = {
            "completedGames": evaluation.get("completedGames"),
            "scheduledGames": evaluation.get("scheduledGames"),
            "fallbackCount": evaluation.get("fallbackCount"),
            "illegalActionCount": evaluation.get("illegalActionCount"),
        }
    return summary


def _write_iterative_run_summary(
    config: NonPlayingIterativeRlRunConfig,
    file_config: dict[str, object],
    *,
    completed_iteration_count: int,
    elapsed_seconds: float,
    state: dict[str, object],
) -> dict[str, object]:
    latest_iteration = completed_iteration_count - 1
    latest_record = (
        _load_completed_nonplaying_iteration(_iterative_iteration_dir(config, latest_iteration))
        if latest_iteration >= 0
        else None
    )
    summary: dict[str, object] = {
        "schemaVersion": 1,
        "runType": "non-playing-iterative-ppo",
        "config": file_config,
        "completedIterationCount": completed_iteration_count,
        "requestedIterations": config.iterations,
        "latestCompletedIteration": latest_iteration,
        "state": state,
        "latestIteration": latest_record,
        "completedAtUnixSeconds": int(time.time()),
        "elapsedSeconds": elapsed_seconds,
    }
    _atomic_write_json(config.output_dir / "run-summary.json", summary)
    return summary


def _load_iterative_state(run_dir: Path) -> dict[str, object]:
    state_path = run_dir / "state.json"
    if not state_path.exists():
        return {
            "schemaVersion": 1,
            "completedIterationCount": 0,
            "latestCompletedIteration": -1,
            "completedEvaluations": [],
        }
    return _load_json_object(state_path)


def _write_iterative_state(
    config: NonPlayingIterativeRlRunConfig,
    state: dict[str, object],
) -> dict[str, object]:
    data = {"schemaVersion": 1, **state}
    _atomic_write_json(config.output_dir / "state.json", data)
    return data


def _prepare_iterative_output_dir(config: NonPlayingIterativeRlRunConfig) -> None:
    if config.output_dir.exists():
        if not config.output_dir.is_dir():
            raise NonPlayingRlOrchestratorError(
                f"output-dir exists and is not a directory: {config.output_dir}"
            )
        if any(config.output_dir.iterdir()):
            if not config.overwrite:
                raise NonPlayingRlOrchestratorError(
                    f"output-dir is not empty; choose a new path or pass --overwrite: "
                    f"{config.output_dir}"
                )
            shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)


def _validate_iterative_resume_config(
    stored_config: dict[str, object],
    requested_config: dict[str, object],
    *,
    provided_config_keys: set[str],
) -> None:
    if stored_config.get("schemaVersion") != ITERATIVE_RUN_CONFIG_SCHEMA_VERSION:
        raise NonPlayingRlOrchestratorError("stored config schemaVersion mismatch.")
    always_check = {
        "schemaVersion",
        "runType",
        "outputDir",
        "gamesPerIterationUnit",
        "rolloutPolicyTopology",
        "rotationOffsets",
        "reward",
        "biddingFrozenOpponentMixRuleVersion",
        "biddingFrozenOpponentPolicyIds",
        "playingPolicyOnnxSha256",
        "playingPolicyMetadataSha256",
        "playingPolicyArtifactId",
    }
    for key in always_check | provided_config_keys:
        requested_value = requested_config.get(key)
        stored_value = stored_config.get(key)
        if stored_value != requested_value:
            raise NonPlayingRlOrchestratorError(
                f"resume config mismatch for {key}: {requested_value!r} != {stored_value!r}"
            )


def _iterative_config_from_file_dict(
    data: dict[str, object],
    *,
    build_typescript: bool,
) -> NonPlayingIterativeRlRunConfig:
    return NonPlayingIterativeRlRunConfig(
        output_dir=Path(_required_str(data["outputDir"])),
        iterations=_required_int(data["iterations"]),
        games_per_iteration=_required_int(data["gamesPerIteration"]),
        evaluation_interval=_required_int(data["evaluationInterval"]),
        evaluation_games=_required_int(data["evaluationGames"]),
        games_per_shard=_required_int(data["gamesPerShard"]),
        epochs=_required_int(data["epochs"]),
        batch_size=_required_int(data["batchSize"]),
        learning_rate=_required_float(data["learningRate"]),
        hidden_dim=_required_int(data["hiddenDim"]),
        hidden_layers=_required_int(data["hiddenLayers"]),
        dropout=_required_float(data["dropout"]),
        ppo_clip_epsilon=_required_float(data["ppoClipEpsilon"]),
        value_loss_coefficient=_required_float(data["valueLossCoefficient"]),
        seed=_required_int(data["seed"]),
        temperature=_required_float(data["temperature"]),
        inference_device=cast(
            Literal["cpu", "auto", "cuda"],
            _required_str(data["inferenceDevice"]),
        ),
        inference_max_batch_size=_required_int(data["inferenceMaxBatchSize"]),
        playing_policy_onnx=Path(_required_str(data["playingPolicyOnnx"])),
        playing_policy_metadata=Path(_required_str(data["playingPolicyMetadata"])),
        playing_policy_artifact_id=_required_str(data["playingPolicyArtifactId"]),
        build_typescript=build_typescript,
        overwrite=False,
    )


def _stage(name: str, fn: Callable[[], _T]) -> _T:
    print(f"[stage] {name}", flush=True)
    try:
        return fn()
    except Exception as error:
        if isinstance(error, NonPlayingRlOrchestratorError) and str(error).startswith("stage "):
            raise
        raise NonPlayingRlOrchestratorError(f"stage '{name}' failed: {error}") from error


def _validate_config(config: NonPlayingRlRunConfig) -> None:
    _validate_positive_int(config.games, "games")
    _validate_positive_int(config.evaluation_games, "evaluation-games")
    _validate_positive_int(config.effective_games_per_shard, "games-per-shard")
    _validate_positive_int(config.epochs, "epochs")
    _validate_positive_int(config.batch_size, "batch-size")
    _validate_positive_int(config.hidden_dim, "hidden-dim")
    _validate_positive_int(config.hidden_layers, "hidden-layers")
    _validate_positive_int(config.inference_max_batch_size, "inference-max-batch-size")
    _validate_positive_float(config.learning_rate, "learning-rate")
    _validate_positive_float(config.ppo_clip_epsilon, "ppo-clip-epsilon")
    _validate_positive_float(config.temperature, "temperature")
    if config.value_loss_coefficient < 0.0:
        raise NonPlayingRlOrchestratorError("value-loss-coefficient must be non-negative.")
    if config.dropout < 0.0 or config.dropout >= 1.0:
        raise NonPlayingRlOrchestratorError("dropout must be in [0.0, 1.0).")
    if config.inference_device not in SUPPORTED_INFERENCE_DEVICES:
        raise NonPlayingRlOrchestratorError(
            f"inference-device must be one of {', '.join(SUPPORTED_INFERENCE_DEVICES)}."
        )
    _ensure_file(config.playing_policy_onnx, "playing policy ONNX")
    _ensure_file(config.playing_policy_metadata, "playing policy metadata")


def _validate_iterative_config(config: NonPlayingIterativeRlRunConfig) -> None:
    _validate_positive_int(config.iterations, "iterations")
    _validate_positive_int(config.games_per_iteration, "games-per-iteration")
    _validate_positive_int(config.evaluation_interval, "evaluation-interval")
    _validate_positive_int(config.evaluation_games, "evaluation-games")
    _validate_positive_int(config.effective_games_per_shard, "games-per-shard")
    _validate_positive_int(config.epochs, "epochs")
    _validate_positive_int(config.batch_size, "batch-size")
    _validate_positive_int(config.hidden_dim, "hidden-dim")
    _validate_positive_int(config.hidden_layers, "hidden-layers")
    _validate_positive_int(config.inference_max_batch_size, "inference-max-batch-size")
    _validate_positive_float(config.learning_rate, "learning-rate")
    _validate_positive_float(config.ppo_clip_epsilon, "ppo-clip-epsilon")
    _validate_positive_float(config.temperature, "temperature")
    if config.value_loss_coefficient < 0.0:
        raise NonPlayingRlOrchestratorError("value-loss-coefficient must be non-negative.")
    if config.dropout < 0.0 or config.dropout >= 1.0:
        raise NonPlayingRlOrchestratorError("dropout must be in [0.0, 1.0).")
    if config.inference_device not in SUPPORTED_INFERENCE_DEVICES:
        raise NonPlayingRlOrchestratorError(
            f"inference-device must be one of {', '.join(SUPPORTED_INFERENCE_DEVICES)}."
        )
    if config.playing_policy_artifact_id != DEFAULT_PLAYING_POLICY_ARTIFACT_ID:
        raise NonPlayingRlOrchestratorError(
            "iterative non-playing RL currently requires frozen playing policy "
            f"{DEFAULT_PLAYING_POLICY_ARTIFACT_ID!r}."
        )
    _ensure_file(config.playing_policy_onnx, "playing policy ONNX")
    _ensure_file(config.playing_policy_metadata, "playing policy metadata")


def _prepare_output_dir(config: NonPlayingRlRunConfig) -> None:
    if config.output_dir.exists():
        if not config.output_dir.is_dir():
            raise NonPlayingRlOrchestratorError(
                f"output-dir exists and is not a directory: {config.output_dir}"
            )
        if any(config.output_dir.iterdir()):
            if not config.overwrite:
                raise NonPlayingRlOrchestratorError(
                    f"output-dir is not empty; choose a new path or pass --overwrite: "
                    f"{config.output_dir}"
                )
            shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)


def _build_typescript_helpers() -> None:
    result = subprocess.run(
        ["pnpm", "--filter", "@napoleon/self-play-cli...", "build"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    if result.returncode != 0:
        raise NonPlayingRlOrchestratorError(
            f"TypeScript helper build failed with exit {result.returncode}."
        )


def _run_node_json(command: Sequence[str], *, cwd: Path) -> dict[str, object]:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    if result.returncode != 0:
        raise NonPlayingRlOrchestratorError(
            f"command failed with exit {result.returncode}: {command}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise NonPlayingRlOrchestratorError(
            f"subprocess did not return JSON: {result.stdout!r}"
        ) from error
    if not isinstance(parsed, dict):
        raise NonPlayingRlOrchestratorError("subprocess JSON output must be an object.")
    return cast(dict[str, object], parsed)


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        temp_path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, sort_keys=True) + "\n")


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise NonPlayingRlOrchestratorError(f"cannot read JSON file {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise NonPlayingRlOrchestratorError(f"invalid JSON file {path}: {error}") from error
    if not isinstance(raw, dict):
        raise NonPlayingRlOrchestratorError(f"JSON file must contain an object: {path}")
    return cast(dict[str, object], raw)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _print_completion(summary: dict[str, object]) -> None:
    evaluation = cast(dict[str, object], summary["evaluation"])
    counts = cast(dict[str, object], evaluation["policyAgentDecisionCounts"])
    print(
        "[complete] "
        f"runSummary={cast(dict[str, object], summary['artifactPaths'])['runSummary']} "
        f"completed={evaluation['completedGames']}/{evaluation['scheduledGames']} "
        f"fallback={evaluation['fallbackCount']} "
        f"illegal={evaluation['illegalActionCount']} "
        f"bidding={counts['biddingOnnxCallCount']} "
        f"adjutant={counts['adjutantOnnxCallCount']} "
        f"exchange={counts['exchangeOnnxCallCount']}",
        flush=True,
    )


def _print_iterative_completion(summary: dict[str, object]) -> None:
    latest_iteration = summary["latestCompletedIteration"]
    latest = summary.get("latestIteration")
    config = cast(dict[str, object], summary["config"])
    fallback: object = "n/a"
    illegal: object = "n/a"
    if isinstance(latest, dict) and isinstance(latest.get("evaluation"), dict):
        evaluation = cast(dict[str, object], latest["evaluation"])
        fallback = evaluation.get("fallbackCount", "n/a")
        illegal = evaluation.get("illegalActionCount", "n/a")
    print(
        "[complete] "
        f"runSummary={config['outputDir']}/run-summary.json "
        f"completedIterations={summary['completedIterationCount']}/"
        f"{summary['requestedIterations']} "
        f"latestIteration={latest_iteration} "
        f"fallback={fallback} "
        f"illegal={illegal}",
        flush=True,
    )


def _quarantine_incomplete_directory(directory: Path, *, label: str) -> None:
    target = directory.with_name(
        f"{directory.name}.incomplete-{int(time.time())}-{os.getpid()}"
    )
    print(f"[resume] moving incomplete {label} aside: {directory} -> {target}", flush=True)
    directory.rename(target)


def _require_dict(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise NonPlayingRlOrchestratorError(f"{label} must be an object.")
    return cast(dict[str, object], value)


def _required_str(value: object) -> str:
    if not isinstance(value, str) or value == "":
        raise NonPlayingRlOrchestratorError("required string is missing.")
    return value


def _required_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise NonPlayingRlOrchestratorError("required integer is missing.")
    return value


def _required_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise NonPlayingRlOrchestratorError("required number is missing.")
    return float(value)


def _validate_positive_int(value: int, label: str) -> None:
    if isinstance(value, bool) or value <= 0:
        raise NonPlayingRlOrchestratorError(f"{label} must be a positive integer.")


def _validate_positive_float(value: float, label: str) -> None:
    if value <= 0.0:
        raise NonPlayingRlOrchestratorError(f"{label} must be positive.")


def _ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise NonPlayingRlOrchestratorError(f"{label} file does not exist: {path}")


def _resolve_repo_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (_repo_root() / expanded).resolve()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]
