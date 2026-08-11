"""Experiment orchestrator for iterative playing self-play REINFORCE runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, cast

from napoleon_ml.cli._policy_common import configure_reproducibility, load_checked_manifest
from napoleon_ml.dataset.constants import PLAYING_SELF_PLAY_BINARY_DATASET_FORMAT, UINT32_MAX
from napoleon_ml.dataset.manifest import DatasetManifest, parse_manifest
from napoleon_ml.dataset.pytorch import create_playing_self_play_dataloader
from napoleon_ml.dataset.reader import iter_raw_samples, load_manifest
from napoleon_ml.dataset.split import DatasetSplit, SplitConfig
from napoleon_ml.dataset.validation import validate_manifest
from napoleon_ml.policy.actor_critic import (
    ACTOR_CRITIC_ALGORITHM,
    DEFAULT_VALUE_LOSS_COEFFICIENT,
    ActorCriticTrainReport,
    ActorCriticTrainSettings,
    train_policy_actor_critic,
)
from napoleon_ml.policy.device import (
    SUPPORTED_TORCH_DEVICES,
    RequestedTorchDevice,
    TorchDeviceResolutionError,
    resolve_torch_device,
)
from napoleon_ml.policy.onnx_export import export_policy_checkpoint_to_onnx
from napoleon_ml.policy.reinforce import (
    REINFORCE_ALGORITHM,
    ReinforceTrainReport,
    ReinforceTrainSettings,
    train_policy_reinforce,
)

ORCHESTRATOR_SCHEMA_VERSION = 1
DEFAULT_ITERATIONS = 100
DEFAULT_GAMES_PER_ITERATION = 200
DEFAULT_GAMES_PER_SHARD = 20
DEFAULT_SELF_PLAY_SEED_BASE = 0
DEFAULT_TEMPERATURE = 1.0
DEFAULT_ROLLOUT_ROSTER = ",".join(["current-policy"] * 5)
DEFAULT_ROLLOUT_WORKERS = 1
DEFAULT_INFERENCE_MAX_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 128
DEFAULT_FULL_DIAGNOSTICS_INTERVAL = 10
DEFAULT_TRAINING_SEED_BASE = 2_000_000_000
DEFAULT_EVALUATION_INTERVAL = 10
DEFAULT_EVALUATION_START_SEED = 1_000_000_000
DEFAULT_EVALUATION_SEED_COUNT = 100
DEFAULT_INFERENCE_DEVICE = "cpu"
DEFAULT_RETAIN_SELF_PLAY_DATA = False
DEFAULT_SIMULATION_BACKEND = "typescript"
DEFAULT_FROZEN_POLICY_ARTIFACT_ID = "rl-v740"
DEFAULT_FROZEN_POLICY_ONNX = Path("benchmarks/playing-policies/rl-v740/policy.onnx")
DEFAULT_FROZEN_POLICY_METADATA = Path("benchmarks/playing-policies/rl-v740/policy.json")
PLAYING_RL_ALGORITHMS = (REINFORCE_ALGORITHM, ACTOR_CRITIC_ALGORITHM)
SUPPORTED_INFERENCE_DEVICES = ("cpu", "auto", "cuda")
SUPPORTED_SIMULATION_BACKENDS = ("typescript", "cpp")


@dataclass(frozen=True)
class PlayingRlRunConfig:
    run_directory: Path
    initial_checkpoint: Path
    supervised_dataset: Path
    iterations: int = DEFAULT_ITERATIONS
    games_per_iteration: int = DEFAULT_GAMES_PER_ITERATION
    games_per_shard: int = DEFAULT_GAMES_PER_SHARD
    self_play_seed_base: int = DEFAULT_SELF_PLAY_SEED_BASE
    temperature: float = DEFAULT_TEMPERATURE
    rollout_roster: str = DEFAULT_ROLLOUT_ROSTER
    rollout_workers: int = DEFAULT_ROLLOUT_WORKERS
    rollout_concurrency: int | None = None
    inference_max_batch_size: int = DEFAULT_INFERENCE_MAX_BATCH_SIZE
    algorithm: str = REINFORCE_ALGORITHM
    learning_rate: float = DEFAULT_LEARNING_RATE
    value_loss_coefficient: float = DEFAULT_VALUE_LOSS_COEFFICIENT
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    full_diagnostics_interval: int = DEFAULT_FULL_DIAGNOSTICS_INTERVAL
    device: RequestedTorchDevice = "cpu"
    training_seed_base: int = DEFAULT_TRAINING_SEED_BASE
    evaluation_interval: int = DEFAULT_EVALUATION_INTERVAL
    evaluation_start_seed: int = DEFAULT_EVALUATION_START_SEED
    evaluation_seed_count: int = DEFAULT_EVALUATION_SEED_COUNT
    inference_device: str = DEFAULT_INFERENCE_DEVICE
    retain_self_play_data: bool = DEFAULT_RETAIN_SELF_PLAY_DATA
    simulation_backend: str = DEFAULT_SIMULATION_BACKEND
    frozen_policy_onnx: Path | None = None
    frozen_policy_metadata: Path | None = None
    frozen_policy_artifact_id: str = DEFAULT_FROZEN_POLICY_ARTIFACT_ID
    build_typescript: bool = True
    build_cpp: bool = True

    def normalized(self) -> PlayingRlRunConfig:
        return PlayingRlRunConfig(
            run_directory=self.run_directory.expanduser().resolve(),
            initial_checkpoint=self.initial_checkpoint.expanduser().resolve(),
            supervised_dataset=self.supervised_dataset.expanduser().resolve(),
            iterations=self.iterations,
            games_per_iteration=self.games_per_iteration,
            games_per_shard=self.games_per_shard,
            self_play_seed_base=self.self_play_seed_base,
            temperature=self.temperature,
            rollout_roster=self.rollout_roster,
            rollout_workers=self.rollout_workers,
            rollout_concurrency=self.rollout_concurrency,
            inference_max_batch_size=self.inference_max_batch_size,
            algorithm=self.algorithm,
            learning_rate=self.learning_rate,
            value_loss_coefficient=self.value_loss_coefficient,
            epochs=self.epochs,
            batch_size=self.batch_size,
            full_diagnostics_interval=self.full_diagnostics_interval,
            device=self.device,
            training_seed_base=self.training_seed_base,
            evaluation_interval=self.evaluation_interval,
            evaluation_start_seed=self.evaluation_start_seed,
            evaluation_seed_count=self.evaluation_seed_count,
            inference_device=self.inference_device,
            retain_self_play_data=self.retain_self_play_data,
            simulation_backend=self.simulation_backend,
            frozen_policy_onnx=(
                self.frozen_policy_onnx.expanduser().resolve()
                if self.frozen_policy_onnx is not None
                else None
            ),
            frozen_policy_metadata=(
                self.frozen_policy_metadata.expanduser().resolve()
                if self.frozen_policy_metadata is not None
                else None
            ),
            frozen_policy_artifact_id=self.frozen_policy_artifact_id,
            build_typescript=self.build_typescript,
            build_cpp=self.build_cpp,
        )

    def to_file_dict(self) -> dict[str, object]:
        return {
            "schemaVersion": ORCHESTRATOR_SCHEMA_VERSION,
            "runDirectory": str(self.run_directory),
            "initialCheckpoint": str(self.initial_checkpoint),
            "initialCheckpointSha256": sha256_file(self.initial_checkpoint),
            "supervisedDataset": str(self.supervised_dataset),
            "supervisedManifestSha256": sha256_file(self.supervised_dataset / "manifest.json"),
            "iterations": self.iterations,
            "gamesPerIteration": self.games_per_iteration,
            "gamesPerShard": self.games_per_shard,
            "selfPlaySeedBase": self.self_play_seed_base,
            "temperature": self.temperature,
            "rolloutRoster": self.rollout_roster,
            "rolloutWorkers": self.rollout_workers,
            "rolloutConcurrency": _effective_rollout_concurrency(self),
            "inferenceMaxBatchSize": self.inference_max_batch_size,
            "algorithm": self.algorithm,
            "learningRate": self.learning_rate,
            "valueLossCoefficient": self.value_loss_coefficient,
            "epochs": self.epochs,
            "batchSize": self.batch_size,
            "fullDiagnosticsInterval": self.full_diagnostics_interval,
            "device": self.device,
            "trainingSeedBase": self.training_seed_base,
            "evaluationInterval": self.evaluation_interval,
            "evaluationStartSeed": self.evaluation_start_seed,
            "evaluationSeedCount": self.evaluation_seed_count,
            "inferenceDevice": self.inference_device,
            "retainSelfPlayData": self.retain_self_play_data,
            "simulationBackend": self.simulation_backend,
            "frozenPolicyOnnx": (
                str(self.frozen_policy_onnx) if self.frozen_policy_onnx is not None else None
            ),
            "frozenPolicyOnnxSha256": (
                sha256_file(self.frozen_policy_onnx)
                if self.frozen_policy_onnx is not None
                else None
            ),
            "frozenPolicyMetadata": (
                str(self.frozen_policy_metadata)
                if self.frozen_policy_metadata is not None
                else None
            ),
            "frozenPolicyMetadataSha256": (
                sha256_file(self.frozen_policy_metadata)
                if self.frozen_policy_metadata is not None
                else None
            ),
            "frozenPolicyArtifactId": self.frozen_policy_artifact_id,
        }


class PlayingRlOrchestratorError(RuntimeError):
    """Raised when the experiment cannot safely continue."""


def run_playing_rl_experiment(
    config: PlayingRlRunConfig,
    *,
    resume: bool,
    provided_config_keys: Iterable[str] = (),
) -> None:
    config = config.normalized()
    _validate_config(config)
    _ensure_inputs(config)
    run_directory = config.run_directory
    file_config = config.to_file_dict()
    config_path = run_directory / "config.json"

    if resume:
        stored_config = _load_json_object(config_path)
        _validate_resume_config(
            stored_config=stored_config,
            requested_config=file_config,
            provided_config_keys=set(provided_config_keys),
        )
        file_config = stored_config
        config = _config_from_file_dict(stored_config, build_typescript=config.build_typescript)
        _validate_config(config)
        _ensure_inputs(config)
    else:
        if config_path.exists():
            raise PlayingRlOrchestratorError(
                f"run directory already contains config.json; use --resume: {run_directory}"
            )
        if run_directory.exists() and any(run_directory.iterdir()):
            raise PlayingRlOrchestratorError(
                f"run directory is not empty; refusing to start a new run: {run_directory}"
            )
        run_directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(config_path, file_config)

    if config.simulation_backend == "typescript" and config.build_typescript:
        _build_typescript_helpers()
    if config.simulation_backend == "cpp" and config.build_cpp:
        _build_cpp_helpers()

    load_checked_manifest(config.supervised_dataset, command_label="playing-rl-orchestrator")
    _validate_completed_artifacts(config)
    state = _load_state(run_directory)
    state = _ensure_evaluation(config, state, generation=0)
    completed_generation = _next_iteration(config)
    state = _ensure_due_evaluations(
        config,
        state,
        completed_generation=completed_generation,
    )

    for iteration in range(completed_generation, config.iterations):
        state = _run_iteration(config, iteration, state)
        generation = iteration + 1
        state = _ensure_due_evaluations(config, state, completed_generation=generation)

    _validate_completed_artifacts(config)


def _run_iteration(
    config: PlayingRlRunConfig,
    iteration: int,
    state: dict[str, object],
) -> dict[str, object]:
    total = config.iterations
    iteration_dir = _iteration_dir(config, iteration)
    completed = _load_completed_iteration(iteration_dir)
    if completed is not None:
        _validate_iteration_artifacts(iteration_dir, completed)
        _discard_completed_self_play_cache_if_due(completed)
        return state

    if iteration_dir.exists():
        _quarantine_incomplete_iteration(iteration_dir)

    iteration_dir.mkdir(parents=True)
    self_play_dir = iteration_dir / "selfplay"
    input_checkpoint = _checkpoint_for_generation(config, iteration)
    input_checkpoint_sha256 = sha256_file(input_checkpoint)
    behavior_onnx = iteration_dir / "behavior.onnx"
    behavior_metadata = iteration_dir / "behavior.json"
    output_checkpoint = iteration_dir / "output-checkpoint.pt"

    print(f"[iter {iteration}/{total}] export policy-v{iteration}", flush=True)
    export_policy_checkpoint_to_onnx(
        dataset_directory=config.supervised_dataset,
        checkpoint_path=input_checkpoint,
        onnx_path=behavior_onnx,
        metadata_path=behavior_metadata,
        manifest=load_manifest(config.supervised_dataset),
        verify_integrity=True,
    )
    behavior_onnx_sha256 = sha256_file(behavior_onnx)
    behavior_metadata_sha256 = sha256_file(behavior_metadata)

    start_seed = _self_play_start_seed(config, iteration)
    print(
        f"[iter {iteration}/{total}] self-play "
        f"{config.games_per_iteration} games start_seed={start_seed} "
        f"workers={config.rollout_workers} "
        f"concurrency={_effective_rollout_concurrency(config)} "
        f"max_batch={config.inference_max_batch_size}",
        flush=True,
    )
    rollout_summary = _run_rollout_backend(
        config,
        behavior_onnx=behavior_onnx,
        behavior_metadata=behavior_metadata,
        self_play_dir=self_play_dir,
        start_seed=start_seed,
        artifact_id=f"policy-v{iteration}",
        progress_prefix=f"[iter {iteration}/{total}] self-play ",
    )
    atomic_write_json(
        iteration_dir / "selfplay-rollout-timing.json",
        {
            "schemaVersion": ORCHESTRATOR_SCHEMA_VERSION,
            "simulationBackend": config.simulation_backend,
            "rolloutWorkers": rollout_summary["rolloutWorkers"],
            "rolloutConcurrency": rollout_summary["rolloutConcurrency"],
            "inferenceMaxBatchSize": rollout_summary["inferenceMaxBatchSize"],
            "rolloutElapsedSeconds": rollout_summary["rolloutElapsedSeconds"],
            "inferenceRequestCount": rollout_summary["inferenceRequestCount"],
            "inferenceSessionRunCount": rollout_summary["inferenceSessionRunCount"],
            "inferenceMeanBatchSize": rollout_summary["inferenceMeanBatchSize"],
            "inferenceMaxObservedBatchSize": rollout_summary["inferenceMaxObservedBatchSize"],
            "inferenceBatchSizeHistogram": rollout_summary["inferenceBatchSizeHistogram"],
        },
    )

    manifest = load_manifest(self_play_dir)
    _consume_samples_for_integrity(self_play_dir)
    if manifest.behavior_policy is None:
        raise PlayingRlOrchestratorError("self-play manifest has no behaviorPolicy.")
    if manifest.behavior_policy.onnx_sha256 != behavior_onnx_sha256:
        raise PlayingRlOrchestratorError("self-play behavior ONNX SHA mismatch.")
    if manifest.behavior_policy.metadata_sha256 != behavior_metadata_sha256:
        raise PlayingRlOrchestratorError("self-play behavior metadata SHA mismatch.")
    _validate_self_play_manifest_matches_request(
        manifest=manifest,
        start_seed=start_seed,
        game_count=config.games_per_iteration,
    )
    self_play_manifest_sha256 = sha256_file(self_play_dir / "manifest.json")
    self_play_shard_byte_length = sum(shard.byte_length for shard in manifest.shards)
    self_play_tensor_compression = _self_play_tensor_compression(manifest)
    print(
        f"[iter {iteration}/{total}] self-play complete: {manifest.sample_count} samples",
        flush=True,
    )

    print(f"[iter {iteration}/{total}] {config.algorithm}", flush=True)
    training_seed = _training_seed(config, iteration)
    diagnostics_performed = _full_diagnostics_is_due(config, generation=iteration + 1)
    configure_reproducibility(training_seed)
    dataloader = create_playing_self_play_dataloader(
        self_play_dir,
        split=DatasetSplit.TRAIN,
        split_config=SplitConfig(train=100, validation=0, test=0),
        batch_size=config.batch_size,
        verify_integrity=True,
    )
    report: ActorCriticTrainReport | ReinforceTrainReport
    if config.algorithm == ACTOR_CRITIC_ALGORITHM:
        actor_critic_settings = ActorCriticTrainSettings(
            seed=training_seed,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            value_loss_coefficient=config.value_loss_coefficient,
            verify_integrity=True,
            device=config.device,
            full_diagnostics=diagnostics_performed,
            behavior_parity_execution_provider=cast(str, rollout_summary["executionProvider"]),
            behavior_parity_max_observed_batch_size=cast(
                int,
                rollout_summary["inferenceMaxObservedBatchSize"],
            ),
        )
        report = train_policy_actor_critic(
            input_checkpoint=input_checkpoint,
            self_play_dataset_directory=self_play_dir,
            output_checkpoint=output_checkpoint,
            manifest=manifest,
            dataloader=dataloader,
            settings=actor_critic_settings,
        )
    else:
        reinforce_settings = ReinforceTrainSettings(
            seed=training_seed,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            verify_integrity=True,
            device=config.device,
            full_diagnostics=diagnostics_performed,
            behavior_parity_execution_provider=cast(str, rollout_summary["executionProvider"]),
            behavior_parity_max_observed_batch_size=cast(
                int,
                rollout_summary["inferenceMaxObservedBatchSize"],
            ),
        )
        report = train_policy_reinforce(
            input_checkpoint=input_checkpoint,
            self_play_dataset_directory=self_play_dir,
            output_checkpoint=output_checkpoint,
            manifest=manifest,
            dataloader=dataloader,
            settings=reinforce_settings,
        )
    if report.optimizer_step_count <= 0:
        raise PlayingRlOrchestratorError("optimizerStepCount must be > 0.")
    if report.parameter_delta_norm <= 0:
        raise PlayingRlOrchestratorError("parameterDeltaNorm must be > 0.")
    if report.changed_parameter_count <= 0:
        raise PlayingRlOrchestratorError("changedParameterCount must be > 0.")
    if config.algorithm == ACTOR_CRITIC_ALGORITHM:
        actor_critic_report = cast(ActorCriticTrainReport, report)
        if actor_critic_report.critic_parameter_delta_norm <= 0:
            raise PlayingRlOrchestratorError("criticParameterDeltaNorm must be > 0.")
        if actor_critic_report.changed_critic_parameter_count <= 0:
            raise PlayingRlOrchestratorError("changedCriticParameterCount must be > 0.")

    output_checkpoint_sha256 = sha256_file(output_checkpoint)
    train_report = report.to_dict()
    atomic_write_json(iteration_dir / "train-report.json", train_report)

    iteration_record = {
        "schemaVersion": ORCHESTRATOR_SCHEMA_VERSION,
        "completionState": "completed",
        "iteration": iteration,
        "algorithm": config.algorithm,
        "simulationBackend": config.simulation_backend,
        "inputGeneration": iteration,
        "outputGeneration": iteration + 1,
        "inputCheckpointPath": str(input_checkpoint),
        "inputCheckpointSha256": input_checkpoint_sha256,
        "behaviorOnnxPath": str(behavior_onnx),
        "behaviorOnnxSha256": behavior_onnx_sha256,
        "behaviorMetadataPath": str(behavior_metadata),
        "behaviorMetadataSha256": behavior_metadata_sha256,
        "selfPlayDirectory": str(self_play_dir),
        "selfPlayManifestSha256": self_play_manifest_sha256,
        "selfPlayFormat": manifest.format,
        "selfPlayTensorCompression": self_play_tensor_compression,
        "selfPlayShardCount": manifest.shard_count,
        "selfPlayShardByteLength": self_play_shard_byte_length,
        "selfPlayCacheRetained": config.retain_self_play_data,
        "selfPlayStartSeed": manifest.start_seed,
        "selfPlayEndSeed": manifest.end_seed,
        "gameCount": manifest.game_count,
        "sampleCount": manifest.sample_count,
        "rolloutWorkers": config.rollout_workers,
        "rolloutConcurrency": rollout_summary["rolloutConcurrency"],
        "inferenceMaxBatchSize": rollout_summary["inferenceMaxBatchSize"],
        "trainingSeed": training_seed,
        "requestedDevice": report.requested_device,
        "resolvedDevice": report.resolved_device,
        "requestedInferenceDevice": rollout_summary["requestedInferenceDevice"],
        "resolvedInferenceDevice": rollout_summary["resolvedInferenceDevice"],
        "executionProvider": rollout_summary["executionProvider"],
        "rolloutElapsedSeconds": rollout_summary["rolloutElapsedSeconds"],
        "inferenceRequestCount": rollout_summary["inferenceRequestCount"],
        "inferenceSessionRunCount": rollout_summary["inferenceSessionRunCount"],
        "inferenceMeanBatchSize": rollout_summary["inferenceMeanBatchSize"],
        "inferenceMaxObservedBatchSize": rollout_summary["inferenceMaxObservedBatchSize"],
        "inferenceBatchSizeHistogram": rollout_summary["inferenceBatchSizeHistogram"],
        "cudaDeviceName": report.cuda_device_name,
        "diagnosticsPerformed": report.diagnostics_performed,
        "fullDiagnosticsInterval": config.full_diagnostics_interval,
        "safetyValidationElapsedSeconds": report.safety_validation_elapsed_seconds,
        "preEvalElapsedSeconds": report.pre_eval_elapsed_seconds,
        "optimizerTrainingElapsedSeconds": report.optimizer_training_elapsed_seconds,
        "postEvalElapsedSeconds": report.post_eval_elapsed_seconds,
        "totalElapsedSeconds": report.total_elapsed_seconds,
        "valueLossCoefficient": (
            config.value_loss_coefficient if config.algorithm == ACTOR_CRITIC_ALGORITHM else None
        ),
        "optimizerStepCount": report.optimizer_step_count,
        "behaviorParityDiagnostics": train_report["behaviorParityDiagnostics"],
        "behaviorPolicyProvenance": train_report["behaviorPolicyProvenance"],
        "parameterDeltaNorm": report.parameter_delta_norm,
        "changedParameterCount": report.changed_parameter_count,
        "outputCheckpointPath": str(output_checkpoint),
        "outputCheckpointSha256": output_checkpoint_sha256,
        "completedAtUnixSeconds": int(time.time()),
    }
    atomic_write_json(iteration_dir / "iteration.json", iteration_record)
    _validate_iteration_artifacts(iteration_dir, iteration_record)
    if not config.retain_self_play_data:
        _discard_self_play_shards(self_play_dir, manifest)

    summary = {
        "iteration": iteration,
        "algorithm": config.algorithm,
        "simulationBackend": config.simulation_backend,
        "selfPlayGameCount": manifest.game_count,
        "sampleCount": manifest.sample_count,
        "rolloutConcurrency": rollout_summary["rolloutConcurrency"],
        "inferenceMaxBatchSize": rollout_summary["inferenceMaxBatchSize"],
        "selfPlayFormat": manifest.format,
        "selfPlayTensorCompression": self_play_tensor_compression,
        "selfPlayShardByteLength": self_play_shard_byte_length,
        "selfPlayCacheRetained": config.retain_self_play_data,
        "diagnosticsPerformed": report.diagnostics_performed,
        "positiveRewardCount": report.positive_reward_count,
        "negativeRewardCount": report.negative_reward_count,
        "meanReward": report.mean_reward,
        "forcedSampleCount": report.forced_sample_count,
        "nonForcedSampleCount": report.non_forced_sample_count,
        "requestedDevice": report.requested_device,
        "resolvedDevice": report.resolved_device,
        "requestedInferenceDevice": rollout_summary["requestedInferenceDevice"],
        "resolvedInferenceDevice": rollout_summary["resolvedInferenceDevice"],
        "executionProvider": rollout_summary["executionProvider"],
        "rolloutElapsedSeconds": rollout_summary["rolloutElapsedSeconds"],
        "inferenceRequestCount": rollout_summary["inferenceRequestCount"],
        "inferenceSessionRunCount": rollout_summary["inferenceSessionRunCount"],
        "inferenceMeanBatchSize": rollout_summary["inferenceMeanBatchSize"],
        "inferenceMaxObservedBatchSize": rollout_summary["inferenceMaxObservedBatchSize"],
        "cudaDeviceName": report.cuda_device_name,
        "safetyValidationElapsedSeconds": report.safety_validation_elapsed_seconds,
        "preEvalElapsedSeconds": report.pre_eval_elapsed_seconds,
        "optimizerTrainingElapsedSeconds": report.optimizer_training_elapsed_seconds,
        "postEvalElapsedSeconds": report.post_eval_elapsed_seconds,
        "totalElapsedSeconds": report.total_elapsed_seconds,
        "optimizerStepCount": report.optimizer_step_count,
        "meanSelectedLogProbabilityBefore": report.mean_selected_log_probability_before,
        "meanSelectedLogProbabilityAfter": report.mean_selected_log_probability_after,
        "maxBehaviorLogProbabilityParityError": (
            report.max_behavior_log_probability_parity_error
        ),
        "behaviorParityDiagnostics": train_report["behaviorParityDiagnostics"],
        "behaviorPolicyProvenance": train_report["behaviorPolicyProvenance"],
        "parameterDeltaNorm": report.parameter_delta_norm,
        "changedParameterCount": report.changed_parameter_count,
        "inputCheckpointSha256": input_checkpoint_sha256,
        "behaviorOnnxSha256": behavior_onnx_sha256,
        "outputCheckpointSha256": output_checkpoint_sha256,
    }
    if config.algorithm == ACTOR_CRITIC_ALGORITHM:
        actor_critic_report = cast(ActorCriticTrainReport, report)
        summary.update(
            {
                "actorLossBefore": actor_critic_report.actor_loss_before,
                "actorLossAfter": actor_critic_report.actor_loss_after,
                "valueLossBefore": actor_critic_report.value_loss_before,
                "valueLossAfter": actor_critic_report.value_loss_after,
                "totalLossBefore": actor_critic_report.total_loss_before,
                "totalLossAfter": actor_critic_report.total_loss_after,
                "meanValuePredictionBefore": (
                    actor_critic_report.mean_value_prediction_before
                ),
                "meanValuePredictionAfter": actor_critic_report.mean_value_prediction_after,
                "meanAdvantageBefore": actor_critic_report.mean_advantage_before,
                "meanAdvantageAfter": actor_critic_report.mean_advantage_after,
                "advantageStdBefore": actor_critic_report.advantage_std_before,
                "advantageStdAfter": actor_critic_report.advantage_std_after,
                "actorParameterDeltaNorm": actor_critic_report.actor_parameter_delta_norm,
                "criticParameterDeltaNorm": actor_critic_report.critic_parameter_delta_norm,
                "changedActorParameterCount": (
                    actor_critic_report.changed_actor_parameter_count
                ),
                "changedCriticParameterCount": (
                    actor_critic_report.changed_critic_parameter_count
                ),
            }
        )
    else:
        reinforce_report = cast(ReinforceTrainReport, report)
        summary.update(
            {
                "meanPolicyLossBefore": reinforce_report.mean_policy_loss_before,
                "meanPolicyLossAfter": reinforce_report.mean_policy_loss_after,
            }
        )
    append_jsonl(config.run_directory / "summary.jsonl", summary)
    state = _write_state(
        config,
        {
            **state,
            "completedIterationCount": iteration + 1,
            "latestGeneration": iteration + 1,
            "latestCheckpointPath": str(output_checkpoint),
            "latestCheckpointSha256": output_checkpoint_sha256,
        },
    )
    print(
        "[iter "
        f"{iteration}/{total}] completed -> policy-v{iteration + 1} "
        f"samples={report.sample_count} optimizer_steps={report.optimizer_step_count} "
        f"parameter_delta={report.parameter_delta_norm:.8g}",
        flush=True,
    )
    return state


def _ensure_due_evaluations(
    config: PlayingRlRunConfig,
    state: dict[str, object],
    *,
    completed_generation: int,
) -> dict[str, object]:
    for generation in range(1, completed_generation + 1):
        if _evaluation_is_due(config, generation):
            state = _ensure_evaluation(config, state, generation=generation)
    return state


def _evaluation_is_due(config: PlayingRlRunConfig, generation: int) -> bool:
    return generation % config.evaluation_interval == 0 or generation == config.iterations


def _full_diagnostics_is_due(config: PlayingRlRunConfig, generation: int) -> bool:
    # Full diagnostics follow output generation cadence exactly: interval=10
    # means v10, v20, ... and does not force an extra final-generation pass.
    return generation % config.full_diagnostics_interval == 0


def _ensure_evaluation(
    config: PlayingRlRunConfig,
    state: dict[str, object],
    *,
    generation: int,
) -> dict[str, object]:
    evaluation_dir = config.run_directory / "evaluations" / f"policy-v{generation:03d}"
    summary_path = evaluation_dir / "summary.json"
    if summary_path.exists():
        summary = _load_json_object(summary_path)
        _validate_evaluation_artifacts(config, generation, evaluation_dir, summary)
        return state

    if evaluation_dir.exists() and any(evaluation_dir.iterdir()):
        _quarantine_incomplete_directory(evaluation_dir, label="evaluation")

    evaluation_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _checkpoint_for_generation(config, generation)
    checkpoint_sha256 = sha256_file(checkpoint)
    onnx_path = evaluation_dir / "policy.onnx"
    metadata_path = evaluation_dir / "policy.json"
    result_path = evaluation_dir / "evaluation.json"
    print(f"[eval v{generation}] export", flush=True)
    export_policy_checkpoint_to_onnx(
        dataset_directory=config.supervised_dataset,
        checkpoint_path=checkpoint,
        onnx_path=onnx_path,
        metadata_path=metadata_path,
        manifest=load_manifest(config.supervised_dataset),
        verify_integrity=True,
    )
    onnx_sha256 = sha256_file(onnx_path)
    metadata_sha256 = sha256_file(metadata_path)
    print(
        f"[eval v{generation}] run {config.evaluation_seed_count} seeds x 5 rotations",
        flush=True,
    )
    evaluation_summary = _run_evaluation_backend(
        config,
        onnx_path=onnx_path,
        metadata_path=metadata_path,
        result_path=result_path,
        candidate_id=f"policy-v{generation}",
        progress_prefix=f"[eval v{generation}] ",
    )
    result = _load_json_object(result_path)
    _validate_evaluation_run(config, result)
    baseline_summary = None
    baseline_result = None
    if generation != 0:
        baseline_summary = _load_json_object(
            config.run_directory / "evaluations" / "policy-v000" / "summary.json"
        )
        baseline_result = _load_json_object(
            config.run_directory / "evaluations" / "policy-v000" / "evaluation.json"
        )
    summary = _build_evaluation_summary(
        generation=generation,
        checkpoint_sha256=checkpoint_sha256,
        onnx_sha256=onnx_sha256,
        metadata_sha256=metadata_sha256,
        node_summary=evaluation_summary,
        result=result,
        baseline_summary=baseline_summary,
        baseline_result=baseline_result,
        result_sha256=sha256_file(result_path),
    )
    _validate_evaluation_artifacts(config, generation, evaluation_dir, summary)
    atomic_write_json(summary_path, summary)
    append_jsonl(config.run_directory / "evaluations.jsonl", summary)
    state = _write_state(
        config,
        {
            **state,
            "completedEvaluations": sorted(
                {
                    *cast(list[int], state.get("completedEvaluations", [])),
                    generation,
                }
            ),
        },
    )
    win_rate = cast(float | None, summary["winRate"])
    delta_vs_v0 = cast(float | None, summary["winRateDeltaVsV0"])
    print(
        f"[eval v{generation}] winRate={_format_optional(win_rate)} "
        f"deltaVsV0={_format_optional(delta_vs_v0)}",
        flush=True,
    )
    return state


def _build_evaluation_summary(
    *,
    generation: int,
    checkpoint_sha256: str,
    onnx_sha256: str,
    metadata_sha256: str,
    node_summary: dict[str, object],
    result: dict[str, object],
    baseline_summary: dict[str, object] | None,
    baseline_result: dict[str, object] | None,
    result_sha256: str,
) -> dict[str, object]:
    if "comparison" not in result:
        return _build_cpp_evaluation_summary(
            generation=generation,
            checkpoint_sha256=checkpoint_sha256,
            onnx_sha256=onnx_sha256,
            metadata_sha256=metadata_sha256,
            node_summary=node_summary,
            result=result,
            baseline_summary=baseline_summary,
            baseline_result=baseline_result,
            result_sha256=result_sha256,
        )

    comparison = _object(result["comparison"])
    policy = _object(comparison["policy"])
    rule_based = _object(comparison["ruleBased"])
    policy_comparison = _object(policy["comparison"])
    run = _object(result["run"])
    win_rate = _rate_value(_object(policy["winRate"]))
    contract_success_rate = _rate_value(_object(policy["contractSuccessRate"]))
    average_point_cards = _optional_float(policy["averagePointCards"])
    baseline_win_rate = (
        _optional_float(baseline_summary.get("winRate")) if baseline_summary else None
    )
    baseline_contract = (
        _optional_float(baseline_summary.get("contractSuccessRate"))
        if baseline_summary
        else None
    )
    baseline_points = (
        _optional_float(baseline_summary.get("averagePointCards"))
        if baseline_summary
        else None
    )

    return {
        "schemaVersion": ORCHESTRATOR_SCHEMA_VERSION,
        "generation": generation,
        "checkpointSha256": checkpoint_sha256,
        "onnxSha256": onnx_sha256,
        "metadataSha256": metadata_sha256,
        "evaluationResultSha256": result_sha256,
        "evaluationStartSeed": run["startSeed"],
        "evaluationEndSeed": run["endSeed"],
        "evaluationSeedCount": run["gameCount"],
        "requestedInferenceDevice": _policy_runtime_value(
            result, "requestedInferenceDevice"
        ),
        "resolvedInferenceDevice": _policy_runtime_value(result, "resolvedInferenceDevice"),
        "executionProvider": _policy_runtime_value(result, "executionProvider"),
        "scheduledGames": node_summary["scheduledGames"],
        "completedGames": node_summary["completedGames"],
        "failedGames": node_summary["failedGames"],
        "illegalActionCount": node_summary["illegalActionCount"],
        "wins": policy["wins"],
        "losses": policy["losses"],
        "winRate": win_rate,
        "winRateConfidenceInterval": policy["winRate"],
        "winRateDeltaVsRuleBased": policy_comparison["winRateDelta"],
        "winRateDeltaVsRuleBasedConfidenceInterval": (
            policy_comparison["winRateDeltaConfidenceInterval"]
        ),
        "contractSuccessRate": contract_success_rate,
        "contractSuccessDeltaVsRuleBased": policy_comparison["contractSuccessRateDelta"],
        "averagePointCards": average_point_cards,
        "averagePointCardsDeltaVsRuleBased": policy_comparison["averagePointCardsDelta"],
        "napoleonBreakdown": _role_breakdown(policy, "napoleon"),
        "adjutantBreakdown": _role_breakdown(policy, "adjutant"),
        "allianceBreakdown": _role_breakdown(policy, "alliance"),
        "seatBreakdown": policy["seatResults"],
        "ruleBased": rule_based,
        "winRateDeltaVsV0": subtract_optional(win_rate, baseline_win_rate),
        "contractSuccessDeltaVsV0": subtract_optional(contract_success_rate, baseline_contract),
        "averagePointCardsDeltaVsV0": subtract_optional(average_point_cards, baseline_points),
        "pairedComparisonVsV0": (
            _paired_comparison(baseline_result, result) if baseline_result is not None else None
        ),
    }


def _build_cpp_evaluation_summary(
    *,
    generation: int,
    checkpoint_sha256: str,
    onnx_sha256: str,
    metadata_sha256: str,
    node_summary: dict[str, object],
    result: dict[str, object],
    baseline_summary: dict[str, object] | None,
    baseline_result: dict[str, object] | None,
    result_sha256: str,
) -> dict[str, object]:
    configuration = _object(result["configuration"])
    summary = _object(result["summary"])
    candidate = _object(summary["candidate"])
    candidate_comparison = {
        "winRateDelta": None,
        "winRateDeltaConfidenceInterval": None,
        "contractSuccessRateDelta": None,
        "averagePointCardsDelta": None,
    }
    baseline_win_rate = (
        _optional_float(baseline_summary.get("winRate")) if baseline_summary else None
    )
    baseline_contract = (
        _optional_float(baseline_summary.get("contractSuccessRate"))
        if baseline_summary
        else None
    )
    baseline_points = (
        _optional_float(baseline_summary.get("averagePointCards"))
        if baseline_summary
        else None
    )
    win_rate = _rate_value(_object(candidate["winRate"]))
    contract_success_rate = _rate_value(_object(candidate["contractSuccessRate"]))
    average_point_cards = _optional_float(candidate["averagePointCards"])

    return {
        "schemaVersion": ORCHESTRATOR_SCHEMA_VERSION,
        "simulationBackend": "cpp",
        "generation": generation,
        "checkpointSha256": checkpoint_sha256,
        "onnxSha256": onnx_sha256,
        "metadataSha256": metadata_sha256,
        "evaluationResultSha256": result_sha256,
        "evaluationStartSeed": configuration["startSeed"],
        "evaluationEndSeed": configuration["endSeed"],
        "evaluationSeedCount": configuration["seedCount"],
        "requestedInferenceDevice": node_summary["requestedInferenceDevice"],
        "resolvedInferenceDevice": node_summary["resolvedInferenceDevice"],
        "executionProvider": node_summary["executionProvider"],
        "scheduledGames": node_summary["scheduledGames"],
        "completedGames": node_summary["completedGames"],
        "failedGames": node_summary["failedGames"],
        "illegalActionCount": node_summary["illegalActionCount"],
        "wins": candidate["wins"],
        "losses": candidate["losses"],
        "winRate": win_rate,
        "winRateConfidenceInterval": candidate["winRate"],
        "winRateDeltaVsRuleBased": candidate_comparison["winRateDelta"],
        "winRateDeltaVsRuleBasedConfidenceInterval": (
            candidate_comparison["winRateDeltaConfidenceInterval"]
        ),
        "contractSuccessRate": contract_success_rate,
        "contractSuccessDeltaVsRuleBased": candidate_comparison["contractSuccessRateDelta"],
        "averagePointCards": average_point_cards,
        "averagePointCardsDeltaVsRuleBased": candidate_comparison["averagePointCardsDelta"],
        "napoleonBreakdown": None,
        "adjutantBreakdown": None,
        "allianceBreakdown": None,
        "seatBreakdown": None,
        "ruleBased": _cpp_agent_summary(summary, "rule-based"),
        "timing": result.get("metrics"),
        "winRateDeltaVsV0": subtract_optional(win_rate, baseline_win_rate),
        "contractSuccessDeltaVsV0": subtract_optional(contract_success_rate, baseline_contract),
        "averagePointCardsDeltaVsV0": subtract_optional(average_point_cards, baseline_points),
        "pairedComparisonVsV0": (
            _paired_comparison(baseline_result, result) if baseline_result is not None else None
        ),
    }


def _validate_self_play_manifest_matches_request(
    *,
    manifest: DatasetManifest,
    start_seed: int,
    game_count: int,
) -> None:
    end_seed = start_seed + game_count - 1
    if manifest.start_seed != start_seed:
        raise PlayingRlOrchestratorError(
            f"self-play manifest startSeed mismatch: {manifest.start_seed} != {start_seed}"
        )
    if manifest.end_seed != end_seed:
        raise PlayingRlOrchestratorError(
            f"self-play manifest endSeed mismatch: {manifest.end_seed} != {end_seed}"
        )
    if manifest.game_count != game_count:
        raise PlayingRlOrchestratorError(
            f"self-play manifest gameCount mismatch: {manifest.game_count} != {game_count}"
        )


def _validate_evaluation_run(
    config: PlayingRlRunConfig,
    result: dict[str, object],
) -> None:
    if "comparison" not in result:
        _validate_cpp_evaluation_run(config, result)
        return

    run = _object(result["run"])
    start_seed = _required_int(run["startSeed"])
    game_count = _required_int(run["gameCount"])
    end_seed = _required_int(run["endSeed"])
    rotation_offsets = cast(list[object], run["rotationOffsets"])
    scheduled_games = game_count * len(rotation_offsets)
    completed_count = _required_int(run["completedCount"])
    failed_count = _required_int(run["failedCount"])

    if start_seed != config.evaluation_start_seed:
        raise PlayingRlOrchestratorError(
            f"evaluation startSeed mismatch: {start_seed} != {config.evaluation_start_seed}"
        )
    if game_count != config.evaluation_seed_count:
        raise PlayingRlOrchestratorError(
            f"evaluation gameCount mismatch: {game_count} != {config.evaluation_seed_count}"
        )
    if end_seed != config.evaluation_start_seed + config.evaluation_seed_count - 1:
        raise PlayingRlOrchestratorError(f"evaluation endSeed mismatch: {end_seed}")
    if completed_count + failed_count != scheduled_games:
        raise PlayingRlOrchestratorError(
            "evaluation completedCount + failedCount must equal scheduled games."
        )


def _validate_cpp_evaluation_run(
    config: PlayingRlRunConfig,
    result: dict[str, object],
) -> None:
    configuration = _object(result["configuration"])
    summary = _object(result["summary"])
    start_seed = _required_int(configuration["startSeed"])
    seed_count = _required_int(configuration["seedCount"])
    end_seed = _required_int(configuration["endSeed"])
    rotation_offsets = cast(list[object], configuration["rotationOffsets"])
    scheduled_games = seed_count * len(rotation_offsets)
    completed_count = _required_int(summary["completedGames"])
    failed_count = _required_int(summary["failedGames"])

    if start_seed != config.evaluation_start_seed:
        raise PlayingRlOrchestratorError(
            f"evaluation startSeed mismatch: {start_seed} != {config.evaluation_start_seed}"
        )
    if seed_count != config.evaluation_seed_count:
        raise PlayingRlOrchestratorError(
            f"evaluation gameCount mismatch: {seed_count} != {config.evaluation_seed_count}"
        )
    if end_seed != config.evaluation_start_seed + config.evaluation_seed_count - 1:
        raise PlayingRlOrchestratorError(f"evaluation endSeed mismatch: {end_seed}")
    if completed_count + failed_count != scheduled_games:
        raise PlayingRlOrchestratorError(
            "evaluation completedGames + failedGames must equal scheduled games."
        )


def _policy_runtime_value(result: dict[str, object], key: str) -> object:
    configuration = _object(result["configuration"])
    runtime = configuration.get("policyRuntime")
    if runtime is None:
        runtime = configuration.get("candidateRuntime")
    if runtime is None:
        raise PlayingRlOrchestratorError("evaluation result has no policy runtime metadata.")
    return _object(runtime)[key]


def _paired_comparison(
    baseline_result: dict[str, object],
    current_result: dict[str, object],
) -> dict[str, int]:
    baseline = _policy_win_by_seed_rotation(baseline_result)
    current = _policy_win_by_seed_rotation(current_result)
    keys = sorted(set(baseline) & set(current))
    counts = {
        "v0LoseCurrentWin": 0,
        "v0WinCurrentLose": 0,
        "bothWin": 0,
        "bothLose": 0,
    }
    for key in keys:
        before = baseline[key]
        after = current[key]
        if not before and after:
            counts["v0LoseCurrentWin"] += 1
        elif before and not after:
            counts["v0WinCurrentLose"] += 1
        elif before and after:
            counts["bothWin"] += 1
        else:
            counts["bothLose"] += 1
    return counts


def _policy_win_by_seed_rotation(result: dict[str, object]) -> dict[tuple[int, int], bool]:
    if "run" not in result:
        return _cpp_policy_win_by_seed_rotation(result)

    run = _object(result["run"])
    games = cast(list[object], run["games"])
    wins: dict[tuple[int, int], bool] = {}
    for item in games:
        game = _object(item)
        if game["status"] != "completed":
            continue
        seats = cast(list[object], game["seats"])
        policy_seat: dict[str, object] | None = None
        for seat in seats:
            seat_object = _object(seat)
            if seat_object["sourceAgentIndex"] == 0:
                policy_seat = seat_object
                break
        if policy_seat is None:
            raise PlayingRlOrchestratorError("evaluation game has no policy seat.")
        role = policy_seat["role"]
        winner = game["winner"]
        policy_won = (
            (role == "alliance" and winner == "alliance")
            or (role in {"napoleon", "adjutant"} and winner == "napoleon-team")
        )
        wins[(_required_int(game["seed"]), _required_int(game["rotationOffset"]))] = policy_won
    return wins


def _cpp_policy_win_by_seed_rotation(result: dict[str, object]) -> dict[tuple[int, int], bool]:
    games = cast(list[object], result["games"])
    wins: dict[tuple[int, int], bool] = {}
    for item in games:
        game = _object(item)
        if game["status"] != "completed":
            continue
        roster = _object(game["roster"])
        seats = cast(list[object], roster["seats"])
        policy_seat_index: int | None = None
        for seat in seats:
            seat_object = _object(seat)
            agent = _object(seat_object["agent"])
            if agent["type"] == "current-policy":
                policy_seat_index = _required_int(seat_object["seatIndex"])
                break
        if policy_seat_index is None:
            raise PlayingRlOrchestratorError("C++ evaluation game has no current-policy seat.")
        contract = _object(game["contract"])
        napoleon_id = _required_str(contract["napoleonPlayerId"])
        adjutant_id = contract["adjutantPlayerId"]
        napoleon_team_ids = {napoleon_id}
        if isinstance(adjutant_id, str):
            napoleon_team_ids.add(adjutant_id)
        policy_id = f"player-{policy_seat_index}"
        policy_team = "napoleon-team" if policy_id in napoleon_team_ids else "alliance"
        wins[(_required_int(game["seed"]), _required_int(game["rotationOffset"]))] = (
            policy_team == game["winner"]
        )
    return wins


def _cpp_agent_summary(summary: Mapping[str, object], agent_type: str) -> object:
    agents = cast(list[object], summary["agents"])
    for item in agents:
        obj = _object(item)
        if _required_str(obj["key"]).startswith(f"{agent_type}:"):
            return obj["stats"]
    return None


def _validate_completed_artifacts(config: PlayingRlRunConfig) -> None:
    expected_input_sha256 = sha256_file(config.initial_checkpoint)
    for iteration in range(config.iterations):
        iteration_dir = _iteration_dir(config, iteration)
        record = _load_completed_iteration(iteration_dir)
        if record is None:
            break
        _validate_iteration_artifacts(iteration_dir, record)
        _discard_completed_self_play_cache_if_due(record)
        if record["inputCheckpointSha256"] != expected_input_sha256:
            raise PlayingRlOrchestratorError(
                f"{iteration_dir}: input checkpoint chain mismatch: "
                f"{record['inputCheckpointSha256']} != {expected_input_sha256}"
            )
        expected_input_sha256 = _required_str(record["outputCheckpointSha256"])


def _load_completed_iteration(iteration_dir: Path) -> dict[str, object] | None:
    path = iteration_dir / "iteration.json"
    if not path.exists():
        return None
    record = _load_json_object(path)
    if record.get("completionState") != "completed":
        return None
    return record


def _validate_iteration_artifacts(iteration_dir: Path, record: Mapping[str, object]) -> None:
    required_paths = {
        "inputCheckpointPath": "inputCheckpointSha256",
        "behaviorOnnxPath": "behaviorOnnxSha256",
        "behaviorMetadataPath": "behaviorMetadataSha256",
        "outputCheckpointPath": "outputCheckpointSha256",
    }
    for path_key, sha_key in required_paths.items():
        path = Path(cast(str, record[path_key]))
        if not path.is_file():
            raise PlayingRlOrchestratorError(f"completed artifact is missing: {path}")
        actual = sha256_file(path)
        if actual != record[sha_key]:
            raise PlayingRlOrchestratorError(
                f"completed artifact SHA mismatch for {path}: {actual} != {record[sha_key]}"
            )
    self_play_dir = Path(cast(str, record["selfPlayDirectory"]))
    if bool(record.get("selfPlayCacheRetained", True)):
        manifest = load_manifest(self_play_dir)
        _consume_samples_for_integrity(self_play_dir)
    else:
        # Completed ephemeral-cache iterations may retain only manifest.json.
        # Shard integrity was already verified before training consumed them.
        manifest = _load_manifest_file(self_play_dir)
    if manifest.game_count != record["gameCount"]:
        raise PlayingRlOrchestratorError(f"{iteration_dir}: gameCount mismatch.")
    if manifest.sample_count != record["sampleCount"]:
        raise PlayingRlOrchestratorError(f"{iteration_dir}: sampleCount mismatch.")
    if manifest.start_seed != record["selfPlayStartSeed"]:
        raise PlayingRlOrchestratorError(f"{iteration_dir}: self-play start seed mismatch.")
    if manifest.end_seed != record["selfPlayEndSeed"]:
        raise PlayingRlOrchestratorError(f"{iteration_dir}: self-play end seed mismatch.")
    manifest_sha = sha256_file(self_play_dir / "manifest.json")
    if manifest_sha != record["selfPlayManifestSha256"]:
        raise PlayingRlOrchestratorError(f"{iteration_dir}: self-play manifest SHA mismatch.")
    if record.get("selfPlayFormat", manifest.format) != manifest.format:
        raise PlayingRlOrchestratorError(f"{iteration_dir}: self-play format mismatch.")
    if record.get("selfPlayShardCount", manifest.shard_count) != manifest.shard_count:
        raise PlayingRlOrchestratorError(f"{iteration_dir}: self-play shard count mismatch.")
    if record.get("selfPlayShardByteLength", _manifest_shard_byte_length(manifest)) != (
        _manifest_shard_byte_length(manifest)
    ):
        raise PlayingRlOrchestratorError(f"{iteration_dir}: self-play shard byte length mismatch.")
    if record.get("selfPlayTensorCompression", _self_play_tensor_compression(manifest)) != (
        _self_play_tensor_compression(manifest)
    ):
        raise PlayingRlOrchestratorError(
            f"{iteration_dir}: self-play tensor compression mismatch."
        )
    if int(cast(int, record["optimizerStepCount"])) <= 0:
        raise PlayingRlOrchestratorError(f"{iteration_dir}: optimizerStepCount must be > 0.")
    if float(cast(float, record["parameterDeltaNorm"])) <= 0:
        raise PlayingRlOrchestratorError(f"{iteration_dir}: parameterDeltaNorm must be > 0.")
    if int(cast(int, record["changedParameterCount"])) <= 0:
        raise PlayingRlOrchestratorError(f"{iteration_dir}: changedParameterCount must be > 0.")


def _validate_evaluation_artifacts(
    config: PlayingRlRunConfig,
    generation: int,
    evaluation_dir: Path,
    summary: Mapping[str, object],
) -> None:
    if summary.get("schemaVersion") != ORCHESTRATOR_SCHEMA_VERSION:
        raise PlayingRlOrchestratorError("evaluation summary schemaVersion mismatch.")
    for key in (
        "checkpointSha256",
        "onnxSha256",
        "metadataSha256",
        "evaluationResultSha256",
    ):
        value = summary.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise PlayingRlOrchestratorError(f"evaluation summary {key} is invalid.")
    checks = {
        _checkpoint_for_generation(config, generation): summary["checkpointSha256"],
        evaluation_dir / "policy.onnx": summary["onnxSha256"],
        evaluation_dir / "policy.json": summary["metadataSha256"],
        evaluation_dir / "evaluation.json": summary["evaluationResultSha256"],
    }
    for path, expected_sha256 in checks.items():
        if not path.is_file():
            raise PlayingRlOrchestratorError(f"completed evaluation artifact is missing: {path}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise PlayingRlOrchestratorError(
                f"completed evaluation artifact SHA mismatch for {path}: "
                f"{actual_sha256} != {expected_sha256}"
            )
    result = _load_json_object(evaluation_dir / "evaluation.json")
    _validate_evaluation_run(config, result)


def _next_iteration(config: PlayingRlRunConfig) -> int:
    next_index = 0
    expected_input_sha256 = sha256_file(config.initial_checkpoint)
    for iteration in range(config.iterations):
        iteration_dir = _iteration_dir(config, iteration)
        record = _load_completed_iteration(iteration_dir)
        if record is None:
            break
        _validate_iteration_artifacts(iteration_dir, record)
        if record["inputCheckpointSha256"] != expected_input_sha256:
            raise PlayingRlOrchestratorError(
                f"{iteration_dir}: input checkpoint chain mismatch: "
                f"{record['inputCheckpointSha256']} != {expected_input_sha256}"
            )
        expected_input_sha256 = _required_str(record["outputCheckpointSha256"])
        next_index = iteration + 1
    return next_index


def _checkpoint_for_generation(config: PlayingRlRunConfig, generation: int) -> Path:
    if generation == 0:
        return config.initial_checkpoint
    if generation < 0 or generation > config.iterations:
        raise PlayingRlOrchestratorError(f"generation out of range: {generation}")
    path = _iteration_dir(config, generation - 1) / "output-checkpoint.pt"
    if not path.is_file():
        raise PlayingRlOrchestratorError(
            f"checkpoint for policy-v{generation} is missing: {path}"
        )
    return path


def _iteration_dir(config: PlayingRlRunConfig, iteration: int) -> Path:
    return config.run_directory / "iterations" / f"iter-{iteration:03d}"


def _quarantine_incomplete_iteration(iteration_dir: Path) -> None:
    _quarantine_incomplete_directory(iteration_dir, label="iteration")


def _quarantine_incomplete_directory(directory: Path, *, label: str) -> None:
    target = directory.with_name(
        f"{directory.name}.incomplete-{int(time.time())}-{os.getpid()}"
    )
    print(f"[resume] moving incomplete {label} aside: {directory} -> {target}", flush=True)
    directory.rename(target)


def _load_state(run_directory: Path) -> dict[str, object]:
    state_path = run_directory / "state.json"
    if not state_path.exists():
        return {
            "schemaVersion": ORCHESTRATOR_SCHEMA_VERSION,
            "completedIterationCount": 0,
            "latestGeneration": 0,
            "completedEvaluations": [],
        }
    return _load_json_object(state_path)


def _write_state(config: PlayingRlRunConfig, state: Mapping[str, object]) -> dict[str, object]:
    data = {
        "schemaVersion": ORCHESTRATOR_SCHEMA_VERSION,
        **dict(state),
    }
    atomic_write_json(config.run_directory / "state.json", data)
    return data


def _validate_resume_config(
    *,
    stored_config: dict[str, object],
    requested_config: dict[str, object],
    provided_config_keys: set[str],
) -> None:
    if stored_config.get("schemaVersion") != ORCHESTRATOR_SCHEMA_VERSION:
        raise PlayingRlOrchestratorError("stored config schemaVersion mismatch.")
    always_check = {
        "initialCheckpointSha256",
        "supervisedManifestSha256",
        "algorithm",
        "rolloutRoster",
        "inferenceDevice",
        "rolloutConcurrency",
        "inferenceMaxBatchSize",
        "simulationBackend",
        "frozenPolicyOnnxSha256",
        "frozenPolicyMetadataSha256",
        "frozenPolicyArtifactId",
    }
    for key in always_check | provided_config_keys:
        if requested_config.get(key) != _stored_config_value(stored_config, key):
            raise PlayingRlOrchestratorError(
                f"resume config mismatch for {key}: "
                f"{requested_config.get(key)!r} != {_stored_config_value(stored_config, key)!r}"
            )


def _config_from_file_dict(
    data: Mapping[str, object],
    *,
    build_typescript: bool,
) -> PlayingRlRunConfig:
    rollout_workers = _required_int(_stored_config_value(data, "rolloutWorkers"))
    return PlayingRlRunConfig(
        run_directory=Path(_required_str(data["runDirectory"])),
        initial_checkpoint=Path(_required_str(data["initialCheckpoint"])),
        supervised_dataset=Path(_required_str(data["supervisedDataset"])),
        iterations=_required_int(data["iterations"]),
        games_per_iteration=_required_int(data["gamesPerIteration"]),
        games_per_shard=_required_int(data["gamesPerShard"]),
        self_play_seed_base=_required_int(data["selfPlaySeedBase"]),
        temperature=_required_float(data["temperature"]),
        rollout_roster=_required_str(_stored_config_value(data, "rolloutRoster")),
        rollout_workers=rollout_workers,
        rollout_concurrency=_required_int(
            data.get("rolloutConcurrency", rollout_workers)
        ),
        inference_max_batch_size=_required_int(
            _stored_config_value(data, "inferenceMaxBatchSize")
        ),
        algorithm=_required_str(_stored_config_value(data, "algorithm")),
        learning_rate=_required_float(data["learningRate"]),
        value_loss_coefficient=_required_float(
            _stored_config_value(data, "valueLossCoefficient")
        ),
        epochs=_required_int(data["epochs"]),
        batch_size=_required_int(data["batchSize"]),
        full_diagnostics_interval=_required_int(
            _stored_config_value(data, "fullDiagnosticsInterval")
        ),
        device=_required_device(_stored_config_value(data, "device")),
        training_seed_base=_required_int(data["trainingSeedBase"]),
        evaluation_interval=_required_int(data["evaluationInterval"]),
        evaluation_start_seed=_required_int(data["evaluationStartSeed"]),
        evaluation_seed_count=_required_int(data["evaluationSeedCount"]),
        inference_device=_required_inference_device(
            _stored_config_value(data, "inferenceDevice")
        ),
        retain_self_play_data=_required_bool(_stored_config_value(data, "retainSelfPlayData")),
        simulation_backend=_required_simulation_backend(
            _stored_config_value(data, "simulationBackend")
        ),
        frozen_policy_onnx=(
            Path(_required_str(data["frozenPolicyOnnx"]))
            if data.get("frozenPolicyOnnx") is not None
            else None
        ),
        frozen_policy_metadata=(
            Path(_required_str(data["frozenPolicyMetadata"]))
            if data.get("frozenPolicyMetadata") is not None
            else None
        ),
        frozen_policy_artifact_id=_required_str(
            _stored_config_value(data, "frozenPolicyArtifactId")
        ),
        build_typescript=build_typescript,
        build_cpp=True,
    ).normalized()


def _validate_config(config: PlayingRlRunConfig) -> None:
    positive_ints = {
        "iterations": config.iterations,
        "games_per_iteration": config.games_per_iteration,
        "games_per_shard": config.games_per_shard,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "full_diagnostics_interval": config.full_diagnostics_interval,
        "evaluation_interval": config.evaluation_interval,
        "evaluation_seed_count": config.evaluation_seed_count,
        "rollout_workers": config.rollout_workers,
        "rollout_concurrency": _effective_rollout_concurrency(config),
        "inference_max_batch_size": config.inference_max_batch_size,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise PlayingRlOrchestratorError(f"{name} must be positive, got {value}.")
    if config.games_per_shard > config.games_per_iteration:
        raise PlayingRlOrchestratorError("games_per_shard must be <= games_per_iteration.")
    if config.temperature <= 0 or not math.isfinite(config.temperature):
        raise PlayingRlOrchestratorError("temperature must be finite and positive.")
    if config.rollout_roster.strip() == "":
        raise PlayingRlOrchestratorError("rollout_roster must be non-empty.")
    if config.algorithm not in PLAYING_RL_ALGORITHMS:
        raise PlayingRlOrchestratorError(
            "algorithm must be one of "
            f"{', '.join(PLAYING_RL_ALGORITHMS)}, got {config.algorithm!r}."
        )
    if config.learning_rate <= 0 or not math.isfinite(config.learning_rate):
        raise PlayingRlOrchestratorError("learning_rate must be finite and positive.")
    if config.device not in SUPPORTED_TORCH_DEVICES:
        raise PlayingRlOrchestratorError(
            "device must be one of "
            f"{', '.join(SUPPORTED_TORCH_DEVICES)}, got {config.device!r}."
        )
    if config.inference_device not in SUPPORTED_INFERENCE_DEVICES:
        raise PlayingRlOrchestratorError(
            "inference_device must be one of "
            f"{', '.join(SUPPORTED_INFERENCE_DEVICES)}, got {config.inference_device!r}."
        )
    if config.simulation_backend not in SUPPORTED_SIMULATION_BACKENDS:
        raise PlayingRlOrchestratorError(
            "simulation_backend must be one of "
            f"{', '.join(SUPPORTED_SIMULATION_BACKENDS)}, got {config.simulation_backend!r}."
        )
    if config.simulation_backend == "cpp":
        if config.frozen_policy_onnx is None or config.frozen_policy_metadata is None:
            raise PlayingRlOrchestratorError(
                "C++ backend requires frozen_policy_onnx and frozen_policy_metadata."
            )
        if config.frozen_policy_artifact_id.strip() == "":
            raise PlayingRlOrchestratorError("frozen_policy_artifact_id must be non-empty.")
        if not config.frozen_policy_onnx.is_file():
            raise PlayingRlOrchestratorError(
                f"frozen policy ONNX does not exist: {config.frozen_policy_onnx}"
            )
        if not config.frozen_policy_metadata.is_file():
            raise PlayingRlOrchestratorError(
                f"frozen policy metadata does not exist: {config.frozen_policy_metadata}"
            )
    if config.device == "cuda":
        try:
            resolve_torch_device(config.device)
        except TorchDeviceResolutionError as error:
            raise PlayingRlOrchestratorError(str(error)) from error
    if (
        not math.isfinite(config.value_loss_coefficient)
        or config.value_loss_coefficient < 0.0
    ):
        raise PlayingRlOrchestratorError(
            "value_loss_coefficient must be finite and non-negative."
        )

    self_play_start = config.self_play_seed_base
    self_play_end = config.self_play_seed_base + config.iterations * config.games_per_iteration - 1
    evaluation_start = config.evaluation_start_seed
    evaluation_end = config.evaluation_start_seed + config.evaluation_seed_count - 1
    training_start = config.training_seed_base
    training_end = config.training_seed_base + config.iterations - 1
    ranges = {
        "self-play": (self_play_start, self_play_end),
        "evaluation": (evaluation_start, evaluation_end),
        "training": (training_start, training_end),
    }
    for name, (start, end) in ranges.items():
        if start < 0 or end > UINT32_MAX or not Number_is_safe_integer(start, end):
            raise PlayingRlOrchestratorError(f"{name} seed range exceeds uint32: {start}..{end}")
    if _ranges_overlap((self_play_start, self_play_end), (evaluation_start, evaluation_end)):
        raise PlayingRlOrchestratorError("self-play and evaluation seed ranges must not overlap.")
    if _ranges_overlap((self_play_start, self_play_end), (training_start, training_end)):
        raise PlayingRlOrchestratorError("self-play and training seed ranges must not overlap.")
    if _ranges_overlap((evaluation_start, evaluation_end), (training_start, training_end)):
        raise PlayingRlOrchestratorError("evaluation and training seed ranges must not overlap.")


def _ensure_inputs(config: PlayingRlRunConfig) -> None:
    if not config.initial_checkpoint.is_file():
        raise PlayingRlOrchestratorError(
            f"initial checkpoint does not exist: {config.initial_checkpoint}"
        )
    if not (config.supervised_dataset / "manifest.json").is_file():
        raise PlayingRlOrchestratorError(
            f"supervised dataset manifest does not exist: {config.supervised_dataset}"
        )


def _build_typescript_helpers() -> None:
    print("[setup] build TypeScript self-play/evaluation helpers", flush=True)
    result = subprocess.run(
        ["pnpm", "--filter", "@napoleon/self-play-cli...", "build"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    if result.returncode != 0:
        raise PlayingRlOrchestratorError(
            f"TypeScript helper build failed with exit {result.returncode}."
        )


def _build_cpp_helpers() -> None:
    print("[setup] build C++ simulation helpers", flush=True)
    result = subprocess.run(
        ["pnpm", "--filter", "@napoleon/cpp-core", "build"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    if result.returncode != 0:
        raise PlayingRlOrchestratorError(
            f"C++ helper build failed with exit {result.returncode}."
        )


def _run_rollout_backend(
    config: PlayingRlRunConfig,
    *,
    behavior_onnx: Path,
    behavior_metadata: Path,
    self_play_dir: Path,
    start_seed: int,
    artifact_id: str,
    progress_prefix: str,
) -> dict[str, object]:
    if config.simulation_backend == "typescript":
        return _run_node_json(
            [
                "node",
                str(_repo_root() / "apps/self-play-cli/dist/playingSelfPlayCli.js"),
                "--onnx",
                str(behavior_onnx),
                "--metadata",
                str(behavior_metadata),
                "--output",
                str(self_play_dir),
                "--start-seed",
                str(start_seed),
                "--games",
                str(config.games_per_iteration),
                "--games-per-shard",
                str(config.games_per_shard),
                "--temperature",
                repr(config.temperature),
                "--rollout-roster",
                config.rollout_roster,
                "--rollout-workers",
                str(config.rollout_workers),
                "--rollout-concurrency",
                str(_effective_rollout_concurrency(config)),
                "--inference-max-batch-size",
                str(config.inference_max_batch_size),
                "--inference-device",
                config.inference_device,
                "--artifact-id",
                artifact_id,
                "--progress-prefix",
                progress_prefix,
            ],
            cwd=_repo_root(),
        )

    frozen_onnx, frozen_metadata = _require_frozen_policy_paths(config)
    started = time.monotonic()
    summary = _run_node_json(
        [
            str(_repo_root() / "packages/cpp-core/build/napoleon_rl_dataset_cli"),
            "--output",
            str(self_play_dir),
            "--start-seed",
            str(start_seed),
            "--games",
            str(config.games_per_iteration),
            "--games-per-shard",
            str(config.games_per_shard),
            "--policy-onnx",
            str(behavior_onnx),
            "--policy-metadata",
            str(behavior_metadata),
            "--policy-artifact-id",
            artifact_id,
            "--frozen-onnx",
            str(frozen_onnx),
            "--frozen-metadata",
            str(frozen_metadata),
            "--frozen-artifact-id",
            config.frozen_policy_artifact_id,
            "--roster-seed",
            str(start_seed),
            "--temperature",
            repr(config.temperature),
            "--max-concurrent-games",
            str(_effective_rollout_concurrency(config)),
            "--inference-max-batch-size",
            str(config.inference_max_batch_size),
            "--inference-device",
            config.inference_device,
            "--policy-backend",
            _cpp_policy_backend(config),
        ],
        cwd=_repo_root(),
    )
    elapsed = time.monotonic() - started
    return _normalize_cpp_rollout_summary(config, summary, elapsed)


def _run_evaluation_backend(
    config: PlayingRlRunConfig,
    *,
    onnx_path: Path,
    metadata_path: Path,
    result_path: Path,
    candidate_id: str,
    progress_prefix: str,
) -> dict[str, object]:
    if config.simulation_backend == "typescript":
        return _run_node_json(
            [
                "node",
                str(_repo_root() / "apps/self-play-cli/dist/policyEvaluationCli.js"),
                "--onnx",
                str(onnx_path),
                "--metadata",
                str(metadata_path),
                "--output",
                str(result_path),
                "--start-seed",
                str(config.evaluation_start_seed),
                "--seed-count",
                str(config.evaluation_seed_count),
                "--inference-device",
                config.inference_device,
                "--progress-prefix",
                progress_prefix,
            ],
            cwd=_repo_root(),
        )

    frozen_onnx, frozen_metadata = _require_frozen_policy_paths(config)
    return _run_node_json(
        [
            str(_repo_root() / "packages/cpp-core/build/napoleon_eval_cli"),
            "--scenario",
            "candidate-vs-opponent-pool",
            "--start-seed",
            str(config.evaluation_start_seed),
            "--seed-count",
            str(config.evaluation_seed_count),
            "--roster-seed",
            str(config.evaluation_start_seed),
            "--max-concurrent-games",
            str(_effective_rollout_concurrency(config)),
            "--inference-max-batch-size",
            str(config.inference_max_batch_size),
            "--candidate-id",
            candidate_id,
            "--frozen-policy-id",
            config.frozen_policy_artifact_id,
            "--candidate-onnx",
            str(onnx_path),
            "--candidate-metadata",
            str(metadata_path),
            "--frozen-onnx",
            str(frozen_onnx),
            "--frozen-metadata",
            str(frozen_metadata),
            "--inference-device",
            config.inference_device,
            "--policy-backend",
            _cpp_policy_backend(config),
            "--output",
            str(result_path),
        ],
        cwd=_repo_root(),
    )


def _normalize_cpp_rollout_summary(
    config: PlayingRlRunConfig,
    summary: Mapping[str, object],
    elapsed_seconds: float,
) -> dict[str, object]:
    inference = _object(summary.get("inference", {}))
    return {
        "outputDirectory": summary["outputDirectory"],
        "gameCount": summary["gameCount"],
        "sampleCount": summary["sampleCount"],
        "shardCount": summary["shardCount"],
        "format": summary["format"],
        "rolloutWorkers": config.rollout_workers,
        "rolloutConcurrency": _effective_rollout_concurrency(config),
        "inferenceMaxBatchSize": config.inference_max_batch_size,
        "requestedInferenceDevice": summary.get(
            "requestedInferenceDevice",
            config.inference_device,
        ),
        "resolvedInferenceDevice": summary.get("resolvedInferenceDevice", "cpu"),
        "executionProvider": summary.get("executionProvider", "cpu"),
        "rolloutElapsedSeconds": summary.get("rolloutElapsedSeconds", elapsed_seconds),
        "inferenceRequestCount": inference.get("requestCount", 0),
        "inferenceSessionRunCount": inference.get("sessionRunCount", 0),
        "inferenceMeanBatchSize": inference.get("meanBatchSize", 0),
        "inferenceMaxObservedBatchSize": inference.get("maxObservedBatchSize", 0),
        "inferenceBatchSizeHistogram": inference.get("batchSizeHistogram", {}),
    }


def _cpp_policy_backend(config: PlayingRlRunConfig) -> str:
    return "onnx" if config.inference_device == "cuda" else "deterministic"


def _require_frozen_policy_paths(config: PlayingRlRunConfig) -> tuple[Path, Path]:
    if config.frozen_policy_onnx is None or config.frozen_policy_metadata is None:
        raise PlayingRlOrchestratorError(
            "C++ backend requires frozen_policy_onnx and frozen_policy_metadata."
        )
    return config.frozen_policy_onnx, config.frozen_policy_metadata


def _run_node_json(command: Sequence[str], *, cwd: Path) -> dict[str, object]:
    output = _run_subprocess(command, cwd=cwd)
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError as error:
        raise PlayingRlOrchestratorError(
            f"subprocess did not return JSON: {output!r}"
        ) from error
    if not isinstance(parsed, dict):
        raise PlayingRlOrchestratorError("subprocess JSON output must be an object.")
    return cast(dict[str, object], parsed)


def _run_subprocess(command: Sequence[str], *, cwd: Path) -> str:
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stderr_lines: list[str] = []
    stdout_lines: list[str] = []
    stdout_thread = threading.Thread(
        target=_read_stdout_lines,
        args=(process.stdout, stdout_lines),
    )
    stdout_thread.start()
    if process.stderr is not None:
        for line in process.stderr:
            stderr_lines.append(line)
            print(line, end="", file=sys.stderr, flush=True)
    stdout_thread.join()
    return_code = process.wait()
    stdout = "".join(stdout_lines)
    if return_code != 0:
        raise PlayingRlOrchestratorError(
            f"command failed with exit {return_code}: {command}\n"
            f"stdout:\n{stdout}\nstderr:\n{''.join(stderr_lines)}"
        )
    return stdout.strip()


def _read_stdout_lines(
    stream: TextIO | None,
    output: list[str],
) -> None:
    if stream is None:
        return
    for line in stream:
        output.append(line)


def _consume_samples_for_integrity(directory: Path) -> None:
    manifest = load_manifest(directory)
    if manifest.format == PLAYING_SELF_PLAY_BINARY_DATASET_FORMAT:
        loader = create_playing_self_play_dataloader(
            directory,
            split=DatasetSplit.TRAIN,
            split_config=SplitConfig(train=100, validation=0, test=0),
            batch_size=4096,
            verify_integrity=True,
        )
        for _ in loader:
            pass
        return

    for _ in iter_raw_samples(directory, verify_integrity=True):
        pass


def _load_manifest_file(directory: Path) -> DatasetManifest:
    path = directory / "manifest.json"
    if not path.is_file():
        raise PlayingRlOrchestratorError(f"self-play manifest is missing: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PlayingRlOrchestratorError(f"self-play manifest is invalid JSON: {path}") from error
    manifest = parse_manifest(raw)
    validate_manifest(manifest)
    return manifest


def _discard_self_play_shards(directory: Path, manifest: DatasetManifest) -> None:
    deleted_count = 0
    deleted_bytes = 0
    for shard in manifest.shards:
        path = directory / shard.file
        path.unlink(missing_ok=True)
        deleted_count += 1
        deleted_bytes += shard.byte_length
    print(
        f"[cache] discarded self-play shards: files={deleted_count} bytes={deleted_bytes}",
        flush=True,
    )


def _discard_completed_self_play_cache_if_due(record: Mapping[str, object]) -> None:
    if bool(record.get("selfPlayCacheRetained", True)):
        return
    self_play_dir = Path(cast(str, record["selfPlayDirectory"]))
    manifest = _load_manifest_file(self_play_dir)
    if any((self_play_dir / shard.file).exists() for shard in manifest.shards):
        _discard_self_play_shards(self_play_dir, manifest)


def _manifest_shard_byte_length(manifest: DatasetManifest) -> int:
    return sum(shard.byte_length for shard in manifest.shards)


def _self_play_tensor_compression(manifest: DatasetManifest) -> str | None:
    if manifest.tensor_schema is None:
        return None
    return manifest.tensor_schema.compression


def _self_play_start_seed(config: PlayingRlRunConfig, iteration: int) -> int:
    return config.self_play_seed_base + iteration * config.games_per_iteration


def _training_seed(config: PlayingRlRunConfig, iteration: int) -> int:
    return config.training_seed_base + iteration


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp.replace(path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, sort_keys=True) + "\n")


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PlayingRlOrchestratorError(f"JSON file not found: {path}") from error
    if not isinstance(parsed, dict):
        raise PlayingRlOrchestratorError(f"JSON file must contain an object: {path}")
    return cast(dict[str, object], parsed)


def _stored_config_value(data: Mapping[str, object], key: str) -> object:
    if key in data:
        return data[key]
    if key == "algorithm":
        return REINFORCE_ALGORITHM
    if key == "valueLossCoefficient":
        return DEFAULT_VALUE_LOSS_COEFFICIENT
    if key == "rolloutRoster":
        return DEFAULT_ROLLOUT_ROSTER
    if key == "rolloutWorkers":
        return DEFAULT_ROLLOUT_WORKERS
    if key == "rolloutConcurrency":
        return data.get("rolloutWorkers", DEFAULT_ROLLOUT_WORKERS)
    if key == "inferenceMaxBatchSize":
        return DEFAULT_INFERENCE_MAX_BATCH_SIZE
    if key == "fullDiagnosticsInterval":
        return 1
    if key == "device":
        return "cpu"
    if key == "inferenceDevice":
        return DEFAULT_INFERENCE_DEVICE
    if key == "retainSelfPlayData":
        return DEFAULT_RETAIN_SELF_PLAY_DATA
    if key == "simulationBackend":
        return DEFAULT_SIMULATION_BACKEND
    if key == "frozenPolicyOnnxSha256":
        return data.get("frozenPolicyOnnxSha256")
    if key == "frozenPolicyMetadataSha256":
        return data.get("frozenPolicyMetadataSha256")
    if key == "frozenPolicyArtifactId":
        return DEFAULT_FROZEN_POLICY_ARTIFACT_ID
    raise KeyError(key)


def _effective_rollout_concurrency(config: PlayingRlRunConfig) -> int:
    if config.rollout_concurrency is None:
        return config.rollout_workers
    return config.rollout_concurrency


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise PlayingRlOrchestratorError("expected JSON object.")
    return cast(dict[str, object], value)


def _rate_value(value: Mapping[str, object]) -> float | None:
    return _optional_float(value["rate"])


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float):
        raise PlayingRlOrchestratorError(f"expected number or null, got {value!r}")
    return float(value)


def _required_int(value: object) -> int:
    if not isinstance(value, int):
        raise PlayingRlOrchestratorError(f"expected integer, got {value!r}")
    return value


def _required_float(value: object) -> float:
    if not isinstance(value, int | float):
        raise PlayingRlOrchestratorError(f"expected number, got {value!r}")
    return float(value)


def _required_str(value: object) -> str:
    if not isinstance(value, str):
        raise PlayingRlOrchestratorError(f"expected string, got {value!r}")
    return value


def _required_bool(value: object) -> bool:
    if not isinstance(value, bool):
        raise PlayingRlOrchestratorError(f"expected boolean, got {value!r}")
    return value


def _required_device(value: object) -> RequestedTorchDevice:
    text = _required_str(value)
    if text not in SUPPORTED_TORCH_DEVICES:
        raise PlayingRlOrchestratorError(
            "device must be one of "
            f"{', '.join(SUPPORTED_TORCH_DEVICES)}, got {text!r}."
        )
    return text


def _required_inference_device(value: object) -> str:
    text = _required_str(value)
    if text not in SUPPORTED_INFERENCE_DEVICES:
        raise PlayingRlOrchestratorError(
            "inferenceDevice must be one of "
            f"{', '.join(SUPPORTED_INFERENCE_DEVICES)}, got {text!r}."
        )
    return text


def _required_simulation_backend(value: object) -> str:
    text = _required_str(value)
    if text not in SUPPORTED_SIMULATION_BACKENDS:
        raise PlayingRlOrchestratorError(
            "simulationBackend must be one of "
            f"{', '.join(SUPPORTED_SIMULATION_BACKENDS)}, got {text!r}."
        )
    return text


def subtract_optional(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _role_breakdown(policy: Mapping[str, object], role: str) -> object:
    roles = cast(list[object], policy["roleResults"])
    for item in roles:
        obj = _object(item)
        if obj["role"] == role:
            return obj
    raise PlayingRlOrchestratorError(f"role breakdown missing: {role}")


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def _ranges_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] <= right[1] and right[0] <= left[1]


def Number_is_safe_integer(*values: int) -> bool:
    return all(abs(value) <= 9_007_199_254_740_991 for value in values)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]
