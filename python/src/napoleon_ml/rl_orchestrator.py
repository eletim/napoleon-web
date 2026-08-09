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
from napoleon_ml.dataset.constants import UINT32_MAX
from napoleon_ml.dataset.manifest import DatasetManifest
from napoleon_ml.dataset.pytorch import create_playing_self_play_dataloader
from napoleon_ml.dataset.reader import iter_raw_samples, load_manifest
from napoleon_ml.dataset.split import DatasetSplit, SplitConfig
from napoleon_ml.policy.actor_critic import (
    ACTOR_CRITIC_ALGORITHM,
    DEFAULT_VALUE_LOSS_COEFFICIENT,
    ActorCriticTrainReport,
    ActorCriticTrainSettings,
    train_policy_actor_critic,
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
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 128
DEFAULT_TRAINING_SEED_BASE = 2_000_000_000
DEFAULT_EVALUATION_INTERVAL = 10
DEFAULT_EVALUATION_START_SEED = 1_000_000_000
DEFAULT_EVALUATION_SEED_COUNT = 100
PLAYING_RL_ALGORITHMS = (REINFORCE_ALGORITHM, ACTOR_CRITIC_ALGORITHM)


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
    algorithm: str = REINFORCE_ALGORITHM
    learning_rate: float = DEFAULT_LEARNING_RATE
    value_loss_coefficient: float = DEFAULT_VALUE_LOSS_COEFFICIENT
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    training_seed_base: int = DEFAULT_TRAINING_SEED_BASE
    evaluation_interval: int = DEFAULT_EVALUATION_INTERVAL
    evaluation_start_seed: int = DEFAULT_EVALUATION_START_SEED
    evaluation_seed_count: int = DEFAULT_EVALUATION_SEED_COUNT
    build_typescript: bool = True

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
            algorithm=self.algorithm,
            learning_rate=self.learning_rate,
            value_loss_coefficient=self.value_loss_coefficient,
            epochs=self.epochs,
            batch_size=self.batch_size,
            training_seed_base=self.training_seed_base,
            evaluation_interval=self.evaluation_interval,
            evaluation_start_seed=self.evaluation_start_seed,
            evaluation_seed_count=self.evaluation_seed_count,
            build_typescript=self.build_typescript,
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
            "algorithm": self.algorithm,
            "learningRate": self.learning_rate,
            "valueLossCoefficient": self.value_loss_coefficient,
            "epochs": self.epochs,
            "batchSize": self.batch_size,
            "trainingSeedBase": self.training_seed_base,
            "evaluationInterval": self.evaluation_interval,
            "evaluationStartSeed": self.evaluation_start_seed,
            "evaluationSeedCount": self.evaluation_seed_count,
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

    if config.build_typescript:
        _build_typescript_helpers()

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
        f"{config.games_per_iteration} games start_seed={start_seed}",
        flush=True,
    )
    _run_node_json(
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
            "--artifact-id",
            f"policy-v{iteration}",
            "--progress-prefix",
            f"[iter {iteration}/{total}] self-play ",
        ],
        cwd=_repo_root(),
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
    print(
        f"[iter {iteration}/{total}] self-play complete: {manifest.sample_count} samples",
        flush=True,
    )

    print(f"[iter {iteration}/{total}] {config.algorithm}", flush=True)
    training_seed = _training_seed(config, iteration)
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
        "selfPlayStartSeed": manifest.start_seed,
        "selfPlayEndSeed": manifest.end_seed,
        "gameCount": manifest.game_count,
        "sampleCount": manifest.sample_count,
        "trainingSeed": training_seed,
        "valueLossCoefficient": (
            config.value_loss_coefficient if config.algorithm == ACTOR_CRITIC_ALGORITHM else None
        ),
        "optimizerStepCount": report.optimizer_step_count,
        "parameterDeltaNorm": report.parameter_delta_norm,
        "changedParameterCount": report.changed_parameter_count,
        "outputCheckpointPath": str(output_checkpoint),
        "outputCheckpointSha256": output_checkpoint_sha256,
        "completedAtUnixSeconds": int(time.time()),
    }
    atomic_write_json(iteration_dir / "iteration.json", iteration_record)
    _validate_iteration_artifacts(iteration_dir, iteration_record)

    summary = {
        "iteration": iteration,
        "algorithm": config.algorithm,
        "selfPlayGameCount": manifest.game_count,
        "sampleCount": manifest.sample_count,
        "positiveRewardCount": report.positive_reward_count,
        "negativeRewardCount": report.negative_reward_count,
        "meanReward": report.mean_reward,
        "forcedSampleCount": report.forced_sample_count,
        "nonForcedSampleCount": report.non_forced_sample_count,
        "optimizerStepCount": report.optimizer_step_count,
        "meanSelectedLogProbabilityBefore": report.mean_selected_log_probability_before,
        "meanSelectedLogProbabilityAfter": report.mean_selected_log_probability_after,
        "maxBehaviorLogProbabilityParityError": (
            report.max_behavior_log_probability_parity_error
        ),
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
    node_summary = _run_node_json(
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
            "--progress-prefix",
            f"[eval v{generation}] ",
        ],
        cwd=_repo_root(),
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
        node_summary=node_summary,
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


def _validate_completed_artifacts(config: PlayingRlRunConfig) -> None:
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
    manifest = load_manifest(self_play_dir)
    _consume_samples_for_integrity(self_play_dir)
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
        algorithm=_required_str(_stored_config_value(data, "algorithm")),
        learning_rate=_required_float(data["learningRate"]),
        value_loss_coefficient=_required_float(
            _stored_config_value(data, "valueLossCoefficient")
        ),
        epochs=_required_int(data["epochs"]),
        batch_size=_required_int(data["batchSize"]),
        training_seed_base=_required_int(data["trainingSeedBase"]),
        evaluation_interval=_required_int(data["evaluationInterval"]),
        evaluation_start_seed=_required_int(data["evaluationStartSeed"]),
        evaluation_seed_count=_required_int(data["evaluationSeedCount"]),
        build_typescript=build_typescript,
    ).normalized()


def _validate_config(config: PlayingRlRunConfig) -> None:
    positive_ints = {
        "iterations": config.iterations,
        "games_per_iteration": config.games_per_iteration,
        "games_per_shard": config.games_per_shard,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "evaluation_interval": config.evaluation_interval,
        "evaluation_seed_count": config.evaluation_seed_count,
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
    for _ in iter_raw_samples(directory, verify_integrity=True):
        pass


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
    raise KeyError(key)


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
