"""Actor-Critic v1 training for playing self-play trajectories."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, cast

import torch
from torch import Tensor, optim
from torch.nn import functional as F

from napoleon_ml.dataset.constants import (
    DATASET_SCHEMA_VERSION,
    SELF_ROLE_ORDER,
)
from napoleon_ml.dataset.manifest import DatasetManifest
from napoleon_ml.dataset.pytorch import PlayingSelfPlayTorchSample
from napoleon_ml.dataset.tensors import MODEL_INPUT_FEATURE_COUNT, MODEL_INPUT_SCHEMA_VERSION
from napoleon_ml.dataset.playing_variants import (
    model_input_feature_count_for_variant,
    normalize_playing_observation_variant,
    playing_model_input_schema_version_for_variant,
)
from napoleon_ml.policy.behavior_parity import (
    BehaviorParityDiagnostics,
    BehaviorPolicyProvenanceDiagnostic,
    select_behavior_parity_tolerance,
    validate_behavior_policy_provenance,
)
from napoleon_ml.policy.checkpoint import (
    ACTOR_CRITIC_MODEL_ARCHITECTURE,
    ACTOR_CRITIC_MODEL_ARCHITECTURES,
    CHECKPOINT_SCHEMA_VERSION,
    POLICY_MODEL_ARCHITECTURE,
    SEPARATED_ACTOR_CRITIC_MODEL_ARCHITECTURE,
    PolicyCheckpointCompatibilityError,
)
from napoleon_ml.policy.device import (
    RequestedTorchDevice,
    ResolvedTorchDevice,
    cpu_state_dict,
    elapsed_seconds_since,
    playing_self_play_batch_to_device,
    resolve_torch_device,
    start_timing,
)
from napoleon_ml.policy.model import (
    PolicyActorCriticModel,
    PolicyMlpConfig,
    PolicyMlpModel,
    PolicySeparatedActorCriticModel,
    create_actor_critic_from_policy_model,
    create_separated_actor_critic_from_policy_model,
)
from napoleon_ml.policy.reinforce import (
    DEFAULT_BEHAVIOR_PARITY_SUBSET_SIZE,
    _require_manifest_temperature,
    _slice_playing_self_play_batch,
    _TrainingSampleStats,
    _validate_self_play_manifest,
    masked_selected_log_probability,
)

ACTOR_CRITIC_ALGORITHM = "actor-critic-v1"
SEPARATED_ACTOR_CRITIC_ALGORITHM = "actor-critic-separated-v1"
PPO_SEPARATED_ACTOR_CRITIC_ALGORITHM = "ppo-separated-v1"
ACTOR_CRITIC_ALGORITHMS = (
    ACTOR_CRITIC_ALGORITHM,
    SEPARATED_ACTOR_CRITIC_ALGORITHM,
    PPO_SEPARATED_ACTOR_CRITIC_ALGORITHM,
)
DEFAULT_VALUE_LOSS_COEFFICIENT = 0.5
DEFAULT_PPO_CLIP_EPSILON = 0.2
PPO_APPROXIMATE_KL_DEFINITION = "mean(behavior_log_probability - new_log_probability)"
PlayingActorCriticModel = PolicyActorCriticModel | PolicySeparatedActorCriticModel


@dataclass(frozen=True)
class ActorCriticTrainSettings:
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    verify_integrity: bool
    device: RequestedTorchDevice = "cpu"
    value_loss_coefficient: float = DEFAULT_VALUE_LOSS_COEFFICIENT
    optimizer: str = "AdamW"
    full_diagnostics: bool = True
    behavior_parity_subset_size: int = DEFAULT_BEHAVIOR_PARITY_SUBSET_SIZE
    behavior_parity_execution_provider: str | None = None
    behavior_parity_max_observed_batch_size: int | None = None
    algorithm: str = ACTOR_CRITIC_ALGORITHM
    ppo_clip_epsilon: float = DEFAULT_PPO_CLIP_EPSILON

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "epochs": self.epochs,
            "batchSize": self.batch_size,
            "learningRate": self.learning_rate,
            "verifyIntegrity": self.verify_integrity,
            "device": self.device,
            "valueLossCoefficient": self.value_loss_coefficient,
            "optimizer": self.optimizer,
            "fullDiagnostics": self.full_diagnostics,
            "behaviorParitySubsetSize": self.behavior_parity_subset_size,
            "behaviorParityExecutionProvider": self.behavior_parity_execution_provider,
            "behaviorParityMaxObservedBatchSize": self.behavior_parity_max_observed_batch_size,
            "algorithm": self.algorithm,
            "ppoClipEpsilon": self.ppo_clip_epsilon,
        }


@dataclass(frozen=True)
class ActorCriticTrainReport:
    sample_count: int
    batch_count: int
    optimizer_step_count: int
    mean_actor_loss: float
    diagnostics_performed: bool
    behavior_parity_sample_count: int
    behavior_parity_subset_size: int
    actor_loss_before: float | None
    actor_loss_after: float | None
    value_loss_before: float | None
    value_loss_after: float | None
    total_loss_before: float | None
    total_loss_after: float | None
    mean_selected_log_probability_before: float | None
    mean_selected_log_probability_after: float | None
    mean_reward: float
    mean_value_prediction_before: float | None
    mean_value_prediction_after: float | None
    mean_advantage_before: float | None
    mean_advantage_after: float | None
    advantage_std_before: float | None
    advantage_std_after: float | None
    min_value_prediction_before: float | None
    max_value_prediction_before: float | None
    min_value_prediction_after: float | None
    max_value_prediction_after: float | None
    positive_reward_count: int
    negative_reward_count: int
    forced_sample_count: int
    non_forced_sample_count: int
    max_behavior_log_probability_parity_error: float
    behavior_parity_diagnostics: BehaviorParityDiagnostics
    behavior_policy_provenance: BehaviorPolicyProvenanceDiagnostic
    parameter_delta_norm: float
    changed_parameter_count: int
    actor_parameter_delta_norm: float
    critic_parameter_delta_norm: float
    changed_actor_parameter_count: int
    changed_critic_parameter_count: int
    ppo_clip_epsilon: float | None
    ppo_mean_probability_ratio: float | None
    ppo_probability_ratio_std: float | None
    ppo_clipped_sample_count: int | None
    ppo_clipped_fraction: float | None
    ppo_approximate_kl: float | None
    ppo_approximate_kl_definition: str | None
    role_stats_before: dict[str, dict[str, float | int]] | None
    role_stats_after: dict[str, dict[str, float | int]] | None
    requested_device: str
    resolved_device: str
    cuda_device_name: str | None
    safety_validation_elapsed_seconds: float
    pre_eval_elapsed_seconds: float | None
    optimizer_training_elapsed_seconds: float
    post_eval_elapsed_seconds: float | None
    total_elapsed_seconds: float
    output_checkpoint_path: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "sampleCount": self.sample_count,
            "batchCount": self.batch_count,
            "optimizerStepCount": self.optimizer_step_count,
            "meanActorLoss": self.mean_actor_loss,
            "diagnosticsPerformed": self.diagnostics_performed,
            "behaviorParitySampleCount": self.behavior_parity_sample_count,
            "behaviorParitySubsetSize": self.behavior_parity_subset_size,
            "actorLossBefore": self.actor_loss_before,
            "actorLossAfter": self.actor_loss_after,
            "valueLossBefore": self.value_loss_before,
            "valueLossAfter": self.value_loss_after,
            "totalLossBefore": self.total_loss_before,
            "totalLossAfter": self.total_loss_after,
            "meanSelectedLogProbabilityBefore": (
                self.mean_selected_log_probability_before
            ),
            "meanSelectedLogProbabilityAfter": self.mean_selected_log_probability_after,
            "meanReward": self.mean_reward,
            "meanValuePredictionBefore": self.mean_value_prediction_before,
            "meanValuePredictionAfter": self.mean_value_prediction_after,
            "meanAdvantageBefore": self.mean_advantage_before,
            "meanAdvantageAfter": self.mean_advantage_after,
            "advantageStdBefore": self.advantage_std_before,
            "advantageStdAfter": self.advantage_std_after,
            "minValuePredictionBefore": self.min_value_prediction_before,
            "maxValuePredictionBefore": self.max_value_prediction_before,
            "minValuePredictionAfter": self.min_value_prediction_after,
            "maxValuePredictionAfter": self.max_value_prediction_after,
            "positiveRewardCount": self.positive_reward_count,
            "negativeRewardCount": self.negative_reward_count,
            "forcedSampleCount": self.forced_sample_count,
            "nonForcedSampleCount": self.non_forced_sample_count,
            "maxBehaviorLogProbabilityParityError": (
                self.max_behavior_log_probability_parity_error
            ),
            "behaviorParityDiagnostics": self.behavior_parity_diagnostics.to_dict(),
            "behaviorPolicyProvenance": self.behavior_policy_provenance.to_dict(),
            "parameterDeltaNorm": self.parameter_delta_norm,
            "changedParameterCount": self.changed_parameter_count,
            "actorParameterDeltaNorm": self.actor_parameter_delta_norm,
            "criticParameterDeltaNorm": self.critic_parameter_delta_norm,
            "changedActorParameterCount": self.changed_actor_parameter_count,
            "changedCriticParameterCount": self.changed_critic_parameter_count,
            "ppoClipEpsilon": self.ppo_clip_epsilon,
            "ppoMeanProbabilityRatio": self.ppo_mean_probability_ratio,
            "ppoProbabilityRatioStd": self.ppo_probability_ratio_std,
            "ppoClippedSampleCount": self.ppo_clipped_sample_count,
            "ppoClippedFraction": self.ppo_clipped_fraction,
            "ppoApproximateKl": self.ppo_approximate_kl,
            "ppoApproximateKlDefinition": self.ppo_approximate_kl_definition,
            "roleStatsBefore": self.role_stats_before,
            "roleStatsAfter": self.role_stats_after,
            "requestedDevice": self.requested_device,
            "resolvedDevice": self.resolved_device,
            "cudaDeviceName": self.cuda_device_name,
            "safetyValidationElapsedSeconds": self.safety_validation_elapsed_seconds,
            "preEvalElapsedSeconds": self.pre_eval_elapsed_seconds,
            "optimizerTrainingElapsedSeconds": self.optimizer_training_elapsed_seconds,
            "postEvalElapsedSeconds": self.post_eval_elapsed_seconds,
            "totalElapsedSeconds": self.total_elapsed_seconds,
            "outputCheckpointPath": str(self.output_checkpoint_path),
        }


class _LoadedCheckpoint(NamedTuple):
    behavior_model: PolicyMlpModel | PlayingActorCriticModel
    training_model: PlayingActorCriticModel
    checkpoint: dict[str, object]
    sha256: str
    migrated_from_policy: bool


@dataclass
class _RoleAccumulator:
    sample_count: int = 0
    reward_sum: float = 0.0
    value_sum: float = 0.0
    advantage_sum: float = 0.0

    def update(self, *, reward: Tensor, value: Tensor, advantage: Tensor) -> None:
        count = int(reward.shape[0])
        self.sample_count += count
        self.reward_sum += float(reward.detach().sum().item())
        self.value_sum += float(value.detach().sum().item())
        self.advantage_sum += float(advantage.detach().sum().item())

    def to_dict(self) -> dict[str, float | int]:
        if self.sample_count == 0:
            return {
                "sampleCount": 0,
                "meanReward": 0.0,
                "meanValue": 0.0,
                "meanAdvantage": 0.0,
            }
        return {
            "sampleCount": self.sample_count,
            "meanReward": self.reward_sum / self.sample_count,
            "meanValue": self.value_sum / self.sample_count,
            "meanAdvantage": self.advantage_sum / self.sample_count,
        }


@dataclass
class _EvaluationAccumulator:
    sample_count: int = 0
    actor_loss_sum: float = 0.0
    value_loss_sum: float = 0.0
    total_loss_sum: float = 0.0
    selected_log_probability_sum: float = 0.0
    reward_sum: float = 0.0
    value_sum: float = 0.0
    advantage_sum: float = 0.0
    advantage_squared_sum: float = 0.0
    min_value: float = math.inf
    max_value: float = -math.inf
    positive_reward_count: int = 0
    negative_reward_count: int = 0
    forced_sample_count: int = 0
    non_forced_sample_count: int = 0
    behavior_parity_diagnostics: BehaviorParityDiagnostics | None = None
    role_accumulators: dict[str, _RoleAccumulator] | None = None

    def __post_init__(self) -> None:
        self.role_accumulators = {role: _RoleAccumulator() for role in SELF_ROLE_ORDER}

    def update(
        self,
        *,
        selected_log_probability: Tensor,
        reward: Tensor,
        value_prediction: Tensor,
        actor_loss: Tensor,
        value_loss: Tensor,
        total_loss: Tensor,
        legal_mask: Tensor,
        role_index: Tensor,
        behavior_log_probability: Tensor | None = None,
    ) -> None:
        batch_size = int(reward.shape[0])
        advantage = reward - value_prediction
        legal_counts = legal_mask.to(dtype=torch.int64).sum(dim=1)
        forced = legal_counts.eq(1)

        self.sample_count += batch_size
        self.actor_loss_sum += float(actor_loss.detach().item()) * batch_size
        self.value_loss_sum += float(value_loss.detach().item()) * batch_size
        self.total_loss_sum += float(total_loss.detach().item()) * batch_size
        self.selected_log_probability_sum += float(
            selected_log_probability.detach().sum().item()
        )
        self.reward_sum += float(reward.detach().sum().item())
        self.value_sum += float(value_prediction.detach().sum().item())
        self.advantage_sum += float(advantage.detach().sum().item())
        self.advantage_squared_sum += float((advantage.detach() * advantage.detach()).sum().item())
        self.min_value = min(self.min_value, float(value_prediction.detach().min().item()))
        self.max_value = max(self.max_value, float(value_prediction.detach().max().item()))
        self.positive_reward_count += int(reward.eq(1.0).sum().item())
        self.negative_reward_count += int(reward.eq(-1.0).sum().item())
        self.forced_sample_count += int(forced.sum().item())
        self.non_forced_sample_count += int(legal_counts.gt(1).sum().item())

        if self.role_accumulators is None:
            raise AssertionError("role_accumulators must be initialized.")
        for index, role in enumerate(SELF_ROLE_ORDER):
            mask = role_index.eq(index)
            if bool(mask.any().item()):
                self.role_accumulators[role].update(
                    reward=reward[mask],
                    value=value_prediction[mask],
                    advantage=advantage[mask],
                )

        if behavior_log_probability is not None:
            if self.behavior_parity_diagnostics is None:
                raise AssertionError("behavior_parity_diagnostics must be initialized.")
            self.behavior_parity_diagnostics.update(
                selected_log_probability=selected_log_probability,
                behavior_log_probability=behavior_log_probability,
                legal_mask=legal_mask,
            )

    def mean_actor_loss(self) -> float:
        return self.actor_loss_sum / self._nonzero_sample_count()

    def mean_value_loss(self) -> float:
        return self.value_loss_sum / self._nonzero_sample_count()

    def mean_total_loss(self) -> float:
        return self.total_loss_sum / self._nonzero_sample_count()

    def mean_selected_log_probability(self) -> float:
        return self.selected_log_probability_sum / self._nonzero_sample_count()

    def mean_reward(self) -> float:
        return self.reward_sum / self._nonzero_sample_count()

    def mean_value(self) -> float:
        return self.value_sum / self._nonzero_sample_count()

    def mean_advantage(self) -> float:
        return self.advantage_sum / self._nonzero_sample_count()

    def advantage_std(self) -> float:
        mean = self.mean_advantage()
        variance = self.advantage_squared_sum / self._nonzero_sample_count() - mean * mean
        return math.sqrt(max(0.0, variance))

    def role_stats(self) -> dict[str, dict[str, float | int]]:
        if self.role_accumulators is None:
            raise AssertionError("role_accumulators must be initialized.")
        return {
            role: accumulator.to_dict()
            for role, accumulator in self.role_accumulators.items()
        }

    def _nonzero_sample_count(self) -> int:
        if self.sample_count == 0:
            raise ValueError("self-play dataset contains no training samples.")
        return self.sample_count


@dataclass
class _PpoDiagnosticsAccumulator:
    sample_count: int = 0
    ratio_sum: float = 0.0
    ratio_squared_sum: float = 0.0
    clipped_sample_count: int = 0
    approximate_kl_sum: float = 0.0

    def update(
        self,
        *,
        selected_log_probability: Tensor,
        behavior_log_probability: Tensor,
        clip_epsilon: float,
    ) -> None:
        ratio = torch.exp(
            selected_log_probability.detach() - behavior_log_probability.detach()
        )
        clipped = ratio.lt(1.0 - clip_epsilon) | ratio.gt(1.0 + clip_epsilon)
        approximate_kl = behavior_log_probability.detach() - selected_log_probability.detach()
        batch_size = int(ratio.shape[0])

        self.sample_count += batch_size
        self.ratio_sum += float(ratio.sum().item())
        self.ratio_squared_sum += float((ratio * ratio).sum().item())
        self.clipped_sample_count += int(clipped.sum().item())
        self.approximate_kl_sum += float(approximate_kl.sum().item())

    def mean_ratio(self) -> float:
        return self.ratio_sum / self._nonzero_sample_count()

    def ratio_std(self) -> float:
        mean = self.mean_ratio()
        variance = self.ratio_squared_sum / self._nonzero_sample_count() - mean * mean
        return math.sqrt(max(0.0, variance))

    def clipped_fraction(self) -> float:
        return self.clipped_sample_count / self._nonzero_sample_count()

    def approximate_kl(self) -> float:
        return self.approximate_kl_sum / self._nonzero_sample_count()

    def _nonzero_sample_count(self) -> int:
        if self.sample_count == 0:
            raise ValueError("PPO diagnostics contain no samples.")
        return self.sample_count


def train_policy_actor_critic(
    *,
    input_checkpoint: Path | str,
    self_play_dataset_directory: Path | str,
    output_checkpoint: Path | str,
    manifest: DatasetManifest,
    dataloader: Iterable[PlayingSelfPlayTorchSample],
    settings: ActorCriticTrainSettings,
) -> ActorCriticTrainReport:
    """Run fail-close behavior parity and Actor-Critic updates on the requested device."""

    device = resolve_torch_device(settings.device)
    total_start = start_timing(device)
    _validate_settings(settings)
    _validate_self_play_manifest(manifest)
    loaded = load_checkpoint_for_actor_critic(
        input_checkpoint,
        manifest=manifest,
        value_head_seed=settings.seed,
        target_model_architecture=_model_architecture_for_algorithm(settings.algorithm),
    )
    behavior_provenance = validate_behavior_policy_provenance(
        model=loaded.behavior_model,
        checkpoint=loaded.checkpoint,
        manifest=manifest,
        self_play_dataset_directory=Path(self_play_dataset_directory),
        source_checkpoint_sha256=loaded.sha256,
    )
    loaded.training_model.to(device.torch_device)
    frozen_old_critic = (
        _clone_frozen_old_critic(loaded.training_model)
        if _is_ppo_algorithm(settings.algorithm)
        else None
    )

    temperature = _require_manifest_temperature(manifest)
    safety_start = start_timing(device)
    parity = evaluate_actor_critic_policy(
        loaded.training_model,
        dataloader,
        temperature=temperature,
        value_loss_coefficient=settings.value_loss_coefficient,
        require_behavior_parity=True,
        device=device,
        max_samples=None if settings.full_diagnostics else settings.behavior_parity_subset_size,
        behavior_parity_execution_provider=settings.behavior_parity_execution_provider,
        behavior_parity_max_observed_batch_size=(
            settings.behavior_parity_max_observed_batch_size
        ),
        algorithm=settings.algorithm,
        ppo_clip_epsilon=settings.ppo_clip_epsilon,
    )
    safety_elapsed = elapsed_seconds_since(safety_start, device)
    _assert_behavior_parity(parity)
    pre_eval_elapsed = safety_elapsed if settings.full_diagnostics else None

    before_parameters = _clone_parameters(loaded.training_model)
    optimizer = optim.AdamW(loaded.training_model.parameters(), lr=settings.learning_rate)
    batch_count = 0
    optimizer_step_count = 0
    actor_loss_sum = 0.0
    trained_sample_count = 0
    sample_stats = _TrainingSampleStats()
    ppo_training_diagnostics = (
        _PpoDiagnosticsAccumulator() if _is_ppo_algorithm(settings.algorithm) else None
    )

    loaded.training_model.train()
    training_start = start_timing(device)
    for epoch_index in range(settings.epochs):
        for batch in dataloader:
            optimizer.zero_grad(set_to_none=True)
            (
                model_input,
                selected,
                legal_mask,
                reward,
                behavior_log_probability,
            ) = playing_self_play_batch_to_device(batch, device)
            if epoch_index == 0:
                sample_stats.update(reward=reward, legal_mask=legal_mask)

            logits, value_prediction = loaded.training_model.forward_actor_critic(model_input)
            selected_log_probability = masked_selected_log_probability(
                logits,
                selected,
                legal_mask,
                temperature=temperature,
            )
            if _is_ppo_algorithm(settings.algorithm):
                if frozen_old_critic is None or ppo_training_diagnostics is None:
                    raise AssertionError("PPO training requires a frozen old Critic.")
                with torch.no_grad():
                    old_value_prediction = frozen_old_critic(model_input)
                actor_loss, value_loss, total_loss = ppo_actor_critic_losses(
                    selected_log_probability,
                    behavior_log_probability,
                    reward,
                    value_prediction,
                    old_value_prediction,
                    clip_epsilon=settings.ppo_clip_epsilon,
                    value_loss_coefficient=settings.value_loss_coefficient,
                )
                ppo_training_diagnostics.update(
                    selected_log_probability=selected_log_probability,
                    behavior_log_probability=behavior_log_probability,
                    clip_epsilon=settings.ppo_clip_epsilon,
                )
            else:
                actor_loss, value_loss, total_loss = actor_critic_losses(
                    selected_log_probability,
                    reward,
                    value_prediction,
                    value_loss_coefficient=settings.value_loss_coefficient,
                )
            _assert_finite_tensor(total_loss, label="actor-critic loss")
            total_loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

            batch_size = int(reward.shape[0])
            batch_count += 1
            optimizer_step_count += 1
            trained_sample_count += batch_size
            actor_loss_sum += float(actor_loss.detach().item()) * batch_size
    training_elapsed = elapsed_seconds_since(training_start, device)

    if optimizer_step_count == 0:
        raise ValueError("optimizer.step() was not executed; self-play dataset may be empty.")

    after: _EvaluationAccumulator | None = None
    post_eval_elapsed: float | None = None
    if settings.full_diagnostics:
        post_eval_start = start_timing(device)
        after = evaluate_actor_critic_policy(
            loaded.training_model,
            dataloader,
            temperature=temperature,
            value_loss_coefficient=settings.value_loss_coefficient,
            require_behavior_parity=False,
            device=device,
            algorithm=settings.algorithm,
            ppo_clip_epsilon=settings.ppo_clip_epsilon,
        )
        post_eval_elapsed = elapsed_seconds_since(post_eval_start, device)
    parameter_delta_norm, changed_parameter_count = _parameter_delta(
        before_parameters, loaded.training_model
    )
    actor_delta_norm, changed_actor_parameter_count = _parameter_delta_for_prefixes(
        before_parameters,
        loaded.training_model,
        prefixes=_actor_parameter_prefixes(loaded.training_model),
    )
    critic_delta_norm, changed_critic_parameter_count = _parameter_delta_for_prefixes(
        before_parameters,
        loaded.training_model,
        prefixes=_critic_parameter_prefixes(loaded.training_model),
    )
    total_elapsed = elapsed_seconds_since(total_start, device)

    output = Path(output_checkpoint)
    report = ActorCriticTrainReport(
        sample_count=sample_stats.sample_count,
        batch_count=batch_count,
        optimizer_step_count=optimizer_step_count,
        mean_actor_loss=actor_loss_sum / trained_sample_count,
        diagnostics_performed=settings.full_diagnostics,
        behavior_parity_sample_count=parity.sample_count,
        behavior_parity_subset_size=settings.behavior_parity_subset_size,
        actor_loss_before=parity.mean_actor_loss() if settings.full_diagnostics else None,
        actor_loss_after=after.mean_actor_loss() if after is not None else None,
        value_loss_before=parity.mean_value_loss() if settings.full_diagnostics else None,
        value_loss_after=after.mean_value_loss() if after is not None else None,
        total_loss_before=parity.mean_total_loss() if settings.full_diagnostics else None,
        total_loss_after=after.mean_total_loss() if after is not None else None,
        mean_selected_log_probability_before=(
            parity.mean_selected_log_probability() if settings.full_diagnostics else None
        ),
        mean_selected_log_probability_after=(
            after.mean_selected_log_probability() if after is not None else None
        ),
        mean_reward=sample_stats.mean_reward(),
        mean_value_prediction_before=parity.mean_value() if settings.full_diagnostics else None,
        mean_value_prediction_after=after.mean_value() if after is not None else None,
        mean_advantage_before=parity.mean_advantage() if settings.full_diagnostics else None,
        mean_advantage_after=after.mean_advantage() if after is not None else None,
        advantage_std_before=parity.advantage_std() if settings.full_diagnostics else None,
        advantage_std_after=after.advantage_std() if after is not None else None,
        min_value_prediction_before=parity.min_value if settings.full_diagnostics else None,
        max_value_prediction_before=parity.max_value if settings.full_diagnostics else None,
        min_value_prediction_after=after.min_value if after is not None else None,
        max_value_prediction_after=after.max_value if after is not None else None,
        positive_reward_count=sample_stats.positive_reward_count,
        negative_reward_count=sample_stats.negative_reward_count,
        forced_sample_count=sample_stats.forced_sample_count,
        non_forced_sample_count=sample_stats.non_forced_sample_count,
        max_behavior_log_probability_parity_error=(
            _require_behavior_parity_diagnostics(parity).max_abs_error
        ),
        behavior_parity_diagnostics=_require_behavior_parity_diagnostics(parity),
        behavior_policy_provenance=behavior_provenance,
        parameter_delta_norm=parameter_delta_norm,
        changed_parameter_count=changed_parameter_count,
        actor_parameter_delta_norm=actor_delta_norm,
        critic_parameter_delta_norm=critic_delta_norm,
        changed_actor_parameter_count=changed_actor_parameter_count,
        changed_critic_parameter_count=changed_critic_parameter_count,
        ppo_clip_epsilon=(
            settings.ppo_clip_epsilon if _is_ppo_algorithm(settings.algorithm) else None
        ),
        ppo_mean_probability_ratio=(
            ppo_training_diagnostics.mean_ratio()
            if ppo_training_diagnostics is not None
            else None
        ),
        ppo_probability_ratio_std=(
            ppo_training_diagnostics.ratio_std()
            if ppo_training_diagnostics is not None
            else None
        ),
        ppo_clipped_sample_count=(
            ppo_training_diagnostics.clipped_sample_count
            if ppo_training_diagnostics is not None
            else None
        ),
        ppo_clipped_fraction=(
            ppo_training_diagnostics.clipped_fraction()
            if ppo_training_diagnostics is not None
            else None
        ),
        ppo_approximate_kl=(
            ppo_training_diagnostics.approximate_kl()
            if ppo_training_diagnostics is not None
            else None
        ),
        ppo_approximate_kl_definition=(
            PPO_APPROXIMATE_KL_DEFINITION
            if _is_ppo_algorithm(settings.algorithm)
            else None
        ),
        role_stats_before=parity.role_stats() if settings.full_diagnostics else None,
        role_stats_after=after.role_stats() if after is not None else None,
        requested_device=device.requested,
        resolved_device=device.resolved,
        cuda_device_name=device.cuda_device_name,
        safety_validation_elapsed_seconds=safety_elapsed,
        pre_eval_elapsed_seconds=pre_eval_elapsed,
        optimizer_training_elapsed_seconds=training_elapsed,
        post_eval_elapsed_seconds=post_eval_elapsed,
        total_elapsed_seconds=total_elapsed,
        output_checkpoint_path=output,
    )
    _assert_report_finite(report)

    provenance = _build_rl_provenance(
        input_checkpoint_sha256=loaded.sha256,
        self_play_dataset_directory=Path(self_play_dataset_directory),
        manifest=manifest,
        settings=settings,
        optimizer_step_count=optimizer_step_count,
        sample_count=sample_stats.sample_count,
        migrated_from_policy=loaded.migrated_from_policy,
        device=device,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _save_actor_critic_checkpoint(
        output,
        model=loaded.training_model,
        parent_checkpoint=loaded.checkpoint,
        rl_provenance=provenance,
        source_checkpoint_sha256=loaded.sha256,
        migrated_from_policy=loaded.migrated_from_policy,
    )
    return report


@torch.no_grad()
def evaluate_actor_critic_policy(
    model: PlayingActorCriticModel,
    dataloader: Iterable[PlayingSelfPlayTorchSample],
    *,
    temperature: float,
    value_loss_coefficient: float,
    require_behavior_parity: bool,
    device: ResolvedTorchDevice | None = None,
    max_samples: int | None = None,
    behavior_parity_execution_provider: str | None = None,
    behavior_parity_max_observed_batch_size: int | None = None,
    algorithm: str = ACTOR_CRITIC_ALGORITHM,
    ppo_clip_epsilon: float = DEFAULT_PPO_CLIP_EPSILON,
) -> _EvaluationAccumulator:
    if device is None:
        device = resolve_torch_device("cpu")
    if max_samples is not None and max_samples <= 0:
        raise ValueError(f"max_samples must be positive, got {max_samples}.")
    model.to(device.torch_device)
    model.eval()
    parity_tolerance = select_behavior_parity_tolerance(
        execution_provider=behavior_parity_execution_provider,
        max_observed_batch_size=behavior_parity_max_observed_batch_size,
    )
    accumulator = _EvaluationAccumulator(
        behavior_parity_diagnostics=BehaviorParityDiagnostics(
            tolerance=parity_tolerance,
            execution_provider=behavior_parity_execution_provider,
            max_observed_batch_size=behavior_parity_max_observed_batch_size,
        )
        if require_behavior_parity
        else None
    )

    for batch in dataloader:
        if max_samples is not None:
            # The safety subset is the deterministic shard-order prefix capped by
            # behavior_parity_subset_size; full diagnostics pass max_samples=None.
            remaining = max_samples - accumulator.sample_count
            if remaining <= 0:
                break
            batch = _slice_playing_self_play_batch(batch, remaining)
        (
            model_input,
            selected,
            legal_mask,
            reward,
            behavior_log_probability,
        ) = playing_self_play_batch_to_device(batch, device)
        role_index = batch["self_role_index"].to(device=device.torch_device, dtype=torch.long)

        logits, value_prediction = model.forward_actor_critic(model_input)
        selected_log_probability = masked_selected_log_probability(
            logits,
            selected,
            legal_mask,
            temperature=temperature,
        )
        if _is_ppo_algorithm(algorithm):
            actor_loss, value_loss, total_loss = ppo_actor_critic_losses(
                selected_log_probability,
                behavior_log_probability,
                reward,
                value_prediction,
                value_prediction,
                clip_epsilon=ppo_clip_epsilon,
                value_loss_coefficient=value_loss_coefficient,
            )
        else:
            actor_loss, value_loss, total_loss = actor_critic_losses(
                selected_log_probability,
                reward,
                value_prediction,
                value_loss_coefficient=value_loss_coefficient,
            )
        accumulator.update(
            selected_log_probability=selected_log_probability,
            reward=reward,
            value_prediction=value_prediction,
            actor_loss=actor_loss,
            value_loss=value_loss,
            total_loss=total_loss,
            legal_mask=legal_mask,
            role_index=role_index,
            behavior_log_probability=(
                behavior_log_probability if require_behavior_parity else None
            ),
        )

        if max_samples is not None and accumulator.sample_count >= max_samples:
            break

    if accumulator.sample_count == 0:
        raise ValueError("self-play dataset contains no training samples.")

    return accumulator


def actor_critic_losses(
    selected_log_probability: Tensor,
    terminal_reward: Tensor,
    value_prediction: Tensor,
    *,
    value_loss_coefficient: float,
) -> tuple[Tensor, Tensor, Tensor]:
    _validate_actor_critic_loss_inputs(
        selected_log_probability,
        terminal_reward,
        value_prediction,
        value_loss_coefficient=value_loss_coefficient,
    )

    reward = terminal_reward.to(dtype=value_prediction.dtype)
    advantage = reward - value_prediction
    actor_loss = -torch.mean(advantage.detach() * selected_log_probability)
    value_loss = F.mse_loss(value_prediction, reward)
    total_loss = actor_loss + value_loss_coefficient * value_loss
    return actor_loss, value_loss, total_loss


def _validate_actor_critic_loss_inputs(
    selected_log_probability: Tensor,
    terminal_reward: Tensor,
    value_prediction: Tensor,
    *,
    value_loss_coefficient: float,
) -> None:
    if selected_log_probability.shape != terminal_reward.shape:
        raise ValueError(
            "terminal_reward must have the same shape as selected_log_probability: "
            f"{tuple(terminal_reward.shape)} != {tuple(selected_log_probability.shape)}."
        )
    if value_prediction.shape != terminal_reward.shape:
        raise ValueError(
            "value_prediction must have the same shape as terminal_reward: "
            f"{tuple(value_prediction.shape)} != {tuple(terminal_reward.shape)}."
        )
    if bool((~((terminal_reward == 1.0) | (terminal_reward == -1.0))).any().item()):
        raise ValueError("terminal_reward must contain only +1 or -1.")
    if not math.isfinite(value_loss_coefficient) or value_loss_coefficient < 0.0:
        raise ValueError(
            "value_loss_coefficient must be finite and non-negative, "
            f"got {value_loss_coefficient}."
        )


def ppo_clipped_actor_loss(
    selected_log_probability: Tensor,
    behavior_log_probability: Tensor,
    advantage: Tensor,
    *,
    clip_epsilon: float,
) -> Tensor:
    if selected_log_probability.shape != behavior_log_probability.shape:
        raise ValueError(
            "behavior_log_probability must have the same shape as "
            "selected_log_probability: "
            f"{tuple(behavior_log_probability.shape)} != "
            f"{tuple(selected_log_probability.shape)}."
        )
    if advantage.shape != selected_log_probability.shape:
        raise ValueError(
            "advantage must have the same shape as selected_log_probability: "
            f"{tuple(advantage.shape)} != {tuple(selected_log_probability.shape)}."
        )
    _validate_ppo_clip_epsilon(clip_epsilon)

    ratio = torch.exp(selected_log_probability - behavior_log_probability)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    fixed_advantage = advantage.detach()
    objective = torch.minimum(ratio * fixed_advantage, clipped_ratio * fixed_advantage)
    return -torch.mean(objective)


def ppo_actor_critic_losses(
    selected_log_probability: Tensor,
    behavior_log_probability: Tensor,
    terminal_reward: Tensor,
    value_prediction: Tensor,
    old_value_prediction: Tensor,
    *,
    clip_epsilon: float,
    value_loss_coefficient: float,
) -> tuple[Tensor, Tensor, Tensor]:
    _validate_actor_critic_loss_inputs(
        selected_log_probability,
        terminal_reward,
        value_prediction,
        value_loss_coefficient=value_loss_coefficient,
    )
    if old_value_prediction.shape != terminal_reward.shape:
        raise ValueError(
            "old_value_prediction must have the same shape as terminal_reward: "
            f"{tuple(old_value_prediction.shape)} != {tuple(terminal_reward.shape)}."
        )

    reward = terminal_reward.to(dtype=value_prediction.dtype)
    old_value = old_value_prediction.to(dtype=value_prediction.dtype)
    advantage = reward - old_value
    actor_loss = ppo_clipped_actor_loss(
        selected_log_probability,
        behavior_log_probability,
        advantage,
        clip_epsilon=clip_epsilon,
    )
    value_loss = F.mse_loss(value_prediction, reward)
    total_loss = actor_loss + value_loss_coefficient * value_loss
    return actor_loss, value_loss, total_loss


def load_checkpoint_for_actor_critic(
    path: Path | str,
    *,
    manifest: DatasetManifest,
    value_head_seed: int | None = None,
    target_model_architecture: str = ACTOR_CRITIC_MODEL_ARCHITECTURE,
) -> _LoadedCheckpoint:
    checkpoint_path = Path(path)
    raw = _load_raw_checkpoint(checkpoint_path)
    checkpoint = cast(dict[str, object], raw)
    _validate_checkpoint_for_actor_critic(checkpoint, manifest=manifest)
    _validate_target_model_architecture(target_model_architecture)

    model_config_raw = checkpoint.get("model_config")
    if not isinstance(model_config_raw, dict):
        raise PolicyCheckpointCompatibilityError("checkpoint model_config must be a dictionary.")
    model_config = PolicyMlpConfig.from_dict(cast(dict[str, Any], model_config_raw))

    architecture = checkpoint.get("model_architecture", POLICY_MODEL_ARCHITECTURE)
    training_model: PlayingActorCriticModel
    if architecture == POLICY_MODEL_ARCHITECTURE:
        policy_model = PolicyMlpModel(model_config)
        _load_model_state(policy_model, checkpoint)
        if target_model_architecture == ACTOR_CRITIC_MODEL_ARCHITECTURE:
            training_model = create_actor_critic_from_policy_model(
                policy_model,
                seed=value_head_seed,
            )
        elif target_model_architecture == SEPARATED_ACTOR_CRITIC_MODEL_ARCHITECTURE:
            training_model = create_separated_actor_critic_from_policy_model(
                policy_model,
                seed=value_head_seed,
            )
        else:
            raise AssertionError("target model architecture must be validated.")
        behavior_model: PolicyMlpModel | PlayingActorCriticModel = policy_model
        migrated = True
    elif architecture in ACTOR_CRITIC_MODEL_ARCHITECTURES:
        if architecture != target_model_architecture:
            raise PolicyCheckpointCompatibilityError(
                "checkpoint model_architecture mismatch for Actor-Critic training: "
                f"expected {target_model_architecture!r}, got {architecture!r}."
            )
        if architecture == ACTOR_CRITIC_MODEL_ARCHITECTURE:
            actor_critic_model: PlayingActorCriticModel = PolicyActorCriticModel(model_config)
        elif architecture == SEPARATED_ACTOR_CRITIC_MODEL_ARCHITECTURE:
            actor_critic_model = PolicySeparatedActorCriticModel(model_config)
        else:
            raise AssertionError("architecture must be validated.")
        _load_model_state(actor_critic_model, checkpoint)
        behavior_model = actor_critic_model
        training_model = actor_critic_model
        migrated = False
    else:
        raise PolicyCheckpointCompatibilityError(
            f"checkpoint model_architecture is unsupported: {architecture!r}."
        )

    return _LoadedCheckpoint(
        behavior_model=behavior_model,
        training_model=training_model,
        checkpoint=checkpoint,
        sha256=_sha256_file(checkpoint_path),
        migrated_from_policy=migrated,
    )


def _validate_checkpoint_for_actor_critic(
    checkpoint: dict[str, object],
    *,
    manifest: DatasetManifest,
) -> None:
    variant = normalize_playing_observation_variant(manifest.playing_observation_variant)
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "playing_encoder_schema_version": manifest.playing_encoder_schema_version,
        "model_input_schema_version": playing_model_input_schema_version_for_variant(variant),
        "card_ids_sha256": manifest.card_ids_sha256,
    }
    for key, expected_value in expected.items():
        actual = checkpoint.get(key)
        if actual != expected_value:
            raise PolicyCheckpointCompatibilityError(
                f"checkpoint {key} mismatch: expected {expected_value!r}, got {actual!r}."
            )

    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise PolicyCheckpointCompatibilityError("checkpoint model_config must be a dictionary.")
    checkpoint_variant = normalize_playing_observation_variant(
        checkpoint.get("playing_observation_variant")
    )
    if checkpoint_variant != variant:
        raise PolicyCheckpointCompatibilityError(
            "checkpoint playing_observation_variant mismatch: "
            f"expected {variant!r}, got {checkpoint_variant!r}."
        )
    expected_input_dim = model_input_feature_count_for_variant(variant)
    if checkpoint.get("model_input_feature_count", expected_input_dim) != expected_input_dim:
        raise PolicyCheckpointCompatibilityError(
            "checkpoint model_input_feature_count mismatch: "
            f"expected {expected_input_dim}, got {checkpoint.get('model_input_feature_count')!r}."
        )
    if model_config.get("input_dim") != expected_input_dim:
        raise PolicyCheckpointCompatibilityError(
            "checkpoint model_config.input_dim mismatch: "
            f"expected {expected_input_dim}, got {model_config.get('input_dim')!r}."
        )


def _assert_behavior_parity(parity: _EvaluationAccumulator) -> None:
    diagnostics = _require_behavior_parity_diagnostics(parity)
    max_error = diagnostics.max_abs_error
    if not math.isfinite(max_error):
        raise ValueError("behavior log probability parity error is not finite.")
    if diagnostics.failed():
        raise PolicyCheckpointCompatibilityError(
            "behavior log probability parity failed: "
            f"{diagnostics.failure_detail()}; "
            f"max abs error {max_error:.8g}; diagnostics={diagnostics.to_dict()}."
        )


def _require_behavior_parity_diagnostics(
    parity: _EvaluationAccumulator,
) -> BehaviorParityDiagnostics:
    diagnostics = parity.behavior_parity_diagnostics
    if diagnostics is None:
        raise AssertionError("behavior parity diagnostics were not collected.")
    return diagnostics


def _load_raw_checkpoint(path: Path) -> dict[Any, Any]:
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except OSError as error:
        raise PolicyCheckpointCompatibilityError(f"checkpoint cannot be read: {error}") from error
    except RuntimeError as error:
        raise PolicyCheckpointCompatibilityError(f"checkpoint cannot be loaded: {error}") from error
    except pickle.UnpicklingError as error:
        raise PolicyCheckpointCompatibilityError(f"checkpoint cannot be loaded: {error}") from error

    if not isinstance(raw, dict):
        raise PolicyCheckpointCompatibilityError("checkpoint must be a dictionary.")
    return raw


def _load_model_state(
    model: PolicyMlpModel | PlayingActorCriticModel,
    checkpoint: dict[str, object],
) -> None:
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, dict):
        raise PolicyCheckpointCompatibilityError(
            "checkpoint model_state must be a state dictionary."
        )
    try:
        model.load_state_dict(model_state)
    except RuntimeError as error:
        raise PolicyCheckpointCompatibilityError(
            f"checkpoint model_state is incompatible with model_config: {error}"
        ) from error


def _clone_parameters(model: PlayingActorCriticModel) -> dict[str, Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def _parameter_delta(
    before: dict[str, Tensor],
    model: PlayingActorCriticModel,
) -> tuple[float, int]:
    return _parameter_delta_for_prefixes(before, model, prefixes=None)


def _parameter_delta_for_prefixes(
    before: dict[str, Tensor],
    model: PlayingActorCriticModel,
    *,
    prefixes: tuple[str, ...] | None,
) -> tuple[float, int]:
    squared_sum = 0.0
    changed = 0
    for name, parameter in model.named_parameters():
        if prefixes is not None and not name.startswith(prefixes):
            continue
        diff = parameter.detach() - before[name]
        squared_sum += float(torch.sum(diff * diff).item())
        changed += int(torch.ne(parameter.detach(), before[name]).sum().item())
    return math.sqrt(squared_sum), changed


def _actor_parameter_prefixes(model: PlayingActorCriticModel) -> tuple[str, ...]:
    if isinstance(model, PolicySeparatedActorCriticModel):
        return ("actor.",)
    return ("trunk.", "policy_head.")


def _critic_parameter_prefixes(model: PlayingActorCriticModel) -> tuple[str, ...]:
    if isinstance(model, PolicySeparatedActorCriticModel):
        return ("critic.",)
    return ("value_head.",)


def _clone_frozen_old_critic(model: PlayingActorCriticModel) -> torch.nn.Module:
    if not isinstance(model, PolicySeparatedActorCriticModel):
        raise ValueError("PPO Actor-Critic training requires separated Actor/Critic parameters.")
    frozen = deepcopy(model.critic)
    frozen.eval()
    frozen.requires_grad_(False)
    return cast(torch.nn.Module, frozen)


def _build_rl_provenance(
    *,
    input_checkpoint_sha256: str,
    self_play_dataset_directory: Path,
    manifest: DatasetManifest,
    settings: ActorCriticTrainSettings,
    optimizer_step_count: int,
    sample_count: int,
    migrated_from_policy: bool,
    device: ResolvedTorchDevice,
) -> dict[str, object]:
    if manifest.behavior_policy is None or manifest.reward is None:
        raise ValueError("self-play manifest behaviorPolicy and reward metadata are required.")

    return {
        "algorithm": settings.algorithm,
        "modelArchitecture": _model_architecture_for_algorithm(settings.algorithm),
        "parentCheckpointSha256": input_checkpoint_sha256,
        "selfPlayManifestSha256": _sha256_file(self_play_dataset_directory / "manifest.json"),
        "behaviorOnnxSha256": manifest.behavior_policy.onnx_sha256,
        "behaviorMetadataSha256": manifest.behavior_policy.metadata_sha256,
        "temperature": _require_manifest_temperature(manifest),
        "reward": {
            "type": manifest.reward.type,
            "version": manifest.reward.version,
        },
        "optimizer": settings.optimizer,
        "learningRate": settings.learning_rate,
        "valueLossCoefficient": settings.value_loss_coefficient,
        "ppoClipEpsilon": (
            settings.ppo_clip_epsilon if _is_ppo_algorithm(settings.algorithm) else None
        ),
        "epochs": settings.epochs,
        "optimizerSteps": optimizer_step_count,
        "trainingSeed": settings.seed,
        "sampleCount": sample_count,
        "fullDiagnostics": settings.full_diagnostics,
        "behaviorParitySubsetSize": settings.behavior_parity_subset_size,
        "migratedFromPolicyCheckpoint": migrated_from_policy,
        "valueHeadInitializationSeed": settings.seed if migrated_from_policy else None,
        **device.to_metadata(),
    }


def _save_actor_critic_checkpoint(
    path: Path,
    *,
    model: PlayingActorCriticModel,
    parent_checkpoint: dict[str, object],
    rl_provenance: dict[str, object],
    source_checkpoint_sha256: str,
    migrated_from_policy: bool,
) -> None:
    checkpoint = dict(parent_checkpoint)
    model_architecture = _model_architecture_for_model(model)
    checkpoint["model_architecture"] = model_architecture
    checkpoint["model_state"] = cpu_state_dict(model)
    checkpoint["model_config"] = model.config.to_dict()
    checkpoint["rl_provenance"] = rl_provenance
    if migrated_from_policy:
        checkpoint["actor_critic_migration_provenance"] = {
            "migration": _migration_name_for_architecture(model_architecture),
            "sourceCheckpointSha256": source_checkpoint_sha256,
            "sourceModelArchitecture": parent_checkpoint.get(
                "model_architecture", POLICY_MODEL_ARCHITECTURE
            ),
            "targetModelArchitecture": model_architecture,
            "copiedParameters": _copied_parameters_for_architecture(model_architecture),
            "newParameters": _new_parameters_for_architecture(model_architecture),
            "policyLogitsPreserved": True,
        }
    torch.save(checkpoint, path)


def _validate_settings(settings: ActorCriticTrainSettings) -> None:
    if settings.algorithm not in ACTOR_CRITIC_ALGORITHMS:
        raise ValueError(
            "algorithm must be one of "
            f"{', '.join(ACTOR_CRITIC_ALGORITHMS)}, got {settings.algorithm!r}."
        )
    if settings.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {settings.epochs}.")
    if settings.batch_size <= 0:
        raise ValueError(f"batch-size must be positive, got {settings.batch_size}.")
    if settings.learning_rate <= 0.0:
        raise ValueError(f"learning-rate must be positive, got {settings.learning_rate}.")
    if (
        not math.isfinite(settings.value_loss_coefficient)
        or settings.value_loss_coefficient < 0.0
    ):
        raise ValueError(
            "value-loss-coefficient must be finite and non-negative, "
            f"got {settings.value_loss_coefficient}."
        )
    if settings.behavior_parity_subset_size <= 0:
        raise ValueError(
            "behavior_parity_subset_size must be positive, "
            f"got {settings.behavior_parity_subset_size}."
        )
    if (
        settings.behavior_parity_max_observed_batch_size is not None
        and settings.behavior_parity_max_observed_batch_size <= 0
    ):
        raise ValueError(
            "behavior_parity_max_observed_batch_size must be positive when set, "
            f"got {settings.behavior_parity_max_observed_batch_size}."
        )
    _validate_ppo_clip_epsilon(settings.ppo_clip_epsilon)
    if _is_ppo_algorithm(settings.algorithm) and settings.value_loss_coefficient < 0.0:
        raise ValueError("PPO value-loss-coefficient must be non-negative.")


def _model_architecture_for_algorithm(algorithm: str) -> str:
    if algorithm == ACTOR_CRITIC_ALGORITHM:
        return ACTOR_CRITIC_MODEL_ARCHITECTURE
    if algorithm in {
        SEPARATED_ACTOR_CRITIC_ALGORITHM,
        PPO_SEPARATED_ACTOR_CRITIC_ALGORITHM,
    }:
        return SEPARATED_ACTOR_CRITIC_MODEL_ARCHITECTURE
    raise ValueError(
        f"algorithm must be one of {', '.join(ACTOR_CRITIC_ALGORITHMS)}, got {algorithm!r}."
    )


def _validate_target_model_architecture(model_architecture: str) -> None:
    if model_architecture not in ACTOR_CRITIC_MODEL_ARCHITECTURES:
        raise PolicyCheckpointCompatibilityError(
            "target model architecture must be one of "
            f"{', '.join(ACTOR_CRITIC_MODEL_ARCHITECTURES)}, got {model_architecture!r}."
        )


def _model_architecture_for_model(model: PlayingActorCriticModel) -> str:
    if isinstance(model, PolicySeparatedActorCriticModel):
        return SEPARATED_ACTOR_CRITIC_MODEL_ARCHITECTURE
    return ACTOR_CRITIC_MODEL_ARCHITECTURE


def _migration_name_for_architecture(model_architecture: str) -> str:
    if model_architecture == SEPARATED_ACTOR_CRITIC_MODEL_ARCHITECTURE:
        return "policy-mlp-to-playing-separated-actor-critic-v1"
    return "policy-mlp-to-playing-actor-critic-v1"


def _copied_parameters_for_architecture(model_architecture: str) -> list[str]:
    if model_architecture == SEPARATED_ACTOR_CRITIC_MODEL_ARCHITECTURE:
        return ["actor"]
    return ["trunk", "policy_head"]


def _new_parameters_for_architecture(model_architecture: str) -> list[str]:
    if model_architecture == SEPARATED_ACTOR_CRITIC_MODEL_ARCHITECTURE:
        return ["critic"]
    return ["value_head"]


def _is_ppo_algorithm(algorithm: str) -> bool:
    return algorithm == PPO_SEPARATED_ACTOR_CRITIC_ALGORITHM


def _validate_ppo_clip_epsilon(value: float) -> None:
    if not math.isfinite(value) or value <= 0.0 or value >= 1.0:
        raise ValueError(f"ppo_clip_epsilon must be finite and in (0, 1), got {value}.")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_finite_tensor(value: Tensor, *, label: str) -> None:
    if bool((~torch.isfinite(value)).any().item()):
        raise ValueError(f"{label} contains NaN or Infinity.")


def _assert_report_finite(report: ActorCriticTrainReport) -> None:
    def check(value: object, *, path: str) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"train report {path} contains NaN or Infinity.")
        if isinstance(value, dict):
            for key, child in value.items():
                check(child, path=f"{path}.{key}")
        if isinstance(value, list):
            for index, child in enumerate(value):
                check(child, path=f"{path}[{index}]")

    data = report.to_dict()
    check(data, path="report")
    json.dumps(data)
