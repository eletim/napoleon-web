"""REINFORCE v1 training for playing self-play trajectories."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, cast

import torch
from torch import Tensor, optim
from torch.nn import functional as F

from napoleon_ml.dataset.constants import (
    CARD_COUNT,
    DATASET_SCHEMA_VERSION,
    PLAYING_ENCODER_SCHEMA_VERSION,
    PLAYING_MODEL_INPUT_SCHEMA_VERSION,
    PLAYING_SELF_PLAY_DATASET_SCHEMA_VERSION,
    PLAYING_SELF_PLAY_LEGACY_DATASET_SCHEMA_VERSION,
    PLAYING_SELF_PLAY_REWARD_TYPE,
    PLAYING_SELF_PLAY_REWARD_VERSION,
    PLAYING_SELF_PLAY_SAMPLING_ALGORITHM,
)
from napoleon_ml.dataset.manifest import DatasetManifest
from napoleon_ml.dataset.pytorch import PlayingSelfPlayTorchSample
from napoleon_ml.dataset.tensors import MODEL_INPUT_FEATURE_COUNT, MODEL_INPUT_SCHEMA_VERSION
from napoleon_ml.policy.behavior_parity import (
    BehaviorParityDiagnostics,
    BehaviorPolicyProvenanceDiagnostic,
    select_behavior_parity_tolerance,
    validate_behavior_policy_provenance,
)
from napoleon_ml.policy.checkpoint import (
    ACTOR_CRITIC_MODEL_ARCHITECTURE,
    CHECKPOINT_SCHEMA_VERSION,
    POLICY_MODEL_ARCHITECTURE,
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
from napoleon_ml.policy.model import PolicyMlpConfig, PolicyMlpModel

BEHAVIOR_LOG_PROB_PARITY_RTOL = 1e-4
BEHAVIOR_LOG_PROB_PARITY_ATOL = 1e-5
DEFAULT_BEHAVIOR_PARITY_SUBSET_SIZE = 4096
REINFORCE_ALGORITHM = "reinforce-v1"


@dataclass(frozen=True)
class ReinforceTrainSettings:
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    verify_integrity: bool
    device: RequestedTorchDevice = "cpu"
    optimizer: str = "AdamW"
    full_diagnostics: bool = True
    behavior_parity_subset_size: int = DEFAULT_BEHAVIOR_PARITY_SUBSET_SIZE
    behavior_parity_execution_provider: str | None = None
    behavior_parity_max_observed_batch_size: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "epochs": self.epochs,
            "batchSize": self.batch_size,
            "learningRate": self.learning_rate,
            "verifyIntegrity": self.verify_integrity,
            "device": self.device,
            "optimizer": self.optimizer,
            "fullDiagnostics": self.full_diagnostics,
            "behaviorParitySubsetSize": self.behavior_parity_subset_size,
            "behaviorParityExecutionProvider": self.behavior_parity_execution_provider,
            "behaviorParityMaxObservedBatchSize": self.behavior_parity_max_observed_batch_size,
        }


@dataclass(frozen=True)
class ReinforceTrainReport:
    sample_count: int
    batch_count: int
    optimizer_step_count: int
    mean_policy_loss: float
    diagnostics_performed: bool
    behavior_parity_sample_count: int
    behavior_parity_subset_size: int
    mean_policy_loss_before: float | None
    mean_policy_loss_after: float | None
    mean_selected_log_probability_before: float | None
    mean_selected_log_probability_after: float | None
    mean_reward: float
    positive_reward_count: int
    negative_reward_count: int
    forced_sample_count: int
    non_forced_sample_count: int
    max_behavior_log_probability_parity_error: float
    behavior_parity_diagnostics: BehaviorParityDiagnostics
    behavior_policy_provenance: BehaviorPolicyProvenanceDiagnostic
    parameter_delta_norm: float
    changed_parameter_count: int
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
            "meanPolicyLoss": self.mean_policy_loss,
            "diagnosticsPerformed": self.diagnostics_performed,
            "behaviorParitySampleCount": self.behavior_parity_sample_count,
            "behaviorParitySubsetSize": self.behavior_parity_subset_size,
            "meanPolicyLossBefore": self.mean_policy_loss_before,
            "meanPolicyLossAfter": self.mean_policy_loss_after,
            "meanSelectedLogProbabilityBefore": self.mean_selected_log_probability_before,
            "meanSelectedLogProbabilityAfter": self.mean_selected_log_probability_after,
            "meanReward": self.mean_reward,
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


class _LoadedPolicyCheckpoint(NamedTuple):
    model: PolicyMlpModel
    checkpoint: dict[str, object]
    sha256: str


@dataclass
class _EvaluationAccumulator:
    sample_count: int = 0
    loss_sum: float = 0.0
    selected_log_probability_sum: float = 0.0
    reward_sum: float = 0.0
    positive_reward_count: int = 0
    negative_reward_count: int = 0
    forced_sample_count: int = 0
    non_forced_sample_count: int = 0
    behavior_parity_diagnostics: BehaviorParityDiagnostics | None = None

    def update(
        self,
        *,
        selected_log_probability: Tensor,
        reward: Tensor,
        loss: Tensor,
        legal_mask: Tensor,
        behavior_log_probability: Tensor | None = None,
    ) -> None:
        batch_size = int(reward.shape[0])
        legal_counts = legal_mask.to(dtype=torch.int64).sum(dim=1)
        forced = legal_counts.eq(1)

        self.sample_count += batch_size
        self.loss_sum += float(loss.detach().item()) * batch_size
        self.selected_log_probability_sum += float(selected_log_probability.detach().sum().item())
        self.reward_sum += float(reward.detach().sum().item())
        self.positive_reward_count += int(reward.eq(1.0).sum().item())
        self.negative_reward_count += int(reward.eq(-1.0).sum().item())
        self.forced_sample_count += int(forced.sum().item())
        self.non_forced_sample_count += int(legal_counts.gt(1).sum().item())

        if behavior_log_probability is not None:
            if self.behavior_parity_diagnostics is None:
                raise AssertionError("behavior_parity_diagnostics must be initialized.")
            self.behavior_parity_diagnostics.update(
                selected_log_probability=selected_log_probability,
                behavior_log_probability=behavior_log_probability,
                legal_mask=legal_mask,
            )

    def mean_loss(self) -> float:
        if self.sample_count == 0:
            raise ValueError("self-play dataset contains no training samples.")
        return self.loss_sum / self.sample_count

    def mean_selected_log_probability(self) -> float:
        if self.sample_count == 0:
            raise ValueError("self-play dataset contains no training samples.")
        return self.selected_log_probability_sum / self.sample_count

    def mean_reward(self) -> float:
        if self.sample_count == 0:
            raise ValueError("self-play dataset contains no training samples.")
        return self.reward_sum / self.sample_count


@dataclass
class _TrainingSampleStats:
    sample_count: int = 0
    reward_sum: float = 0.0
    positive_reward_count: int = 0
    negative_reward_count: int = 0
    forced_sample_count: int = 0
    non_forced_sample_count: int = 0

    def update(self, *, reward: Tensor, legal_mask: Tensor) -> None:
        legal_counts = legal_mask.to(dtype=torch.int64).sum(dim=1)
        self.sample_count += int(reward.shape[0])
        self.reward_sum += float(reward.detach().sum().item())
        self.positive_reward_count += int(reward.eq(1.0).sum().item())
        self.negative_reward_count += int(reward.eq(-1.0).sum().item())
        self.forced_sample_count += int(legal_counts.eq(1).sum().item())
        self.non_forced_sample_count += int(legal_counts.gt(1).sum().item())

    def mean_reward(self) -> float:
        if self.sample_count == 0:
            raise ValueError("self-play dataset contains no training samples.")
        return self.reward_sum / self.sample_count


def train_policy_reinforce(
    *,
    input_checkpoint: Path | str,
    self_play_dataset_directory: Path | str,
    output_checkpoint: Path | str,
    manifest: DatasetManifest,
    dataloader: Iterable[PlayingSelfPlayTorchSample],
    settings: ReinforceTrainSettings,
) -> ReinforceTrainReport:
    """Run fail-close behavior parity and REINFORCE updates on the requested device."""

    device = resolve_torch_device(settings.device)
    total_start = start_timing(device)
    _validate_settings(settings)
    _validate_self_play_manifest(manifest)
    loaded = load_policy_checkpoint_for_reinforce(input_checkpoint, manifest=manifest)
    behavior_provenance = validate_behavior_policy_provenance(
        model=loaded.model,
        checkpoint=loaded.checkpoint,
        manifest=manifest,
        self_play_dataset_directory=Path(self_play_dataset_directory),
        source_checkpoint_sha256=loaded.sha256,
    )
    loaded.model.to(device.torch_device)

    temperature = _require_manifest_temperature(manifest)
    safety_start = start_timing(device)
    parity = evaluate_reinforce_policy(
        loaded.model,
        dataloader,
        temperature=temperature,
        require_behavior_parity=True,
        device=device,
        max_samples=None if settings.full_diagnostics else settings.behavior_parity_subset_size,
        behavior_parity_execution_provider=settings.behavior_parity_execution_provider,
        behavior_parity_max_observed_batch_size=(
            settings.behavior_parity_max_observed_batch_size
        ),
    )
    safety_elapsed = elapsed_seconds_since(safety_start, device)
    _assert_behavior_parity(parity)
    pre_eval_elapsed = safety_elapsed if settings.full_diagnostics else None

    before_parameters = _clone_parameters(loaded.model)
    optimizer = optim.AdamW(loaded.model.parameters(), lr=settings.learning_rate)
    batch_count = 0
    optimizer_step_count = 0
    loss_sum = 0.0
    trained_sample_count = 0
    sample_stats = _TrainingSampleStats()

    loaded.model.train()
    training_start = start_timing(device)
    for epoch_index in range(settings.epochs):
        for batch in dataloader:
            optimizer.zero_grad(set_to_none=True)
            model_input, selected, legal_mask, reward, _ = playing_self_play_batch_to_device(
                batch,
                device,
            )
            if epoch_index == 0:
                sample_stats.update(reward=reward, legal_mask=legal_mask)

            logits = loaded.model(model_input)
            selected_log_probability = masked_selected_log_probability(
                logits,
                selected,
                legal_mask,
                temperature=temperature,
            )
            loss = reinforce_policy_loss(selected_log_probability, reward)
            _assert_finite_tensor(loss, label="policy loss")
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

            batch_size = int(reward.shape[0])
            batch_count += 1
            optimizer_step_count += 1
            trained_sample_count += batch_size
            loss_sum += float(loss.detach().item()) * batch_size
    training_elapsed = elapsed_seconds_since(training_start, device)

    if optimizer_step_count == 0:
        raise ValueError("optimizer.step() was not executed; self-play dataset may be empty.")

    after: _EvaluationAccumulator | None = None
    post_eval_elapsed: float | None = None
    if settings.full_diagnostics:
        post_eval_start = start_timing(device)
        after = evaluate_reinforce_policy(
            loaded.model,
            dataloader,
            temperature=temperature,
            require_behavior_parity=False,
            device=device,
        )
        post_eval_elapsed = elapsed_seconds_since(post_eval_start, device)
    parameter_delta_norm, changed_parameter_count = _parameter_delta(
        before_parameters, loaded.model
    )
    total_elapsed = elapsed_seconds_since(total_start, device)

    output = Path(output_checkpoint)
    report = ReinforceTrainReport(
        sample_count=sample_stats.sample_count,
        batch_count=batch_count,
        optimizer_step_count=optimizer_step_count,
        mean_policy_loss=loss_sum / trained_sample_count,
        diagnostics_performed=settings.full_diagnostics,
        behavior_parity_sample_count=parity.sample_count,
        behavior_parity_subset_size=settings.behavior_parity_subset_size,
        mean_policy_loss_before=parity.mean_loss() if settings.full_diagnostics else None,
        mean_policy_loss_after=after.mean_loss() if after is not None else None,
        mean_selected_log_probability_before=(
            parity.mean_selected_log_probability() if settings.full_diagnostics else None
        ),
        mean_selected_log_probability_after=(
            after.mean_selected_log_probability() if after is not None else None
        ),
        mean_reward=sample_stats.mean_reward(),
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
        device=device,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _save_reinforce_checkpoint(
        output,
        model=loaded.model,
        parent_checkpoint=loaded.checkpoint,
        rl_provenance=provenance,
    )
    return report


@torch.no_grad()
def evaluate_reinforce_policy(
    model: PolicyMlpModel,
    dataloader: Iterable[PlayingSelfPlayTorchSample],
    *,
    temperature: float,
    require_behavior_parity: bool,
    device: ResolvedTorchDevice | None = None,
    max_samples: int | None = None,
    behavior_parity_execution_provider: str | None = None,
    behavior_parity_max_observed_batch_size: int | None = None,
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

        logits = model(model_input)
        selected_log_probability = masked_selected_log_probability(
            logits,
            selected,
            legal_mask,
            temperature=temperature,
        )
        loss = reinforce_policy_loss(selected_log_probability, reward)
        accumulator.update(
            selected_log_probability=selected_log_probability,
            reward=reward,
            loss=loss,
            legal_mask=legal_mask,
            behavior_log_probability=behavior_log_probability if require_behavior_parity else None,
        )

        if max_samples is not None and accumulator.sample_count >= max_samples:
            break

    if accumulator.sample_count == 0:
        raise ValueError("self-play dataset contains no training samples.")

    return accumulator


def masked_selected_log_probability(
    logits: Tensor,
    selected_card_index: Tensor,
    legal_play_mask: Tensor,
    *,
    temperature: float,
) -> Tensor:
    _validate_reinforce_shapes(logits, selected_card_index, legal_play_mask)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"temperature must be finite and greater than 0, got {temperature}.")

    selected = selected_card_index.to(dtype=torch.long)
    bool_mask = legal_play_mask.to(dtype=torch.bool)
    selected_is_legal = bool_mask.gather(1, selected.unsqueeze(1)).squeeze(1)
    if bool((~selected_is_legal).any().item()):
        raise ValueError("selected_card_index must be legal according to legal_play_mask.")

    legal_counts = bool_mask.to(dtype=torch.int64).sum(dim=1)
    if bool(legal_counts.eq(0).any().item()):
        raise ValueError("legal_play_mask must contain at least one legal card per row.")

    scaled_logits = logits / temperature
    masked_logits = scaled_logits.masked_fill(~bool_mask, torch.finfo(logits.dtype).min)
    log_probabilities = F.log_softmax(masked_logits, dim=1)
    selected_log_probability = log_probabilities.gather(1, selected.unsqueeze(1)).squeeze(1)
    forced_zero = torch.zeros_like(selected_log_probability)
    return torch.where(legal_counts.eq(1), forced_zero, selected_log_probability)


def _slice_playing_self_play_batch(
    batch: PlayingSelfPlayTorchSample,
    max_count: int,
) -> PlayingSelfPlayTorchSample:
    batch_size = int(batch["model_input"].shape[0])
    count = min(max_count, batch_size)
    sliced: dict[str, Tensor] = {}
    for key, value in batch.items():
        if not isinstance(value, Tensor):
            raise TypeError(f"playing self-play batch {key} must be a Tensor.")
        if int(value.shape[0]) != batch_size:
            raise ValueError(
                f"playing self-play batch {key} has inconsistent batch dimension."
            )
        sliced[key] = value[:count]
    return cast(PlayingSelfPlayTorchSample, sliced)


def reinforce_policy_loss(selected_log_probability: Tensor, terminal_reward: Tensor) -> Tensor:
    if selected_log_probability.shape != terminal_reward.shape:
        raise ValueError(
            "terminal_reward must have the same shape as selected_log_probability: "
            f"{tuple(terminal_reward.shape)} != {tuple(selected_log_probability.shape)}."
        )
    if bool((~((terminal_reward == 1.0) | (terminal_reward == -1.0))).any().item()):
        raise ValueError("terminal_reward must contain only +1 or -1.")

    weighted_log_probability = (
        terminal_reward.to(dtype=selected_log_probability.dtype) * selected_log_probability
    )
    return -torch.mean(weighted_log_probability)


def load_policy_checkpoint_for_reinforce(
    path: Path | str,
    *,
    manifest: DatasetManifest,
) -> _LoadedPolicyCheckpoint:
    checkpoint_path = Path(path)
    try:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except OSError as error:
        raise PolicyCheckpointCompatibilityError(f"checkpoint cannot be read: {error}") from error
    except RuntimeError as error:
        raise PolicyCheckpointCompatibilityError(f"checkpoint cannot be loaded: {error}") from error
    except pickle.UnpicklingError as error:
        raise PolicyCheckpointCompatibilityError(f"checkpoint cannot be loaded: {error}") from error

    if not isinstance(raw, dict):
        raise PolicyCheckpointCompatibilityError("checkpoint must be a dictionary.")

    checkpoint = cast(dict[str, object], raw)
    _validate_checkpoint_for_reinforce(checkpoint, manifest=manifest)
    model_config_raw = checkpoint.get("model_config")
    if not isinstance(model_config_raw, dict):
        raise PolicyCheckpointCompatibilityError("checkpoint model_config must be a dictionary.")

    model_config = PolicyMlpConfig.from_dict(cast(dict[str, Any], model_config_raw))
    model = PolicyMlpModel(model_config)
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

    return _LoadedPolicyCheckpoint(
        model=model,
        checkpoint=checkpoint,
        sha256=_sha256_file(checkpoint_path),
    )


def _validate_checkpoint_for_reinforce(
    checkpoint: dict[str, object],
    *,
    manifest: DatasetManifest,
) -> None:
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "playing_encoder_schema_version": manifest.playing_encoder_schema_version,
        "model_input_schema_version": MODEL_INPUT_SCHEMA_VERSION,
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
    architecture = checkpoint.get("model_architecture", POLICY_MODEL_ARCHITECTURE)
    if architecture == ACTOR_CRITIC_MODEL_ARCHITECTURE:
        raise PolicyCheckpointCompatibilityError(
            "REINFORCE training cannot resume from an Actor-Critic checkpoint."
        )
    if architecture != POLICY_MODEL_ARCHITECTURE:
        raise PolicyCheckpointCompatibilityError(
            f"checkpoint model_architecture is unsupported: {architecture!r}."
        )
    rl_provenance = checkpoint.get("rl_provenance")
    if isinstance(rl_provenance, dict) and rl_provenance.get("algorithm") == "actor-critic-v1":
        raise PolicyCheckpointCompatibilityError(
            "REINFORCE training cannot resume from an Actor-Critic checkpoint."
        )
    if model_config.get("input_dim") != MODEL_INPUT_FEATURE_COUNT:
        raise PolicyCheckpointCompatibilityError(
            "checkpoint model_config.input_dim mismatch: "
            f"expected {MODEL_INPUT_FEATURE_COUNT}, got {model_config.get('input_dim')!r}."
        )


def _validate_self_play_manifest(manifest: DatasetManifest) -> None:
    if manifest.dataset_schema_version not in {
        PLAYING_SELF_PLAY_DATASET_SCHEMA_VERSION,
        PLAYING_SELF_PLAY_LEGACY_DATASET_SCHEMA_VERSION,
    }:
        raise ValueError(
            "REINFORCE training requires a playing self-play dataset schema v3 or v4, "
            f"got {manifest.dataset_schema_version}."
        )
    if manifest.playing_encoder_schema_version != PLAYING_ENCODER_SCHEMA_VERSION:
        raise ValueError("self-play playing encoder schema is incompatible.")
    if manifest.playing_model_input_schema_version != PLAYING_MODEL_INPUT_SCHEMA_VERSION:
        raise ValueError("self-play model input schema is incompatible.")
    if manifest.sampling_algorithm != PLAYING_SELF_PLAY_SAMPLING_ALGORITHM:
        raise ValueError(
            "self-play samplingAlgorithm mismatch: "
            f"expected {PLAYING_SELF_PLAY_SAMPLING_ALGORITHM!r}, "
            f"got {manifest.sampling_algorithm!r}."
        )
    if manifest.reward is None:
        raise ValueError("self-play manifest reward metadata is required.")
    if (
        manifest.reward.type != PLAYING_SELF_PLAY_REWARD_TYPE
        or manifest.reward.version != PLAYING_SELF_PLAY_REWARD_VERSION
    ):
        raise ValueError(
            "self-play reward metadata mismatch: expected "
            f"{PLAYING_SELF_PLAY_REWARD_TYPE} v{PLAYING_SELF_PLAY_REWARD_VERSION}."
        )
    _require_manifest_temperature(manifest)


def _assert_behavior_parity(parity: _EvaluationAccumulator) -> None:
    diagnostics = _require_behavior_parity_diagnostics(parity)
    max_error = diagnostics.max_abs_error
    if not math.isfinite(max_error):
        raise ValueError("behavior log probability parity error is not finite.")
    if diagnostics.failed():
        raise PolicyCheckpointCompatibilityError(
            "behavior log probability parity failed: "
            f"{diagnostics.failure_detail()}; "
            f"max abs error {max_error:.8g}; diagnostics={diagnostics.to_dict()}. "
            "このtrajectoryは別policyから生成された可能性がある"
        )


def _require_behavior_parity_diagnostics(
    parity: _EvaluationAccumulator,
) -> BehaviorParityDiagnostics:
    diagnostics = parity.behavior_parity_diagnostics
    if diagnostics is None:
        raise AssertionError("behavior parity diagnostics were not collected.")
    return diagnostics


def _validate_reinforce_shapes(
    logits: Tensor,
    selected_card_index: Tensor,
    legal_mask: Tensor,
) -> None:
    if logits.ndim != 2 or logits.shape[1] != CARD_COUNT:
        raise ValueError(
            f"logits must have shape (batch, {CARD_COUNT}), got {tuple(logits.shape)}."
        )
    if legal_mask.shape != logits.shape:
        raise ValueError(
            f"legal_play_mask must have shape {tuple(logits.shape)}, got {tuple(legal_mask.shape)}."
        )
    if selected_card_index.shape != logits.shape[:1]:
        raise ValueError(
            "selected_card_index must have shape "
            f"{tuple(logits.shape[:1])}, got {tuple(selected_card_index.shape)}."
        )


def _validate_settings(settings: ReinforceTrainSettings) -> None:
    if settings.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {settings.epochs}.")
    if settings.batch_size <= 0:
        raise ValueError(f"batch-size must be positive, got {settings.batch_size}.")
    if settings.learning_rate <= 0.0:
        raise ValueError(f"learning-rate must be positive, got {settings.learning_rate}.")
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


def _clone_parameters(model: PolicyMlpModel) -> dict[str, Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def _parameter_delta(before: dict[str, Tensor], model: PolicyMlpModel) -> tuple[float, int]:
    squared_sum = 0.0
    changed = 0
    for name, parameter in model.named_parameters():
        diff = parameter.detach() - before[name]
        squared_sum += float(torch.sum(diff * diff).item())
        changed += int(torch.ne(parameter.detach(), before[name]).sum().item())
    return math.sqrt(squared_sum), changed


def _build_rl_provenance(
    *,
    input_checkpoint_sha256: str,
    self_play_dataset_directory: Path,
    manifest: DatasetManifest,
    settings: ReinforceTrainSettings,
    optimizer_step_count: int,
    sample_count: int,
    device: ResolvedTorchDevice,
) -> dict[str, object]:
    if manifest.behavior_policy is None or manifest.reward is None:
        raise ValueError("self-play manifest behaviorPolicy and reward metadata are required.")

    return {
        "algorithm": REINFORCE_ALGORITHM,
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
        "epochs": settings.epochs,
        "optimizerSteps": optimizer_step_count,
        "trainingSeed": settings.seed,
        "sampleCount": sample_count,
        "fullDiagnostics": settings.full_diagnostics,
        "behaviorParitySubsetSize": settings.behavior_parity_subset_size,
        **device.to_metadata(),
    }


def _save_reinforce_checkpoint(
    path: Path,
    *,
    model: PolicyMlpModel,
    parent_checkpoint: dict[str, object],
    rl_provenance: dict[str, object],
) -> None:
    checkpoint = dict(parent_checkpoint)
    checkpoint["model_state"] = cpu_state_dict(model)
    checkpoint["rl_provenance"] = rl_provenance
    torch.save(checkpoint, path)


def _require_manifest_temperature(manifest: DatasetManifest) -> float:
    temperature = manifest.temperature
    if temperature is None or not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"self-play manifest temperature must be > 0, got {temperature!r}.")
    return float(temperature)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_finite_tensor(value: Tensor, *, label: str) -> None:
    if bool((~torch.isfinite(value)).any().item()):
        raise ValueError(f"{label} contains NaN or Infinity.")


def _assert_report_finite(report: ReinforceTrainReport) -> None:
    data = report.to_dict()
    for key, value in data.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"train report {key} contains NaN or Infinity.")

    json.dumps(data)
