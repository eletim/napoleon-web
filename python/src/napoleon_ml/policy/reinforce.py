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
    PLAYING_SELF_PLAY_REWARD_TYPE,
    PLAYING_SELF_PLAY_REWARD_VERSION,
    PLAYING_SELF_PLAY_SAMPLING_ALGORITHM,
)
from napoleon_ml.dataset.manifest import DatasetManifest
from napoleon_ml.dataset.pytorch import PlayingSelfPlayTorchSample
from napoleon_ml.dataset.tensors import MODEL_INPUT_FEATURE_COUNT, MODEL_INPUT_SCHEMA_VERSION
from napoleon_ml.policy.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    PolicyCheckpointCompatibilityError,
)
from napoleon_ml.policy.model import PolicyMlpConfig, PolicyMlpModel
from napoleon_ml.policy.onnx_export import build_policy_onnx_metadata

BEHAVIOR_LOG_PROB_PARITY_RTOL = 1e-4
BEHAVIOR_LOG_PROB_PARITY_ATOL = 1e-5
REINFORCE_ALGORITHM = "reinforce-v1"


@dataclass(frozen=True)
class ReinforceTrainSettings:
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    verify_integrity: bool
    optimizer: str = "AdamW"

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "epochs": self.epochs,
            "batchSize": self.batch_size,
            "learningRate": self.learning_rate,
            "verifyIntegrity": self.verify_integrity,
            "optimizer": self.optimizer,
        }


@dataclass(frozen=True)
class ReinforceTrainReport:
    sample_count: int
    batch_count: int
    optimizer_step_count: int
    mean_policy_loss: float
    mean_policy_loss_before: float
    mean_policy_loss_after: float
    mean_selected_log_probability_before: float
    mean_selected_log_probability_after: float
    mean_reward: float
    positive_reward_count: int
    negative_reward_count: int
    forced_sample_count: int
    non_forced_sample_count: int
    max_behavior_log_probability_parity_error: float
    parameter_delta_norm: float
    changed_parameter_count: int
    output_checkpoint_path: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "sampleCount": self.sample_count,
            "batchCount": self.batch_count,
            "optimizerStepCount": self.optimizer_step_count,
            "meanPolicyLoss": self.mean_policy_loss,
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
            "parameterDeltaNorm": self.parameter_delta_norm,
            "changedParameterCount": self.changed_parameter_count,
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
    max_behavior_error: float = 0.0
    behavior_parity_failed: bool = False

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
            batch_error = torch.abs(
                selected_log_probability.detach() - behavior_log_probability
            ).max()
            self.max_behavior_error = max(self.max_behavior_error, float(batch_error.item()))
            close = torch.isclose(
                selected_log_probability.detach(),
                behavior_log_probability,
                rtol=BEHAVIOR_LOG_PROB_PARITY_RTOL,
                atol=BEHAVIOR_LOG_PROB_PARITY_ATOL,
            )
            if bool((~close).any().item()):
                self.behavior_parity_failed = True

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


def train_policy_reinforce(
    *,
    input_checkpoint: Path | str,
    self_play_dataset_directory: Path | str,
    output_checkpoint: Path | str,
    manifest: DatasetManifest,
    dataloader: Iterable[PlayingSelfPlayTorchSample],
    settings: ReinforceTrainSettings,
) -> ReinforceTrainReport:
    """Run fail-close behavior parity and REINFORCE updates on CPU."""

    _validate_settings(settings)
    _validate_self_play_manifest(manifest)
    loaded = load_policy_checkpoint_for_reinforce(input_checkpoint, manifest=manifest)
    _validate_behavior_metadata(loaded.model, loaded.checkpoint, manifest=manifest)

    temperature = _require_manifest_temperature(manifest)
    parity = evaluate_reinforce_policy(
        loaded.model,
        dataloader,
        temperature=temperature,
        require_behavior_parity=True,
    )
    _assert_behavior_parity(parity)

    before_parameters = _clone_parameters(loaded.model)
    optimizer = optim.AdamW(loaded.model.parameters(), lr=settings.learning_rate)
    batch_count = 0
    optimizer_step_count = 0
    loss_sum = 0.0
    trained_sample_count = 0

    loaded.model.train()
    for _ in range(settings.epochs):
        for batch in dataloader:
            optimizer.zero_grad(set_to_none=True)
            model_input = batch["model_input"].to(dtype=torch.float32)
            selected = batch["selected_card_index"].to(dtype=torch.long)
            legal_mask = batch["legal_play_mask"].to(dtype=torch.bool)
            reward = batch["terminal_reward"].to(dtype=torch.float32)

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

    if optimizer_step_count == 0:
        raise ValueError("optimizer.step() was not executed; self-play dataset may be empty.")

    after = evaluate_reinforce_policy(
        loaded.model,
        dataloader,
        temperature=temperature,
        require_behavior_parity=False,
    )
    parameter_delta_norm, changed_parameter_count = _parameter_delta(
        before_parameters, loaded.model
    )

    output = Path(output_checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance = _build_rl_provenance(
        input_checkpoint_sha256=loaded.sha256,
        self_play_dataset_directory=Path(self_play_dataset_directory),
        manifest=manifest,
        settings=settings,
        optimizer_step_count=optimizer_step_count,
        sample_count=parity.sample_count,
    )
    _save_reinforce_checkpoint(
        output,
        model=loaded.model,
        parent_checkpoint=loaded.checkpoint,
        rl_provenance=provenance,
    )

    report = ReinforceTrainReport(
        sample_count=parity.sample_count,
        batch_count=batch_count,
        optimizer_step_count=optimizer_step_count,
        mean_policy_loss=loss_sum / trained_sample_count,
        mean_policy_loss_before=parity.mean_loss(),
        mean_policy_loss_after=after.mean_loss(),
        mean_selected_log_probability_before=parity.mean_selected_log_probability(),
        mean_selected_log_probability_after=after.mean_selected_log_probability(),
        mean_reward=parity.mean_reward(),
        positive_reward_count=parity.positive_reward_count,
        negative_reward_count=parity.negative_reward_count,
        forced_sample_count=parity.forced_sample_count,
        non_forced_sample_count=parity.non_forced_sample_count,
        max_behavior_log_probability_parity_error=parity.max_behavior_error,
        parameter_delta_norm=parameter_delta_norm,
        changed_parameter_count=changed_parameter_count,
        output_checkpoint_path=output,
    )
    _assert_report_finite(report)
    return report


@torch.no_grad()
def evaluate_reinforce_policy(
    model: PolicyMlpModel,
    dataloader: Iterable[PlayingSelfPlayTorchSample],
    *,
    temperature: float,
    require_behavior_parity: bool,
) -> _EvaluationAccumulator:
    model.eval()
    accumulator = _EvaluationAccumulator()

    for batch in dataloader:
        model_input = batch["model_input"].to(dtype=torch.float32)
        selected = batch["selected_card_index"].to(dtype=torch.long)
        legal_mask = batch["legal_play_mask"].to(dtype=torch.bool)
        reward = batch["terminal_reward"].to(dtype=torch.float32)
        behavior_log_probability = batch["behavior_log_probability"].to(dtype=torch.float32)

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
    if model_config.get("input_dim") != MODEL_INPUT_FEATURE_COUNT:
        raise PolicyCheckpointCompatibilityError(
            "checkpoint model_config.input_dim mismatch: "
            f"expected {MODEL_INPUT_FEATURE_COUNT}, got {model_config.get('input_dim')!r}."
        )


def _validate_self_play_manifest(manifest: DatasetManifest) -> None:
    if manifest.dataset_schema_version != PLAYING_SELF_PLAY_DATASET_SCHEMA_VERSION:
        raise ValueError(
            "REINFORCE training requires a playing self-play dataset schema v3, "
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


def _validate_behavior_metadata(
    model: PolicyMlpModel,
    checkpoint: dict[str, object],
    *,
    manifest: DatasetManifest,
) -> None:
    if manifest.behavior_policy is None:
        raise ValueError("self-play manifest behaviorPolicy metadata is required.")

    expected_metadata = build_policy_onnx_metadata(model=model, checkpoint=checkpoint)
    if manifest.behavior_policy.metadata != expected_metadata:
        raise PolicyCheckpointCompatibilityError(
            "self-play behaviorPolicy.metadata does not match the input checkpoint. "
            "このtrajectoryは別policyから生成された可能性がある"
        )


def _assert_behavior_parity(parity: _EvaluationAccumulator) -> None:
    max_error = parity.max_behavior_error
    if not math.isfinite(max_error):
        raise ValueError("behavior log probability parity error is not finite.")
    if parity.behavior_parity_failed:
        raise PolicyCheckpointCompatibilityError(
            "behavior log probability parity failed: "
            f"max abs error {max_error:.8g} exceeds rtol={BEHAVIOR_LOG_PROB_PARITY_RTOL} "
            f"and atol={BEHAVIOR_LOG_PROB_PARITY_ATOL}. "
            "このtrajectoryは別policyから生成された可能性がある"
        )


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
    }


def _save_reinforce_checkpoint(
    path: Path,
    *,
    model: PolicyMlpModel,
    parent_checkpoint: dict[str, object],
    rl_provenance: dict[str, object],
) -> None:
    checkpoint = dict(parent_checkpoint)
    checkpoint["model_state"] = model.state_dict()
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
