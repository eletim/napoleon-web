"""Exchange-only PPO training utilities for non-playing RL samples."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from napoleon_ml.dataset.constants import (
    CARD_COUNT,
    EXCHANGE_ENCODER_SCHEMA_VERSION,
)
from napoleon_ml.dataset.tensors import (
    EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
    EXCHANGE_MODEL_INPUT_SCHEMA_VERSION,
)
from napoleon_ml.dataset.validation import calculate_card_ids_sha256
from napoleon_ml.policy.device import (
    RequestedTorchDevice,
    ResolvedTorchDevice,
    cpu_state_dict,
    resolve_torch_device,
)

from .model import (
    ExchangeActorCriticModel,
    ExchangeMlpConfig,
    ExchangeMlpModel,
    create_seeded_exchange_actor_critic_model,
)

NON_PLAYING_RL_SAMPLE_TYPE = "non-playing-exchange-rl-sample"
NON_PLAYING_RL_REWARD_ID = "non-playing-terminal-role-reward-v3"
NON_PLAYING_RL_REWARD_TYPE = "non-playing-terminal-role-reward"
NON_PLAYING_RL_REWARD_VERSION = 3
NON_PLAYING_RL_TERMINAL_REWARD_TRANSFORM_ID = (
    "non-playing-terminal-role-reward-v3-minus-game-player-mean-v1"
)
NON_PLAYING_RL_TERMINAL_REWARD_TRANSFORM_TYPE = "raw-reward-minus-game-player-mean"
NON_PLAYING_RL_TERMINAL_REWARD_TRANSFORM_VERSION = 1
NON_PLAYING_RL_ALL_PASS_RULE_ID = "all-pass-immediate-zero-raw-terminal-reward-v1"
EXCHANGE_PPO_ALGORITHM = "exchange-ppo-sequential-card-v1"
EXCHANGE_ACTOR_CRITIC_MODEL_ARCHITECTURE = "exchange-separated-actor-critic-v1"
EXCHANGE_PPO_CHECKPOINT_SCHEMA_VERSION = 1
EXCHANGE_DECISION_MODE = "sequential-card-v1"
DEFAULT_PPO_CLIP_EPSILON = 0.2
DEFAULT_VALUE_LOSS_COEFFICIENT = 0.5


class ExchangePpoCompatibilityError(ValueError):
    """Raised when a exchange PPO artifact is incompatible with the current schema."""


@dataclass(frozen=True)
class NonPlayingExchangeRlManifest:
    dataset_schema_version: int
    generator_version: int
    sample_type: str
    sample_schema_version: int
    phase_scope: str
    sample_count: int
    behavior_policy: dict[str, object]
    fixed_playing_policy: dict[str, object]
    reward: dict[str, object]
    terminal_reward_transform: dict[str, object]
    raw: dict[str, object]


@dataclass(frozen=True)
class NonPlayingExchangeRlSample:
    seed: int
    step: int
    acting_player_index: int
    model_input: np.ndarray
    legal_discard_card_mask: np.ndarray
    selected_action_index: int
    behavior_log_probability: float
    terminal_reward: float


class ExchangePpoBatch(NamedTuple):
    model_input: Tensor
    legal_discard_card_mask: Tensor
    selected_action_index: Tensor
    behavior_log_probability: Tensor
    terminal_reward: Tensor
    seed: Tensor
    step: Tensor
    acting_player_index: Tensor


@dataclass(frozen=True)
class ExchangePpoLoss:
    actor_loss: Tensor
    value_loss: Tensor
    total_loss: Tensor
    new_log_probability: Tensor
    probability_ratio: Tensor
    clipped_sample_mask: Tensor
    advantage: Tensor


@dataclass(frozen=True)
class ExchangePpoTrainSettings:
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    ppo_clip_epsilon: float = DEFAULT_PPO_CLIP_EPSILON
    value_loss_coefficient: float = DEFAULT_VALUE_LOSS_COEFFICIENT
    optimizer: str = "AdamW"
    training_device: RequestedTorchDevice = "cpu"
    parent_actor_checkpoint: str | None = None
    parent_checkpoint: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "epochs": self.epochs,
            "batchSize": self.batch_size,
            "learningRate": self.learning_rate,
            "ppoClipEpsilon": self.ppo_clip_epsilon,
            "valueLossCoefficient": self.value_loss_coefficient,
            "optimizer": self.optimizer,
            "trainingDevice": self.training_device,
            "algorithm": EXCHANGE_PPO_ALGORITHM,
            "parentActorCheckpoint": self.parent_actor_checkpoint,
            "parentCheckpoint": self.parent_checkpoint,
        }


@dataclass(frozen=True)
class ExchangePpoTrainReport:
    sample_count: int
    optimizer_step_count: int
    mean_actor_loss: float
    mean_value_loss: float
    mean_total_loss: float
    mean_reward: float
    mean_probability_ratio: float
    clipped_fraction: float
    forced_sample_count: int
    output_checkpoint_path: Path
    requested_training_device: str
    resolved_training_device: str
    cuda_device_name: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "sampleCount": self.sample_count,
            "optimizerStepCount": self.optimizer_step_count,
            "meanActorLoss": self.mean_actor_loss,
            "meanValueLoss": self.mean_value_loss,
            "meanTotalLoss": self.mean_total_loss,
            "meanReward": self.mean_reward,
            "meanProbabilityRatio": self.mean_probability_ratio,
            "clippedFraction": self.clipped_fraction,
            "forcedSampleCount": self.forced_sample_count,
            "outputCheckpointPath": str(self.output_checkpoint_path),
            "trainingDevice": {
                "requestedDevice": self.requested_training_device,
                "resolvedDevice": self.resolved_training_device,
                "cudaDeviceName": self.cuda_device_name,
            },
            "requestedTrainingDevice": self.requested_training_device,
            "resolvedTrainingDevice": self.resolved_training_device,
            "cudaDeviceName": self.cuda_device_name,
        }


class NonPlayingExchangeRlDataset(Dataset[NonPlayingExchangeRlSample]):
    def __init__(self, samples: list[NonPlayingExchangeRlSample]) -> None:
        if len(samples) == 0:
            raise ExchangePpoCompatibilityError("non-playing exchange RL dataset has no samples.")
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> NonPlayingExchangeRlSample:
        return self._samples[index]


def load_non_playing_exchange_rl_manifest(
    dataset_directory: Path | str,
) -> NonPlayingExchangeRlManifest:
    path = Path(dataset_directory) / "manifest.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ExchangePpoCompatibilityError(f"manifest cannot be read: {error}") from error
    except json.JSONDecodeError as error:
        raise ExchangePpoCompatibilityError(f"manifest is not valid JSON: {error}") from error

    if not isinstance(raw, dict):
        raise ExchangePpoCompatibilityError("manifest must be a JSON object.")
    if raw.get("sampleType") != NON_PLAYING_RL_SAMPLE_TYPE:
        raise ExchangePpoCompatibilityError(
            f"manifest sampleType must be {NON_PLAYING_RL_SAMPLE_TYPE!r}."
        )
    if raw.get("phaseScope") != "exchange-only":
        raise ExchangePpoCompatibilityError("manifest phaseScope must be 'exchange-only'.")
    if raw.get("decisionMode") != EXCHANGE_DECISION_MODE:
        raise ExchangePpoCompatibilityError("manifest decisionMode must be 'sequential-card-v1'.")
    if raw.get("exchangeEncoderSchemaVersion") != EXCHANGE_ENCODER_SCHEMA_VERSION:
        raise ExchangePpoCompatibilityError("manifest exchangeEncoderSchemaVersion mismatch.")
    if raw.get("exchangeModelInputSchemaVersion") != EXCHANGE_MODEL_INPUT_SCHEMA_VERSION:
        raise ExchangePpoCompatibilityError("manifest exchangeModelInputSchemaVersion mismatch.")
    if raw.get("exchangeModelInputFeatureCount") != EXCHANGE_MODEL_INPUT_FEATURE_COUNT:
        raise ExchangePpoCompatibilityError("manifest exchangeModelInputFeatureCount mismatch.")
    if raw.get("actionCount") != CARD_COUNT:
        raise ExchangePpoCompatibilityError("manifest actionCount mismatch.")
    if raw.get("cardIdsSha256") != calculate_card_ids_sha256():
        raise ExchangePpoCompatibilityError("manifest cardIdsSha256 mismatch.")

    reward = _require_dict(raw.get("reward"), "manifest.reward")
    if (
        reward.get("type") != NON_PLAYING_RL_REWARD_TYPE
        or reward.get("version") != NON_PLAYING_RL_REWARD_VERSION
        or reward.get("id") != NON_PLAYING_RL_REWARD_ID
    ):
        raise ExchangePpoCompatibilityError("manifest reward metadata mismatch.")
    all_pass_rule = _require_dict(raw.get("allPassRule"), "manifest.allPassRule")
    if (
        all_pass_rule.get("id") != NON_PLAYING_RL_ALL_PASS_RULE_ID
        or all_pass_rule.get("starterPayoff") != 0
        or all_pass_rule.get("otherPayoff") != 0
    ):
        raise ExchangePpoCompatibilityError("manifest allPassRule metadata mismatch.")
    terminal_reward_transform = _require_terminal_reward_transform(
        raw.get("terminalRewardTransform"), "manifest.terminalRewardTransform"
    )

    sample_count = _require_non_negative_int(raw.get("sampleCount"), "manifest.sampleCount")

    return NonPlayingExchangeRlManifest(
        dataset_schema_version=_require_int(
            raw.get("datasetSchemaVersion"), "manifest.datasetSchemaVersion"
        ),
        generator_version=_require_int(raw.get("generatorVersion"), "manifest.generatorVersion"),
        sample_type=cast(str, raw["sampleType"]),
        sample_schema_version=_require_int(
            raw.get("sampleSchemaVersion"), "manifest.sampleSchemaVersion"
        ),
        phase_scope=cast(str, raw["phaseScope"]),
        sample_count=sample_count,
        behavior_policy=_require_dict(raw.get("behaviorPolicy"), "manifest.behaviorPolicy"),
        fixed_playing_policy=_require_dict(
            raw.get("fixedPlayingPolicy"), "manifest.fixedPlayingPolicy"
        ),
        reward=reward,
        terminal_reward_transform=terminal_reward_transform,
        raw=cast(dict[str, object], raw),
    )


def iter_non_playing_exchange_rl_samples(
    dataset_directory: Path | str,
) -> Iterator[NonPlayingExchangeRlSample]:
    directory = Path(dataset_directory)
    manifest = load_non_playing_exchange_rl_manifest(directory)
    shards = manifest.raw.get("shards")
    if not isinstance(shards, list):
        raise ExchangePpoCompatibilityError("manifest.shards must be a list.")

    count = 0
    for shard_index, shard in enumerate(shards):
        shard_obj = _require_dict(shard, f"manifest.shards[{shard_index}]")
        file_name = shard_obj.get("file")
        if not isinstance(file_name, str) or "/" in file_name or "\\" in file_name:
            raise ExchangePpoCompatibilityError("manifest shard file must be a bare file name.")
        shard_path = directory / file_name
        try:
            lines = shard_path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise ExchangePpoCompatibilityError(f"shard cannot be read: {error}") from error
        expected_count = _require_non_negative_int(
            shard_obj.get("sampleCount"), f"manifest.shards[{shard_index}].sampleCount"
        )
        if len(lines) != expected_count:
            raise ExchangePpoCompatibilityError(
                f"{file_name}: expected {expected_count} samples, got {len(lines)}."
            )
        for line_number, line in enumerate(lines, start=1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ExchangePpoCompatibilityError(
                    f"{file_name}:{line_number}: invalid JSON: {error}"
                ) from error
            yield _parse_rl_sample(raw, context=f"{file_name}:{line_number}")
            count += 1

    if count != manifest.sample_count:
        raise ExchangePpoCompatibilityError(
            f"manifest sampleCount mismatch: expected {manifest.sample_count}, got {count}."
        )


def create_non_playing_exchange_rl_dataloader(
    dataset_directory: Path | str,
    *,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader[ExchangePpoBatch]:
    samples = list(iter_non_playing_exchange_rl_samples(dataset_directory))
    dataset = NonPlayingExchangeRlDataset(samples)
    return cast(
        DataLoader[ExchangePpoBatch],
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_collate_exchange_ppo_batch,
            num_workers=0,
        ),
    )


def masked_selected_log_probability(
    logits: Tensor,
    selected_action_index: Tensor,
    legal_mask: Tensor,
) -> Tensor:
    if logits.ndim != 2 or logits.shape[1] != CARD_COUNT:
        raise ValueError(f"logits must have shape (batch, {CARD_COUNT}).")
    if legal_mask.shape != logits.shape:
        raise ValueError("legal_mask must have the same shape as logits.")
    if selected_action_index.ndim != 1 or selected_action_index.shape[0] != logits.shape[0]:
        raise ValueError("selected_action_index must have shape (batch,).")

    mask = legal_mask.to(dtype=torch.bool)
    selected_legal = mask.gather(1, selected_action_index.reshape(-1, 1)).squeeze(1)
    if not bool(selected_legal.all().item()):
        raise ValueError("selected_action_index contains an illegal action.")

    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    log_probabilities = F.log_softmax(masked_logits, dim=1)
    forced = mask.to(dtype=torch.int64).sum(dim=1).eq(1)
    selected = log_probabilities.gather(1, selected_action_index.reshape(-1, 1)).squeeze(1)
    return torch.where(forced, torch.zeros_like(selected), selected)


def exchange_ppo_loss(
    *,
    logits: Tensor,
    value_prediction: Tensor,
    selected_action_index: Tensor,
    legal_discard_card_mask: Tensor,
    behavior_log_probability: Tensor,
    terminal_reward: Tensor,
    clip_epsilon: float = DEFAULT_PPO_CLIP_EPSILON,
    value_loss_coefficient: float = DEFAULT_VALUE_LOSS_COEFFICIENT,
) -> ExchangePpoLoss:
    if clip_epsilon <= 0.0:
        raise ValueError("clip_epsilon must be positive.")
    if value_loss_coefficient < 0.0:
        raise ValueError("value_loss_coefficient must be non-negative.")

    new_log_probability = masked_selected_log_probability(
        logits,
        selected_action_index,
        legal_discard_card_mask,
    )
    advantage = terminal_reward - value_prediction.detach()
    probability_ratio = torch.exp(new_log_probability - behavior_log_probability)
    clipped_ratio = torch.clamp(probability_ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    unclipped_objective = probability_ratio * advantage
    clipped_objective = clipped_ratio * advantage
    actor_loss = -torch.minimum(unclipped_objective, clipped_objective).mean()
    value_loss = F.mse_loss(value_prediction, terminal_reward)
    total_loss = actor_loss + value_loss_coefficient * value_loss
    clipped_sample_mask = (probability_ratio - clipped_ratio).abs().gt(1e-12)
    return ExchangePpoLoss(
        actor_loss=actor_loss,
        value_loss=value_loss,
        total_loss=total_loss,
        new_log_probability=new_log_probability,
        probability_ratio=probability_ratio,
        clipped_sample_mask=clipped_sample_mask,
        advantage=advantage,
    )


def train_exchange_ppo(
    *,
    dataset_directory: Path | str,
    output_checkpoint_path: Path | str,
    settings: ExchangePpoTrainSettings,
    model_config: ExchangeMlpConfig,
) -> ExchangePpoTrainReport:
    manifest = load_non_playing_exchange_rl_manifest(dataset_directory)
    training_device = resolve_torch_device(
        settings.training_device,
        flag_name="--training-device",
    )
    torch.manual_seed(settings.seed)
    model = create_seeded_exchange_actor_critic_model(model_config, seed=settings.seed)
    parent_sha256: str | None = None
    parent_checkpoint_sha256: str | None = None
    if settings.parent_checkpoint is not None:
        parent_checkpoint_sha256 = initialize_model_from_checkpoint(
            model,
            settings.parent_checkpoint,
        )
    elif settings.parent_actor_checkpoint is not None:
        parent_sha256 = initialize_actor_from_checkpoint(
            model,
            settings.parent_actor_checkpoint,
        )
    model.to(training_device.torch_device)

    dataloader = create_non_playing_exchange_rl_dataloader(
        dataset_directory,
        batch_size=settings.batch_size,
        shuffle=False,
    )
    optimizer = optim.AdamW(model.parameters(), lr=settings.learning_rate)
    totals = _LossTotals()

    for _epoch in range(settings.epochs):
        model.train()
        for batch in dataloader:
            batch = _exchange_ppo_batch_to_device(batch, training_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.model_input)
            values = model.value(batch.model_input)
            loss = exchange_ppo_loss(
                logits=logits,
                value_prediction=values,
                selected_action_index=batch.selected_action_index,
                legal_discard_card_mask=batch.legal_discard_card_mask,
                behavior_log_probability=batch.behavior_log_probability,
                terminal_reward=batch.terminal_reward,
                clip_epsilon=settings.ppo_clip_epsilon,
                value_loss_coefficient=settings.value_loss_coefficient,
            )
            loss.total_loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            totals.update(batch=batch, loss=loss)

    if totals.sample_count == 0:
        raise ExchangePpoCompatibilityError("training produced no PPO batches.")

    output = Path(output_checkpoint_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_exchange_ppo_checkpoint(
        output,
        model=model,
        settings=settings,
        manifest=manifest,
        parent_actor_checkpoint_sha256=parent_sha256,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
    )
    return totals.to_report(output, device=training_device)


def initialize_model_from_checkpoint(
    model: ExchangeActorCriticModel,
    checkpoint_path: Path | str,
) -> str:
    checkpoint_file = Path(checkpoint_path)
    raw = _load_raw_checkpoint(checkpoint_file)
    _validate_exchange_ppo_checkpoint(raw)
    model_config_raw = raw.get("model_config")
    if not isinstance(model_config_raw, dict):
        raise ExchangePpoCompatibilityError("parent checkpoint model_config must be a dictionary.")
    parent_config = ExchangeMlpConfig.from_dict(model_config_raw)
    if parent_config != model.config:
        raise ExchangePpoCompatibilityError(
            "parent checkpoint model_config must match PPO model_config."
        )
    state = raw.get("model_state")
    if not isinstance(state, dict):
        raise ExchangePpoCompatibilityError("parent checkpoint model_state must be a dictionary.")
    try:
        model.load_state_dict(state)
    except RuntimeError as error:
        raise ExchangePpoCompatibilityError(
            f"parent checkpoint model_state is incompatible: {error}"
        ) from error
    return _sha256_file(checkpoint_file)


def initialize_actor_from_checkpoint(
    model: ExchangeActorCriticModel,
    checkpoint_path: Path | str,
) -> str:
    checkpoint_file = Path(checkpoint_path)
    raw = _load_raw_checkpoint(checkpoint_file)
    model_config_raw = raw.get("model_config")
    if not isinstance(model_config_raw, dict):
        raise ExchangePpoCompatibilityError("parent checkpoint model_config must be a dictionary.")
    parent_config = ExchangeMlpConfig.from_dict(model_config_raw)
    if parent_config != model.config:
        raise ExchangePpoCompatibilityError(
            "parent checkpoint model_config must match PPO actor model_config."
        )
    state = raw.get("model_state")
    if not isinstance(state, dict):
        raise ExchangePpoCompatibilityError("parent checkpoint model_state must be a dictionary.")

    actor_state = _extract_actor_state_dict(state)
    try:
        model.actor.load_state_dict(actor_state)
    except RuntimeError as error:
        raise ExchangePpoCompatibilityError(
            f"parent checkpoint actor state is incompatible: {error}"
        ) from error

    return _sha256_file(checkpoint_file)


def save_exchange_ppo_checkpoint(
    path: Path | str,
    *,
    model: ExchangeActorCriticModel,
    settings: ExchangePpoTrainSettings,
    manifest: NonPlayingExchangeRlManifest,
    parent_actor_checkpoint_sha256: str | None,
    parent_checkpoint_sha256: str | None = None,
) -> None:
    checkpoint = {
        "checkpoint_schema_version": EXCHANGE_PPO_CHECKPOINT_SCHEMA_VERSION,
        "model_architecture": EXCHANGE_ACTOR_CRITIC_MODEL_ARCHITECTURE,
        "algorithm": EXCHANGE_PPO_ALGORITHM,
        "model_state": cpu_state_dict(model),
        "model_config": model.config.to_dict(),
        "training_config": settings.to_dict(),
        "dataset_schema_version": manifest.dataset_schema_version,
        "sample_type": manifest.sample_type,
        "sample_schema_version": manifest.sample_schema_version,
        "phase_scope": manifest.phase_scope,
        "decision_mode": EXCHANGE_DECISION_MODE,
        "exchange_encoder_schema_version": EXCHANGE_ENCODER_SCHEMA_VERSION,
        "exchange_model_input_schema_version": EXCHANGE_MODEL_INPUT_SCHEMA_VERSION,
        "model_input_schema_version": EXCHANGE_MODEL_INPUT_SCHEMA_VERSION,
        "model_input_feature_count": EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
        "action_count": CARD_COUNT,
        "card_ids_sha256": calculate_card_ids_sha256(),
        "seed": settings.seed,
        "reward": dict(manifest.reward),
        "terminal_reward_transform": dict(manifest.terminal_reward_transform),
        "behavior_policy": dict(manifest.behavior_policy),
        "fixed_playing_policy": dict(manifest.fixed_playing_policy),
        "parent_actor_checkpoint_sha256": parent_actor_checkpoint_sha256,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
    }
    torch.save(checkpoint, Path(path))


def load_exchange_ppo_checkpoint(
    path: Path | str,
) -> tuple[ExchangeActorCriticModel, dict[str, object]]:
    raw = _load_raw_checkpoint(Path(path))
    _validate_exchange_ppo_checkpoint(raw)
    config_raw = raw["model_config"]
    if not isinstance(config_raw, dict):
        raise ExchangePpoCompatibilityError("checkpoint model_config must be a dictionary.")
    model = ExchangeActorCriticModel(ExchangeMlpConfig.from_dict(config_raw))
    state = raw.get("model_state")
    if not isinstance(state, dict):
        raise ExchangePpoCompatibilityError("checkpoint model_state must be a dictionary.")
    try:
        model.load_state_dict(state)
    except RuntimeError as error:
        raise ExchangePpoCompatibilityError(
            f"checkpoint model_state is incompatible: {error}"
        ) from error
    return model, raw


def load_exchange_logits_checkpoint(
    path: Path | str,
) -> tuple[nn.Module, dict[str, object]]:
    raw = _load_raw_checkpoint(Path(path))
    architecture = raw.get("model_architecture")
    if architecture == EXCHANGE_ACTOR_CRITIC_MODEL_ARCHITECTURE:
        return load_exchange_ppo_checkpoint(path)

    config_raw = raw.get("model_config")
    if not isinstance(config_raw, dict):
        raise ExchangePpoCompatibilityError("checkpoint model_config must be a dictionary.")
    model = ExchangeMlpModel(ExchangeMlpConfig.from_dict(config_raw))
    state = raw.get("model_state")
    if not isinstance(state, dict):
        raise ExchangePpoCompatibilityError("checkpoint model_state must be a dictionary.")
    try:
        model.load_state_dict(state)
    except RuntimeError as error:
        raise ExchangePpoCompatibilityError(
            f"checkpoint model_state is incompatible: {error}"
        ) from error
    return model, raw


@dataclass
class _LossTotals:
    sample_count: int = 0
    optimizer_step_count: int = 0
    actor_loss_sum: float = 0.0
    value_loss_sum: float = 0.0
    total_loss_sum: float = 0.0
    reward_sum: float = 0.0
    probability_ratio_sum: float = 0.0
    clipped_sample_count: int = 0
    forced_sample_count: int = 0

    def update(self, *, batch: ExchangePpoBatch, loss: ExchangePpoLoss) -> None:
        batch_size = int(batch.terminal_reward.shape[0])
        self.sample_count += batch_size
        self.optimizer_step_count += 1
        self.actor_loss_sum += float(loss.actor_loss.detach().item()) * batch_size
        self.value_loss_sum += float(loss.value_loss.detach().item()) * batch_size
        self.total_loss_sum += float(loss.total_loss.detach().item()) * batch_size
        self.reward_sum += float(batch.terminal_reward.detach().sum().item())
        self.probability_ratio_sum += float(loss.probability_ratio.detach().sum().item())
        self.clipped_sample_count += int(loss.clipped_sample_mask.detach().sum().item())
        self.forced_sample_count += int(
            batch.legal_discard_card_mask.to(dtype=torch.int64).sum(dim=1).eq(1).sum().item()
        )

    def to_report(
        self,
        output: Path,
        *,
        device: ResolvedTorchDevice,
    ) -> ExchangePpoTrainReport:
        return ExchangePpoTrainReport(
            sample_count=self.sample_count,
            optimizer_step_count=self.optimizer_step_count,
            mean_actor_loss=self.actor_loss_sum / self.sample_count,
            mean_value_loss=self.value_loss_sum / self.sample_count,
            mean_total_loss=self.total_loss_sum / self.sample_count,
            mean_reward=self.reward_sum / self.sample_count,
            mean_probability_ratio=self.probability_ratio_sum / self.sample_count,
            clipped_fraction=self.clipped_sample_count / self.sample_count,
            forced_sample_count=self.forced_sample_count,
            output_checkpoint_path=output,
            requested_training_device=device.requested,
            resolved_training_device=device.resolved,
            cuda_device_name=device.cuda_device_name,
        )


def _parse_rl_sample(raw: object, *, context: str) -> NonPlayingExchangeRlSample:
    obj = _require_dict(raw, context)
    if obj.get("sampleType") != NON_PLAYING_RL_SAMPLE_TYPE:
        raise ExchangePpoCompatibilityError(f"{context}: sampleType mismatch.")
    model_input = _float_array(
        obj.get("modelInput"),
        length=EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
        path=f"{context}.modelInput",
    )
    legal_mask = _int_mask_array(
        obj.get("legalDiscardCardMask"), length=CARD_COUNT, path=f"{context}.legalDiscardCardMask"
    )
    selected = _require_int(obj.get("selectedActionIndex"), f"{context}.selectedActionIndex")
    if selected < 0 or selected >= CARD_COUNT:
        raise ExchangePpoCompatibilityError(f"{context}: selectedActionIndex is out of range.")
    if int(legal_mask[selected]) != 1:
        raise ExchangePpoCompatibilityError(f"{context}: selectedActionIndex is illegal.")
    behavior_log_probability = _require_finite_number(
        obj.get("behaviorLogProbability"), f"{context}.behaviorLogProbability"
    )
    if behavior_log_probability > 1e-12:
        raise ExchangePpoCompatibilityError(
            f"{context}: behaviorLogProbability must be <= 0."
        )
    return NonPlayingExchangeRlSample(
        seed=_require_int(obj.get("seed"), f"{context}.seed"),
        step=_require_int(obj.get("step"), f"{context}.step"),
        acting_player_index=_require_int(
            obj.get("actingPlayerIndex"), f"{context}.actingPlayerIndex"
        ),
        model_input=model_input,
        legal_discard_card_mask=legal_mask,
        selected_action_index=selected,
        behavior_log_probability=behavior_log_probability,
        terminal_reward=_require_finite_number(
            obj.get("terminalReward"), f"{context}.terminalReward"
        ),
    )


def _collate_exchange_ppo_batch(samples: list[NonPlayingExchangeRlSample]) -> ExchangePpoBatch:
    return ExchangePpoBatch(
        model_input=torch.from_numpy(np.stack([sample.model_input for sample in samples])),
        legal_discard_card_mask=torch.from_numpy(
            np.stack([sample.legal_discard_card_mask for sample in samples])
        ).to(
            dtype=torch.bool,
        ),
        selected_action_index=torch.tensor(
            [sample.selected_action_index for sample in samples], dtype=torch.int64
        ),
        behavior_log_probability=torch.tensor(
            [sample.behavior_log_probability for sample in samples], dtype=torch.float32
        ),
        terminal_reward=torch.tensor(
            [sample.terminal_reward for sample in samples], dtype=torch.float32
        ),
        seed=torch.tensor([sample.seed for sample in samples], dtype=torch.int64),
        step=torch.tensor([sample.step for sample in samples], dtype=torch.int64),
        acting_player_index=torch.tensor(
            [sample.acting_player_index for sample in samples], dtype=torch.int64
        ),
    )


def _exchange_ppo_batch_to_device(
    batch: ExchangePpoBatch,
    device: ResolvedTorchDevice,
) -> ExchangePpoBatch:
    torch_device = device.torch_device
    return ExchangePpoBatch(
        model_input=batch.model_input.to(device=torch_device, dtype=torch.float32),
        legal_discard_card_mask=batch.legal_discard_card_mask.to(
            device=torch_device,
            dtype=torch.bool,
        ),
        selected_action_index=batch.selected_action_index.to(
            device=torch_device,
            dtype=torch.int64,
        ),
        behavior_log_probability=batch.behavior_log_probability.to(
            device=torch_device,
            dtype=torch.float32,
        ),
        terminal_reward=batch.terminal_reward.to(device=torch_device, dtype=torch.float32),
        seed=batch.seed.to(device=torch_device, dtype=torch.int64),
        step=batch.step.to(device=torch_device, dtype=torch.int64),
        acting_player_index=batch.acting_player_index.to(
            device=torch_device,
            dtype=torch.int64,
        ),
    )


def _extract_actor_state_dict(state: dict[Any, Any]) -> dict[str, Tensor]:
    if any(isinstance(key, str) and key.startswith("actor.") for key in state):
        return {
            key.removeprefix("actor."): value
            for key, value in state.items()
            if isinstance(key, str) and key.startswith("actor.")
        }
    return cast(dict[str, Tensor], state)


def _validate_exchange_ppo_checkpoint(raw: dict[str, object]) -> None:
    expected = {
        "checkpoint_schema_version": EXCHANGE_PPO_CHECKPOINT_SCHEMA_VERSION,
        "model_architecture": EXCHANGE_ACTOR_CRITIC_MODEL_ARCHITECTURE,
        "algorithm": EXCHANGE_PPO_ALGORITHM,
        "sample_type": NON_PLAYING_RL_SAMPLE_TYPE,
        "phase_scope": "exchange-only",
        "decision_mode": EXCHANGE_DECISION_MODE,
        "exchange_encoder_schema_version": EXCHANGE_ENCODER_SCHEMA_VERSION,
        "exchange_model_input_schema_version": EXCHANGE_MODEL_INPUT_SCHEMA_VERSION,
        "model_input_schema_version": EXCHANGE_MODEL_INPUT_SCHEMA_VERSION,
        "model_input_feature_count": EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
        "action_count": CARD_COUNT,
        "card_ids_sha256": calculate_card_ids_sha256(),
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise ExchangePpoCompatibilityError(
                f"checkpoint {key} mismatch: expected {value!r}, got {raw.get(key)!r}."
            )
    reward = raw.get("reward")
    expected_reward = {
        "type": NON_PLAYING_RL_REWARD_TYPE,
        "version": NON_PLAYING_RL_REWARD_VERSION,
        "id": NON_PLAYING_RL_REWARD_ID,
    }
    if not isinstance(reward, dict) or reward != expected_reward:
        raise ExchangePpoCompatibilityError("checkpoint reward metadata mismatch.")
    _require_terminal_reward_transform(
        raw.get("terminal_reward_transform"), "checkpoint terminal_reward_transform"
    )


def _require_terminal_reward_transform(value: object, path: str) -> dict[str, object]:
    transform = _require_dict(value, path)
    expected = {
        "type": NON_PLAYING_RL_TERMINAL_REWARD_TRANSFORM_TYPE,
        "version": NON_PLAYING_RL_TERMINAL_REWARD_TRANSFORM_VERSION,
        "id": NON_PLAYING_RL_TERMINAL_REWARD_TRANSFORM_ID,
        "sourceRewardId": NON_PLAYING_RL_REWARD_ID,
        "baseline": "meanRawRewardAllPlayers",
        "formula": "relative_reward_i = raw_reward_i - mean(raw_reward_all_players)",
    }
    if transform != expected:
        raise ExchangePpoCompatibilityError(f"{path} metadata mismatch.")
    return transform


def _load_raw_checkpoint(path: Path) -> dict[str, object]:
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except OSError as error:
        raise ExchangePpoCompatibilityError(f"checkpoint cannot be read: {error}") from error
    except RuntimeError as error:
        raise ExchangePpoCompatibilityError(f"checkpoint cannot be loaded: {error}") from error
    except pickle.UnpicklingError as error:
        raise ExchangePpoCompatibilityError(f"checkpoint cannot be loaded: {error}") from error
    if not isinstance(raw, dict):
        raise ExchangePpoCompatibilityError("checkpoint must be a dictionary.")
    return cast(dict[str, object], raw)


def _require_dict(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ExchangePpoCompatibilityError(f"{path} must be an object.")
    return cast(dict[str, object], value)


def _require_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExchangePpoCompatibilityError(f"{path} must be an integer.")
    return value


def _require_positive_int(value: object, path: str) -> int:
    item = _require_int(value, path)
    if item <= 0:
        raise ExchangePpoCompatibilityError(f"{path} must be positive.")
    return item


def _require_non_negative_int(value: object, path: str) -> int:
    item = _require_int(value, path)
    if item < 0:
        raise ExchangePpoCompatibilityError(f"{path} must be non-negative.")
    return item


def _require_finite_number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ExchangePpoCompatibilityError(f"{path} must be a number.")
    item = float(value)
    if not math.isfinite(item):
        raise ExchangePpoCompatibilityError(f"{path} must be finite.")
    return item


def _float_array(value: object, *, length: int, path: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != length:
        raise ExchangePpoCompatibilityError(f"{path} must be a list of length {length}.")
    array = np.asarray(
        [_require_finite_number(item, f"{path}[]") for item in value],
        dtype=np.float32,
    )
    array.setflags(write=False)
    return array


def _int_mask_array(value: object, *, length: int, path: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != length:
        raise ExchangePpoCompatibilityError(f"{path} must be a list of length {length}.")
    items = []
    for index, item in enumerate(value):
        if item not in (0, 1):
            raise ExchangePpoCompatibilityError(f"{path}[{index}] must be 0 or 1.")
        items.append(item)
    if sum(items) == 0:
        raise ExchangePpoCompatibilityError(f"{path} must contain at least one legal action.")
    array = np.asarray(items, dtype=np.uint8)
    array.setflags(write=False)
    return array


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
