"""PyTorch IterableDataset adapter for streamed self-play data."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict, TypeVar

import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .binary import iter_binary_playing_self_play_batches
from .constants import (
    ADJUTANT_DATASET_SAMPLE_TYPE,
    BIDDING_ACTION_COUNT,
    BIDDING_DATASET_SAMPLE_TYPE,
    CARD_COUNT,
    EXCHANGE_DATASET_SAMPLE_TYPE,
    PLAYING_DATASET_SAMPLE_TYPE,
    PLAYING_SELF_PLAY_BINARY_DATASET_FORMAT,
    PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE,
)
from .errors import DatasetError
from .manifest import DatasetManifest
from .reader import _iter_tensorized_samples_with_manifest, load_manifest
from .split import DatasetSplit, SplitConfig
from .tensors import (
    TensorizedAdjutantSample,
    TensorizedBiddingSample,
    TensorizedExchangeSample,
    TensorizedPlayingSample,
    TensorizedPlayingSelfPlaySample,
    TensorizedTrainingSample,
)


class PlayingTorchSample(TypedDict):
    model_input: Tensor
    actor_target: Tensor
    legal_play_mask: Tensor
    belief_target: Tensor
    belief_hidden_ownership_loss_mask: Tensor
    seed: Tensor
    step: Tensor


class BiddingTorchSample(TypedDict):
    model_input: Tensor
    legal_bid_mask: Tensor
    actor_target: Tensor
    seed: Tensor
    step: Tensor


class ExchangeTorchSample(TypedDict):
    model_input: Tensor
    legal_discard_card_mask: Tensor
    discard_target_mask: Tensor
    seed: Tensor
    step: Tensor


class AdjutantTorchSample(TypedDict):
    model_input: Tensor
    legal_adjutant_mask: Tensor
    actor_target: Tensor
    seed: Tensor
    step: Tensor


class PlayingSelfPlayTorchSample(TypedDict):
    model_input: Tensor
    legal_play_mask: Tensor
    selected_card_index: Tensor
    behavior_log_probability: Tensor
    terminal_reward: Tensor
    seed: Tensor
    step: Tensor
    acting_player_index: Tensor
    self_role_index: Tensor


TrainingTorchSample = (
    PlayingTorchSample
    | BiddingTorchSample
    | ExchangeTorchSample
    | AdjutantTorchSample
    | PlayingSelfPlayTorchSample
)
_MASK_DTYPES = {torch.bool, torch.uint8}
_TorchSample = TypeVar("_TorchSample", bound=TrainingTorchSample)


class _TensorizedIterableDataset(IterableDataset[_TorchSample]):
    """Stream split-filtered samples as PyTorch tensors.

    The underlying JSONL dataset is reopened on every ``__iter__`` call, so
    the same dataset instance can be reused across epochs without buffering
    the full dataset in memory. The first version explicitly supports only
    single-process loading (``num_workers=0``).
    """

    def __init__(
        self,
        dataset_directory: Path | str,
        *,
        split: DatasetSplit | str,
        sample_type: str,
        split_config: SplitConfig | None = None,
        verify_integrity: bool = True,
        mask_dtype: torch.dtype = torch.bool,
    ) -> None:
        super().__init__()
        self.dataset_directory = Path(dataset_directory)
        self.split = _coerce_split(split)
        self.sample_type = sample_type
        self.split_config = _coerce_split_config(split_config)
        self.verify_integrity = _coerce_bool(verify_integrity, name="verify_integrity")
        self.mask_dtype = _coerce_mask_dtype(mask_dtype)

    def __iter__(self) -> Iterator[_TorchSample]:
        if get_worker_info() is not None:
            raise DatasetError(f"{type(self).__name__} only supports DataLoader num_workers=0.")

        manifest = load_manifest(self.dataset_directory)
        _require_manifest_sample_type(manifest, self.sample_type, dataset_name=type(self).__name__)

        for sample in _iter_tensorized_samples_with_manifest(
            self.dataset_directory,
            manifest,
            split=self.split,
            split_config=self.split_config,
            verify_integrity=self.verify_integrity,
        ):
            yield self._torch_sample(sample)

    def _torch_sample(self, sample: TensorizedTrainingSample) -> _TorchSample:
        raise NotImplementedError


class PlayingIterableDataset(_TensorizedIterableDataset[PlayingTorchSample]):
    """Stream split-filtered playing samples as PyTorch tensors."""

    def __init__(
        self,
        dataset_directory: Path | str,
        *,
        split: DatasetSplit | str,
        split_config: SplitConfig | None = None,
        verify_integrity: bool = True,
        mask_dtype: torch.dtype = torch.bool,
    ) -> None:
        super().__init__(
            dataset_directory,
            split=split,
            sample_type=PLAYING_DATASET_SAMPLE_TYPE,
            split_config=split_config,
            verify_integrity=verify_integrity,
            mask_dtype=mask_dtype,
        )

    def _torch_sample(self, sample: TensorizedTrainingSample) -> PlayingTorchSample:
        if not isinstance(sample, TensorizedPlayingSample):
            raise DatasetError("PlayingIterableDataset requires a playing-training-sample dataset.")

        return _torch_playing_sample(sample, mask_dtype=self.mask_dtype)


class BiddingIterableDataset(_TensorizedIterableDataset[BiddingTorchSample]):
    """Stream split-filtered bidding samples as PyTorch tensors."""

    def __init__(
        self,
        dataset_directory: Path | str,
        *,
        split: DatasetSplit | str,
        split_config: SplitConfig | None = None,
        verify_integrity: bool = True,
    ) -> None:
        super().__init__(
            dataset_directory,
            split=split,
            sample_type=BIDDING_DATASET_SAMPLE_TYPE,
            split_config=split_config,
            verify_integrity=verify_integrity,
        )

    def _torch_sample(self, sample: TensorizedTrainingSample) -> BiddingTorchSample:
        if not isinstance(sample, TensorizedBiddingSample):
            raise DatasetError("BiddingIterableDataset requires a bidding-training-sample dataset.")

        return _torch_bidding_sample(sample)


class ExchangeIterableDataset(_TensorizedIterableDataset[ExchangeTorchSample]):
    """Stream split-filtered exchange samples as PyTorch tensors."""

    def __init__(
        self,
        dataset_directory: Path | str,
        *,
        split: DatasetSplit | str,
        split_config: SplitConfig | None = None,
        verify_integrity: bool = True,
    ) -> None:
        super().__init__(
            dataset_directory,
            split=split,
            sample_type=EXCHANGE_DATASET_SAMPLE_TYPE,
            split_config=split_config,
            verify_integrity=verify_integrity,
        )

    def _torch_sample(self, sample: TensorizedTrainingSample) -> ExchangeTorchSample:
        if not isinstance(sample, TensorizedExchangeSample):
            raise DatasetError(
                "ExchangeIterableDataset requires an exchange-training-sample dataset."
            )

        return _torch_exchange_sample(sample)


class AdjutantIterableDataset(_TensorizedIterableDataset[AdjutantTorchSample]):
    """Stream split-filtered adjutant samples as PyTorch tensors."""

    def __init__(
        self,
        dataset_directory: Path | str,
        *,
        split: DatasetSplit | str,
        split_config: SplitConfig | None = None,
        verify_integrity: bool = True,
    ) -> None:
        super().__init__(
            dataset_directory,
            split=split,
            sample_type=ADJUTANT_DATASET_SAMPLE_TYPE,
            split_config=split_config,
            verify_integrity=verify_integrity,
        )

    def _torch_sample(self, sample: TensorizedTrainingSample) -> AdjutantTorchSample:
        if not isinstance(sample, TensorizedAdjutantSample):
            raise DatasetError(
                "AdjutantIterableDataset requires an adjutant-training-sample dataset."
            )

        return _torch_adjutant_sample(sample)


class PlayingSelfPlayIterableDataset(_TensorizedIterableDataset[PlayingSelfPlayTorchSample]):
    """Stream split-filtered playing self-play samples as PyTorch tensors."""

    def __init__(
        self,
        dataset_directory: Path | str,
        *,
        split: DatasetSplit | str,
        split_config: SplitConfig | None = None,
        verify_integrity: bool = True,
    ) -> None:
        super().__init__(
            dataset_directory,
            split=split,
            sample_type=PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE,
            split_config=split_config,
            verify_integrity=verify_integrity,
        )

    def _torch_sample(self, sample: TensorizedTrainingSample) -> PlayingSelfPlayTorchSample:
        if not isinstance(sample, TensorizedPlayingSelfPlaySample):
            raise DatasetError(
                "PlayingSelfPlayIterableDataset requires a playing-self-play-sample dataset."
            )

        return _torch_playing_self_play_sample(sample)


class BinaryPlayingSelfPlayBatchIterableDataset(IterableDataset[PlayingSelfPlayTorchSample]):
    """Yield pre-batched tensors from binary playing self-play shards."""

    def __init__(
        self,
        dataset_directory: Path | str,
        *,
        split: DatasetSplit | str,
        split_config: SplitConfig | None = None,
        batch_size: int,
        verify_integrity: bool = True,
        drop_last: bool = False,
    ) -> None:
        super().__init__()
        _validate_dataloader_options(
            batch_size=batch_size,
            num_workers=0,
            function_name=type(self).__name__,
        )
        self.dataset_directory = Path(dataset_directory)
        self.split = _coerce_split(split)
        self.split_config = _coerce_split_config(split_config)
        self.batch_size = batch_size
        self.verify_integrity = _coerce_bool(verify_integrity, name="verify_integrity")
        self.drop_last = _coerce_bool(drop_last, name="drop_last")

    def __iter__(self) -> Iterator[PlayingSelfPlayTorchSample]:
        if get_worker_info() is not None:
            raise DatasetError(
                f"{type(self).__name__} only supports DataLoader num_workers=0."
            )
        manifest = load_manifest(self.dataset_directory)
        _require_manifest_sample_type(
            manifest,
            PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE,
            dataset_name=type(self).__name__,
        )
        if manifest.format != PLAYING_SELF_PLAY_BINARY_DATASET_FORMAT:
            raise DatasetError(
                f"{type(self).__name__} requires binary self-play format, "
                f"got {manifest.format!r}."
            )
        yield from iter_binary_playing_self_play_batches(
            self.dataset_directory,
            manifest,
            split=self.split,
            split_config=self.split_config,
            batch_size=self.batch_size,
            verify_integrity=self.verify_integrity,
            drop_last=self.drop_last,
        )


def create_playing_dataloader(
    dataset_directory: Path | str,
    *,
    split: DatasetSplit | str,
    batch_size: int,
    split_config: SplitConfig | None = None,
    verify_integrity: bool = True,
    mask_dtype: torch.dtype = torch.bool,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader[PlayingTorchSample]:
    """Create a deterministic, fixed-shape PyTorch DataLoader.

    ``shuffle`` is intentionally not exposed: ordering comes from the
    generated shard order plus deterministic seed-level split filtering.
    """

    _validate_dataloader_options(
        batch_size=batch_size,
        num_workers=num_workers,
        function_name="create_playing_dataloader",
    )

    dataset = PlayingIterableDataset(
        dataset_directory,
        split=split,
        split_config=split_config,
        verify_integrity=verify_integrity,
        mask_dtype=mask_dtype,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=drop_last,
    )


def create_playing_self_play_dataloader(
    dataset_directory: Path | str,
    *,
    split: DatasetSplit | str,
    batch_size: int,
    split_config: SplitConfig | None = None,
    verify_integrity: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader[PlayingSelfPlayTorchSample]:
    """Create a deterministic DataLoader for playing self-play RL batches."""

    _validate_dataloader_options(
        batch_size=batch_size,
        num_workers=num_workers,
        function_name="create_playing_self_play_dataloader",
    )
    manifest = load_manifest(dataset_directory)
    if manifest.format == PLAYING_SELF_PLAY_BINARY_DATASET_FORMAT:
        dataset = BinaryPlayingSelfPlayBatchIterableDataset(
            dataset_directory,
            split=split,
            split_config=split_config,
            batch_size=batch_size,
            verify_integrity=verify_integrity,
            drop_last=drop_last,
        )
        return DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)

    legacy_dataset = PlayingSelfPlayIterableDataset(
        dataset_directory,
        split=split,
        split_config=split_config,
        verify_integrity=verify_integrity,
    )
    return DataLoader(
        legacy_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=drop_last
    )


def create_bidding_dataloader(
    dataset_directory: Path | str,
    *,
    split: DatasetSplit | str,
    batch_size: int,
    split_config: SplitConfig | None = None,
    verify_integrity: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader[BiddingTorchSample]:
    """Create a deterministic bidding PyTorch DataLoader."""

    _validate_dataloader_options(
        batch_size=batch_size,
        num_workers=num_workers,
        function_name="create_bidding_dataloader",
    )
    dataset = BiddingIterableDataset(
        dataset_directory,
        split=split,
        split_config=split_config,
        verify_integrity=verify_integrity,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=drop_last
    )


def create_exchange_dataloader(
    dataset_directory: Path | str,
    *,
    split: DatasetSplit | str,
    batch_size: int,
    split_config: SplitConfig | None = None,
    verify_integrity: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader[ExchangeTorchSample]:
    """Create a deterministic exchange PyTorch DataLoader."""

    _validate_dataloader_options(
        batch_size=batch_size,
        num_workers=num_workers,
        function_name="create_exchange_dataloader",
    )
    dataset = ExchangeIterableDataset(
        dataset_directory,
        split=split,
        split_config=split_config,
        verify_integrity=verify_integrity,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=drop_last
    )


def create_adjutant_dataloader(
    dataset_directory: Path | str,
    *,
    split: DatasetSplit | str,
    batch_size: int,
    split_config: SplitConfig | None = None,
    verify_integrity: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader[AdjutantTorchSample]:
    """Create a deterministic adjutant PyTorch DataLoader."""

    _validate_dataloader_options(
        batch_size=batch_size,
        num_workers=num_workers,
        function_name="create_adjutant_dataloader",
    )
    dataset = AdjutantIterableDataset(
        dataset_directory,
        split=split,
        split_config=split_config,
        verify_integrity=verify_integrity,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=drop_last
    )


def create_training_dataloader(
    dataset_directory: Path | str,
    *,
    split: DatasetSplit | str,
    batch_size: int,
    split_config: SplitConfig | None = None,
    verify_integrity: bool = True,
    mask_dtype: torch.dtype = torch.bool,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader[TrainingTorchSample]:
    """Create a DataLoader whose batch type is selected from manifest sampleType."""

    manifest = load_manifest(dataset_directory)
    _coerce_mask_dtype(mask_dtype)

    if manifest.sample_type == PLAYING_DATASET_SAMPLE_TYPE:
        return create_playing_dataloader(
            dataset_directory,
            split=split,
            batch_size=batch_size,
            split_config=split_config,
            verify_integrity=verify_integrity,
            mask_dtype=mask_dtype,
            drop_last=drop_last,
            num_workers=num_workers,
        )
    if manifest.sample_type == PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE:
        raise DatasetError(
            "create_training_dataloader does not treat playing-self-play-sample as "
            "supervised data; use create_playing_self_play_dataloader instead."
        )
    if mask_dtype != torch.bool:
        raise DatasetError(
            "mask_dtype is only supported for playing-training-sample datasets; "
            "non-playing DataLoader masks are always torch.bool."
        )
    if manifest.sample_type == BIDDING_DATASET_SAMPLE_TYPE:
        return create_bidding_dataloader(
            dataset_directory,
            split=split,
            batch_size=batch_size,
            split_config=split_config,
            verify_integrity=verify_integrity,
            drop_last=drop_last,
            num_workers=num_workers,
        )
    if manifest.sample_type == EXCHANGE_DATASET_SAMPLE_TYPE:
        return create_exchange_dataloader(
            dataset_directory,
            split=split,
            batch_size=batch_size,
            split_config=split_config,
            verify_integrity=verify_integrity,
            drop_last=drop_last,
            num_workers=num_workers,
        )
    if manifest.sample_type == ADJUTANT_DATASET_SAMPLE_TYPE:
        return create_adjutant_dataloader(
            dataset_directory,
            split=split,
            batch_size=batch_size,
            split_config=split_config,
            verify_integrity=verify_integrity,
            drop_last=drop_last,
            num_workers=num_workers,
        )

    raise DatasetError(f"Unsupported dataset sampleType: {manifest.sample_type!r}.")


def _validate_dataloader_options(*, batch_size: int, num_workers: int, function_name: str) -> None:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise DatasetError(
            f"batch_size must be a positive integer, got {type(batch_size).__name__}."
        )

    if batch_size <= 0:
        raise DatasetError(f"batch_size must be a positive integer, got {batch_size}.")

    if isinstance(num_workers, bool) or not isinstance(num_workers, int):
        raise DatasetError(f"num_workers must be integer 0, got {type(num_workers).__name__}.")

    if num_workers != 0:
        raise DatasetError(f"{function_name} only supports num_workers=0 in this version.")


def _coerce_split(split: DatasetSplit | str) -> DatasetSplit:
    if isinstance(split, DatasetSplit):
        return split

    if isinstance(split, str):
        try:
            return DatasetSplit(split)
        except ValueError as error:
            allowed = ", ".join(item.value for item in DatasetSplit)
            raise DatasetError(f"split must be one of {allowed}, got {split!r}.") from error

    raise DatasetError(f"split must be a DatasetSplit or string, got {type(split).__name__}.")


def _coerce_split_config(split_config: SplitConfig | None) -> SplitConfig:
    if split_config is None:
        return SplitConfig()

    if not isinstance(split_config, SplitConfig):
        raise DatasetError(
            f"split_config must be a SplitConfig, got {type(split_config).__name__}."
        )

    return split_config


def _coerce_bool(value: bool, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise DatasetError(f"{name} must be a bool, got {type(value).__name__}.")

    return value


def _coerce_mask_dtype(mask_dtype: torch.dtype) -> torch.dtype:
    if mask_dtype not in _MASK_DTYPES:
        raise DatasetError("mask_dtype must be torch.bool or torch.uint8.")

    return mask_dtype


def _require_manifest_sample_type(
    manifest: DatasetManifest, expected_sample_type: str, *, dataset_name: str
) -> None:
    if manifest.sample_type != expected_sample_type:
        raise DatasetError(
            f"{dataset_name} requires {_sample_type_article(expected_sample_type)} "
            f"{expected_sample_type} dataset, got {manifest.sample_type}."
        )


def _sample_type_article(sample_type: str) -> str:
    return "an" if sample_type[0] in {"a", "e", "i", "o", "u"} else "a"


def _torch_playing_sample(
    sample: TensorizedPlayingSample, *, mask_dtype: torch.dtype
) -> PlayingTorchSample:
    actor_target_value = int(sample.actor_target)
    legal_play_mask = torch.from_numpy(sample.legal_play_mask.copy()).to(dtype=torch.bool)

    if actor_target_value < 0 or actor_target_value >= CARD_COUNT:
        raise DatasetError(
            f"actor_target must be between 0 and {CARD_COUNT - 1}, got {actor_target_value} "
            f"(seed={sample.seed}, step={sample.step})."
        )

    if not bool(legal_play_mask[actor_target_value].item()):
        raise DatasetError(
            "actor_target must be legal according to legal_play_mask "
            f"(seed={sample.seed}, step={sample.step}, actor_target={actor_target_value})."
        )

    return {
        "model_input": torch.from_numpy(sample.model_input.copy()),
        "actor_target": torch.tensor(actor_target_value, dtype=torch.int64),
        "legal_play_mask": legal_play_mask,
        "belief_target": torch.from_numpy(sample.belief_target.copy()),
        "belief_hidden_ownership_loss_mask": torch.from_numpy(
            sample.belief_hidden_ownership_loss_mask.copy()
        ).to(dtype=mask_dtype),
        "seed": torch.tensor(sample.seed, dtype=torch.int64),
        "step": torch.tensor(sample.step, dtype=torch.int64),
    }


def _torch_bidding_sample(sample: TensorizedBiddingSample) -> BiddingTorchSample:
    actor_target_value = int(sample.actor_target)
    legal_bid_mask = torch.from_numpy(sample.legal_bid_mask.copy()).to(dtype=torch.bool)

    _validate_actor_target(
        actor_target_value,
        legal_bid_mask,
        action_count=BIDDING_ACTION_COUNT,
        legal_mask_name="legal_bid_mask",
        seed=sample.seed,
        step=sample.step,
    )

    return {
        "model_input": torch.from_numpy(sample.model_input.copy()),
        "legal_bid_mask": legal_bid_mask,
        "actor_target": torch.tensor(actor_target_value, dtype=torch.int64),
        "seed": torch.tensor(sample.seed, dtype=torch.int64),
        "step": torch.tensor(sample.step, dtype=torch.int64),
    }


def _torch_exchange_sample(sample: TensorizedExchangeSample) -> ExchangeTorchSample:
    legal_mask = torch.from_numpy(sample.legal_discard_card_mask.copy()).to(dtype=torch.bool)
    target_mask = torch.from_numpy(sample.discard_target_mask.copy()).to(dtype=torch.bool)

    legal_count = int(legal_mask.to(dtype=torch.int64).sum().item())
    target_count = int(target_mask.to(dtype=torch.int64).sum().item())

    if legal_count != 13:
        raise DatasetError(
            "legal_discard_card_mask must be 13-hot "
            f"(seed={sample.seed}, step={sample.step}, got {legal_count})."
        )

    if target_count != 3:
        raise DatasetError(
            "discard_target_mask must be exactly 3-hot "
            f"(seed={sample.seed}, step={sample.step}, got {target_count})."
        )

    if not bool(torch.logical_or(torch.logical_not(target_mask), legal_mask).all().item()):
        raise DatasetError(
            "discard_target_mask must be a subset of legal_discard_card_mask "
            f"(seed={sample.seed}, step={sample.step})."
        )

    return {
        "model_input": torch.from_numpy(sample.model_input.copy()),
        "legal_discard_card_mask": legal_mask,
        "discard_target_mask": target_mask,
        "seed": torch.tensor(sample.seed, dtype=torch.int64),
        "step": torch.tensor(sample.step, dtype=torch.int64),
    }


def _torch_adjutant_sample(sample: TensorizedAdjutantSample) -> AdjutantTorchSample:
    actor_target_value = int(sample.actor_target)
    legal_adjutant_mask = torch.from_numpy(sample.legal_adjutant_mask.copy()).to(dtype=torch.bool)

    _validate_actor_target(
        actor_target_value,
        legal_adjutant_mask,
        action_count=CARD_COUNT,
        legal_mask_name="legal_adjutant_mask",
        seed=sample.seed,
        step=sample.step,
    )

    return {
        "model_input": torch.from_numpy(sample.model_input.copy()),
        "legal_adjutant_mask": legal_adjutant_mask,
        "actor_target": torch.tensor(actor_target_value, dtype=torch.int64),
        "seed": torch.tensor(sample.seed, dtype=torch.int64),
        "step": torch.tensor(sample.step, dtype=torch.int64),
    }


def _torch_playing_self_play_sample(
    sample: TensorizedPlayingSelfPlaySample,
) -> PlayingSelfPlayTorchSample:
    selected_value = int(sample.selected_card_index)
    legal_play_mask = torch.from_numpy(sample.legal_play_mask.copy()).to(dtype=torch.bool)

    if selected_value < 0 or selected_value >= CARD_COUNT:
        raise DatasetError(
            f"selected_card_index must be between 0 and {CARD_COUNT - 1}, got "
            f"{selected_value} (seed={sample.seed}, step={sample.step})."
        )

    if not bool(legal_play_mask[selected_value].item()):
        raise DatasetError(
            "selected_card_index must be legal according to legal_play_mask "
            f"(seed={sample.seed}, step={sample.step}, selected_card_index={selected_value})."
        )

    return {
        "model_input": torch.from_numpy(sample.model_input.copy()),
        "legal_play_mask": legal_play_mask,
        "selected_card_index": torch.tensor(selected_value, dtype=torch.int64),
        "behavior_log_probability": torch.tensor(
            float(sample.behavior_log_probability), dtype=torch.float32
        ),
        "terminal_reward": torch.tensor(float(sample.terminal_reward), dtype=torch.float32),
        "seed": torch.tensor(sample.seed, dtype=torch.int64),
        "step": torch.tensor(sample.step, dtype=torch.int64),
        "acting_player_index": torch.tensor(sample.acting_player_index, dtype=torch.int64),
    }


def _validate_actor_target(
    actor_target_value: int,
    legal_mask: Tensor,
    *,
    action_count: int,
    legal_mask_name: str,
    seed: int,
    step: int,
) -> None:
    if actor_target_value < 0 or actor_target_value >= action_count:
        raise DatasetError(
            f"actor_target must be between 0 and {action_count - 1}, got "
            f"{actor_target_value} (seed={seed}, step={step})."
        )

    if not bool(legal_mask[actor_target_value].item()):
        raise DatasetError(
            f"actor_target must be legal according to {legal_mask_name} "
            f"(seed={seed}, step={step}, actor_target={actor_target_value})."
        )
