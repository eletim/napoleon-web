"""PyTorch IterableDataset adapter for streamed self-play data."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict

import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .constants import CARD_COUNT
from .errors import DatasetError
from .reader import iter_tensorized_samples
from .split import DatasetSplit, SplitConfig
from .tensors import TensorizedPlayingSample


class PlayingTorchSample(TypedDict):
    model_input: Tensor
    actor_target: Tensor
    legal_play_mask: Tensor
    belief_target: Tensor
    belief_hidden_ownership_loss_mask: Tensor
    seed: Tensor
    step: Tensor


_MASK_DTYPES = {torch.bool, torch.uint8}


class PlayingIterableDataset(IterableDataset[PlayingTorchSample]):
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
        split_config: SplitConfig | None = None,
        verify_integrity: bool = True,
        mask_dtype: torch.dtype = torch.bool,
    ) -> None:
        super().__init__()
        self.dataset_directory = Path(dataset_directory)
        self.split = _coerce_split(split)
        self.split_config = _coerce_split_config(split_config)
        self.verify_integrity = _coerce_bool(verify_integrity, name="verify_integrity")
        self.mask_dtype = _coerce_mask_dtype(mask_dtype)

    def __iter__(self) -> Iterator[PlayingTorchSample]:
        if get_worker_info() is not None:
            raise DatasetError(
                "PlayingIterableDataset only supports DataLoader num_workers=0 in this version."
            )

        for sample in iter_tensorized_samples(
            self.dataset_directory,
            split=self.split,
            split_config=self.split_config,
            verify_integrity=self.verify_integrity,
        ):
            if not isinstance(sample, TensorizedPlayingSample):
                raise DatasetError(
                    "PlayingIterableDataset requires a playing-training-sample dataset."
                )

            yield _torch_sample(sample, mask_dtype=self.mask_dtype)


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

    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise DatasetError(
            f"batch_size must be a positive integer, got {type(batch_size).__name__}."
        )

    if batch_size <= 0:
        raise DatasetError(f"batch_size must be a positive integer, got {batch_size}.")

    if isinstance(num_workers, bool) or not isinstance(num_workers, int):
        raise DatasetError(f"num_workers must be integer 0, got {type(num_workers).__name__}.")

    if num_workers != 0:
        raise DatasetError("create_playing_dataloader only supports num_workers=0 in this version.")

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


def _torch_sample(
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
