"""Deterministic, game-level train/validation/test splitting.

Splitting must happen per game (per seed), never per sample: a single game
produces ~50 sequential decisions, and letting those decisions land in
different splits would leak information about a game the model is being
"tested" on into what it was trained on. ``split_for_seed`` is a pure
function of ``seed`` and ``config`` alone (no RNG, no process-order
dependence, no use of Python's salted built-in ``hash()``), so the same
inputs always produce the same split across processes and Python runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .constants import UINT32_MAX
from .errors import DatasetError


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class SplitConfig:
    train: int = 80
    validation: int = 10
    test: int = 10

    def __post_init__(self) -> None:
        for name, value in (
            ("train", self.train),
            ("validation", self.validation),
            ("test", self.test),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise DatasetError(
                    f"SplitConfig.{name} must be an integer, got {type(value).__name__}."
                )

            if value < 0:
                raise DatasetError(f"SplitConfig.{name} must be non-negative, got {value}.")

        total = self.train + self.validation + self.test

        if total != 100:
            raise DatasetError(
                f"SplitConfig ratios must sum to 100, got {total} "
                f"(train={self.train}, validation={self.validation}, test={self.test})."
            )


_DEFAULT_SPLIT_CONFIG = SplitConfig()


def split_for_seed(seed: int, config: SplitConfig = _DEFAULT_SPLIT_CONFIG) -> DatasetSplit:
    """Return the split a game's seed belongs to.

    Deterministic: ``seed % 100`` is bucketed into ``[0, config.train)`` for
    train, the next ``config.validation`` values for validation, and the
    remainder for test. The same ``(seed, config)`` pair always returns the
    same split, in any process, on any run.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise DatasetError(f"seed must be an integer, got {type(seed).__name__}.")

    if seed < 0 or seed > UINT32_MAX:
        raise DatasetError(f"seed must be between 0 and {UINT32_MAX}, got {seed}.")

    bucket = seed % 100

    if bucket < config.train:
        return DatasetSplit.TRAIN

    if bucket < config.train + config.validation:
        return DatasetSplit.VALIDATION

    return DatasetSplit.TEST
