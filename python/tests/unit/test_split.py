from __future__ import annotations

import pytest

from napoleon_ml.dataset.constants import UINT32_MAX
from napoleon_ml.dataset.errors import DatasetError
from napoleon_ml.dataset.split import DatasetSplit, SplitConfig, split_for_seed


@pytest.mark.parametrize(
    ("seed", "expected"),
    [
        (0, DatasetSplit.TRAIN),
        (79, DatasetSplit.TRAIN),
        (80, DatasetSplit.VALIDATION),
        (89, DatasetSplit.VALIDATION),
        (90, DatasetSplit.TEST),
        (99, DatasetSplit.TEST),
        (100, DatasetSplit.TRAIN),
    ],
)
def test_default_split_boundaries(seed: int, expected: DatasetSplit) -> None:
    assert split_for_seed(seed) is expected


def test_same_seed_always_returns_same_split() -> None:
    results = {split_for_seed(42) for _ in range(10)}
    assert len(results) == 1


def test_split_is_a_pure_function_of_seed_and_config() -> None:
    config = SplitConfig(train=50, validation=25, test=25)
    first = split_for_seed(123, config)
    second = split_for_seed(123, SplitConfig(train=50, validation=25, test=25))

    assert first is second


def test_config_ratios_must_sum_to_100() -> None:
    with pytest.raises(DatasetError, match="sum to 100"):
        SplitConfig(train=80, validation=10, test=5)


def test_config_rejects_negative_ratio() -> None:
    with pytest.raises(DatasetError, match="non-negative"):
        SplitConfig(train=101, validation=-1, test=0)


def test_config_rejects_bool_ratio() -> None:
    with pytest.raises(DatasetError, match="integer"):
        SplitConfig(train=True, validation=0, test=0)


def test_split_for_seed_rejects_negative_seed() -> None:
    with pytest.raises(DatasetError, match="between 0"):
        split_for_seed(-1)


def test_split_for_seed_rejects_seed_above_uint32_max() -> None:
    with pytest.raises(DatasetError, match="between 0"):
        split_for_seed(UINT32_MAX + 1)


def test_split_for_seed_accepts_uint32_max() -> None:
    # Must not raise.
    split_for_seed(UINT32_MAX)


def test_split_for_seed_rejects_bool_seed() -> None:
    with pytest.raises(DatasetError, match="integer"):
        split_for_seed(True)


def test_custom_config_changes_boundaries() -> None:
    config = SplitConfig(train=50, validation=25, test=25)

    assert split_for_seed(49, config) is DatasetSplit.TRAIN
    assert split_for_seed(50, config) is DatasetSplit.VALIDATION
    assert split_for_seed(74, config) is DatasetSplit.VALIDATION
    assert split_for_seed(75, config) is DatasetSplit.TEST
    assert split_for_seed(99, config) is DatasetSplit.TEST
