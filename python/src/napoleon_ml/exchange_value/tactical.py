"""Visible candidate-local tactical features for compact396 exchange inputs."""

from __future__ import annotations

import numpy as np

from napoleon_ml.dataset.constants import CARD_COUNT, EXPECTED_CARD_IDS

EXCHANGE_TACTICAL_FEATURE_NAMES = (
    "buriedPointCardCount",
    "buriedTrumpCount",
    "remainingTrumpCount",
    "retainedPointCardCount",
    "buriedJoker",
    "buriedOruma",
    "buriedSeiJack",
    "buriedUraJack",
    "buriedYoromeki",
    "buriedCalledAdjutant",
)
EXCHANGE_TACTICAL_FEATURE_COUNT = len(EXCHANGE_TACTICAL_FEATURE_NAMES)
EXCHANGE_COMPACT_STATE_FEATURE_COUNT = 343
EXCHANGE_COMPACT_VALUE_FEATURE_COUNT = 396

_ORIGINAL = slice(0, CARD_COUNT)
_KITTY = slice(CARD_COUNT, CARD_COUNT * 2)
_CALLED = slice(CARD_COUNT * 2, CARD_COUNT * 3)
_TRUMP_SUIT = slice(CARD_COUNT * 3, CARD_COUNT * 3 + 4)
_DISCARD = slice(EXCHANGE_COMPACT_STATE_FEATURE_COUNT, EXCHANGE_COMPACT_VALUE_FEATURE_COUNT)
_POINT_CARD_INDICES = np.asarray(
    [
        index
        for index, card_id in enumerate(EXPECTED_CARD_IDS)
        if card_id.split("-")[-1] in {"A", "K", "Q", "J", "10"}
    ],
    dtype=np.int64,
)
_JOKER_INDEX = EXPECTED_CARD_IDS.index("joker")
_ORUMA_INDEX = EXPECTED_CARD_IDS.index("spades-A")
_YOROMEKI_INDEX = EXPECTED_CARD_IDS.index("hearts-Q")
_SEI_JACK_INDICES = np.asarray(
    [EXPECTED_CARD_IDS.index(f"{suit}-J") for suit in ("spades", "hearts", "diamonds", "clubs")],
    dtype=np.int64,
)
_URA_JACK_INDICES = np.asarray(
    [
        EXPECTED_CARD_IDS.index("clubs-J"),
        EXPECTED_CARD_IDS.index("diamonds-J"),
        EXPECTED_CARD_IDS.index("hearts-J"),
        EXPECTED_CARD_IDS.index("spades-J"),
    ],
    dtype=np.int64,
)


def compact396_tactical_features(value_input: np.ndarray) -> np.ndarray:
    """Return ten normalized features using only exchange-visible compact396 fields."""

    values = np.asarray(value_input, dtype=np.float32)
    if values.shape[-1] != EXCHANGE_COMPACT_VALUE_FEATURE_COUNT:
        raise ValueError(
            f"compact396 tactical input must end in 396 features, got {values.shape[-1]}."
        )
    original = values[..., _ORIGINAL]
    kitty = values[..., _KITTY]
    called = values[..., _CALLED]
    trump_one_hot = values[..., _TRUMP_SUIT]
    discard = values[..., _DISCARD]
    pickup = np.maximum(original, kitty)
    trump_index = np.argmax(trump_one_hot, axis=-1)

    trump_masks = np.zeros_like(discard)
    for suit_index in range(4):
        start = suit_index * 13
        trump_masks[..., start : start + 13] = np.expand_dims(trump_index == suit_index, axis=-1)

    buried_point_count = discard[..., _POINT_CARD_INDICES].sum(axis=-1)
    buried_trump_count = (discard * trump_masks).sum(axis=-1)
    remaining_trump_count = (pickup * trump_masks).sum(axis=-1) - buried_trump_count
    retained_point_count = pickup[..., _POINT_CARD_INDICES].sum(axis=-1) - buried_point_count
    sei_indices = _SEI_JACK_INDICES[trump_index]
    ura_indices = _URA_JACK_INDICES[trump_index]
    buried_called = (discard * called).sum(axis=-1)

    features = np.stack(
        (
            buried_point_count / 3.0,
            buried_trump_count / 3.0,
            remaining_trump_count / 13.0,
            retained_point_count / 13.0,
            discard[..., _JOKER_INDEX],
            discard[..., _ORUMA_INDEX],
            np.take_along_axis(discard, np.expand_dims(sei_indices, -1), axis=-1)[..., 0],
            np.take_along_axis(discard, np.expand_dims(ura_indices, -1), axis=-1)[..., 0],
            discard[..., _YOROMEKI_INDEX],
            buried_called,
        ),
        axis=-1,
    )
    return features.astype(np.float32, copy=False)


def compact406_value_input(value_input: np.ndarray) -> np.ndarray:
    """Append visible tactical aggregates to compact396."""

    values = np.asarray(value_input, dtype=np.float32)
    return np.concatenate((values, compact396_tactical_features(values)), axis=-1).astype(
        np.float32, copy=False
    )
