"""NumPy tensorization of validated samples, with an explicit fixed feature layout.

Every array here has a fixed shape and dtype derived directly from schema
v1's constants (see :mod:`napoleon_ml.dataset.constants`), never inferred by
recursively flattening whatever a sample happens to contain. ``flat_observation``
concatenates only the fields that are already good MLP input as-is (binary
masks, one-hots, and small scalars) in the fixed order recorded in
:data:`FLAT_OBSERVATION_LAYOUT`. Card/player/trick *index* fields
(``currentTrickCardIndices``, the bidding history index arrays, etc.) are
deliberately kept out of ``flat_observation`` and exposed only as separate
``int64`` arrays on :class:`PlayingObservationTensors`: concatenating a raw
category index into a float vector would silently imply an ordinal
relationship between card ids that is not there.

``model_input`` (schema version :data:`MODEL_INPUT_SCHEMA_VERSION`) is the
complete fixed-length model input: ``flat_observation`` followed by every
one of those index fields, one-hot encoded, in the fixed order recorded in
:data:`MODEL_INPUT_LAYOUT` (which starts with :data:`FLAT_OBSERVATION_LAYOUT`
verbatim and appends the one-hot slices after it — see
:data:`MODEL_INPUT_ONEHOT_LAYOUT` for just the appended part). An empty slot
(index ``-1``, or a bidding-history target when the slot has no bid) encodes
to an all-zero region rather than a one-hot at some placeholder class, so
"no value" is never confusable with a real class 0. ``flat_observation``
remains untouched and continues to omit these fields, for existing
consumers.

Deliberately excluded from both ``flat_observation`` and ``model_input``:
``schemaVersion``, ``seed``/``step``/player-id strings, and the sample's
``actorTarget``/``beliefTarget`` (the training labels and hidden-ownership
ground truth a player could not actually observe) — see
:func:`napoleon_ml.dataset.tensors.tensorize_sample`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, overload

import numpy as np

from .constants import (
    BIDDING_ACTION_COUNT,
    BIDDING_HISTORY_SUIT_ORDER,
    CARD_COUNT,
    CARDS_PER_TRICK,
    MAX_BIDDING_ACTION_COUNT,
    MAX_BIDDING_TARGET_POINT_CARDS,
    MIN_BIDDING_TARGET_POINT_CARDS,
    MIN_CONTRACT_TARGET_POINT_CARDS,
    NOT_IN_HAND_CLASS_INDEX,
    PLAYER_COUNT,
    REVEALED_ADJUTANT_CLASS_COUNT,
    TRICK_COUNT,
)
from .errors import SampleValidationError
from .sample import (
    AdjutantTrainingSample,
    BiddingTrainingSample,
    EncodedAdjutantObservation,
    EncodedBiddingHistory,
    EncodedBiddingObservation,
    EncodedExchangeObservation,
    EncodedPlayingObservation,
    ExchangeTrainingSample,
    PlayingTrainingSample,
    TrainingSample,
)

MODEL_INPUT_SCHEMA_VERSION = 1
BIDDING_MODEL_INPUT_SCHEMA_VERSION = 1
EXCHANGE_MODEL_INPUT_SCHEMA_VERSION = 1
ADJUTANT_MODEL_INPUT_SCHEMA_VERSION = 1

_TRUMP_SUIT_OPTION_COUNT = 4
_COMPLETED_TRICK_CARD_SLOT_COUNT = TRICK_COUNT * CARDS_PER_TRICK
_SPECIAL_CARD_INDEX_COUNT = 4
_BIDDING_ACTION_TYPE_CLASS_COUNT = 2
_BIDDING_SUIT_CLASS_COUNT = len(BIDDING_HISTORY_SUIT_ORDER)
_BIDDING_TARGET_POINT_CARDS_CLASS_COUNT = (
    MAX_BIDDING_TARGET_POINT_CARDS - MIN_BIDDING_TARGET_POINT_CARDS + 1
)


@dataclass(frozen=True)
class FeatureSlice:
    """One named, contiguous region of a feature layout.

    Used by :data:`FLAT_OBSERVATION_LAYOUT`, :data:`MODEL_INPUT_ONEHOT_LAYOUT`,
    and :data:`MODEL_INPUT_LAYOUT` (their concatenation).
    """

    name: str
    start: int
    stop: int
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class PlayingObservationTensors:
    # Binary masks (uint8), each element 0 or 1.
    called_adjutant_card_mask: np.ndarray
    self_hand_mask: np.ndarray
    legal_play_mask: np.ndarray
    captured_point_card_mask_by_player: np.ndarray
    current_trick_slot_mask: np.ndarray
    completed_trick_slot_mask: np.ndarray
    completed_trick_mask: np.ndarray
    latest_buried_event_point_card_mask: np.ndarray
    bidding_history_action_mask: np.ndarray

    # One-hot vectors (float32).
    trump_suit_one_hot: np.ndarray
    napoleon_player_one_hot: np.ndarray
    revealed_adjutant_player_one_hot: np.ndarray

    # Scalars and small count vectors (float32), cast but not rescaled.
    trick_number: np.float32
    completed_trick_count: np.float32
    contract_target_point_cards: np.float32
    latest_buried_event_hidden_non_point_count: np.float32
    latest_buried_event_present: np.float32
    hand_count_by_player: np.ndarray

    # Category indices (int64); -1 marks an empty slot. Not part of
    # flat_observation -- see the module docstring.
    special_card_indices: np.ndarray
    current_trick_card_indices: np.ndarray
    current_trick_player_indices: np.ndarray
    completed_trick_card_indices: np.ndarray
    completed_trick_player_indices: np.ndarray
    completed_trick_winner_indices: np.ndarray
    bidding_history_action_type_indices: np.ndarray
    bidding_history_player_indices: np.ndarray
    bidding_history_suit_indices: np.ndarray
    bidding_history_target_point_cards: np.ndarray


@dataclass(frozen=True)
class TensorizedPlayingSample:
    seed: int
    step: int
    acting_player_index: int
    observation: PlayingObservationTensors
    flat_observation: np.ndarray
    model_input: np.ndarray
    legal_play_mask: np.ndarray
    actor_target: np.int64
    belief_target: np.ndarray
    belief_hidden_ownership_loss_mask: np.ndarray


@dataclass(frozen=True)
class BiddingObservationTensors:
    self_hand_mask: np.ndarray
    legal_bid_mask: np.ndarray
    starter_player_index: np.int64
    highest_bid_present: np.float32
    highest_bid_player_index: np.int64
    highest_bid_suit_index: np.int64
    highest_bid_target_point_cards: np.int64
    consecutive_pass_count: np.int64
    bidding_history_action_mask: np.ndarray
    bidding_history_action_type_indices: np.ndarray
    bidding_history_player_indices: np.ndarray
    bidding_history_suit_indices: np.ndarray
    bidding_history_target_point_cards: np.ndarray


@dataclass(frozen=True)
class TensorizedBiddingSample:
    seed: int
    step: int
    acting_player_index: int
    observation: BiddingObservationTensors
    model_input: np.ndarray
    legal_bid_mask: np.ndarray
    actor_target: np.int64


@dataclass(frozen=True)
class ExchangeObservationTensors:
    trump_suit_one_hot: np.ndarray
    called_adjutant_card_mask: np.ndarray
    self_hand_mask: np.ndarray
    legal_discard_card_mask: np.ndarray
    contract_target_point_cards: np.int64
    hand_count_by_player: np.ndarray
    special_card_indices: np.ndarray
    bidding_history_action_mask: np.ndarray
    bidding_history_action_type_indices: np.ndarray
    bidding_history_player_indices: np.ndarray
    bidding_history_suit_indices: np.ndarray
    bidding_history_target_point_cards: np.ndarray


@dataclass(frozen=True)
class TensorizedExchangeSample:
    seed: int
    step: int
    acting_player_index: int
    observation: ExchangeObservationTensors
    model_input: np.ndarray
    legal_discard_card_mask: np.ndarray
    discard_target_mask: np.ndarray


@dataclass(frozen=True)
class AdjutantObservationTensors:
    trump_suit_one_hot: np.ndarray
    self_hand_mask: np.ndarray
    legal_adjutant_mask: np.ndarray
    contract_target_point_cards: np.int64
    special_card_indices: np.ndarray
    bidding_history_action_mask: np.ndarray
    bidding_history_action_type_indices: np.ndarray
    bidding_history_player_indices: np.ndarray
    bidding_history_suit_indices: np.ndarray
    bidding_history_target_point_cards: np.ndarray


@dataclass(frozen=True)
class TensorizedAdjutantSample:
    seed: int
    step: int
    acting_player_index: int
    observation: AdjutantObservationTensors
    model_input: np.ndarray
    legal_adjutant_mask: np.ndarray
    actor_target: np.int64


TensorizedTrainingSample = (
    TensorizedPlayingSample
    | TensorizedBiddingSample
    | TensorizedExchangeSample
    | TensorizedAdjutantSample
)
_SpecialCardObservation = (
    EncodedPlayingObservation | EncodedExchangeObservation | EncodedAdjutantObservation
)


_FLAT_LAYOUT_SPEC: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("trumpSuitOneHot", (_TRUMP_SUIT_OPTION_COUNT,)),
    ("napoleonPlayerOneHot", (PLAYER_COUNT,)),
    ("revealedAdjutantPlayerOneHot", (REVEALED_ADJUTANT_CLASS_COUNT,)),
    ("calledAdjutantCardMask", (CARD_COUNT,)),
    ("selfHandMask", (CARD_COUNT,)),
    ("legalPlayMask", (CARD_COUNT,)),
    ("handCountByPlayer", (PLAYER_COUNT,)),
    ("capturedPointCardMaskByPlayer", (PLAYER_COUNT, CARD_COUNT)),
    ("currentTrickSlotMask", (CARDS_PER_TRICK,)),
    ("completedTrickSlotMask", (_COMPLETED_TRICK_CARD_SLOT_COUNT,)),
    ("completedTrickMask", (TRICK_COUNT,)),
    ("biddingHistoryActionMask", (MAX_BIDDING_ACTION_COUNT,)),
    ("latestBuriedEventPointCardMask", (CARD_COUNT,)),
    ("trickNumber", (1,)),
    ("completedTrickCount", (1,)),
    ("contractTargetPointCards", (1,)),
    ("latestBuriedEventHiddenNonPointCount", (1,)),
    ("latestBuriedEventPresent", (1,)),
)


class _OneHotField(NamedTuple):
    """One category-index field of :class:`PlayingObservationTensors` to one-hot encode.

    The single source of truth for the one-hot block appended to
    ``model_input``: ``name``/``slot_count``/``num_classes`` drive
    :data:`MODEL_INPUT_ONEHOT_LAYOUT`, and every field (including
    ``tensor_attr``, the matching ``int64`` attribute name on
    :class:`PlayingObservationTensors`) drives both :func:`_model_input`'s
    encoding and :func:`_validate_model_input`'s cross-check -- so the
    layout, the encoder, and the validator can never silently drift apart
    from one another.
    """

    name: str
    tensor_attr: str
    slot_count: int
    num_classes: int
    min_value: int


# One-hot regions appended after FLAT_OBSERVATION_LAYOUT to build
# MODEL_INPUT_LAYOUT (see the module docstring). Order mirrors issue #17:
# special card indices, current/completed trick card indices,
# current/completed trick player indices, completed trick winner indices,
# bidding history action type/player/suit/target.
_MODEL_INPUT_ONEHOT_FIELDS: tuple[_OneHotField, ...] = (
    _OneHotField(
        "specialCardIndicesOneHot", "special_card_indices", _SPECIAL_CARD_INDEX_COUNT, CARD_COUNT, 0
    ),
    _OneHotField(
        "currentTrickCardIndicesOneHot",
        "current_trick_card_indices",
        CARDS_PER_TRICK,
        CARD_COUNT,
        0,
    ),
    _OneHotField(
        "completedTrickCardIndicesOneHot",
        "completed_trick_card_indices",
        _COMPLETED_TRICK_CARD_SLOT_COUNT,
        CARD_COUNT,
        0,
    ),
    _OneHotField(
        "currentTrickPlayerIndicesOneHot",
        "current_trick_player_indices",
        CARDS_PER_TRICK,
        PLAYER_COUNT,
        0,
    ),
    _OneHotField(
        "completedTrickPlayerIndicesOneHot",
        "completed_trick_player_indices",
        _COMPLETED_TRICK_CARD_SLOT_COUNT,
        PLAYER_COUNT,
        0,
    ),
    _OneHotField(
        "completedTrickWinnerIndicesOneHot",
        "completed_trick_winner_indices",
        TRICK_COUNT,
        PLAYER_COUNT,
        0,
    ),
    _OneHotField(
        "biddingHistoryActionTypeIndicesOneHot",
        "bidding_history_action_type_indices",
        MAX_BIDDING_ACTION_COUNT,
        _BIDDING_ACTION_TYPE_CLASS_COUNT,
        0,
    ),
    _OneHotField(
        "biddingHistoryPlayerIndicesOneHot",
        "bidding_history_player_indices",
        MAX_BIDDING_ACTION_COUNT,
        PLAYER_COUNT,
        0,
    ),
    _OneHotField(
        "biddingHistorySuitIndicesOneHot",
        "bidding_history_suit_indices",
        MAX_BIDDING_ACTION_COUNT,
        _BIDDING_SUIT_CLASS_COUNT,
        0,
    ),
    _OneHotField(
        "biddingHistoryTargetPointCardsOneHot",
        "bidding_history_target_point_cards",
        MAX_BIDDING_ACTION_COUNT,
        _BIDDING_TARGET_POINT_CARDS_CLASS_COUNT,
        MIN_BIDDING_TARGET_POINT_CARDS,
    ),
)

_MODEL_INPUT_ONEHOT_SPEC: tuple[tuple[str, tuple[int, ...]], ...] = tuple(
    (field.name, (field.slot_count, field.num_classes)) for field in _MODEL_INPUT_ONEHOT_FIELDS
)


def _build_layout(
    spec: tuple[tuple[str, tuple[int, ...]], ...], *, start: int
) -> tuple[FeatureSlice, ...]:
    slices: list[FeatureSlice] = []
    offset = start

    for name, shape in spec:
        length = 1

        for dimension in shape:
            length *= dimension

        slices.append(
            FeatureSlice(
                name=name, start=offset, stop=offset + length, shape=shape, dtype="float32"
            )
        )
        offset += length

    return tuple(slices)


def _validate_layout(layout: tuple[FeatureSlice, ...], *, expected_start: int, label: str) -> None:
    for feature in layout:
        if feature.start != expected_start:
            raise AssertionError(
                f"{label} has a gap or overlap before {feature.name!r}: "
                f"expected start {expected_start}, got {feature.start}."
            )

        if feature.stop <= feature.start:
            raise AssertionError(f"{label} slice {feature.name!r} is empty.")

        expected_start = feature.stop

    names = [feature.name for feature in layout]

    if len(names) != len(set(names)):
        raise AssertionError(f"{label} contains duplicate feature names.")


FLAT_OBSERVATION_LAYOUT: tuple[FeatureSlice, ...] = _build_layout(_FLAT_LAYOUT_SPEC, start=0)
_validate_layout(FLAT_OBSERVATION_LAYOUT, expected_start=0, label="FLAT_OBSERVATION_LAYOUT")
FLAT_OBSERVATION_FEATURE_COUNT: int = FLAT_OBSERVATION_LAYOUT[-1].stop

MODEL_INPUT_ONEHOT_LAYOUT: tuple[FeatureSlice, ...] = _build_layout(
    _MODEL_INPUT_ONEHOT_SPEC, start=FLAT_OBSERVATION_FEATURE_COUNT
)
_validate_layout(
    MODEL_INPUT_ONEHOT_LAYOUT,
    expected_start=FLAT_OBSERVATION_FEATURE_COUNT,
    label="MODEL_INPUT_ONEHOT_LAYOUT",
)

MODEL_INPUT_LAYOUT: tuple[FeatureSlice, ...] = FLAT_OBSERVATION_LAYOUT + MODEL_INPUT_ONEHOT_LAYOUT
_validate_layout(MODEL_INPUT_LAYOUT, expected_start=0, label="MODEL_INPUT_LAYOUT")
MODEL_INPUT_FEATURE_COUNT: int = MODEL_INPUT_LAYOUT[-1].stop


_CONTRACT_TARGET_POINT_CARDS_CLASS_COUNT = (
    MAX_BIDDING_TARGET_POINT_CARDS - MIN_CONTRACT_TARGET_POINT_CARDS + 1
)
_CONSECUTIVE_PASS_COUNT_CLASS_COUNT = PLAYER_COUNT + 1

_BIDDING_HISTORY_ONEHOT_SPEC: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("biddingHistoryActionTypeIndicesOneHot", (MAX_BIDDING_ACTION_COUNT, 2)),
    ("biddingHistoryPlayerIndicesOneHot", (MAX_BIDDING_ACTION_COUNT, PLAYER_COUNT)),
    ("biddingHistorySuitIndicesOneHot", (MAX_BIDDING_ACTION_COUNT, _BIDDING_SUIT_CLASS_COUNT)),
    (
        "biddingHistoryTargetPointCardsOneHot",
        (MAX_BIDDING_ACTION_COUNT, _BIDDING_TARGET_POINT_CARDS_CLASS_COUNT),
    ),
)

_BIDDING_MODEL_INPUT_SPEC: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("selfHandMask", (CARD_COUNT,)),
    ("legalBidMask", (BIDDING_ACTION_COUNT,)),
    ("starterPlayerOneHot", (PLAYER_COUNT,)),
    ("highestBidPresent", (1,)),
    ("highestBidPlayerOneHot", (PLAYER_COUNT,)),
    ("highestBidSuitOneHot", (_BIDDING_SUIT_CLASS_COUNT,)),
    ("highestBidTargetPointCardsOneHot", (_BIDDING_TARGET_POINT_CARDS_CLASS_COUNT,)),
    ("consecutivePassCountOneHot", (_CONSECUTIVE_PASS_COUNT_CLASS_COUNT,)),
    ("biddingHistoryActionMask", (MAX_BIDDING_ACTION_COUNT,)),
) + _BIDDING_HISTORY_ONEHOT_SPEC

_EXCHANGE_MODEL_INPUT_SPEC: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("trumpSuitOneHot", (_TRUMP_SUIT_OPTION_COUNT,)),
    ("selfHandMask", (CARD_COUNT,)),
    ("legalDiscardCardMask", (CARD_COUNT,)),
    ("calledAdjutantCardMask", (CARD_COUNT,)),
    ("contractTargetPointCardsOneHot", (_CONTRACT_TARGET_POINT_CARDS_CLASS_COUNT,)),
    ("handCountByPlayer", (PLAYER_COUNT,)),
    ("specialCardIndicesOneHot", (_SPECIAL_CARD_INDEX_COUNT, CARD_COUNT)),
    ("biddingHistoryActionMask", (MAX_BIDDING_ACTION_COUNT,)),
) + _BIDDING_HISTORY_ONEHOT_SPEC

_ADJUTANT_MODEL_INPUT_SPEC: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("trumpSuitOneHot", (_TRUMP_SUIT_OPTION_COUNT,)),
    ("selfHandMask", (CARD_COUNT,)),
    ("legalAdjutantMask", (CARD_COUNT,)),
    ("contractTargetPointCardsOneHot", (_CONTRACT_TARGET_POINT_CARDS_CLASS_COUNT,)),
    ("specialCardIndicesOneHot", (_SPECIAL_CARD_INDEX_COUNT, CARD_COUNT)),
    ("biddingHistoryActionMask", (MAX_BIDDING_ACTION_COUNT,)),
) + _BIDDING_HISTORY_ONEHOT_SPEC

BIDDING_MODEL_INPUT_LAYOUT: tuple[FeatureSlice, ...] = _build_layout(
    _BIDDING_MODEL_INPUT_SPEC, start=0
)
_validate_layout(
    BIDDING_MODEL_INPUT_LAYOUT, expected_start=0, label="BIDDING_MODEL_INPUT_LAYOUT"
)
BIDDING_MODEL_INPUT_FEATURE_COUNT: int = BIDDING_MODEL_INPUT_LAYOUT[-1].stop

EXCHANGE_MODEL_INPUT_LAYOUT: tuple[FeatureSlice, ...] = _build_layout(
    _EXCHANGE_MODEL_INPUT_SPEC, start=0
)
_validate_layout(
    EXCHANGE_MODEL_INPUT_LAYOUT, expected_start=0, label="EXCHANGE_MODEL_INPUT_LAYOUT"
)
EXCHANGE_MODEL_INPUT_FEATURE_COUNT: int = EXCHANGE_MODEL_INPUT_LAYOUT[-1].stop

ADJUTANT_MODEL_INPUT_LAYOUT: tuple[FeatureSlice, ...] = _build_layout(
    _ADJUTANT_MODEL_INPUT_SPEC, start=0
)
_validate_layout(
    ADJUTANT_MODEL_INPUT_LAYOUT, expected_start=0, label="ADJUTANT_MODEL_INPUT_LAYOUT"
)
ADJUTANT_MODEL_INPUT_FEATURE_COUNT: int = ADJUTANT_MODEL_INPUT_LAYOUT[-1].stop

if BIDDING_MODEL_INPUT_FEATURE_COUNT != 2333:
    raise AssertionError(
        f"BIDDING_MODEL_INPUT_FEATURE_COUNT must be 2333, got {BIDDING_MODEL_INPUT_FEATURE_COUNT}."
    )
if EXCHANGE_MODEL_INPUT_FEATURE_COUNT != 2611:
    raise AssertionError(
        "EXCHANGE_MODEL_INPUT_FEATURE_COUNT must be 2611, "
        f"got {EXCHANGE_MODEL_INPUT_FEATURE_COUNT}."
    )
if ADJUTANT_MODEL_INPUT_FEATURE_COUNT != 2553:
    raise AssertionError(
        "ADJUTANT_MODEL_INPUT_FEATURE_COUNT must be 2553, "
        f"got {ADJUTANT_MODEL_INPUT_FEATURE_COUNT}."
    )


def _mask_array(values: tuple[int, ...]) -> np.ndarray:
    array = np.array(values, dtype=np.uint8)
    array.setflags(write=False)
    return array


def _one_hot_array(values: tuple[int, ...]) -> np.ndarray:
    array = np.array(values, dtype=np.float32)
    array.setflags(write=False)
    return array


def _index_array(values: tuple[int, ...]) -> np.ndarray:
    array = np.array(values, dtype=np.int64)
    array.setflags(write=False)
    return array


def _one_hot_encode_indices(indices: np.ndarray, *, num_classes: int, min_value: int) -> np.ndarray:
    """One-hot encode a 1-D int64 array of category indices.

    Class ``c`` (``0 <= c < num_classes``) is encoded at ``min_value + c``.
    Any value outside ``[min_value, min_value + num_classes)`` -- in
    particular the ``-1`` empty-slot sentinel used throughout this schema,
    and a bidding-history slot's ``target_point_cards`` of ``0`` when that
    slot has no bid -- produces an all-zero row rather than a one-hot at a
    placeholder class, so "no value" is never confusable with a real class.
    Returns shape ``(len(indices), num_classes)``, dtype ``float32``.
    """

    local_class = indices.astype(np.int64) - min_value
    in_range = (local_class >= 0) & (local_class < num_classes)

    one_hot = np.zeros((indices.shape[0], num_classes), dtype=np.float32)
    row_indices = np.nonzero(in_range)[0]
    one_hot[row_indices, local_class[row_indices]] = 1.0

    return one_hot


def _single_one_hot(value: int, *, num_classes: int, min_value: int = 0) -> np.ndarray:
    return _one_hot_encode_indices(
        np.array([value], dtype=np.int64), num_classes=num_classes, min_value=min_value
    ).reshape(num_classes)


def _special_card_indices_array(observation: _SpecialCardObservation) -> np.ndarray:
    array = np.array(
        [
            observation.special_card_indices.oruma,
            observation.special_card_indices.yoromeki,
            observation.special_card_indices.sei_jack,
            observation.special_card_indices.ura_jack,
        ],
        dtype=np.int64,
    )
    array.setflags(write=False)
    return array


def _bidding_history_index_arrays(
    history: EncodedBiddingHistory,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        _mask_array(history.action_mask),
        _index_array(history.action_type_indices),
        _index_array(history.player_indices),
        _index_array(history.suit_indices),
        _index_array(history.target_point_cards),
    )


def _bidding_history_one_hot_parts(
    action_type_indices: np.ndarray,
    player_indices: np.ndarray,
    suit_indices: np.ndarray,
    target_point_cards: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        _one_hot_encode_indices(
            action_type_indices, num_classes=_BIDDING_ACTION_TYPE_CLASS_COUNT, min_value=0
        ).reshape(-1),
        _one_hot_encode_indices(player_indices, num_classes=PLAYER_COUNT, min_value=0).reshape(-1),
        _one_hot_encode_indices(
            suit_indices, num_classes=_BIDDING_SUIT_CLASS_COUNT, min_value=0
        ).reshape(-1),
        _one_hot_encode_indices(
            target_point_cards,
            num_classes=_BIDDING_TARGET_POINT_CARDS_CLASS_COUNT,
            min_value=MIN_BIDDING_TARGET_POINT_CARDS,
        ).reshape(-1),
    )


def tensorize_observation(observation: EncodedPlayingObservation) -> PlayingObservationTensors:
    """Convert one validated observation into fixed-shape, fixed-dtype NumPy arrays."""

    captured_point_card_mask_by_player = np.array(
        observation.captured_point_card_mask_by_player, dtype=np.uint8
    )
    captured_point_card_mask_by_player.setflags(write=False)

    hand_count_by_player = np.array(observation.hand_count_by_player, dtype=np.float32)
    hand_count_by_player.setflags(write=False)

    special_card_indices = np.array(
        [
            observation.special_card_indices.oruma,
            observation.special_card_indices.yoromeki,
            observation.special_card_indices.sei_jack,
            observation.special_card_indices.ura_jack,
        ],
        dtype=np.int64,
    )
    special_card_indices.setflags(write=False)

    return PlayingObservationTensors(
        called_adjutant_card_mask=_mask_array(observation.called_adjutant_card_mask),
        self_hand_mask=_mask_array(observation.self_hand_mask),
        legal_play_mask=_mask_array(observation.legal_play_mask),
        captured_point_card_mask_by_player=captured_point_card_mask_by_player,
        current_trick_slot_mask=_mask_array(observation.current_trick_slot_mask),
        completed_trick_slot_mask=_mask_array(observation.completed_trick_slot_mask),
        completed_trick_mask=_mask_array(observation.completed_trick_mask),
        latest_buried_event_point_card_mask=_mask_array(
            observation.latest_buried_event_point_card_mask
        ),
        bidding_history_action_mask=_mask_array(observation.bidding_history.action_mask),
        trump_suit_one_hot=_one_hot_array(observation.trump_suit_one_hot),
        napoleon_player_one_hot=_one_hot_array(observation.napoleon_player_one_hot),
        revealed_adjutant_player_one_hot=_one_hot_array(
            observation.revealed_adjutant_player_one_hot
        ),
        trick_number=np.float32(observation.trick_number),
        completed_trick_count=np.float32(observation.completed_trick_count),
        contract_target_point_cards=np.float32(observation.contract_target_point_cards),
        latest_buried_event_hidden_non_point_count=np.float32(
            observation.latest_buried_event_hidden_non_point_count
        ),
        latest_buried_event_present=np.float32(observation.latest_buried_event_present),
        hand_count_by_player=hand_count_by_player,
        special_card_indices=special_card_indices,
        current_trick_card_indices=_index_array(observation.current_trick_card_indices),
        current_trick_player_indices=_index_array(observation.current_trick_player_indices),
        completed_trick_card_indices=_index_array(observation.completed_trick_card_indices),
        completed_trick_player_indices=_index_array(observation.completed_trick_player_indices),
        completed_trick_winner_indices=_index_array(observation.completed_trick_winner_indices),
        bidding_history_action_type_indices=_index_array(
            observation.bidding_history.action_type_indices
        ),
        bidding_history_player_indices=_index_array(observation.bidding_history.player_indices),
        bidding_history_suit_indices=_index_array(observation.bidding_history.suit_indices),
        bidding_history_target_point_cards=_index_array(
            observation.bidding_history.target_point_cards
        ),
    )


def tensorize_bidding_observation(
    observation: EncodedBiddingObservation,
) -> BiddingObservationTensors:
    (
        action_mask,
        action_type_indices,
        player_indices,
        suit_indices,
        target_point_cards,
    ) = _bidding_history_index_arrays(observation.bidding_history)

    return BiddingObservationTensors(
        self_hand_mask=_mask_array(observation.self_hand_mask),
        legal_bid_mask=_mask_array(observation.legal_bid_mask),
        starter_player_index=np.int64(observation.starter_player_index),
        highest_bid_present=np.float32(observation.highest_bid_present),
        highest_bid_player_index=np.int64(observation.highest_bid_player_index),
        highest_bid_suit_index=np.int64(observation.highest_bid_suit_index),
        highest_bid_target_point_cards=np.int64(observation.highest_bid_target_point_cards),
        consecutive_pass_count=np.int64(observation.consecutive_pass_count),
        bidding_history_action_mask=action_mask,
        bidding_history_action_type_indices=action_type_indices,
        bidding_history_player_indices=player_indices,
        bidding_history_suit_indices=suit_indices,
        bidding_history_target_point_cards=target_point_cards,
    )


def tensorize_exchange_observation(
    observation: EncodedExchangeObservation,
) -> ExchangeObservationTensors:
    (
        action_mask,
        action_type_indices,
        player_indices,
        suit_indices,
        target_point_cards,
    ) = _bidding_history_index_arrays(observation.bidding_history)

    hand_count_by_player = np.array(observation.hand_count_by_player, dtype=np.float32)
    hand_count_by_player.setflags(write=False)

    return ExchangeObservationTensors(
        trump_suit_one_hot=_one_hot_array(observation.trump_suit_one_hot),
        called_adjutant_card_mask=_mask_array(observation.called_adjutant_card_mask),
        self_hand_mask=_mask_array(observation.self_hand_mask),
        legal_discard_card_mask=_mask_array(observation.legal_discard_card_mask),
        contract_target_point_cards=np.int64(observation.contract_target_point_cards),
        hand_count_by_player=hand_count_by_player,
        special_card_indices=_special_card_indices_array(observation),
        bidding_history_action_mask=action_mask,
        bidding_history_action_type_indices=action_type_indices,
        bidding_history_player_indices=player_indices,
        bidding_history_suit_indices=suit_indices,
        bidding_history_target_point_cards=target_point_cards,
    )


def tensorize_adjutant_observation(
    observation: EncodedAdjutantObservation,
) -> AdjutantObservationTensors:
    (
        action_mask,
        action_type_indices,
        player_indices,
        suit_indices,
        target_point_cards,
    ) = _bidding_history_index_arrays(observation.bidding_history)

    return AdjutantObservationTensors(
        trump_suit_one_hot=_one_hot_array(observation.trump_suit_one_hot),
        self_hand_mask=_mask_array(observation.self_hand_mask),
        legal_adjutant_mask=_mask_array(observation.legal_adjutant_mask),
        contract_target_point_cards=np.int64(observation.contract_target_point_cards),
        special_card_indices=_special_card_indices_array(observation),
        bidding_history_action_mask=action_mask,
        bidding_history_action_type_indices=action_type_indices,
        bidding_history_player_indices=player_indices,
        bidding_history_suit_indices=suit_indices,
        bidding_history_target_point_cards=target_point_cards,
    )


def _flat_observation(tensors: PlayingObservationTensors) -> np.ndarray:
    parts = (
        tensors.trump_suit_one_hot,
        tensors.napoleon_player_one_hot,
        tensors.revealed_adjutant_player_one_hot,
        tensors.called_adjutant_card_mask.astype(np.float32),
        tensors.self_hand_mask.astype(np.float32),
        tensors.legal_play_mask.astype(np.float32),
        tensors.hand_count_by_player,
        tensors.captured_point_card_mask_by_player.astype(np.float32).reshape(-1),
        tensors.current_trick_slot_mask.astype(np.float32),
        tensors.completed_trick_slot_mask.astype(np.float32),
        tensors.completed_trick_mask.astype(np.float32),
        tensors.bidding_history_action_mask.astype(np.float32),
        tensors.latest_buried_event_point_card_mask.astype(np.float32),
        np.array([tensors.trick_number], dtype=np.float32),
        np.array([tensors.completed_trick_count], dtype=np.float32),
        np.array([tensors.contract_target_point_cards], dtype=np.float32),
        np.array([tensors.latest_buried_event_hidden_non_point_count], dtype=np.float32),
        np.array([tensors.latest_buried_event_present], dtype=np.float32),
    )
    flat = np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)

    if flat.shape != (FLAT_OBSERVATION_FEATURE_COUNT,):
        raise AssertionError(
            f"flat_observation length {flat.shape[0]} does not match "
            f"FLAT_OBSERVATION_FEATURE_COUNT {FLAT_OBSERVATION_FEATURE_COUNT}."
        )

    flat.setflags(write=False)
    return flat


def _one_hot_field(tensors: PlayingObservationTensors, field: _OneHotField) -> np.ndarray:
    indices: np.ndarray = getattr(tensors, field.tensor_attr)
    return _one_hot_encode_indices(
        indices, num_classes=field.num_classes, min_value=field.min_value
    )


def _model_input(tensors: PlayingObservationTensors, flat: np.ndarray) -> np.ndarray:
    """Concatenate ``flat`` with the one-hot regions in :data:`MODEL_INPUT_ONEHOT_LAYOUT`.

    Iterates :data:`_MODEL_INPUT_ONEHOT_FIELDS` in order -- the single
    source of truth also used to build :data:`MODEL_INPUT_ONEHOT_LAYOUT`
    and to cross-check ``model_input`` in :func:`_validate_model_input`.
    """

    parts = [flat]
    parts.extend(_one_hot_field(tensors, field).reshape(-1) for field in _MODEL_INPUT_ONEHOT_FIELDS)
    model_input = np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)

    if model_input.shape != (MODEL_INPUT_FEATURE_COUNT,):
        raise AssertionError(
            f"model_input length {model_input.shape[0]} does not match "
            f"MODEL_INPUT_FEATURE_COUNT {MODEL_INPUT_FEATURE_COUNT}."
        )

    model_input.setflags(write=False)
    return model_input


def _bidding_model_input(tensors: BiddingObservationTensors) -> np.ndarray:
    parts = (
        tensors.self_hand_mask.astype(np.float32),
        tensors.legal_bid_mask.astype(np.float32),
        _single_one_hot(int(tensors.starter_player_index), num_classes=PLAYER_COUNT),
        np.array([tensors.highest_bid_present], dtype=np.float32),
        _single_one_hot(int(tensors.highest_bid_player_index), num_classes=PLAYER_COUNT),
        _single_one_hot(int(tensors.highest_bid_suit_index), num_classes=_BIDDING_SUIT_CLASS_COUNT),
        _single_one_hot(
            int(tensors.highest_bid_target_point_cards),
            num_classes=_BIDDING_TARGET_POINT_CARDS_CLASS_COUNT,
            min_value=MIN_BIDDING_TARGET_POINT_CARDS,
        ),
        _single_one_hot(
            int(tensors.consecutive_pass_count), num_classes=_CONSECUTIVE_PASS_COUNT_CLASS_COUNT
        ),
        tensors.bidding_history_action_mask.astype(np.float32),
        *_bidding_history_one_hot_parts(
            tensors.bidding_history_action_type_indices,
            tensors.bidding_history_player_indices,
            tensors.bidding_history_suit_indices,
            tensors.bidding_history_target_point_cards,
        ),
    )
    return _checked_model_input(
        parts,
        feature_count=BIDDING_MODEL_INPUT_FEATURE_COUNT,
        label="bidding model_input",
    )


def _exchange_model_input(tensors: ExchangeObservationTensors) -> np.ndarray:
    parts = (
        tensors.trump_suit_one_hot,
        tensors.self_hand_mask.astype(np.float32),
        tensors.legal_discard_card_mask.astype(np.float32),
        tensors.called_adjutant_card_mask.astype(np.float32),
        _single_one_hot(
            int(tensors.contract_target_point_cards),
            num_classes=_CONTRACT_TARGET_POINT_CARDS_CLASS_COUNT,
            min_value=MIN_CONTRACT_TARGET_POINT_CARDS,
        ),
        tensors.hand_count_by_player,
        _one_hot_encode_indices(
            tensors.special_card_indices, num_classes=CARD_COUNT, min_value=0
        ).reshape(-1),
        tensors.bidding_history_action_mask.astype(np.float32),
        *_bidding_history_one_hot_parts(
            tensors.bidding_history_action_type_indices,
            tensors.bidding_history_player_indices,
            tensors.bidding_history_suit_indices,
            tensors.bidding_history_target_point_cards,
        ),
    )
    return _checked_model_input(
        parts,
        feature_count=EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
        label="exchange model_input",
    )


def _adjutant_model_input(tensors: AdjutantObservationTensors) -> np.ndarray:
    parts = (
        tensors.trump_suit_one_hot,
        tensors.self_hand_mask.astype(np.float32),
        tensors.legal_adjutant_mask.astype(np.float32),
        _single_one_hot(
            int(tensors.contract_target_point_cards),
            num_classes=_CONTRACT_TARGET_POINT_CARDS_CLASS_COUNT,
            min_value=MIN_CONTRACT_TARGET_POINT_CARDS,
        ),
        _one_hot_encode_indices(
            tensors.special_card_indices, num_classes=CARD_COUNT, min_value=0
        ).reshape(-1),
        tensors.bidding_history_action_mask.astype(np.float32),
        *_bidding_history_one_hot_parts(
            tensors.bidding_history_action_type_indices,
            tensors.bidding_history_player_indices,
            tensors.bidding_history_suit_indices,
            tensors.bidding_history_target_point_cards,
        ),
    )
    return _checked_model_input(
        parts,
        feature_count=ADJUTANT_MODEL_INPUT_FEATURE_COUNT,
        label="adjutant model_input",
    )


def _checked_model_input(
    parts: tuple[np.ndarray, ...], *, feature_count: int, label: str
) -> np.ndarray:
    model_input = np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)

    if model_input.shape != (feature_count,):
        raise AssertionError(
            f"{label} length {model_input.shape[0]} does not match expected {feature_count}."
        )

    model_input.setflags(write=False)
    return model_input


def tensorize_playing_sample(sample: PlayingTrainingSample) -> TensorizedPlayingSample:
    """Convert one validated sample into a :class:`TensorizedPlayingSample`.

    Callers should validate ``sample`` first (see
    :func:`napoleon_ml.dataset.validation.validate_sample`); this function
    does not repeat schema validation, only the tensor-shape/dtype
    invariants that tensorization itself must uphold (see
    :func:`validate_tensorized_sample`).
    """

    observation_tensors = tensorize_observation(sample.observation)
    flat = _flat_observation(observation_tensors)
    model_input = _model_input(observation_tensors, flat)

    tensorized = TensorizedPlayingSample(
        seed=sample.seed,
        step=sample.step,
        acting_player_index=sample.relative_player_ids.index(sample.acting_player_id),
        observation=observation_tensors,
        flat_observation=flat,
        model_input=model_input,
        legal_play_mask=observation_tensors.legal_play_mask,
        actor_target=np.int64(sample.actor_target.selected_card_index),
        belief_target=_index_array(sample.belief_target.owner_class_by_card),
        belief_hidden_ownership_loss_mask=_mask_array(
            sample.belief_target.hidden_ownership_loss_mask
        ),
    )

    validate_tensorized_sample(tensorized)

    return tensorized


@overload
def tensorize_sample(sample: PlayingTrainingSample) -> TensorizedPlayingSample: ...


@overload
def tensorize_sample(sample: BiddingTrainingSample) -> TensorizedBiddingSample: ...


@overload
def tensorize_sample(sample: ExchangeTrainingSample) -> TensorizedExchangeSample: ...


@overload
def tensorize_sample(sample: AdjutantTrainingSample) -> TensorizedAdjutantSample: ...


@overload
def tensorize_sample(sample: TrainingSample) -> TensorizedTrainingSample: ...


def tensorize_sample(sample: TrainingSample) -> TensorizedTrainingSample:
    if isinstance(sample, PlayingTrainingSample):
        return tensorize_playing_sample(sample)
    if isinstance(sample, BiddingTrainingSample):
        return tensorize_bidding_sample(sample)
    if isinstance(sample, ExchangeTrainingSample):
        return tensorize_exchange_sample(sample)
    if isinstance(sample, AdjutantTrainingSample):
        return tensorize_adjutant_sample(sample)

    raise SampleValidationError(f"Unsupported sample class: {type(sample).__name__}.")


def tensorize_bidding_sample(sample: BiddingTrainingSample) -> TensorizedBiddingSample:
    observation_tensors = tensorize_bidding_observation(sample.observation)
    model_input = _bidding_model_input(observation_tensors)
    tensorized = TensorizedBiddingSample(
        seed=sample.seed,
        step=sample.step,
        acting_player_index=sample.relative_player_ids.index(sample.acting_player_id),
        observation=observation_tensors,
        model_input=model_input,
        legal_bid_mask=observation_tensors.legal_bid_mask,
        actor_target=np.int64(sample.actor_target),
    )
    validate_tensorized_sample(tensorized)
    return tensorized


def tensorize_exchange_sample(sample: ExchangeTrainingSample) -> TensorizedExchangeSample:
    observation_tensors = tensorize_exchange_observation(sample.observation)
    model_input = _exchange_model_input(observation_tensors)
    tensorized = TensorizedExchangeSample(
        seed=sample.seed,
        step=sample.step,
        acting_player_index=sample.relative_player_ids.index(sample.acting_player_id),
        observation=observation_tensors,
        model_input=model_input,
        legal_discard_card_mask=observation_tensors.legal_discard_card_mask,
        discard_target_mask=_mask_array(sample.actor_target.discard_target_mask),
    )
    validate_tensorized_sample(tensorized)
    return tensorized


def tensorize_adjutant_sample(sample: AdjutantTrainingSample) -> TensorizedAdjutantSample:
    observation_tensors = tensorize_adjutant_observation(sample.observation)
    model_input = _adjutant_model_input(observation_tensors)
    tensorized = TensorizedAdjutantSample(
        seed=sample.seed,
        step=sample.step,
        acting_player_index=sample.relative_player_ids.index(sample.acting_player_id),
        observation=observation_tensors,
        model_input=model_input,
        legal_adjutant_mask=observation_tensors.legal_adjutant_mask,
        actor_target=np.int64(sample.actor_target),
    )
    validate_tensorized_sample(tensorized)
    return tensorized


def validate_tensorized_sample(tensorized: TensorizedTrainingSample) -> None:
    """Check the shape/dtype/value invariants tensorization must uphold."""

    if isinstance(tensorized, TensorizedPlayingSample):
        validate_tensorized_playing_sample(tensorized)
        return
    if isinstance(tensorized, TensorizedBiddingSample):
        _validate_actor_mask_sample(
            tensorized.model_input,
            feature_count=BIDDING_MODEL_INPUT_FEATURE_COUNT,
            actor_target=tensorized.actor_target,
            legal_mask=tensorized.legal_bid_mask,
            legal_mask_name="legal_bid_mask",
            legal_mask_length=BIDDING_ACTION_COUNT,
        )
        return
    if isinstance(tensorized, TensorizedExchangeSample):
        _validate_model_input_basic(
            tensorized.model_input, feature_count=EXCHANGE_MODEL_INPUT_FEATURE_COUNT
        )
        _validate_mask_shape(tensorized.legal_discard_card_mask, "legal_discard_card_mask")
        _validate_mask_shape(tensorized.discard_target_mask, "discard_target_mask")

        if int(tensorized.discard_target_mask.sum()) != 3:
            raise SampleValidationError("discard_target_mask must be exactly 3-hot.")

        for index, value in enumerate(tensorized.discard_target_mask):
            if value == 1 and tensorized.legal_discard_card_mask[index] != 1:
                raise SampleValidationError("discard_target_mask must be within legal mask.")
        return
    if isinstance(tensorized, TensorizedAdjutantSample):
        _validate_actor_mask_sample(
            tensorized.model_input,
            feature_count=ADJUTANT_MODEL_INPUT_FEATURE_COUNT,
            actor_target=tensorized.actor_target,
            legal_mask=tensorized.legal_adjutant_mask,
            legal_mask_name="legal_adjutant_mask",
            legal_mask_length=CARD_COUNT,
        )
        return

    raise SampleValidationError(f"Unsupported tensorized sample: {type(tensorized).__name__}.")


def validate_tensorized_playing_sample(tensorized: TensorizedPlayingSample) -> None:
    """Check playing tensor shape/dtype/value invariants."""

    flat = tensorized.flat_observation

    if flat.ndim != 1:
        raise SampleValidationError(
            f"flat_observation must be 1-dimensional, got shape {flat.shape}."
        )

    if flat.dtype != np.float32:
        raise SampleValidationError(f"flat_observation dtype must be float32, got {flat.dtype}.")

    if not flat.flags["C_CONTIGUOUS"]:
        raise SampleValidationError("flat_observation must be C-contiguous.")

    if not np.isfinite(flat).all():
        raise SampleValidationError("flat_observation must not contain NaN or Infinity.")

    if tensorized.actor_target.dtype != np.int64:
        raise SampleValidationError(
            f"actor_target dtype must be int64, got {tensorized.actor_target.dtype}."
        )

    actor_target_value = int(tensorized.actor_target)

    if actor_target_value < 0 or actor_target_value >= CARD_COUNT:
        raise SampleValidationError(
            f"actor_target must be between 0 and {CARD_COUNT - 1}, got {actor_target_value}."
        )

    legal_mask = tensorized.legal_play_mask

    if legal_mask.shape != (CARD_COUNT,):
        raise SampleValidationError(
            f"legal_play_mask shape must be ({CARD_COUNT},), got {legal_mask.shape}."
        )

    if legal_mask[actor_target_value] != 1:
        raise SampleValidationError("legal_play_mask at the actor_target index must be 1.")

    belief_target = tensorized.belief_target

    if belief_target.shape != (CARD_COUNT,):
        raise SampleValidationError(
            f"belief_target shape must be ({CARD_COUNT},), got {belief_target.shape}."
        )

    if belief_target.dtype != np.int64:
        raise SampleValidationError(
            f"belief_target dtype must be int64, got {belief_target.dtype}."
        )

    if bool(((belief_target < 0) | (belief_target > NOT_IN_HAND_CLASS_INDEX)).any()):
        raise SampleValidationError(
            f"belief_target values must be between 0 and {NOT_IN_HAND_CLASS_INDEX}."
        )

    _validate_model_input(tensorized.model_input, flat, tensorized.observation)


def _validate_model_input_basic(model_input: np.ndarray, *, feature_count: int) -> None:
    if model_input.ndim != 1:
        raise SampleValidationError(
            f"model_input must be 1-dimensional, got shape {model_input.shape}."
        )

    if model_input.shape != (feature_count,):
        raise SampleValidationError(
            f"model_input shape must be ({feature_count},), got {model_input.shape}."
        )

    if model_input.dtype != np.float32:
        raise SampleValidationError(f"model_input dtype must be float32, got {model_input.dtype}.")

    if not model_input.flags["C_CONTIGUOUS"]:
        raise SampleValidationError("model_input must be C-contiguous.")

    if not np.isfinite(model_input).all():
        raise SampleValidationError("model_input must not contain NaN or Infinity.")


def _validate_mask_shape(mask: np.ndarray, name: str, *, length: int = CARD_COUNT) -> None:
    if mask.shape != (length,):
        raise SampleValidationError(f"{name} shape must be ({length},), got {mask.shape}.")

    if mask.dtype != np.uint8:
        raise SampleValidationError(f"{name} dtype must be uint8, got {mask.dtype}.")


def _validate_actor_mask_sample(
    model_input: np.ndarray,
    *,
    feature_count: int,
    actor_target: np.int64,
    legal_mask: np.ndarray,
    legal_mask_name: str,
    legal_mask_length: int,
) -> None:
    _validate_model_input_basic(model_input, feature_count=feature_count)

    if actor_target.dtype != np.int64:
        raise SampleValidationError(f"actor_target dtype must be int64, got {actor_target.dtype}.")

    actor_target_value = int(actor_target)

    if actor_target_value < 0 or actor_target_value >= legal_mask_length:
        raise SampleValidationError(
            f"actor_target must be between 0 and {legal_mask_length - 1}, "
            f"got {actor_target_value}."
        )

    _validate_mask_shape(legal_mask, legal_mask_name, length=legal_mask_length)

    if legal_mask[actor_target_value] != 1:
        raise SampleValidationError(f"{legal_mask_name} at actor_target index must be 1.")


def _validate_model_input(
    model_input: np.ndarray, flat: np.ndarray, observation: PlayingObservationTensors
) -> None:
    _validate_model_input_basic(model_input, feature_count=MODEL_INPUT_FEATURE_COUNT)

    if not np.array_equal(model_input[:FLAT_OBSERVATION_FEATURE_COUNT], flat):
        raise SampleValidationError("model_input must start with flat_observation unchanged.")

    # Recompute each one-hot region straight from observation's underlying
    # int64 index array (the same source _model_input() itself encoded from)
    # and require an exact match. This is stricter than checking each
    # region is merely well-formed (0/1 values, <=1 set bit per row): it
    # also catches a region that is well-formed but encodes the wrong
    # class -- a shifted, swapped, or all-zeroed-out index that a
    # structural-only check would silently accept.
    for feature, field in zip(MODEL_INPUT_ONEHOT_LAYOUT, _MODEL_INPUT_ONEHOT_FIELDS, strict=True):
        region = model_input[feature.start : feature.stop].reshape(feature.shape)
        expected = _one_hot_field(observation, field)

        if not np.array_equal(region, expected):
            raise SampleValidationError(
                f"model_input region {feature.name!r} does not match "
                f"observation.{field.tensor_attr}."
            )
