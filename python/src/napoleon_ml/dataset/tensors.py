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
relationship between card ids that is not there. A future consumer that
wants those as model input should one-hot encode them explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import (
    CARD_COUNT,
    CARDS_PER_TRICK,
    MAX_BIDDING_ACTION_COUNT,
    NOT_IN_HAND_CLASS_INDEX,
    PLAYER_COUNT,
    REVEALED_ADJUTANT_CLASS_COUNT,
    TRICK_COUNT,
)
from .errors import SampleValidationError
from .sample import EncodedPlayingObservation, PlayingTrainingSample

_TRUMP_SUIT_OPTION_COUNT = 4
_COMPLETED_TRICK_CARD_SLOT_COUNT = TRICK_COUNT * CARDS_PER_TRICK


@dataclass(frozen=True)
class FeatureSlice:
    """One named, contiguous region of :data:`FLAT_OBSERVATION_LAYOUT`."""

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
    legal_play_mask: np.ndarray
    actor_target: np.int64
    belief_target: np.ndarray
    belief_hidden_ownership_loss_mask: np.ndarray


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


def _build_flat_layout() -> tuple[FeatureSlice, ...]:
    slices: list[FeatureSlice] = []
    offset = 0

    for name, shape in _FLAT_LAYOUT_SPEC:
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


def _validate_flat_layout(layout: tuple[FeatureSlice, ...]) -> None:
    expected_start = 0

    for feature in layout:
        if feature.start != expected_start:
            raise AssertionError(
                f"FLAT_OBSERVATION_LAYOUT has a gap or overlap before {feature.name!r}: "
                f"expected start {expected_start}, got {feature.start}."
            )

        if feature.stop <= feature.start:
            raise AssertionError(f"FLAT_OBSERVATION_LAYOUT slice {feature.name!r} is empty.")

        expected_start = feature.stop


FLAT_OBSERVATION_LAYOUT: tuple[FeatureSlice, ...] = _build_flat_layout()
_validate_flat_layout(FLAT_OBSERVATION_LAYOUT)
FLAT_OBSERVATION_FEATURE_COUNT: int = FLAT_OBSERVATION_LAYOUT[-1].stop


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


def tensorize_sample(sample: PlayingTrainingSample) -> TensorizedPlayingSample:
    """Convert one validated sample into a :class:`TensorizedPlayingSample`.

    Callers should validate ``sample`` first (see
    :func:`napoleon_ml.dataset.validation.validate_sample`); this function
    does not repeat schema validation, only the tensor-shape/dtype
    invariants that tensorization itself must uphold (see
    :func:`validate_tensorized_sample`).
    """

    observation_tensors = tensorize_observation(sample.observation)
    flat = _flat_observation(observation_tensors)

    tensorized = TensorizedPlayingSample(
        seed=sample.seed,
        step=sample.step,
        acting_player_index=sample.relative_player_ids.index(sample.acting_player_id),
        observation=observation_tensors,
        flat_observation=flat,
        legal_play_mask=observation_tensors.legal_play_mask,
        actor_target=np.int64(sample.actor_target.selected_card_index),
        belief_target=_index_array(sample.belief_target.owner_class_by_card),
        belief_hidden_ownership_loss_mask=_mask_array(
            sample.belief_target.hidden_ownership_loss_mask
        ),
    )

    validate_tensorized_sample(tensorized)

    return tensorized


def validate_tensorized_sample(tensorized: TensorizedPlayingSample) -> None:
    """Check the shape/dtype/value invariants tensorization must uphold."""

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
