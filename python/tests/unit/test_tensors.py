from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from napoleon_ml.dataset.constants import (
    CARD_COUNT,
    CARDS_PER_TRICK,
    MAX_BIDDING_ACTION_COUNT,
    PLAYER_COUNT,
    REVEALED_ADJUTANT_CLASS_COUNT,
    TRICK_COUNT,
)
from napoleon_ml.dataset.errors import SampleValidationError
from napoleon_ml.dataset.sample import parse_sample
from napoleon_ml.dataset.tensors import (
    FLAT_OBSERVATION_FEATURE_COUNT,
    FLAT_OBSERVATION_LAYOUT,
    FeatureSlice,
    tensorize_sample,
    validate_tensorized_sample,
)

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "valid_sample.json"
_COMPLETED_TRICK_CARD_SLOT_COUNT = TRICK_COUNT * CARDS_PER_TRICK


def _load_valid_sample() -> dict[str, Any]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _find_slice(name: str) -> FeatureSlice:
    for feature in FLAT_OBSERVATION_LAYOUT:
        if feature.name == name:
            return feature

    raise KeyError(name)


def test_flat_layout_has_no_gaps_or_overlaps() -> None:
    expected_start = 0

    for feature in FLAT_OBSERVATION_LAYOUT:
        assert feature.start == expected_start
        assert feature.stop > feature.start
        expected_start = feature.stop

    assert expected_start == FLAT_OBSERVATION_FEATURE_COUNT


def test_flat_layout_slice_names_are_unique() -> None:
    names = [feature.name for feature in FLAT_OBSERVATION_LAYOUT]
    assert len(names) == len(set(names))


def test_tensorize_sample_shapes_and_dtypes() -> None:
    sample = parse_sample(_load_valid_sample())
    tensorized = tensorize_sample(sample)
    observation = tensorized.observation

    assert tensorized.flat_observation.shape == (FLAT_OBSERVATION_FEATURE_COUNT,)
    assert tensorized.flat_observation.dtype == np.float32
    assert tensorized.legal_play_mask.shape == (CARD_COUNT,)
    assert tensorized.legal_play_mask.dtype == np.uint8
    assert tensorized.belief_target.shape == (CARD_COUNT,)
    assert tensorized.belief_target.dtype == np.int64
    assert tensorized.belief_hidden_ownership_loss_mask.shape == (CARD_COUNT,)
    assert tensorized.belief_hidden_ownership_loss_mask.dtype == np.uint8
    assert tensorized.actor_target.dtype == np.int64

    assert observation.called_adjutant_card_mask.shape == (CARD_COUNT,)
    assert observation.called_adjutant_card_mask.dtype == np.uint8
    assert observation.captured_point_card_mask_by_player.shape == (PLAYER_COUNT, CARD_COUNT)
    assert observation.captured_point_card_mask_by_player.dtype == np.uint8
    assert observation.trump_suit_one_hot.shape == (4,)
    assert observation.trump_suit_one_hot.dtype == np.float32
    assert observation.napoleon_player_one_hot.shape == (PLAYER_COUNT,)
    assert observation.revealed_adjutant_player_one_hot.shape == (REVEALED_ADJUTANT_CLASS_COUNT,)
    assert observation.hand_count_by_player.shape == (PLAYER_COUNT,)
    assert observation.hand_count_by_player.dtype == np.float32
    assert observation.special_card_indices.shape == (4,)
    assert observation.special_card_indices.dtype == np.int64
    assert observation.current_trick_card_indices.shape == (CARDS_PER_TRICK,)
    assert observation.current_trick_card_indices.dtype == np.int64
    assert observation.completed_trick_card_indices.shape == (_COMPLETED_TRICK_CARD_SLOT_COUNT,)
    assert observation.completed_trick_winner_indices.shape == (TRICK_COUNT,)
    assert observation.bidding_history_action_mask.shape == (MAX_BIDDING_ACTION_COUNT,)
    assert observation.bidding_history_action_mask.dtype == np.uint8
    assert observation.bidding_history_action_type_indices.shape == (MAX_BIDDING_ACTION_COUNT,)
    assert observation.bidding_history_action_type_indices.dtype == np.int64


def test_flat_observation_is_c_contiguous() -> None:
    sample = parse_sample(_load_valid_sample())
    tensorized = tensorize_sample(sample)

    assert tensorized.flat_observation.flags["C_CONTIGUOUS"]


def test_flat_observation_has_no_nan_or_infinity() -> None:
    sample = parse_sample(_load_valid_sample())
    tensorized = tensorize_sample(sample)

    assert np.isfinite(tensorized.flat_observation).all()


def test_actor_target_position_is_legal() -> None:
    sample = parse_sample(_load_valid_sample())
    tensorized = tensorize_sample(sample)

    assert tensorized.legal_play_mask[int(tensorized.actor_target)] == 1


def test_validate_tensorized_sample_rejects_illegal_actor_target() -> None:
    sample = parse_sample(_load_valid_sample())
    tensorized = tensorize_sample(sample)

    all_zero_mask = tensorized.legal_play_mask * 0
    bad = dataclasses.replace(tensorized, legal_play_mask=all_zero_mask)

    with pytest.raises(SampleValidationError, match="legal_play_mask"):
        validate_tensorized_sample(bad)


def test_same_sample_produces_byte_identical_tensors() -> None:
    raw = _load_valid_sample()
    first = tensorize_sample(parse_sample(json.loads(json.dumps(raw))))
    second = tensorize_sample(parse_sample(json.loads(json.dumps(raw))))

    assert first.flat_observation.tobytes() == second.flat_observation.tobytes()
    assert first.belief_target.tobytes() == second.belief_target.tobytes()
    assert bytes(first.actor_target) == bytes(second.actor_target)


def test_changing_one_field_only_changes_its_flat_slice() -> None:
    raw = _load_valid_sample()
    baseline = tensorize_sample(parse_sample(json.loads(json.dumps(raw)))).flat_observation

    mutated_raw = json.loads(json.dumps(raw))
    self_hand_mask = list(mutated_raw["observation"]["selfHandMask"])
    flip_index = self_hand_mask.index(0)  # a card currently marked "not in hand"
    self_hand_mask[flip_index] = 1
    mutated_raw["observation"]["selfHandMask"] = self_hand_mask

    mutated = tensorize_sample(parse_sample(mutated_raw)).flat_observation

    changed_indices = {index for index in range(len(baseline)) if baseline[index] != mutated[index]}

    self_hand_mask_slice = _find_slice("selfHandMask")
    expected_changed_index = self_hand_mask_slice.start + flip_index

    assert changed_indices == {expected_changed_index}
