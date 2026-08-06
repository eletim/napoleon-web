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
    MODEL_INPUT_FEATURE_COUNT,
    MODEL_INPUT_LAYOUT,
    MODEL_INPUT_ONEHOT_LAYOUT,
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


def _find_model_input_slice(name: str) -> FeatureSlice:
    for feature in MODEL_INPUT_LAYOUT:
        if feature.name == name:
            return feature

    raise KeyError(name)


def _region(model_input: np.ndarray, name: str) -> np.ndarray:
    feature = _find_model_input_slice(name)
    return model_input[feature.start : feature.stop].reshape(feature.shape)


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
    assert tensorized.model_input.shape == (MODEL_INPUT_FEATURE_COUNT,)
    assert tensorized.model_input.dtype == np.float32
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


# --- model_input ----------------------------------------------------------


def test_model_input_layout_has_no_gaps_or_overlaps() -> None:
    expected_start = 0

    for feature in MODEL_INPUT_LAYOUT:
        assert feature.start == expected_start
        assert feature.stop > feature.start
        expected_start = feature.stop

    assert expected_start == MODEL_INPUT_FEATURE_COUNT


def test_model_input_layout_slice_names_are_unique() -> None:
    names = [feature.name for feature in MODEL_INPUT_LAYOUT]
    assert len(names) == len(set(names))


def test_model_input_layout_starts_with_flat_observation_layout() -> None:
    assert MODEL_INPUT_LAYOUT[: len(FLAT_OBSERVATION_LAYOUT)] == FLAT_OBSERVATION_LAYOUT
    assert MODEL_INPUT_LAYOUT[len(FLAT_OBSERVATION_LAYOUT) :] == MODEL_INPUT_ONEHOT_LAYOUT


def test_model_input_feature_count_matches_layout_and_expected_total() -> None:
    onehot_length = sum(feature.stop - feature.start for feature in MODEL_INPUT_ONEHOT_LAYOUT)

    assert FLAT_OBSERVATION_FEATURE_COUNT + onehot_length == MODEL_INPUT_FEATURE_COUNT
    assert MODEL_INPUT_FEATURE_COUNT == 6242


def test_model_input_is_c_contiguous_and_finite() -> None:
    sample = parse_sample(_load_valid_sample())
    tensorized = tensorize_sample(sample)

    assert tensorized.model_input.flags["C_CONTIGUOUS"]
    assert np.isfinite(tensorized.model_input).all()


def test_model_input_prefix_equals_flat_observation() -> None:
    sample = parse_sample(_load_valid_sample())
    tensorized = tensorize_sample(sample)

    prefix = tensorized.model_input[:FLAT_OBSERVATION_FEATURE_COUNT]

    assert prefix.tobytes() == tensorized.flat_observation.tobytes()


def test_model_input_one_hot_regions_have_at_most_one_set_bit_per_row() -> None:
    sample = parse_sample(_load_valid_sample())
    tensorized = tensorize_sample(sample)

    for feature in MODEL_INPUT_ONEHOT_LAYOUT:
        region = _region(tensorized.model_input, feature.name)
        assert set(np.unique(region)) <= {0.0, 1.0}
        assert bool(np.all((region.sum(axis=-1) == 0.0) | (region.sum(axis=-1) == 1.0)))


def test_model_input_special_card_indices_one_hot_matches_fixture() -> None:
    # Fixture: oruma=0, yoromeki=15, seiJack=29, uraJack=16 (see valid_sample.json).
    sample = parse_sample(_load_valid_sample())
    tensorized = tensorize_sample(sample)

    region = _region(tensorized.model_input, "specialCardIndicesOneHot")

    assert region[0, 0] == 1.0
    assert region[1, 15] == 1.0
    assert region[2, 29] == 1.0
    assert region[3, 16] == 1.0
    assert region.sum() == 4.0


def test_model_input_current_and_completed_trick_indices_are_all_zero_when_empty() -> None:
    # Fixture: no trick has started yet, so every trick card/player index is -1.
    sample = parse_sample(_load_valid_sample())
    tensorized = tensorize_sample(sample)

    for name in (
        "currentTrickCardIndicesOneHot",
        "completedTrickCardIndicesOneHot",
        "currentTrickPlayerIndicesOneHot",
        "completedTrickPlayerIndicesOneHot",
        "completedTrickWinnerIndicesOneHot",
    ):
        assert _region(tensorized.model_input, name).sum() == 0.0


def test_model_input_bidding_history_one_hot_for_bid_pass_and_empty_slots() -> None:
    # Fixture bidding history (see valid_sample.json):
    #   slot 0: bid,  player 1, suit 1 (hearts),   target 13
    #   slot 1: pass, player 2
    #   slot 4: bid,  player 0, suit 2 (diamonds),  target 15
    #   slot 9: empty (actionMask 0)
    sample = parse_sample(_load_valid_sample())
    tensorized = tensorize_sample(sample)

    action_type = _region(tensorized.model_input, "biddingHistoryActionTypeIndicesOneHot")
    player = _region(tensorized.model_input, "biddingHistoryPlayerIndicesOneHot")
    suit = _region(tensorized.model_input, "biddingHistorySuitIndicesOneHot")
    target = _region(tensorized.model_input, "biddingHistoryTargetPointCardsOneHot")

    # Slot 0: bid (class 1), player 1, suit 1 (hearts), target 13 -> class 0.
    assert action_type[0, 1] == 1.0
    assert player[0, 1] == 1.0
    assert suit[0, 1] == 1.0
    assert target[0, 0] == 1.0

    # Slot 1: pass (class 0), player 2; suit/target have no meaning for a pass.
    assert action_type[1, 0] == 1.0
    assert player[1, 2] == 1.0
    assert suit[1].sum() == 0.0
    assert target[1].sum() == 0.0

    # Slot 4: bid (class 1), player 0, suit 2 (diamonds), target 15 -> class 2.
    assert action_type[4, 1] == 1.0
    assert player[4, 0] == 1.0
    assert suit[4, 2] == 1.0
    assert target[4, 2] == 1.0

    # Slot 9: unused slot (actionMask 0) -- every field is the empty sentinel.
    assert action_type[9].sum() == 0.0
    assert player[9].sum() == 0.0
    assert suit[9].sum() == 0.0
    assert target[9].sum() == 0.0


def test_same_sample_produces_byte_identical_model_input() -> None:
    raw = _load_valid_sample()
    first = tensorize_sample(parse_sample(json.loads(json.dumps(raw))))
    second = tensorize_sample(parse_sample(json.loads(json.dumps(raw))))

    assert first.model_input.tobytes() == second.model_input.tobytes()


def test_changing_one_index_field_only_changes_its_model_input_one_hot_slice() -> None:
    raw = _load_valid_sample()
    baseline = tensorize_sample(parse_sample(json.loads(json.dumps(raw)))).model_input

    mutated_raw = json.loads(json.dumps(raw))
    # oruma=0 in the fixture; move it to a different, still-valid card index.
    mutated_raw["observation"]["specialCardIndices"]["oruma"] = 7

    mutated = tensorize_sample(parse_sample(mutated_raw)).model_input

    changed_indices = {index for index in range(len(baseline)) if baseline[index] != mutated[index]}

    special_card_slice = _find_model_input_slice("specialCardIndicesOneHot")
    # Row 0 of the (4, CARD_COUNT) region is `oruma`; only its old (col 0) and
    # new (col 7) one-hot positions should flip.
    expected_changed_indices = {special_card_slice.start + 0, special_card_slice.start + 7}

    assert changed_indices == expected_changed_indices


def test_validate_tensorized_sample_rejects_wrong_model_input_shape() -> None:
    sample = parse_sample(_load_valid_sample())
    tensorized = tensorize_sample(sample)

    truncated = tensorized.model_input[:-1]
    bad = dataclasses.replace(tensorized, model_input=truncated)

    with pytest.raises(SampleValidationError, match="model_input"):
        validate_tensorized_sample(bad)


def test_validate_tensorized_sample_rejects_model_input_not_starting_with_flat_observation() -> (
    None
):
    sample = parse_sample(_load_valid_sample())
    tensorized = tensorize_sample(sample)

    tampered = tensorized.model_input.copy()
    tampered.setflags(write=True)
    tampered[0] = tampered[0] + 1.0
    tampered.setflags(write=False)
    bad = dataclasses.replace(tensorized, model_input=tampered)

    with pytest.raises(SampleValidationError, match="flat_observation"):
        validate_tensorized_sample(bad)
