from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from napoleon_ml.dataset.errors import SampleValidationError
from napoleon_ml.dataset.sample import PlayingSelfPlaySample, parse_sample
from napoleon_ml.dataset.validation import validate_sample

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "valid_sample.json"


def _load_valid_sample() -> dict[str, Any]:
    """Load a single real sample produced by the TypeScript self-play CLI (seed 0, step 12)."""

    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _parse_and_validate(raw: dict[str, Any]) -> None:
    validate_sample(parse_sample(raw))


def _load_valid_self_play_sample() -> dict[str, Any]:
    raw = _load_valid_sample()
    selected_card_index = raw["actorTarget"]["selectedCardIndex"]
    del raw["actorTarget"]
    del raw["beliefTarget"]
    raw.update(
        {
            "sampleType": "playing-self-play-sample",
            "selectedCardIndex": selected_card_index,
            "behaviorLogProbability": -0.5,
            "terminalReward": 1,
            "outcome": {
                "winner": "napoleon-team",
                "napoleonPlayerId": "player-4",
                "actingPlayerTeam": "napoleon-team",
                "actingPlayerRole": "napoleon",
            },
        }
    )
    return raw


def _parse_and_validate_self_play(raw: dict[str, Any]) -> PlayingSelfPlaySample:
    sample = parse_sample(raw, sample_type="playing-self-play-sample")
    validate_sample(sample)
    assert isinstance(sample, PlayingSelfPlaySample)
    return sample


def test_valid_sample_passes() -> None:
    raw = _load_valid_sample()
    sample = parse_sample(raw)
    validate_sample(sample)

    assert sample.seed == 0
    assert sample.step == 12
    assert sample.acting_player_id == "player-4"
    assert len(sample.observation.legal_play_mask) == 53
    assert len(sample.belief_target.owner_class_by_card) == 53


def test_valid_self_play_sample_passes() -> None:
    sample = _parse_and_validate_self_play(_load_valid_self_play_sample())

    assert sample.sample_type == "playing-self-play-sample"
    assert sample.observation.legal_play_mask[sample.selected_card_index] == 1
    assert sample.terminal_reward == 1
    assert sample.outcome.acting_player_role == "napoleon"


def test_self_play_unknown_hidden_fields_rejected() -> None:
    for hidden_key in (
        "actualState",
        "actualHands",
        "opponentPrivateHand",
        "unusedCards",
        "teacherAction",
        "futureAction",
        "actorTarget",
        "beliefTarget",
    ):
        raw = _load_valid_self_play_sample()
        raw[hidden_key] = {}

        with pytest.raises(SampleValidationError, match="unknown key"):
            _parse_and_validate_self_play(raw)


def test_self_play_illegal_selected_action_rejected() -> None:
    raw = _load_valid_self_play_sample()
    raw["selectedCardIndex"] = raw["observation"]["legalPlayMask"].index(0)

    with pytest.raises(SampleValidationError, match="selectedCardIndex"):
        _parse_and_validate_self_play(raw)


def test_self_play_positive_log_probability_rejected() -> None:
    raw = _load_valid_self_play_sample()
    raw["behaviorLogProbability"] = 0.1

    with pytest.raises(SampleValidationError, match="behaviorLogProbability"):
        _parse_and_validate_self_play(raw)


def test_self_play_forced_action_requires_zero_log_probability() -> None:
    raw = _load_valid_self_play_sample()
    selected = raw["selectedCardIndex"]
    raw["observation"]["legalPlayMask"] = [1 if index == selected else 0 for index in range(53)]
    raw["behaviorLogProbability"] = -0.1

    with pytest.raises(SampleValidationError, match="forced"):
        _parse_and_validate_self_play(raw)


def test_self_play_outcome_reward_mismatch_rejected() -> None:
    raw = _load_valid_self_play_sample()
    raw["terminalReward"] = -1

    with pytest.raises(SampleValidationError, match="terminalReward"):
        _parse_and_validate_self_play(raw)


def test_self_play_role_team_mismatch_rejected() -> None:
    raw = _load_valid_self_play_sample()
    raw["outcome"]["actingPlayerRole"] = "alliance"

    with pytest.raises(SampleValidationError, match="actingPlayerTeam"):
        _parse_and_validate_self_play(raw)


def test_self_play_missing_outcome_key_rejected() -> None:
    raw = _load_valid_self_play_sample()
    del raw["outcome"]["winner"]

    with pytest.raises(SampleValidationError, match="winner"):
        parse_sample(raw, sample_type="playing-self-play-sample")


def test_schema_version_mismatch_rejected() -> None:
    raw = _load_valid_sample()
    raw["schemaVersion"] = 2

    with pytest.raises(SampleValidationError, match="schemaVersion"):
        _parse_and_validate(raw)


def test_seed_invalid_rejected() -> None:
    raw = _load_valid_sample()
    raw["seed"] = -1

    with pytest.raises(SampleValidationError, match="seed"):
        _parse_and_validate(raw)


def test_step_invalid_rejected() -> None:
    raw = _load_valid_sample()
    raw["step"] = 0

    with pytest.raises(SampleValidationError, match="step"):
        _parse_and_validate(raw)


def test_relative_player_ids_length_invalid_rejected() -> None:
    raw = _load_valid_sample()
    raw["relativePlayerIds"] = raw["relativePlayerIds"][:-1]

    with pytest.raises(SampleValidationError, match="relativePlayerIds"):
        _parse_and_validate(raw)


def test_relative_player_ids_duplicate_rejected() -> None:
    raw = _load_valid_sample()
    ids = list(raw["relativePlayerIds"])
    ids[-1] = ids[0]
    raw["relativePlayerIds"] = ids

    with pytest.raises(SampleValidationError, match="unique"):
        _parse_and_validate(raw)


def test_acting_player_id_mismatch_rejected() -> None:
    raw = _load_valid_sample()
    raw["actingPlayerId"] = raw["relativePlayerIds"][1]

    with pytest.raises(SampleValidationError, match="actingPlayerId"):
        _parse_and_validate(raw)


def test_observation_relative_player_mismatch_rejected() -> None:
    raw = _load_valid_sample()
    ids = list(raw["relativePlayerIds"])
    ids[1], ids[2] = ids[2], ids[1]
    raw["relativePlayerIds"] = ids  # actingPlayerId (index 0) is unchanged

    with pytest.raises(SampleValidationError, match="observation.relativePlayerIds"):
        _parse_and_validate(raw)


def test_mask_length_invalid_rejected() -> None:
    raw = _load_valid_sample()
    raw["observation"]["selfHandMask"] = raw["observation"]["selfHandMask"][:-1]

    with pytest.raises(SampleValidationError, match="selfHandMask"):
        _parse_and_validate(raw)


def test_mask_non_binary_value_rejected() -> None:
    raw = _load_valid_sample()
    mask = list(raw["observation"]["selfHandMask"])
    mask[0] = 2
    raw["observation"]["selfHandMask"] = mask

    with pytest.raises(SampleValidationError, match="0/1"):
        _parse_and_validate(raw)


def test_one_hot_invalid_rejected() -> None:
    raw = _load_valid_sample()
    raw["observation"]["trumpSuitOneHot"] = [0, 0, 0, 0]

    with pytest.raises(SampleValidationError, match="sum to 1"):
        _parse_and_validate(raw)


def test_card_index_out_of_range_rejected() -> None:
    raw = _load_valid_sample()
    raw["observation"]["specialCardIndices"]["oruma"] = 53

    with pytest.raises(SampleValidationError, match="specialCardIndices.oruma"):
        _parse_and_validate(raw)


def test_actor_target_out_of_range_rejected() -> None:
    raw = _load_valid_sample()
    raw["actorTarget"]["selectedCardIndex"] = 53

    with pytest.raises(SampleValidationError, match="selectedCardIndex"):
        _parse_and_validate(raw)


def test_actor_target_illegal_rejected() -> None:
    raw = _load_valid_sample()
    legal_mask = raw["observation"]["legalPlayMask"]
    illegal_index = legal_mask.index(0)
    raw["actorTarget"]["selectedCardIndex"] = illegal_index

    with pytest.raises(SampleValidationError, match="legal in observation.legalPlayMask"):
        _parse_and_validate(raw)


def test_legal_mask_outside_self_hand_rejected() -> None:
    raw = _load_valid_sample()
    self_hand_mask = raw["observation"]["selfHandMask"]
    legal_mask = list(raw["observation"]["legalPlayMask"])
    not_in_hand_index = self_hand_mask.index(0)
    legal_mask[not_in_hand_index] = 1
    raw["observation"]["legalPlayMask"] = legal_mask

    with pytest.raises(SampleValidationError, match="within selfHandMask"):
        _parse_and_validate(raw)


def test_belief_target_length_invalid_rejected() -> None:
    raw = _load_valid_sample()
    raw["beliefTarget"]["ownerClassByCard"] = raw["beliefTarget"]["ownerClassByCard"][:-1]

    with pytest.raises(SampleValidationError, match="ownerClassByCard"):
        _parse_and_validate(raw)


def test_belief_owner_class_out_of_range_rejected() -> None:
    raw = _load_valid_sample()
    owner_classes = list(raw["beliefTarget"]["ownerClassByCard"])
    owner_classes[0] = 6
    raw["beliefTarget"]["ownerClassByCard"] = owner_classes

    with pytest.raises(SampleValidationError, match="owner class"):
        _parse_and_validate(raw)


def test_unknown_observation_key_rejected() -> None:
    raw = _load_valid_sample()
    raw["observation"]["extraField"] = 1

    with pytest.raises(SampleValidationError, match="unknown key"):
        _parse_and_validate(raw)


def test_missing_observation_key_rejected() -> None:
    raw = _load_valid_sample()
    del raw["observation"]["legalPlayMask"]

    with pytest.raises(SampleValidationError, match="legalPlayMask"):
        _parse_and_validate(raw)


def test_bool_rejected_where_integer_expected() -> None:
    raw = _load_valid_sample()
    raw["step"] = True

    with pytest.raises(SampleValidationError, match="integer"):
        _parse_and_validate(raw)


def test_error_message_includes_context_prefix() -> None:
    raw = _load_valid_sample()
    del raw["observation"]["legalPlayMask"]

    with pytest.raises(SampleValidationError, match=r"^shard-00003\.jsonl:127:"):
        parse_sample(raw, context="shard-00003.jsonl:127")


def test_sample_dict_is_never_mutated_by_validation() -> None:
    raw = _load_valid_sample()
    snapshot = copy.deepcopy(raw)
    _parse_and_validate(raw)

    assert raw == snapshot
