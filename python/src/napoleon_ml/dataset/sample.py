"""Typed representation of one JSONL row (TypeScript ``PlayingTrainingSample``).

Mirrors, field-for-field, the following TypeScript sources:

- ``packages/ai-observation/src/createPlayingTrainingSample.ts`` (``PlayingTrainingSample``)
- ``packages/ai-observation/src/encodePlayingObservation.ts`` (``EncodedPlayingObservation``)
- ``packages/ai-observation/src/encodeBiddingHistory.ts`` (``EncodedBiddingHistory``)
- ``packages/ai-observation/src/encodePlayAction.ts`` (``EncodedPlayAction``)
- ``packages/ai-observation/src/encodeBeliefTarget.ts`` (``EncodedBeliefTarget``)

Parsing here only enforces JSON *shape*: required keys present, no unknown
keys, every value has the declared JSON type, and every fixed-length array
has its schema-v1 length. No numeric-range or cross-field checking is done
here; that lives in :mod:`napoleon_ml.dataset.validation`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._strict import (
    ErrorFactory,
    require_dict,
    require_exact_keys,
    require_fixed_length_int_tuple,
    require_int,
    require_list,
    require_str,
    require_str_tuple,
)
from .constants import (
    CARD_COUNT,
    CARDS_PER_TRICK,
    MAX_BIDDING_ACTION_COUNT,
    PLAYER_COUNT,
    REVEALED_ADJUTANT_CLASS_COUNT,
    TRICK_COUNT,
)
from .errors import SampleValidationError

_COMPLETED_TRICK_CARD_SLOT_COUNT = TRICK_COUNT * CARDS_PER_TRICK
_TRUMP_SUIT_OPTION_COUNT = 4

_SPECIAL_CARD_INDICES_KEYS = frozenset({"oruma", "yoromeki", "seiJack", "uraJack"})
_BIDDING_HISTORY_KEYS = frozenset(
    {"actionTypeIndices", "playerIndices", "suitIndices", "targetPointCards", "actionMask"}
)
_OBSERVATION_KEYS = frozenset(
    {
        "schemaVersion",
        "relativePlayerIds",
        "trickNumber",
        "completedTrickCount",
        "contractTargetPointCards",
        "trumpSuitOneHot",
        "napoleonPlayerOneHot",
        "revealedAdjutantPlayerOneHot",
        "calledAdjutantCardMask",
        "selfHandMask",
        "legalPlayMask",
        "handCountByPlayer",
        "capturedPointCardMaskByPlayer",
        "specialCardIndices",
        "currentTrickCardIndices",
        "currentTrickPlayerIndices",
        "currentTrickSlotMask",
        "completedTrickCardIndices",
        "completedTrickPlayerIndices",
        "completedTrickSlotMask",
        "completedTrickWinnerIndices",
        "completedTrickMask",
        "biddingHistory",
        "latestBuriedEventPointCardMask",
        "latestBuriedEventHiddenNonPointCount",
        "latestBuriedEventPresent",
    }
)
_ACTOR_TARGET_KEYS = frozenset({"selectedCardIndex"})
_BELIEF_TARGET_KEYS = frozenset({"ownerClassByCard", "hiddenOwnershipLossMask"})
_SAMPLE_KEYS = frozenset(
    {
        "schemaVersion",
        "seed",
        "step",
        "actingPlayerId",
        "relativePlayerIds",
        "observation",
        "actorTarget",
        "beliefTarget",
    }
)


@dataclass(frozen=True)
class SpecialCardIndices:
    oruma: int
    yoromeki: int
    sei_jack: int
    ura_jack: int


@dataclass(frozen=True)
class EncodedBiddingHistory:
    action_type_indices: tuple[int, ...]
    player_indices: tuple[int, ...]
    suit_indices: tuple[int, ...]
    target_point_cards: tuple[int, ...]
    action_mask: tuple[int, ...]


@dataclass(frozen=True)
class EncodedPlayingObservation:
    schema_version: int
    relative_player_ids: tuple[str, ...]
    trick_number: int
    completed_trick_count: int
    contract_target_point_cards: int
    trump_suit_one_hot: tuple[int, ...]
    napoleon_player_one_hot: tuple[int, ...]
    revealed_adjutant_player_one_hot: tuple[int, ...]
    called_adjutant_card_mask: tuple[int, ...]
    self_hand_mask: tuple[int, ...]
    legal_play_mask: tuple[int, ...]
    hand_count_by_player: tuple[int, ...]
    captured_point_card_mask_by_player: tuple[tuple[int, ...], ...]
    special_card_indices: SpecialCardIndices
    current_trick_card_indices: tuple[int, ...]
    current_trick_player_indices: tuple[int, ...]
    current_trick_slot_mask: tuple[int, ...]
    completed_trick_card_indices: tuple[int, ...]
    completed_trick_player_indices: tuple[int, ...]
    completed_trick_slot_mask: tuple[int, ...]
    completed_trick_winner_indices: tuple[int, ...]
    completed_trick_mask: tuple[int, ...]
    bidding_history: EncodedBiddingHistory
    latest_buried_event_point_card_mask: tuple[int, ...]
    latest_buried_event_hidden_non_point_count: int
    latest_buried_event_present: int


@dataclass(frozen=True)
class EncodedPlayAction:
    selected_card_index: int


@dataclass(frozen=True)
class EncodedBeliefTarget:
    owner_class_by_card: tuple[int, ...]
    hidden_ownership_loss_mask: tuple[int, ...]


@dataclass(frozen=True)
class PlayingTrainingSample:
    schema_version: int
    seed: int
    step: int
    acting_player_id: str
    relative_player_ids: tuple[str, ...]
    observation: EncodedPlayingObservation
    actor_target: EncodedPlayAction
    belief_target: EncodedBeliefTarget


def _make_error(context: str) -> ErrorFactory:
    def factory(message: str) -> SampleValidationError:
        return SampleValidationError(f"{context}: {message}")

    return factory


def _parse_special_card_indices(raw: object, *, error: ErrorFactory) -> SpecialCardIndices:
    path = "observation.specialCardIndices"
    obj = require_dict(raw, path=path, error=error)
    require_exact_keys(obj, _SPECIAL_CARD_INDICES_KEYS, path=path, error=error)

    return SpecialCardIndices(
        oruma=require_int(obj["oruma"], path=f"{path}.oruma", error=error),
        yoromeki=require_int(obj["yoromeki"], path=f"{path}.yoromeki", error=error),
        sei_jack=require_int(obj["seiJack"], path=f"{path}.seiJack", error=error),
        ura_jack=require_int(obj["uraJack"], path=f"{path}.uraJack", error=error),
    )


def _parse_bidding_history(raw: object, *, error: ErrorFactory) -> EncodedBiddingHistory:
    path = "observation.biddingHistory"
    obj = require_dict(raw, path=path, error=error)
    require_exact_keys(obj, _BIDDING_HISTORY_KEYS, path=path, error=error)

    return EncodedBiddingHistory(
        action_type_indices=require_fixed_length_int_tuple(
            obj["actionTypeIndices"],
            path=f"{path}.actionTypeIndices",
            length=MAX_BIDDING_ACTION_COUNT,
            error=error,
        ),
        player_indices=require_fixed_length_int_tuple(
            obj["playerIndices"],
            path=f"{path}.playerIndices",
            length=MAX_BIDDING_ACTION_COUNT,
            error=error,
        ),
        suit_indices=require_fixed_length_int_tuple(
            obj["suitIndices"],
            path=f"{path}.suitIndices",
            length=MAX_BIDDING_ACTION_COUNT,
            error=error,
        ),
        target_point_cards=require_fixed_length_int_tuple(
            obj["targetPointCards"],
            path=f"{path}.targetPointCards",
            length=MAX_BIDDING_ACTION_COUNT,
            error=error,
        ),
        action_mask=require_fixed_length_int_tuple(
            obj["actionMask"],
            path=f"{path}.actionMask",
            length=MAX_BIDDING_ACTION_COUNT,
            error=error,
        ),
    )


def _parse_captured_point_card_mask_by_player(
    raw: object, *, error: ErrorFactory
) -> tuple[tuple[int, ...], ...]:
    path = "observation.capturedPointCardMaskByPlayer"
    items = require_list(raw, path=path, error=error)

    if len(items) != PLAYER_COUNT:
        raise error(f"{path} must have length {PLAYER_COUNT}, got {len(items)}.")

    return tuple(
        require_fixed_length_int_tuple(
            item, path=f"{path}[{index}]", length=CARD_COUNT, error=error
        )
        for index, item in enumerate(items)
    )


def _parse_observation(raw: object, *, error: ErrorFactory) -> EncodedPlayingObservation:
    path = "observation"
    obj = require_dict(raw, path=path, error=error)
    require_exact_keys(obj, _OBSERVATION_KEYS, path=path, error=error)

    return EncodedPlayingObservation(
        schema_version=require_int(obj["schemaVersion"], path=f"{path}.schemaVersion", error=error),
        relative_player_ids=require_str_tuple(
            obj["relativePlayerIds"], path=f"{path}.relativePlayerIds", error=error
        ),
        trick_number=require_int(obj["trickNumber"], path=f"{path}.trickNumber", error=error),
        completed_trick_count=require_int(
            obj["completedTrickCount"], path=f"{path}.completedTrickCount", error=error
        ),
        contract_target_point_cards=require_int(
            obj["contractTargetPointCards"], path=f"{path}.contractTargetPointCards", error=error
        ),
        trump_suit_one_hot=require_fixed_length_int_tuple(
            obj["trumpSuitOneHot"],
            path=f"{path}.trumpSuitOneHot",
            length=_TRUMP_SUIT_OPTION_COUNT,
            error=error,
        ),
        napoleon_player_one_hot=require_fixed_length_int_tuple(
            obj["napoleonPlayerOneHot"],
            path=f"{path}.napoleonPlayerOneHot",
            length=PLAYER_COUNT,
            error=error,
        ),
        revealed_adjutant_player_one_hot=require_fixed_length_int_tuple(
            obj["revealedAdjutantPlayerOneHot"],
            path=f"{path}.revealedAdjutantPlayerOneHot",
            length=REVEALED_ADJUTANT_CLASS_COUNT,
            error=error,
        ),
        called_adjutant_card_mask=require_fixed_length_int_tuple(
            obj["calledAdjutantCardMask"],
            path=f"{path}.calledAdjutantCardMask",
            length=CARD_COUNT,
            error=error,
        ),
        self_hand_mask=require_fixed_length_int_tuple(
            obj["selfHandMask"], path=f"{path}.selfHandMask", length=CARD_COUNT, error=error
        ),
        legal_play_mask=require_fixed_length_int_tuple(
            obj["legalPlayMask"], path=f"{path}.legalPlayMask", length=CARD_COUNT, error=error
        ),
        hand_count_by_player=require_fixed_length_int_tuple(
            obj["handCountByPlayer"],
            path=f"{path}.handCountByPlayer",
            length=PLAYER_COUNT,
            error=error,
        ),
        captured_point_card_mask_by_player=_parse_captured_point_card_mask_by_player(
            obj["capturedPointCardMaskByPlayer"], error=error
        ),
        special_card_indices=_parse_special_card_indices(obj["specialCardIndices"], error=error),
        current_trick_card_indices=require_fixed_length_int_tuple(
            obj["currentTrickCardIndices"],
            path=f"{path}.currentTrickCardIndices",
            length=CARDS_PER_TRICK,
            error=error,
        ),
        current_trick_player_indices=require_fixed_length_int_tuple(
            obj["currentTrickPlayerIndices"],
            path=f"{path}.currentTrickPlayerIndices",
            length=CARDS_PER_TRICK,
            error=error,
        ),
        current_trick_slot_mask=require_fixed_length_int_tuple(
            obj["currentTrickSlotMask"],
            path=f"{path}.currentTrickSlotMask",
            length=CARDS_PER_TRICK,
            error=error,
        ),
        completed_trick_card_indices=require_fixed_length_int_tuple(
            obj["completedTrickCardIndices"],
            path=f"{path}.completedTrickCardIndices",
            length=_COMPLETED_TRICK_CARD_SLOT_COUNT,
            error=error,
        ),
        completed_trick_player_indices=require_fixed_length_int_tuple(
            obj["completedTrickPlayerIndices"],
            path=f"{path}.completedTrickPlayerIndices",
            length=_COMPLETED_TRICK_CARD_SLOT_COUNT,
            error=error,
        ),
        completed_trick_slot_mask=require_fixed_length_int_tuple(
            obj["completedTrickSlotMask"],
            path=f"{path}.completedTrickSlotMask",
            length=_COMPLETED_TRICK_CARD_SLOT_COUNT,
            error=error,
        ),
        completed_trick_winner_indices=require_fixed_length_int_tuple(
            obj["completedTrickWinnerIndices"],
            path=f"{path}.completedTrickWinnerIndices",
            length=TRICK_COUNT,
            error=error,
        ),
        completed_trick_mask=require_fixed_length_int_tuple(
            obj["completedTrickMask"],
            path=f"{path}.completedTrickMask",
            length=TRICK_COUNT,
            error=error,
        ),
        bidding_history=_parse_bidding_history(obj["biddingHistory"], error=error),
        latest_buried_event_point_card_mask=require_fixed_length_int_tuple(
            obj["latestBuriedEventPointCardMask"],
            path=f"{path}.latestBuriedEventPointCardMask",
            length=CARD_COUNT,
            error=error,
        ),
        latest_buried_event_hidden_non_point_count=require_int(
            obj["latestBuriedEventHiddenNonPointCount"],
            path=f"{path}.latestBuriedEventHiddenNonPointCount",
            error=error,
        ),
        latest_buried_event_present=require_int(
            obj["latestBuriedEventPresent"], path=f"{path}.latestBuriedEventPresent", error=error
        ),
    )


def _parse_actor_target(raw: object, *, error: ErrorFactory) -> EncodedPlayAction:
    path = "actorTarget"
    obj = require_dict(raw, path=path, error=error)
    require_exact_keys(obj, _ACTOR_TARGET_KEYS, path=path, error=error)

    return EncodedPlayAction(
        selected_card_index=require_int(
            obj["selectedCardIndex"], path=f"{path}.selectedCardIndex", error=error
        )
    )


def _parse_belief_target(raw: object, *, error: ErrorFactory) -> EncodedBeliefTarget:
    path = "beliefTarget"
    obj = require_dict(raw, path=path, error=error)
    require_exact_keys(obj, _BELIEF_TARGET_KEYS, path=path, error=error)

    return EncodedBeliefTarget(
        owner_class_by_card=require_fixed_length_int_tuple(
            obj["ownerClassByCard"], path=f"{path}.ownerClassByCard", length=CARD_COUNT, error=error
        ),
        hidden_ownership_loss_mask=require_fixed_length_int_tuple(
            obj["hiddenOwnershipLossMask"],
            path=f"{path}.hiddenOwnershipLossMask",
            length=CARD_COUNT,
            error=error,
        ),
    )


def parse_sample(raw: object, *, context: str = "sample") -> PlayingTrainingSample:
    """Parse an already JSON-decoded sample value into a typed dataclass.

    ``context`` prefixes every error message (e.g. ``"shard-00003.jsonl:127"``)
    so a failure can be located without a debugger. Raises
    :class:`~napoleon_ml.dataset.errors.SampleValidationError` on any
    structural mismatch; call
    :func:`napoleon_ml.dataset.validation.validate_sample` on the result for
    semantic (range/cross-field) validation.
    """

    error = _make_error(context)
    obj = require_dict(raw, path="sample", error=error)
    require_exact_keys(obj, _SAMPLE_KEYS, path="sample", error=error)

    return PlayingTrainingSample(
        schema_version=require_int(obj["schemaVersion"], path="schemaVersion", error=error),
        seed=require_int(obj["seed"], path="seed", error=error),
        step=require_int(obj["step"], path="step", error=error),
        acting_player_id=require_str(obj["actingPlayerId"], path="actingPlayerId", error=error),
        relative_player_ids=require_str_tuple(
            obj["relativePlayerIds"], path="relativePlayerIds", error=error
        ),
        observation=_parse_observation(obj["observation"], error=error),
        actor_target=_parse_actor_target(obj["actorTarget"], error=error),
        belief_target=_parse_belief_target(obj["beliefTarget"], error=error),
    )
