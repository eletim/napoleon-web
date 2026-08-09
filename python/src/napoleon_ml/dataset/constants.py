"""Fixed schema values mirrored from the TypeScript source of truth.

Every value in this module is copied from a specific TypeScript file rather
than re-derived, so a schema change on the TypeScript side is a merge
conflict here instead of a silent divergence. The TypeScript sources are:

- ``packages/ai-observation/src/schema.ts``
- ``packages/ai-observation/src/cardIndex.ts``
- ``packages/training-data/src/schema.ts``

Playing samples currently use encoder/model-input schema v2, while non-playing
phase samples remain on their phase-specific v1 schemas. Keep these names in
lockstep with TypeScript so version drift fails during validation instead of
silently tensorizing incompatible artifacts.
"""

from __future__ import annotations

# --- packages/training-data/src/schema.ts -----------------------------------

DATASET_SCHEMA_VERSION = 1
DATASET_GENERATOR_VERSION = 1
MULTIPHASE_DATASET_SCHEMA_VERSION = 2
MULTIPHASE_DATASET_GENERATOR_VERSION = 2
RULE_BASED_AGENT_VERSION = 1
DATASET_FORMAT = "jsonl"
PLAYING_DATASET_SAMPLE_TYPE = "playing-training-sample"
BIDDING_DATASET_SAMPLE_TYPE = "bidding-training-sample"
EXCHANGE_DATASET_SAMPLE_TYPE = "exchange-training-sample"
ADJUTANT_DATASET_SAMPLE_TYPE = "adjutant-training-sample"
DATASET_SAMPLE_TYPE = PLAYING_DATASET_SAMPLE_TYPE
DATASET_SAMPLE_TYPES = (
    PLAYING_DATASET_SAMPLE_TYPE,
    BIDDING_DATASET_SAMPLE_TYPE,
    EXCHANGE_DATASET_SAMPLE_TYPE,
    ADJUTANT_DATASET_SAMPLE_TYPE,
)
PLAYING_SELF_PLAY_DATASET_SCHEMA_VERSION = 3
PLAYING_SELF_PLAY_DATASET_GENERATOR_VERSION = 1
PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE = "playing-self-play-sample"
PLAYING_SELF_PLAY_SAMPLE_SCHEMA_VERSION = 3
PLAYING_SELF_PLAY_SAMPLING_ALGORITHM = "masked-categorical"
PLAYING_SELF_PLAY_ROSTER_ASSIGNMENT = "rotate-by-seed"
CURRENT_POLICY_ROSTER_SOURCE = "current-policy"
RULE_BASED_ROSTER_SOURCE = "rule-based"
FROZEN_ONNX_ROSTER_SOURCE = "frozen-onnx"
ROLLOUT_ROSTER_SOURCES = (
    CURRENT_POLICY_ROSTER_SOURCE,
    RULE_BASED_ROSTER_SOURCE,
    FROZEN_ONNX_ROSTER_SOURCE,
)
PLAYING_SELF_PLAY_REWARD_TYPE = "terminal-team-win"
PLAYING_SELF_PLAY_REWARD_VERSION = 1
PLAYING_SELF_PLAY_BEHAVIOR_POLICY_TYPE = "playing-onnx"
SHARD_FILE_DIGITS = 5
MAX_SHARD_COUNT = 10**SHARD_FILE_DIGITS
UINT32_MAX = 0xFFFFFFFF

# --- packages/ai-observation/src/schema.ts -----------------------------------

PLAYING_ENCODER_SCHEMA_VERSION = 2
PLAYING_MODEL_INPUT_SCHEMA_VERSION = 2
BIDDING_ENCODER_SCHEMA_VERSION = 1
EXCHANGE_ENCODER_SCHEMA_VERSION = 1
ADJUTANT_ENCODER_SCHEMA_VERSION = 1

CARD_COUNT = 53
PLAYER_COUNT = 5
TRICK_COUNT = 10
CARDS_PER_TRICK = 5

EMPTY_CARD_INDEX = -1
EMPTY_PLAYER_INDEX = -1
NOT_IN_HAND_CLASS_INDEX = 5
BELIEF_OWNER_CLASS_COUNT = NOT_IN_HAND_CLASS_INDEX + 1
SELF_ROLE_ORDER: tuple[str, ...] = ("napoleon", "adjutant", "alliance", "napoleon-solo")
SELF_ROLE_COUNT = len(SELF_ROLE_ORDER)

# packages/ai-observation/src/schema.ts: BIDDING_HISTORY_SUIT_ORDER
BIDDING_HISTORY_SUIT_ORDER: tuple[str, ...] = ("spades", "hearts", "diamonds", "clubs")
BIDDING_ACTION_TYPE_PASS = 0
BIDDING_ACTION_TYPE_BID = 1
EMPTY_BIDDING_ACTION_TYPE = -1
EMPTY_BIDDING_SUIT_INDEX = -1

# packages/game-core/src/bidding.ts
MIN_BIDDING_TARGET_POINT_CARDS = 13
MAX_BIDDING_TARGET_POINT_CARDS = 19
# packages/game-core/src/bidding.ts: biddingSuitOrder length (clubs/diamonds/hearts/spades)
_BIDDING_SUIT_COUNT = 4
_BIDDING_BID_ACTION_COUNT = (
    MAX_BIDDING_TARGET_POINT_CARDS - MIN_BIDDING_TARGET_POINT_CARDS + 1
) * _BIDDING_SUIT_COUNT
BIDDING_ACTION_COUNT = 1 + _BIDDING_BID_ACTION_COUNT
MAX_BIDDING_ACTION_COUNT = PLAYER_COUNT + _BIDDING_BID_ACTION_COUNT * (PLAYER_COUNT - 1)

# packages/ai-observation/src/encodePlayingObservation.ts
TRUMP_SUIT_ORDER: tuple[str, ...] = ("spades", "hearts", "diamonds", "clubs")
REVEALED_ADJUTANT_CLASS_COUNT = PLAYER_COUNT + 1
COMPLETED_TRICK_CARD_SLOT_COUNT = TRICK_COUNT * CARDS_PER_TRICK

# packages/ai-observation/src/encodePlayingObservation.ts: contractTargetPointCards range
MIN_CONTRACT_TARGET_POINT_CARDS = 12
MAX_CONTRACT_TARGET_POINT_CARDS = MAX_BIDDING_TARGET_POINT_CARDS
MAX_HAND_COUNT = 13
MAX_LATEST_BURIED_EVENT_HIDDEN_NON_POINT_COUNT = 3

# --- packages/ai-observation/src/cardIndex.ts --------------------------------

_CARD_SUIT_ORDER: tuple[str, ...] = ("spades", "hearts", "diamonds", "clubs")
_CARD_RANK_ORDER: tuple[str, ...] = (
    "A",
    "K",
    "Q",
    "J",
    "10",
    "9",
    "8",
    "7",
    "6",
    "5",
    "4",
    "3",
    "2",
)

EXPECTED_CARD_IDS: tuple[str, ...] = tuple(
    f"{suit}-{rank}" for suit in _CARD_SUIT_ORDER for rank in _CARD_RANK_ORDER
) + ("joker",)

if len(EXPECTED_CARD_IDS) != CARD_COUNT:
    raise AssertionError(
        f"EXPECTED_CARD_IDS must contain {CARD_COUNT} cards, got {len(EXPECTED_CARD_IDS)}."
    )

if len(set(EXPECTED_CARD_IDS)) != CARD_COUNT:
    raise AssertionError("EXPECTED_CARD_IDS must not contain duplicates.")
