import {
  maxBidTargetPointCards,
  minBidTargetPointCards
} from "@napoleon/game-core";
import type { Suit } from "@napoleon/game-core";

export const PLAYING_ENCODER_SCHEMA_VERSION = 1 as const;
export const BIDDING_ENCODER_SCHEMA_VERSION = 1 as const;
export const EXCHANGE_ENCODER_SCHEMA_VERSION = 1 as const;
export const MODEL_INPUT_SCHEMA_VERSION = 1 as const;

export const CARD_COUNT = 53;
export const PLAYER_COUNT = 5;
export const TRICK_COUNT = 10;
export const CARDS_PER_TRICK = 5;

export const EMPTY_CARD_INDEX = -1;
export const EMPTY_PLAYER_INDEX = -1;
export const NOT_IN_HAND_CLASS_INDEX = 5;

export const BIDDING_HISTORY_SUIT_ORDER: readonly Suit[] = [
  "spades",
  "hearts",
  "diamonds",
  "clubs"
];
export const BIDDING_ACTION_TYPE_PASS = 0;
export const BIDDING_ACTION_TYPE_BID = 1;
export const EMPTY_BIDDING_ACTION_TYPE = -1;
export const EMPTY_BIDDING_SUIT_INDEX = -1;

export const MIN_BIDDING_TARGET_POINT_CARDS = minBidTargetPointCards;
export const MAX_BIDDING_TARGET_POINT_CARDS = maxBidTargetPointCards;
const BIDDING_BID_ACTION_COUNT =
  (MAX_BIDDING_TARGET_POINT_CARDS - MIN_BIDDING_TARGET_POINT_CARDS + 1) *
  BIDDING_HISTORY_SUIT_ORDER.length;
export const BIDDING_ACTION_COUNT =
  1 + BIDDING_BID_ACTION_COUNT;

export const MAX_BIDDING_ACTION_COUNT =
  PLAYER_COUNT + BIDDING_BID_ACTION_COUNT * (PLAYER_COUNT - 1);

export const FLAT_OBSERVATION_FEATURE_COUNT = 684 as const;
export const MODEL_INPUT_FEATURE_COUNT = 6242 as const;
