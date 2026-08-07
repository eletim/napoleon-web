import type { BidAction, GameAction, PassAction, PlayerId, Suit } from "@napoleon/game-core";
import { getBiddingSuitIndex } from "./encodeBiddingHistory.js";
import {
  BIDDING_ACTION_COUNT,
  BIDDING_HISTORY_SUIT_ORDER,
  MAX_BIDDING_TARGET_POINT_CARDS,
  MIN_BIDDING_TARGET_POINT_CARDS
} from "./schema.js";

export type DecodedBiddingAction = BidAction | PassAction;

export function encodeBiddingAction(
  action: GameAction,
  legalBidMask?: readonly number[],
  expectedPlayerId?: PlayerId
): number {
  if (action.type !== "pass" && action.type !== "bid") {
    throw new Error(`encodeBiddingAction requires a pass or bid action, got ${action.type}.`);
  }

  if (expectedPlayerId !== undefined && action.playerId !== expectedPlayerId) {
    throw new Error(
      `Decision action playerId must match decision playerId: ${action.playerId} !== ${expectedPlayerId}`
    );
  }

  const actionIndex = action.type === "pass"
    ? 0
    : encodeBidActionIndex(action.targetPointCards, action.suit);

  if (legalBidMask !== undefined) {
    validateLegalBidMask(legalBidMask);

    if (legalBidMask[actionIndex] !== 1) {
      throw new Error(`Bidding actorTarget ${actionIndex} is not legal for this observation.`);
    }
  }

  return actionIndex;
}

export function decodeBiddingAction(
  actionIndex: number,
  playerId: PlayerId
): DecodedBiddingAction {
  expectIntegerInRange("bidding action index", actionIndex, 0, BIDDING_ACTION_COUNT - 1);

  if (actionIndex === 0) {
    return { type: "pass", playerId };
  }

  const bidOffset = actionIndex - 1;
  const suitIndex = bidOffset % BIDDING_HISTORY_SUIT_ORDER.length;
  const targetPointCards =
    MIN_BIDDING_TARGET_POINT_CARDS +
    Math.floor(bidOffset / BIDDING_HISTORY_SUIT_ORDER.length);
  const suit = BIDDING_HISTORY_SUIT_ORDER[suitIndex];

  if (suit === undefined) {
    throw new Error(`Bidding action index has no suit mapping: ${actionIndex}`);
  }

  return {
    type: "bid",
    playerId,
    suit,
    targetPointCards
  };
}

export function encodeLegalBidMask(
  legalActions: readonly GameAction[],
  expectedPlayerId: PlayerId
): readonly number[] {
  const mask = Array(BIDDING_ACTION_COUNT).fill(0);

  for (const action of legalActions) {
    if (action.type !== "pass" && action.type !== "bid") {
      throw new Error(`Bidding legalActions must contain only pass or bid actions, got ${action.type}.`);
    }

    if (action.playerId !== expectedPlayerId) {
      throw new Error(
        `Legal bidding action playerId must match observation playerId: ${action.playerId} !== ${expectedPlayerId}`
      );
    }

    mask[encodeBiddingAction(action)] = 1;
  }

  return mask;
}

export function validateLegalBidMask(mask: readonly number[]): void {
  if (mask.length !== BIDDING_ACTION_COUNT) {
    throw new Error(`legalBidMask must have length ${BIDDING_ACTION_COUNT}, got ${mask.length}.`);
  }

  for (const value of mask) {
    if (!Number.isFinite(value) || !Number.isInteger(value) || (value !== 0 && value !== 1)) {
      throw new Error("legalBidMask must contain only 0/1 values.");
    }
  }
}

function encodeBidActionIndex(targetPointCards: number, suit: Suit): number {
  expectIntegerInRange(
    "bid targetPointCards",
    targetPointCards,
    MIN_BIDDING_TARGET_POINT_CARDS,
    MAX_BIDDING_TARGET_POINT_CARDS
  );

  return 1 +
    (targetPointCards - MIN_BIDDING_TARGET_POINT_CARDS) * BIDDING_HISTORY_SUIT_ORDER.length +
    getBiddingSuitIndex(suit);
}

function expectIntegerInRange(name: string, value: number, min: number, max: number): void {
  if (!Number.isFinite(value) || !Number.isInteger(value)) {
    throw new Error(`${name} must be a finite integer.`);
  }

  if (value < min || value > max) {
    throw new Error(`${name} must be between ${min} and ${max}, got ${value}.`);
  }
}
