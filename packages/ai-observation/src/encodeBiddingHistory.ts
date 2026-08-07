import type { AutomatedGameRecord, DecisionRecord, PublicActionRecord } from "@napoleon/ai";
import type { GameAction, PlayerId, Suit } from "@napoleon/game-core";
import { getRelativePlayerIndex } from "./playerIndex.js";
import {
  BIDDING_ACTION_TYPE_BID,
  BIDDING_ACTION_TYPE_PASS,
  BIDDING_HISTORY_SUIT_ORDER,
  EMPTY_BIDDING_ACTION_TYPE,
  EMPTY_BIDDING_SUIT_INDEX,
  EMPTY_PLAYER_INDEX,
  MAX_BIDDING_ACTION_COUNT,
  MAX_BIDDING_TARGET_POINT_CARDS,
  MIN_BIDDING_TARGET_POINT_CARDS,
  PLAYER_COUNT
} from "./schema.js";

export interface EncodedBiddingHistory {
  actionTypeIndices: readonly number[];
  playerIndices: readonly number[];
  suitIndices: readonly number[];
  targetPointCards: readonly number[];
  actionMask: readonly number[];
}

export function createEmptyEncodedBiddingHistory(): EncodedBiddingHistory {
  return {
    actionTypeIndices: Array(MAX_BIDDING_ACTION_COUNT).fill(EMPTY_BIDDING_ACTION_TYPE),
    playerIndices: Array(MAX_BIDDING_ACTION_COUNT).fill(EMPTY_PLAYER_INDEX),
    suitIndices: Array(MAX_BIDDING_ACTION_COUNT).fill(EMPTY_BIDDING_SUIT_INDEX),
    targetPointCards: Array(MAX_BIDDING_ACTION_COUNT).fill(0),
    actionMask: Array(MAX_BIDDING_ACTION_COUNT).fill(0)
  };
}

export function encodeBiddingHistory(
  record: AutomatedGameRecord,
  playingDecision: DecisionRecord,
  relativePlayerIds: readonly PlayerId[]
): EncodedBiddingHistory {
  if (playingDecision.phase !== "playing") {
    throw new Error(
      `encodeBiddingHistory requires a playing decision, got ${playingDecision.phase}.`
    );
  }

  return encodeBiddingHistoryFromPublicActions(
    record.decisions
      .filter((decision) => decision.step < playingDecision.step)
      .flatMap(toPublicBiddingRecord),
    relativePlayerIds
  );
}

export function encodeBiddingHistoryFromPublicActions(
  publicActionHistory: readonly PublicActionRecord[],
  relativePlayerIds: readonly PlayerId[]
): EncodedBiddingHistory {
  const biddingDecisions = publicActionHistory;

  if (biddingDecisions.length > MAX_BIDDING_ACTION_COUNT) {
    throw new Error(
      `Bidding history cannot exceed ${MAX_BIDDING_ACTION_COUNT} actions, got ${biddingDecisions.length}.`
    );
  }

  const encoded = createEmptyEncodedBiddingHistory();
  const actionTypeIndices = [...encoded.actionTypeIndices];
  const playerIndices = [...encoded.playerIndices];
  const suitIndices = [...encoded.suitIndices];
  const targetPointCards = [...encoded.targetPointCards];
  const actionMask = [...encoded.actionMask];

  biddingDecisions.forEach((record, index) => {
    const action = record.action;

    if (action.playerId !== record.playerId) {
      throw new Error(
        `Bidding action playerId must match record/decision playerId: ${action.playerId} !== ${record.playerId}`
      );
    }

    actionTypeIndices[index] = encodeBiddingActionType(action);
    playerIndices[index] = getRelativePlayerIndex(relativePlayerIds, record.playerId);
    actionMask[index] = 1;

    if (action.type === "bid") {
      suitIndices[index] = getBiddingSuitIndex(action.suit);
      targetPointCards[index] = action.targetPointCards;
      return;
    }

    if (action.type === "pass") {
      suitIndices[index] = EMPTY_BIDDING_SUIT_INDEX;
      targetPointCards[index] = 0;
      return;
    }

    throw new Error(
      `Bidding history supports only pass and bid actions, got ${(action as GameAction).type}.`
    );
  });

  const result: EncodedBiddingHistory = {
    actionTypeIndices,
    playerIndices,
    suitIndices,
    targetPointCards,
    actionMask
  };

  validateEncodedBiddingHistory(result);

  return result;
}

export function getBiddingSuitIndex(suit: Suit): number {
  const index = BIDDING_HISTORY_SUIT_ORDER.indexOf(suit);

  if (index === -1) {
    throw new Error(`Unsupported bidding suit: ${suit}`);
  }

  return index;
}

export function validateEncodedBiddingHistory(history: EncodedBiddingHistory): void {
  expectLength(
    "biddingHistory.actionTypeIndices",
    history.actionTypeIndices,
    MAX_BIDDING_ACTION_COUNT
  );
  expectLength("biddingHistory.playerIndices", history.playerIndices, MAX_BIDDING_ACTION_COUNT);
  expectLength("biddingHistory.suitIndices", history.suitIndices, MAX_BIDDING_ACTION_COUNT);
  expectLength(
    "biddingHistory.targetPointCards",
    history.targetPointCards,
    MAX_BIDDING_ACTION_COUNT
  );
  expectLength("biddingHistory.actionMask", history.actionMask, MAX_BIDDING_ACTION_COUNT);
  validateMask("biddingHistory.actionMask", history.actionMask);
  validateContiguousMask("biddingHistory.actionMask", history.actionMask);

  for (let index = 0; index < MAX_BIDDING_ACTION_COUNT; index += 1) {
    const slotName = `biddingHistory[${index}]`;
    const mask = history.actionMask[index];
    const actionType = history.actionTypeIndices[index];
    const playerIndex = history.playerIndices[index];
    const suitIndex = history.suitIndices[index];
    const target = history.targetPointCards[index];

    if (mask === 0) {
      expectInteger(`${slotName}.actionType`, actionType);
      expectInteger(`${slotName}.playerIndex`, playerIndex);
      expectInteger(`${slotName}.suitIndex`, suitIndex);
      expectInteger(`${slotName}.targetPointCards`, target);

      if (
        actionType !== EMPTY_BIDDING_ACTION_TYPE ||
        playerIndex !== EMPTY_PLAYER_INDEX ||
        suitIndex !== EMPTY_BIDDING_SUIT_INDEX ||
        target !== 0
      ) {
        throw new Error(`${slotName} must contain only empty values when actionMask is 0.`);
      }

      continue;
    }

    expectIntegerInRange(`${slotName}.playerIndex`, playerIndex, 0, PLAYER_COUNT - 1);

    if (actionType === BIDDING_ACTION_TYPE_PASS) {
      if (suitIndex !== EMPTY_BIDDING_SUIT_INDEX || target !== 0) {
        throw new Error(`${slotName} pass must use suit -1 and target 0.`);
      }

      continue;
    }

    if (actionType === BIDDING_ACTION_TYPE_BID) {
      expectIntegerInRange(
        `${slotName}.suitIndex`,
        suitIndex,
        0,
        BIDDING_HISTORY_SUIT_ORDER.length - 1
      );
      expectIntegerInRange(
        `${slotName}.targetPointCards`,
        target,
        MIN_BIDDING_TARGET_POINT_CARDS,
        MAX_BIDDING_TARGET_POINT_CARDS
      );
      continue;
    }

    throw new Error(`${slotName}.actionType must be pass, bid, or empty.`);
  }
}

function encodeBiddingActionType(action: GameAction): number {
  switch (action.type) {
    case "pass":
      return BIDDING_ACTION_TYPE_PASS;
    case "bid":
      return BIDDING_ACTION_TYPE_BID;
    default:
      throw new Error(`Bidding history supports only pass and bid actions, got ${action.type}.`);
  }
}

function toPublicBiddingRecord(decision: DecisionRecord): readonly PublicActionRecord[] {
  if (decision.phase !== "bidding") {
    return [];
  }

  if (decision.action.type !== "bid" && decision.action.type !== "pass") {
    throw new Error(
      `Bidding history supports only pass and bid actions, got ${decision.action.type}.`
    );
  }

  return [{
    step: decision.step,
    playerId: decision.playerId,
    phase: decision.phase,
    action: decision.action
  }];
}

function expectLength(name: string, value: readonly unknown[], expectedLength: number): void {
  if (value.length !== expectedLength) {
    throw new Error(`${name} must have length ${expectedLength}, got ${value.length}.`);
  }
}

function validateMask(name: string, mask: readonly number[]): void {
  for (const value of mask) {
    expectInteger(name, value);

    if (value !== 0 && value !== 1) {
      throw new Error(`${name} must contain only 0/1 values.`);
    }
  }
}

function validateContiguousMask(name: string, mask: readonly number[]): void {
  let seenEmpty = false;

  for (const value of mask) {
    if (value === 0) {
      seenEmpty = true;
      continue;
    }

    if (seenEmpty) {
      throw new Error(`${name} must contain contiguous 1 values followed by 0 values.`);
    }
  }
}

function expectInteger(name: string, value: number): void {
  if (!Number.isFinite(value) || !Number.isInteger(value)) {
    throw new Error(`${name} must be a finite integer.`);
  }
}

function expectIntegerInRange(name: string, value: number, min: number, max: number): void {
  expectInteger(name, value);

  if (value < min || value > max) {
    throw new Error(`${name} must be between ${min} and ${max}, got ${value}.`);
  }
}
