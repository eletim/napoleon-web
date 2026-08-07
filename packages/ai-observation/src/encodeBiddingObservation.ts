import type { PlayerObservation, PublicActionRecord } from "@napoleon/ai";
import type { PlayerId, PublicPlayerState } from "@napoleon/game-core";
import { getCardIndex } from "./cardIndex.js";
import { encodeLegalBidMask, validateLegalBidMask } from "./encodeBiddingAction.js";
import {
  encodeBiddingHistoryFromPublicActions,
  getBiddingSuitIndex,
  validateEncodedBiddingHistory
} from "./encodeBiddingHistory.js";
import type { EncodedBiddingHistory } from "./encodeBiddingHistory.js";
import { createRelativePlayerOrder, getRelativePlayerIndex } from "./playerIndex.js";
import {
  BIDDING_ENCODER_SCHEMA_VERSION,
  CARD_COUNT,
  EMPTY_BIDDING_SUIT_INDEX,
  EMPTY_PLAYER_INDEX,
  MAX_BIDDING_TARGET_POINT_CARDS,
  MIN_BIDDING_TARGET_POINT_CARDS,
  PLAYER_COUNT
} from "./schema.js";

export interface EncodedBiddingObservation {
  schemaVersion: typeof BIDDING_ENCODER_SCHEMA_VERSION;
  relativePlayerIds: readonly string[];
  selfHandMask: readonly number[];
  legalBidMask: readonly number[];
  starterPlayerIndex: number;
  highestBidPresent: number;
  highestBidPlayerIndex: number;
  highestBidSuitIndex: number;
  highestBidTargetPointCards: number;
  consecutivePassCount: number;
  biddingHistory: EncodedBiddingHistory;
}

export function encodeBiddingObservation(
  observation: PlayerObservation,
  absolutePlayerIds: readonly PlayerId[]
): EncodedBiddingObservation {
  const view = observation.view;

  if (observation.playerId !== view.selfId) {
    throw new Error(
      `Observation playerId must match view.selfId: ${observation.playerId} !== ${view.selfId}`
    );
  }

  if (view.phase !== "bidding") {
    throw new Error(`encodeBiddingObservation requires a bidding observation, got ${view.phase}.`);
  }

  if (view.bidding === null) {
    throw new Error("Bidding observations must include public bidding state.");
  }

  const relativePlayerIds = createRelativePlayerOrder(absolutePlayerIds, observation.playerId);
  const playersById = createPlayersById(view.players);
  const self = playersById.get(observation.playerId);

  if (self?.hand === undefined) {
    throw new Error(`Self hand is missing from observation for ${observation.playerId}.`);
  }

  const highestBid = view.bidding.highestBid;
  const encoded: EncodedBiddingObservation = {
    schemaVersion: BIDDING_ENCODER_SCHEMA_VERSION,
    relativePlayerIds,
    selfHandMask: createMask(self.hand.map((card) => card.id)),
    legalBidMask: encodeLegalBidMask(observation.legalActions, observation.playerId),
    starterPlayerIndex: getRelativePlayerIndex(relativePlayerIds, view.bidding.starterPlayerId),
    highestBidPresent: highestBid === null ? 0 : 1,
    highestBidPlayerIndex:
      highestBid === null
        ? EMPTY_PLAYER_INDEX
        : getRelativePlayerIndex(relativePlayerIds, highestBid.playerId),
    highestBidSuitIndex:
      highestBid === null ? EMPTY_BIDDING_SUIT_INDEX : getBiddingSuitIndex(highestBid.suit),
    highestBidTargetPointCards: highestBid === null ? 0 : highestBid.targetPointCards,
    consecutivePassCount: view.bidding.consecutivePassCount,
    biddingHistory: encodeBiddingHistoryFromPublicActions(
      getPublicBiddingHistory(observation),
      relativePlayerIds
    )
  };

  validateEncodedBiddingObservation(encoded);

  if (encoded.relativePlayerIds[0] !== observation.playerId) {
    throw new Error(
      `Bidding observation relative player index 0 must be the acting player: ${encoded.relativePlayerIds[0]} !== ${observation.playerId}`
    );
  }

  return encoded;
}

export function validateEncodedBiddingObservation(
  observation: EncodedBiddingObservation
): void {
  if (observation.schemaVersion !== BIDDING_ENCODER_SCHEMA_VERSION) {
    throw new Error(`Unsupported bidding encoder schema version: ${observation.schemaVersion}`);
  }

  expectLength("relativePlayerIds", observation.relativePlayerIds, PLAYER_COUNT);
  expectLength("selfHandMask", observation.selfHandMask, CARD_COUNT);
  validateLegalBidMask(observation.legalBidMask);
  validateEncodedBiddingHistory(observation.biddingHistory);
  validateRelativePlayerIds(observation.relativePlayerIds);
  validateMask("selfHandMask", observation.selfHandMask);
  validateMask("highestBidPresent", [observation.highestBidPresent]);
  expectIntegerInRange("starterPlayerIndex", observation.starterPlayerIndex, 0, PLAYER_COUNT - 1);
  expectIntegerInRange("consecutivePassCount", observation.consecutivePassCount, 0, PLAYER_COUNT);

  if (observation.highestBidPresent === 0) {
    if (
      observation.highestBidPlayerIndex !== EMPTY_PLAYER_INDEX ||
      observation.highestBidSuitIndex !== EMPTY_BIDDING_SUIT_INDEX ||
      observation.highestBidTargetPointCards !== 0
    ) {
      throw new Error("Empty highest bid fields must use player -1, suit -1, and target 0.");
    }

    return;
  }

  expectIntegerInRange(
    "highestBidPlayerIndex",
    observation.highestBidPlayerIndex,
    0,
    PLAYER_COUNT - 1
  );
  expectIntegerInRange("highestBidSuitIndex", observation.highestBidSuitIndex, 0, 3);
  expectIntegerInRange(
    "highestBidTargetPointCards",
    observation.highestBidTargetPointCards,
    MIN_BIDDING_TARGET_POINT_CARDS,
    MAX_BIDDING_TARGET_POINT_CARDS
  );
}

function getPublicBiddingHistory(
  observation: PlayerObservation
): readonly PublicActionRecord[] {
  if (observation.publicActionHistory !== undefined) {
    return observation.publicActionHistory;
  }

  const history = observation.view.bidding?.history ?? [];

  return history.map((action, index) => ({
    step: index + 1,
    playerId: action.playerId,
    phase: "bidding" as const,
    action
  }));
}

function createPlayersById(
  players: readonly PublicPlayerState[]
): ReadonlyMap<PlayerId, PublicPlayerState> {
  const playersById = new Map(players.map((player) => [player.id, player]));

  if (playersById.size !== players.length) {
    throw new Error("Observation player ids must be unique.");
  }

  return playersById;
}

function createMask(cardIds: readonly string[]): readonly number[] {
  const mask = Array(CARD_COUNT).fill(0);

  for (const cardId of cardIds) {
    mask[getCardIndex(cardId)] = 1;
  }

  return mask;
}

function expectLength(name: string, value: readonly unknown[], expectedLength: number): void {
  if (value.length !== expectedLength) {
    throw new Error(`${name} must have length ${expectedLength}, got ${value.length}.`);
  }
}

function validateMask(name: string, mask: readonly number[]): void {
  for (const value of mask) {
    if (!Number.isFinite(value) || !Number.isInteger(value) || (value !== 0 && value !== 1)) {
      throw new Error(`${name} must contain only 0/1 values.`);
    }
  }
}

function validateRelativePlayerIds(playerIds: readonly string[]): void {
  const uniquePlayerIds = new Set(playerIds);

  if (uniquePlayerIds.size !== playerIds.length) {
    throw new Error("relativePlayerIds must be unique.");
  }

  for (const [index, playerId] of playerIds.entries()) {
    if (typeof playerId !== "string" || playerId.length === 0) {
      throw new Error(`relativePlayerIds[${index}] must be a non-empty string.`);
    }
  }
}

function expectIntegerInRange(name: string, value: number, min: number, max: number): void {
  if (!Number.isFinite(value) || !Number.isInteger(value)) {
    throw new Error(`${name} must be a finite integer.`);
  }

  if (value < min || value > max) {
    throw new Error(`${name} must be between ${min} and ${max}, got ${value}.`);
  }
}
