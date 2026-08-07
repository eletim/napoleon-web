import type { PlayerObservation } from "@napoleon/ai";
import type { PlayerId, PublicPlayerState, Suit } from "@napoleon/game-core";
import { getCardIndex } from "./cardIndex.js";
import { encodeLegalAdjutantMask, validateLegalAdjutantMask } from "./encodeAdjutantAction.js";
import { validateEncodedBiddingHistory } from "./encodeBiddingHistory.js";
import type { EncodedBiddingHistory } from "./encodeBiddingHistory.js";
import { createRelativePlayerOrder } from "./playerIndex.js";
import {
  ADJUTANT_ENCODER_SCHEMA_VERSION,
  CARD_COUNT,
  MAX_BIDDING_TARGET_POINT_CARDS,
  PLAYER_COUNT
} from "./schema.js";

const SUIT_ORDER: readonly Suit[] = ["spades", "hearts", "diamonds", "clubs"];
// All-pass bidding resolves to a forced spades-12 contract before adjutant choice.
const MIN_ADJUTANT_CONTRACT_TARGET_POINT_CARDS = 12;

export interface EncodedAdjutantObservation {
  schemaVersion: typeof ADJUTANT_ENCODER_SCHEMA_VERSION;
  relativePlayerIds: readonly string[];
  trumpSuitOneHot: readonly number[];
  contractTargetPointCards: number;
  selfHandMask: readonly number[];
  legalAdjutantMask: readonly number[];
  specialCardIndices: {
    oruma: number;
    yoromeki: number;
    seiJack: number;
    uraJack: number;
  };
  biddingHistory: EncodedBiddingHistory;
}

export function encodeAdjutantObservation(
  observation: PlayerObservation,
  absolutePlayerIds: readonly PlayerId[],
  biddingHistory: EncodedBiddingHistory
): EncodedAdjutantObservation {
  const view = observation.view;

  if (observation.playerId !== view.selfId) {
    throw new Error(
      `Observation playerId must match view.selfId: ${observation.playerId} !== ${view.selfId}`
    );
  }

  if (view.phase !== "choosing-adjutant") {
    throw new Error(
      `encodeAdjutantObservation requires a choosing-adjutant observation, got ${view.phase}.`
    );
  }

  if (view.trumpSuit === null || view.contract === null) {
    throw new Error("Adjutant observations must include a resolved trump suit and contract.");
  }

  if (view.adjutant !== null) {
    throw new Error("Adjutant observations must be encoded before the adjutant card is chosen.");
  }

  if (view.adjutantChoiceRequirement === null) {
    throw new Error("Adjutant observations require an adjutant choice requirement.");
  }

  if (view.contract.napoleonPlayerId !== observation.playerId) {
    throw new Error(
      `Adjutant observation player must be Napoleon: ${observation.playerId} !== ${view.contract.napoleonPlayerId}`
    );
  }

  if (view.currentPlayerId !== observation.playerId) {
    throw new Error(
      `Adjutant observation player must be the current player: ${observation.playerId} !== ${view.currentPlayerId}`
    );
  }

  if (
    view.currentTrick.length !== 0 ||
    view.completedTricks.length !== 0 ||
    view.completedTrickCount !== 0 ||
    view.trickNumber !== 1
  ) {
    throw new Error("Adjutant observations must be before exchange, discard, and play.");
  }

  const playersById = createPlayersById(view.players);
  const self = playersById.get(observation.playerId);

  if (self?.hand === undefined) {
    throw new Error(`Self hand is missing from adjutant observation for ${observation.playerId}.`);
  }

  if (self.hand.length !== 10) {
    throw new Error(`Napoleon must have 10 cards before adjutant choice, got ${self.hand.length}.`);
  }

  const relativePlayerIds = createRelativePlayerOrder(absolutePlayerIds, observation.playerId);
  const encoded: EncodedAdjutantObservation = {
    schemaVersion: ADJUTANT_ENCODER_SCHEMA_VERSION,
    relativePlayerIds,
    trumpSuitOneHot: encodeSuitOneHot(view.trumpSuit),
    contractTargetPointCards: view.contract.targetPointCards,
    selfHandMask: createMask(self.hand.map((card) => card.id)),
    legalAdjutantMask: encodeLegalAdjutantMask(observation.legalActions, observation.playerId),
    specialCardIndices: {
      oruma: getCardIndex(view.specialCards.orumaCardId),
      yoromeki: getCardIndex(view.specialCards.yoromekiCardId),
      seiJack: getCardIndex(requiredSpecialCardId("seiJack", view.specialCards.seiJackCardId)),
      uraJack: getCardIndex(requiredSpecialCardId("uraJack", view.specialCards.uraJackCardId))
    },
    biddingHistory
  };

  validateEncodedAdjutantObservation(encoded);

  if (encoded.relativePlayerIds[0] !== observation.playerId) {
    throw new Error(
      `Adjutant observation relative player index 0 must be Napoleon: ${encoded.relativePlayerIds[0]} !== ${observation.playerId}`
    );
  }

  return encoded;
}

export function validateEncodedAdjutantObservation(
  observation: EncodedAdjutantObservation
): void {
  if (observation.schemaVersion !== ADJUTANT_ENCODER_SCHEMA_VERSION) {
    throw new Error(`Unsupported adjutant encoder schema version: ${observation.schemaVersion}`);
  }

  expectLength("relativePlayerIds", observation.relativePlayerIds, PLAYER_COUNT);
  expectLength("trumpSuitOneHot", observation.trumpSuitOneHot, SUIT_ORDER.length);
  expectLength("selfHandMask", observation.selfHandMask, CARD_COUNT);
  validateLegalAdjutantMask(observation.legalAdjutantMask);
  validateEncodedBiddingHistory(observation.biddingHistory);
  validateRelativePlayerIds(observation.relativePlayerIds);
  validateOneHot("trumpSuitOneHot", observation.trumpSuitOneHot);
  validateMask("selfHandMask", observation.selfHandMask);
  expectSum("selfHandMask", observation.selfHandMask, 10);
  expectIntegerInRange(
    "contractTargetPointCards",
    observation.contractTargetPointCards,
    MIN_ADJUTANT_CONTRACT_TARGET_POINT_CARDS,
    MAX_BIDDING_TARGET_POINT_CARDS
  );
  validateSpecialCardIndices(observation.specialCardIndices);
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

function encodeSuitOneHot(trumpSuit: Suit): readonly number[] {
  return SUIT_ORDER.map((suit) => (suit === trumpSuit ? 1 : 0));
}

function createMask(cardIds: readonly string[]): readonly number[] {
  const mask = Array(CARD_COUNT).fill(0);

  for (const cardId of cardIds) {
    mask[getCardIndex(cardId)] = 1;
  }

  return mask;
}

function requiredSpecialCardId(name: string, cardId: string | null): string {
  if (cardId === null) {
    throw new Error(`Adjutant observations must include ${name}.`);
  }

  return cardId;
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

function validateOneHot(name: string, values: readonly number[]): void {
  validateMask(name, values);
  expectSum(name, values, 1);
}

function expectSum(name: string, values: readonly number[], expectedSum: number): void {
  const actualSum = values.reduce((total, value) => total + value, 0);

  if (actualSum !== expectedSum) {
    throw new Error(`${name} must sum to ${expectedSum}, got ${actualSum}.`);
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

function validateSpecialCardIndices(
  specialCardIndices: EncodedAdjutantObservation["specialCardIndices"]
): void {
  expectIntegerInRange("specialCardIndices.oruma", specialCardIndices.oruma, 0, CARD_COUNT - 1);
  expectIntegerInRange(
    "specialCardIndices.yoromeki",
    specialCardIndices.yoromeki,
    0,
    CARD_COUNT - 1
  );
  expectIntegerInRange("specialCardIndices.seiJack", specialCardIndices.seiJack, 0, CARD_COUNT - 1);
  expectIntegerInRange("specialCardIndices.uraJack", specialCardIndices.uraJack, 0, CARD_COUNT - 1);
}
