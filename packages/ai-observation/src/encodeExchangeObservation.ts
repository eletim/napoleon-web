import type { PlayerObservation } from "@napoleon/ai";
import type { PlayerId, PublicPlayerState, Suit } from "@napoleon/game-core";
import { getCardIndex } from "./cardIndex.js";
import { validateEncodedBiddingHistory } from "./encodeBiddingHistory.js";
import type { EncodedBiddingHistory } from "./encodeBiddingHistory.js";
import { validateLegalDiscardCardMask } from "./encodeExchangeAction.js";
import { createRelativePlayerOrder } from "./playerIndex.js";
import {
  CARD_COUNT,
  EXCHANGE_ENCODER_SCHEMA_VERSION,
  MAX_BIDDING_TARGET_POINT_CARDS,
  MIN_CONTRACT_TARGET_POINT_CARDS,
  PLAYER_COUNT
} from "./schema.js";

const SUIT_ORDER: readonly Suit[] = ["spades", "hearts", "diamonds", "clubs"];

export interface EncodedExchangeObservation {
  schemaVersion: typeof EXCHANGE_ENCODER_SCHEMA_VERSION;
  relativePlayerIds: readonly string[];
  contractTargetPointCards: number;
  trumpSuitOneHot: readonly number[];
  calledAdjutantCardMask: readonly number[];
  selfHandMask: readonly number[];
  legalDiscardCardMask: readonly number[];
  handCountByPlayer: readonly number[];
  specialCardIndices: {
    oruma: number;
    yoromeki: number;
    seiJack: number;
    uraJack: number;
  };
  biddingHistory: EncodedBiddingHistory;
}

export function encodeExchangeObservation(
  observation: PlayerObservation,
  absolutePlayerIds: readonly PlayerId[],
  biddingHistory: EncodedBiddingHistory
): EncodedExchangeObservation {
  const view = observation.view;

  if (observation.playerId !== view.selfId) {
    throw new Error(
      `Observation playerId must match view.selfId: ${observation.playerId} !== ${view.selfId}`
    );
  }

  if (view.phase !== "exchanging") {
    throw new Error(`encodeExchangeObservation requires an exchanging observation, got ${view.phase}.`);
  }

  if (view.trumpSuit === null || view.contract === null) {
    throw new Error("Exchange observations must include a resolved trump suit and contract.");
  }

  if (view.adjutant === null) {
    throw new Error("Exchange observations must include the called adjutant card.");
  }

  if (view.exchangeRequirement?.discardCount !== 3) {
    throw new Error(
      `Exchange observations require discardCount 3, got ${view.exchangeRequirement?.discardCount ?? "missing"}.`
    );
  }

  if (view.contract.napoleonPlayerId !== observation.playerId) {
    throw new Error(
      `Exchange observation player must be Napoleon: ${observation.playerId} !== ${view.contract.napoleonPlayerId}`
    );
  }

  if (view.currentPlayerId !== observation.playerId) {
    throw new Error(
      `Exchange observation player must be the current player: ${observation.playerId} !== ${view.currentPlayerId}`
    );
  }

  const playersById = createPlayersById(view.players);
  const self = playersById.get(observation.playerId);

  if (self?.hand === undefined) {
    throw new Error(`Self hand is missing from exchange observation for ${observation.playerId}.`);
  }

  if (self.hand.length !== 13) {
    throw new Error(`Napoleon must have 13 cards at exchange, got ${self.hand.length}.`);
  }

  const relativePlayerIds = createRelativePlayerOrder(absolutePlayerIds, observation.playerId);
  const selfHandMask = createMask(self.hand.map((card) => card.id));
  const encoded: EncodedExchangeObservation = {
    schemaVersion: EXCHANGE_ENCODER_SCHEMA_VERSION,
    relativePlayerIds,
    contractTargetPointCards: view.contract.targetPointCards,
    trumpSuitOneHot: encodeSuitOneHot(view.trumpSuit),
    calledAdjutantCardMask: createMask([view.adjutant.calledCardId]),
    selfHandMask,
    legalDiscardCardMask: selfHandMask,
    handCountByPlayer: relativePlayerIds.map((playerId) => {
      const player = playersById.get(playerId);

      if (player === undefined) {
        throw new Error(`Observation is missing player state for ${playerId}.`);
      }

      return player.handCount;
    }),
    specialCardIndices: {
      oruma: getCardIndex(view.specialCards.orumaCardId),
      yoromeki: getCardIndex(view.specialCards.yoromekiCardId),
      seiJack: getCardIndex(requiredSpecialCardId("seiJack", view.specialCards.seiJackCardId)),
      uraJack: getCardIndex(requiredSpecialCardId("uraJack", view.specialCards.uraJackCardId))
    },
    biddingHistory
  };

  validateEncodedExchangeObservation(encoded);

  if (encoded.relativePlayerIds[0] !== observation.playerId) {
    throw new Error(
      `Exchange observation relative player index 0 must be Napoleon: ${encoded.relativePlayerIds[0]} !== ${observation.playerId}`
    );
  }

  return encoded;
}

export function validateEncodedExchangeObservation(
  observation: EncodedExchangeObservation
): void {
  if (observation.schemaVersion !== EXCHANGE_ENCODER_SCHEMA_VERSION) {
    throw new Error(`Unsupported exchange encoder schema version: ${observation.schemaVersion}`);
  }

  expectLength("relativePlayerIds", observation.relativePlayerIds, PLAYER_COUNT);
  expectLength("trumpSuitOneHot", observation.trumpSuitOneHot, SUIT_ORDER.length);
  expectLength("calledAdjutantCardMask", observation.calledAdjutantCardMask, CARD_COUNT);
  expectLength("selfHandMask", observation.selfHandMask, CARD_COUNT);
  validateLegalDiscardCardMask(observation.legalDiscardCardMask);
  expectLength("handCountByPlayer", observation.handCountByPlayer, PLAYER_COUNT);
  validateEncodedBiddingHistory(observation.biddingHistory);
  validateRelativePlayerIds(observation.relativePlayerIds);
  expectIntegerInRange(
    "contractTargetPointCards",
    observation.contractTargetPointCards,
    MIN_CONTRACT_TARGET_POINT_CARDS,
    MAX_BIDDING_TARGET_POINT_CARDS
  );
  validateOneHot("trumpSuitOneHot", observation.trumpSuitOneHot);
  validateMask("calledAdjutantCardMask", observation.calledAdjutantCardMask);
  expectSum("calledAdjutantCardMask", observation.calledAdjutantCardMask, 1);
  validateMask("selfHandMask", observation.selfHandMask);
  expectSum("selfHandMask", observation.selfHandMask, 13);

  if (!sameMask(observation.legalDiscardCardMask, observation.selfHandMask)) {
    throw new Error("legalDiscardCardMask must equal selfHandMask at exchange.");
  }

  observation.handCountByPlayer.forEach((value, index) =>
    expectIntegerInRange(`handCountByPlayer[${index}]`, value, 0, 13)
  );

  if (observation.handCountByPlayer[0] !== 13) {
    throw new Error("handCountByPlayer[0] must be 13 for Napoleon at exchange.");
  }

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
    throw new Error(`Exchange observations must include ${name}.`);
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
  specialCardIndices: EncodedExchangeObservation["specialCardIndices"]
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

function sameMask(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}
