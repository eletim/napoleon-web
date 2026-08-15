import type { ActualCardState, PlayerObservation } from "@napoleon/ai";
import type { PlayerId, PublicPlayerState, Suit } from "@napoleon/game-core";
import { CARD_IDS, getCardIndex } from "./cardIndex.js";
import { encodeBeliefTarget } from "./encodeBeliefTarget.js";
import { createRelativePlayerOrder, getRelativePlayerIndex } from "./playerIndex.js";
import {
  CARD_COUNT,
  CARDS_PER_TRICK,
  COMPLETE_INFO_PLAYING_ENCODER_SCHEMA_VERSION,
  EMPTY_CARD_INDEX,
  EMPTY_PLAYER_INDEX,
  MAX_BIDDING_TARGET_POINT_CARDS,
  MIN_CONTRACT_TARGET_POINT_CARDS,
  NOT_IN_HAND_CLASS_INDEX,
  PLAYER_COUNT,
  PLAYING_ENCODER_SCHEMA_VERSION,
  SELF_ROLE_COUNT,
  SELF_ROLE_ORDER,
  TRICK_COUNT
} from "./schema.js";

const SUIT_ORDER: readonly Suit[] = ["spades", "hearts", "diamonds", "clubs"];
const REVEALED_ADJUTANT_CLASS_COUNT = PLAYER_COUNT + 1;

export interface EncodedCompleteInfoPlayingObservation {
  schemaVersion: typeof COMPLETE_INFO_PLAYING_ENCODER_SCHEMA_VERSION;
  sourcePublicPlayingEncoderSchemaVersion: typeof PLAYING_ENCODER_SCHEMA_VERSION;
  relativePlayerIds: readonly string[];
  trickNumber: number;
  completedTrickCount: number;
  contractTargetPointCards: number;
  cardOwnerClassByCard: readonly number[];
  capturedPointCardCountByPlayer: readonly number[];
  trumpSuitOneHot: readonly number[];
  napoleonPlayerOneHot: readonly number[];
  revealedAdjutantPlayerOneHot: readonly number[];
  selfRoleOneHot: readonly number[];
  calledAdjutantCardIndex: number;
  specialCardIndices: {
    oruma: number;
    yoromeki: number;
    seiJack: number;
    uraJack: number;
  };
  currentTrickCardIndices: readonly number[];
  currentTrickPlayerIndices: readonly number[];
  currentTrickSlotMask: readonly number[];
  legalPlayMask: readonly number[];
}

export function encodeCompleteInfoPlayingObservation(
  observation: PlayerObservation,
  actualState: ActualCardState,
  absolutePlayerIds: readonly PlayerId[]
): EncodedCompleteInfoPlayingObservation {
  const view = observation.view;

  if (observation.playerId !== view.selfId) {
    throw new Error(
      `Observation playerId must match view.selfId: ${observation.playerId} !== ${view.selfId}`
    );
  }

  if (view.phase !== "playing") {
    throw new Error(
      `encodeCompleteInfoPlayingObservation requires a playing observation, got ${view.phase}.`
    );
  }

  if (view.trumpSuit === null || view.contract === null) {
    throw new Error("Complete-info playing observations must include a resolved trump suit and contract.");
  }

  if (view.adjutant === null) {
    throw new Error("Complete-info playing observations must include an adjutant view.");
  }

  if (view.playingSelfRole === null) {
    throw new Error("Complete-info playing observations must include a self playing role.");
  }

  const relativePlayerIds = createRelativePlayerOrder(absolutePlayerIds, observation.playerId);
  const playersById = createPlayersById(view.players);
  const ownerTarget = encodeBeliefTarget(observation, actualState, absolutePlayerIds);
  const encoded: EncodedCompleteInfoPlayingObservation = {
    schemaVersion: COMPLETE_INFO_PLAYING_ENCODER_SCHEMA_VERSION,
    sourcePublicPlayingEncoderSchemaVersion: PLAYING_ENCODER_SCHEMA_VERSION,
    relativePlayerIds,
    trickNumber: view.trickNumber,
    completedTrickCount: view.completedTrickCount,
    contractTargetPointCards: view.contract.targetPointCards,
    cardOwnerClassByCard: ownerTarget.ownerClassByCard,
    capturedPointCardCountByPlayer: relativePlayerIds.map(
      (playerId) => actualState.awardedPointCardIds[playerId]?.length ?? 0
    ),
    trumpSuitOneHot: encodeSuitOneHot(view.trumpSuit),
    napoleonPlayerOneHot: encodeRelativePlayerOneHot(
      relativePlayerIds,
      view.contract.napoleonPlayerId
    ),
    revealedAdjutantPlayerOneHot: encodeRevealedAdjutantPlayerOneHot(
      relativePlayerIds,
      view.adjutant.revealedPlayerId
    ),
    selfRoleOneHot: encodeSelfRoleOneHot(view.playingSelfRole),
    calledAdjutantCardIndex: getCardIndex(view.adjutant.calledCardId),
    specialCardIndices: {
      oruma: getCardIndex(view.specialCards.orumaCardId),
      yoromeki: getCardIndex(view.specialCards.yoromekiCardId),
      seiJack:
        view.specialCards.seiJackCardId === null
          ? EMPTY_CARD_INDEX
          : getCardIndex(view.specialCards.seiJackCardId),
      uraJack:
        view.specialCards.uraJackCardId === null
          ? EMPTY_CARD_INDEX
          : getCardIndex(view.specialCards.uraJackCardId)
    },
    ...encodeCurrentTrick(relativePlayerIds, view.currentTrick),
    legalPlayMask: encodeLegalPlayMask(observation)
  };

  for (const playerId of relativePlayerIds) {
    if (!playersById.has(playerId)) {
      throw new Error(`Observation is missing player state for ${playerId}.`);
    }
  }

  validateEncodedCompleteInfoPlayingObservation(encoded);

  return encoded;
}

export function validateEncodedCompleteInfoPlayingObservation(
  observation: EncodedCompleteInfoPlayingObservation
): void {
  if (observation.schemaVersion !== COMPLETE_INFO_PLAYING_ENCODER_SCHEMA_VERSION) {
    throw new Error(
      `Unsupported complete-info playing encoder schema version: ${observation.schemaVersion}`
    );
  }

  if (observation.sourcePublicPlayingEncoderSchemaVersion !== PLAYING_ENCODER_SCHEMA_VERSION) {
    throw new Error(
      `Unsupported source public playing encoder schema version: ${observation.sourcePublicPlayingEncoderSchemaVersion}`
    );
  }

  expectLength("relativePlayerIds", observation.relativePlayerIds, PLAYER_COUNT);
  expectLength("cardOwnerClassByCard", observation.cardOwnerClassByCard, CARD_COUNT);
  expectLength(
    "capturedPointCardCountByPlayer",
    observation.capturedPointCardCountByPlayer,
    PLAYER_COUNT
  );
  expectLength("trumpSuitOneHot", observation.trumpSuitOneHot, SUIT_ORDER.length);
  expectLength("napoleonPlayerOneHot", observation.napoleonPlayerOneHot, PLAYER_COUNT);
  expectLength(
    "revealedAdjutantPlayerOneHot",
    observation.revealedAdjutantPlayerOneHot,
    REVEALED_ADJUTANT_CLASS_COUNT
  );
  expectLength("selfRoleOneHot", observation.selfRoleOneHot, SELF_ROLE_COUNT);
  expectLength("currentTrickCardIndices", observation.currentTrickCardIndices, CARDS_PER_TRICK);
  expectLength("currentTrickPlayerIndices", observation.currentTrickPlayerIndices, CARDS_PER_TRICK);
  expectLength("currentTrickSlotMask", observation.currentTrickSlotMask, CARDS_PER_TRICK);
  expectLength("legalPlayMask", observation.legalPlayMask, CARD_COUNT);

  validateRelativePlayerIds(observation.relativePlayerIds);
  expectIntegerInRange("trickNumber", observation.trickNumber, 1, TRICK_COUNT);
  expectIntegerInRange("completedTrickCount", observation.completedTrickCount, 0, TRICK_COUNT);
  expectIntegerInRange(
    "contractTargetPointCards",
    observation.contractTargetPointCards,
    MIN_CONTRACT_TARGET_POINT_CARDS,
    MAX_BIDDING_TARGET_POINT_CARDS
  );
  validateOwnerClasses(observation.cardOwnerClassByCard);
  observation.capturedPointCardCountByPlayer.forEach((value, index) =>
    expectIntegerInRange(`capturedPointCardCountByPlayer[${index}]`, value, 0, CARD_COUNT)
  );
  validateOneHot("trumpSuitOneHot", observation.trumpSuitOneHot);
  validateOneHot("napoleonPlayerOneHot", observation.napoleonPlayerOneHot);
  validateOneHot("revealedAdjutantPlayerOneHot", observation.revealedAdjutantPlayerOneHot);
  validateOneHot("selfRoleOneHot", observation.selfRoleOneHot);
  expectIntegerInRange("calledAdjutantCardIndex", observation.calledAdjutantCardIndex, 0, CARD_COUNT - 1);
  validateSpecialCardIndices(observation.specialCardIndices);
  validateCardIndexArray("currentTrickCardIndices", observation.currentTrickCardIndices);
  validatePlayerIndexArray("currentTrickPlayerIndices", observation.currentTrickPlayerIndices);
  validateMask("currentTrickSlotMask", observation.currentTrickSlotMask);
  validateContiguousMask("currentTrickSlotMask", observation.currentTrickSlotMask);
  validateMask("legalPlayMask", observation.legalPlayMask);
  validateCurrentTrickSlots(observation);
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

function encodeRelativePlayerOneHot(
  relativePlayerIds: readonly PlayerId[],
  playerId: PlayerId
): readonly number[] {
  const playerIndex = getRelativePlayerIndex(relativePlayerIds, playerId);

  return relativePlayerIds.map((_, index) => (index === playerIndex ? 1 : 0));
}

function encodeRevealedAdjutantPlayerOneHot(
  relativePlayerIds: readonly PlayerId[],
  revealedPlayerId: PlayerId | null
): readonly number[] {
  const oneHot = Array(REVEALED_ADJUTANT_CLASS_COUNT).fill(0);

  if (revealedPlayerId === null) {
    oneHot[PLAYER_COUNT] = 1;
    return oneHot;
  }

  oneHot[getRelativePlayerIndex(relativePlayerIds, revealedPlayerId)] = 1;
  return oneHot;
}

function encodeSelfRoleOneHot(role: PlayerObservation["view"]["playingSelfRole"]): readonly number[] {
  if (role === null) {
    throw new Error("Complete-info playing self role is required.");
  }

  return SELF_ROLE_ORDER.map((candidate) => (candidate === role ? 1 : 0));
}

function encodeCurrentTrick(
  relativePlayerIds: readonly PlayerId[],
  currentTrick: PlayerObservation["view"]["currentTrick"]
): Pick<
  EncodedCompleteInfoPlayingObservation,
  "currentTrickCardIndices" | "currentTrickPlayerIndices" | "currentTrickSlotMask"
> {
  if (currentTrick.length > CARDS_PER_TRICK) {
    throw new Error(`Current trick cannot contain more than ${CARDS_PER_TRICK} cards.`);
  }

  const cardIndices = Array(CARDS_PER_TRICK).fill(EMPTY_CARD_INDEX);
  const playerIndices = Array(CARDS_PER_TRICK).fill(EMPTY_PLAYER_INDEX);
  const slotMask = Array(CARDS_PER_TRICK).fill(0);

  currentTrick.forEach((playedCard, index) => {
    cardIndices[index] = getCardIndex(playedCard.card.id);
    playerIndices[index] = getRelativePlayerIndex(relativePlayerIds, playedCard.playerId);
    slotMask[index] = 1;
  });

  return {
    currentTrickCardIndices: cardIndices,
    currentTrickPlayerIndices: playerIndices,
    currentTrickSlotMask: slotMask
  };
}

function encodeLegalPlayMask(observation: PlayerObservation): readonly number[] {
  const legalCardIds: string[] = [];

  for (const action of observation.legalActions) {
    if (action.type !== "play-card") {
      throw new Error(`Playing legalActions must contain only play-card actions, got ${action.type}.`);
    }

    if (action.playerId !== observation.playerId) {
      throw new Error(
        `Legal play action playerId must match observation playerId: ${action.playerId} !== ${observation.playerId}`
      );
    }

    legalCardIds.push(action.cardId);
  }

  return createMask(legalCardIds);
}

function createMask(cardIds: readonly string[]): readonly number[] {
  const mask = Array(CARD_COUNT).fill(0);

  for (const cardId of cardIds) {
    mask[getCardIndex(cardId)] = 1;
  }

  return mask;
}

function validateOwnerClasses(ownerClassByCard: readonly number[]): void {
  const classCounts = Array(NOT_IN_HAND_CLASS_INDEX + 1).fill(0);

  for (const [cardIndex, ownerClass] of ownerClassByCard.entries()) {
    getCardIndex(CARD_IDS[cardIndex]);
    expectIntegerInRange(
      `cardOwnerClassByCard[${cardIndex}]`,
      ownerClass,
      0,
      NOT_IN_HAND_CLASS_INDEX
    );
    classCounts[ownerClass] += 1;
  }

  if (classCounts.reduce((sum, count) => sum + count, 0) !== CARD_COUNT) {
    throw new Error("cardOwnerClassByCard must assign every card exactly once.");
  }
}

function validateSpecialCardIndices(
  specialCardIndices: EncodedCompleteInfoPlayingObservation["specialCardIndices"]
): void {
  expectIntegerInRange("specialCardIndices.oruma", specialCardIndices.oruma, 0, CARD_COUNT - 1);
  expectIntegerInRange(
    "specialCardIndices.yoromeki",
    specialCardIndices.yoromeki,
    0,
    CARD_COUNT - 1
  );
  validateOptionalCardIndex("specialCardIndices.seiJack", specialCardIndices.seiJack);
  validateOptionalCardIndex("specialCardIndices.uraJack", specialCardIndices.uraJack);
}

function validateCardIndexArray(name: string, values: readonly number[]): void {
  values.forEach((value, index) => validateOptionalCardIndex(`${name}[${index}]`, value));
}

function validateOptionalCardIndex(name: string, value: number): void {
  if (value === EMPTY_CARD_INDEX) {
    return;
  }

  expectIntegerInRange(name, value, 0, CARD_COUNT - 1);
}

function validatePlayerIndexArray(name: string, values: readonly number[]): void {
  values.forEach((value, index) => validateOptionalPlayerIndex(`${name}[${index}]`, value));
}

function validateOptionalPlayerIndex(name: string, value: number): void {
  if (value === EMPTY_PLAYER_INDEX) {
    return;
  }

  expectIntegerInRange(name, value, 0, PLAYER_COUNT - 1);
}

function validateCurrentTrickSlots(observation: EncodedCompleteInfoPlayingObservation): void {
  for (let index = 0; index < CARDS_PER_TRICK; index += 1) {
    validateCardPlayerSlot(
      `currentTrick[${index}]`,
      observation.currentTrickSlotMask[index],
      observation.currentTrickCardIndices[index],
      observation.currentTrickPlayerIndices[index]
    );
  }
}

function validateCardPlayerSlot(
  name: string,
  mask: number,
  cardIndex: number,
  playerIndex: number
): void {
  if (mask === 1) {
    expectIntegerInRange(`${name}.cardIndex`, cardIndex, 0, CARD_COUNT - 1);
    expectIntegerInRange(`${name}.playerIndex`, playerIndex, 0, PLAYER_COUNT - 1);
    return;
  }

  if (cardIndex !== EMPTY_CARD_INDEX || playerIndex !== EMPTY_PLAYER_INDEX) {
    throw new Error(`${name} must use empty card and player indices when slot mask is 0.`);
  }
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
