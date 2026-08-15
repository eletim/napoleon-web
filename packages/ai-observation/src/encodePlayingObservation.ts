import type { PlayerObservation } from "@napoleon/ai";
import type { PlayerId, PublicPlayerState, Suit } from "@napoleon/game-core";
import { getCardIndex } from "./cardIndex.js";
import { validateEncodedBiddingHistory } from "./encodeBiddingHistory.js";
import type { EncodedBiddingHistory } from "./encodeBiddingHistory.js";
import { createRelativePlayerOrder, getRelativePlayerIndex } from "./playerIndex.js";
import {
  CARD_COUNT,
  CARDS_PER_TRICK,
  EMPTY_CARD_INDEX,
  EMPTY_PLAYER_INDEX,
  MAX_BIDDING_TARGET_POINT_CARDS,
  MIN_CONTRACT_TARGET_POINT_CARDS,
  PLAYER_COUNT,
  PLAYING_ENCODER_SCHEMA_VERSION,
  SELF_ROLE_COUNT,
  SELF_ROLE_ORDER,
  TRICK_COUNT
} from "./schema.js";

const SUIT_ORDER: readonly Suit[] = ["spades", "hearts", "diamonds", "clubs"];
const REVEALED_ADJUTANT_CLASS_COUNT = PLAYER_COUNT + 1;
const COMPLETED_TRICK_CARD_SLOT_COUNT = TRICK_COUNT * CARDS_PER_TRICK;

export interface EncodedPlayingObservation {
  schemaVersion: typeof PLAYING_ENCODER_SCHEMA_VERSION;
  relativePlayerIds: readonly string[];
  trickNumber: number;
  completedTrickCount: number;
  contractTargetPointCards: number;
  trumpSuitOneHot: readonly number[];
  napoleonPlayerOneHot: readonly number[];
  revealedAdjutantPlayerOneHot: readonly number[];
  selfRoleOneHot: readonly number[];
  calledAdjutantCardMask: readonly number[];
  selfHandMask: readonly number[];
  legalPlayMask: readonly number[];
  handCountByPlayer: readonly number[];
  capturedPointCardMaskByPlayer: readonly (readonly number[])[];
  specialCardIndices: {
    oruma: number;
    yoromeki: number;
    seiJack: number;
    uraJack: number;
  };
  currentTrickCardIndices: readonly number[];
  currentTrickPlayerIndices: readonly number[];
  currentTrickSlotMask: readonly number[];
  completedTrickCardIndices: readonly number[];
  completedTrickPlayerIndices: readonly number[];
  completedTrickSlotMask: readonly number[];
  completedTrickWinnerIndices: readonly number[];
  completedTrickMask: readonly number[];
  biddingHistory: EncodedBiddingHistory;
  latestBuriedEventPointCardMask: readonly number[];
  latestBuriedEventHiddenNonPointCount: number;
  latestBuriedEventPresent: number;
}

export function encodePlayingObservation(
  observation: PlayerObservation,
  absolutePlayerIds: readonly PlayerId[],
  biddingHistory: EncodedBiddingHistory
): EncodedPlayingObservation {
  const view = observation.view;

  if (observation.playerId !== view.selfId) {
    throw new Error(
      `Observation playerId must match view.selfId: ${observation.playerId} !== ${view.selfId}`
    );
  }

  if (view.phase !== "playing") {
    throw new Error(`encodePlayingObservation requires a playing observation, got ${view.phase}.`);
  }

  if (view.trumpSuit === null || view.contract === null) {
    throw new Error("Playing observations must include a resolved trump suit and contract.");
  }

  if (view.adjutant === null) {
    throw new Error("Playing observations must include an adjutant view.");
  }

  if (view.playingSelfRole === null) {
    throw new Error("Playing observations must include a self playing role.");
  }

  const relativePlayerIds = createRelativePlayerOrder(absolutePlayerIds, observation.playerId);
  const playersById = createPlayersById(view.players);
  const self = playersById.get(observation.playerId);

  if (self?.hand === undefined) {
    throw new Error(`Self hand is missing from observation for ${observation.playerId}.`);
  }

  const encoded: EncodedPlayingObservation = {
    schemaVersion: PLAYING_ENCODER_SCHEMA_VERSION,
    relativePlayerIds,
    trickNumber: view.trickNumber,
    completedTrickCount: view.completedTrickCount,
    contractTargetPointCards: view.contract.targetPointCards,
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
    calledAdjutantCardMask: createMask([view.adjutant.calledCardId]),
    selfHandMask: createMask(self.hand.map((card) => card.id)),
    legalPlayMask: encodeLegalPlayMask(observation),
    handCountByPlayer: relativePlayerIds.map((playerId) => {
      const player = playersById.get(playerId);

      if (player === undefined) {
        throw new Error(`Observation is missing player state for ${playerId}.`);
      }

      return player.handCount;
    }),
    capturedPointCardMaskByPlayer: relativePlayerIds.map((playerId) => {
      const player = playersById.get(playerId);

      if (player === undefined) {
        throw new Error(`Observation is missing player state for ${playerId}.`);
      }

      return createMask(player.capturedPointCards.map((card) => card.id));
    }),
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
    ...encodeCompletedTricks(relativePlayerIds, view.completedTricks),
    biddingHistory,
    ...encodeLatestBuriedEvent(view.latestEvent)
  };

  validateEncodedPlayingObservation(encoded);

  return encoded;
}

export function validateEncodedPlayingObservation(
  observation: EncodedPlayingObservation
): void {
  if (observation.schemaVersion !== PLAYING_ENCODER_SCHEMA_VERSION) {
    throw new Error(`Unsupported playing encoder schema version: ${observation.schemaVersion}`);
  }

  expectLength("relativePlayerIds", observation.relativePlayerIds, PLAYER_COUNT);
  expectLength("trumpSuitOneHot", observation.trumpSuitOneHot, SUIT_ORDER.length);
  expectLength("napoleonPlayerOneHot", observation.napoleonPlayerOneHot, PLAYER_COUNT);
  expectLength(
    "revealedAdjutantPlayerOneHot",
    observation.revealedAdjutantPlayerOneHot,
    REVEALED_ADJUTANT_CLASS_COUNT
  );
  expectLength("selfRoleOneHot", observation.selfRoleOneHot, SELF_ROLE_COUNT);
  expectLength("calledAdjutantCardMask", observation.calledAdjutantCardMask, CARD_COUNT);
  expectLength("selfHandMask", observation.selfHandMask, CARD_COUNT);
  expectLength("legalPlayMask", observation.legalPlayMask, CARD_COUNT);
  expectLength("handCountByPlayer", observation.handCountByPlayer, PLAYER_COUNT);
  expectLength(
    "capturedPointCardMaskByPlayer",
    observation.capturedPointCardMaskByPlayer,
    PLAYER_COUNT
  );

  observation.capturedPointCardMaskByPlayer.forEach((mask, index) =>
    expectLength(`capturedPointCardMaskByPlayer[${index}]`, mask, CARD_COUNT)
  );
  expectLength("currentTrickCardIndices", observation.currentTrickCardIndices, CARDS_PER_TRICK);
  expectLength("currentTrickPlayerIndices", observation.currentTrickPlayerIndices, CARDS_PER_TRICK);
  expectLength("currentTrickSlotMask", observation.currentTrickSlotMask, CARDS_PER_TRICK);
  expectLength(
    "completedTrickCardIndices",
    observation.completedTrickCardIndices,
    COMPLETED_TRICK_CARD_SLOT_COUNT
  );
  expectLength(
    "completedTrickPlayerIndices",
    observation.completedTrickPlayerIndices,
    COMPLETED_TRICK_CARD_SLOT_COUNT
  );
  expectLength(
    "completedTrickSlotMask",
    observation.completedTrickSlotMask,
    COMPLETED_TRICK_CARD_SLOT_COUNT
  );
  expectLength("completedTrickWinnerIndices", observation.completedTrickWinnerIndices, TRICK_COUNT);
  expectLength("completedTrickMask", observation.completedTrickMask, TRICK_COUNT);
  validateEncodedBiddingHistory(observation.biddingHistory);
  expectLength(
    "latestBuriedEventPointCardMask",
    observation.latestBuriedEventPointCardMask,
    CARD_COUNT
  );

  validateRelativePlayerIds(observation.relativePlayerIds);
  expectIntegerInRange("trickNumber", observation.trickNumber, 1, TRICK_COUNT);
  expectIntegerInRange("completedTrickCount", observation.completedTrickCount, 0, TRICK_COUNT);
  expectIntegerInRange(
    "contractTargetPointCards",
    observation.contractTargetPointCards,
    MIN_CONTRACT_TARGET_POINT_CARDS,
    MAX_BIDDING_TARGET_POINT_CARDS
  );
  expectIntegerInRange(
    "latestBuriedEventHiddenNonPointCount",
    observation.latestBuriedEventHiddenNonPointCount,
    0,
    3
  );

  validateOneHot("trumpSuitOneHot", observation.trumpSuitOneHot);
  validateOneHot("napoleonPlayerOneHot", observation.napoleonPlayerOneHot);
  validateOneHot("revealedAdjutantPlayerOneHot", observation.revealedAdjutantPlayerOneHot);
  validateOneHot("selfRoleOneHot", observation.selfRoleOneHot);
  validateMask("calledAdjutantCardMask", observation.calledAdjutantCardMask);
  expectSum("calledAdjutantCardMask", observation.calledAdjutantCardMask, 1);
  validateMask("selfHandMask", observation.selfHandMask);
  validateMask("legalPlayMask", observation.legalPlayMask);
  observation.capturedPointCardMaskByPlayer.forEach((mask, index) =>
    validateMask(`capturedPointCardMaskByPlayer[${index}]`, mask)
  );
  validateMask("currentTrickSlotMask", observation.currentTrickSlotMask);
  validateMask("completedTrickSlotMask", observation.completedTrickSlotMask);
  validateMask("completedTrickMask", observation.completedTrickMask);
  validateMask("latestBuriedEventPointCardMask", observation.latestBuriedEventPointCardMask);
  validateMask("latestBuriedEventPresent", [observation.latestBuriedEventPresent]);

  observation.handCountByPlayer.forEach((value, index) =>
    expectIntegerInRange(`handCountByPlayer[${index}]`, value, 0, 13)
  );
  validateCardIndexArray("currentTrickCardIndices", observation.currentTrickCardIndices);
  validateCardIndexArray("completedTrickCardIndices", observation.completedTrickCardIndices);
  validatePlayerIndexArray("currentTrickPlayerIndices", observation.currentTrickPlayerIndices);
  validatePlayerIndexArray("completedTrickPlayerIndices", observation.completedTrickPlayerIndices);
  validatePlayerIndexArray("completedTrickWinnerIndices", observation.completedTrickWinnerIndices);
  validateSpecialCardIndices(observation.specialCardIndices);
  validateMaskIsSubset("legalPlayMask", observation.legalPlayMask, "selfHandMask", observation.selfHandMask);
  validateContiguousMask("currentTrickSlotMask", observation.currentTrickSlotMask);
  validateContiguousMask("completedTrickMask", observation.completedTrickMask);
  validateCurrentTrickSlots(observation);
  validateCompletedTrickSlots(observation);
  validateLatestBuriedEvent(observation);

  const completedTrickCount = sum(observation.completedTrickMask);

  if (observation.completedTrickCount !== completedTrickCount) {
    throw new Error(
      `completedTrickCount must equal completedTrickMask sum: ${observation.completedTrickCount} !== ${completedTrickCount}.`
    );
  }

  if (observation.trickNumber !== observation.completedTrickCount + 1) {
    throw new Error(
      `trickNumber must equal completedTrickCount + 1: ${observation.trickNumber} !== ${observation.completedTrickCount + 1}.`
    );
  }
}

function encodeSelfRoleOneHot(role: PlayerObservation["view"]["playingSelfRole"]): readonly number[] {
  if (role === null) {
    throw new Error("Playing self role is required.");
  }

  return SELF_ROLE_ORDER.map((candidate) => (candidate === role ? 1 : 0));
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

function createMask(cardIds: readonly string[]): readonly number[] {
  const mask = Array(CARD_COUNT).fill(0);

  for (const cardId of cardIds) {
    mask[getCardIndex(cardId)] = 1;
  }

  return mask;
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

function encodeCurrentTrick(
  relativePlayerIds: readonly PlayerId[],
  currentTrick: PlayerObservation["view"]["currentTrick"]
): Pick<
  EncodedPlayingObservation,
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

function encodeCompletedTricks(
  relativePlayerIds: readonly PlayerId[],
  completedTricks: PlayerObservation["view"]["completedTricks"]
): Pick<
  EncodedPlayingObservation,
  | "completedTrickCardIndices"
  | "completedTrickPlayerIndices"
  | "completedTrickSlotMask"
  | "completedTrickWinnerIndices"
  | "completedTrickMask"
> {
  if (completedTricks.length > TRICK_COUNT) {
    throw new Error(`Completed tricks cannot exceed ${TRICK_COUNT}.`);
  }

  const cardIndices = Array(COMPLETED_TRICK_CARD_SLOT_COUNT).fill(EMPTY_CARD_INDEX);
  const playerIndices = Array(COMPLETED_TRICK_CARD_SLOT_COUNT).fill(EMPTY_PLAYER_INDEX);
  const slotMask = Array(COMPLETED_TRICK_CARD_SLOT_COUNT).fill(0);
  const winnerIndices = Array(TRICK_COUNT).fill(EMPTY_PLAYER_INDEX);
  const trickMask = Array(TRICK_COUNT).fill(0);

  completedTricks.forEach((trick, trickIndex) => {
    if (trick.cards.length !== CARDS_PER_TRICK) {
      throw new Error(
        `Completed trick ${trick.trickNumber} must contain ${CARDS_PER_TRICK} cards, got ${trick.cards.length}.`
      );
    }

    trick.cards.forEach((playedCard, cardOffset) => {
      const slotIndex = trickIndex * CARDS_PER_TRICK + cardOffset;
      cardIndices[slotIndex] = getCardIndex(playedCard.card.id);
      playerIndices[slotIndex] = getRelativePlayerIndex(relativePlayerIds, playedCard.playerId);
      slotMask[slotIndex] = 1;
    });
    winnerIndices[trickIndex] = getRelativePlayerIndex(relativePlayerIds, trick.winnerId);
    trickMask[trickIndex] = 1;
  });

  return {
    completedTrickCardIndices: cardIndices,
    completedTrickPlayerIndices: playerIndices,
    completedTrickSlotMask: slotMask,
    completedTrickWinnerIndices: winnerIndices,
    completedTrickMask: trickMask
  };
}

function encodeLatestBuriedEvent(
  latestEvent: PlayerObservation["view"]["latestEvent"]
): Pick<
  EncodedPlayingObservation,
  | "latestBuriedEventPointCardMask"
  | "latestBuriedEventHiddenNonPointCount"
  | "latestBuriedEventPresent"
> {
  if (latestEvent?.type !== "buried-cards-resolved") {
    return {
      latestBuriedEventPointCardMask: Array(CARD_COUNT).fill(0),
      latestBuriedEventHiddenNonPointCount: 0,
      latestBuriedEventPresent: 0
    };
  }

  return {
    latestBuriedEventPointCardMask: createMask(
      latestEvent.awardedPointCards.map((card) => card.id)
    ),
    latestBuriedEventHiddenNonPointCount: latestEvent.hiddenNonPointCardCount,
    latestBuriedEventPresent: 1
  };
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
  const actualSum = sum(values);

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

function validateSpecialCardIndices(
  specialCardIndices: EncodedPlayingObservation["specialCardIndices"]
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

function validateMaskIsSubset(
  childName: string,
  child: readonly number[],
  parentName: string,
  parent: readonly number[]
): void {
  child.forEach((value, index) => {
    if (value === 1 && parent[index] !== 1) {
      throw new Error(`${childName}[${index}] must be within ${parentName}.`);
    }
  });
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

function validateCurrentTrickSlots(observation: EncodedPlayingObservation): void {
  for (let index = 0; index < CARDS_PER_TRICK; index += 1) {
    validateCardPlayerSlot(
      `currentTrick[${index}]`,
      observation.currentTrickSlotMask[index],
      observation.currentTrickCardIndices[index],
      observation.currentTrickPlayerIndices[index]
    );
  }
}

function validateCompletedTrickSlots(observation: EncodedPlayingObservation): void {
  for (let trickIndex = 0; trickIndex < TRICK_COUNT; trickIndex += 1) {
    const trickMask = observation.completedTrickMask[trickIndex];
    const winnerIndex = observation.completedTrickWinnerIndices[trickIndex];

    if (trickMask === 1) {
      expectIntegerInRange(`completedTrickWinnerIndices[${trickIndex}]`, winnerIndex, 0, PLAYER_COUNT - 1);
    } else if (winnerIndex !== EMPTY_PLAYER_INDEX) {
      throw new Error(
        `completedTrickWinnerIndices[${trickIndex}] must be ${EMPTY_PLAYER_INDEX} when completedTrickMask is 0.`
      );
    }

    for (let cardOffset = 0; cardOffset < CARDS_PER_TRICK; cardOffset += 1) {
      const slotIndex = trickIndex * CARDS_PER_TRICK + cardOffset;
      const slotMask = observation.completedTrickSlotMask[slotIndex];

      if (slotMask !== trickMask) {
        throw new Error(
          `completedTrickSlotMask[${slotIndex}] must match completedTrickMask[${trickIndex}].`
        );
      }

      validateCardPlayerSlot(
        `completedTrick[${slotIndex}]`,
        slotMask,
        observation.completedTrickCardIndices[slotIndex],
        observation.completedTrickPlayerIndices[slotIndex]
      );
    }
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

function validateLatestBuriedEvent(observation: EncodedPlayingObservation): void {
  const publicPointCardCount = sum(observation.latestBuriedEventPointCardMask);

  if (observation.latestBuriedEventPresent === 0) {
    if (publicPointCardCount !== 0 || observation.latestBuriedEventHiddenNonPointCount !== 0) {
      throw new Error("latestBuriedEvent fields must be empty when latestBuriedEventPresent is 0.");
    }

    return;
  }

  if (publicPointCardCount + observation.latestBuriedEventHiddenNonPointCount !== 3) {
    throw new Error("latestBuriedEvent public point cards plus hidden non-point count must equal 3.");
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

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}
