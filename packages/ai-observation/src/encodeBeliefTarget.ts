import type { PlayerObservation } from "@napoleon/ai";
import type { ActualCardState } from "@napoleon/ai";
import type { PlayerId } from "@napoleon/game-core";
import { CARD_IDS, getCardIndex } from "./cardIndex.js";
import { createRelativePlayerOrder, getRelativePlayerIndex } from "./playerIndex.js";
import { CARD_COUNT, NOT_IN_HAND_CLASS_INDEX } from "./schema.js";

export interface EncodedBeliefTarget {
  ownerClassByCard: readonly number[];
  hiddenOwnershipLossMask: readonly number[];
}

export function encodeBeliefTarget(
  observation: PlayerObservation,
  actualState: ActualCardState,
  absolutePlayerIds: readonly PlayerId[]
): EncodedBeliefTarget {
  const relativePlayerIds = createRelativePlayerOrder(absolutePlayerIds, observation.playerId);
  validateCompleteActualCardState(actualState, absolutePlayerIds);

  const ownerClassByCard = CARD_IDS.map((cardId) => {
    for (const playerId of relativePlayerIds) {
      if (actualState.hands[playerId]?.includes(cardId) === true) {
        return getRelativePlayerIndex(relativePlayerIds, playerId);
      }
    }

    return NOT_IN_HAND_CLASS_INDEX;
  });
  const hiddenOwnershipLossMask = encodeHiddenOwnershipLossMask(observation);
  const target: EncodedBeliefTarget = {
    ownerClassByCard,
    hiddenOwnershipLossMask
  };

  validateEncodedBeliefTarget(target);

  return target;
}

export function validateEncodedBeliefTarget(target: EncodedBeliefTarget): void {
  if (target.ownerClassByCard.length !== CARD_COUNT) {
    throw new Error(
      `ownerClassByCard must have length ${CARD_COUNT}, got ${target.ownerClassByCard.length}.`
    );
  }

  if (target.hiddenOwnershipLossMask.length !== CARD_COUNT) {
    throw new Error(
      `hiddenOwnershipLossMask must have length ${CARD_COUNT}, got ${target.hiddenOwnershipLossMask.length}.`
    );
  }

  for (const ownerClass of target.ownerClassByCard) {
    if (
      !Number.isInteger(ownerClass) ||
      ownerClass < 0 ||
      ownerClass > NOT_IN_HAND_CLASS_INDEX
    ) {
      throw new Error(`ownerClassByCard contains an invalid owner class: ${ownerClass}.`);
    }
  }

  for (const value of target.hiddenOwnershipLossMask) {
    if (value !== 0 && value !== 1) {
      throw new Error("hiddenOwnershipLossMask must contain only 0/1 values.");
    }
  }
}

function validateCompleteActualCardState(
  actualState: ActualCardState,
  absolutePlayerIds: readonly PlayerId[]
): void {
  const allCardIds = [
    ...absolutePlayerIds.flatMap((playerId) => {
      const hand = actualState.hands[playerId];

      if (hand === undefined) {
        throw new Error(`Actual card state is missing hand for ${playerId}.`);
      }

      return hand;
    }),
    ...actualState.unusedCardIds,
    ...actualState.excludedCardIds,
    ...Object.values(actualState.awardedPointCardIds).flat(),
    ...actualState.currentTrickCardIds,
    ...actualState.completedTrickCardIds
  ];

  if (allCardIds.length !== CARD_COUNT) {
    throw new Error(`Actual card state must contain ${CARD_COUNT} card locations, got ${allCardIds.length}.`);
  }

  const seenCardIds = new Set(allCardIds);

  if (seenCardIds.size !== CARD_COUNT) {
    throw new Error("Actual card state must assign every card to exactly one location.");
  }

  for (const cardId of CARD_IDS) {
    if (!seenCardIds.has(cardId)) {
      throw new Error(`Actual card state is missing card: ${cardId}`);
    }
  }

  for (const cardId of allCardIds) {
    getCardIndex(cardId);
  }
}

function encodeHiddenOwnershipLossMask(observation: PlayerObservation): readonly number[] {
  const mask = Array(CARD_COUNT).fill(1);
  const knownCardIds: string[] = [];

  for (const player of observation.view.players) {
    if (player.id === observation.playerId) {
      knownCardIds.push(...(player.hand?.map((card) => card.id) ?? []));
    }

    knownCardIds.push(...player.capturedPointCards.map((card) => card.id));
  }

  knownCardIds.push(
    ...observation.view.currentTrick.map((playedCard) => playedCard.card.id),
    ...observation.view.completedTricks.flatMap((trick) =>
      trick.cards.map((playedCard) => playedCard.card.id)
    )
  );

  if (observation.view.latestEvent?.type === "buried-cards-resolved") {
    knownCardIds.push(...observation.view.latestEvent.awardedPointCards.map((card) => card.id));
  }

  for (const cardId of knownCardIds) {
    mask[getCardIndex(cardId)] = 0;
  }

  return mask;
}
