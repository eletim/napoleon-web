import type { GameAction } from "@napoleon/game-core";
import { getCardIndex } from "./cardIndex.js";
import { CARD_COUNT } from "./schema.js";

export interface EncodedPlayAction {
  selectedCardIndex: number;
}

export function encodePlayAction(
  action: GameAction,
  legalPlayMask: readonly number[]
): EncodedPlayAction {
  if (action.type !== "play-card") {
    throw new Error(`encodePlayAction requires a play-card action, got ${action.type}.`);
  }

  if (legalPlayMask.length !== CARD_COUNT) {
    throw new Error(`legalPlayMask must have length ${CARD_COUNT}, got ${legalPlayMask.length}.`);
  }

  for (const value of legalPlayMask) {
    if (value !== 0 && value !== 1) {
      throw new Error("legalPlayMask must contain only 0/1 values.");
    }
  }

  const selectedCardIndex = getCardIndex(action.cardId);

  if (legalPlayMask[selectedCardIndex] !== 1) {
    throw new Error(`Selected card is not legal for this observation: ${action.cardId}`);
  }

  return { selectedCardIndex };
}
