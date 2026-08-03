import type { GameAction, PlayerId } from "@napoleon/game-core";
import { getCardIndex } from "./cardIndex.js";
import { CARD_COUNT } from "./schema.js";

export interface EncodedPlayAction {
  selectedCardIndex: number;
}

export function encodePlayAction(
  action: GameAction,
  legalPlayMask: readonly number[],
  expectedPlayerId?: PlayerId
): EncodedPlayAction {
  if (action.type !== "play-card") {
    throw new Error(`encodePlayAction requires a play-card action, got ${action.type}.`);
  }

  if (expectedPlayerId !== undefined && action.playerId !== expectedPlayerId) {
    throw new Error(
      `Decision action playerId must match decision playerId: ${action.playerId} !== ${expectedPlayerId}`
    );
  }

  if (legalPlayMask.length !== CARD_COUNT) {
    throw new Error(`legalPlayMask must have length ${CARD_COUNT}, got ${legalPlayMask.length}.`);
  }

  for (const value of legalPlayMask) {
    if (!Number.isFinite(value) || !Number.isInteger(value) || (value !== 0 && value !== 1)) {
      throw new Error("legalPlayMask must contain only 0/1 values.");
    }
  }

  const selectedCardIndex = getCardIndex(action.cardId);

  if (legalPlayMask[selectedCardIndex] !== 1) {
    throw new Error(`Selected card is not legal for this observation: ${action.cardId}`);
  }

  return { selectedCardIndex };
}
