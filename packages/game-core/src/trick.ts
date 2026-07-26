import { GameRuleError } from "./errors.js";
import { getRankValue } from "./ranks.js";
import type { Card, PlayedCard, PlayerId, Suit } from "./types.js";

export type TrickCardCategory = "trump" | "lead" | "other";

export interface TrickContext {
  trumpSuit: Suit | null;
}

export interface TrickCardStrength {
  category: TrickCardCategory;
  categoryPriority: number;
  rankPriority: number;
}

const categoryPriorities: Record<TrickCardCategory, number> = {
  other: 0,
  lead: 1,
  trump: 2
};

export function getLeadSuit(currentTrick: readonly PlayedCard[]): Suit | undefined {
  return currentTrick[0]?.card.suit;
}

export function getPlayableCards(
  hand: readonly Card[],
  currentTrick: readonly PlayedCard[]
): readonly Card[] {
  const leadSuit = getLeadSuit(currentTrick);

  if (leadSuit === undefined) {
    return hand;
  }

  const followCards = hand.filter((card) => card.suit === leadSuit);
  return followCards.length > 0 ? followCards : hand;
}

export function getTrickCardStrength(
  card: Card,
  leadSuit: Suit,
  context: TrickContext
): TrickCardStrength {
  const category = getTrickCardCategory(card, leadSuit, context);

  return {
    category,
    categoryPriority: categoryPriorities[category],
    rankPriority: getRankValue(card.rank)
  };
}

export function determineTrickWinner(
  trick: readonly PlayedCard[],
  context: TrickContext
): PlayerId {
  const leadSuit = getLeadSuit(trick);

  if (leadSuit === undefined) {
    throw new GameRuleError("TRICK_NOT_COMPLETE", "Cannot determine a winner for an empty trick.");
  }

  const winningCard = trick.reduce((winner, candidate) =>
    comparePlayedCards(candidate, winner, leadSuit, context) > 0 ? candidate : winner
  );

  return winningCard.playerId;
}

function getTrickCardCategory(
  card: Card,
  leadSuit: Suit,
  context: TrickContext
): TrickCardCategory {
  if (context.trumpSuit !== null && card.suit === context.trumpSuit) {
    return "trump";
  }

  if (card.suit === leadSuit) {
    return "lead";
  }

  return "other";
}

function comparePlayedCards(
  left: PlayedCard,
  right: PlayedCard,
  leadSuit: Suit,
  context: TrickContext
): number {
  const leftStrength = getTrickCardStrength(left.card, leadSuit, context);
  const rightStrength = getTrickCardStrength(right.card, leadSuit, context);

  if (leftStrength.categoryPriority !== rightStrength.categoryPriority) {
    return leftStrength.categoryPriority - rightStrength.categoryPriority;
  }

  return leftStrength.rankPriority - rightStrength.rankPriority;
}
