import { GameRuleError } from "./errors.js";
import { getRankValue } from "./ranks.js";
import { isJokerCard, isOrumaCard, isStandardCard, isYoromekiCard } from "./cards.js";
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
const jokerRankPriority = 1;

export function getLeadSuit(
  currentTrick: readonly PlayedCard[],
  context: TrickContext
): Suit | undefined {
  const leadCard = currentTrick[0]?.card;

  if (leadCard === undefined) {
    return undefined;
  }

  if (isStandardCard(leadCard)) {
    return leadCard.suit;
  }

  if (context.trumpSuit === null) {
    throw new GameRuleError(
      "TRUMP_NOT_SET",
      "Trump suit must be set when joker leads a trick."
    );
  }

  return context.trumpSuit;
}

export function getPlayableCards(
  hand: readonly Card[],
  currentTrick: readonly PlayedCard[],
  context: TrickContext
): readonly Card[] {
  const leadSuit = getLeadSuit(currentTrick, context);

  if (leadSuit === undefined) {
    return hand;
  }

  const hasFollowCards = hand.some((card) => isStandardCard(card) && card.suit === leadSuit);
  return hasFollowCards
    ? hand.filter((card) => isJokerCard(card) || (isStandardCard(card) && card.suit === leadSuit))
    : hand;
}

export function getTrickCardStrength(
  card: Card,
  leadSuit: Suit,
  context: TrickContext,
  options: { isLeadCard?: boolean } = {}
): TrickCardStrength {
  const category = getTrickCardCategory(card, leadSuit, context, options);

  return {
    category,
    categoryPriority: categoryPriorities[category],
    rankPriority: isJokerCard(card) ? jokerRankPriority : getRankValue(card.rank)
  };
}

export function determineTrickWinner(
  trick: readonly PlayedCard[],
  context: TrickContext
): PlayerId {
  const specialWinner = determineOrumaYoromekiWinner(trick);

  if (specialWinner !== undefined) {
    return specialWinner;
  }

  const leadSuit = getLeadSuit(trick, context);

  if (leadSuit === undefined) {
    throw new GameRuleError("TRICK_NOT_COMPLETE", "Cannot determine a winner for an empty trick.");
  }

  const winningCard = trick.reduce((winner, candidate, candidateIndex) =>
    comparePlayedCards(
      candidate,
      winner,
      leadSuit,
      context,
      candidateIndex,
      trick.indexOf(winner)
    ) > 0
      ? candidate
      : winner
  );

  return winningCard.playerId;
}

function determineOrumaYoromekiWinner(trick: readonly PlayedCard[]): PlayerId | undefined {
  const orumaPlay = trick.find((playedCard) => isOrumaCard(playedCard.card));
  const yoromekiPlay = trick.find((playedCard) => isYoromekiCard(playedCard.card));

  if (orumaPlay !== undefined && yoromekiPlay !== undefined) {
    return yoromekiPlay.playerId;
  }

  return orumaPlay?.playerId;
}

function getTrickCardCategory(
  card: Card,
  leadSuit: Suit,
  context: TrickContext,
  options: { isLeadCard?: boolean }
): TrickCardCategory {
  if (isJokerCard(card)) {
    return options.isLeadCard === true ? "trump" : "lead";
  }

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
  context: TrickContext,
  leftIndex: number,
  rightIndex: number
): number {
  const leftStrength = getTrickCardStrength(left.card, leadSuit, context, {
    isLeadCard: leftIndex === 0
  });
  const rightStrength = getTrickCardStrength(right.card, leadSuit, context, {
    isLeadCard: rightIndex === 0
  });

  if (leftStrength.categoryPriority !== rightStrength.categoryPriority) {
    return leftStrength.categoryPriority - rightStrength.categoryPriority;
  }

  return leftStrength.rankPriority - rightStrength.rankPriority;
}
