import { GameRuleError } from "./errors.js";
import { getRankValue } from "./ranks.js";
import type { Card, PlayedCard, PlayerId, Suit } from "./types.js";

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

export function determineTrickWinner(trick: readonly PlayedCard[]): PlayerId {
  const leadSuit = getLeadSuit(trick);

  if (leadSuit === undefined) {
    throw new GameRuleError("TRICK_NOT_COMPLETE", "Cannot determine a winner for an empty trick.");
  }

  const leadSuitCards = trick.filter((playedCard) => playedCard.card.suit === leadSuit);
  const winningCard = leadSuitCards.reduce((winner, candidate) =>
    getRankValue(candidate.card.rank) > getRankValue(winner.card.rank) ? candidate : winner
  );

  return winningCard.playerId;
}
