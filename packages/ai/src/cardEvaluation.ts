import {
  getSeiJackCardId,
  getUraJackCardId,
  isJokerCard,
  isStandardCard,
  orumaCardId,
  yoromekiCardId
} from "@napoleon/game-core";
import type { Card, Rank, SpecialCardsView, Suit } from "@napoleon/game-core";

export const aiRankValues: Record<Rank, number> = {
  A: 20,
  K: 13,
  Q: 12,
  J: 11,
  "10": 10,
  "9": 9,
  "8": 8,
  "7": 7,
  "6": 6,
  "5": 5,
  "4": 4,
  "3": 3,
  "2": 12
};

export function getAiRankValue(rank: Rank): number {
  return aiRankValues[rank];
}

export function createSpecialCardsForTrump(trumpSuit: Suit): SpecialCardsView {
  return {
    orumaCardId,
    yoromekiCardId,
    seiJackCardId: getSeiJackCardId(trumpSuit),
    uraJackCardId: getUraJackCardId(trumpSuit)
  };
}

export function evaluateCardForTrump(
  card: Card,
  trumpSuit: Suit,
  specialCards: SpecialCardsView
): number {
  if (isJokerCard(card)) {
    return 20;
  }

  if (card.id === specialCards.orumaCardId) {
    return 60;
  }

  if (card.id === specialCards.seiJackCardId) {
    return 55;
  }

  if (card.id === specialCards.uraJackCardId) {
    return 50;
  }

  if (card.id === specialCards.yoromekiCardId) {
    return card.suit === trumpSuit ? 45 : 30;
  }

  return getAiRankValue(card.rank) + (card.suit === trumpSuit ? 30 : 0);
}

export function countStandardCardsOfSuit(hand: readonly Card[], suit: Suit): number {
  return hand.filter((card) => isStandardCard(card) && card.suit === suit).length;
}
