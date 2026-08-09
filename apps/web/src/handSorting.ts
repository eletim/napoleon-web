import type { PublicCard, PublicRank, PublicSuit } from "@napoleon/protocol";

export type HandOrderMode = "original" | "riipai";

const suitOrder: Readonly<Record<PublicSuit, number>> = {
  spades: 0,
  hearts: 1,
  diamonds: 2,
  clubs: 3
};

const rankOrder: Readonly<Record<PublicRank, number>> = {
  A: 0,
  K: 1,
  Q: 2,
  J: 3,
  "10": 4,
  "9": 5,
  "8": 6,
  "7": 7,
  "6": 8,
  "5": 9,
  "4": 10,
  "3": 11,
  "2": 12
};

export function getDisplayedHandCards(
  hand: readonly PublicCard[],
  orderMode: HandOrderMode
): readonly PublicCard[] {
  if (orderMode === "original") {
    return hand;
  }

  return [...hand].sort(compareCardsForRiipai);
}

export function compareCardsForRiipai(left: PublicCard, right: PublicCard): number {
  if (left.type === "joker" && right.type === "joker") {
    return 0;
  }
  if (left.type === "joker") {
    return 1;
  }
  if (right.type === "joker") {
    return -1;
  }

  const suitComparison = suitOrder[left.suit] - suitOrder[right.suit];
  if (suitComparison !== 0) {
    return suitComparison;
  }

  const rankComparison = rankOrder[left.rank] - rankOrder[right.rank];
  if (rankComparison !== 0) {
    return rankComparison;
  }

  return left.id.localeCompare(right.id);
}
