import type { Card, Rank, Suit } from "./types.js";

export const suits: readonly Suit[] = ["spades", "hearts", "diamonds", "clubs"];
export const ranks: readonly Rank[] = [
  "A",
  "2",
  "3",
  "4",
  "5",
  "6",
  "7",
  "8",
  "9",
  "10",
  "J",
  "Q",
  "K"
];

export function createDeck(): readonly Card[] {
  return suits.flatMap((suit) =>
    ranks.map((rank) => ({
      id: `${suit}-${rank}`,
      suit,
      rank
    }))
  );
}

export function shuffleDeck(
  cards: readonly Card[],
  rng: () => number = Math.random
): readonly Card[] {
  const shuffled = [...cards];

  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(rng() * (index + 1));
    [shuffled[index], shuffled[swapIndex]] = [shuffled[swapIndex], shuffled[index]];
  }

  return shuffled;
}
