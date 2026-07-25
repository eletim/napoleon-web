import type { Suit } from "@napoleon/game-core";

export const suitSymbols: Record<Suit, string> = {
  spades: "♠",
  hearts: "♥",
  diamonds: "♦",
  clubs: "♣"
};

export function isRedSuit(suit: Suit): boolean {
  return suit === "hearts" || suit === "diamonds";
}
