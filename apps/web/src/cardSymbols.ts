import type { PublicSuit } from "@napoleon/protocol";

export const suitSymbols: Record<PublicSuit, string> = {
  spades: "♠",
  hearts: "♥",
  diamonds: "♦",
  clubs: "♣"
};

export function isRedSuit(suit: PublicSuit): boolean {
  return suit === "hearts" || suit === "diamonds";
}
