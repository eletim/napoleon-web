import type { MockPlayingCardSuit } from "./mockPlayingCardAdapter";

export const cardDesignSuitSymbols = {
  spades: "♠",
  hearts: "♥",
  diamonds: "♦",
  clubs: "♣"
} as const satisfies Record<MockPlayingCardSuit, string>;

export const cardDesignSuitLabels = {
  spades: "Spades",
  hearts: "Hearts",
  diamonds: "Diamonds",
  clubs: "Clubs"
} as const satisfies Record<MockPlayingCardSuit, string>;

export const cardDesignSuitOrder = ["spades", "clubs", "hearts", "diamonds"] as const satisfies readonly MockPlayingCardSuit[];
