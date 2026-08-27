import type { MockPlayingCardSuit } from "./mockPlayingCardAdapter";

export const fourColorSuitColors = {
  spades: "#111827",
  hearts: "#dc2626",
  diamonds: "#2563eb",
  clubs: "#15803d"
} as const satisfies Record<MockPlayingCardSuit, string>;

const cardmeisterSuitColorOrder = ["spades", "hearts", "diamonds", "clubs"] as const satisfies readonly MockPlayingCardSuit[];

export const cardmeisterFourColorCsv = cardmeisterSuitColorOrder
  .map((suit) => fourColorSuitColors[suit])
  .join(",");
