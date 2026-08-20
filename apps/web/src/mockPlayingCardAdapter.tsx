import type { ComponentType, SVGProps } from "react";
import * as playingCardDeck from "@letele/playing-cards/dist/index.esm.js";

export type MockPlayingCardSuit = "spades" | "hearts" | "diamonds" | "clubs";
export type MockPlayingCardRank = "A" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "10" | "J" | "Q" | "K";

export interface MockPlayingCard {
  rank: MockPlayingCardRank;
  suit: MockPlayingCardSuit;
}

type PlayingCardSuitPrefix = "S" | "H" | "D" | "C";
type PlayingCardRankSuffix = "a" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "10" | "j" | "q" | "k";

export type MockPlayingCardFrontComponentName = `${PlayingCardSuitPrefix}${PlayingCardRankSuffix}`;
export type MockPlayingCardBackComponentName = "B1" | "B2";
export type MockPlayingCardComponentName = MockPlayingCardFrontComponentName | MockPlayingCardBackComponentName;
export type PlayingCardSvgComponent = ComponentType<SVGProps<SVGSVGElement> & { title?: string }>;

const suitComponentPrefixes = {
  spades: "S",
  hearts: "H",
  diamonds: "D",
  clubs: "C"
} as const satisfies Record<MockPlayingCardSuit, PlayingCardSuitPrefix>;

const rankComponentSuffixes = {
  A: "a",
  "2": "2",
  "3": "3",
  "4": "4",
  "5": "5",
  "6": "6",
  "7": "7",
  "8": "8",
  "9": "9",
  "10": "10",
  J: "j",
  Q: "q",
  K: "k"
} as const satisfies Record<MockPlayingCardRank, PlayingCardRankSuffix>;

const cardComponents = playingCardDeck as Record<MockPlayingCardComponentName, PlayingCardSvgComponent>;

export const mockCardBackComponentName = "B1" satisfies MockPlayingCardBackComponentName;

export function mockPlayingCardComponentName(card: MockPlayingCard): MockPlayingCardFrontComponentName {
  return `${suitComponentPrefixes[card.suit]}${rankComponentSuffixes[card.rank]}`;
}

export function mockPlayingCardComponent(card: MockPlayingCard): PlayingCardSvgComponent {
  return cardComponents[mockPlayingCardComponentName(card)];
}

export function mockCardBackComponent(
  componentName: MockPlayingCardBackComponentName = mockCardBackComponentName
): PlayingCardSvgComponent {
  return cardComponents[componentName];
}
