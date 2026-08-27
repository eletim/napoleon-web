import type { ComponentType, SVGProps } from "react";
import * as playingCardDeck from "@letele/playing-cards/dist/index.esm.js";

export type MockPlayingCardSuit = "spades" | "hearts" | "diamonds" | "clubs";
export type MockPlayingCardRank = "A" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "10" | "J" | "Q" | "K";

export interface MockPlayingCard {
  rank: MockPlayingCardRank;
  suit: MockPlayingCardSuit;
}

export type MockPlayingCardBackComponentName = "B1" | "B2";
export type PlayingCardSvgComponent = ComponentType<SVGProps<SVGSVGElement> & { title?: string }>;

const cardBackComponents = playingCardDeck as Record<MockPlayingCardBackComponentName, PlayingCardSvgComponent>;

export const mockCardBackComponentName = "B1" satisfies MockPlayingCardBackComponentName;

export function mockCardBackComponent(
  componentName: MockPlayingCardBackComponentName = mockCardBackComponentName
): PlayingCardSvgComponent {
  return cardBackComponents[componentName];
}
