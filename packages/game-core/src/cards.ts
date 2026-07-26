import type { Card, JokerCard, StandardCard } from "./types.js";
import { ranks, suits } from "./deck.js";

export function isStandardCard(card: Card): card is StandardCard {
  return card.type === "standard";
}

export function isJokerCard(card: Card): card is JokerCard {
  return card.type === "joker";
}

const standardCardIds = new Set<string>(
  suits.flatMap((suit) => ranks.map((rank) => `${suit}-${rank}`))
);

export function isStandardCardId(cardId: string): boolean {
  return standardCardIds.has(cardId);
}
