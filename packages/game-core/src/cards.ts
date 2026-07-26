import type { Card, JokerCard, StandardCard } from "./types.js";

export function isStandardCard(card: Card): card is StandardCard {
  return card.type === "standard";
}

export function isJokerCard(card: Card): card is JokerCard {
  return card.type === "joker";
}
