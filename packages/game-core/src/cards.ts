import type { Card, JokerCard, Rank, StandardCard, Suit } from "./types.js";
import { ranks, suits } from "./deck.js";

export const orumaCardId = "spades-A";
export const yoromekiCardId = "hearts-Q";

export function isStandardCard(card: Card): card is StandardCard {
  return card.type === "standard";
}

export function isJokerCard(card: Card): card is JokerCard {
  return card.type === "joker";
}

const pointRanks = new Set<Rank>(["10", "J", "Q", "K", "A"]);

export function isPointCard(card: Card): card is StandardCard {
  return isStandardCard(card) && pointRanks.has(card.rank);
}

export function isOrumaCard(card: Card): boolean {
  return card.id === orumaCardId;
}

export function isYoromekiCard(card: Card): boolean {
  return card.id === yoromekiCardId;
}

export function getSeiJackCardId(trumpSuit: Suit): string {
  return `${trumpSuit}-J`;
}

export function getUraJackCardId(trumpSuit: Suit): string {
  switch (trumpSuit) {
    case "spades":
      return "clubs-J";
    case "clubs":
      return "spades-J";
    case "hearts":
      return "diamonds-J";
    case "diamonds":
      return "hearts-J";
  }
}

export function isSeiJackCard(card: Card, trumpSuit: Suit): boolean {
  return card.id === getSeiJackCardId(trumpSuit);
}

export function isUraJackCard(card: Card, trumpSuit: Suit): boolean {
  return card.id === getUraJackCardId(trumpSuit);
}

const standardCardIds = new Set<string>(
  suits.flatMap((suit) => ranks.map((rank) => `${suit}-${rank}`))
);

export function isStandardCardId(cardId: string): boolean {
  return standardCardIds.has(cardId);
}
