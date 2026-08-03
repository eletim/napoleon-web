import { CARD_COUNT } from "./schema.js";

const SUIT_ORDER = ["spades", "hearts", "diamonds", "clubs"] as const;
const RANK_ORDER = ["A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"] as const;

export const CARD_IDS: readonly string[] = [
  ...SUIT_ORDER.flatMap((suit) => RANK_ORDER.map((rank) => `${suit}-${rank}`)),
  "joker"
];

const cardIndexById = new Map(CARD_IDS.map((cardId, index) => [cardId, index]));

if (CARD_IDS.length !== CARD_COUNT) {
  throw new Error(`CARD_IDS must contain ${CARD_COUNT} cards, got ${CARD_IDS.length}.`);
}

export function getCardIndex(cardId: string): number {
  const index = cardIndexById.get(cardId);

  if (index === undefined) {
    throw new Error(`Unknown card id: ${cardId}`);
  }

  return index;
}

export function getCardId(cardIndex: number): string {
  if (!Number.isInteger(cardIndex) || cardIndex < 0 || cardIndex >= CARD_IDS.length) {
    throw new Error(`Card index must be an integer between 0 and ${CARD_IDS.length - 1}: ${cardIndex}`);
  }

  return CARD_IDS[cardIndex];
}
