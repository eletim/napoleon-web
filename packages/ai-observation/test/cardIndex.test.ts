import { describe, expect, it } from "vitest";
import { createDeck, isJokerCard, isStandardCard } from "@napoleon/game-core";
import { CARD_COUNT, CARD_IDS, getCardId, getCardIndex } from "../src/index.js";

describe("card indexing", () => {
  it("defines a fixed 53-card order", () => {
    expect(CARD_IDS).toHaveLength(CARD_COUNT);
    expect(new Set(CARD_IDS).size).toBe(CARD_COUNT);
    expect(CARD_IDS.slice(0, 13)).toEqual([
      "spades-A",
      "spades-K",
      "spades-Q",
      "spades-J",
      "spades-10",
      "spades-9",
      "spades-8",
      "spades-7",
      "spades-6",
      "spades-5",
      "spades-4",
      "spades-3",
      "spades-2"
    ]);
    expect(CARD_IDS.at(-1)).toBe("joker");
  });

  it("matches the game deck card set without depending on deck order", () => {
    const deck = createDeck();

    expect(new Set(CARD_IDS)).toEqual(new Set(deck.map((card) => card.id)));
    expect(deck.filter(isJokerCard)).toHaveLength(1);
    expect(deck.filter(isStandardCard)).toHaveLength(52);
    expect(CARD_IDS.filter((cardId) => cardId === "joker")).toHaveLength(1);
    expect(CARD_IDS.filter((cardId) => cardId !== "joker")).toHaveLength(52);
  });

  it("round-trips card ids and indices", () => {
    for (const [index, cardId] of CARD_IDS.entries()) {
      expect(getCardIndex(cardId)).toBe(index);
      expect(getCardId(index)).toBe(cardId);
    }
  });

  it("rejects invalid card ids and indices", () => {
    expect(() => getCardIndex("spades-1")).toThrow("Unknown card id");
    expect(() => getCardId(-1)).toThrow("Card index must be an integer");
    expect(() => getCardId(53)).toThrow("Card index must be an integer");
    expect(() => getCardId(1.5)).toThrow("Card index must be an integer");
  });
});
