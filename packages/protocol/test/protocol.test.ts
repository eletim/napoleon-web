import { describe, expect, it } from "vitest";
import type { PublicCard, PublicGameAction } from "../src/index.js";

describe("protocol DTOs", () => {
  it("represents public standard cards and joker cards as a discriminated union", () => {
    const standard: PublicCard = {
      type: "standard",
      id: "spades-A",
      suit: "spades",
      rank: "A"
    };
    const joker: PublicCard = {
      type: "joker",
      id: "joker"
    };

    expect(standard).toEqual({
      type: "standard",
      id: "spades-A",
      suit: "spades",
      rank: "A"
    });
    expect(joker).toEqual({
      type: "joker",
      id: "joker"
    });
    expect(Object.prototype.hasOwnProperty.call(joker, "suit")).toBe(false);
    expect(Object.prototype.hasOwnProperty.call(joker, "rank")).toBe(false);
  });

  it("uses play-card cardId for both standard cards and the joker", () => {
    const action: PublicGameAction = {
      type: "play-card",
      cardId: "joker"
    };

    expect(action).toEqual({
      type: "play-card",
      cardId: "joker"
    });
  });
});
