import { describe, expect, it } from "vitest";
import type { PublicCard, PublicGameAction, PublicGameState } from "../src/index.js";

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

  it("uses discard-cards cardIds without a playerId", () => {
    const action: PublicGameAction = {
      type: "discard-cards",
      cardIds: ["joker", "spades-A", "hearts-2"]
    };

    expect(action).toEqual({
      type: "discard-cards",
      cardIds: ["joker", "spades-A", "hearts-2"]
    });
    expect(Object.prototype.hasOwnProperty.call(action, "playerId")).toBe(false);
  });

  it("uses choose-adjutant cardId without a playerId", () => {
    const action: PublicGameAction = {
      type: "choose-adjutant",
      cardId: "spades-A"
    };

    expect(action).toEqual({
      type: "choose-adjutant",
      cardId: "spades-A"
    });
    expect(Object.prototype.hasOwnProperty.call(action, "playerId")).toBe(false);
  });

  it("exposes exchange and adjutant state without private fields", () => {
    const state: PublicGameState = {
      self: {
        id: "player-0",
        handCount: 13,
        hand: []
      },
      opponents: [],
      phase: "exchanging",
      trumpSuit: "spades",
      contract: {
        napoleonPlayerId: "player-0",
        trumpSuit: "spades",
        targetPointCards: 13
      },
      adjutant: {
        calledCardId: "hearts-A",
        revealedPlayerId: null
      },
      bidding: null,
      exchange: {
        napoleonPlayerId: "player-0",
        requiredDiscardCount: 3
      },
      adjutantChoice: {
        napoleonPlayerId: "player-0",
        standardCardsOnly: true
      },
      currentPlayerId: "player-0",
      currentTrick: [],
      completedTrickCount: 0,
      trickNumber: 1,
      isTrickComplete: false,
      isGameOver: false,
      legalActions: []
    };

    expect(state.exchange).toEqual({
      napoleonPlayerId: "player-0",
      requiredDiscardCount: 3
    });
    expect(Object.prototype.hasOwnProperty.call(state, "buriedCards")).toBe(false);
    expect(Object.prototype.hasOwnProperty.call(state.adjutant, "playerId")).toBe(false);
  });
});
