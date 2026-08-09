import { describe, expect, it } from "vitest";
import type {
  CreateGameRequest,
  GetAgentsResponse,
  PublicCard,
  PublicGameAction,
  PublicGameEvent,
  PublicGameResult,
  PublicGameState
} from "../src/index.js";

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
      cardId: "joker"
    };

    expect(action).toEqual({
      type: "choose-adjutant",
      cardId: "joker"
    });
    expect(Object.prototype.hasOwnProperty.call(action, "playerId")).toBe(false);
  });

  it("describes available agents and selected game agents", () => {
    const agents: GetAgentsResponse = {
      agents: [
        {
          id: "rule-based",
          displayName: "Rule-based AI",
          isAvailable: true
        }
      ]
    };
    const request: CreateGameRequest = {
      aiAgents: [
        {
          playerId: "player-1",
          agentId: "rule-based"
        }
      ]
    };

    expect(agents.agents[0]).toEqual({
      id: "rule-based",
      displayName: "Rule-based AI",
      isAvailable: true
    });
    expect(request.aiAgents?.[0]).toEqual({
      playerId: "player-1",
      agentId: "rule-based"
    });
  });

  it("exposes exchange and adjutant state without private fields", () => {
    const state: PublicGameState = {
      self: {
        id: "player-0",
        handCount: 13,
        hand: [],
        capturedPointCards: []
      },
      opponents: [
        {
          id: "player-1",
          handCount: 10,
          capturedPointCards: []
        }
      ],
      phase: "exchanging",
      trumpSuit: "spades",
      contract: {
        napoleonPlayerId: "player-0",
        trumpSuit: "spades",
        targetPointCards: 13
      },
      specialCards: {
        orumaCardId: "spades-A",
        yoromekiCardId: "hearts-Q",
        seiJackCardId: "spades-J",
        uraJackCardId: "clubs-J"
      },
      adjutant: {
        calledCardId: "hearts-A",
        revealedPlayerId: null
      },
      latestEvent: null,
      result: null,
      bidding: null,
      exchange: {
        napoleonPlayerId: "player-0",
        requiredDiscardCount: 3
      },
      adjutantChoice: {
        napoleonPlayerId: "player-0",
        jokerAllowed: true
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
    expect(state.latestEvent).toBeNull();
    expect(state.self.capturedPointCards).toEqual([]);
    expect(state.opponents[0]?.capturedPointCards).toEqual([]);
    expect(Object.prototype.hasOwnProperty.call(state.adjutant, "playerId")).toBe(false);
    expect(state.adjutantChoice).toEqual({
      napoleonPlayerId: "player-0",
      jokerAllowed: true
    });
    expect(state.specialCards).toEqual({
      orumaCardId: "spades-A",
      yoromekiCardId: "hearts-Q",
      seiJackCardId: "spades-J",
      uraJackCardId: "clubs-J"
    });
  });

  it("exposes special card ids and allows undecided jack ids before trump is set", () => {
    const specialCards: PublicGameState["specialCards"] = {
      orumaCardId: "spades-A",
      yoromekiCardId: "hearts-Q",
      seiJackCardId: null,
      uraJackCardId: null
    };

    expect(specialCards).toEqual({
      orumaCardId: "spades-A",
      yoromekiCardId: "hearts-Q",
      seiJackCardId: null,
      uraJackCardId: null
    });
    expect(Object.prototype.hasOwnProperty.call(specialCards, "trumpSuit")).toBe(false);
    expect(Object.prototype.hasOwnProperty.call(specialCards, "orumaSuit")).toBe(false);
    expect(Object.prototype.hasOwnProperty.call(specialCards, "seiJackCardId")).toBe(true);
    expect(Object.prototype.hasOwnProperty.call(specialCards, "uraJackCardId")).toBe(true);
  });

  it("exposes buried resolution events and game result without internal-only fields", () => {
    const latestEvent: PublicGameEvent = {
      type: "buried-cards-resolved",
      napoleonPlayerId: "player-0",
      awardedPointCards: [
        {
          type: "standard",
          id: "spades-A",
          suit: "spades",
          rank: "A"
        }
      ],
      hiddenNonPointCardCount: 2
    };
    const result: PublicGameResult = {
      winner: "napoleon-team",
      napoleonTeamPointCards: 15,
      alliancePointCards: 5,
      targetPointCards: 15,
      napoleonPlayerId: "player-0",
      adjutantPlayerId: null
    };

    expect(latestEvent).toEqual({
      type: "buried-cards-resolved",
      napoleonPlayerId: "player-0",
      awardedPointCards: [
        {
          type: "standard",
          id: "spades-A",
          suit: "spades",
          rank: "A"
        }
      ],
      hiddenNonPointCardCount: 2
    });
    expect(Object.keys(latestEvent)).toEqual([
      "type",
      "napoleonPlayerId",
      "awardedPointCards",
      "hiddenNonPointCardCount"
    ]);
    expect(Object.prototype.hasOwnProperty.call(latestEvent, "hiddenCards")).toBe(false);
    expect(Object.prototype.hasOwnProperty.call(result, "buriedPointCards")).toBe(false);
    expect(Object.prototype.hasOwnProperty.call(result, "adjutant")).toBe(false);
    expect(Object.prototype.hasOwnProperty.call(result, "completedTricks")).toBe(false);
  });
});
