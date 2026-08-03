import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { FastifyInstance } from "fastify";
import { RuleBasedAgent, type Agent } from "@napoleon/ai";
import {
  advanceToNextTrick,
  applyAction,
  createDeck,
  createInitialGame,
  getLegalActions,
  isJokerCard,
  isPointCard,
  isStandardCard,
  type Bid,
  type Card,
  type GameAction,
  type GameState,
  type Rank,
  type Suit
} from "@napoleon/game-core";
import type {
  ApiError,
  CreateGameResponse,
  GetGameResponse,
  NextTrickResponse,
  RunAutomatedSimulationResponse,
  SendActionResponse
} from "@napoleon/protocol";
import { buildApp } from "../src/app.js";
import { createAgents, games } from "../src/store.js";

let app: FastifyInstance;

class ThrowingAgent implements Agent {
  async selectAction(): Promise<GameAction> {
    throw new Error("AI failure for test");
  }
}

class InspectingAgent implements Agent {
  observedPhase: GameState["phase"] | undefined;
  observedTrumpSuit: Suit | null | undefined;
  observedHighestBid: Bid | null | undefined;
  observedHistoryLength: number | undefined;
  observedContract: GameState["contract"] | undefined;
  observedSpecialCards: Parameters<Agent["selectAction"]>[0]["view"]["specialCards"] | undefined;
  observedLegalActions: readonly GameAction[] | undefined;
  observedOpponentHandLeak = false;

  async selectAction(input: Parameters<Agent["selectAction"]>[0]): Promise<GameAction> {
    this.observedPhase = input.view.phase;
    this.observedTrumpSuit = input.view.trumpSuit;
    this.observedHighestBid = input.view.bidding?.highestBid;
    this.observedHistoryLength = input.view.bidding?.history.length;
    this.observedContract = input.view.contract;
    this.observedSpecialCards = input.view.specialCards;
    this.observedLegalActions = input.legalActions;
    this.observedOpponentHandLeak = input.view.players
      .filter((player) => player.id !== input.playerId)
      .some((player) => hasOwn(player, "hand"));
    const exchangeAction = createExchangeAction(input);

    if (exchangeAction !== undefined) {
      return exchangeAction;
    }

    const adjutantAction = createAdjutantAction(input);

    if (adjutantAction !== undefined) {
      return adjutantAction;
    }

    const action = input.legalActions[0];

    if (action === undefined) {
      throw new Error("Expected legal action for inspecting agent.");
    }

    return action;
  }
}

class PassAgent implements Agent {
  async selectAction(input: Parameters<Agent["selectAction"]>[0]): Promise<GameAction> {
    const exchangeAction = createExchangeAction(input);

    if (exchangeAction !== undefined) {
      return exchangeAction;
    }

    const adjutantAction = createAdjutantAction(input);

    if (adjutantAction !== undefined) {
      return adjutantAction;
    }

    const passAction = input.legalActions.find((action) => action.type === "pass");
    const fallback = input.legalActions[0];

    if (passAction !== undefined) {
      return passAction;
    }

    if (fallback === undefined) {
      throw new Error("Expected a legal action.");
    }

    return fallback;
  }
}

class PreferBidAgent implements Agent {
  constructor(
    private readonly suit: Suit,
    private readonly targetPointCards: number
  ) {}

  async selectAction(input: Parameters<Agent["selectAction"]>[0]): Promise<GameAction> {
    const exchangeAction = createExchangeAction(input);

    if (exchangeAction !== undefined) {
      return exchangeAction;
    }

    const adjutantAction = createAdjutantAction(input);

    if (adjutantAction !== undefined) {
      return adjutantAction;
    }

    const preferred = input.legalActions.find(
      (action) =>
        action.type === "bid" &&
        action.suit === this.suit &&
        action.targetPointCards === this.targetPointCards
    );

    if (preferred !== undefined) {
      return preferred;
    }

    const passAction = input.legalActions.find((action) => action.type === "pass");
    const fallback = input.legalActions[0];

    if (passAction !== undefined) {
      return passAction;
    }

    if (fallback === undefined) {
      throw new Error("Expected a legal action.");
    }

    return fallback;
  }
}

class PreferredCardAgent implements Agent {
  constructor(private readonly cardId: string) {}

  async selectAction(input: Parameters<Agent["selectAction"]>[0]): Promise<GameAction> {
    const exchangeAction = createExchangeAction(input);

    if (exchangeAction !== undefined) {
      return exchangeAction;
    }

    const adjutantAction = createAdjutantAction(input);

    if (adjutantAction !== undefined) {
      return adjutantAction;
    }

    const preferred = input.legalActions.find(
      (action) => action.type === "play-card" && action.cardId === this.cardId
    );
    const fallback = input.legalActions[0];

    if (preferred !== undefined) {
      return preferred;
    }

    if (fallback === undefined) {
      throw new Error("Expected a legal action.");
    }

    return fallback;
  }
}

function createExchangeAction(input: Parameters<Agent["selectAction"]>[0]): GameAction | undefined {
  if (input.view.phase !== "exchanging" || input.view.exchangeRequirement === null) {
    return undefined;
  }

  const self = input.view.players.find((player) => player.id === input.playerId);

  if (self?.hand === undefined) {
    return undefined;
  }

  return {
    type: "discard-cards",
    playerId: input.playerId,
    cardIds: self.hand.slice(0, input.view.exchangeRequirement.discardCount).map((card) => card.id)
  };
}

function createAdjutantAction(input: Parameters<Agent["selectAction"]>[0]): GameAction | undefined {
  if (input.view.phase !== "choosing-adjutant" || input.view.adjutantChoiceRequirement === null) {
    return undefined;
  }

  return {
    type: "choose-adjutant",
    playerId: input.playerId,
    cardId: "spades-A"
  };
}

beforeEach(async () => {
  games.clear();
  app = await buildApp();
});

afterEach(async () => {
  games.clear();
  await app.close();
});

describe("server API", () => {
  it("uses RuleBasedAgent as the standard AI agent", () => {
    const agents = createAgents(["player-1"]);

    expect(agents.get("player-1")).toBeInstanceOf(RuleBasedAgent);
  });

  it("returns health status", async () => {
    const response = await app.inject({
      method: "GET",
      url: "/api/health"
    });

    expect(response.statusCode).toBe(200);
    expect(response.json<{ ok: boolean }>()).toEqual({ ok: true });
  });

  it("creates a game with public self and opponent DTOs", async () => {
    const response = await app.inject({
      method: "POST",
      url: "/api/games",
      payload: {}
    });
    const body = response.json<CreateGameResponse>();

    expect(response.statusCode).toBe(201);
    expect(typeof body.gameId).toBe("string");
    expect(body.playerId).toBe("player-0");
    expect(body.state.self.hand).toHaveLength(10);
    expect(body.state.opponents).toHaveLength(4);
    expect(body.state.self.capturedPointCards).toEqual([]);
    expect(body.state.opponents.every((opponent) => opponent.capturedPointCards.length === 0)).toBe(
      true
    );
    expect(body.state.phase).toBe("bidding");
    expect(body.state.trumpSuit).toBeNull();
    expect(body.state.contract).toBeNull();
    expect(body.state.specialCards).toEqual({
      orumaCardId: "spades-A",
      yoromekiCardId: "hearts-Q",
      seiJackCardId: null,
      uraJackCardId: null
    });
    expect(body.state.adjutant).toBeNull();
    expect(body.state.adjutantChoice).toBeNull();
    expect(body.state.bidding).toMatchObject({
      starterPlayerId: "player-0",
      highestBid: null,
      consecutivePassCount: 0
    });
    expect(body.state.opponents.some((opponent) => hasOwn(opponent, "hand"))).toBe(false);
    expect(body.state.legalActions.some((action) => action.type === "pass")).toBe(true);
    expect(body.state.legalActions.some((action) => action.type === "bid")).toBe(true);
    expect(body.state.legalActions.some((action) => hasOwn(action, "playerId"))).toBe(false);
    expect(hasOwn(body.state, "unusedCards")).toBe(false);

    const record = games.get(body.gameId);
    expect(record).toBeDefined();
    if (record !== undefined) {
      const allCards = [
        ...record.state.players.flatMap((player) => player.hand),
        ...record.state.unusedCards
      ];
      expect(allCards).toHaveLength(53);
      expect(record.state.unusedCards).toHaveLength(3);
      expect(allCards.filter(isStandardCard)).toHaveLength(52);
      expect(allCards.filter(isJokerCard)).toHaveLength(1);
    }
  });

  it("runs a seeded automated simulation to completion", async () => {
    const response = await app.inject({
      method: "POST",
      url: "/api/simulations",
      payload: {
        seed: 12345
      }
    });
    const body = response.json<RunAutomatedSimulationResponse>();

    expect(response.statusCode).toBe(200);
    expect(body.schemaVersion).toBe(1);
    expect(body.seed).toBe(12345);
    expect(body.playerIds).toEqual(["player-0", "player-1", "player-2", "player-3", "player-4"]);
    expect(body.decisions.length).toBeGreaterThan(0);
    expect(body.summary.totalDecisionCount).toBe(body.decisions.length);
    expect(body.result.winner).toMatch(/^(napoleon-team|alliance)$/);
    expect(body.decisions[0]).toMatchObject({
      step: 1,
      playerId: "player-0",
      phase: "bidding",
      trickNumber: 1
    });
    expect(body.decisions[0].legalActionCount).toBeGreaterThan(0);
    expect(body.decisions[0].action).not.toHaveProperty("playerId");
    expect(
      body.decisions[0].observation.players.find((player) => player.id === "player-0")?.hand
    ).toBeDefined();
    expect(
      body.decisions[0].observation.players.find((player) => player.id === "player-1")
    ).not.toHaveProperty("hand");
  });

  it("returns the same automated simulation response for the same seed", async () => {
    const first = await app.inject({
      method: "POST",
      url: "/api/simulations",
      payload: {
        seed: 2468
      }
    });
    const second = await app.inject({
      method: "POST",
      url: "/api/simulations",
      payload: {
        seed: 2468
      }
    });

    expect(first.statusCode).toBe(200);
    expect(second.statusCode).toBe(200);
    expect(second.json<RunAutomatedSimulationResponse>()).toEqual(
      first.json<RunAutomatedSimulationResponse>()
    );
  });

  it("returns different automated simulation initial hands for different seeds", async () => {
    const first = await app.inject({
      method: "POST",
      url: "/api/simulations",
      payload: {
        seed: 1
      }
    });
    const second = await app.inject({
      method: "POST",
      url: "/api/simulations",
      payload: {
        seed: 2
      }
    });

    expect(first.statusCode).toBe(200);
    expect(second.statusCode).toBe(200);
    expect(second.json<RunAutomatedSimulationResponse>().initialHands).not.toEqual(
      first.json<RunAutomatedSimulationResponse>().initialHands
    );
  });

  it("rejects invalid automated simulation seeds", async () => {
    const invalidBodies = [
      {},
      { seed: "123" },
      { seed: 1.5 },
      { seed: Number.POSITIVE_INFINITY }
    ];

    for (const payload of invalidBodies) {
      const response = await app.inject({
        method: "POST",
        url: "/api/simulations",
        payload
      });
      const body = response.json<ApiError>();

      expect(response.statusCode).toBe(400);
      expect(body.error.code).toBe("INVALID_SIMULATION_REQUEST");
    }
  });

  it("returns joker cards as public discriminated DTOs without suit or rank", async () => {
    games.set("joker-dto", {
      state: createStateWithHands([
        [joker(), card("hearts", "3")],
        [card("spades", "A")],
        [card("clubs", "A")],
        [card("diamonds", "A")],
        [card("hearts", "A")]
      ]),
      humanPlayerId: "player-0",
      agents: createAgents(["player-1", "player-2", "player-3", "player-4"])
    });

    const response = await app.inject({
      method: "GET",
      url: "/api/games/joker-dto"
    });
    const body = response.json<GetGameResponse>();
    const publicJoker = body.state.self.hand.find((card) => card.type === "joker");

    expect(response.statusCode).toBe(200);
    expect(body.state.self.hand.every((card) => card.type === "standard" || card.type === "joker")).toBe(
      true
    );
    expect(publicJoker).toEqual({ type: "joker", id: "joker" });
    expect(publicJoker).toBeDefined();
    if (publicJoker !== undefined) {
      expect(hasOwn(publicJoker, "suit")).toBe(false);
      expect(hasOwn(publicJoker, "rank")).toBe(false);
    }
    expect(body.state.opponents.some((opponent) => hasOwn(opponent, "hand"))).toBe(false);
  });

  it("gets an existing game and returns 404 for a missing game", async () => {
    const created = await createGame();

    const found = await app.inject({
      method: "GET",
      url: `/api/games/${created.gameId}`
    });
    const missing = await app.inject({
      method: "GET",
      url: "/api/games/missing"
    });

    expect(found.statusCode).toBe(200);
    expect(found.json<GetGameResponse>()).toMatchObject({
      gameId: created.gameId,
      state: {
        phase: "bidding",
        trumpSuit: null,
        contract: null
      }
    });
    expect(missing.statusCode).toBe(404);
  });

  it("plays a legal card and advances all four AI players", async () => {
    const state = createStateWithHands([
      [card("hearts", "A"), card("clubs", "6")],
      [card("hearts", "K"), card("clubs", "2")],
      [card("clubs", "A"), card("clubs", "3")],
      [card("hearts", "3"), card("diamonds", "4")],
      [card("clubs", "Q"), card("spades", "5")]
    ]);
    games.set("playing", {
      state,
      humanPlayerId: "player-0",
      agents: createAgents(["player-1", "player-2", "player-3", "player-4"])
    });
    const cardToPlay = state.players[0].hand[0];

    const response = await app.inject({
      method: "POST",
      url: "/api/games/playing/actions",
      payload: {
        action: {
          type: "play-card",
          cardId: cardToPlay.id
        }
      }
    });
    const body = response.json<SendActionResponse>();

    expect(response.statusCode).toBe(200);
    expect(body.state.currentTrick).toHaveLength(5);
    expect(body.state.isTrickComplete).toBe(true);
    expect(body.state.trumpSuit).toBe("spades");
    expect(body.state.self.handCount).toBe(1);
    expect(body.state.opponents.some((opponent) => hasOwn(opponent, "hand"))).toBe(false);
  });

  it("returns captured point cards by player after a completed trick", async () => {
    games.set("captured-points", {
      state: createStateWithHands([
        [card("hearts", "A"), card("clubs", "2")],
        [card("hearts", "K"), card("clubs", "3")],
        [card("hearts", "Q"), card("clubs", "4")],
        [card("hearts", "7"), card("clubs", "5")],
        [card("hearts", "2"), card("clubs", "6")]
      ]),
      humanPlayerId: "player-0",
      agents: new Map([
        ["player-1", new PreferredCardAgent("hearts-K")],
        ["player-2", new PreferredCardAgent("hearts-Q")],
        ["player-3", new PreferredCardAgent("hearts-7")],
        ["player-4", new PreferredCardAgent("hearts-2")]
      ])
    });

    const response = await app.inject({
      method: "POST",
      url: "/api/games/captured-points/actions",
      payload: {
        action: {
          type: "play-card",
          cardId: "hearts-A"
        }
      }
    });
    const body = response.json<SendActionResponse>();

    expect(response.statusCode).toBe(200);
    expect(body.state.currentPlayerId).toBe("player-0");
    expect(body.state.self.capturedPointCards).toEqual([
      { type: "standard", id: "hearts-A", suit: "hearts", rank: "A" },
      { type: "standard", id: "hearts-K", suit: "hearts", rank: "K" },
      { type: "standard", id: "hearts-Q", suit: "hearts", rank: "Q" }
    ]);
    expect(body.state.opponents.every((opponent) => opponent.capturedPointCards.length === 0)).toBe(
      true
    );
  });

  it("allows the human to play joker while holding the lead suit and continues AI play", async () => {
    games.set("joker-follow", {
      state: {
        ...createStateWithHands([
          [card("hearts", "3"), card("spades", "A"), joker()],
          [card("hearts", "K")],
          [card("clubs", "A")],
          [card("diamonds", "A")],
          [card("clubs", "Q")]
        ]),
        currentPlayerId: "player-0",
        currentTrick: [{ playerId: "player-4", card: card("hearts", "J") }]
      },
      humanPlayerId: "player-0",
      agents: createAgents(["player-1", "player-2", "player-3", "player-4"])
    });

    const response = await app.inject({
      method: "POST",
      url: "/api/games/joker-follow/actions",
      payload: {
        action: {
          type: "play-card",
          cardId: "joker"
        }
      }
    });
    const body = response.json<SendActionResponse>();
    const playedJoker = body.state.currentTrick.find((played) => played.card.type === "joker");

    expect(response.statusCode).toBe(200);
    expect(body.state.currentTrick).toHaveLength(5);
    expect(playedJoker).toMatchObject({
      playerId: "player-0",
      card: {
        type: "joker",
        id: "joker"
      }
    });
    expect(body.state.opponents.some((opponent) => hasOwn(opponent, "hand"))).toBe(false);
  });

  it("uses trump follow and trump winner when joker leads", async () => {
    games.set("joker-lead", {
      state: createStateWithHands([
        [joker(), card("clubs", "2")],
        [card("spades", "2")],
        [card("hearts", "A")],
        [card("clubs", "A")],
        [card("diamonds", "A")]
      ]),
      humanPlayerId: "player-0",
      agents: createAgents(["player-1", "player-2", "player-3", "player-4"])
    });

    const response = await app.inject({
      method: "POST",
      url: "/api/games/joker-lead/actions",
      payload: {
        action: {
          type: "play-card",
          cardId: "joker"
        }
      }
    });
    const body = response.json<SendActionResponse>();

    expect(response.statusCode).toBe(200);
    expect(body.state.currentTrick.map((played) => played.card)).toEqual([
      { type: "joker", id: "joker" },
      { type: "standard", id: "spades-2", suit: "spades", rank: "2" },
      { type: "standard", id: "hearts-A", suit: "hearts", rank: "A" },
      { type: "standard", id: "clubs-A", suit: "clubs", rank: "A" },
      { type: "standard", id: "diamonds-A", suit: "diamonds", rank: "A" }
    ]);
    expect(body.state.isTrickComplete).toBe(true);
    expect(body.state.currentPlayerId).toBe("player-1");
  });

  it("uses oruma as the trick winner over trump, lead cards, and joker", async () => {
    games.set("oruma-winner", {
      state: {
        ...createStateWithHands([
          [card("hearts", "A"), card("clubs", "6")],
          [card("spades", "A")],
          [card("hearts", "K")],
          [card("clubs", "A")],
          [joker()]
        ]),
        trumpSuit: "hearts",
        contract: {
          napoleonPlayerId: "player-0",
          trumpSuit: "hearts",
          targetPointCards: 13
        }
      },
      humanPlayerId: "player-0",
      agents: new Map([
        ["player-1", new PreferredCardAgent("spades-A")],
        ["player-2", new PreferredCardAgent("hearts-K")],
        ["player-3", new PreferredCardAgent("clubs-A")],
        ["player-4", new PreferredCardAgent("joker")]
      ])
    });

    const response = await app.inject({
      method: "POST",
      url: "/api/games/oruma-winner/actions",
      payload: {
        action: {
          type: "play-card",
          cardId: "hearts-A"
        }
      }
    });
    const body = response.json<SendActionResponse>();

    expect(response.statusCode).toBe(200);
    expect(body.state.currentTrick.map((played) => played.card.id)).toEqual([
      "hearts-A",
      "spades-A",
      "hearts-K",
      "clubs-A",
      "joker"
    ]);
    expect(body.state.isTrickComplete).toBe(true);
    expect(body.state.currentPlayerId).toBe("player-1");
  });

  it("uses yoromeki over oruma only when both cards are in the same trick", async () => {
    games.set("yoromeki-winner", {
      state: createStateWithHands([
        [card("spades", "A"), card("clubs", "6")],
        [card("hearts", "Q")],
        [joker()],
        [card("clubs", "A")],
        [card("diamonds", "A")]
      ]),
      humanPlayerId: "player-0",
      agents: new Map([
        ["player-1", new PreferredCardAgent("hearts-Q")],
        ["player-2", new PreferredCardAgent("joker")],
        ["player-3", new PreferredCardAgent("clubs-A")],
        ["player-4", new PreferredCardAgent("diamonds-A")]
      ])
    });

    const response = await app.inject({
      method: "POST",
      url: "/api/games/yoromeki-winner/actions",
      payload: {
        action: {
          type: "play-card",
          cardId: "spades-A"
        }
      }
    });
    const body = response.json<SendActionResponse>();

    expect(response.statusCode).toBe(200);
    expect(body.state.currentTrick.map((played) => played.card.id)).toEqual([
      "spades-A",
      "hearts-Q",
      "joker",
      "clubs-A",
      "diamonds-A"
    ]);
    expect(body.state.isTrickComplete).toBe(true);
    expect(body.state.currentPlayerId).toBe("player-1");
  });

  it("keeps hearts-Q and clubs-A as normal cards when oruma is absent", async () => {
    games.set("yoromeki-normal", {
      state: createStateWithHands([
        [card("spades", "K"), card("clubs", "6")],
        [card("hearts", "Q")],
        [card("spades", "10")],
        [card("clubs", "A")],
        [card("diamonds", "A")]
      ]),
      humanPlayerId: "player-0",
      agents: new Map([
        ["player-1", new PreferredCardAgent("hearts-Q")],
        ["player-2", new PreferredCardAgent("spades-10")],
        ["player-3", new PreferredCardAgent("clubs-A")],
        ["player-4", new PreferredCardAgent("diamonds-A")]
      ])
    });

    const yoromekiResponse = await app.inject({
      method: "POST",
      url: "/api/games/yoromeki-normal/actions",
      payload: {
        action: {
          type: "play-card",
          cardId: "spades-K"
        }
      }
    });
    const yoromekiBody = yoromekiResponse.json<SendActionResponse>();

    expect(yoromekiResponse.statusCode).toBe(200);
    expect(yoromekiBody.state.currentPlayerId).toBe("player-0");

    games.set("clubs-a-normal", {
      state: createStateWithHands([
        [card("clubs", "A"), card("hearts", "6")],
        [card("hearts", "Q")],
        [card("spades", "K")],
        [card("clubs", "2")],
        [card("diamonds", "2")]
      ]),
      humanPlayerId: "player-0",
      agents: new Map([
        ["player-1", new PreferredCardAgent("hearts-Q")],
        ["player-2", new PreferredCardAgent("spades-K")],
        ["player-3", new PreferredCardAgent("clubs-2")],
        ["player-4", new PreferredCardAgent("diamonds-2")]
      ])
    });

    const clubsResponse = await app.inject({
      method: "POST",
      url: "/api/games/clubs-a-normal/actions",
      payload: {
        action: {
          type: "play-card",
          cardId: "clubs-A"
        }
      }
    });
    const clubsBody = clubsResponse.json<SendActionResponse>();

    expect(clubsResponse.statusCode).toBe(200);
    expect(clubsBody.state.currentPlayerId).toBe("player-2");
  });

  it("uses sei jack as the trick winner over ura jack, normal trump, and joker", async () => {
    games.set("sei-jack-winner", {
      state: {
        ...createStateWithHands([
          [card("hearts", "A"), card("clubs", "6")],
          [card("hearts", "J")],
          [card("diamonds", "J")],
          [card("clubs", "A")],
          [joker()]
        ]),
        trumpSuit: "hearts",
        contract: {
          napoleonPlayerId: "player-0",
          trumpSuit: "hearts",
          targetPointCards: 13
        }
      },
      humanPlayerId: "player-0",
      agents: new Map([
        ["player-1", new PreferredCardAgent("hearts-J")],
        ["player-2", new PreferredCardAgent("diamonds-J")],
        ["player-3", new PreferredCardAgent("clubs-A")],
        ["player-4", new PreferredCardAgent("joker")]
      ])
    });

    const response = await app.inject({
      method: "POST",
      url: "/api/games/sei-jack-winner/actions",
      payload: {
        action: {
          type: "play-card",
          cardId: "hearts-A"
        }
      }
    });
    const body = response.json<SendActionResponse>();

    expect(response.statusCode).toBe(200);
    expect(body.state.currentTrick.map((played) => played.card.id)).toEqual([
      "hearts-A",
      "hearts-J",
      "diamonds-J",
      "clubs-A",
      "joker"
    ]);
    expect(body.state.currentPlayerId).toBe("player-1");
  });

  it("uses ura jack as the trick winner over normal trump when sei jack and oruma are absent", async () => {
    games.set("ura-jack-winner", {
      state: {
        ...createStateWithHands([
          [card("hearts", "A"), card("clubs", "6")],
          [card("diamonds", "J")],
          [card("hearts", "K")],
          [card("clubs", "A")],
          [joker()]
        ]),
        trumpSuit: "hearts",
        contract: {
          napoleonPlayerId: "player-0",
          trumpSuit: "hearts",
          targetPointCards: 13
        }
      },
      humanPlayerId: "player-0",
      agents: new Map([
        ["player-1", new PreferredCardAgent("diamonds-J")],
        ["player-2", new PreferredCardAgent("hearts-K")],
        ["player-3", new PreferredCardAgent("clubs-A")],
        ["player-4", new PreferredCardAgent("joker")]
      ])
    });

    const response = await app.inject({
      method: "POST",
      url: "/api/games/ura-jack-winner/actions",
      payload: {
        action: {
          type: "play-card",
          cardId: "hearts-A"
        }
      }
    });
    const body = response.json<SendActionResponse>();

    expect(response.statusCode).toBe(200);
    expect(body.state.currentTrick.map((played) => played.card.id)).toEqual([
      "hearts-A",
      "diamonds-J",
      "hearts-K",
      "clubs-A",
      "joker"
    ]);
    expect(body.state.currentPlayerId).toBe("player-1");
  });

  it("uses same two as the trick winner through the API and starts the next trick from that winner", async () => {
    games.set("same-two-winner", {
      state: {
        ...createStateWithHands([
          [card("hearts", "A"), card("clubs", "6")],
          [card("hearts", "K"), card("clubs", "2")],
          [card("hearts", "Q"), card("clubs", "3")],
          [card("hearts", "7"), card("diamonds", "4")],
          [card("hearts", "2"), card("spades", "5")]
        ]),
        trickNumber: 2
      },
      humanPlayerId: "player-0",
      agents: new Map([
        ["player-1", new PreferredCardAgent("hearts-K")],
        ["player-2", new PreferredCardAgent("hearts-Q")],
        ["player-3", new PreferredCardAgent("hearts-7")],
        ["player-4", new PreferredCardAgent("hearts-2")]
      ])
    });

    const playResponse = await app.inject({
      method: "POST",
      url: "/api/games/same-two-winner/actions",
      payload: {
        action: {
          type: "play-card",
          cardId: "hearts-A"
        }
      }
    });
    const playBody = playResponse.json<SendActionResponse>();

    expect(playResponse.statusCode).toBe(200);
    expect(playBody.state.currentTrick.map((played) => played.card.id)).toEqual([
      "hearts-A",
      "hearts-K",
      "hearts-Q",
      "hearts-7",
      "hearts-2"
    ]);
    expect(playBody.state.currentPlayerId).toBe("player-4");
    expect(playBody.state.isTrickComplete).toBe(true);
    expect(playBody.state.specialCards).toEqual({
      orumaCardId: "spades-A",
      yoromekiCardId: "hearts-Q",
      seiJackCardId: "spades-J",
      uraJackCardId: "clubs-J"
    });

    const nextResponse = await app.inject({
      method: "POST",
      url: "/api/games/same-two-winner/next-trick"
    });
    const nextBody = nextResponse.json<NextTrickResponse>();

    expect(nextResponse.statusCode).toBe(200);
    expect(nextBody.state.currentTrick.map((played) => played.playerId)).toEqual(["player-4"]);
    expect(nextBody.state.currentPlayerId).toBe("player-0");
    expect(nextBody.state.trickNumber).toBe(3);
  });

  it("passes bidding public view fields to AI agents", async () => {
    const created = await createGame();
    const record = games.get(created.gameId);

    if (record === undefined) {
      throw new Error("Expected created game to be stored.");
    }

    const inspectingAgent = new InspectingAgent();
    games.set(created.gameId, {
      ...record,
      agents: new Map(record.agents).set("player-1", inspectingAgent)
    });

    const response = await app.inject({
      method: "POST",
      url: `/api/games/${created.gameId}/actions`,
      payload: {
        action: {
          type: "bid",
          suit: "hearts",
          targetPointCards: 13
        }
      }
    });

    expect(response.statusCode).toBe(200);
    expect(inspectingAgent.observedPhase).toBe("bidding");
    expect(inspectingAgent.observedTrumpSuit).toBeNull();
    expect(inspectingAgent.observedHighestBid).toMatchObject({
      playerId: "player-0",
      suit: "hearts",
      targetPointCards: 13
    });
    expect(inspectingAgent.observedHistoryLength).toBe(1);
    expect(inspectingAgent.observedContract).toBeNull();
    expect(inspectingAgent.observedSpecialCards).toEqual({
      orumaCardId: "spades-A",
      yoromekiCardId: "hearts-Q",
      seiJackCardId: null,
      uraJackCardId: null
    });
    expect(inspectingAgent.observedLegalActions?.length).toBeGreaterThan(0);
    expect(inspectingAgent.observedOpponentHandLeak).toBe(false);
  });

  it("advances AI bidding after a human bid and finalizes the contract", async () => {
    const created = await createGame();
    const record = games.get(created.gameId);

    if (record === undefined) {
      throw new Error("Expected created game to be stored.");
    }

    games.set(created.gameId, {
      ...record,
      agents: new Map([
        ["player-1", new PassAgent()],
        ["player-2", new PassAgent()],
        ["player-3", new PassAgent()],
        ["player-4", new PassAgent()]
      ])
    });

    const response = await app.inject({
      method: "POST",
      url: `/api/games/${created.gameId}/actions`,
      payload: {
        action: {
          type: "bid",
          suit: "hearts",
          targetPointCards: 13
        }
      }
    });
    const body = response.json<SendActionResponse>();

    expect(response.statusCode).toBe(200);
    expect(body.state.phase).toBe("choosing-adjutant");
    expect(body.state.contract).toEqual({
      napoleonPlayerId: "player-0",
      trumpSuit: "hearts",
      targetPointCards: 13
    });
    expect(body.state.currentPlayerId).toBe("player-0");
    expect(body.state.trumpSuit).toBe("hearts");
    expect(body.state.specialCards).toEqual({
      orumaCardId: "spades-A",
      yoromekiCardId: "hearts-Q",
      seiJackCardId: "hearts-J",
      uraJackCardId: "diamonds-J"
    });
    expect(body.state.exchange).toBeNull();
    expect(body.state.adjutant).toBeNull();
    expect(body.state.adjutantChoice).toEqual({
      napoleonPlayerId: "player-0",
      jokerAllowed: true
    });
    expect(body.state.self.hand).toHaveLength(10);
    expect(body.state.opponents.some((opponent) => hasOwn(opponent, "hand"))).toBe(false);
    expect(body.state.latestEvent).toBeNull();
    expect(body.state.result).toBeNull();
  });

  it("creates the special spades-12 contract when everyone passes", async () => {
    const created = await createGame();
    const record = games.get(created.gameId);

    if (record === undefined) {
      throw new Error("Expected created game to be stored.");
    }

    games.set(created.gameId, {
      ...record,
      agents: new Map([
        ["player-1", new PassAgent()],
        ["player-2", new PassAgent()],
        ["player-3", new PassAgent()],
        ["player-4", new PassAgent()]
      ])
    });

    const response = await app.inject({
      method: "POST",
      url: `/api/games/${created.gameId}/actions`,
      payload: {
        action: {
          type: "pass"
        }
      }
    });
    const body = response.json<SendActionResponse>();

    expect(response.statusCode).toBe(200);
    expect(body.state.phase).toBe("choosing-adjutant");
    expect(body.state.contract).toEqual({
      napoleonPlayerId: "player-0",
      trumpSuit: "spades",
      targetPointCards: 12
    });
    expect(body.state.currentPlayerId).toBe("player-0");
    expect(body.state.trumpSuit).toBe("spades");
    expect(body.state.exchange).toBeNull();
    expect(body.state.adjutant).toBeNull();
    expect(body.state.adjutantChoice).toEqual({
      napoleonPlayerId: "player-0",
      jokerAllowed: true
    });
    expect(body.state.self.hand).toHaveLength(10);
  });

  it("finalizes a normal AI winning bid and advances play until the human turn", async () => {
    const created = await createGame();
    const record = games.get(created.gameId);

    if (record === undefined) {
      throw new Error("Expected created game to be stored.");
    }

    games.set(created.gameId, {
      ...record,
      agents: new Map([
        ["player-1", new PreferBidAgent("spades", 13)],
        ["player-2", new PassAgent()],
        ["player-3", new PassAgent()],
        ["player-4", new PassAgent()]
      ])
    });

    const bidResponse = await app.inject({
      method: "POST",
      url: `/api/games/${created.gameId}/actions`,
      payload: {
        action: {
          type: "bid",
          suit: "hearts",
          targetPointCards: 13
        }
      }
    });
    const biddingBody = bidResponse.json<SendActionResponse>();

    expect(bidResponse.statusCode).toBe(200);
    expect(biddingBody.state.phase).toBe("bidding");
    expect(biddingBody.state.currentPlayerId).toBe("player-0");
    expect(biddingBody.state.bidding?.highestBid).toEqual({
      playerId: "player-1",
      suit: "spades",
      targetPointCards: 13
    });

    const passResponse = await app.inject({
      method: "POST",
      url: `/api/games/${created.gameId}/actions`,
      payload: {
        action: {
          type: "pass"
        }
      }
    });
    const playingBody = passResponse.json<SendActionResponse>();

    expect(passResponse.statusCode).toBe(200);
    expect(playingBody.state.phase).toBe("playing");
    expect(playingBody.state.contract).toEqual({
      napoleonPlayerId: "player-1",
      trumpSuit: "spades",
      targetPointCards: 13
    });
    expect(playingBody.state.adjutant?.calledCardId).toBe("spades-A");
    expect(
      playingBody.state.adjutant?.revealedPlayerId === null ||
        typeof playingBody.state.adjutant?.revealedPlayerId === "string"
    ).toBe(true);
    expect(playingBody.state.adjutantChoice).toBeNull();
    expect(playingBody.state.currentPlayerId).toBe("player-0");
    expect(playingBody.state.currentTrick.map((played) => played.playerId)).toEqual([
      "player-1",
      "player-2",
      "player-3",
      "player-4"
    ]);
    expect(playingBody.state.legalActions.length).toBeGreaterThan(0);
  });

  it("lets a human Napoleon discard three buried cards and starts play", async () => {
    const state = createAllPassExchangeState();
    const discardIds = state.players[0].hand.slice(0, 3).map((card) => card.id);
    games.set("human-exchange", {
      state,
      humanPlayerId: "player-0",
      agents: createAgents(["player-1", "player-2", "player-3", "player-4"])
    });

    const response = await app.inject({
      method: "POST",
      url: "/api/games/human-exchange/actions",
      payload: {
        action: {
          type: "discard-cards",
          cardIds: discardIds
        }
      }
    });
    const body = response.json<SendActionResponse>();

    expect(response.statusCode).toBe(200);
    expect(body.state.phase).toBe("playing");
    expect(body.state.exchange).toBeNull();
    expect(body.state.adjutantChoice).toBeNull();
    expect(body.state.adjutant).toEqual({
      calledCardId: state.adjutant?.calledCardId,
      revealedPlayerId: null
    });
    expect(body.state.self.hand).toHaveLength(10);
    expect(body.state.legalActions.some((action) => action.type === "play-card")).toBe(true);
    expect(body.state.latestEvent).toMatchObject({
      type: "buried-cards-resolved",
      napoleonPlayerId: "player-0"
    });
    const stored = games.get("human-exchange")?.state;
    expect([
      ...(stored?.awardedPointCards.flatMap((award) => award.cards.map((card) => card.id)) ?? []),
      ...(stored?.excludedCards.map((card) => card.id) ?? [])
    ].sort()).toEqual([...discardIds].sort());
  });

  it("reveals only point cards from buried cards after human exchange", async () => {
    const state = {
      ...createAllPassExchangeState(),
      players: [
        {
          id: "player-0",
          hand: [
            card("spades", "A"),
            card("hearts", "10"),
            card("clubs", "3"),
            card("spades", "2"),
            card("spades", "3"),
            card("spades", "4"),
            card("spades", "5"),
            card("spades", "6"),
            card("spades", "7"),
            card("spades", "8"),
            card("hearts", "2"),
            card("hearts", "3"),
            card("hearts", "4")
          ]
        },
        { id: "player-1", hand: [card("clubs", "A")] },
        { id: "player-2", hand: [card("diamonds", "A")] },
        { id: "player-3", hand: [card("clubs", "K")] },
        { id: "player-4", hand: [card("diamonds", "K")] }
      ]
    };
    games.set("buried-summary", {
      state,
      humanPlayerId: "player-0",
      agents: createAgents(["player-1", "player-2", "player-3", "player-4"])
    });

    const response = await app.inject({
      method: "POST",
      url: "/api/games/buried-summary/actions",
      payload: {
        action: {
          type: "discard-cards",
          cardIds: ["spades-A", "hearts-10", "clubs-3"]
        }
      }
    });
    const body = response.json<SendActionResponse>();
    const serializedEvent = JSON.stringify(body.state.latestEvent);

    expect(response.statusCode).toBe(200);
    expect(body.state.latestEvent).toEqual({
      type: "buried-cards-resolved",
      napoleonPlayerId: "player-0",
      awardedPointCards: [
        { type: "standard", id: "spades-A", suit: "spades", rank: "A" },
        { type: "standard", id: "hearts-10", suit: "hearts", rank: "10" }
      ],
      hiddenNonPointCardCount: 1
    });
    expect(body.state.self.capturedPointCards.map((card) => card.id)).toEqual([
      "spades-A",
      "hearts-10"
    ]);
    expect(serializedEvent).not.toContain("clubs-3");
    expect(hasOwn(body.state, "unusedCards")).toBe(false);
    expect(hasOwn(body.state, "buriedCards")).toBe(false);
  });

  it("counts buried joker as a hidden card without exposing the joker id", async () => {
    const state = {
      ...createAllPassExchangeState(),
      players: [
        {
          id: "player-0",
          hand: [
            joker(),
            card("clubs", "3"),
            card("diamonds", "8"),
            card("spades", "2"),
            card("spades", "3"),
            card("spades", "4"),
            card("spades", "5"),
            card("spades", "6"),
            card("spades", "7"),
            card("spades", "8"),
            card("hearts", "2"),
            card("hearts", "3"),
            card("hearts", "4")
          ]
        },
        { id: "player-1", hand: [card("clubs", "A")] },
        { id: "player-2", hand: [card("diamonds", "A")] },
        { id: "player-3", hand: [card("clubs", "K")] },
        { id: "player-4", hand: [card("diamonds", "K")] }
      ]
    };
    games.set("buried-joker", {
      state,
      humanPlayerId: "player-0",
      agents: createAgents(["player-1", "player-2", "player-3", "player-4"])
    });

    const response = await app.inject({
      method: "POST",
      url: "/api/games/buried-joker/actions",
      payload: {
        action: {
          type: "discard-cards",
          cardIds: ["joker", "clubs-3", "diamonds-8"]
        }
      }
    });
    const body = response.json<SendActionResponse>();
    const serializedEvent = JSON.stringify(body.state.latestEvent);

    expect(response.statusCode).toBe(200);
    expect(body.state.latestEvent).toEqual({
      type: "buried-cards-resolved",
      napoleonPlayerId: "player-0",
      awardedPointCards: [],
      hiddenNonPointCardCount: 3
    });
    expect(serializedEvent).not.toContain("joker");
    expect(serializedEvent).not.toContain("clubs-3");
    expect(serializedEvent).not.toContain("diamonds-8");
  });

  it("lets a human Napoleon choose an adjutant card and starts exchange", async () => {
    const choosing = createAllPassAdjutantChoiceState();
    const calledCard = choosing.players[1].hand.find(isStandardCard);

    expect(calledCard).toBeDefined();
    if (calledCard === undefined) {
      throw new Error("Expected player-1 to have a standard card.");
    }

    games.set("human-adjutant", {
      state: choosing,
      humanPlayerId: "player-0",
      agents: createAgents(["player-1", "player-2", "player-3", "player-4"])
    });

    const response = await app.inject({
      method: "POST",
      url: "/api/games/human-adjutant/actions",
      payload: {
        action: {
          type: "choose-adjutant",
          cardId: calledCard.id
        }
      }
    });
    const body = response.json<SendActionResponse>();

    expect(response.statusCode).toBe(200);
    expect(body.state.phase).toBe("exchanging");
    expect(body.state.adjutant).toEqual({
      calledCardId: calledCard.id,
      revealedPlayerId: null
    });
    expect(body.state.adjutantChoice).toBeNull();
    expect(body.state.exchange).toEqual({
      napoleonPlayerId: "player-0",
      requiredDiscardCount: 3
    });
    expect(body.state.self.hand).toHaveLength(13);
    expect(body.state.legalActions).toEqual([]);
    expect(body.state.latestEvent).toBeNull();
    expect(games.get("human-adjutant")?.state.adjutant).toEqual({
      calledCardId: calledCard.id,
      playerId: "player-1",
      revealed: false
    });
  });

  it("lets a human Napoleon choose the joker as an adjutant card", async () => {
    const choosing = {
      ...createStateWithHands([
        [
          card("spades", "2"),
          card("spades", "3"),
          card("spades", "4"),
          card("spades", "5"),
          card("spades", "6"),
          card("spades", "7"),
          card("spades", "8"),
          card("spades", "9"),
          card("spades", "10"),
          card("spades", "J")
        ],
        [joker()],
        [card("clubs", "3")],
        [card("diamonds", "4")],
        [card("hearts", "5")]
      ]),
      phase: "choosing-adjutant" as const,
      currentPlayerId: "player-0",
      unusedCards: [card("clubs", "2"), card("clubs", "4"), card("diamonds", "3")]
    };
    games.set("human-joker-adjutant", {
      state: choosing,
      humanPlayerId: "player-0",
      agents: createAgents(["player-1", "player-2", "player-3", "player-4"])
    });

    const response = await app.inject({
      method: "POST",
      url: "/api/games/human-joker-adjutant/actions",
      payload: {
        action: {
          type: "choose-adjutant",
          cardId: "joker"
        }
      }
    });
    const body = response.json<SendActionResponse>();

    expect(response.statusCode).toBe(200);
    expect(body.state.phase).toBe("exchanging");
    expect(body.state.adjutant).toEqual({
      calledCardId: "joker",
      revealedPlayerId: null
    });
    expect(body.state.self.hand).toHaveLength(13);
    expect(body.state.exchange).toEqual({
      napoleonPlayerId: "player-0",
      requiredDiscardCount: 3
    });
    expect(games.get("human-joker-adjutant")?.state.adjutant).toEqual({
      calledCardId: "joker",
      playerId: "player-1",
      revealed: false
    });
  });

  it("reveals a joker adjutant after the owner plays the joker through the API", async () => {
    games.set("joker-adjutant-reveal", {
      state: {
        ...createStateWithHands([
          [card("hearts", "2"), card("spades", "2")],
          [joker(), card("spades", "3")],
          [card("clubs", "3"), card("spades", "4")],
          [card("diamonds", "4"), card("spades", "5")],
          [card("clubs", "5"), card("spades", "6")]
        ]),
        adjutant: {
          calledCardId: "joker",
          playerId: "player-1",
          revealed: false
        }
      },
      humanPlayerId: "player-0",
      agents: new Map([
        ["player-1", new PreferredCardAgent("joker")],
        ["player-2", new PreferredCardAgent("clubs-3")],
        ["player-3", new PreferredCardAgent("diamonds-4")],
        ["player-4", new PreferredCardAgent("clubs-5")]
      ])
    });

    const response = await app.inject({
      method: "POST",
      url: "/api/games/joker-adjutant-reveal/actions",
      payload: {
        action: {
          type: "play-card",
          cardId: "hearts-2"
        }
      }
    });
    const body = response.json<SendActionResponse>();

    expect(response.statusCode).toBe(200);
    expect(body.state.currentTrick.map((played) => played.card.id)).toContain("joker");
    expect(body.state.adjutant).toEqual({
      calledCardId: "joker",
      revealedPlayerId: "player-1"
    });
    expect(body.state.opponents.some((opponent) => hasOwn(opponent, "hand"))).toBe(false);
  });

  it("rejects invalid discard requests and leaves exchange state unchanged", async () => {
    const state = createAllPassExchangeState();
    games.set("invalid-exchange", {
      state,
      humanPlayerId: "player-0",
      agents: createAgents(["player-1", "player-2", "player-3", "player-4"])
    });

    const malformedPayloads = [
      { action: { type: "discard-cards", cardIds: "not-array" } },
      { action: { type: "discard-cards", cardIds: ["a", 1, "c"] } },
      { action: { type: "discard-cards", cardIds: ["a", "b", "c"], extra: true } }
    ];

    for (const payload of malformedPayloads) {
      const snapshot = createStateSnapshot(games.get("invalid-exchange")?.state ?? state);
      const response = await app.inject({
        method: "POST",
        url: "/api/games/invalid-exchange/actions",
        payload
      });

      expect(response.statusCode).toBe(400);
      expect(createStateSnapshot(games.get("invalid-exchange")?.state ?? state)).toEqual(
        snapshot
      );
    }

    const validHandIds = state.players[0].hand.map((card) => card.id);
    const coreInvalidCases = [
      {
        action: { type: "discard-cards", cardIds: validHandIds.slice(0, 2) },
        code: "INVALID_DISCARD_COUNT"
      },
      {
        action: {
          type: "discard-cards",
          cardIds: [validHandIds[0], validHandIds[0], validHandIds[1]]
        },
        code: "DUPLICATE_CARD_ID"
      },
      {
        action: {
          type: "discard-cards",
          cardIds: [validHandIds[0], validHandIds[1], "not-in-hand"]
        },
        code: "CARD_NOT_IN_HAND"
      }
    ] as const;

    for (const testCase of coreInvalidCases) {
      const snapshot = createStateSnapshot(games.get("invalid-exchange")?.state ?? state);
      const response = await app.inject({
        method: "POST",
        url: "/api/games/invalid-exchange/actions",
        payload: { action: testCase.action }
      });

      expect(response.statusCode).toBe(400);
      expect(response.json<ApiError>().error.code).toBe(testCase.code);
      expect(createStateSnapshot(games.get("invalid-exchange")?.state ?? state)).toEqual(
        snapshot
      );
    }
  });

  it("rejects invalid adjutant requests and leaves choosing state unchanged", async () => {
    const choosing = createAllPassAdjutantChoiceState();
    games.set("invalid-adjutant", {
      state: choosing,
      humanPlayerId: "player-0",
      agents: createAgents(["player-1", "player-2", "player-3", "player-4"])
    });

    const malformedPayloads = [
      { action: { type: "choose-adjutant", cardId: 1 } },
      { action: { type: "choose-adjutant", cardId: "spades-A", extra: true } }
    ];

    for (const payload of malformedPayloads) {
      const snapshot = createStateSnapshot(games.get("invalid-adjutant")?.state ?? choosing);
      const response = await app.inject({
        method: "POST",
        url: "/api/games/invalid-adjutant/actions",
        payload
      });

      expect(response.statusCode).toBe(400);
      expect(response.json<ApiError>().error.code).toBe("INVALID_ACTION");
      expect(createStateSnapshot(games.get("invalid-adjutant")?.state ?? choosing)).toEqual(
        snapshot
      );
    }

    for (const testCase of [
      { action: { type: "choose-adjutant", cardId: "spades-1" }, code: "INVALID_ADJUTANT_CARD" }
    ] as const) {
      const snapshot = createStateSnapshot(games.get("invalid-adjutant")?.state ?? choosing);
      const response = await app.inject({
        method: "POST",
        url: "/api/games/invalid-adjutant/actions",
        payload: { action: testCase.action }
      });

      expect(response.statusCode).toBe(400);
      expect(response.json<ApiError>().error.code).toBe(testCase.code);
      expect(createStateSnapshot(games.get("invalid-adjutant")?.state ?? choosing)).toEqual(
        snapshot
      );
    }

    games.set("non-napoleon-adjutant", {
      state: choosing,
      humanPlayerId: "player-1",
      agents: createAgents(["player-0", "player-2", "player-3", "player-4"])
    });

    const nonNapoleonResponse = await app.inject({
      method: "POST",
      url: "/api/games/non-napoleon-adjutant/actions",
      payload: {
        action: {
          type: "choose-adjutant",
          cardId: "spades-A"
        }
      }
    });

    expect(nonNapoleonResponse.statusCode).toBe(400);
    expect(nonNapoleonResponse.json<ApiError>().error.code).toBe("NOT_NAPOLEON");
  });

  it("rejects discard requests outside exchange or by non-Napoleon humans", async () => {
    const playing = createStateWithHands([
      [card("hearts", "A"), card("clubs", "6"), card("spades", "A")],
      [card("hearts", "K")],
      [card("clubs", "A")],
      [card("hearts", "3")],
      [card("clubs", "Q")]
    ]);
    games.set("playing-discard", {
      state: playing,
      humanPlayerId: "player-0",
      agents: createAgents(["player-1", "player-2", "player-3", "player-4"])
    });

    const playingResponse = await app.inject({
      method: "POST",
      url: "/api/games/playing-discard/actions",
      payload: {
        action: {
          type: "discard-cards",
          cardIds: playing.players[0].hand.slice(0, 3).map((card) => card.id)
        }
      }
    });

    expect(playingResponse.statusCode).toBe(400);
    expect(playingResponse.json<ApiError>().error.code).toBe("INVALID_ACTION_FOR_PHASE");

    const exchange = createAllPassExchangeState();
    games.set("non-napoleon-discard", {
      state: exchange,
      humanPlayerId: "player-1",
      agents: createAgents(["player-0", "player-2", "player-3", "player-4"])
    });
    const nonNapoleonResponse = await app.inject({
      method: "POST",
      url: "/api/games/non-napoleon-discard/actions",
      payload: {
        action: {
          type: "discard-cards",
          cardIds: exchange.players[1].hand.slice(0, 3).map((card) => card.id)
        }
      }
    });

    expect(nonNapoleonResponse.statusCode).toBe(400);
    expect(nonNapoleonResponse.json<ApiError>().error.code).toBe("NOT_NAPOLEON");
  });

  it("rejects invalid bidding actions and leaves stored state unchanged", async () => {
    const created = await createGame();
    const record = games.get(created.gameId);

    if (record === undefined) {
      throw new Error("Expected created game to be stored.");
    }

    const cases = [
      { action: { type: "play-card", cardId: created.state.self.hand[0].id }, code: "INVALID_ACTION_FOR_PHASE" },
      { action: { type: "bid", suit: "spades", targetPointCards: 12 }, code: "INVALID_BID" },
      { action: { type: "bid", suit: "spades", targetPointCards: 20 }, code: "INVALID_BID" }
    ] as const;

    for (const testCase of cases) {
      const snapshot = createStateSnapshot(record.state);
      const response = await app.inject({
        method: "POST",
        url: `/api/games/${created.gameId}/actions`,
        payload: {
          action: testCase.action
        }
      });
      const body = response.json<ApiError>();
      const storedAfter = games.get(created.gameId);

      expect(response.statusCode).toBe(400);
      expect(body.error.code).toBe(testCase.code);
      expect(storedAfter).toBeDefined();
      if (storedAfter !== undefined) {
        expect(createStateSnapshot(storedAfter.state)).toEqual(snapshot);
      }
    }
  });

  it("rejects bids below the current highest bid and playing-phase bid or pass", async () => {
    const biddingState = applyAction(createInitialGame(), {
      type: "bid",
      playerId: "player-0",
      suit: "hearts",
      targetPointCards: 13
    });
    games.set("low-bid", {
      state: {
        ...biddingState,
        currentPlayerId: "player-0"
      },
      humanPlayerId: "player-0",
      agents: createAgents(["player-1", "player-2", "player-3", "player-4"])
    });
    const lowBidSnapshot = createStateSnapshot(games.get("low-bid")?.state ?? biddingState);

    const lowBidResponse = await app.inject({
      method: "POST",
      url: "/api/games/low-bid/actions",
      payload: {
        action: {
          type: "bid",
          suit: "diamonds",
          targetPointCards: 13
        }
      }
    });

    expect(lowBidResponse.statusCode).toBe(400);
    expect(lowBidResponse.json<ApiError>().error.code).toBe("BID_TOO_LOW");
    expect(createStateSnapshot(games.get("low-bid")?.state ?? biddingState)).toEqual(
      lowBidSnapshot
    );

    games.set("playing-action", {
      state: createStateWithHands([
        [card("hearts", "A")],
        [card("hearts", "K")],
        [card("clubs", "A")],
        [card("hearts", "3")],
        [card("clubs", "Q")]
      ]),
      humanPlayerId: "player-0",
      agents: createAgents(["player-1", "player-2", "player-3", "player-4"])
    });

    for (const action of [
      { type: "bid" as const, suit: "spades" as const, targetPointCards: 13 },
      { type: "pass" as const }
    ]) {
      const response = await app.inject({
        method: "POST",
        url: "/api/games/playing-action/actions",
        payload: { action }
      });

      expect(response.statusCode).toBe(400);
      expect(response.json<ApiError>().error.code).toBe("INVALID_ACTION_FOR_PHASE");
    }
  });

  it("does not persist partial state when AI advancement fails during bidding", async () => {
    const created = await createGame();
    const record = games.get(created.gameId);

    if (record === undefined) {
      throw new Error("Expected created game to be stored.");
    }

    games.set(created.gameId, {
      ...record,
      agents: new Map(record.agents).set("player-1", new ThrowingAgent())
    });
    const snapshot = createStateSnapshot(record.state);

    const response = await app.inject({
      method: "POST",
      url: `/api/games/${created.gameId}/actions`,
      payload: {
        action: {
          type: "bid",
          suit: "hearts",
          targetPointCards: 13
        }
      }
    });
    const storedAfter = games.get(created.gameId);

    expect(response.statusCode).toBeGreaterThanOrEqual(500);
    expect(storedAfter).toBeDefined();
    if (storedAfter !== undefined) {
      expect(createStateSnapshot(storedAfter.state)).toEqual(snapshot);
    }
  });

  it("does not persist partial state when AI exchange succeeds but later AI play fails", async () => {
    const created = await createGame();
    const record = games.get(created.gameId);

    if (record === undefined) {
      throw new Error("Expected created game to be stored.");
    }

    games.set(created.gameId, {
      ...record,
      agents: new Map([
        ["player-1", new PreferBidAgent("spades", 13)],
        ["player-2", new PassAgent()],
        ["player-3", new PassAgent()],
        ["player-4", new PassAgent()]
      ])
    });

    await app.inject({
      method: "POST",
      url: `/api/games/${created.gameId}/actions`,
      payload: {
        action: {
          type: "bid",
          suit: "hearts",
          targetPointCards: 13
        }
      }
    });
    const beforePass = games.get(created.gameId);

    if (beforePass === undefined) {
      throw new Error("Expected game to remain stored.");
    }

    games.set(created.gameId, {
      ...beforePass,
      agents: new Map([
        ["player-1", new PreferBidAgent("spades", 13)],
        ["player-2", new ThrowingAgent()],
        ["player-3", new PassAgent()],
        ["player-4", new PassAgent()]
      ])
    });
    const snapshot = createStateSnapshot(beforePass.state);

    const response = await app.inject({
      method: "POST",
      url: `/api/games/${created.gameId}/actions`,
      payload: {
        action: {
          type: "pass"
        }
      }
    });
    const storedAfter = games.get(created.gameId);

    expect(response.statusCode).toBeGreaterThanOrEqual(500);
    expect(storedAfter).toBeDefined();
    if (storedAfter !== undefined) {
      expect(createStateSnapshot(storedAfter.state)).toEqual(snapshot);
      expect(storedAfter.state.awardedPointCards).toEqual([]);
      expect(storedAfter.state.excludedCards).toEqual([]);
      expect(storedAfter.state.phase).toBe("bidding");
    }
  });

  it("does not persist result or final trick when AI fails during the final trick", async () => {
    const state = createFinalTrickScoringState();
    games.set("final-result-failure", {
      state,
      humanPlayerId: "player-0",
      agents: new Map([
        ["player-1", new ThrowingAgent()],
        ...createAgents(["player-2", "player-3", "player-4"])
      ])
    });
    const snapshot = createStateSnapshot(state);

    const response = await app.inject({
      method: "POST",
      url: "/api/games/final-result-failure/actions",
      payload: {
        action: {
          type: "play-card",
          cardId: "spades-A"
        }
      }
    });
    const storedAfter = games.get("final-result-failure");

    expect(response.statusCode).toBeGreaterThanOrEqual(500);
    expect(storedAfter).toBeDefined();
    if (storedAfter !== undefined) {
      expect(createStateSnapshot(storedAfter.state)).toEqual(snapshot);
      expect(storedAfter.state.result).toBeNull();
      expect(storedAfter.state.phase).toBe("playing");
      expect(storedAfter.state.completedTricks).toHaveLength(9);
      expect(storedAfter.state.currentTrick).toEqual([]);
    }
  });

  it("rejects an invalid card id", async () => {
    games.set("invalid-card", {
      state: createStateWithHands([
        [card("hearts", "A")],
        [card("hearts", "K")],
        [card("clubs", "A")],
        [card("hearts", "3")],
        [card("clubs", "Q")]
      ]),
      humanPlayerId: "player-0",
      agents: createAgents(["player-1", "player-2", "player-3", "player-4"])
    });

    const response = await app.inject({
      method: "POST",
      url: "/api/games/invalid-card/actions",
      payload: {
        action: {
          type: "play-card",
          cardId: "not-in-hand"
        }
      }
    });
    const body = response.json<ApiError>();

    expect(response.statusCode).toBeGreaterThanOrEqual(400);
    expect(response.statusCode).toBeLessThan(500);
    expect(body.error.code).toBe("CARD_NOT_IN_HAND");
  });

  it("returns scoring result and fully reveals the adjutant after the final trick", async () => {
    const state = createFinalTrickScoringState();
    games.set("final-result", {
      state,
      humanPlayerId: "player-0",
      agents: createAgents(["player-1", "player-2", "player-3", "player-4"])
    });

    const response = await app.inject({
      method: "POST",
      url: "/api/games/final-result/actions",
      payload: {
        action: {
          type: "play-card",
          cardId: "spades-A"
        }
      }
    });
    const body = response.json<SendActionResponse>();

    expect(response.statusCode).toBe(200);
    expect(body.state.phase).toBe("finished");
    expect(body.state.isGameOver).toBe(true);
    expect(body.state.result).toEqual({
      winner: "napoleon-team",
      napoleonTeamPointCards: 20,
      alliancePointCards: 0,
      targetPointCards: 13,
      napoleonPlayerId: "player-0",
      adjutantPlayerId: "player-1"
    });
    expect(body.state.adjutant).toEqual({
      calledCardId: "hearts-A",
      revealedPlayerId: "player-1"
    });
    expect(body.state.latestEvent).toBeNull();
    expect(hasOwn(body.state, "buriedCards")).toBe(false);
  });

  it("rejects follow-suit violations and leaves stored state unchanged", async () => {
    const state = createFollowSuitViolationState();
    games.set("follow-suit", {
      state,
      humanPlayerId: "player-0",
      agents: createAgents(["player-1", "player-2", "player-3", "player-4"])
    });
    const snapshot = createStateSnapshot(state);

    const response = await app.inject({
      method: "POST",
      url: "/api/games/follow-suit/actions",
      payload: {
        action: {
          type: "play-card",
          cardId: "spades-A"
        }
      }
    });
    const body = response.json<ApiError>();
    const storedAfter = games.get("follow-suit");

    expect(response.statusCode).toBe(400);
    expect(body.error.code).toBe("MUST_FOLLOW_SUIT");
    expect(storedAfter).toBeDefined();
    if (storedAfter !== undefined) {
      expect(createStateSnapshot(storedAfter.state)).toEqual(snapshot);
    }
  });

  it("does not treat ura jack as trump suit for follow-suit validation", async () => {
    const state = {
      ...createStateWithHands([
        [card("clubs", "J"), card("spades", "5")],
        [card("hearts", "K")],
        [card("clubs", "3")],
        [card("diamonds", "4")],
        [card("clubs", "5")]
      ]),
      currentPlayerId: "player-0",
      currentTrick: [{ playerId: "player-1", card: card("spades", "2") }]
    };
    games.set("ura-jack-follow-suit", {
      state,
      humanPlayerId: "player-0",
      agents: createAgents(["player-1", "player-2", "player-3", "player-4"])
    });
    const snapshot = createStateSnapshot(state);

    const response = await app.inject({
      method: "POST",
      url: "/api/games/ura-jack-follow-suit/actions",
      payload: {
        action: {
          type: "play-card",
          cardId: "clubs-J"
        }
      }
    });
    const body = response.json<ApiError>();
    const storedAfter = games.get("ura-jack-follow-suit");

    expect(response.statusCode).toBe(400);
    expect(body.error.code).toBe("MUST_FOLLOW_SUIT");
    expect(storedAfter).toBeDefined();
    if (storedAfter !== undefined) {
      expect(createStateSnapshot(storedAfter.state)).toEqual(snapshot);
    }
  });

  it("rejects a request action that includes a client-supplied playerId", async () => {
    const created = await createGame();
    const card = created.state.self.hand[0];

    const response = await app.inject({
      method: "POST",
      url: `/api/games/${created.gameId}/actions`,
      payload: {
        action: {
          type: "play-card",
          playerId: "player-1",
          cardId: card.id
        }
      }
    });

    expect(response.statusCode).toBe(400);
  });

  it("rejects next-trick before the current trick is complete", async () => {
    const created = await createGame();

    const response = await app.inject({
      method: "POST",
      url: `/api/games/${created.gameId}/next-trick`
    });

    expect(response.statusCode).toBeGreaterThanOrEqual(400);
    expect(response.statusCode).toBeLessThan(500);
  });

  it("advances to the next trick after completion", async () => {
    games.set("human-lead", {
      state: createCompletedTrickWonByHuman(),
      humanPlayerId: "player-0",
      agents: createAgents(["player-1", "player-2", "player-3", "player-4"])
    });

    const response = await app.inject({
      method: "POST",
      url: "/api/games/human-lead/next-trick"
    });
    const body = response.json<NextTrickResponse>();

    expect(response.statusCode).toBe(200);
    expect(body.state.currentTrick).toEqual([]);
    expect(body.state.currentPlayerId).toBe("player-0");
    expect(body.state.trumpSuit).toBe("spades");
    expect(body.state.trickNumber).toBe(2);
  });

  it("advances to the next trick even when the next lead player is an AI", async () => {
    const state = createCompletedTrickWonByAi();
    expect(state.completedTricks[0].winnerId).toBe("player-1");
    expect(state.currentPlayerId).toBe("player-1");
    games.set("ai-lead", {
      state,
      humanPlayerId: "player-0",
      agents: createAgents(["player-1", "player-2", "player-3", "player-4"])
    });

    const response = await app.inject({
      method: "POST",
      url: "/api/games/ai-lead/next-trick"
    });
    const body = response.json<NextTrickResponse>();

    expect(response.statusCode).toBe(200);
    expect(body.state.currentPlayerId).toBe("player-0");
    expect(body.state.currentTrick).toHaveLength(4);
    expect(body.state.trumpSuit).toBe("spades");
    expect(body.state.currentTrick.map((played) => played.playerId)).toEqual([
      "player-1",
      "player-2",
      "player-3",
      "player-4"
    ]);
    expect(body.state.isTrickComplete).toBe(false);
    expect(body.state.legalActions.length).toBeGreaterThan(0);
    expect(body.state.trickNumber).toBe(2);
  });

  it("does not persist partial state when AI advancement fails after a human action", async () => {
    const state = createStateWithHands([
      [card("hearts", "A"), card("clubs", "6")],
      [card("hearts", "K"), card("clubs", "2")],
      [card("clubs", "A"), card("clubs", "3")],
      [card("hearts", "3"), card("diamonds", "4")],
      [card("clubs", "Q"), card("spades", "5")]
    ]);
    games.set("ai-play-failure", {
      state,
      humanPlayerId: "player-0",
      agents: new Map([
        ["player-1", new PassAgent()],
        ["player-2", new ThrowingAgent()],
        ...createAgents(["player-3", "player-4"])
      ])
    });
    const storedBefore = games.get("ai-play-failure");

    if (storedBefore === undefined) {
      throw new Error("Expected game to remain stored.");
    }

    const snapshot = createStateSnapshot(storedBefore.state);
    const response = await app.inject({
      method: "POST",
      url: "/api/games/ai-play-failure/actions",
      payload: {
        action: {
          type: "play-card",
          cardId: state.players[0].hand[0].id
        }
      }
    });
    const storedAfter = games.get("ai-play-failure");

    expect(response.statusCode).toBeGreaterThanOrEqual(500);
    expect(storedAfter).toBeDefined();
    if (storedAfter !== undefined) {
      expect(createStateSnapshot(storedAfter.state)).toEqual(snapshot);
    }
  });

  it("does not persist partial state when AI plays joker before a later AI failure", async () => {
    const state = createStateWithHands([
      [card("hearts", "A"), card("clubs", "6")],
      [joker()],
      [card("hearts", "K")],
      [card("hearts", "3")],
      [card("clubs", "Q")]
    ]);
    games.set("ai-joker-failure", {
      state,
      humanPlayerId: "player-0",
      agents: new Map([
        ["player-1", new PassAgent()],
        ["player-2", new ThrowingAgent()],
        ...createAgents(["player-3", "player-4"])
      ])
    });
    const snapshot = createStateSnapshot(state);

    const response = await app.inject({
      method: "POST",
      url: "/api/games/ai-joker-failure/actions",
      payload: {
        action: {
          type: "play-card",
          cardId: "hearts-A"
        }
      }
    });
    const storedAfter = games.get("ai-joker-failure");

    expect(response.statusCode).toBeGreaterThanOrEqual(500);
    expect(storedAfter).toBeDefined();
    if (storedAfter !== undefined) {
      expect(createStateSnapshot(storedAfter.state)).toEqual(snapshot);
      expect(storedAfter.state.currentTrick.some((played) => isJokerCard(played.card))).toBe(false);
    }
  });

  it("does not persist partial state when AI plays yoromeki before a later AI failure", async () => {
    const state = createStateWithHands([
      [card("spades", "A"), card("clubs", "6")],
      [card("hearts", "Q")],
      [card("clubs", "A")],
      [card("hearts", "3")],
      [card("clubs", "Q")]
    ]);
    games.set("ai-yoromeki-failure", {
      state,
      humanPlayerId: "player-0",
      agents: new Map([
        ["player-1", new PreferredCardAgent("hearts-Q")],
        ["player-2", new ThrowingAgent()],
        ...createAgents(["player-3", "player-4"])
      ])
    });
    const snapshot = createStateSnapshot(state);

    const response = await app.inject({
      method: "POST",
      url: "/api/games/ai-yoromeki-failure/actions",
      payload: {
        action: {
          type: "play-card",
          cardId: "spades-A"
        }
      }
    });
    const storedAfter = games.get("ai-yoromeki-failure");

    expect(response.statusCode).toBeGreaterThanOrEqual(500);
    expect(storedAfter).toBeDefined();
    if (storedAfter !== undefined) {
      expect(createStateSnapshot(storedAfter.state)).toEqual(snapshot);
      expect(storedAfter.state.currentTrick).toEqual([]);
    }
  });

  it("does not persist partial state when AI plays sei jack before a later AI failure", async () => {
    const state = {
      ...createStateWithHands([
        [card("hearts", "A"), card("clubs", "6")],
        [card("hearts", "J")],
        [card("diamonds", "J")],
        [card("clubs", "A")],
        [joker()]
      ]),
      trumpSuit: "hearts" as const,
      contract: {
        napoleonPlayerId: "player-0",
        trumpSuit: "hearts" as const,
        targetPointCards: 13
      }
    };
    games.set("ai-sei-jack-failure", {
      state,
      humanPlayerId: "player-0",
      agents: new Map([
        ["player-1", new PreferredCardAgent("hearts-J")],
        ["player-2", new ThrowingAgent()],
        ...createAgents(["player-3", "player-4"])
      ])
    });
    const snapshot = createStateSnapshot(state);

    const response = await app.inject({
      method: "POST",
      url: "/api/games/ai-sei-jack-failure/actions",
      payload: {
        action: {
          type: "play-card",
          cardId: "hearts-A"
        }
      }
    });
    const storedAfter = games.get("ai-sei-jack-failure");

    expect(response.statusCode).toBeGreaterThanOrEqual(500);
    expect(storedAfter).toBeDefined();
    if (storedAfter !== undefined) {
      expect(createStateSnapshot(storedAfter.state)).toEqual(snapshot);
      expect(storedAfter.state.currentTrick).toEqual([]);
      expect(storedAfter.state.adjutant).toBeNull();
      expect(storedAfter.state.result).toBeNull();
    }
  });

  it("does not persist partial state when AI plays toward same two before a later AI failure", async () => {
    const state = {
      ...createStateWithHands([
        [card("hearts", "A"), card("clubs", "6")],
        [card("hearts", "K")],
        [card("hearts", "Q")],
        [card("hearts", "7")],
        [card("hearts", "2")]
      ]),
      trickNumber: 2
    };
    games.set("ai-same-two-failure", {
      state,
      humanPlayerId: "player-0",
      agents: new Map([
        ["player-1", new PreferredCardAgent("hearts-K")],
        ["player-2", new ThrowingAgent()],
        ...createAgents(["player-3", "player-4"])
      ])
    });
    const snapshot = createStateSnapshot(state);

    const response = await app.inject({
      method: "POST",
      url: "/api/games/ai-same-two-failure/actions",
      payload: {
        action: {
          type: "play-card",
          cardId: "hearts-A"
        }
      }
    });
    const storedAfter = games.get("ai-same-two-failure");

    expect(response.statusCode).toBeGreaterThanOrEqual(500);
    expect(storedAfter).toBeDefined();
    if (storedAfter !== undefined) {
      expect(createStateSnapshot(storedAfter.state)).toEqual(snapshot);
      expect(storedAfter.state.currentTrick).toEqual([]);
      expect(storedAfter.state.currentPlayerId).toBe("player-0");
    }
  });

  it("does not persist partial state when AI advancement fails after next-trick", async () => {
    const state = createCompletedTrickWonByAi();
    games.set("ai-lead-failure", {
      state,
      humanPlayerId: "player-0",
      agents: new Map([
        ["player-1", new ThrowingAgent()],
        ...createAgents(["player-2", "player-3", "player-4"])
      ])
    });
    const snapshot = createStateSnapshot(state);

    const response = await app.inject({
      method: "POST",
      url: "/api/games/ai-lead-failure/next-trick"
    });
    const storedAfter = games.get("ai-lead-failure");

    expect(response.statusCode).toBeGreaterThanOrEqual(500);
    expect(storedAfter).toBeDefined();
    if (storedAfter !== undefined) {
      expect(createStateSnapshot(storedAfter.state)).toEqual(snapshot);
      expect(storedAfter.state.currentTrick).toHaveLength(5);
      expect(storedAfter.state.isTrickComplete).toBe(true);
      expect(storedAfter.state.trickNumber).toBe(1);
      expect(storedAfter.state.currentPlayerId).toBe("player-1");
    }
  });
});

async function createGame(): Promise<CreateGameResponse> {
  const response = await app.inject({
    method: "POST",
    url: "/api/games",
    payload: {}
  });

  expect(response.statusCode).toBe(201);
  return response.json<CreateGameResponse>();
}

function createCompletedTrickWonByAi(): GameState {
  return [
    { playerId: "player-0", cardId: "hearts-A" },
    { playerId: "player-1", cardId: "spades-2" },
    { playerId: "player-2", cardId: "hearts-K" },
    { playerId: "player-3", cardId: "clubs-A" },
    { playerId: "player-4", cardId: "diamonds-A" }
  ].reduce(
    (state, action) =>
      applyAction(state, {
        type: "play-card",
        playerId: action.playerId,
        cardId: action.cardId
      }),
    createStateWithHands([
      [card("hearts", "A"), card("clubs", "6")],
      [card("spades", "2"), card("clubs", "2")],
      [card("hearts", "K"), card("clubs", "3")],
      [card("clubs", "A"), card("diamonds", "4")],
      [card("diamonds", "A"), card("spades", "5")]
    ])
  );
}

function createCompletedTrickWonByHuman(): GameState {
  return [
    { playerId: "player-0", cardId: "hearts-A" },
    { playerId: "player-1", cardId: "hearts-K" },
    { playerId: "player-2", cardId: "clubs-A" },
    { playerId: "player-3", cardId: "hearts-3" },
    { playerId: "player-4", cardId: "clubs-Q" }
  ].reduce(
    (state, action) =>
      applyAction(state, {
        type: "play-card",
        playerId: action.playerId,
        cardId: action.cardId
      }),
    createStateWithHands([
      [card("hearts", "A"), card("clubs", "6")],
      [card("hearts", "K"), card("clubs", "2")],
      [card("clubs", "A"), card("clubs", "3")],
      [card("hearts", "3"), card("diamonds", "4")],
      [card("clubs", "Q"), card("spades", "5")]
    ])
  );
}

function createFollowSuitViolationState(): GameState {
  return {
    ...createStateWithHands([
      [card("hearts", "2"), card("spades", "A")],
      [card("hearts", "K")],
      [card("clubs", "3")],
      [card("diamonds", "4")],
      [card("clubs", "5")]
    ]),
    currentPlayerId: "player-0",
    currentTrick: [{ playerId: "player-1", card: card("hearts", "J") }]
  };
}

function createStateSnapshot(state: GameState) {
  return {
    phase: state.phase,
    humanHandCount: state.players[0].hand.length,
    currentTrick: structuredClone(state.currentTrick),
    currentPlayerId: state.currentPlayerId,
    completedTricks: structuredClone(state.completedTricks),
    trumpSuit: state.trumpSuit,
    contract: structuredClone(state.contract),
    adjutant: structuredClone(state.adjutant),
    result: structuredClone(state.result),
    bidding: structuredClone(state.bidding),
    awardedPointCards: structuredClone(state.awardedPointCards),
    excludedCards: structuredClone(state.excludedCards),
    latestEvent: structuredClone(state.latestEvent),
    unusedCards: structuredClone(state.unusedCards),
    trickNumber: state.trickNumber
  };
}

function hasOwn(value: object, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
}

function card(suit: Suit, rank: Rank): Card {
  return {
    type: "standard",
    id: `${suit}-${rank}`,
    suit,
    rank
  };
}

function joker(): Card {
  return {
    type: "joker",
    id: "joker"
  };
}

function createStateWithHands(
  hands: readonly [
    readonly Card[],
    readonly Card[],
    readonly Card[],
    readonly Card[],
    readonly Card[]
  ]
): GameState {
  const playerIds = ["player-0", "player-1", "player-2", "player-3", "player-4"];

  return {
    players: playerIds.map((id, index) => ({
      id,
      hand: hands[index]
    })),
    phase: "playing",
    currentPlayerId: "player-0",
    currentTrick: [],
    completedTricks: [],
    trumpSuit: "spades",
    contract: {
      napoleonPlayerId: "player-0",
      trumpSuit: "spades",
      targetPointCards: 13
    },
    adjutant: null,
    bidding: null,
    awardedPointCards: [],
    excludedCards: [],
    latestEvent: null,
    result: null,
    trickNumber: 1,
    isTrickComplete: false,
    isGameOver: false,
    unusedCards: []
  };
}

function createFinalTrickScoringState(): GameState {
  return {
    ...createStateWithHands([
      [card("spades", "A")],
      [card("spades", "K")],
      [card("spades", "Q")],
      [card("spades", "J")],
      [card("spades", "10")]
    ]),
    completedTricks: createCompletedTricksFromPointCounts([5, 5, 5, 0, 0, 0, 0, 0, 0]),
    awardedPointCards: [],
    excludedCards: [card("clubs", "2"), card("clubs", "3"), joker()],
    adjutant: {
      calledCardId: "hearts-A",
      playerId: "player-1",
      revealed: false
    },
    trickNumber: 10
  };
}

function createCompletedTricksFromPointCounts(
  pointCounts: readonly number[]
): GameState["completedTricks"] {
  return pointCounts.map((pointCount, index) =>
    createCompletedTrickWithPointCount(index + 1, pointCount, index * 5)
  );
}

function createCompletedTrickWithPointCount(
  trickNumber: number,
  pointCount: number,
  cardOffset: number
): GameState["completedTricks"][number] {
  const pointCards = createDeck().filter(isPointCard);
  const nonPointCards = createDeck().filter((candidate) => !isPointCard(candidate));
  const cards = [
    ...Array.from(
      { length: pointCount },
      (_, index) => pointCards[(cardOffset + index) % pointCards.length]
    ),
    ...Array.from(
      { length: 5 - pointCount },
      (_, index) => nonPointCards[(cardOffset + index) % nonPointCards.length]
    )
  ];
  const playerIds = ["player-0", "player-1", "player-2", "player-3", "player-4"];

  return {
    trickNumber,
    winnerId: "player-0",
    cards: playerIds.map((playerId, index) => ({
      playerId,
      card: cards[index]
    }))
  };
}

function createAllPassExchangeState(): GameState {
  const choosing = createAllPassAdjutantChoiceState();

  return applyAction(choosing, {
    type: "choose-adjutant",
    playerId: choosing.currentPlayerId,
    cardId: "spades-A"
  });
}

function createAllPassAdjutantChoiceState(): GameState {
  return Array.from({ length: 5 }).reduce<GameState>(
    (state) => applyAction(state, { type: "pass", playerId: state.currentPlayerId }),
    createInitialGame({ rng: () => 0 })
  );
}
