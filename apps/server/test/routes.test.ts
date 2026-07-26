import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { FastifyInstance } from "fastify";
import type { Agent } from "@napoleon/ai";
import {
  advanceToNextTrick,
  applyAction,
  createInitialGame,
  getLegalActions,
  type GameAction,
  type GameState
} from "@napoleon/game-core";
import type {
  ApiError,
  CreateGameResponse,
  GetGameResponse,
  NextTrickResponse,
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

beforeEach(async () => {
  games.clear();
  app = await buildApp();
});

afterEach(async () => {
  games.clear();
  await app.close();
});

describe("server API", () => {
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
    expect(body.state.opponents.some((opponent) => hasOwn(opponent, "hand"))).toBe(false);
    expect(body.state.legalActions.length).toBeGreaterThan(0);
    expect(body.state.legalActions.some((action) => hasOwn(action, "playerId"))).toBe(false);
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
    expect(found.json<GetGameResponse>().gameId).toBe(created.gameId);
    expect(missing.statusCode).toBe(404);
  });

  it("plays a legal card and advances all four AI players", async () => {
    const created = await createGame();
    const card = created.state.self.hand[0];

    const response = await app.inject({
      method: "POST",
      url: `/api/games/${created.gameId}/actions`,
      payload: {
        action: {
          type: "play-card",
          cardId: card.id
        }
      }
    });
    const body = response.json<SendActionResponse>();

    expect(response.statusCode).toBe(200);
    expect(body.state.currentTrick).toHaveLength(5);
    expect(body.state.isTrickComplete).toBe(true);
    expect(body.state.self.handCount).toBe(9);
    expect(body.state.opponents.some((opponent) => hasOwn(opponent, "hand"))).toBe(false);
  });

  it("rejects an invalid card id", async () => {
    const created = await createGame();

    const response = await app.inject({
      method: "POST",
      url: `/api/games/${created.gameId}/actions`,
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
    const completed = await playOneFullTrick();

    const response = await app.inject({
      method: "POST",
      url: `/api/games/${completed.gameId}/next-trick`
    });
    const body = response.json<NextTrickResponse>();

    expect(response.statusCode).toBe(200);
    expect(body.state.currentTrick).toEqual([]);
    expect(body.state.trickNumber).toBe(2);
  });

  it("advances to the next trick even when the next lead player is an AI", async () => {
    const state = createCompletedTrickWithAiLead();
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
    const created = await createGame();
    const record = games.get(created.gameId);

    if (record === undefined) {
      throw new Error("Expected created game to be stored.");
    }

    games.set(created.gameId, {
      ...record,
      agents: new Map(record.agents).set("player-2", new ThrowingAgent())
    });
    const storedBefore = games.get(created.gameId);

    if (storedBefore === undefined) {
      throw new Error("Expected game to remain stored.");
    }

    const snapshot = createStateSnapshot(storedBefore.state);
    const response = await app.inject({
      method: "POST",
      url: `/api/games/${created.gameId}/actions`,
      payload: {
        action: {
          type: "play-card",
          cardId: created.state.self.hand[0].id
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

  it("does not persist partial state when AI advancement fails after next-trick", async () => {
    const state = createCompletedTrickWithAiLead();
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

async function playOneFullTrick(): Promise<SendActionResponse> {
  const created = await createGame();
  const card = created.state.self.hand[0];
  const response = await app.inject({
    method: "POST",
    url: `/api/games/${created.gameId}/actions`,
    payload: {
      action: {
        type: "play-card",
        cardId: card.id
      }
    }
  });

  expect(response.statusCode).toBe(200);
  return response.json<SendActionResponse>();
}

function createCompletedTrickWithAiLead(): GameState {
  const initial = createInitialGame({ rng: () => 0 });
  const completed = Array.from({ length: 5 }).reduce<GameState>((state) => {
    const action = getLegalActions(state, state.currentPlayerId)[0];
    return applyAction(state, action);
  }, initial);
  const advanced = advanceToNextTrick(completed);

  return {
    ...completed,
    currentPlayerId: "player-1",
    trickNumber: advanced.trickNumber - 1
  };
}

function createStateSnapshot(state: GameState) {
  return {
    humanHandCount: state.players[0].hand.length,
    currentTrick: structuredClone(state.currentTrick),
    currentPlayerId: state.currentPlayerId,
    completedTricks: structuredClone(state.completedTricks),
    trickNumber: state.trickNumber
  };
}

function hasOwn(value: object, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
}
