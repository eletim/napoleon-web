import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { FastifyInstance } from "fastify";
import {
  advanceToNextTrick,
  applyAction,
  createInitialGame,
  getLegalActions,
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
    expect(body.state.currentTrick).toEqual([]);
    expect(body.state.currentPlayerId).toBe("player-1");
    expect(body.state.trickNumber).toBe(2);
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

function hasOwn(value: object, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
}
