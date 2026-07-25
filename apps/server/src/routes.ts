import type { FastifyInstance, FastifyReply } from "fastify";
import { NoLegalActionsError } from "@napoleon/ai";
import {
  applyAction,
  createInitialGame,
  createPlayerView,
  GameRuleError,
  getLegalActions
} from "@napoleon/game-core";
import type {
  CreateGameResponse,
  GetGameResponse,
  NextTrickResponse,
  SendActionResponse
} from "@napoleon/protocol";
import { createAgents, createGameId, games, type InternalGameState } from "./store.js";
import { readActionBody } from "./validation.js";

interface GameParams {
  gameId: string;
}

const humanPlayerId = "player-0";

export async function registerRoutes(app: FastifyInstance): Promise<void> {
  app.get("/api/health", async () => ({ ok: true }));

  app.post("/api/games", async (_request, reply): Promise<CreateGameResponse> => {
    const state = createInitialGame();
    const aiPlayerIds = state.players
      .map((player) => player.id)
      .filter((playerId) => playerId !== humanPlayerId);
    const gameId = createGameId();

    games.set(gameId, {
      state,
      humanPlayerId,
      agents: createAgents(aiPlayerIds)
    });

    reply.code(201);
    return createGameResponse(gameId, games.get(gameId));
  });

  app.get<{ Params: GameParams }>(
    "/api/games/:gameId",
    async (request, reply): Promise<GetGameResponse | FastifyReply> => {
      const record = games.get(request.params.gameId);

      if (record === undefined) {
        return sendError(reply, 404, "GAME_NOT_FOUND", "Game was not found.");
      }

      return {
        gameId: request.params.gameId,
        playerId: record.humanPlayerId,
        state: createPlayerView(record.state, record.humanPlayerId)
      };
    }
  );

  app.post<{ Params: GameParams }>(
    "/api/games/:gameId/actions",
    async (request, reply): Promise<SendActionResponse | FastifyReply> => {
      const record = games.get(request.params.gameId);

      if (record === undefined) {
        return sendError(reply, 404, "GAME_NOT_FOUND", "Game was not found.");
      }

      const action = readActionBody(request.body);

      if (action === undefined || action.type !== "play-card") {
        return sendError(reply, 400, "INVALID_ACTION", "A play-card action is required.");
      }

      if (action.playerId !== record.humanPlayerId) {
        return sendError(reply, 403, "FORBIDDEN_ACTION", "Only the human player can submit actions.");
      }

      try {
        record.state = applyAction(record.state, action);
        await advanceAiTurns(record);
      } catch (error) {
        return handleActionError(reply, error);
      }

      return {
        gameId: request.params.gameId,
        playerId: record.humanPlayerId,
        state: createPlayerView(record.state, record.humanPlayerId)
      };
    }
  );

  app.post<{ Params: GameParams }>(
    "/api/games/:gameId/next-trick",
    async (request, reply): Promise<NextTrickResponse | FastifyReply> => {
      const record = games.get(request.params.gameId);

      if (record === undefined) {
        return sendError(reply, 404, "GAME_NOT_FOUND", "Game was not found.");
      }

      try {
        record.state = applyAction(record.state, {
          type: "next-trick",
          playerId: record.humanPlayerId
        });
      } catch (error) {
        return handleActionError(reply, error);
      }

      return {
        gameId: request.params.gameId,
        playerId: record.humanPlayerId,
        state: createPlayerView(record.state, record.humanPlayerId)
      };
    }
  );
}

async function advanceAiTurns(record: InternalGameState): Promise<void> {
  let guard = 0;

  while (
    !record.state.isGameOver &&
    !record.state.isTrickComplete &&
    record.state.currentPlayerId !== record.humanPlayerId
  ) {
    guard += 1;

    if (guard > 20) {
      throw new Error("AI turn guard exceeded.");
    }

    const playerId = record.state.currentPlayerId;
    const agent = record.agents.get(playerId);

    if (agent === undefined) {
      throw new Error(`No AI agent registered for ${playerId}.`);
    }

    const legalActions = getLegalActions(record.state, playerId);
    const view = createPlayerView(record.state, playerId);
    const action = await agent.selectAction({ playerId, view, legalActions });
    record.state = applyAction(record.state, action);
  }
}

function createGameResponse(
  gameId: string,
  record: InternalGameState | undefined
): CreateGameResponse {
  if (record === undefined) {
    throw new Error("Game creation failed.");
  }

  return {
    gameId,
    playerId: record.humanPlayerId,
    state: createPlayerView(record.state, record.humanPlayerId)
  };
}

function handleActionError(reply: FastifyReply, error: unknown): FastifyReply {
  if (error instanceof GameRuleError) {
    return sendError(reply, 400, error.code, error.message);
  }

  if (error instanceof NoLegalActionsError) {
    return sendError(reply, 409, "NO_LEGAL_ACTIONS", error.message);
  }

  throw error;
}

function sendError(
  reply: FastifyReply,
  statusCode: number,
  code: string,
  message: string
): FastifyReply {
  return reply.code(statusCode).send({
    error: {
      code,
      message
    }
  });
}
