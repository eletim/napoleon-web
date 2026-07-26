import type { FastifyInstance, FastifyReply } from "fastify";
import { NoLegalActionsError } from "@napoleon/ai";
import {
  advanceToNextTrick,
  applyAction,
  createInitialGame,
  createPlayerView,
  GameRuleError,
  getLegalActions
} from "@napoleon/game-core";
import type { GameAction, GameState, PlayerId } from "@napoleon/game-core";
import type { Agent } from "@napoleon/ai";
import type {
  CreateGameResponse,
  GetGameResponse,
  NextTrickResponse,
  PublicGameAction,
  SendActionResponse
} from "@napoleon/protocol";
import { createAgents, createGameId, games, type InternalGameState } from "./store.js";
import { toPublicGameState } from "./publicState.js";
import { readActionBody } from "./validation.js";

interface GameParams {
  gameId: string;
}

const humanPlayerId = "player-0";
const maxAutomaticAiActions = 100;

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
        state: toPublicGameState(record.state, record.humanPlayerId)
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

      if (action === undefined) {
        return sendError(reply, 400, "INVALID_ACTION", "A valid action is required.");
      }

      try {
        let nextState = applyAction(record.state, toInternalAction(action, record.humanPlayerId));
        nextState = await advanceAiTurns(nextState, record.humanPlayerId, record.agents);
        record.state = nextState;
      } catch (error) {
        return handleActionError(reply, error);
      }

      return {
        gameId: request.params.gameId,
        playerId: record.humanPlayerId,
        state: toPublicGameState(record.state, record.humanPlayerId)
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
        let nextState = advanceToNextTrick(record.state);
        nextState = await advanceAiTurns(nextState, record.humanPlayerId, record.agents);
        record.state = nextState;
      } catch (error) {
        return handleActionError(reply, error);
      }

      return {
        gameId: request.params.gameId,
        playerId: record.humanPlayerId,
        state: toPublicGameState(record.state, record.humanPlayerId)
      };
    }
  );
}

async function advanceAiTurns(
  initialState: GameState,
  humanPlayerId: PlayerId,
  agents: ReadonlyMap<PlayerId, Agent>
): Promise<GameState> {
  let state = initialState;
  let guard = 0;

  while (
    !state.isGameOver &&
    !state.isTrickComplete &&
    state.currentPlayerId !== humanPlayerId
  ) {
    guard += 1;

    if (guard > maxAutomaticAiActions) {
      throw new Error("AI turn guard exceeded.");
    }

    const playerId = state.currentPlayerId;
    const agent = agents.get(playerId);

    if (agent === undefined) {
      throw new Error(`No AI agent registered for ${playerId}.`);
    }

    const legalActions = getLegalActions(state, playerId);
    const view = createPlayerView(state, playerId);
    const action = await agent.selectAction({ playerId, view, legalActions });
    state = applyAction(state, action);
  }

  return state;
}

function toInternalAction(action: PublicGameAction, playerId: PlayerId): GameAction {
  switch (action.type) {
    case "play-card":
      return {
        type: "play-card",
        playerId,
        cardId: action.cardId
      };
    case "bid":
      return {
        type: "bid",
        playerId,
        suit: action.suit,
        targetPointCards: action.targetPointCards
      };
    case "pass":
      return {
        type: "pass",
        playerId
      };
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
    state: toPublicGameState(record.state, record.humanPlayerId)
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
