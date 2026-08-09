import type { FastifyInstance, FastifyReply } from "fastify";
import { NoLegalActionsError, RuleBasedAgent, runAutomatedGame } from "@napoleon/ai";
import {
  advanceToNextTrick,
  applyAction,
  clearLatestEvent,
  createInitialGame,
  createPlayerView,
  GameRuleError,
  getLegalActions
} from "@napoleon/game-core";
import type { GameAction, GameState, PlayerId } from "@napoleon/game-core";
import type { Agent, PublicActionRecord } from "@napoleon/ai";
import type {
  CreateGameResponse,
  GetAgentsResponse,
  GetGameResponse,
  NextTrickResponse,
  PublicGameAction,
  RunAutomatedSimulationResponse,
  SendActionResponse
} from "@napoleon/protocol";
import {
  AgentUnavailableError,
  UnknownAgentIdError,
  createAgentRegistryFromEnvironment,
  type AgentRegistry
} from "./agentRegistry.js";
import {
  InvalidAgentSelectionError,
  createAgentConfiguration,
  createGameId,
  games,
  type InternalGameState
} from "./store.js";
import { toPublicGameState } from "./publicState.js";
import { toPublicSimulationResponse } from "./simulationResponse.js";
import {
  readActionBody,
  readCreateGameBody,
  readRunAutomatedSimulationBody
} from "./validation.js";

interface GameParams {
  gameId: string;
}

const humanPlayerId = "player-0";
const maxAutomaticAiActions = 100;

export interface RegisterRoutesOptions {
  agentRegistry?: AgentRegistry;
}

export async function registerRoutes(
  app: FastifyInstance,
  options: RegisterRoutesOptions = {}
): Promise<void> {
  const agentRegistry = options.agentRegistry ?? createAgentRegistryFromEnvironment();

  app.get("/api/health", async () => ({ ok: true }));

  app.get("/api/agents", async (): Promise<GetAgentsResponse> => ({
    agents: agentRegistry.listAgents()
  }));

  app.post("/api/games", async (request, reply): Promise<CreateGameResponse | FastifyReply> => {
    const body = readCreateGameBody(request.body);

    if (body === undefined) {
      return sendError(
        reply,
        400,
        "INVALID_CREATE_GAME_REQUEST",
        "A valid game creation request is required."
      );
    }

    const state = createInitialGame();
    const aiPlayerIds = state.players
      .map((player) => player.id)
      .filter((playerId) => playerId !== humanPlayerId);
    const gameId = createGameId();
    let agentConfiguration;

    try {
      agentConfiguration = createAgentConfiguration(
        aiPlayerIds,
        agentRegistry,
        body.aiAgents ?? []
      );
    } catch (error) {
      return handleCreateGameError(reply, error);
    }

    games.set(gameId, {
      state,
      humanPlayerId,
      agents: agentConfiguration.agents,
      agentIds: agentConfiguration.agentIds,
      publicActionHistory: []
    });

    reply.code(201);
    return createGameResponse(gameId, games.get(gameId));
  });

  app.post(
    "/api/simulations",
    async (request, reply): Promise<RunAutomatedSimulationResponse | FastifyReply> => {
      const body = readRunAutomatedSimulationBody(request.body);

      if (body === undefined) {
        return sendError(
          reply,
          400,
          "INVALID_SIMULATION_REQUEST",
          "seed must be an integer between 0 and 4294967295."
        );
      }

      const record = await runAutomatedGame({
        seed: body.seed,
        createAgent: ({ rng }) => new RuleBasedAgent(rng)
      });

      return toPublicSimulationResponse(record);
    }
  );

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
        const internalAction = toInternalAction(action, record.humanPlayerId);
        let publicActionHistory = appendPublicBiddingAction(
          record.publicActionHistory ?? [],
          internalAction
        );
        let nextState = applyAction(
          clearLatestEvent(record.state),
          internalAction
        );
        const advanced = await advanceAiTurns(
          nextState,
          record.humanPlayerId,
          record.agents,
          publicActionHistory
        );
        record.state = advanced.state;
        record.publicActionHistory = advanced.publicActionHistory;
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
        const nextState = advanceToNextTrick(clearLatestEvent(record.state));
        const advanced = await advanceAiTurns(
          nextState,
          record.humanPlayerId,
          record.agents,
          record.publicActionHistory ?? []
        );
        record.state = advanced.state;
        record.publicActionHistory = advanced.publicActionHistory;
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
  agents: ReadonlyMap<PlayerId, Agent>,
  initialPublicActionHistory: readonly PublicActionRecord[]
): Promise<{
  state: GameState;
  publicActionHistory: readonly PublicActionRecord[];
}> {
  let state = initialState;
  let publicActionHistory = initialPublicActionHistory;
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
    const action = await agent.selectAction({
      playerId,
      view,
      legalActions,
      publicActionHistory
    });
    publicActionHistory = appendPublicBiddingAction(publicActionHistory, action);
    state = applyAction(state, action);
  }

  return {
    state,
    publicActionHistory
  };
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
    case "discard-cards":
      return {
        type: "discard-cards",
        playerId,
        cardIds: action.cardIds
      };
    case "choose-adjutant":
      return {
        type: "choose-adjutant",
        playerId,
        cardId: action.cardId
      };
  }
}

function appendPublicBiddingAction(
  history: readonly PublicActionRecord[],
  action: GameAction
): readonly PublicActionRecord[] {
  if (action.type !== "bid" && action.type !== "pass") {
    return history;
  }

  return [
    ...history,
    {
      step: history.length + 1,
      playerId: action.playerId,
      phase: "bidding",
      action
    }
  ];
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

  if (error instanceof AgentUnavailableError) {
    return sendError(reply, 503, "AGENT_UNAVAILABLE", error.message);
  }

  throw error;
}

function handleCreateGameError(reply: FastifyReply, error: unknown): FastifyReply {
  if (error instanceof UnknownAgentIdError) {
    return sendError(reply, 400, "UNKNOWN_AGENT_ID", error.message);
  }

  if (error instanceof InvalidAgentSelectionError) {
    return sendError(reply, 400, "INVALID_AGENT_SELECTION", error.message);
  }

  if (error instanceof AgentUnavailableError) {
    return sendError(reply, 503, "AGENT_UNAVAILABLE", error.message);
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
