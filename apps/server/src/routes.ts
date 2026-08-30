import type { FastifyInstance, FastifyReply } from "fastify";
import { NoLegalActionsError, RuleBasedAgent, runAutomatedGame } from "@napoleon/ai";
import {
  advanceToNextTrick,
  applyAction,
  clearLatestEvent,
  completeCurrentRound,
  createInitialMatch,
  createPlayerView,
  GameRuleError,
  getLegalActions,
  MatchRuleError,
  updateCurrentGame
} from "@napoleon/game-core";
import type { GameAction, GameState, PlayerId } from "@napoleon/game-core";
import type { Agent, PublicActionRecord } from "@napoleon/ai";
import type {
  AiPreset,
  AdvanceMatchResponse,
  CreateGameResponse,
  GetAiPresetsResponse,
  GetAgentsResponse,
  GetGameResponse,
  GetGamePolicyDiagnosticsResponse,
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
  InvalidAiPresetCompositionError,
  UnknownAiPresetIdError,
  createAiPresetRegistry,
  type AiPresetRegistry
} from "./aiPresetRegistry.js";
import {
  InvalidAgentSelectionError,
  createAgentConfiguration,
  createGameId,
  games,
  type InternalGameState
} from "./store.js";
import { toPublicGameState } from "./publicState.js";
import { toPublicMatchState } from "./publicMatch.js";
import { toPublicSimulationResponse } from "./simulationResponse.js";
import {
  readActionBody,
  readCreateGameBody,
  readRunAutomatedSimulationBody,
  readUpdateAiPresetBody
} from "./validation.js";

interface GameParams {
  gameId: string;
}

interface AiPresetParams {
  presetId: string;
}

const humanPlayerId = "player-0";
const maxAutomaticAiActions = 100;

export interface RegisterRoutesOptions {
  agentRegistry?: AgentRegistry;
  aiPresetRegistry?: AiPresetRegistry;
}

export async function registerRoutes(
  app: FastifyInstance,
  options: RegisterRoutesOptions = {}
): Promise<void> {
  const agentRegistry = options.agentRegistry ?? createAgentRegistryFromEnvironment();
  await agentRegistry.initializePhasePolicies();
  const aiPresetRegistry = options.aiPresetRegistry ?? createAiPresetRegistry(agentRegistry);

  app.get("/api/health", async () => ({ ok: true }));

  app.get("/api/agents", async (): Promise<GetAgentsResponse> => ({
    agents: agentRegistry.listAgents(),
    policyRegistry: agentRegistry.listPhasePolicies()
  }));

  app.get("/api/ai-presets", async (): Promise<GetAiPresetsResponse> => ({
    presets: aiPresetRegistry.list(),
    policyRegistry: agentRegistry.listPhasePolicies()
  }));

  app.put<{ Params: AiPresetParams }>(
    "/api/ai-presets/:presetId",
    async (request, reply): Promise<AiPreset | FastifyReply> => {
      const body = readUpdateAiPresetBody(request.body);
      if (body === undefined) {
        return sendError(
          reply,
          400,
          "INVALID_AI_PRESET_REQUEST",
          "A valid AI preset composition is required."
        );
      }
      try {
        return aiPresetRegistry.update(request.params.presetId, body.composition);
      } catch (error) {
        return handleAiPresetError(reply, error);
      }
    }
  );

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

    const match = createInitialMatch();
    const state = match.currentGame;
    if (state === null) {
      throw new Error("A new match did not contain an initial game.");
    }
    const aiPlayerIds = state.players
      .map((player) => player.id)
      .filter((playerId) => playerId !== humanPlayerId);
    const gameId = createGameId();
    let agentConfiguration;
    let selectedPreset: AiPreset | undefined;

    try {
      selectedPreset = body.aiPresetId === undefined
        ? undefined
        : aiPresetRegistry.resolve(body.aiPresetId);
      const selectedComposition = selectedPreset?.composition;
      agentConfiguration = createAgentConfiguration(
        aiPlayerIds,
        agentRegistry,
        selectedComposition === undefined
          ? body.aiAgents ?? []
          : aiPlayerIds.map((playerId) => ({
              playerId,
              policyComposition: selectedComposition
            }))
      );
    } catch (error) {
      return handleCreateGameError(reply, error);
    }

    games.set(gameId, {
      state,
      match,
      humanPlayerId,
      agents: agentConfiguration.agents,
      agentIds: agentConfiguration.agentIds,
      policyDiagnostics: agentConfiguration.policyDiagnostics,
      ...(selectedPreset === undefined ? {} : { aiPresetId: selectedPreset.id }),
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

      let simulationComposition = body.policyComposition;
      let simulationPreset: AiPreset | undefined;
      try {
        simulationPreset = body.aiPresetId === undefined
          ? undefined
          : aiPresetRegistry.resolve(body.aiPresetId);
        simulationComposition ??= simulationPreset?.composition;
      } catch (error) {
        return handleAiPresetError(reply, error);
      }
      const diagnostics = new Map();
      try {
        const record = await runAutomatedGame({
          seed: body.seed,
          createAgent: ({ rng, playerId }) => {
            if (simulationComposition === undefined) {
              return new RuleBasedAgent(rng);
            }
            const phaseDiagnostics = agentRegistry.createCompositionDiagnostics(
              simulationComposition
            );
            diagnostics.set(playerId, phaseDiagnostics);
            return agentRegistry.createComposedAgent(
              simulationComposition,
              phaseDiagnostics,
              rng
            );
          }
        });

        return {
          ...toPublicSimulationResponse(record),
          ...(simulationPreset === undefined ? {} : { presetId: simulationPreset.id }),
          ...(simulationComposition === undefined
            ? {}
            : { policyDiagnostics: Object.fromEntries(diagnostics) })
        };
      } catch (error) {
        if (error instanceof AgentUnavailableError) {
          return sendError(reply, 503, "AGENT_UNAVAILABLE", error.message);
        }
        throw error;
      }
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
        state: toPublicGameState(record.state, record.humanPlayerId),
        ...publicMatchProperty(record)
      };
    }
  );

  app.get<{ Params: GameParams }>(
    "/api/games/:gameId/diagnostics",
    async (request, reply): Promise<GetGamePolicyDiagnosticsResponse | FastifyReply> => {
      const record = games.get(request.params.gameId);
      if (record === undefined) {
        return sendError(reply, 404, "GAME_NOT_FOUND", "Game was not found.");
      }
      return {
        gameId: request.params.gameId,
        ...(record.aiPresetId === undefined ? {} : { presetId: record.aiPresetId }),
        diagnostics: Object.fromEntries(record.policyDiagnostics ?? [])
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
        syncMatchGame(record);
        record.publicActionHistory = advanced.publicActionHistory;
      } catch (error) {
        return handleActionError(reply, error);
      }

      return {
        gameId: request.params.gameId,
        playerId: record.humanPlayerId,
        state: toPublicGameState(record.state, record.humanPlayerId),
        ...publicMatchProperty(record)
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
        syncMatchGame(record);
        record.publicActionHistory = advanced.publicActionHistory;
      } catch (error) {
        return handleActionError(reply, error);
      }

      return {
        gameId: request.params.gameId,
        playerId: record.humanPlayerId,
        state: toPublicGameState(record.state, record.humanPlayerId),
        ...publicMatchProperty(record)
      };
    }
  );

  app.post<{ Params: GameParams }>(
    "/api/games/:gameId/next-round",
    async (request, reply): Promise<AdvanceMatchResponse | FastifyReply> => {
      const record = games.get(request.params.gameId);
      if (record === undefined) {
        return sendError(reply, 404, "GAME_NOT_FOUND", "Game was not found.");
      }
      if (record.match === undefined) {
        return sendError(reply, 409, "MATCH_NOT_AVAILABLE", "This is a standalone game.");
      }

      try {
        record.match = completeCurrentRound(updateCurrentGame(record.match, record.state));
        if (record.match.currentGame !== null) {
          record.state = record.match.currentGame;
        }
        record.publicActionHistory = [];
      } catch (error) {
        return handleActionError(reply, error);
      }

      return {
        gameId: request.params.gameId,
        playerId: record.humanPlayerId,
        state: toPublicGameState(record.state, record.humanPlayerId),
        match: toPublicMatchState(
          record.match,
          record.state.players.map(({ id }) => id)
        )
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
    state: toPublicGameState(record.state, record.humanPlayerId),
    ...publicMatchProperty(record)
  };
}

function syncMatchGame(record: InternalGameState): void {
  if (record.match !== undefined) {
    record.match = updateCurrentGame(record.match, record.state);
  }
}

function publicMatchProperty(
  record: InternalGameState
): { match?: ReturnType<typeof toPublicMatchState> } {
  return record.match === undefined
    ? {}
    : {
        match: toPublicMatchState(
          record.match,
          record.state.players.map(({ id }) => id)
        )
      };
}

function handleActionError(reply: FastifyReply, error: unknown): FastifyReply {
  if (error instanceof GameRuleError || error instanceof MatchRuleError) {
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
  if (
    error instanceof UnknownAiPresetIdError ||
    error instanceof InvalidAiPresetCompositionError
  ) {
    return handleAiPresetError(reply, error);
  }
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

function handleAiPresetError(reply: FastifyReply, error: unknown): FastifyReply {
  if (error instanceof UnknownAiPresetIdError) {
    return sendError(reply, 400, "UNKNOWN_AI_PRESET_ID", error.message);
  }
  if (error instanceof InvalidAiPresetCompositionError) {
    return sendError(reply, 400, "INVALID_AI_PRESET_COMPOSITION", error.message);
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
