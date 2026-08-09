import type { Agent, PublicActionRecord } from "@napoleon/ai";
import type { GameState, PlayerId } from "@napoleon/game-core";
import type { CreateGameAgentSelection } from "@napoleon/protocol";
import { randomUUID } from "node:crypto";
import {
  RULE_BASED_AGENT_ID,
  createAgentRegistry,
  type AgentRegistry
} from "./agentRegistry.js";

export interface InternalGameState {
  state: GameState;
  humanPlayerId: PlayerId;
  agents: ReadonlyMap<PlayerId, Agent>;
  agentIds?: ReadonlyMap<PlayerId, string>;
  publicActionHistory?: readonly PublicActionRecord[];
}

export const games = new Map<string, InternalGameState>();

export function createGameId(): string {
  return randomUUID();
}

export interface AgentConfiguration {
  agents: ReadonlyMap<PlayerId, Agent>;
  agentIds: ReadonlyMap<PlayerId, string>;
}

export class InvalidAgentSelectionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "InvalidAgentSelectionError";
  }
}

const defaultAgentRegistry = createAgentRegistry();

export function createAgentConfiguration(
  playerIds: readonly PlayerId[],
  selections: readonly CreateGameAgentSelection[] = [],
  registry: AgentRegistry = defaultAgentRegistry
): AgentConfiguration {
  const playerIdSet = new Set(playerIds);
  const selectedPlayerIds = new Set<string>();
  const agentIds = new Map<PlayerId, string>(
    playerIds.map((playerId) => [playerId, RULE_BASED_AGENT_ID])
  );

  for (const selection of selections) {
    if (!playerIdSet.has(selection.playerId)) {
      throw new InvalidAgentSelectionError(
        `AI agent selection references unknown seat ${selection.playerId}.`
      );
    }

    if (selectedPlayerIds.has(selection.playerId)) {
      throw new InvalidAgentSelectionError(
        `AI agent selection repeats seat ${selection.playerId}.`
      );
    }

    selectedPlayerIds.add(selection.playerId);
    agentIds.set(selection.playerId, selection.agentId);
  }

  return {
    agentIds,
    agents: new Map(
      playerIds.map((playerId) => [
        playerId,
        registry.createAgent(agentIds.get(playerId) ?? RULE_BASED_AGENT_ID)
      ])
    )
  };
}

export function createAgents(
  playerIds: readonly PlayerId[],
  selections: readonly CreateGameAgentSelection[] = [],
  registry: AgentRegistry = defaultAgentRegistry
): ReadonlyMap<PlayerId, Agent> {
  return createAgentConfiguration(playerIds, selections, registry).agents;
}
