import { RuleBasedAgent, type Agent, type PublicActionRecord } from "@napoleon/ai";
import type { GameState, PlayerId } from "@napoleon/game-core";
import type {
  AiPolicyComposition,
  AiPresetId,
  CreateGameAgentSelection,
  PublicAiPhaseCallDiagnostics
} from "@napoleon/protocol";
import { randomUUID } from "node:crypto";
import { RULE_BASED_AGENT_ID, type AgentRegistry } from "./agentRegistry.js";

export interface InternalGameState {
  state: GameState;
  humanPlayerId: PlayerId;
  agents: ReadonlyMap<PlayerId, Agent>;
  agentIds?: ReadonlyMap<PlayerId, string>;
  policyDiagnostics?: ReadonlyMap<PlayerId, PublicAiPhaseCallDiagnostics>;
  aiPresetId?: AiPresetId;
  publicActionHistory?: readonly PublicActionRecord[];
}

export const games = new Map<string, InternalGameState>();

export function createGameId(): string {
  return randomUUID();
}

export interface AgentConfiguration {
  agents: ReadonlyMap<PlayerId, Agent>;
  agentIds: ReadonlyMap<PlayerId, string>;
  policyDiagnostics: ReadonlyMap<PlayerId, PublicAiPhaseCallDiagnostics>;
}

export class InvalidAgentSelectionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "InvalidAgentSelectionError";
  }
}

export function createAgentConfiguration(
  playerIds: readonly PlayerId[],
  registry: AgentRegistry,
  selections: readonly CreateGameAgentSelection[] = []
): AgentConfiguration {
  const playerIdSet = new Set(playerIds);
  const selectedPlayerIds = new Set<string>();
  const agentIds = new Map<PlayerId, string>(
    playerIds.map((playerId) => [playerId, RULE_BASED_AGENT_ID])
  );
  const compositions = new Map<PlayerId, AiPolicyComposition>();

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
    if ("agentId" in selection) {
      agentIds.set(selection.playerId, selection.agentId);
    } else {
      compositions.set(selection.playerId, selection.policyComposition);
      agentIds.set(selection.playerId, compositionLabel(selection.policyComposition));
    }
  }

  const policyDiagnostics = new Map<PlayerId, PublicAiPhaseCallDiagnostics>();

  return {
    agentIds,
    policyDiagnostics,
    agents: new Map(
      playerIds.map((playerId) => {
        const composition = compositions.get(playerId);
        if (composition === undefined) {
          return [
            playerId,
            registry.createAgent(agentIds.get(playerId) ?? RULE_BASED_AGENT_ID)
          ];
        }
        const diagnostics = registry.createCompositionDiagnostics(composition);
        policyDiagnostics.set(playerId, diagnostics);
        return [playerId, registry.createComposedAgent(composition, diagnostics)];
      })
    )
  };
}

function compositionLabel(composition: AiPolicyComposition): string {
  return `composition:${composition.playing}/${composition.bidding}/${composition.nonPlaying}`;
}

export function createAgents(
  playerIds: readonly PlayerId[]
): ReadonlyMap<PlayerId, Agent> {
  return new Map(playerIds.map((playerId) => [playerId, new RuleBasedAgent()]));
}
