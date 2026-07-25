import { RandomAgent, type Agent } from "@napoleon/ai";
import type { GameState, PlayerId } from "@napoleon/game-core";
import { randomUUID } from "node:crypto";

export interface InternalGameState {
  state: GameState;
  humanPlayerId: PlayerId;
  agents: ReadonlyMap<PlayerId, Agent>;
}

export const games = new Map<string, InternalGameState>();

export function createGameId(): string {
  return randomUUID();
}

export function createAgents(playerIds: readonly PlayerId[]): ReadonlyMap<PlayerId, Agent> {
  return new Map(playerIds.map((playerId) => [playerId, new RandomAgent()]));
}
