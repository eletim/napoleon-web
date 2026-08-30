import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { FastifyInstance } from "fastify";
import { RuleBasedAgent, type Agent } from "@napoleon/ai";
import type { GameAction } from "@napoleon/game-core";
import type {
  AdvanceMatchResponse,
  CreateGameResponse,
  SendActionResponse
} from "@napoleon/protocol";
import type { AgentRegistry } from "../src/agentRegistry.js";
import type { AiPresetRegistry } from "../src/aiPresetRegistry.js";
import { buildApp } from "../src/app.js";
import { games } from "../src/store.js";

let app: FastifyInstance;

class PassAgent extends RuleBasedAgent implements Agent {
  async selectAction(input: Parameters<Agent["selectAction"]>[0]): Promise<GameAction> {
    const pass = input.legalActions.find((action) => action.type === "pass");
    if (pass === undefined) {
      throw new Error("Expected all-pass bidding in this integration test.");
    }
    return pass;
  }
}

beforeEach(async () => {
  app = await buildApp({ agentRegistry: testAgentRegistry(), aiPresetRegistry: emptyPresetRegistry() });
});

afterEach(async () => {
  games.clear();
  await app.close();
});

describe("match progression API", () => {
  it("reviews rounds 1-4, resets each next game, and completes after round 5", async () => {
    const createdResponse = await app.inject({ method: "POST", url: "/api/games", payload: {} });
    let session = createdResponse.json<CreateGameResponse>();

    expect(session.match).toMatchObject({ currentRound: 1, roundCount: 5, completed: false });

    for (let round = 1; round <= 5; round += 1) {
      const finishedResponse = await app.inject({
        method: "POST",
        url: `/api/games/${session.gameId}/actions`,
        payload: { action: { type: "pass" } }
      });
      const finished = finishedResponse.json<SendActionResponse>();

      expect(finished.state.isGameOver).toBe(true);
      expect(finished.match?.completedRoundCount).toBe(round);
      expect(finished.match?.players.find(({ playerId }) => playerId === "player-0")).toEqual({
        playerId: "player-0",
        roundScores: Array.from({ length: round }, () => 1),
        rawMatchScore: round
      });

      const advancedResponse = await app.inject({
        method: "POST",
        url: `/api/games/${session.gameId}/next-round`
      });
      const advanced = advancedResponse.json<AdvanceMatchResponse>();
      expect(advancedResponse.statusCode).toBe(200);

      if (round < 5) {
        expect(advanced.match).toMatchObject({
          currentRound: round + 1,
          completedRoundCount: round,
          remainingRounds: 5 - round,
          completed: false,
          finalScores: null
        });
        expect(advanced.state.phase).toBe("bidding");
        expect(advanced.state.result).toBeNull();
        expect(advanced.state.currentTrick).toEqual([]);
        expect(advanced.state.completedTrickCount).toBe(0);
        expect(advanced.state.self.hand).toHaveLength(10);
      } else {
        expect(advanced.match).toMatchObject({
          currentRound: 5,
          completedRoundCount: 5,
          remainingRounds: 0,
          completed: true
        });
        expect(advanced.match.finalScores).not.toBeNull();
      }

      session = advanced;
    }
  });
});

function testAgentRegistry(): AgentRegistry {
  return {
    initializePhasePolicies: async () => undefined,
    listAgents: () => [{ id: "rule-based", displayName: "Rule-based AI", isAvailable: true }],
    listPhasePolicies: () => ({ playing: [], bidding: [], nonPlaying: [] }),
    createAgent: () => new PassAgent(),
    createComposedAgent: () => new PassAgent(),
    createCompositionDiagnostics: () => ({
      playing: { callCount: 0, fallbackCount: 0 },
      bidding: { callCount: 0, fallbackCount: 0 },
      nonPlaying: { callCount: 0, fallbackCount: 0 }
    })
  };
}

function emptyPresetRegistry(): AiPresetRegistry {
  return {
    list: () => [],
    resolve: () => { throw new Error("No test presets."); },
    update: () => { throw new Error("No test presets."); }
  };
}
