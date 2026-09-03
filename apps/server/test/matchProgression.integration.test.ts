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

const expectedStarterRotation = ["player-0", "player-1", "player-2", "player-3", "player-4"];

describe("match progression API", () => {
  it("reviews rounds 1-4, resets each next game, and completes after round 5", async () => {
    const createdResponse = await app.inject({ method: "POST", url: "/api/games", payload: {} });
    let session = createdResponse.json<CreateGameResponse>();

    expect(session.match).toMatchObject({ currentRound: 1, roundCount: 5, completed: false });
    // Round 1's starter is the human seat, matching the pre-rotation default.
    expect(session.state.bidding?.starterPlayerId).toBe(expectedStarterRotation[0]);

    // Each all-pass round pays +1 to that round's starter and -1 to everyone else,
    // so tracking the starter rotation directly gives player-0's expected scores.
    const expectedPlayerZeroScores: number[] = [];

    for (let round = 1; round <= 5; round += 1) {
      expectedPlayerZeroScores.push(
        expectedStarterRotation[round - 1] === "player-0" ? 1 : -1
      );

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
        roundScores: expectedPlayerZeroScores,
        rawMatchScore: expectedPlayerZeroScores.reduce((sum, score) => sum + score, 0)
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
        // The starter rotates to the next player each round.
        expect(advanced.state.bidding?.starterPlayerId).toBe(expectedStarterRotation[round]);
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

  it("rotates the starter through all five players exactly once before completing", async () => {
    const createdResponse = await app.inject({ method: "POST", url: "/api/games", payload: {} });
    let session = createdResponse.json<CreateGameResponse>();

    const starters = [session.state.bidding?.starterPlayerId];

    for (let round = 1; round <= 5; round += 1) {
      await app.inject({
        method: "POST",
        url: `/api/games/${session.gameId}/actions`,
        payload: { action: { type: "pass" } }
      });
      const advancedResponse = await app.inject({
        method: "POST",
        url: `/api/games/${session.gameId}/next-round`
      });
      const advanced = advancedResponse.json<AdvanceMatchResponse>();

      if (round < 5) {
        starters.push(advanced.state.bidding?.starterPlayerId);
        expect(advanced.match.completed).toBe(false);
      } else {
        // The fifth (and final) round's starter has already been recorded from
        // the previous iteration's next-round response; the match is now over.
        expect(advanced.match.completed).toBe(true);
      }

      session = advanced;
    }

    expect(starters).toEqual(expectedStarterRotation);
    expect(new Set(starters).size).toBe(5);
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
