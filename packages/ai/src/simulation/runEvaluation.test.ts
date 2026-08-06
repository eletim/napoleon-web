import { describe, expect, it } from "vitest";
import type { Agent, PlayerObservation } from "../types.js";
import { RuleBasedAgent } from "../ruleBasedAgent.js";
import { runAutomatedGame } from "./runAutomatedGame.js";
import { runEvaluation } from "./runEvaluation.js";
import type { EvaluationAgentDefinition } from "./types.js";

const playerIds = ["player-0", "player-1", "player-2", "player-3", "player-4"] as const;

function createRuleBasedDefinitions(prefix = "RuleBasedAgent"): readonly EvaluationAgentDefinition[] {
  return playerIds.map((_, index) => ({
    name: `${prefix}-${index}`,
    createAgent: ({ rng }) => new RuleBasedAgent(rng)
  }));
}

describe("runEvaluation", () => {
  it("runs deterministic RuleBasedAgent smoke games across seeds and seat rotations", async () => {
    const options = {
      startSeed: 100,
      gameCount: 3,
      playerIds,
      rotationOffsets: [0, 1, 2, 3, 4],
      agents: createRuleBasedDefinitions()
    };

    const first = await runEvaluation(options);
    const second = await runEvaluation(options);

    expect(second).toEqual(first);
    expect(first.schemaVersion).toBe(1);
    expect(first.startSeed).toBe(100);
    expect(first.endSeed).toBe(102);
    expect(first.games).toHaveLength(15);
    expect(first.completedCount).toBe(15);
    expect(first.failedCount).toBe(0);
    expect(first.games.map((game) => game.seed)).toEqual([
      100, 100, 100, 100, 100,
      101, 101, 101, 101, 101,
      102, 102, 102, 102, 102
    ]);
    expect(first.games.map((game) => game.rotationOffset)).toEqual([
      0, 1, 2, 3, 4,
      0, 1, 2, 3, 4,
      0, 1, 2, 3, 4
    ]);

    for (const game of first.games) {
      expect("result" in game || "failureReason" in game).toBe(true);
      expect("result" in game && "failureReason" in game).toBe(false);
      expect(game.seats).toHaveLength(5);
      expect(game.seats.map((seat) => seat.playerId)).toEqual(playerIds);
      expect(game.seats.every((seat) => seat.agentName.startsWith("RuleBasedAgent-"))).toBe(true);

      if (game.status === "completed") {
        expect(game.contract.targetPointCards).toBeGreaterThanOrEqual(13);
        expect(game.contract.targetPointCards).toBeLessThanOrEqual(20);
        expect(game.pointCards.napoleonTeam + game.pointCards.alliance).toBe(20);
        expect(game.contractSucceeded).toBe(game.winner === "napoleon-team");
        expect(game.seats.filter((seat) => seat.role === "napoleon")).toHaveLength(1);
        expect(game.seats.every((seat) => seat.role !== "unknown")).toBe(true);
      }
    }

    const assignmentsByRotation = first.games.slice(0, 5).map((game) =>
      game.seats.map((seat) => seat.sourceAgentIndex)
    );

    expect(assignmentsByRotation).toEqual([
      [0, 1, 2, 3, 4],
      [4, 0, 1, 2, 3],
      [3, 4, 0, 1, 2],
      [2, 3, 4, 0, 1],
      [1, 2, 3, 4, 0]
    ]);

    for (let sourceAgentIndex = 0; sourceAgentIndex < 5; sourceAgentIndex += 1) {
      expect(assignmentsByRotation.map((assignment) =>
        assignment.indexOf(sourceAgentIndex)
      )).toEqual([
        sourceAgentIndex,
        (sourceAgentIndex + 1) % 5,
        (sourceAgentIndex + 2) % 5,
        (sourceAgentIndex + 3) % 5,
        (sourceAgentIndex + 4) % 5
      ]);
    }
  });

  it("records one failed game entry instead of dropping agent errors", async () => {
    const record = await runEvaluation({
      startSeed: 1,
      gameCount: 1,
      playerIds,
      agents: playerIds.map((_, index) => ({
        name: `FailingAgent-${index}`,
        createAgent: () => new ThrowingAgent()
      }))
    });

    expect(record.games).toHaveLength(1);
    expect(record.completedCount).toBe(0);
    expect(record.failedCount).toBe(1);
    expect(record.games[0]).toMatchObject({
      status: "failed",
      seed: 1,
      failureReason: "intentional evaluation failure"
    });
    expect(record.games[0].seats).toHaveLength(5);
    expect(record.games[0].seats.every((seat) => seat.role === "unknown")).toBe(true);
  });

  it("keeps the existing automated game runner usable", async () => {
    const record = await runAutomatedGame({
      seed: 200,
      playerIds,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });

    expect(record.result.winner).toMatch(/^(napoleon-team|alliance)$/);
    expect(record.playerIds).toEqual(playerIds);
    expect(record.decisions.length).toBeGreaterThan(0);
  });
});

class ThrowingAgent implements Agent {
  async selectAction(_observation: PlayerObservation): Promise<never> {
    throw new Error("intentional evaluation failure");
  }
}
