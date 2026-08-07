import { describe, expect, it } from "vitest";
import { RuleBasedAgent } from "../ruleBasedAgent.js";
import { createEvaluationReport } from "./evaluationReport.js";
import { runEvaluation } from "./runEvaluation.js";
import type {
  CompletedEvaluationGameRecord,
  EvaluationGameRecord,
  EvaluationRunRecord
} from "./types.js";

const playerIds = ["player-0", "player-1", "player-2", "player-3", "player-4"] as const;

describe("createEvaluationReport", () => {
  it("aggregates a fixed evaluation fixture by agent, seat, role, contract, point cards, and failures", () => {
    const report = createEvaluationReport(createFixtureRun());

    expect(report.schemaVersion).toBe(1);
    expect(report.sourceSchemaVersion).toBe(1);
    expect(report.summary).toMatchObject({
      expectedGameCount: 3,
      games: {
        total: 3,
        completed: 2,
        failed: 1
      },
      sampleCount: 2,
      wins: 1,
      losses: 1,
      contractSuccesses: 1,
      winRate: {
        numerator: 1,
        denominator: 2,
        rate: 0.5
      },
      contractSuccessRate: {
        numerator: 1,
        denominator: 2,
        rate: 0.5
      },
      averagePointCards: 12,
      failures: {
        total: 1,
        byReason: {
          "agent crashed": 1
        }
      }
    });
    expect(report.summary.winRate.confidenceInterval).toEqual({
      level: 0.95,
      method: "wilson",
      lower: 0.09453120573423074,
      upper: 0.9054687942657693
    });
    expect(report.summary.contractSuccessRate.confidenceInterval).toEqual(
      report.summary.winRate.confidenceInterval
    );

    expect(report.agents.map((agent) => agent.agentName)).toEqual([
      "Agent-0",
      "Agent-1",
      "Agent-2",
      "Agent-3",
      "Agent-4"
    ]);
    expect(report.agents.map((agent) => agent.sourceAgentIndex)).toEqual([0, 1, 2, 3, 4]);

    const agent0 = report.agents[0];
    expect(agent0.games).toEqual({
      total: 3,
      completed: 2,
      failed: 1
    });
    expect(agent0.sampleCount).toBe(2);
    expect(agent0.wins).toBe(2);
    expect(agent0.losses).toBe(0);
    expect(agent0.averagePointCards).toBe(12);
    expect(agent0.failures).toEqual({
      total: 1,
      byReason: {
        "agent crashed": 1
      }
    });
    expect(agent0.comparison.contractSuccessRateDelta).toBe(0);
    expect(agent0.comparison.winRateDelta).toBeCloseTo(0.4);
    expect(agent0.comparison.averagePointCardsDelta).toBeCloseTo(2.4);
    expect(agent0.winRate.confidenceInterval).toEqual({
      level: 0.95,
      method: "wilson",
      lower: 0.34238022750665303,
      upper: 1
    });
    expect(agent0.comparison.winRateDeltaConfidenceInterval).toEqual({
      level: 0.95,
      method: "newcombe-wilson",
      lower: -0.29728338909607677,
      upper: 0.6873262302663418
    });
    expect(agent0.comparison.contractSuccessRateDeltaConfidenceInterval).toEqual({
      level: 0.95,
      method: "newcombe-wilson",
      lower: -0.48351643517987997,
      upper: 0.48351643517987997
    });
    expect(agent0.comparison.averagePointCardsDeltaConfidenceInterval).toEqual({
      level: 0.95,
      method: "normal",
      lower: -1.9257568984351092,
      upper: 6.72575689843511
    });

    expect(agent0.roleResults.map((role) => [role.role, role.games.total])).toEqual([
      ["napoleon", 1],
      ["adjutant", 0],
      ["alliance", 1]
    ]);
    expect(agent0.roleResults.map((role) => [role.role, role.sampleCount])).toEqual([
      ["napoleon", 1],
      ["adjutant", 0],
      ["alliance", 1]
    ]);
    expect(agent0.roleResults[1].winRate.rate).toBeNull();
    expect(agent0.roleResults[1].winRate.confidenceInterval).toEqual({
      level: 0.95,
      method: "wilson",
      lower: null,
      upper: null
    });
    expect(agent0.roleResults[1].averagePointCards).toBeNull();

    expect(report.seats.map((seat) => [seat.seatIndex, seat.games.total])).toEqual([
      [0, 3],
      [1, 3],
      [2, 3],
      [3, 3],
      [4, 3]
    ]);
    expect(report.seats.map((seat) => [seat.seatIndex, seat.sampleCount])).toEqual([
      [0, 2],
      [1, 2],
      [2, 2],
      [3, 2],
      [4, 2]
    ]);
    expect(report.seats[0]).toMatchObject({
      seatIndex: 0,
      sampleCount: 2,
      wins: 2,
      losses: 0,
      failures: {
        total: 1,
        byReason: {
          "agent crashed": 1
        }
      }
    });

    expect(report.roles.map((role) => [role.role, role.games.completed, role.wins])).toEqual([
      ["napoleon", 2, 1],
      ["adjutant", 1, 1],
      ["alliance", 7, 4]
    ]);
    expect(report.roles.map((role) => [role.role, role.sampleCount])).toEqual([
      ["napoleon", 2],
      ["adjutant", 1],
      ["alliance", 7]
    ]);
    expectReportHasNoNonFiniteNumbers(report);
  });

  it("creates the same report when evaluation games and seats are provided in a different order", () => {
    const run = createFixtureRun();
    const shuffled: EvaluationRunRecord = {
      ...run,
      games: [...run.games]
        .reverse()
        .map((game) => ({
          ...game,
          seats: [...game.seats].reverse()
        }))
    };

    expect(createEvaluationReport(shuffled)).toEqual(createEvaluationReport(run));
  });

  it("keeps expected game count separate from the provided game record count", () => {
    const run = createFixtureRun();
    const incompleteRun: EvaluationRunRecord = {
      ...run,
      gameCount: 4,
      games: run.games.slice(0, 2),
      completedCount: 2,
      failedCount: 0
    };

    const report = createEvaluationReport(incompleteRun);

    expect(report.summary.expectedGameCount).toBe(4);
    expect(report.summary.games).toEqual({
      total: 2,
      completed: 2,
      failed: 0
    });
  });

  it("reports empty runs without NaN or Infinity values", () => {
    const report = createEvaluationReport({
      schemaVersion: 1,
      startSeed: 10,
      endSeed: 9,
      gameCount: 0,
      rotationOffsets: [0],
      playerIds,
      games: [],
      completedCount: 0,
      failedCount: 0
    });

    expect(report.summary).toMatchObject({
      expectedGameCount: 0,
      sampleCount: 0,
      wins: 0,
      losses: 0,
      winRate: {
        numerator: 0,
        denominator: 0,
        rate: null,
        confidenceInterval: {
          level: 0.95,
          method: "wilson",
          lower: null,
          upper: null
        }
      },
      contractSuccessRate: {
        numerator: 0,
        denominator: 0,
        rate: null,
        confidenceInterval: {
          level: 0.95,
          method: "wilson",
          lower: null,
          upper: null
        }
      },
      averagePointCards: null
    });
    expect(report.agents).toEqual([]);
    expect(report.seats).toEqual([]);
    expect(report.roles.map((role) => [role.role, role.sampleCount])).toEqual([
      ["napoleon", 0],
      ["adjutant", 0],
      ["alliance", 0]
    ]);
    expectReportHasNoNonFiniteNumbers(report);
  });

  it("does not report a point-card delta interval when a mean variance is not estimable", () => {
    const run = createFixtureRun();
    const singletonRun: EvaluationRunRecord = {
      ...run,
      games: run.games.slice(0, 1),
      gameCount: 1,
      endSeed: 10,
      completedCount: 1,
      failedCount: 0
    };

    const report = createEvaluationReport(singletonRun);

    expect(report.agents).toHaveLength(5);
    expect(report.agents.every((agent) => agent.sampleCount === 1)).toBe(true);
    expect(report.agents.every((agent) =>
      agent.comparison.averagePointCardsDeltaConfidenceInterval.lower === null
      && agent.comparison.averagePointCardsDeltaConfidenceInterval.upper === null
    )).toBe(true);
    expectReportHasNoNonFiniteNumbers(report);
  });

  it("keeps agent definitions separate when they share the same display name", () => {
    const run = createFixtureRun();
    const duplicateNameRun: EvaluationRunRecord = {
      ...run,
      games: run.games.map((game) => ({
        ...game,
        seats: game.seats.map((seat) => ({
          ...seat,
          agentName: "SharedAgent"
        }))
      }))
    };

    const report = createEvaluationReport(duplicateNameRun);

    expect(report.agents).toHaveLength(5);
    expect(report.agents.map((agent) => agent.agentName)).toEqual([
      "SharedAgent",
      "SharedAgent",
      "SharedAgent",
      "SharedAgent",
      "SharedAgent"
    ]);
    expect(report.agents.map((agent) => agent.sourceAgentIndex)).toEqual([0, 1, 2, 3, 4]);
    expect(report.agents.map((agent) => agent.games.total)).toEqual([3, 3, 3, 3, 3]);
  });

  it("summarizes deterministic RuleBasedAgent runner output for smoke coverage", async () => {
    const run = await runEvaluation({
      startSeed: 100,
      gameCount: 3,
      playerIds,
      rotationOffsets: [0, 1, 2, 3, 4],
      agents: playerIds.map((_, index) => ({
        name: `RuleBasedAgent-${index}`,
        createAgent: ({ rng }) => new RuleBasedAgent(rng)
      }))
    });
    const report = createEvaluationReport(run);
    const reorderedRun: EvaluationRunRecord = {
      ...run,
      games: [...run.games]
        .reverse()
        .map((game) => ({
          ...game,
          seats: [...game.seats].reverse()
        }))
    };

    expect(report.summary.games).toEqual({
      total: 15,
      completed: 15,
      failed: 0
    });
    expect(report.summary.failures.total).toBe(0);
    expect(report.summary.failures.byReason).toEqual({});
    expect(report.summary.contractSuccessRate.denominator).toBe(15);
    expect(report.summary.contractSuccessRate.rate).not.toBeNull();
    expect(report.summary.averagePointCards).not.toBeNull();
    expect(report.agents).toHaveLength(5);
    expect(report.agents.every((agent) => agent.games.total === 15)).toBe(true);
    expect(report.agents.every((agent) => agent.roleResults.length === 3)).toBe(true);
    expect(report.agents.every((agent) => agent.seatResults.length === 5)).toBe(true);
    expect(report.agents.every((agent) => agent.comparison.winRateDelta === 0)).toBe(true);
    expect(report.agents.every((agent) => agent.comparison.averagePointCardsDelta === 0)).toBe(true);
    expect(report.seats).toHaveLength(5);
    expect(report.seats.every((seat) => seat.games.total === 15)).toBe(true);
    expect(report.roles.map((role) => role.role)).toEqual(["napoleon", "adjutant", "alliance"]);
    expect(report.roles.reduce((sum, role) => sum + role.games.completed, 0)).toBe(75);
    expect(createEvaluationReport(reorderedRun)).toEqual(report);
  });
});

function expectReportHasNoNonFiniteNumbers(value: unknown): void {
  if (typeof value === "number") {
    expect(Number.isFinite(value)).toBe(true);
    return;
  }

  if (Array.isArray(value)) {
    for (const item of value) {
      expectReportHasNoNonFiniteNumbers(item);
    }
    return;
  }

  if (value !== null && typeof value === "object") {
    for (const item of Object.values(value)) {
      expectReportHasNoNonFiniteNumbers(item);
    }
  }
}

function createFixtureRun(): EvaluationRunRecord {
  const games: readonly EvaluationGameRecord[] = [
    completedGame({
      gameIndex: 0,
      seed: 10,
      rotationOffset: 0,
      napoleonIndex: 0,
      adjutantIndex: 1,
      napoleonTeamPointCards: 14,
      alliancePointCards: 6,
      targetPointCards: 13
    }),
    completedGame({
      gameIndex: 1,
      seed: 11,
      rotationOffset: 0,
      napoleonIndex: 2,
      adjutantIndex: null,
      napoleonTeamPointCards: 10,
      alliancePointCards: 10,
      targetPointCards: 15
    }),
    {
      schemaVersion: 1,
      status: "failed",
      gameIndex: 2,
      seed: 12,
      rotationOffset: 0,
      playerIds,
      seats: playerIds.map((playerId, index) => ({
        playerId,
        seatIndex: index,
        agentName: `Agent-${index}`,
        sourceAgentIndex: index,
        role: "unknown"
      })),
      failureReason: "agent crashed"
    }
  ];

  return {
    schemaVersion: 1,
    startSeed: 10,
    endSeed: 12,
    gameCount: 3,
    rotationOffsets: [0],
    playerIds,
    games,
    completedCount: 2,
    failedCount: 1
  };
}

function completedGame(args: {
  gameIndex: number;
  seed: number;
  rotationOffset: number;
  napoleonIndex: number;
  adjutantIndex: number | null;
  napoleonTeamPointCards: number;
  alliancePointCards: number;
  targetPointCards: number;
}): CompletedEvaluationGameRecord {
  const napoleonPlayerId = playerIds[args.napoleonIndex];
  const adjutantPlayerId = args.adjutantIndex === null ? null : playerIds[args.adjutantIndex];
  const winner = args.napoleonTeamPointCards >= args.targetPointCards
    ? "napoleon-team"
    : "alliance";

  return {
    schemaVersion: 1,
    status: "completed",
    gameIndex: args.gameIndex,
    seed: args.seed,
    rotationOffset: args.rotationOffset,
    playerIds,
    seats: playerIds.map((playerId, index) => ({
      playerId,
      seatIndex: index,
      agentName: `Agent-${index}`,
      sourceAgentIndex: index,
      role: playerId === napoleonPlayerId
        ? "napoleon"
        : playerId === adjutantPlayerId
          ? "adjutant"
          : "alliance"
    })),
    contract: {
      napoleonPlayerId,
      targetPointCards: args.targetPointCards,
      adjutantPlayerId
    },
    pointCards: {
      napoleonTeam: args.napoleonTeamPointCards,
      alliance: args.alliancePointCards
    },
    winner,
    contractSucceeded: winner === "napoleon-team",
    result: {
      winner,
      napoleonTeamPointCards: args.napoleonTeamPointCards,
      alliancePointCards: args.alliancePointCards,
      targetPointCards: args.targetPointCards,
      napoleonPlayerId,
      adjutantPlayerId
    }
  };
}
