import { readFileSync } from "node:fs";
import assert from "node:assert/strict";

interface EvaluationArtifact {
  schemaVersion: 1;
  application: string;
  scenario: string;
  configuration: {
    seedCount: number;
    rotationOffsets: readonly number[];
    baseline: {
      tsCudaBatch1Workers4SecondsPer2000Games: number;
      tsBatchedPathSecondsPer2000Games: { min: number; max: number };
    };
    usesRlDatasetGeneration: boolean;
    browserOrWebRuntimeIntegration: boolean;
  };
  summary: {
    scheduledGames: number;
    completedGames: number;
    failedGames: number;
    candidate: {
      wins: number;
      losses: number;
      winRate: { numerator: number; denominator: number; rate: number | null };
      contractSuccesses: number;
      averagePointCards: number | null;
    };
    agents: readonly { key: string }[];
  };
  metrics: {
    totalElapsedSeconds: number;
    simulationCpuElapsedSeconds: number;
    inferenceElapsedSeconds: number;
    gamesPerSecond: number;
    decisionsPerSecond: number;
    requestCount: number;
    sessionRunCount: number;
    meanBatchSize: number | null;
    maxBatchSize: number;
    policyStats: readonly {
      key: string;
      requestCount: number;
      sessionRunCount: number;
      meanBatchSize: number | null;
      maxBatchSize: number;
    }[];
  };
  games: readonly {
    gameIndex: number;
    seed: number;
    runtimeSeed: number;
    rotationOffset: number;
    roster: {
      currentSeatIndex: number | null;
      seats: readonly {
        seatIndex: number;
        agent: { type: string; id: string };
      }[];
    };
    winner: string;
    contractSucceeded: boolean;
    pointCards: { napoleonTeam: number; alliance: number };
  }[];
}

const artifactPath = process.argv[2];
if (artifactPath === undefined) {
  throw new Error("usage: evaluationArtifact.test.js <artifact.json>");
}
const raw = readFileSync(artifactPath, "utf8");
const artifact = JSON.parse(raw) as EvaluationArtifact;

assert.equal(artifact.schemaVersion, 1);
assert.equal(artifact.application, "cpp-evaluation-benchmark-tournament");
assert.equal(artifact.scenario, "candidate-vs-opponent-pool");
assert.equal(artifact.configuration.seedCount, 2);
assert.deepEqual(artifact.configuration.rotationOffsets, [0, 1, 2, 3, 4]);
assert.equal(artifact.configuration.baseline.tsCudaBatch1Workers4SecondsPer2000Games, 11.2);
assert.deepEqual(artifact.configuration.baseline.tsBatchedPathSecondsPer2000Games, {
  min: 18,
  max: 22
});
assert.equal(artifact.configuration.usesRlDatasetGeneration, false);
assert.equal(artifact.configuration.browserOrWebRuntimeIntegration, false);

assert.equal(artifact.summary.scheduledGames, 10);
assert.equal(artifact.summary.completedGames, 10);
assert.equal(artifact.summary.failedGames, 0);
assert.equal(
  artifact.summary.candidate.winRate.denominator,
  artifact.summary.candidate.wins + artifact.summary.candidate.losses
);
assert.equal(artifact.summary.candidate.contractSuccesses >= 0, true);
assert.equal(typeof artifact.summary.candidate.averagePointCards, "number");
assert(artifact.summary.agents.some((agent) => agent.key === "current-policy:candidate"));
assert(artifact.summary.agents.some((agent) => agent.key === "rule-based:RuleBasedAgent"));
assert(artifact.summary.agents.some((agent) => agent.key === "frozen-policy:rl-v740"));

assert(artifact.metrics.totalElapsedSeconds >= 0);
assert(artifact.metrics.simulationCpuElapsedSeconds >= 0);
assert(artifact.metrics.inferenceElapsedSeconds >= 0);
assert(artifact.metrics.gamesPerSecond > 0);
assert(artifact.metrics.decisionsPerSecond > 0);
assert(artifact.metrics.requestCount > 0);
assert(artifact.metrics.sessionRunCount > 0);
assert(artifact.metrics.meanBatchSize !== null && artifact.metrics.meanBatchSize <= 3);
assert(artifact.metrics.maxBatchSize <= 3);
assert(artifact.metrics.policyStats.some((stats) => stats.key === "current-policy:candidate"));
assert(artifact.metrics.policyStats.some((stats) => stats.key === "frozen-policy:rl-v740"));
assert.equal(
  artifact.metrics.policyStats.reduce((sum, stats) => sum + stats.requestCount, 0),
  artifact.metrics.requestCount
);
assert.equal(
  artifact.metrics.policyStats.reduce((sum, stats) => sum + stats.sessionRunCount, 0),
  artifact.metrics.sessionRunCount
);

assert.deepEqual(
  artifact.games.map((game) => game.gameIndex),
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
);
assert.deepEqual(
  artifact.games.map((game) => game.seed),
  [200, 200, 200, 200, 200, 201, 201, 201, 201, 201]
);
assert.deepEqual(
  artifact.games.map((game) => game.runtimeSeed),
  artifact.games.map((game) => game.seed)
);
assert.deepEqual(
  artifact.games.map((game) => game.rotationOffset),
  [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
);
assert.deepEqual(
  artifact.games.map((game) => game.roster.currentSeatIndex),
  [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
);
for (const game of artifact.games) {
  const currentSeats = game.roster.seats.filter((seat) => seat.agent.type === "current-policy");
  assert.equal(currentSeats.length, 1);
  assert.equal(currentSeats[0].seatIndex, game.roster.currentSeatIndex);
  assert.equal(game.pointCards.napoleonTeam + game.pointCards.alliance, 20);
}

console.log("C++ evaluation artifact ok");
