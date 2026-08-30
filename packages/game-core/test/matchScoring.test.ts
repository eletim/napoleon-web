import { describe, expect, it } from "vitest";
import {
  calculateMatchScore,
  calculateMatchUma,
  calculateRawMatchScore,
  calculateRoundScore,
  type MatchPlayerScore,
  type PlayerRoundScores,
  type RawMatchScore
} from "../src/index.js";

const playerIds = ["alice", "bob", "carol", "dave", "eve"] as const;

describe("match scoring", () => {
  it("calculates each role's round score from d without a margin adjustment", () => {
    expect(calculateRoundScore("napoleon", 13)).toBe(21);
    expect(calculateRoundScore("adjutant", 13)).toBe(13);
    expect(calculateRoundScore("alliance", 13)).toBe(0);
    expect(calculateRoundScore("napoleon-solo", 13)).toBe(34);

    expect(calculateRoundScore("napoleon", 19)).toBe(33);
    expect(calculateRoundScore("adjutant", 19)).toBe(19);
    expect(calculateRoundScore("alliance", 19)).toBe(0);
    expect(calculateRoundScore("napoleon-solo", 19)).toBe(52);
  });

  it("accumulates exactly five round scores into rawMatchScore", () => {
    expect(calculateRawMatchScore([21, 13, 0, 34, 25])).toBe(93);
    expect(() => calculateRawMatchScore([21, 13, 0, 34])).toThrow(/exactly 5/);
  });

  it("assigns normal ranks and uma from rawMatchScore", () => {
    expect(calculateMatchUma(rawScores([50, 40, 30, 20, 10]))).toEqual([
      { playerId: "alice", rank: 1, uma: 20 },
      { playerId: "bob", rank: 2, uma: 10 },
      { playerId: "carol", rank: 3, uma: 0 },
      { playerId: "dave", rank: 4, uma: -10 },
      { playerId: "eve", rank: 5, uma: -20 }
    ]);
  });

  it("shares first-place uma equally without using seat order", () => {
    expect(calculateMatchUma(rawScores([50, 50, 30, 20, 10]))).toEqual([
      { playerId: "alice", rank: 1, uma: 15 },
      { playerId: "bob", rank: 1, uma: 15 },
      { playerId: "carol", rank: 3, uma: 0 },
      { playerId: "dave", rank: 4, uma: -10 },
      { playerId: "eve", rank: 5, uma: -20 }
    ]);

    const reordered = calculateMatchUma([
      { playerId: "bob", rawMatchScore: 50 },
      { playerId: "eve", rawMatchScore: 10 },
      { playerId: "alice", rawMatchScore: 50 },
      { playerId: "dave", rawMatchScore: 20 },
      { playerId: "carol", rawMatchScore: 30 }
    ]);
    expect(scoreFor(reordered, "alice").uma).toBe(15);
    expect(scoreFor(reordered, "bob").uma).toBe(15);
  });

  it("averages the occupied uma for a middle-rank tie", () => {
    expect(calculateMatchUma(rawScores([50, 40, 40, 20, 10]))).toEqual([
      { playerId: "alice", rank: 1, uma: 20 },
      { playerId: "bob", rank: 2, uma: 5 },
      { playerId: "carol", rank: 2, uma: 5 },
      { playerId: "dave", rank: 4, uma: -10 },
      { playerId: "eve", rank: 5, uma: -20 }
    ]);
  });

  it("handles multiple independent tie groups", () => {
    expect(calculateMatchUma(rawScores([50, 50, 30, 10, 10]))).toEqual([
      { playerId: "alice", rank: 1, uma: 15 },
      { playerId: "bob", rank: 1, uma: 15 },
      { playerId: "carol", rank: 3, uma: 0 },
      { playerId: "dave", rank: 4, uma: -15 },
      { playerId: "eve", rank: 4, uma: -15 }
    ]);
  });

  it("gives every player zero uma when all five players tie", () => {
    expect(calculateMatchUma(rawScores([30, 30, 30, 30, 30]))).toEqual(
      playerIds.map((playerId) => ({ playerId, rank: 1, uma: 0 }))
    );
  });

  it("keeps roundScore, rawMatchScore, uma, score, and finalValue distinct", () => {
    const result = calculateMatchScore(roundScores([40, 30, 20, 10, 0]));

    expect(result.meanScore).toBe(20);
    expect(result.players).toEqual([
      playerScore("alice", 40, 1, 20, 60, 40),
      playerScore("bob", 30, 2, 10, 40, 20),
      playerScore("carol", 20, 3, 0, 20, 0),
      playerScore("dave", 10, 4, -10, 0, -20),
      playerScore("eve", 0, 5, -20, -20, -40)
    ]);
  });

  it("makes finalValue exactly zero-sum with floating-point round scores", () => {
    const result = calculateMatchScore(roundScores([0.1, 0.2, 0.3, 0.4, 0.5]));
    const finalValueSum = result.players.reduce(
      (total, player) => total + player.finalValue,
      0
    );

    for (const player of result.players) {
      expect(player.finalValue).toBeCloseTo(player.score - result.meanScore, 12);
    }
    expect(finalValueSum).toBe(0);
  });

  it("subtracts only the mean and does not divide by standard deviation", () => {
    const result = calculateMatchScore(roundScores([40, 30, 20, 10, 0]));
    const finalValues = result.players.map(({ finalValue }) => finalValue);

    expect(finalValues).toEqual([40, 20, 0, -20, -40]);
    expect(Math.sqrt(finalValues.reduce((total, value) => total + value ** 2, 0) / 5)).toBe(
      Math.sqrt(800)
    );
  });

  it("is a pure calculation that does not mutate its inputs", () => {
    const input = roundScores([40, 30, 20, 10, 0]);
    const before = structuredClone(input);

    const first = calculateMatchScore(input);
    const second = calculateMatchScore(input);

    expect(input).toEqual(before);
    expect(second).toEqual(first);
  });
});

function rawScores(scores: readonly number[]): RawMatchScore[] {
  return playerIds.map((playerId, index) => ({ playerId, rawMatchScore: scores[index] }));
}

function roundScores(rawMatchScores: readonly number[]): PlayerRoundScores[] {
  return playerIds.map((playerId, index) => ({
    playerId,
    roundScores: [rawMatchScores[index], 0, 0, 0, 0]
  }));
}

function scoreFor<T extends { playerId: string }>(scores: readonly T[], playerId: string): T {
  const score = scores.find((candidate) => candidate.playerId === playerId);
  if (score === undefined) {
    throw new Error(`Missing score for ${playerId}.`);
  }
  return score;
}

function playerScore(
  playerId: string,
  rawMatchScore: number,
  rank: number,
  uma: number,
  score: number,
  finalValue: number
): MatchPlayerScore {
  return {
    playerId,
    roundScores: [rawMatchScore, 0, 0, 0, 0],
    rawMatchScore,
    rank,
    uma,
    score,
    finalValue
  };
}
