import { MATCH_ROUND_COUNT } from "./match.js";
import type { PlayerId, PlayingSelfRole } from "./types.js";

export const MATCH_SCORING_PLAYER_COUNT = 5;
export const MATCH_UMA_BY_POSITION = [20, 10, 0, -10, -20] as const;

export type RoundScoringRole = PlayingSelfRole;

export interface PlayerRoundScores {
  playerId: PlayerId;
  roundScores: readonly number[];
}

export interface RawMatchScore {
  playerId: PlayerId;
  rawMatchScore: number;
}

export interface MatchUma {
  playerId: PlayerId;
  rank: number;
  uma: number;
}

export interface MatchPlayerScore extends RawMatchScore, MatchUma {
  roundScores: readonly number[];
  score: number;
  finalValue: number;
}

export interface MatchScoreResult {
  players: readonly MatchPlayerScore[];
  meanScore: number;
}

/** Calculates one round's unadjusted score from the existing outcome value d. */
export function calculateRoundScore(role: RoundScoringRole, d: number): number {
  requireFinite(d, "d");

  switch (role) {
    case "napoleon":
      return 2 * d - 5;
    case "adjutant":
      return d;
    case "alliance":
      return 0;
    case "napoleon-solo":
      return 3 * d - 5;
  }
}

/** Sums exactly five round scores without adding margin or any other adjustment. */
export function calculateRawMatchScore(roundScores: readonly number[]): number {
  if (roundScores.length !== MATCH_ROUND_COUNT) {
    throw new Error(`A match score requires exactly ${MATCH_ROUND_COUNT} round scores.`);
  }

  roundScores.forEach((roundScore, index) => requireFinite(roundScore, `roundScores[${index}]`));
  return sum(roundScores);
}

/**
 * Assigns uma by raw-match-score rank. Ties share the average of every position
 * occupied by the tied group, so input/seat order cannot affect the result.
 */
export function calculateMatchUma(rawMatchScores: readonly RawMatchScore[]): readonly MatchUma[] {
  validateFiveUniquePlayers(rawMatchScores);

  return rawMatchScores.map(({ playerId, rawMatchScore }) => {
    requireFinite(rawMatchScore, `rawMatchScore for ${playerId}`);
    const higherScoreCount = rawMatchScores.filter(
      (candidate) => candidate.rawMatchScore > rawMatchScore
    ).length;
    const tiedPlayerCount = rawMatchScores.filter(
      (candidate) => candidate.rawMatchScore === rawMatchScore
    ).length;
    const occupiedUma = MATCH_UMA_BY_POSITION.slice(
      higherScoreCount,
      higherScoreCount + tiedPlayerCount
    );

    return {
      playerId,
      rank: higherScoreCount + 1,
      uma: sum(occupiedUma) / tiedPlayerCount
    };
  });
}

/** Calculates rawMatchScore, uma, score, and zero-sum finalValue for a five-player match. */
export function calculateMatchScore(players: readonly PlayerRoundScores[]): MatchScoreResult {
  validateFiveUniquePlayers(players);

  const rawScores = players.map(({ playerId, roundScores }) => ({
    playerId,
    rawMatchScore: calculateRawMatchScore(roundScores)
  }));
  const umaByPlayerId = new Map(
    calculateMatchUma(rawScores).map((entry) => [entry.playerId, entry] as const)
  );
  const scores = rawScores.map(({ playerId, rawMatchScore }) => {
    const ranking = umaByPlayerId.get(playerId);
    if (ranking === undefined) {
      throw new Error(`Uma was not calculated for ${playerId}.`);
    }

    return {
      playerId,
      rawMatchScore,
      rank: ranking.rank,
      uma: ranking.uma,
      score: rawMatchScore + ranking.uma
    };
  });
  const meanScore = sum(scores.map(({ score }) => score)) / MATCH_SCORING_PLAYER_COUNT;
  const finalValues = scores.map(({ score }) => score - meanScore);

  // Computing the final entry as the remainder makes a normal left-to-right sum
  // exactly zero even when tied uma produces non-binary fractions such as thirds.
  const zeroSumRemainder = -sum(finalValues.slice(0, MATCH_SCORING_PLAYER_COUNT - 1));
  finalValues[MATCH_SCORING_PLAYER_COUNT - 1] = zeroSumRemainder === 0 ? 0 : zeroSumRemainder;

  return {
    players: scores.map((entry, index) => ({
      ...entry,
      roundScores: [...players[index].roundScores],
      finalValue: finalValues[index]
    })),
    meanScore
  };
}

function validateFiveUniquePlayers(players: readonly { playerId: PlayerId }[]): void {
  if (players.length !== MATCH_SCORING_PLAYER_COUNT) {
    throw new Error(`Match scoring requires exactly ${MATCH_SCORING_PLAYER_COUNT} players.`);
  }

  if (new Set(players.map(({ playerId }) => playerId)).size !== MATCH_SCORING_PLAYER_COUNT) {
    throw new Error("Match scoring requires five unique player ids.");
  }
}

function requireFinite(value: number, name: string): void {
  if (!Number.isFinite(value)) {
    throw new Error(`${name} must be finite.`);
  }
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}
