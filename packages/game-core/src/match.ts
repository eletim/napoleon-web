import { createInitialGame } from "./game.js";
import type {
  CreateInitialGameOptions,
  GameResult,
  GameState,
  PlayerId
} from "./types.js";

export const MATCH_ROUND_COUNT = 5;

export interface MatchRoundResult {
  roundNumber: number;
  result: GameResult;
}

export interface MatchState {
  currentRound: number;
  currentGame: GameState | null;
  roundResults: readonly MatchRoundResult[];
  /** The number of rounds that have not yet produced a result, including the current round. */
  remainingRounds: number;
  completed: boolean;
}

export type MatchRuleErrorCode =
  | "MATCH_COMPLETED"
  | "ROUND_NOT_COMPLETED"
  | "MATCH_PLAYER_MISMATCH";

export class MatchRuleError extends Error {
  readonly code: MatchRuleErrorCode;

  constructor(code: MatchRuleErrorCode, message: string) {
    super(message);
    this.name = "MatchRuleError";
    this.code = code;
  }
}

export interface CreateNextRoundOptions {
  rng?: () => number;
}

export function createInitialMatch(options: CreateInitialGameOptions = {}): MatchState {
  return {
    currentRound: 1,
    currentGame: createInitialGame(options),
    roundResults: [],
    remainingRounds: MATCH_ROUND_COUNT,
    completed: false
  };
}

/**
 * Stores a new state for the active game without applying or duplicating any in-round rules.
 */
export function updateCurrentGame(match: MatchState, game: GameState): MatchState {
  const currentGame = requireActiveGame(match);

  if (!haveSamePlayers(currentGame, game)) {
    throw new MatchRuleError(
      "MATCH_PLAYER_MISMATCH",
      "Players cannot change during a match."
    );
  }

  return { ...match, currentGame: game };
}

/**
 * Records the active game's result and initializes the next round when one remains.
 */
export function completeCurrentRound(
  match: MatchState,
  options: CreateNextRoundOptions = {}
): MatchState {
  const currentGame = requireActiveGame(match);

  if (
    currentGame.phase !== "finished" ||
    !currentGame.isGameOver ||
    currentGame.result === null
  ) {
    throw new MatchRuleError(
      "ROUND_NOT_COMPLETED",
      "The current round must have a finalized game result."
    );
  }

  const roundResults = [
    ...match.roundResults,
    { roundNumber: match.currentRound, result: currentGame.result }
  ];
  const remainingRounds = MATCH_ROUND_COUNT - roundResults.length;

  if (remainingRounds === 0) {
    return {
      ...match,
      currentGame: null,
      roundResults,
      remainingRounds,
      completed: true
    };
  }

  return {
    currentRound: match.currentRound + 1,
    currentGame: createInitialGame({
      playerIds: currentGame.players.map((player) => player.id),
      rng: options.rng
    }),
    roundResults,
    remainingRounds,
    completed: false
  };
}

function requireActiveGame(match: MatchState): GameState {
  if (match.completed || match.currentGame === null) {
    throw new MatchRuleError("MATCH_COMPLETED", "The match is already completed.");
  }

  return match.currentGame;
}

function haveSamePlayers(first: GameState, second: GameState): boolean {
  return samePlayerIds(
    first.players.map((player) => player.id),
    second.players.map((player) => player.id)
  );
}

function samePlayerIds(first: readonly PlayerId[], second: readonly PlayerId[]): boolean {
  return (
    first.length === second.length &&
    first.every((playerId, index) => playerId === second[index])
  );
}
