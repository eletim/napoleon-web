import { describe, expect, it } from "vitest";
import {
  MATCH_ROUND_COUNT,
  MatchRuleError,
  applyAction,
  completeCurrentRound,
  createInitialMatch,
  updateCurrentGame,
  type GameState,
  type MatchState
} from "../src/index.js";

const noShuffle = (): number => 0;
const playerIds = ["alice", "bob", "carol", "dave", "eve"] as const;

function finishWithAllPasses(game: GameState): GameState {
  let finished = game;

  for (let passCount = 0; passCount < playerIds.length; passCount += 1) {
    finished = applyAction(finished, {
      type: "pass",
      playerId: finished.currentPlayerId
    });
  }

  return finished;
}

function expectMatchError(work: () => void, code: string): void {
  try {
    work();
    throw new Error(`Expected ${code}.`);
  } catch (error) {
    expect(error).toBeInstanceOf(MatchRuleError);
    if (error instanceof MatchRuleError) {
      expect(error.code).toBe(code);
    }
  }
}

function requireCurrentGame(match: MatchState): GameState {
  if (match.currentGame === null) {
    throw new Error("Expected an active game.");
  }

  return match.currentGame;
}

describe("match lifecycle", () => {
  it("starts a five-round match while preserving the standalone game defaults", () => {
    const match = createInitialMatch({ playerIds, rng: noShuffle });
    const game = requireCurrentGame(match);

    expect(MATCH_ROUND_COUNT).toBe(5);
    expect(match.currentRound).toBe(1);
    expect(match.remainingRounds).toBe(5);
    expect(match.completed).toBe(false);
    expect(match.roundResults).toEqual([]);
    expect(game.phase).toBe("bidding");
    expect(game.players.map((player) => player.id)).toEqual(playerIds);
  });

  it("does not record an unfinished game or allow player seats to change", () => {
    const match = createInitialMatch({ playerIds, rng: noShuffle });
    const differentPlayers = createInitialMatch({
      playerIds: ["alice", "bob", "carol", "dave", "frank"],
      rng: noShuffle
    });

    expectMatchError(() => completeCurrentRound(match), "ROUND_NOT_COMPLETED");
    expectMatchError(
      () => updateCurrentGame(match, requireCurrentGame(differentPlayers)),
      "MATCH_PLAYER_MISMATCH"
    );
    expect(match.roundResults).toEqual([]);
  });

  it("records five real game results and creates a fresh game only between rounds", () => {
    let match = createInitialMatch({ playerIds, rng: noShuffle });
    const games: GameState[] = [];

    for (let roundNumber = 1; roundNumber <= MATCH_ROUND_COUNT; roundNumber += 1) {
      const activeGame = requireCurrentGame(match);
      games.push(activeGame);
      const finishedGame = finishWithAllPasses(activeGame);

      match = updateCurrentGame(match, finishedGame);
      match = completeCurrentRound(match, { rng: noShuffle });

      expect(match.roundResults).toHaveLength(roundNumber);
      expect(match.roundResults.at(-1)).toEqual({
        roundNumber,
        result: finishedGame.result
      });
      expect(match.remainingRounds).toBe(MATCH_ROUND_COUNT - roundNumber);

      if (roundNumber < MATCH_ROUND_COUNT) {
        const nextGame = requireCurrentGame(match);
        expect(match.currentRound).toBe(roundNumber + 1);
        expect(match.completed).toBe(false);
        expect(nextGame).not.toBe(activeGame);
        expect(nextGame.phase).toBe("bidding");
        expect(nextGame.result).toBeNull();
        expect(nextGame.players.map((player) => player.id)).toEqual(playerIds);
      }
    }

    expect(games).toHaveLength(5);
    expect(match.currentRound).toBe(5);
    expect(match.currentGame).toBeNull();
    expect(match.completed).toBe(true);
    expect(match.roundResults).toHaveLength(5);
    expectMatchError(() => completeCurrentRound(match), "MATCH_COMPLETED");
    expectMatchError(
      () => updateCurrentGame(match, games[0]),
      "MATCH_COMPLETED"
    );
  });
});
