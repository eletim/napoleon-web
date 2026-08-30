import {
  MATCH_ROUND_COUNT,
  calculateMatchProgressScores,
  calculateMatchScore,
  type MatchState,
  type PlayerId
} from "@napoleon/game-core";
import type { PublicMatchState } from "@napoleon/protocol";

export function toPublicMatchState(
  match: MatchState,
  playerIds: readonly PlayerId[]
): PublicMatchState {
  const results = match.roundResults.map(({ result }) => result);
  const pendingResult = match.currentGame?.isGameOver === true
    ? match.currentGame.result
    : null;
  if (pendingResult !== null) {
    results.push(pendingResult);
  }

  const players = calculateMatchProgressScores(playerIds, results);
  const finalScores = match.completed
    ? calculateMatchScore(players).players
    : null;

  return {
    currentRound: match.currentRound,
    roundCount: MATCH_ROUND_COUNT,
    completedRoundCount: results.length,
    remainingRounds: MATCH_ROUND_COUNT - results.length,
    completed: match.completed,
    players,
    finalScores
  };
}
