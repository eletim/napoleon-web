import { describe, expect, it } from "vitest";
import type { PublicMatchState } from "@napoleon/protocol";
import { getMatchAdvanceLabel } from "./MatchProgress";

describe("MatchProgress", () => {
  it("uses match progression state for the next-round and final-result actions", () => {
    expect(getMatchAdvanceLabel(progressMatch())).toBe("次局へ");
    expect(getMatchAdvanceLabel({ ...progressMatch(), remainingRounds: 0 })).toBe("試合結果へ");
  });
});

function progressMatch(): PublicMatchState {
  return {
    currentRound: 2,
    roundCount: 5,
    completedRoundCount: 2,
    remainingRounds: 3,
    completed: false,
    players: [0, 1, 2, 3, 4].map((index) => ({
      playerId: `player-${index}`,
      roundScores: index === 0 ? [1, 3] : [-1, 0],
      rawMatchScore: index === 0 ? 4 : -1
    })),
    finalScores: null
  };
}
