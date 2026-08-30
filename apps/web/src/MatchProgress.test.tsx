import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { PublicMatchState } from "@napoleon/protocol";
import { getMatchAdvanceLabel, MatchProgress } from "./MatchProgress";
import type { TablePlayer } from "./tableTypes";

const players: readonly TablePlayer[] = [
  { id: "player-0", label: "自分", seat: "self", handCount: 10, capturedPointCards: [], isSelf: true },
  ...([1, 2, 3, 4] as const).map((index) => ({
    id: `player-${index}`,
    label: `AI ${index}`,
    seat: (["left", "top-left", "top-right", "right"] as const)[index - 1],
    handCount: 10,
    capturedPointCards: [],
    isSelf: false
  }))
];

describe("MatchProgress", () => {
  it("shows the current round, raw totals, and per-round history without premature final values", () => {
    const html = renderToStaticMarkup(
      <MatchProgress match={progressMatch()} players={players} />
    );

    expect(html).toContain("2/5");
    expect(html).toContain("1局");
    expect(html).toContain("2局");
    expect(html).toContain("+4");
    expect(html).toContain("ウマ・最終値は5局終了後に確定します。");
    expect(html).not.toContain("順位");
  });

  it("shows rank, uma, and final value only for a completed match", () => {
    const progress = progressMatch();
    const finalPlayer = {
      ...progress.players[0],
      roundScores: [1, 3, 0, 2, 4],
      rawMatchScore: 10,
      rank: 1,
      uma: 20,
      score: 30,
      finalValue: 18
    };
    const html = renderToStaticMarkup(
      <MatchProgress
        match={{
          ...progress,
          currentRound: 5,
          completedRoundCount: 5,
          remainingRounds: 0,
          completed: true,
          finalScores: [finalPlayer]
        }}
        players={players}
      />
    );

    expect(html).toContain("5/5");
    expect(html).toContain("試合結果");
    expect(html).toContain("順位");
    expect(html).toContain("ウマ");
    expect(html).toContain("+18");
    expect(html).not.toContain("終了後に確定");
  });

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
    players: players.map(({ id }, index) => ({
      playerId: id,
      roundScores: index === 0 ? [1, 3] : [-1, 0],
      rawMatchScore: index === 0 ? 4 : -1
    })),
    finalScores: null
  };
}
