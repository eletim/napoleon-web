import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { PublicGameResult, PublicMatchState } from "@napoleon/protocol";
import { RoundResultPanel } from "./RoundResultPanel";
import type { TablePlayer } from "./tableTypes";

const players: readonly TablePlayer[] = [
  tablePlayer("player-0", "自分", "self"),
  tablePlayer("player-1", "左側AI", "left"),
  tablePlayer("player-2", "奥左AI", "top-left"),
  tablePlayer("player-3", "奥右AI", "top-right"),
  tablePlayer("player-4", "右側AI", "right")
];

describe("RoundResultPanel", () => {
  it("leads with the winning side, then napoleon and adjutant, for a napoleon-team win", () => {
    const html = render({
      result: standardResult({
        winner: "napoleon-team",
        napoleonPlayerId: "player-1",
        adjutantPlayerId: "player-2"
      }),
      match: progressMatch()
    });

    expect(html).toContain("round-result-winner-napoleon");
    expect(html).toContain("ナポレオン陣営の勝利");
    expect(html).toContain("ナポレオン 左側AI");
    expect(html).toContain("副官 奥左AI");
    expect(html).not.toContain("ソロ");
  });

  it("shows ソロ instead of a named adjutant when napoleon played solo", () => {
    const html = render({
      result: standardResult({
        winner: "alliance",
        napoleonPlayerId: "player-3",
        adjutantPlayerId: null
      }),
      match: progressMatch()
    });

    expect(html).toContain("round-result-winner-alliance");
    expect(html).toContain("連合軍の勝利");
    expect(html).toContain("ナポレオン 奥右AI");
    expect(html).toContain("ソロ");
    expect(html).not.toContain("副官 ");
  });

  it("compares this round's score against the cumulative score per player, with the self row marked", () => {
    const html = render({
      result: standardResult({
        winner: "napoleon-team",
        napoleonPlayerId: "player-1",
        adjutantPlayerId: "player-2"
      }),
      match: progressMatch()
    });

    // Header labels distinguish this round's score from the running total.
    expect(html).toContain("<th scope=\"col\">今回</th>");
    expect(html).toContain("<th scope=\"col\">累積</th>");
    // player-0 (self): this round +7 (last of [3, 7]), cumulative +10.
    expect(html).toMatch(/<tr class="round-result-scores-self"><th scope="row">自分<\/th><td>\+7<\/td><td>\+10<\/td>/);
    // player-1: this round -2 (last of [-1, -2]), cumulative -3, not marked as self.
    expect(html).toMatch(/<tr><th scope="row">左側AI<\/th><td>-2<\/td><td>-3<\/td>/);
  });

  it("shows the round/match progress and the correct advance label mid-match", () => {
    const html = render({
      result: standardResult({
        winner: "napoleon-team",
        napoleonPlayerId: "player-1",
        adjutantPlayerId: null
      }),
      match: progressMatch()
    });

    expect(html).toContain("第3局 / 全5局");
    expect(html).toContain("次局へ");
  });

  it("shows the final-round advance label at the fifth round", () => {
    const html = render({
      result: standardResult({
        winner: "napoleon-team",
        napoleonPlayerId: "player-1",
        adjutantPlayerId: null
      }),
      match: { ...progressMatch(), currentRound: 5, remainingRounds: 0 }
    });

    expect(html).toContain("試合結果へ");
  });

  it("hides the advance button once the match is completed", () => {
    const html = render({
      result: standardResult({
        winner: "napoleon-team",
        napoleonPlayerId: "player-1",
        adjutantPlayerId: null
      }),
      match: { ...progressMatch(), completed: true }
    });

    expect(html).not.toContain("match-advance-button");
    expect(html).not.toContain("次局へ");
    expect(html).not.toContain("試合結果へ");
  });

  it("renders an all-pass round without napoleon/adjutant chips but still with the score comparison", () => {
    const html = render({
      result: {
        resultType: "all-pass",
        starterPlayerId: "player-4",
        payoffs: players.map(({ id }) => ({
          playerId: id,
          payoff: id === "player-4" ? 1 : -1
        }))
      },
      match: progressMatch()
    });

    expect(html).toContain("round-result-winner-draw");
    expect(html).toContain("流局(全員パス)");
    expect(html).toContain("親 右側AIのみ+1");
    expect(html).not.toContain("round-result-role-napoleon");
    expect(html).toContain("round-result-scores");
  });

  it("omits round progress, the score table, and the advance button for a standalone (non-match) game", () => {
    const html = render({
      result: standardResult({
        winner: "napoleon-team",
        napoleonPlayerId: "player-1",
        adjutantPlayerId: null
      }),
      match: undefined
    });

    expect(html).toContain("ナポレオン陣営の勝利");
    expect(html).not.toContain("round-result-scores");
    expect(html).not.toContain("round-result-progress");
    expect(html).not.toContain("match-advance-button");
  });
});

function render({
  match,
  result
}: {
  match: PublicMatchState | undefined;
  result: PublicGameResult;
}): string {
  return renderToStaticMarkup(
    <RoundResultPanel
      disabled={false}
      match={match}
      onAdvanceMatch={vi.fn()}
      players={players}
      result={result}
      selfPlayerId="player-0"
    />
  );
}

function standardResult(overrides: {
  adjutantPlayerId: string | null;
  napoleonPlayerId: string;
  winner: "napoleon-team" | "alliance";
}): PublicGameResult {
  return {
    resultType: "standard",
    winner: overrides.winner,
    napoleonTeamPointCards: overrides.winner === "napoleon-team" ? 14 : 6,
    alliancePointCards: overrides.winner === "napoleon-team" ? 6 : 14,
    targetPointCards: 13,
    napoleonPlayerId: overrides.napoleonPlayerId,
    adjutantPlayerId: overrides.adjutantPlayerId
  };
}

function progressMatch(): PublicMatchState {
  const scores = new Map<string, readonly number[]>([
    ["player-0", [3, 7]],
    ["player-1", [-1, -2]],
    ["player-2", [1, -3]],
    ["player-3", [-2, 5]],
    ["player-4", [-1, -7]]
  ]);

  return {
    currentRound: 3,
    roundCount: 5,
    completedRoundCount: 2,
    remainingRounds: 3,
    completed: false,
    players: players.map(({ id }) => ({
      playerId: id,
      roundScores: scores.get(id) ?? [],
      rawMatchScore: (scores.get(id) ?? []).reduce((sum, score) => sum + score, 0)
    })),
    finalScores: null
  };
}

function tablePlayer(id: string, label: string, seat: TablePlayer["seat"]): TablePlayer {
  return {
    id,
    label,
    seat,
    handCount: 0,
    capturedPointCards: [],
    isSelf: seat === "self"
  };
}
