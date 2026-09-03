// @vitest-environment happy-dom

import { readFileSync } from "node:fs";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { calculateMatchScore } from "@napoleon/game-core";
import type { PublicMatchPlayerFinalScore, PublicMatchState } from "@napoleon/protocol";
import { hasCompletedMatchResult, MatchFinalResults } from "./MatchFinalResults";
import type { TablePlayer } from "./tableTypes";

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

const players: readonly TablePlayer[] = ([0, 1, 2, 3, 4] as const).map((index) => ({
  id: `player-${index}`,
  label: index === 0 ? "自分" : `AI ${index}`,
  seat: (["self", "left", "top-left", "top-right", "right"] as const)[index],
  handCount: 0,
  capturedPointCards: [],
  isSelf: index === 0
}));

describe("MatchFinalResults", () => {
  it.each([1, 2, 3, 4])("does not show a final result after round %i", (round) => {
    const match = progressMatch(round);

    expect(hasCompletedMatchResult(match)).toBe(false);
    expect(render(match)).toBe("");
  });

  it("does not infer completion merely from being in round 5", () => {
    const match = progressMatch(5);

    expect(hasCompletedMatchResult(match)).toBe(false);
    expect(render(match)).toBe("");
  });

  it("shows all five domain final-score values and optional round history after completion", () => {
    const html = render(completedMatch([
      finalScore("player-3", 4, -8, -10, -18, -20.4),
      finalScore("player-0", 1, 16, 20, 36, 33.6),
      finalScore("player-4", 5, -13, -20, -33, -35.4),
      finalScore("player-2", 3, 2, 0, 2, -0.4),
      finalScore("player-1", 2, 8, 10, 18, 15.6)
    ]));

    expect(html).toContain("全5局 終了");
    expect(html).toContain("最終結果");
    expect(html.match(/class="match-final-player"/g)).toHaveLength(5);
    expect(html).toContain("1位");
    expect(html).toContain("2位");
    expect(html).toContain("3位");
    expect(html).toContain("4位");
    expect(html).toContain("5位");
    expect(html).toContain("5局の素点合計");
    expect(html).toContain("ウマ");
    expect(html).toContain("score");
    expect(html).toContain("finalValue");
    expect(html).toContain("+16");
    expect(html).toContain("+20");
    expect(html).toContain("+36");
    expect(html).toContain("+33.6");
    expect(html).toContain("局ごとの得点");
    expect(html.indexOf("1位")).toBeLessThan(html.indexOf("5位"));
  });

  it("uses domain tie ranks and averaged uma without a seat-order tie break", () => {
    const html = render(completedMatch([
      finalScore("player-0", 2, 10, 5, 15, 10),
      finalScore("player-1", 1, 20, 20, 40, 35),
      finalScore("player-2", 2, 10, 5, 15, 10),
      finalScore("player-3", 4, 0, -10, -10, -15),
      finalScore("player-4", 5, -10, -20, -30, -40)
    ]));

    expect(html.match(/>2位</g)).toHaveLength(2);
    expect(html.match(/<dd>\+5<\/dd>/g)).toHaveLength(2);
    expect(html).not.toContain("3位");
  });

  it("displays the supplied zero-sum finalValues unchanged", () => {
    const finalScores = [
      finalScore("player-0", 1, 10, 20, 30, 28.2),
      finalScore("player-1", 2, 4, 10, 14, 12.2),
      finalScore("player-2", 3, 0, 0, 0, -1.8),
      finalScore("player-3", 4, -3, -10, -13, -14.8),
      finalScore("player-4", 5, -2, -20, -22, -23.8)
    ];
    const html = render(completedMatch(finalScores));

    expect(finalScores.reduce((sum, player) => sum + player.finalValue, 0)).toBe(0);
    for (const player of finalScores) {
      expect(html).toContain(`<strong>${player.finalValue > 0 ? "+" : ""}${player.finalValue}</strong>`);
    }
  });

  it("renders every value from the domain Match result without recalculating scoring", () => {
    const result = calculateMatchScore([
      { playerId: "player-0", roundScores: [2, 2, 2, 2, 2] },
      { playerId: "player-1", roundScores: [2, 2, 2, 2, 2] },
      { playerId: "player-2", roundScores: [0, 0, 0, 0, 0] },
      { playerId: "player-3", roundScores: [-1, -1, -1, -1, -1] },
      { playerId: "player-4", roundScores: [-2, -1, -1, -1, -1] }
    ]);
    const html = render(completedMatch(result.players));

    expect(result.players.reduce((sum, player) => sum + player.finalValue, 0)).toBe(0);
    for (const player of result.players) {
      const expectedValues = [
        player.rank,
        player.rawMatchScore,
        player.uma,
        player.score,
        player.finalValue
      ];
      for (const value of expectedValues) {
        const displayed = `${value > 0 && value !== player.rank ? "+" : ""}${value}`;
        expect(html).toContain(displayed);
      }
    }
    expect(html.match(/>1位</g)).toHaveLength(2);
    expect(html.match(/<dd>\+15<\/dd>/g)).toHaveLength(2);
  });

  it("offers and invokes the existing new-match action", () => {
    const onStartNewMatch = vi.fn();
    const container = document.createElement("div");
    const root = createRoot(container);

    act(() => {
      root.render(
        <MatchFinalResults
          match={completedMatch(defaultFinalScores())}
          onStartNewMatch={onStartNewMatch}
          players={players}
        />
      );
    });

    const button = container.querySelector<HTMLButtonElement>(".match-new-game-button");
    expect(button?.textContent).toBe("新しい試合を始める");

    act(() => button?.click());

    expect(onStartNewMatch).toHaveBeenCalledOnce();
    act(() => root.unmount());
  });

  it("keeps the main values readable without a mobile horizontal table", () => {
    const styles = readFileSync("src/styles.css", "utf8");

    expect(styles).toMatch(
      /\.match-final-player \{[\s\S]*?grid-template-columns: minmax\(130px, 0\.8fr\) minmax\(440px, 2fr\);[\s\S]*?\}/
    );
    expect(styles).toMatch(
      /@media \(max-width: 720px\) \{[\s\S]*?\.match-final-values \{[\s\S]*?grid-template-columns: repeat\(2, minmax\(0, 1fr\)\);/
    );
    expect(styles).toMatch(
      /\.app-shell-game-active\.app-shell-match-completed \{[\s\S]*?height: auto;[\s\S]*?overflow: visible;/
    );
  });
});

function render(match: PublicMatchState): string {
  return renderToStaticMarkup(
    <MatchFinalResults match={match} onStartNewMatch={() => undefined} players={players} />
  );
}

function progressMatch(currentRound: number): PublicMatchState {
  return {
    currentRound,
    roundCount: 5,
    completedRoundCount: Math.min(currentRound, 4),
    remainingRounds: 5 - Math.min(currentRound, 4),
    completed: false,
    players: players.map(({ id }) => ({ playerId: id, roundScores: [], rawMatchScore: 0 })),
    finalScores: null
  };
}

function completedMatch(finalScores: readonly PublicMatchPlayerFinalScore[]): PublicMatchState {
  return {
    currentRound: 5,
    roundCount: 5,
    completedRoundCount: 5,
    remainingRounds: 0,
    completed: true,
    players: finalScores,
    finalScores
  };
}

function finalScore(
  playerId: string,
  rank: number,
  rawMatchScore: number,
  uma: number,
  score: number,
  finalValue: number
): PublicMatchPlayerFinalScore {
  return { playerId, rank, roundScores: [1, 2, 3, 4, 5], rawMatchScore, uma, score, finalValue };
}

function defaultFinalScores(): readonly PublicMatchPlayerFinalScore[] {
  return players.map(({ id }, index) =>
    finalScore(id, index + 1, 10 - index * 5, 20 - index * 10, 30 - index * 15, 30 - index * 15)
  );
}
