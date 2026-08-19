import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type {
  PublicGameState,
  PublicPlayedCard,
  PublicRank,
  PublicStandardCard,
  PublicSuit
} from "@napoleon/protocol";
import { TableSurface } from "./TableSurface";
import { createTablePlayers } from "./tablePlayers";

describe("TableSurface", () => {
  it("renders opponent hands as one facedown card per remaining card", () => {
    const initialHtml = renderTable(createState({ opponentHandCounts: [10, 9, 8, 7] }));
    const afterPlayHtml = renderTable(createState({ opponentHandCounts: [9, 9, 8, 7] }));

    expect(countOccurrences(initialHtml, "class=\"card-back\"")).toBe(34);
    expect(countOccurrences(afterPlayHtml, "class=\"card-back\"")).toBe(33);
    expect(initialHtml).toContain("左側AIの裏向き手札 10枚");
    expect(afterPlayHtml).toContain("左側AIの裏向き手札 9枚");
  });

  it("anchors current trick cards to the seat that played them", () => {
    const html = renderTable(
      createState({
        currentTrick: [
          playedCard("player-1", "spades", "A"),
          playedCard("player-2", "hearts", "K"),
          playedCard("player-3", "diamonds", "Q"),
          playedCard("player-4", "clubs", "J"),
          playedCard("player-0", "spades", "10")
        ],
        opponentHandCounts: [9, 9, 9, 9]
      })
    );

    expect(html).toContain("table-card-from-left");
    expect(html).toContain("table-card-from-top-left");
    expect(html).toContain("table-card-from-top-right");
    expect(html).toContain("table-card-from-right");
    expect(html).toContain("table-card-from-self");
    expect(html).toContain("左側AIがA♠を出しました");
    expect(html).toContain("奥左AIがK♥を出しました");
    expect(html).toContain("奥右AIがQ♦を出しました");
    expect(html).toContain("右側AIがJ♣を出しました");
    expect(html).toContain("自分が10♠を出しました");
  });

  it("renders a compact point river inside every seat", () => {
    const html = renderTable(
      createState({
        capturedPointCards: {
          "player-1": [
            standardCard("spades", "A"),
            standardCard("spades", "K"),
            standardCard("spades", "Q"),
            standardCard("hearts", "A"),
            standardCard("hearts", "K"),
            standardCard("hearts", "10")
          ],
          "player-0": [standardCard("clubs", "K")]
        },
        opponentHandCounts: [8, 8, 8, 8]
      })
    );

    expect(countOccurrences(html, "class=\"point-river ")).toBe(5);
    expect(countOccurrences(html, "class=\"point-river-stack")).toBe(5);
    expect(countOccurrences(html, "point-river-empty-mark")).toBe(3);
    expect(countOccurrences(html, "table-point-card")).toBe(7);
    expect(html).not.toContain("★");
    expect(html).toContain("河 <strong>6</strong>");
    expect(html).toContain("河 <strong>1</strong>");
    expect(html).toContain("左側AIの獲得ポイント札 6枚");
    expect(html).toContain("自分の獲得ポイント札 1枚");
  });

  it("renders role markers as compact table objects", () => {
    const unknownHtml = renderTable(
      createState({ opponentHandCounts: [10, 10, 10, 10], phase: "bidding" })
    );
    const confirmedHtml = renderTable(
      createState({
        adjutantRevealedPlayerId: "player-2",
        opponentHandCounts: [9, 9, 9, 9]
      })
    );

    expect(countOccurrences(unknownHtml, "class=\"table-role-marker")).toBe(5);
    expect(countOccurrences(unknownHtml, "の役職?")).toBe(5);
    expect(unknownHtml).toContain("table-role-marker-unknown");
    expect(confirmedHtml).toContain("左側AIの役職N");
    expect(confirmedHtml).toContain("奥左AIの役職A");
    expect(confirmedHtml).toContain("table-role-marker-napoleon");
    expect(confirmedHtml).toContain("table-role-marker-adjutant");
  });

  it("keeps global contract details in the table HUD", () => {
    const html = renderTable(
      createState({
        contractSuit: "hearts",
        adjutantCardId: "diamonds-J",
        adjutantRevealedPlayerId: "player-2",
        opponentHandCounts: [9, 9, 9, 9]
      })
    );
    const hudHtml = html.slice(
      html.indexOf("class=\"table-hud\""),
      html.indexOf("<aside", html.indexOf("class=\"table-hud\""))
    );

    expect(hudHtml).toContain("♥13");
    expect(hudHtml).toContain("契約");
    expect(hudHtml).toContain("呼札 ");
    expect(hudHtml).toContain("♦J");
    expect(hudHtml).toContain("red-text");
    expect(hudHtml).not.toContain("左側AI");
    expect(hudHtml).not.toContain("自分");
    expect(hudHtml).not.toContain("N");
    expect(hudHtml).not.toContain("A");
  });

  it("uses table seat lanes for bidding declarations", () => {
    const html = renderTable(
      createState({
        biddingHistory: [
          { type: "bid", playerId: "player-1", suit: "spades", targetPointCards: 14 },
          { type: "pass", playerId: "player-2" },
          { type: "bid", playerId: "player-4", suit: "hearts", targetPointCards: 15 }
        ],
        opponentHandCounts: [10, 10, 10, 10],
        phase: "bidding"
      })
    );

    expect(html).toContain("table-surface-bidding");
    expect(countOccurrences(html, "class=\"table-seat-guide\"")).toBe(5);
    expect(countOccurrences(html, "class=\"table-bid-token")).toBe(5);
    expect(html).toContain("左側AIの競り宣言");
    expect(html).toContain("♠");
    expect(html).toContain(">14</strong>");
    expect(html).toContain(">Pass</strong>");
    expect(html).toContain("table-bid-token-red");
    expect(html).toContain(">?</strong>");
  });

  it("renders self hand cards with playing-card face structure", () => {
    const html = renderTable(createState({ opponentHandCounts: [10, 10, 10, 10] }));

    expect(html).toContain("class=\"card-corner card-corner-top\">A♠");
    expect(html).toContain("class=\"card-suit card-center-mark\">♠");
    expect(html).toContain("class=\"card-corner card-corner-bottom\">A♠");
    expect(html).toContain("class=\"card-corner card-corner-top\">K♥");
    expect(html).toContain("card-red");
    expect(html).toContain("card-black");
  });

  it("keeps seat placement separate from stable card orientations", () => {
    const styles = readFileSync(fileURLToPath(new URL("./styles.css", import.meta.url)), "utf8");

    expect(styles).not.toContain("--axis-angle");
    expect(styles).not.toContain("--name-x");
    expect(styles).not.toContain("--hand-x");
    expect(styles).not.toContain("--trick-x");
    expect(styles).not.toContain("--river-x");
    expect(styles).not.toContain("--role-x");
    expect(styles).not.toContain("--zone-x");
    expect(styles).not.toContain("--zone-rotation");
    expect(getCssRule(styles, ".table-hand-zone")).toContain(
      "transform: rotate(var(--seat-hand-rotation));"
    );
    expect(getCssRule(styles, ".table-trick-zone")).toContain(
      "transform: rotate(var(--seat-trick-rotation));"
    );
    expect(getCssRule(styles, ".table-river-zone", 1)).toContain(
      "transform: rotate(var(--seat-river-rotation));"
    );
    expect(getCssRule(styles, ".table-player-left")).not.toContain("--seat-hand-rotation:");
    expect(getCssRule(styles, ".table-player-right")).not.toContain("--seat-hand-rotation:");
    expect(getCssRule(styles, ".table-player-top-left")).not.toContain("--seat-hand-rotation:");
    expect(getCssRule(styles, ".table-player-top-right")).not.toContain("--seat-hand-rotation:");
    expect(styles).toContain("flex-direction: column;\n  max-height: 156px;");
  });

  it("keeps permanent and contextual controls in the left side action rail", () => {
    const html = renderTable(
      createState({ opponentHandCounts: [10, 10, 10, 10] }),
      <div className="action-area">
        <button
          aria-label="次のトリックへ進む"
          className="secondary-button next-trick-button"
          type="button"
        >
          次へ
        </button>
      </div>
    );
    const asideStart = html.indexOf("class=\"table-side-actions\"");
    const seatsStart = html.indexOf("class=\"table-seat-container");
    const asideHtml = html.slice(asideStart, seatsStart);

    expect(asideHtml).toContain("aria-label=\"操作\"");
    expect(asideHtml).toContain("class=\"table-side-tools\"");
    expect(asideHtml).toContain("理牌");
    expect(asideHtml).toContain("勝札");
    expect(asideHtml).toContain("next-trick-button");
    expect(asideHtml).toContain("次のトリックへ進む");
    expect(asideHtml.indexOf("table-side-tools")).toBeLessThan(
      asideHtml.indexOf("next-trick-button")
    );
  });

  it("only renders the next trick action when the completed trick needs clearing", () => {
    const appSource = readFileSync(fileURLToPath(new URL("./App.tsx", import.meta.url)), "utf8");
    const styles = readFileSync(fileURLToPath(new URL("./styles.css", import.meta.url)), "utf8");

    expect(appSource).toContain(
      'session?.state.phase === "playing" &&\n                session.state.isTrickComplete &&\n                !session.state.isGameOver'
    );
    expect(appSource).toContain("disabled={isInteractionLocked}");
    expect(styles).toContain(".table-side-actions .next-trick-button");
    expect(styles).toContain("white-space: nowrap;");
  });

  it("keeps completed trick controls attached to the self seat", () => {
    const html = renderTable(
      createState({ isTrickComplete: true, opponentHandCounts: [9, 9, 9, 9] }),
      <div className="action-area">
        <button
          aria-label="次のトリックへ進む"
          className="secondary-button next-trick-button"
          type="button"
        >
          次へ
        </button>
      </div>
    );
    const asideStart = html.indexOf("class=\"table-side-actions\"");
    const seatsStart = html.indexOf("class=\"table-seat-container");
    const asideHtml = html.slice(asideStart, seatsStart);
    const selfSeatStart = html.indexOf("class=\"table-seat-container table-player-self");
    const selfSeatEnd = html.indexOf("<div class=\"table-core\"", selfSeatStart);
    const selfSeatHtml = html.slice(selfSeatStart, selfSeatEnd);

    expect(asideHtml).not.toContain("next-trick-button");
    expect(selfSeatHtml).toContain("class=\"table-seat-actions\"");
    expect(selfSeatHtml).toContain("理牌");
    expect(selfSeatHtml).toContain("勝札");
    expect(selfSeatHtml).toContain("next-trick-button");
  });

  it("groups each player as a concrete seat container and hides unused trick slots while bidding", () => {
    const styles = readFileSync(fileURLToPath(new URL("./styles.css", import.meta.url)), "utf8");
    const html = renderTable(createState({ opponentHandCounts: [10, 10, 10, 10] }));

    expect(countOccurrences(html, "class=\"table-seat-container")).toBe(5);
    expect(countOccurrences(html, "class=\"table-seat-header\"")).toBe(5);
    expect(html.indexOf("class=\"table-seat-header\"")).toBeLessThan(
      html.indexOf("class=\"table-hand-zone\"")
    );
    expect(styles).toContain(".table-surface::before,");
    expect(getCssRule(styles, ".table-surface::before")).toContain("height: min(56vw, 400px);");
    expect(getCssRule(styles, ".table-seat-container")).toContain("display: flex;");
    expect(getCssRule(styles, ".table-seat-guide")).toContain("inset: -6px;");
    expect(getCssRule(styles, ".table-player-left")).toContain("flex-direction: row;");
    expect(getCssRule(styles, ".table-player-right")).toContain("flex-direction: row-reverse;");
    expect(getCssRule(styles, ".table-player-self")).toContain("flex-direction: column-reverse;");
    expect(styles).toContain("position: relative;\n  transform-origin: center;");
    expect(getCssRule(styles, ".table-role-marker")).toContain("position: relative;");
    expect(getCssRule(styles, ".table-trick-card")).toContain("height: 78px;");
    expect(getCssRule(styles, ".point-river")).toContain("display: inline-grid;");
    expect(styles).toContain("background: rgb(255 255 255 / 10%);");
    expect(styles).toContain(".table-side-actions .adjutant-controls label,");
    expect(styles).toContain("color: rgb(226 232 240 / 88%);");
    expect(getCssRule(styles, ".table-surface-bidding .table-trick-zone")).toContain(
      "display: none;"
    );
  });
});

function renderTable(state: PublicGameState, actionPanel: React.ReactNode = null): string {
  return renderToStaticMarkup(
    <TableSurface
      actionPanel={actionPanel}
      canExchange={false}
      currentTrick={state.currentTrick}
      highlightWinningCard={true}
      isBusy={false}
      legalCardIds={new Set()}
      onPlay={vi.fn()}
      onToggleWinningCardHighlight={vi.fn()}
      players={createTablePlayers(state)}
      selectedDiscardCardIds={[]}
      state={state}
      trickNumber={state.trickNumber}
      trumpSuit={state.trumpSuit}
    />
  );
}

function createState({
  capturedPointCards = {},
  currentTrick = [],
  adjutantCardId = "spades-A",
  adjutantRevealedPlayerId = null,
  biddingHistory = [],
  contractSuit = "spades",
  isTrickComplete = false,
  phase = "playing",
  opponentHandCounts
}: {
  adjutantCardId?: string;
  adjutantRevealedPlayerId?: string | null;
  biddingHistory?: NonNullable<PublicGameState["bidding"]>["history"];
  capturedPointCards?: Partial<Record<string, readonly PublicStandardCard[]>>;
  contractSuit?: PublicSuit;
  currentTrick?: readonly PublicPlayedCard[];
  isTrickComplete?: boolean;
  phase?: PublicGameState["phase"];
  opponentHandCounts: readonly [number, number, number, number];
}): PublicGameState {
  return {
    self: {
      id: "player-0",
      handCount: 10,
      hand: [
        standardCard("spades", "A"),
        standardCard("hearts", "K"),
        standardCard("diamonds", "Q")
      ],
      capturedPointCards: capturedPointCards["player-0"] ?? []
    },
    opponents: opponentHandCounts.map((handCount, index) => ({
      id: `player-${index + 1}`,
      handCount,
      capturedPointCards: capturedPointCards[`player-${index + 1}`] ?? []
    })),
    phase,
    trumpSuit: "spades",
    contract:
      phase === "bidding"
        ? null
        : {
            napoleonPlayerId: "player-1",
            trumpSuit: contractSuit,
            targetPointCards: 13
          },
    specialCards: {
      orumaCardId: "spades-A",
      yoromekiCardId: "hearts-Q",
      seiJackCardId: "spades-J",
      uraJackCardId: "clubs-J"
    },
    adjutant:
      phase === "bidding"
        ? null
        : { calledCardId: adjutantCardId, revealedPlayerId: adjutantRevealedPlayerId },
    latestEvent: null,
    result: null,
    bidding:
      phase === "bidding"
        ? {
            starterPlayerId: "player-0",
            highestBid:
              [...biddingHistory].reverse().find((entry) => entry.type === "bid") ?? null,
            consecutivePassCount: 0,
            history: biddingHistory
          }
        : null,
    exchange: null,
    adjutantChoice: null,
    currentPlayerId: "player-0",
    currentTrick,
    completedTrickCount: 0,
    trickNumber: 1,
    isTrickComplete,
    isGameOver: false,
    legalActions: []
  };
}

function playedCard(playerId: string, suit: PublicSuit, rank: PublicRank): PublicPlayedCard {
  return {
    playerId,
    card: standardCard(suit, rank)
  };
}

function standardCard(suit: PublicSuit, rank: PublicRank): PublicStandardCard {
  return {
    type: "standard",
    id: `${suit}-${rank}`,
    suit,
    rank
  };
}

function countOccurrences(value: string, needle: string): number {
  return value.split(needle).length - 1;
}

function getCssRule(styles: string, selector: string, occurrence = 0): string {
  let start = -1;
  let searchFrom = 0;

  for (let index = 0; index <= occurrence; index += 1) {
    start = styles.indexOf(`${selector} {`, searchFrom);
    expect(start).toBeGreaterThanOrEqual(0);
    searchFrom = start + selector.length;
  }

  expect(start).toBeGreaterThanOrEqual(0);
  const end = styles.indexOf("\n}", start);
  expect(end).toBeGreaterThan(start);

  return styles.slice(start, end);
}
