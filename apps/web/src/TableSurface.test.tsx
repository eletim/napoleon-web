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

  it("reserves a fixed twenty-card point river for every player", () => {
    const html = renderTable(
      createState({
        capturedPointCards: {
          "player-1": [standardCard("spades", "A"), standardCard("hearts", "10")],
          "player-0": [standardCard("clubs", "K")]
        },
        opponentHandCounts: [8, 8, 8, 8]
      })
    );

    expect(countOccurrences(html, "class=\"point-river-grid\"")).toBe(5);
    expect(countOccurrences(html, "point-river-slot-empty")).toBe(97);
    expect(html).not.toContain("★");
    expect(html).toContain("左側AIの獲得ポイント札 2枚");
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
    expect(hudHtml).toContain("副官 ");
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
});

function renderTable(state: PublicGameState): string {
  return renderToStaticMarkup(
    <TableSurface
      actionPanel={null}
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
  phase = "playing",
  opponentHandCounts
}: {
  adjutantCardId?: string;
  adjutantRevealedPlayerId?: string | null;
  biddingHistory?: NonNullable<PublicGameState["bidding"]>["history"];
  capturedPointCards?: Partial<Record<string, readonly PublicStandardCard[]>>;
  contractSuit?: PublicSuit;
  currentTrick?: readonly PublicPlayedCard[];
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
    isTrickComplete: false,
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
