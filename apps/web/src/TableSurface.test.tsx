// @vitest-environment happy-dom

import { act } from "react";
import { createRoot } from "react-dom/client";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type {
  PublicBidAction,
  PublicGameState,
  PublicMatchState,
  PublicPlayedCard,
  PublicRank,
  PublicStandardCard,
  PublicSuit
} from "@napoleon/protocol";
import { TableSurface, productionTableTestExports } from "./TableSurface";

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
import { projectTablePoint, tableDesignMockLayout } from "./TableDesignMock";
import { fourColorSuitColors } from "./cardSuitTheme";
import { createTablePlayers } from "./tablePlayers";

describe("TableSurface", () => {
  it("maps game-state players to mock seat geometry and opponent hand counts", () => {
    const html = renderTable(createState({ opponentHandCounts: [10, 9, 8, 7] }));

    expect(html).toContain("production-table-surface-playing");
    expect(html).toContain("--mock-page-height:1830px");
    expect(html).toContain("--mock-page-width:2200px");
    expect(html).toContain("投影後の実ゲーム卓");
    expect(html).toContain('<div class="mock-projected-card-layer">');
    expect(countOccurrences(html, "mock-player-info mock-player-info-")).toBe(5);
    expect(html).toContain("左側AI プレイヤー");
    expect(html).toContain("奥左AI プレイヤー");
    expect(html).toContain("奥右AI プレイヤー");
    expect(html).toContain("右側AI プレイヤー");
    expect(html).toContain("自分 プレイヤー");
    expect(html).toContain("自分 プレイヤー 現在の手番");
    expect(html).toContain("production-player-info-current");
    expect(html).toContain("production-riipai-sidebar production-riipai-sidebar-active");
    expect(html).toContain(">理</span><strong>ON</strong>");
    expect(html).not.toContain(">理牌</button>");
    expect(countOccurrences(html, "mock-projected-playing-card-opponent-hand")).toBe(34);
    expect(html).toContain("左側AIの裏向き手札 10枚");
    expect(html).toContain("奥左AIの裏向き手札 9枚");
  });

  it("connects the current trick to each seat and uses cardmeister four-color cards", () => {
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

    expect(html).toContain("production-trick-card-from-left");
    expect(html).toContain("production-trick-card-from-top-left");
    expect(html).toContain("production-trick-card-from-top-right");
    expect(html).toContain("production-trick-card-from-right");
    expect(html).toContain("production-trick-card-from-self");
    expect(html).toContain('class="mock-projected-playing-card mock-playing-card mock-projected-playing-card-trick"');
    expect(html).toContain('aria-label="A♠"');
    expect(html).toContain('aria-label="K♥"');
    expect(html).toContain('aria-label="Q♦"');
    expect(html).toContain('aria-label="J♣"');
    expect(html).toContain('aria-label="10♠"');
    expect(countOccurrences(html, "class=\"mock-cardmeister-playing-card\"")).toBe(8);
    expect(html).toContain(`rankcolor="${fourColorSuitColors.spades},${fourColorSuitColors.hearts},${fourColorSuitColors.diamonds},${fourColorSuitColors.clubs}"`);
  });

  it("connects captured point cards to the projected 10x2 river face", () => {
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

    expect(countOccurrences(html, "mock-projected-river-card-face")).toBe(7);
    expect(html).toContain('aria-label="A♠"');
    expect(html).toContain('aria-label="10♥"');
    expect(html).toContain('aria-label="K♣"');
    expect(html).toContain(`--mock-river-face-color:${fourColorSuitColors.clubs}`);
    expect(html).not.toContain("point-river-empty-mark");
  });

  it("connects role board markers to contract and revealed adjutant state", () => {
    const biddingHtml = renderTable(
      createState({ opponentHandCounts: [10, 10, 10, 10], phase: "bidding" })
    );
    const playingHtml = renderTable(
      createState({
        adjutantRevealedPlayerId: "player-2",
        opponentHandCounts: [9, 9, 9, 9]
      })
    );

    // Role glyph and score are separate <text> elements (two per seat).
    expect(countOccurrences(biddingHtml, "mock-projected-role-marker-text")).toBe(10);
    expect(biddingHtml).toContain(">?</text>");
    // Roles show as compact badge glyphs, not spelled-out role names.
    expect(playingHtml).toContain(">♛</text>");
    expect(playingHtml).toContain(">★</text>");
    expect(playingHtml).toContain(">⚑</text>");
    expect(playingHtml).toContain("mock-projected-role-marker-fill-napoleon");
    expect(playingHtml).toContain("mock-projected-role-marker-fill-adjutant");
    expect(playingHtml).toContain("mock-projected-role-marker-fill-citizen");
    // Score sits in its own scoreboard-styled box, separate from the role badge.
    expect(countOccurrences(playingHtml, "mock-projected-role-score-fill")).toBe(5);
    expect(countOccurrences(playingHtml, "mock-projected-role-marker-score")).toBe(5);
    expect(productionTableTestExports.playerRoleLabel("player-1", createState({ opponentHandCounts: [9, 9, 9, 9] }))).toBe("ナポレオン");
    expect(productionTableTestExports.playerRoleLabel("player-3", createState({ opponentHandCounts: [9, 9, 9, 9] }))).toBe("?");
    expect(productionTableTestExports.playerRoleLabel("player-1", createState({
      adjutantRevealedPlayerId: "player-1",
      opponentHandCounts: [9, 9, 9, 9]
    }))).toBe("ナ/副");
  });

  it("places domain match totals and roles in each projected seat sector with the round at center", () => {
    const state = createState({
      adjutantRevealedPlayerId: "player-2",
      opponentHandCounts: [9, 9, 9, 9]
    });
    const match = progressMatch();
    const html = renderTable(state, null, undefined, undefined, match);

    expect(html).toContain('aria-label="左側AI: 役職 ナポレオン, 累積試合スコア +21"');
    expect(html).toContain('aria-label="奥左AI: 役職 副官, 累積試合スコア -3"');
    expect(html).toContain('aria-label="奥右AI: 役職 市民, 累積試合スコア +13"');
    expect(html).toContain('aria-label="右側AI: 役職 市民, 累積試合スコア +7"');
    expect(html).toContain('aria-label="自分: 役職 市民, 累積試合スコア 0"');
    expect(countOccurrences(html, "mock-projected-role-marker-score")).toBe(5);
    expect(html).toContain('aria-label="現在 第3局 / 全5局"');
    expect(html).toContain(">第3局</text>");
    expect(html).toContain('aria-label="局ごとの得点履歴を表示"');
    expect(html).toContain("<caption>局ごとの得点</caption>");
    expect(html).toContain("<th scope=\"row\">左側AI</th><td>+10</td><td>+11</td>");
    expect(html).toContain("<th scope=\"row\">奥左AI</th><td>-1</td><td>-2</td>");

    const expectedCenter = projectTablePoint({
      x: tableDesignMockLayout.center.x,
      y: tableDesignMockLayout.center.y
    }, tableDesignMockLayout.camera);
    const roundPosition = html.match(
      /<text aria-label="現在 第3局 \/ 全5局" class="production-match-round"[^>]* x="([^"]+)" y="([^"]+)"/
    );

    expect(Number(roundPosition?.[1])).toBeCloseTo(expectedCenter.x);
    expect(Number(roundPosition?.[2])).toBeCloseTo(expectedCenter.y);
  });

  it("keeps match information attached to player ids when seat projection changes", () => {
    const state = createState({
      adjutantRevealedPlayerId: "player-2",
      opponentHandCounts: [9, 9, 9, 9]
    });
    const players = createTablePlayers(state);
    const rotatedSeats = ["top-right", "right", "self", "left", "top-left"] as const;
    const rotatedPlayers = players.map((player, index) => ({
      ...player,
      seat: rotatedSeats[index]
    }));
    const html = renderTable(state, null, undefined, undefined, progressMatch(), rotatedPlayers);
    const topRightMarker = html.match(
      /<g aria-label="左側AI: 役職 ナポレオン, 累積試合スコア \+21" class="mock-projected-role-marker mock-projected-role-marker-top-right">/
    );

    expect(topRightMarker).not.toBeNull();
  });

  it("derives every finished Napoleon-solo role from the public result", () => {
    const html = renderTable(createState({
      opponentHandCounts: [0, 0, 0, 0],
      phase: "finished",
      result: {
        resultType: "standard",
        winner: "napoleon-team",
        napoleonTeamPointCards: 14,
        alliancePointCards: 6,
        targetPointCards: 13,
        napoleonPlayerId: "player-1",
        adjutantPlayerId: null
      }
    }));

    expect(html).toContain("左側AI: 役職 ナポレオン");
    expect(html).toContain("奥左AI: 役職 市民");
    expect(html).toContain("奥右AI: 役職 市民");
    expect(html).toContain("右側AI: 役職 市民");
    expect(html).toContain("自分: 役職 市民");
    expect(countOccurrences(html, "役職 市民")).toBe(4);
  });

  it("preserves contract and called-card status after bidding", () => {
    const html = renderTable(
      createState({
        adjutantCardId: "diamonds-J",
        contractSuit: "clubs",
        opponentHandCounts: [9, 9, 9, 9]
      })
    );

    expect(html).toContain("契約と呼札");
    expect(html).toContain(">宣言</span>");
    expect(html).toContain(">♣13</strong>");
    expect(html).toContain(">副官</span>");
    expect(html).toContain(">J♦</strong>");
    expect(html).toContain(`--mock-bidding-action-color:${fourColorSuitColors.clubs}`);
    expect(html).toContain(`--mock-bidding-action-color:${fourColorSuitColors.diamonds}`);
  });

  it("renders bidding bubbles from latest bidding history using shared suit colors", () => {
    const html = renderTable(
      createState({
        biddingHistory: [
          { type: "bid", playerId: "player-1", suit: "spades", targetPointCards: 14 },
          { type: "pass", playerId: "player-2" },
          { type: "bid", playerId: "player-4", suit: "hearts", targetPointCards: 15 },
          { type: "bid", playerId: "player-0", suit: "clubs", targetPointCards: 13 }
        ],
        legalActions: [
          { type: "bid", suit: "spades", targetPointCards: 16 },
          { type: "bid", suit: "diamonds", targetPointCards: 16 },
          { type: "pass" }
        ],
        opponentHandCounts: [10, 10, 10, 10],
        phase: "bidding"
      })
    );

    expect(html).toContain("production-table-surface-bidding");
    expect(countOccurrences(html, "mock-bidding-bubble mock-bidding-bubble-")).toBe(5);
    expect(html).toContain("左側AI 最新宣言 ♠14");
    expect(html).toContain("奥左AI 最新宣言 PASS");
    expect(html).toContain("右側AI 最新宣言 ♥15");
    expect(html).toContain("自分 最新宣言 ♣13");
    expect(html).toContain(`--mock-bidding-action-color:${fourColorSuitColors.hearts}`);
    expect(html).toContain(`--mock-bidding-action-color:${fourColorSuitColors.clubs}`);
  });

  it("connects bidding overlay controls to legal actions only", () => {
    const legalActions: readonly PublicBidAction[] = [
      { type: "bid", suit: "spades", targetPointCards: 16 },
      { type: "bid", suit: "diamonds", targetPointCards: 16 }
    ];
    const html = renderTable(
      createState({
        biddingHistory: [
          { type: "bid", playerId: "player-4", suit: "spades", targetPointCards: 15 }
        ],
        legalActions: [...legalActions, { type: "pass" }],
        opponentHandCounts: [10, 10, 10, 10],
        phase: "bidding"
      }),
      null,
      legalActions,
      true
    );

    expect(html).toContain("競り操作Overlay");
    expect(html).toContain("現在の最高入札");
    expect(html).toContain(">♠15</strong>");
    expect(html).toContain('aria-label="♠を選択"');
    expect(html).toContain('aria-label="♦を選択"');
    expect(html).toContain('aria-label="♥を選択" aria-pressed="false" class="mock-bidding-suit-button" disabled=""');
    expect(html).toContain('aria-label="♣を選択" aria-pressed="false" class="mock-bidding-suit-button" disabled=""');
    expect(html).toContain('class="mock-bidding-declare-button"');
    expect(html).toContain('class="mock-bidding-pass-button"');
  });

  it("defaults each new turn to the minimal legal raise, discarding a stale manual selection", async () => {
    // Turn 1: highest bid is clubs-13, so the minimal legal raise is diamonds-13.
    const turnOneLegalActions: readonly PublicBidAction[] = [
      { type: "bid", suit: "diamonds", targetPointCards: 13 },
      { type: "bid", suit: "hearts", targetPointCards: 13 },
      { type: "bid", suit: "spades", targetPointCards: 13 },
      { type: "bid", suit: "clubs", targetPointCards: 14 },
      { type: "bid", suit: "diamonds", targetPointCards: 14 },
      { type: "bid", suit: "hearts", targetPointCards: 14 },
      { type: "bid", suit: "spades", targetPointCards: 14 }
    ];
    // Turn 2: someone else raised to diamonds-13 (not the user's manual
    // clubs-14 pick below), so the minimal legal raise is now hearts-13.
    // clubs-14 is still a legal bid here - it just should no longer be
    // selected by default.
    const turnTwoLegalActions: readonly PublicBidAction[] = [
      { type: "bid", suit: "hearts", targetPointCards: 13 },
      { type: "bid", suit: "spades", targetPointCards: 13 },
      { type: "bid", suit: "clubs", targetPointCards: 14 },
      { type: "bid", suit: "diamonds", targetPointCards: 14 },
      { type: "bid", suit: "hearts", targetPointCards: 14 },
      { type: "bid", suit: "spades", targetPointCards: 14 }
    ];
    const state = createState({
      legalActions: [...turnOneLegalActions, { type: "pass" }],
      opponentHandCounts: [10, 10, 10, 10],
      phase: "bidding"
    });

    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);

    await act(async () => {
      root.render(
        <TableSurface
          actionPanel={null}
          canExchange={false}
          canPass={true}
          currentTrick={[]}
          highlightWinningCard={true}
          isBusy={false}
          legalBidActions={turnOneLegalActions}
          legalCardIds={new Set()}
          onBid={vi.fn()}
          onPass={vi.fn()}
          onPlay={vi.fn()}
          onToggleWinningCardHighlight={vi.fn()}
          players={createTablePlayers(state)}
          selectedDiscardCardIds={[]}
          selfPlayerId="player-0"
          state={state}
          trickNumber={1}
          trumpSuit="spades"
        />
      );
    });

    expect(selectedSuitButton(container)?.getAttribute("aria-label")).toBe("♦を選択");
    expect(container.querySelector(".mock-bidding-number-value")?.textContent).toBe("13");

    // The user manually escalates their (unsubmitted) pick to clubs-14.
    await act(async () => {
      container.querySelector<HTMLButtonElement>('[aria-label="♣を選択"]')?.click();
    });

    expect(selectedSuitButton(container)?.getAttribute("aria-label")).toBe("♣を選択");
    expect(container.querySelector(".mock-bidding-number-value")?.textContent).toBe("14");

    // Turn 2 arrives (a new legalBidActions list from a new highest bid).
    await act(async () => {
      root.render(
        <TableSurface
          actionPanel={null}
          canExchange={false}
          canPass={true}
          currentTrick={[]}
          highlightWinningCard={true}
          isBusy={false}
          legalBidActions={turnTwoLegalActions}
          legalCardIds={new Set()}
          onBid={vi.fn()}
          onPass={vi.fn()}
          onPlay={vi.fn()}
          onToggleWinningCardHighlight={vi.fn()}
          players={createTablePlayers(state)}
          selectedDiscardCardIds={[]}
          selfPlayerId="player-0"
          state={state}
          trickNumber={1}
          trumpSuit="spades"
        />
      );
    });

    // Resets to the new minimal legal raise (hearts-13), not the stale
    // clubs-14 pick from before - even though clubs-14 is still legal.
    expect(selectedSuitButton(container)?.getAttribute("aria-label")).toBe("♥を選択");
    expect(container.querySelector(".mock-bidding-number-value")?.textContent).toBe("13");

    await act(async () => root.unmount());
    container.remove();
  });

  it("keeps centered match information available during the bidding phase", () => {
    const html = renderTable(
      createState({
        legalActions: [{ type: "bid", suit: "spades", targetPointCards: 13 }, { type: "pass" }],
        opponentHandCounts: [10, 10, 10, 10],
        phase: "bidding"
      }),
      null,
      undefined,
      undefined,
      progressMatch()
    );

    expect(html).toContain("production-bidding-overlay");
    expect(countOccurrences(html, "mock-projected-role-marker-score")).toBe(5);
    expect(html).toContain('aria-label="現在 第3局 / 全5局"');
    expect(html).toContain("mock-projected-role-marker-left");
    expect(html).toContain("mock-projected-role-marker-self");
  });

  it("keeps self hand operations and card size fixed when cards are removed", () => {
    const fullHtml = renderTable(createState({ opponentHandCounts: [10, 10, 10, 10] }));
    const reducedHtml = renderTable(
      createState({
        selfHand: [standardCard("spades", "A")],
        opponentHandCounts: [10, 10, 10, 10]
      })
    );

    expect(countOccurrences(fullHtml, "mock-self-hand-card")).toBe(3);
    expect(countOccurrences(reducedHtml, "mock-self-hand-card")).toBe(1);
    expect(fullHtml).toContain("--mock-self-card-width:135.3846153846154px");
    expect(reducedHtml).toContain("--mock-self-card-width:135.3846153846154px");
    expect(fullHtml).toContain("production-card-blocked");
    expect(fullHtml).toContain('cid="As"');
    expect(fullHtml).toContain('cid="Kh"');
  });

  it("renders ten production cards in a single row inside a stable fixed-footprint hand region", () => {
    const tenCards = (["A", "K", "Q", "J", "10", "9", "8", "7", "6", "5"] as const)
      .map((rank) => standardCard("spades", rank));
    const fullHtml = renderTable(createState({
      opponentHandCounts: [10, 10, 10, 10],
      selfHand: tenCards
    }));
    const reducedHtml = renderTable(createState({
      opponentHandCounts: [8, 8, 8, 8],
      selfHand: tenCards.slice(0, 2)
    }));
    const stylePattern = /aria-label="自分の手札" class="mock-self-hand production-self-hand" style="([^"]+)"/;
    const fullStyle = fullHtml.match(stylePattern)?.[1] ?? "";
    const reducedStyle = reducedHtml.match(stylePattern)?.[1] ?? "";
    // Footprint bounds (sized for the 13-card maximum) must stay fixed regardless of hand size;
    // only --mock-self-content-left (which centers the actual cards inside that footprint) may
    // change with the card count.
    const fixedBounds = (style: string) => style.match(
      /--mock-self-hand-height:[^;]+;--mock-self-hand-left:[^;]+;--mock-self-hand-top:[^;]+;--mock-self-hand-width:[^;]+/
    )?.[0];
    const cardIndexes = (html: string) => Array.from(
      html.matchAll(/mock-self-hand-card[^>]*--mock-self-card-index:(\d+)/g)
    ).map((match) => Number(match[1]));

    expect(countOccurrences(fullHtml, "mock-self-hand-card")).toBe(10);
    expect(fullStyle).toContain("--mock-self-card-step:");
    expect(fixedBounds(fullStyle)).toBeDefined();
    expect(fixedBounds(fullStyle)).toBe(fixedBounds(reducedStyle));
    expect(cardIndexes(fullHtml)).toEqual(Array.from({ length: 10 }, (_, index) => index));
  });

  it("keeps exchange/adjutant/result action panels available outside bidding", () => {
    const exchangeHand = (
      ["A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"] as const
    ).map((rank) => standardCard("spades", rank));
    const html = renderTable(
      createState({
        opponentHandCounts: [9, 9, 9, 9],
        phase: "exchanging",
        selfHand: exchangeHand
      }),
      <section className="exchange-panel" aria-label="埋札交換">
        <button className="secondary-button" type="button">捨てる</button>
      </section>
    );

    expect(html).toContain("production-action-overlay");
    expect(html).toContain("埋札交換");
    expect(html).toContain("捨てる");
    expect(countOccurrences(html, "mock-self-hand-card")).toBe(13);
    expect(html).toContain("--mock-self-card-step:");
    expect(countOccurrences(html, "production-card-selectable")).toBe(13);
  });
});

function renderTable(
  state: PublicGameState,
  actionPanel: React.ReactNode = null,
  legalBidActions: readonly PublicBidAction[] = state.legalActions.filter((action): action is PublicBidAction => action.type === "bid"),
  canPass = state.legalActions.some((action) => action.type === "pass"),
  match?: PublicMatchState,
  players = createTablePlayers(state)
): string {
  return renderToStaticMarkup(
    <TableSurface
      actionPanel={actionPanel}
      canExchange={state.phase === "exchanging"}
      canPass={canPass}
      currentTrick={state.currentTrick}
      highlightWinningCard={true}
      isBusy={false}
      legalBidActions={legalBidActions}
      legalCardIds={new Set(state.legalActions.filter((action) => action.type === "play-card").map((action) => action.cardId))}
      match={match}
      onBid={vi.fn()}
      onPass={vi.fn()}
      onPlay={vi.fn()}
      onToggleWinningCardHighlight={vi.fn()}
      players={players}
      selectedDiscardCardIds={[]}
      selfPlayerId={state.self.id}
      state={state}
      trickNumber={state.trickNumber}
      trumpSuit={state.trumpSuit}
    />
  );
}

function progressMatch(): PublicMatchState {
  const scores = new Map<string, readonly number[]>([
    ["player-0", [0, 0]],
    ["player-1", [10, 11]],
    ["player-2", [-1, -2]],
    ["player-3", [6, 7]],
    ["player-4", [3, 4]]
  ]);

  return {
    currentRound: 3,
    roundCount: 5,
    completedRoundCount: 2,
    remainingRounds: 3,
    completed: false,
    players: ["player-4", "player-2", "player-0", "player-1", "player-3"].map((playerId) => ({
      playerId,
      roundScores: scores.get(playerId) ?? [],
      rawMatchScore: (scores.get(playerId) ?? []).reduce((sum, score) => sum + score, 0)
    })),
    finalScores: null
  };
}

function createState({
  capturedPointCards = {},
  currentTrick = [],
  adjutantCardId = "spades-A",
  adjutantRevealedPlayerId = null,
  biddingHistory = [],
  contractSuit = "spades",
  isTrickComplete = false,
  legalActions = [],
  phase = "playing",
  result = null,
  opponentHandCounts,
  selfHand = [
    standardCard("spades", "A"),
    standardCard("hearts", "K"),
    standardCard("diamonds", "Q")
  ]
}: {
  adjutantCardId?: string;
  adjutantRevealedPlayerId?: string | null;
  biddingHistory?: NonNullable<PublicGameState["bidding"]>["history"];
  capturedPointCards?: Partial<Record<string, readonly PublicStandardCard[]>>;
  contractSuit?: PublicSuit;
  currentTrick?: readonly PublicPlayedCard[];
  isTrickComplete?: boolean;
  legalActions?: PublicGameState["legalActions"];
  phase?: PublicGameState["phase"];
  result?: PublicGameState["result"];
  opponentHandCounts: readonly [number, number, number, number];
  selfHand?: PublicGameState["self"]["hand"];
}): PublicGameState {
  return {
    self: {
      id: "player-0",
      handCount: selfHand.length,
      hand: selfHand,
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
    result,
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
    exchange: phase === "exchanging" ? { napoleonPlayerId: "player-0", requiredDiscardCount: 3 } : null,
    adjutantChoice: null,
    currentPlayerId: "player-0",
    currentTrick,
    completedTrickCount: 0,
    trickNumber: 1,
    isTrickComplete,
    isGameOver: false,
    legalActions
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

function selectedSuitButton(container: HTMLElement): Element | null {
  return container.querySelector('.mock-bidding-suit-button[aria-pressed="true"]');
}
