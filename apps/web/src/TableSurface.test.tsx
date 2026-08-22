import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type {
  PublicBidAction,
  PublicGameState,
  PublicPlayedCard,
  PublicRank,
  PublicStandardCard,
  PublicSuit
} from "@napoleon/protocol";
import { TableSurface, productionTableTestExports } from "./TableSurface";
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

    expect(countOccurrences(biddingHtml, "mock-projected-role-marker-text")).toBe(5);
    expect(biddingHtml).toContain(">?</text>");
    expect(playingHtml).toContain(">ナポ</text>");
    expect(playingHtml).toContain(">副</text>");
    expect(playingHtml).toContain(">市</text>");
    expect(productionTableTestExports.playerRoleLabel("player-1", createState({ opponentHandCounts: [9, 9, 9, 9] }))).toBe("ナポ");
    expect(productionTableTestExports.playerRoleLabel("player-3", createState({ opponentHandCounts: [9, 9, 9, 9] }))).toBe("?");
    expect(productionTableTestExports.playerRoleLabel("player-1", createState({
      adjutantRevealedPlayerId: "player-1",
      opponentHandCounts: [9, 9, 9, 9]
    }))).toBe("ナ/副");
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

  it("keeps exchange/adjutant/result action panels available outside bidding", () => {
    const html = renderTable(
      createState({ opponentHandCounts: [9, 9, 9, 9], phase: "exchanging" }),
      <section className="exchange-panel" aria-label="埋札交換">
        <button className="secondary-button" type="button">捨てる</button>
      </section>
    );

    expect(html).toContain("production-action-overlay");
    expect(html).toContain("埋札交換");
    expect(html).toContain("捨てる");
  });
});

function renderTable(
  state: PublicGameState,
  actionPanel: React.ReactNode = null,
  legalBidActions: readonly PublicBidAction[] = state.legalActions.filter((action): action is PublicBidAction => action.type === "bid"),
  canPass = state.legalActions.some((action) => action.type === "pass")
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
      onBid={vi.fn()}
      onPass={vi.fn()}
      onPlay={vi.fn()}
      onToggleWinningCardHighlight={vi.fn()}
      players={createTablePlayers(state)}
      selectedDiscardCardIds={[]}
      selfPlayerId={state.self.id}
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
  legalActions = [],
  phase = "playing",
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
