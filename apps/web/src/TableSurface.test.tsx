import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type {
  PublicGameState,
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
  opponentHandCounts
}: {
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
      capturedPointCards: []
    },
    opponents: opponentHandCounts.map((handCount, index) => ({
      id: `player-${index + 1}`,
      handCount,
      capturedPointCards: []
    })),
    phase: "playing",
    trumpSuit: "spades",
    contract: {
      napoleonPlayerId: "player-1",
      trumpSuit: "spades",
      targetPointCards: 13
    },
    specialCards: {
      orumaCardId: "spades-A",
      yoromekiCardId: "hearts-Q",
      seiJackCardId: "spades-J",
      uraJackCardId: "clubs-J"
    },
    adjutant: { calledCardId: "spades-A", revealedPlayerId: null },
    latestEvent: null,
    result: null,
    bidding: null,
    exchange: null,
    adjutantChoice: null,
    currentPlayerId: "player-0",
    currentTrick: [],
    completedTrickCount: 0,
    trickNumber: 1,
    isTrickComplete: false,
    isGameOver: false,
    legalActions: []
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
