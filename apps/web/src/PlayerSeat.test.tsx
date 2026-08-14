import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { PublicGameState, PublicRank, PublicStandardCard, PublicSuit } from "@napoleon/protocol";
import { PlayerSeat } from "./PlayerSeat";
import type { TablePlayer } from "./tableTypes";

describe("PlayerSeat", () => {
  it("does not expose internal player ids in the visible seat UI", () => {
    const html = renderToStaticMarkup(
      <PlayerSeat player={createPlayer()} state={createState()} />
    );

    expect(html).not.toContain("player-1");
    expect(html).toContain("aria-label=\"左側AIの手札は残り10枚\"");
    expect(html).toContain("aria-label=\"左側AIの獲得得点札は0枚\"");
  });

  it("keeps the mobile in-progress opponent panels in far-side then side order", () => {
    const styles = readFileSync(fileURLToPath(new URL("./styles.css", import.meta.url)), "utf8");
    const match = styles.match(
      /\.app-shell-game-in-progress \.table-grid \{[\s\S]*?grid-template-areas:\n([\s\S]*?)grid-template-rows:/
    );

    expect(match?.[1]).toContain('"top-left top-right"\n      "left right"\n      "center center";');
  });

  it("reserves ten captured point card slots for opponent panels", () => {
    const emptyHtml = renderToStaticMarkup(
      <PlayerSeat player={createPlayer()} state={createState()} />
    );
    const capturedCards = [
      pointCard("spades", "A"),
      pointCard("hearts", "K"),
      pointCard("diamonds", "Q")
    ];
    const capturedHtml = renderToStaticMarkup(
      <PlayerSeat player={createPlayer(capturedCards)} state={createState()} />
    );

    expect(emptyHtml).toContain("なし");
    expect(countOccurrences(emptyHtml, "class=\"point-card-empty-slot\"")).toBe(10);
    expect(countOccurrences(capturedHtml, "class=\"mini-card ")).toBe(3);
    expect(countOccurrences(capturedHtml, "class=\"point-card-empty-slot\"")).toBe(7);
  });

  it("defines a five by two mobile captured point card grid", () => {
    const styles = readFileSync(fileURLToPath(new URL("./styles.css", import.meta.url)), "utf8");
    const match = styles.match(
      /\.app-shell-game-in-progress \.captured-compact \.compact-points \{[\s\S]*?grid-template-columns: ([^;]+);[\s\S]*?grid-template-rows: ([^;]+);/
    );

    expect(match?.[1]).toBe("repeat(5, minmax(0, 1fr))");
    expect(match?.[2]).toBe("repeat(2, 12px)");
  });
});

function createPlayer(capturedPointCards: readonly PublicStandardCard[] = []): TablePlayer {
  return {
    id: "player-1",
    label: "左側AI",
    seat: "left",
    handCount: 10,
    capturedPointCards,
    isSelf: false,
    biddingDeclaration: undefined
  };
}

function pointCard(suit: PublicSuit, rank: PublicRank): PublicStandardCard {
  return {
    type: "standard",
    id: `${suit}-${rank}`,
    suit,
    rank
  };
}

function countOccurrences(value: string, pattern: string): number {
  return value.split(pattern).length - 1;
}

function createState(): PublicGameState {
  return {
    self: {
      id: "player-0",
      handCount: 10,
      hand: [],
      capturedPointCards: []
    },
    opponents: [{ id: "player-1", handCount: 10, capturedPointCards: [] }],
    phase: "playing",
    trumpSuit: "spades",
    contract: {
      napoleonPlayerId: "player-0",
      trumpSuit: "spades",
      targetPointCards: 13
    },
    specialCards: {
      orumaCardId: "spades-A",
      yoromekiCardId: "hearts-Q",
      seiJackCardId: "spades-J",
      uraJackCardId: "clubs-J"
    },
    adjutant: null,
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
