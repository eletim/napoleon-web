import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type {
  PublicGameState,
  PublicRank,
  PublicStandardCard,
  PublicSuit
} from "@napoleon/protocol";
import { PlayerSeat } from "./PlayerSeat";
import type { TablePlayer } from "./tableTypes";

describe("PlayerSeat", () => {
  it("shows only the opponent name, roles, and captured point cards", () => {
    const html = renderToStaticMarkup(
      <PlayerSeat
        player={createPlayer([pointCard("spades", "A")])}
        state={createState({
          contract: {
            napoleonPlayerId: "player-1",
            trumpSuit: "spades",
            targetPointCards: 13
          },
          adjutant: { calledCardId: "spades-A", revealedPlayerId: "player-1" }
        })}
      />
    );

    expect(html).not.toContain("player-1");
    expect(html).not.toContain("手札");
    expect(html).not.toContain("compact-hand");
    expect(html).not.toContain("small-card-back");
    expect(html).not.toContain("latest-bid-declaration");
    expect(html).toContain("<h2>左側AI</h2>");
    expect(html).toContain("aria-label=\"ナポレオン\"");
    expect(html).toContain("aria-label=\"副官\"");
    expect(html).toContain("aria-label=\"左側AIの獲得得点札は1枚\"");
    expect(html).toContain("A♠");
  });

  it("places mobile opponent seats around the center table", () => {
    const styles = readFileSync(fileURLToPath(new URL("./styles.css", import.meta.url)), "utf8");
    const match = styles.match(
      /\.app-shell-game-in-progress \.table-grid \{[\s\S]*?grid-template-areas:\n([\s\S]*?)grid-template-columns:/
    );

    expect(match?.[1]).toContain(
      '"top-left center center top-right"\n      "left center center right";'
    );
  });

  it("does not reserve fixed captured point card slots for empty seats", () => {
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
    expect(countOccurrences(emptyHtml, "class=\"point-card-empty-slot\"")).toBe(0);
    expect(countOccurrences(capturedHtml, "class=\"mini-card ")).toBe(3);
    expect(countOccurrences(capturedHtml, "class=\"point-card-empty-slot\"")).toBe(0);
  });

  it("keeps captured point cards visible beyond the fixed ten-slot viewport", () => {
    const capturedCards = [
      pointCard("spades", "A"),
      pointCard("spades", "K"),
      pointCard("spades", "Q"),
      pointCard("spades", "J"),
      pointCard("spades", "10"),
      pointCard("hearts", "A"),
      pointCard("hearts", "K"),
      pointCard("hearts", "Q"),
      pointCard("hearts", "J"),
      pointCard("hearts", "10"),
      pointCard("diamonds", "A")
    ];
    const html = renderToStaticMarkup(
      <PlayerSeat player={createPlayer(capturedCards)} state={createState()} />
    );

    expect(countOccurrences(html, "class=\"mini-card ")).toBe(11);
    expect(html).toContain("♦");
    expect(countOccurrences(html, "class=\"point-card-empty-slot\"")).toBe(0);
  });

  it("removes mobile in-progress opponent panel chrome", () => {
    const styles = readFileSync(fileURLToPath(new URL("./styles.css", import.meta.url)), "utf8");
    const seatMatch = styles.match(
      /\.app-shell-game-in-progress \.player-seat \{([\s\S]*?)\n  \}/
    );
    const capturedBlocks = Array.from(
      styles.matchAll(/\.app-shell-game-in-progress \.captured-compact \{([\s\S]*?)\n  \}/g),
      (match) => match[1]
    );

    expect(seatMatch?.[1]).toContain("background: transparent;");
    expect(seatMatch?.[1]).toContain("border: 0;");
    expect(seatMatch?.[1]).toContain("box-shadow: none;");
    expect(seatMatch?.[1]).toContain("min-height: 0;");
    expect(capturedBlocks.some((block) => block.includes("grid-template-columns: 1fr;"))).toBe(
      true
    );
    expect(capturedBlocks.every((block) => block.includes("min-height: 0;"))).toBe(true);
  });

  it("clips mobile captured point cards inside the opponent seat width", () => {
    const styles = readFileSync(fileURLToPath(new URL("./styles.css", import.meta.url)), "utf8");
    const compactPointsBlocks = Array.from(
      styles.matchAll(
        /\.app-shell-game-in-progress \.captured-compact \.compact-points \{([\s\S]*?)\n  \}/g
      ),
      (match) => match[1]
    );

    expect(compactPointsBlocks.some((block) => block.includes("max-width: 100%;"))).toBe(true);
    expect(compactPointsBlocks.some((block) => block.includes("min-width: 0;"))).toBe(true);
    expect(compactPointsBlocks.some((block) => block.includes("width: 100%;"))).toBe(true);
  });

  it("keeps mobile opponent seats close to the trick card band", () => {
    const styles = readFileSync(fileURLToPath(new URL("./styles.css", import.meta.url)), "utf8");
    const sideSeatMatch = styles.match(
      /\.app-shell-game-in-progress \.seat-left,\n  \.app-shell-game-in-progress \.seat-right \{([\s\S]*?)\n  \}/
    );
    const tableCenterMatch = styles.match(
      /\.app-shell-game-in-progress \.table-center \{([\s\S]*?)\n  \}/
    );

    expect(sideSeatMatch?.[1]).toContain("align-self: start;");
    expect(sideSeatMatch?.[1]).toContain("margin-top: 96px;");
    expect(tableCenterMatch?.[1]).toContain("align-content: start;");
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

function createState(overrides: Partial<PublicGameState> = {}): PublicGameState {
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
    legalActions: [],
    ...overrides
  };
}
