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

  it("uses the shared in-progress table layout for opponent seats", () => {
    const styles = readFileSync(fileURLToPath(new URL("./styles.css", import.meta.url)), "utf8");

    expect(styles).toContain(
      'grid-template-areas:\n    ". top-left top-right ."\n    "left center center right"\n    "action action action action"\n    "self self self self";'
    );
    expect(styles).toContain(
      "grid-template-columns:\n    minmax(116px, 0.62fr)\n    minmax(230px, 1.2fr)\n    minmax(230px, 1.2fr)\n    minmax(116px, 0.62fr);"
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

  it("shows only the player name and bidding declaration during bidding", () => {
    const html = renderToStaticMarkup(
      <PlayerSeat
        player={createPlayer([pointCard("spades", "A")], {
          biddingDeclaration: {
            type: "bid",
            label: "♥ 13",
            suit: "hearts",
            targetPointCards: 13,
            color: "red"
          }
        })}
        state={createState({
          phase: "bidding",
          contract: {
            napoleonPlayerId: "player-1",
            trumpSuit: "spades",
            targetPointCards: 13
          },
          adjutant: { calledCardId: "spades-A", revealedPlayerId: "player-1" }
        })}
      />
    );

    expect(html).toContain("<h2>左側AI</h2>");
    expect(html).toContain("13");
    expect(html).toContain("♥");
    expect(html).toContain("latest-bid-declaration");
    expect(html).not.toContain("aria-label=\"ナポレオン\"");
    expect(html).not.toContain("aria-label=\"副官\"");
    expect(html).not.toContain("獲得得点札");
    expect(html).not.toContain("A♠");
  });

  it("uses compact pass and pending labels for bidding seats", () => {
    const passHtml = renderToStaticMarkup(
      <PlayerSeat
        player={createPlayer([], { biddingDeclaration: { type: "pass", label: "パス" } })}
        state={createState({ phase: "bidding" })}
      />
    );
    const pendingHtml = renderToStaticMarkup(
      <PlayerSeat
        player={createPlayer([], { biddingDeclaration: { type: "none", label: "未宣言" } })}
        state={createState({ phase: "bidding" })}
      />
    );

    expect(passHtml).toContain(">Pass</strong>");
    expect(pendingHtml).toContain(">—</strong>");
    expect(pendingHtml).not.toContain(">未宣言</strong>");
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

  it("keeps in-progress opponent seats visible as table seats", () => {
    const styles = readFileSync(fileURLToPath(new URL("./styles.css", import.meta.url)), "utf8");

    expect(styles).toContain("background: rgb(248 250 252 / 92%);");
    expect(styles).toContain("border: 1px solid rgb(255 255 255 / 70%);");
    expect(styles).toContain("box-shadow: 0 6px 16px rgb(15 23 42 / 12%);");
    expect(styles).toContain("min-height: 66px;");
    expect(styles).toContain("width: auto;");
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

  it("keeps in-progress opponent seats close to the common center board", () => {
    const styles = readFileSync(fileURLToPath(new URL("./styles.css", import.meta.url)), "utf8");

    expect(styles).toContain("grid-template-rows: auto minmax(0, 1fr) auto auto;");
    expect(styles).toContain("justify-self: stretch;");
    expect(styles).toContain(
      ".app-shell-game-in-progress .seat-top-left,\n.app-shell-game-in-progress .seat-top-right,\n.app-shell-game-in-progress .seat-left,\n.app-shell-game-in-progress .seat-right"
    );
    expect(styles).toContain("margin-top: 0;");
    expect(styles).toContain("align-content: end;");
    expect(styles).toContain("align-content: start;");
    expect(styles).toContain("overflow: hidden;");
  });

  it("defines a mobile landscape density pass without replacing the shared table structure", () => {
    const styles = readFileSync(fileURLToPath(new URL("./styles.css", import.meta.url)), "utf8");
    const appSource = readFileSync(fileURLToPath(new URL("./App.tsx", import.meta.url)), "utf8");
    const portraitBlock = getMediaBlock(
      styles,
      "@media (max-width: 560px) and (orientation: portrait)"
    );
    const landscapeBlock = getLastMediaBlock(
      styles,
      "@media (max-width: 960px) and (max-height: 560px) and (orientation: landscape)"
    );

    expect(appSource).toContain("mobile-landscape-guide");
    expect(appSource).toContain("横向きでプレイしてください");
    expect(portraitBlock).toContain(".app-shell-game-in-progress .mobile-landscape-guide");
    expect(portraitBlock).toContain("display: grid;");
    expect(portraitBlock).toContain(".app-shell-game-in-progress .table");
    expect(portraitBlock).toContain("display: none;");
    expect(styles).toContain(
      'grid-template-areas:\n    ". top-left top-right ."\n    "left center center right"\n    "action action action action"\n    "self self self self";'
    );
    expect(landscapeBlock).not.toContain('"top-left center center top-right"');
    expect(landscapeBlock).toContain("minmax(86px, 0.74fr)");
    expect(landscapeBlock).not.toContain('"top-left message top-right"');
    expect(landscapeBlock).not.toContain('"left self right";');
    expect(landscapeBlock).toContain(
      ".app-shell-phase-bidding.app-shell-game-active .bidding-panel"
    );
    expect(landscapeBlock).toContain('"header highest highest"');
    expect(landscapeBlock).toContain('"suits stepper submit";');
    expect(landscapeBlock).toContain("grid-template-columns: minmax(86px, 0.7fr)");
    expect(landscapeBlock).toContain("width: min(100%, 420px);");
    expect(landscapeBlock).toContain("grid-template-columns: repeat(5, minmax(0, 56px));");
    expect(landscapeBlock).toContain("grid-template-rows: repeat(2, 38px);");
  });
});

function getMediaBlock(styles: string, query: string): string {
  const start = styles.indexOf(query);

  if (start === -1) {
    throw new Error(`Media query not found: ${query}`);
  }

  const nextMedia = styles.indexOf("@media", start + query.length);

  return styles.slice(start, nextMedia === -1 ? undefined : nextMedia);
}

function getLastMediaBlock(styles: string, query: string): string {
  const firstBlock = getMediaBlock(styles, query);
  let block = firstBlock;
  let searchStart = styles.indexOf(query) + query.length;

  while (true) {
    const nextStart = styles.indexOf(query, searchStart);

    if (nextStart === -1) {
      return block;
    }

    const nextMedia = styles.indexOf("@media", nextStart + query.length);
    block = styles.slice(nextStart, nextMedia === -1 ? undefined : nextMedia);
    searchStart = nextStart + query.length;
  }
}

function createPlayer(
  capturedPointCards: readonly PublicStandardCard[] = [],
  overrides: Partial<Pick<TablePlayer, "biddingDeclaration">> = {}
): TablePlayer {
  return {
    id: "player-1",
    label: "左側AI",
    seat: "left",
    handCount: 10,
    capturedPointCards,
    isSelf: false,
    biddingDeclaration: undefined,
    ...overrides
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
