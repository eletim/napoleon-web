// @vitest-environment happy-dom

import { readFileSync } from "node:fs";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { PublicCard, PublicGameState, PublicRank, PublicSuit } from "@napoleon/protocol";
import { SelfHandPanel } from "./SelfHandPanel";
import type { TablePlayer } from "./tableTypes";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe("SelfHandPanel", () => {
  it("renders the self hand in riipai order without changing card id based state", () => {
    const hand = [
      standardCard("clubs", "2"),
      jokerCard,
      standardCard("spades", "2"),
      standardCard("hearts", "K"),
      standardCard("spades", "A")
    ];
    const state = createState(hand);

    const html = renderToStaticMarkup(
      <SelfHandPanel
        canExchange={false}
        defaultHandOrderMode="riipai"
        isBusy={false}
        legalCardIds={new Set(["spades-2"])}
        onToggleWinningCardHighlight={vi.fn()}
        onPlay={vi.fn()}
        selectedDiscardCardIds={["spades-2"]}
        self={state.self}
        selfPlayer={undefined}
        state={state}
        winningCardHighlightEnabled={true}
      />
    );

    expect(indexOfCard(html, "A♠")).toBeLessThan(indexOfCard(html, "2♠"));
    expect(indexOfCard(html, "2♠")).toBeLessThan(indexOfCard(html, "K♥"));
    expect(indexOfCard(html, "K♥")).toBeLessThan(indexOfCard(html, "2♣"));
    expect(indexOfCard(html, "2♣")).toBeLessThan(indexOfCard(html, "JOKER"));

    expect(buttonMarkup(html, "2♠")).toContain("card-legal");
    expect(buttonMarkup(html, "2♠")).toContain("card-selected");
    expect(buttonMarkup(html, "2♠")).toContain("交換対象に選択済み");
    expect(buttonMarkup(html, "K♥")).toContain("card-blocked");
    expect(buttonMarkup(html, "K♥")).toContain("現在は操作できないカード");
  });

  it("uses riipai order as the default view", () => {
    const hand = [
      standardCard("clubs", "2"),
      standardCard("spades", "A"),
      standardCard("hearts", "K")
    ];
    const state = createState(hand);

    const html = renderToStaticMarkup(
      <SelfHandPanel
        canExchange={false}
        isBusy={false}
        legalCardIds={new Set()}
        onToggleWinningCardHighlight={vi.fn()}
        onPlay={vi.fn()}
        selectedDiscardCardIds={[]}
        self={state.self}
        selfPlayer={undefined}
        state={state}
        winningCardHighlightEnabled={true}
      />
    );

    expect(html).toContain("aria-pressed=\"true\"");
    expect(indexOfCard(html, "A♠")).toBeLessThan(indexOfCard(html, "K♥"));
    expect(indexOfCard(html, "K♥")).toBeLessThan(indexOfCard(html, "2♣"));
    expect(html).toContain("aria-label=\"理牌オン\"");
    expect(html).not.toContain("aria-label=\"配札順\"");
  });

  it("keeps the self hand compact without visible ids or legend text", () => {
    const hand = [standardCard("clubs", "2"), standardCard("spades", "A")];
    const state = createState(hand);

    const html = renderToStaticMarkup(
      <SelfHandPanel
        canExchange={false}
        isBusy={false}
        legalCardIds={new Set(["spades-A"])}
        onToggleWinningCardHighlight={vi.fn()}
        onPlay={vi.fn()}
        selectedDiscardCardIds={[]}
        self={state.self}
        selfPlayer={undefined}
        state={state}
        winningCardHighlightEnabled={true}
      />
    );

    expect(html).not.toContain("player-0");
    expect(html).not.toContain("合法");
    expect(html).not.toContain("不可");
    expect(html).not.toContain("選択");
    expect(html).not.toContain("残り2枚");
    expect(html).toContain("aria-label=\"自分の手札\"");
    expect(html).toContain("aria-label=\"自分の獲得得点札は0枚\"");
  });

  it("switches self seat details between bidding and playing phases", () => {
    const hand = [standardCard("clubs", "2")];
    const biddingState = {
      ...createState(hand),
      phase: "bidding" as const,
      contract: {
        napoleonPlayerId: "player-0",
        trumpSuit: "spades" as const,
        targetPointCards: 13
      }
    };
    const selfPlayer = createSelfPlayer({
      biddingDeclaration: {
        type: "bid",
        label: "♣ 13",
        suit: "clubs",
        targetPointCards: 13,
        color: "black"
      }
    });

    const biddingHtml = renderToStaticMarkup(
      <SelfHandPanel
        canExchange={false}
        isBusy={false}
        legalCardIds={new Set()}
        onToggleWinningCardHighlight={vi.fn()}
        onPlay={vi.fn()}
        selectedDiscardCardIds={[]}
        self={biddingState.self}
        selfPlayer={selfPlayer}
        state={biddingState}
        winningCardHighlightEnabled={true}
      />
    );
    const playingHtml = renderToStaticMarkup(
      <SelfHandPanel
        canExchange={false}
        isBusy={false}
        legalCardIds={new Set(["clubs-2"])}
        onToggleWinningCardHighlight={vi.fn()}
        onPlay={vi.fn()}
        selectedDiscardCardIds={[]}
        self={createState(hand).self}
        selfPlayer={selfPlayer}
        state={createState(hand)}
        winningCardHighlightEnabled={true}
      />
    );

    expect(biddingHtml).toContain("latest-bid-declaration");
    expect(biddingHtml).toContain("13");
    expect(biddingHtml).toContain("♣");
    expect(biddingHtml).not.toContain("aria-label=\"ナポレオン\"");
    expect(biddingHtml).not.toContain("aria-label=\"自分の獲得得点札は");

    expect(playingHtml).not.toContain("latest-bid-declaration");
    expect(playingHtml).toContain("aria-label=\"ナポレオン\"");
    expect(playingHtml).toContain("aria-label=\"自分の獲得得点札は0枚\"");
  });

  it("reserves ten hand slots without changing existing card controls", () => {
    const hand = [
      standardCard("spades", "A"),
      standardCard("spades", "K"),
      standardCard("hearts", "Q"),
      standardCard("diamonds", "J"),
      standardCard("clubs", "10"),
      standardCard("clubs", "9")
    ];
    const state = createState(hand);

    const html = renderToStaticMarkup(
      <SelfHandPanel
        canExchange={false}
        isBusy={false}
        legalCardIds={new Set(["spades-A"])}
        onToggleWinningCardHighlight={vi.fn()}
        onPlay={vi.fn()}
        selectedDiscardCardIds={[]}
        self={state.self}
        selfPlayer={undefined}
        state={state}
        winningCardHighlightEnabled={true}
      />
    );

    expect(countOccurrences(html, "class=\"hand-card-empty-slot\"")).toBe(4);
    expect(countOccurrences(html, "class=\"card ")).toBe(6);
    expect(buttonMarkup(html, "A♠")).toContain("card-legal");
  });

  it("does not reserve empty hand slots before a game or after it ends", () => {
    const unstartedHtml = renderToStaticMarkup(
      <SelfHandPanel
        canExchange={false}
        isBusy={false}
        legalCardIds={new Set()}
        onToggleWinningCardHighlight={vi.fn()}
        onPlay={vi.fn()}
        selectedDiscardCardIds={[]}
        self={undefined}
        selfPlayer={undefined}
        state={undefined}
        winningCardHighlightEnabled={true}
      />
    );
    const finishedState = { ...createState([]), isGameOver: true };
    const finishedHtml = renderToStaticMarkup(
      <SelfHandPanel
        canExchange={false}
        isBusy={false}
        legalCardIds={new Set()}
        onToggleWinningCardHighlight={vi.fn()}
        onPlay={vi.fn()}
        selectedDiscardCardIds={[]}
        self={finishedState.self}
        selfPlayer={undefined}
        state={finishedState}
        winningCardHighlightEnabled={true}
      />
    );

    expect(unstartedHtml).not.toContain("class=\"hand-card-empty-slot\"");
    expect(finishedHtml).not.toContain("class=\"hand-card-empty-slot\"");
  });

  it("defines a five by two mobile hand grid", () => {
    const styles = readFileSync("src/styles.css", "utf8");
    const landscapeBlock = getLastMediaBlock(
      styles,
      "@media (max-width: 960px) and (max-height: 560px) and (orientation: landscape)"
    );

    expect(landscapeBlock).toContain("grid-template-columns: repeat(5, minmax(0, 56px));");
    expect(landscapeBlock).toContain("grid-template-rows: repeat(2, 38px);");
  });

  it("keeps the in-progress self panel compact without changing the fixed hand grid", () => {
    const styles = readFileSync("src/styles.css", "utf8");

    expect(styles).toContain("background: rgb(248 250 252 / 92%);");
    expect(styles).toContain("border: 1px solid rgb(255 255 255 / 70%);");
    expect(styles).toContain("box-shadow: 0 6px 16px rgb(15 23 42 / 12%);");
    expect(styles).toContain("border-color: rgb(250 204 21 / 70%);");
    expect(styles).toContain("border-top-color: rgb(148 163 184 / 28%);");
    expect(styles).toContain("grid-template-columns: repeat(10, minmax(0, 68px));");
  });

  it("keeps the landscape mobile hand as a compact five by two grid", () => {
    const styles = readFileSync("src/styles.css", "utf8");
    const landscapeBlock = getMediaBlock(
      styles,
      "@media (max-width: 960px) and (max-height: 560px) and (orientation: landscape)"
    );

    expect(landscapeBlock).toContain("display: grid;");
    expect(landscapeBlock).toContain("grid-template-columns: repeat(5, minmax(0, 1fr));");
    expect(landscapeBlock).toContain("grid-template-rows: repeat(2, 34px);");
    expect(landscapeBlock).toContain("height: 34px;");
  });

  it("toggles riipai from one sort control without sending a card action", () => {
    const hand = [
      standardCard("clubs", "2"),
      jokerCard,
      standardCard("spades", "2"),
      standardCard("hearts", "K"),
      standardCard("spades", "A")
    ];
    const state = createState(hand);
    const onPlay = vi.fn();
    const onToggleWinningCardHighlight = vi.fn();
    const container = document.createElement("div");
    const root = createRoot(container);

    act(() => {
      root.render(
        <SelfHandPanel
          canExchange={false}
          isBusy={false}
          legalCardIds={new Set(["spades-2"])}
          onToggleWinningCardHighlight={onToggleWinningCardHighlight}
          onPlay={onPlay}
          selectedDiscardCardIds={[]}
          self={state.self}
          selfPlayer={undefined}
          state={state}
          winningCardHighlightEnabled={true}
        />
      );
    });

    expect(cardLabels(container)).toEqual(["A♠", "2♠", "K♥", "2♣", "JOKER"]);

    expect(getHandSortButton(container).getAttribute("aria-pressed")).toBe("true");

    act(() => {
      getHandSortButton(container).click();
    });

    expect(onPlay).not.toHaveBeenCalled();
    expect(cardLabels(container)).toEqual(["2♣", "JOKER", "2♠", "K♥", "A♠"]);
    expect(getHandSortButton(container).getAttribute("aria-pressed")).toBe("false");
    expect(getHandSortButton(container).getAttribute("aria-label")).toBe("理牌オフ");

    act(() => {
      getHandSortButton(container).click();
    });

    expect(cardLabels(container)).toEqual(["A♠", "2♠", "K♥", "2♣", "JOKER"]);
    expect(getHandSortButton(container).getAttribute("aria-pressed")).toBe("true");
    expect(getHandSortButton(container).getAttribute("aria-label")).toBe("理牌オン");

    act(() => {
      getCardButton(container, "2♠").click();
    });

    expect(onPlay).toHaveBeenCalledTimes(1);
    expect(onPlay).toHaveBeenCalledWith(hand[2]);

    act(() => {
      getWinningCardToggleButton(container).click();
    });

    expect(onToggleWinningCardHighlight).toHaveBeenCalledTimes(1);
    expect(onPlay).toHaveBeenCalledTimes(1);

    act(() => {
      root.unmount();
    });
  });

  it("shows the compact winning-card highlight toggle state", () => {
    const state = createState([standardCard("clubs", "2")]);

    const html = renderToStaticMarkup(
      <SelfHandPanel
        canExchange={false}
        isBusy={false}
        legalCardIds={new Set()}
        onToggleWinningCardHighlight={vi.fn()}
        onPlay={vi.fn()}
        selectedDiscardCardIds={[]}
        self={state.self}
        selfPlayer={undefined}
        state={state}
        winningCardHighlightEnabled={false}
      />
    );

    expect(html).toContain("aria-label=\"暫定勝ち札強調オフ\"");
    expect(html).toContain(">勝</button>");
  });
});

const jokerCard: PublicCard = { type: "joker", id: "joker" };

function standardCard(suit: PublicSuit, rank: PublicRank): PublicCard {
  return {
    type: "standard",
    id: `${suit}-${rank}`,
    suit,
    rank
  };
}

function getMediaBlock(styles: string, query: string): string {
  const start = styles.indexOf(query);

  if (start === -1) {
    throw new Error(`Media query not found: ${query}`);
  }

  const nextMedia = styles.indexOf("@media", start + query.length);

  return styles.slice(start, nextMedia === -1 ? undefined : nextMedia);
}

function createState(hand: readonly PublicCard[]): PublicGameState {
  return {
    self: {
      id: "player-0",
      handCount: hand.length,
      hand,
      capturedPointCards: []
    },
    opponents: [],
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
    legalActions: [{ type: "play-card", cardId: "spades-2" }]
  };
}

function createSelfPlayer(overrides: Partial<TablePlayer> = {}): TablePlayer {
  return {
    id: "player-0",
    label: "自分",
    seat: "self",
    handCount: 1,
    capturedPointCards: [],
    isSelf: true,
    biddingDeclaration: undefined,
    ...overrides
  };
}

function indexOfCard(html: string, ariaLabel: string): number {
  const index = html.indexOf(`aria-label="${ariaLabel}"`);

  if (index === -1) {
    throw new Error(`Card button ${ariaLabel} was not rendered.`);
  }

  return index;
}

function buttonMarkup(html: string, ariaLabel: string): string {
  const start = indexOfCard(html, ariaLabel);
  const end = html.indexOf("</button>", start);

  if (end === -1) {
    throw new Error(`Card button ${ariaLabel} closing tag was not rendered.`);
  }

  return html.slice(start, end);
}

function cardLabels(container: Element): string[] {
  return Array.from(container.querySelectorAll(".hand .card")).map((button) => {
    const ariaLabel = button.getAttribute("aria-label");

    if (ariaLabel === null) {
      throw new Error("Card button did not have an aria-label.");
    }

    return ariaLabel;
  });
}

function getHandSortButton(container: Element): HTMLButtonElement {
  const button = container.querySelector(".hand-sort-toggle button");

  if (!(button instanceof HTMLButtonElement)) {
    throw new Error("Hand sort button was not rendered.");
  }

  return button;
}

function getWinningCardToggleButton(container: Element): HTMLButtonElement {
  const button = container.querySelector(".winning-card-toggle button");

  if (!(button instanceof HTMLButtonElement)) {
    throw new Error("Winning card toggle button was not rendered.");
  }

  return button;
}

function getCardButton(container: Element, ariaLabel: string): HTMLButtonElement {
  const button = container.querySelector(`.hand .card[aria-label="${ariaLabel}"]`);

  if (!(button instanceof HTMLButtonElement)) {
    throw new Error(`Card button ${ariaLabel} was not rendered.`);
  }

  return button;
}

function countOccurrences(value: string, pattern: string): number {
  return value.split(pattern).length - 1;
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
