// @vitest-environment happy-dom

import { act } from "react";
import { createRoot } from "react-dom/client";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { PublicCard, PublicGameState, PublicRank, PublicSuit } from "@napoleon/protocol";
import { SelfHandPanel } from "./SelfHandPanel";

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
        onPlay={vi.fn()}
        selectedDiscardCardIds={["spades-2"]}
        self={state.self}
        selfPlayer={undefined}
        state={state}
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
        onPlay={vi.fn()}
        selectedDiscardCardIds={[]}
        self={state.self}
        selfPlayer={undefined}
        state={state}
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
        onPlay={vi.fn()}
        selectedDiscardCardIds={[]}
        self={state.self}
        selfPlayer={undefined}
        state={state}
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
    const container = document.createElement("div");
    const root = createRoot(container);

    act(() => {
      root.render(
        <SelfHandPanel
          canExchange={false}
          isBusy={false}
          legalCardIds={new Set(["spades-2"])}
          onPlay={onPlay}
          selectedDiscardCardIds={[]}
          self={state.self}
          selfPlayer={undefined}
          state={state}
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
      root.unmount();
    });
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

function getCardButton(container: Element, ariaLabel: string): HTMLButtonElement {
  const button = container.querySelector(`.hand .card[aria-label="${ariaLabel}"]`);

  if (!(button instanceof HTMLButtonElement)) {
    throw new Error(`Card button ${ariaLabel} was not rendered.`);
  }

  return button;
}
