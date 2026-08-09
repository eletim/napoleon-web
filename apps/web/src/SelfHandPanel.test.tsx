import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { PublicCard, PublicGameState, PublicRank, PublicSuit } from "@napoleon/protocol";
import { SelfHandPanel } from "./SelfHandPanel";

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
    expect(buttonMarkup(html, "K♥")).toContain("card-blocked");
  });

  it("keeps original order as the default view", () => {
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
    expect(indexOfCard(html, "2♣")).toBeLessThan(indexOfCard(html, "A♠"));
    expect(indexOfCard(html, "A♠")).toBeLessThan(indexOfCard(html, "K♥"));
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
