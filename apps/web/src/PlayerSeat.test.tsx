import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { PublicGameState } from "@napoleon/protocol";
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
});

function createPlayer(): TablePlayer {
  return {
    id: "player-1",
    label: "左側AI",
    seat: "left",
    handCount: 10,
    capturedPointCards: [],
    isSelf: false,
    biddingDeclaration: undefined
  };
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
