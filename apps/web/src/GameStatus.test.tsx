import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { PublicGameState, PublicSuit } from "@napoleon/protocol";
import { createGameStatusDisplay } from "./displayText";
import { GameStatus } from "./GameStatus";
import { createTablePlayers } from "./tablePlayers";

describe("GameStatus", () => {
  it.each([
    ["hearts", "status-chip-suit-red"],
    ["diamonds", "status-chip-suit-red"],
    ["spades", "status-chip-suit-black"],
    ["clubs", "status-chip-suit-black"]
  ] satisfies readonly [PublicSuit, string][])(
    "uses the suit background class for the confirmed %s contract",
    (suit, expectedClassName) => {
      const state = createState(suit);
      const html = renderToStaticMarkup(
        <GameStatus display={createGameStatusDisplay(state, createTablePlayers(state))} />
      );

      expect(html).toContain(`status-chip status-chip-contract ${expectedClassName}`);
    }
  );
});

function createState(trumpSuit: PublicSuit): PublicGameState {
  return {
    self: {
      id: "player-0",
      handCount: 10,
      hand: [],
      capturedPointCards: []
    },
    opponents: [
      { id: "player-1", handCount: 10, capturedPointCards: [] },
      { id: "player-2", handCount: 10, capturedPointCards: [] },
      { id: "player-3", handCount: 10, capturedPointCards: [] },
      { id: "player-4", handCount: 10, capturedPointCards: [] }
    ],
    phase: "playing",
    trumpSuit,
    contract: {
      napoleonPlayerId: "player-0",
      trumpSuit,
      targetPointCards: 15
    },
    specialCards: {
      orumaCardId: "spades-A",
      yoromekiCardId: "hearts-Q",
      seiJackCardId: "spades-J",
      uraJackCardId: "clubs-J"
    },
    adjutant: {
      calledCardId: "spades-A",
      revealedPlayerId: null
    },
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
