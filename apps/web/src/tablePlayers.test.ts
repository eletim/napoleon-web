import { describe, expect, it } from "vitest";
import type { PublicGameState } from "@napoleon/protocol";
import { createTablePlayers } from "./tablePlayers";

describe("createTablePlayers", () => {
  it("assigns opponents and self to the fixed five-player seats", () => {
    const players = createTablePlayers(createPublicState());

    expect(players.map((player) => [player.id, player.label, player.seat])).toEqual([
      ["player-1", "左側AI", "left"],
      ["player-2", "奥左AI", "top-left"],
      ["player-3", "奥右AI", "top-right"],
      ["player-4", "右側AI", "right"],
      ["player-0", "自分", "self"]
    ]);
    expect(players.filter((player) => player.isSelf)).toHaveLength(1);
  });

  it("adds latest bidding declarations only during bidding", () => {
    const biddingPlayers = createTablePlayers(
      createPublicState({
        phase: "bidding",
        trumpSuit: null,
        contract: null,
        bidding: {
          starterPlayerId: "player-0",
          highestBid: {
            playerId: "player-0",
            suit: "clubs",
            targetPointCards: 13
          },
          consecutivePassCount: 1,
          history: [
            { type: "bid", playerId: "player-0", suit: "clubs", targetPointCards: 13 },
            { type: "pass", playerId: "player-1" },
            { type: "bid", playerId: "player-0", suit: "hearts", targetPointCards: 14 }
          ]
        }
      })
    );

    expect(biddingPlayers.map((player) => [player.id, player.biddingDeclaration])).toEqual([
      ["player-1", { type: "pass", label: "パス" }],
      ["player-2", { type: "none", label: "未宣言" }],
      ["player-3", { type: "none", label: "未宣言" }],
      ["player-4", { type: "none", label: "未宣言" }],
      [
        "player-0",
        {
          type: "bid",
          label: "♥ 14",
          suit: "hearts",
          targetPointCards: 14,
          color: "red"
        }
      ]
    ]);

    expect(createTablePlayers(createPublicState())[0].biddingDeclaration).toBeUndefined();
  });
});

function createPublicState(overrides: Partial<PublicGameState> = {}): PublicGameState {
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
