import { describe, expect, it } from "vitest";
import type { PublicBiddingState } from "@napoleon/protocol";
import { createLatestBiddingDeclarations } from "./biddingDeclarations";

describe("createLatestBiddingDeclarations", () => {
  it("keeps each player's latest bidding action", () => {
    const declarations = createLatestBiddingDeclarations({
      starterPlayerId: "player-0",
      highestBid: {
        playerId: "player-2",
        suit: "hearts",
        targetPointCards: 14
      },
      consecutivePassCount: 1,
      history: [
        { type: "bid", playerId: "player-0", suit: "clubs", targetPointCards: 13 },
        { type: "pass", playerId: "player-1" },
        { type: "bid", playerId: "player-2", suit: "hearts", targetPointCards: 14 },
        { type: "bid", playerId: "player-0", suit: "spades", targetPointCards: 15 },
        { type: "pass", playerId: "player-2" }
      ]
    } satisfies PublicBiddingState);

    expect(declarations.get("player-0")).toEqual({
      type: "bid",
      label: "♠ 15",
      suit: "spades",
      targetPointCards: 15,
      color: "black"
    });
    expect(declarations.get("player-1")).toEqual({
      type: "pass",
      label: "パス"
    });
    expect(declarations.get("player-2")).toEqual({
      type: "pass",
      label: "パス"
    });
    expect(declarations.get("player-3")).toBeUndefined();
  });

  it("formats red suit bids with existing suit symbols", () => {
    const declarations = createLatestBiddingDeclarations({
      starterPlayerId: "player-0",
      highestBid: {
        playerId: "player-0",
        suit: "diamonds",
        targetPointCards: 16
      },
      consecutivePassCount: 0,
      history: [
        { type: "bid", playerId: "player-0", suit: "diamonds", targetPointCards: 16 }
      ]
    } satisfies PublicBiddingState);

    expect(declarations.get("player-0")).toEqual({
      type: "bid",
      label: "♦ 16",
      suit: "diamonds",
      targetPointCards: 16,
      color: "red"
    });
  });
});
