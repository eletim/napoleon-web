import { describe, expect, it } from "vitest";
import {
  BIDDING_ACTION_COUNT,
  BIDDING_HISTORY_SUIT_ORDER,
  MAX_BIDDING_TARGET_POINT_CARDS,
  MIN_BIDDING_TARGET_POINT_CARDS,
  decodeBiddingAction,
  encodeBiddingAction,
  encodeLegalBidMask
} from "../src/index.js";

describe("bidding action encoding", () => {
  it("round-trips the fixed 41 bidding action space", () => {
    expect(BIDDING_ACTION_COUNT).toBe(41);
    expect(decodeBiddingAction(0, "player-0")).toEqual({
      type: "pass",
      playerId: "player-0"
    });
    expect(encodeBiddingAction({ type: "pass", playerId: "player-0" })).toBe(0);

    for (
      let targetPointCards = MIN_BIDDING_TARGET_POINT_CARDS;
      targetPointCards <= MAX_BIDDING_TARGET_POINT_CARDS;
      targetPointCards += 1
    ) {
      for (const [suitIndex, suit] of BIDDING_HISTORY_SUIT_ORDER.entries()) {
        const actionIndex =
          1 +
          (targetPointCards - MIN_BIDDING_TARGET_POINT_CARDS) *
            BIDDING_HISTORY_SUIT_ORDER.length +
          suitIndex;
        const action = {
          type: "bid" as const,
          playerId: "player-0",
          suit,
          targetPointCards
        };

        expect(encodeBiddingAction(action)).toBe(actionIndex);
        expect(decodeBiddingAction(actionIndex, "player-0")).toEqual(action);
      }
    }
  });

  it("uses the existing bidding history suit order for bid indices", () => {
    expect(BIDDING_HISTORY_SUIT_ORDER).toEqual(["spades", "hearts", "diamonds", "clubs"]);
    expect(encodeBiddingAction({
      type: "bid",
      playerId: "player-0",
      suit: "spades",
      targetPointCards: 13
    })).toBe(13);
    expect(encodeBiddingAction({
      type: "bid",
      playerId: "player-0",
      suit: "clubs",
      targetPointCards: 13
    })).toBe(16);
    expect(encodeBiddingAction({
      type: "bid",
      playerId: "player-0",
      suit: "spades",
      targetPointCards: 19
    })).toBe(37);
    expect(encodeBiddingAction({
      type: "bid",
      playerId: "player-0",
      suit: "clubs",
      targetPointCards: 19
    })).toBe(40);
  });

  it("builds legalBidMask only from supplied legalActions", () => {
    const mask = encodeLegalBidMask([
      { type: "pass", playerId: "player-0" },
      { type: "bid", playerId: "player-0", suit: "spades", targetPointCards: 10 },
      { type: "bid", playerId: "player-0", suit: "hearts", targetPointCards: 10 },
      { type: "bid", playerId: "player-0", suit: "clubs", targetPointCards: 11 }
    ], "player-0");

    expect(mask).toHaveLength(BIDDING_ACTION_COUNT);
    expect(mask[0]).toBe(1);
    expect(mask[1]).toBe(1);
    expect(mask[2]).toBe(1);
    expect(mask[3]).toBe(0);
    expect(mask[4]).toBe(0);
    expect(mask[8]).toBe(1);
    expect(sum(mask)).toBe(4);
  });

  it("validates actorTarget against the legal mask", () => {
    const passOnlyMask = [1, ...Array(BIDDING_ACTION_COUNT - 1).fill(0)];

    expect(encodeBiddingAction(
      { type: "pass", playerId: "player-0" },
      passOnlyMask,
      "player-0"
    )).toBe(0);
    expect(() => encodeBiddingAction(
      { type: "bid", playerId: "player-0", suit: "spades", targetPointCards: 13 },
      passOnlyMask,
      "player-0"
    )).toThrow("not legal");
    expect(() => encodeBiddingAction(
      { type: "bid", playerId: "player-1", suit: "spades", targetPointCards: 13 },
      passOnlyMask,
      "player-0"
    )).toThrow("Decision action playerId");
  });

  it("rejects invalid action indices and bids outside 10 to 19", () => {
    expect(() => decodeBiddingAction(-1, "player-0")).toThrow("between 0 and 40");
    expect(() => decodeBiddingAction(41, "player-0")).toThrow("between 0 and 40");
    expect(() => encodeBiddingAction({
      type: "bid",
      playerId: "player-0",
      suit: "spades",
      targetPointCards: 9
    })).toThrow("between 10 and 19");
  });
});

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}
