import { describe, expect, it } from "vitest";
import type { PublicBidAction } from "@napoleon/protocol";
import {
  findBidAction,
  getBidStepperState,
  getBidTargetsForSuit,
  normalizeBidSelection
} from "./biddingOptions";

describe("biddingOptions", () => {
  const legalBidActions: readonly PublicBidAction[] = [
    { type: "bid", suit: "hearts", targetPointCards: 13 },
    { type: "bid", suit: "spades", targetPointCards: 13 },
    { type: "bid", suit: "hearts", targetPointCards: 15 },
    { type: "bid", suit: "clubs", targetPointCards: 16 }
  ];

  it("derives target point candidates from legal bid actions only", () => {
    expect(getBidTargetsForSuit(legalBidActions, "hearts")).toEqual([13, 15]);
    expect(getBidTargetsForSuit(legalBidActions, "diamonds")).toEqual([]);
  });

  it("normalizes an empty selection to the first legal bid action", () => {
    expect(normalizeBidSelection(legalBidActions, null)).toEqual({
      suit: "hearts",
      targetPointCards: 13
    });
  });

  it("keeps a selected bid only while it remains legal", () => {
    expect(
      normalizeBidSelection(legalBidActions, {
        suit: "spades",
        targetPointCards: 13
      })
    ).toEqual({
      suit: "spades",
      targetPointCards: 13
    });
    expect(
      normalizeBidSelection(legalBidActions, {
        suit: "spades",
        targetPointCards: 14
      })
    ).toEqual({
      suit: "spades",
      targetPointCards: 13
    });
  });

  it("moves the stepper through legal targets without numeric guessing", () => {
    expect(
      getBidStepperState(legalBidActions, {
        suit: "hearts",
        targetPointCards: 13
      })
    ).toMatchObject({
      previousTarget: null,
      nextTarget: 15
    });
    expect(
      getBidStepperState(legalBidActions, {
        suit: "hearts",
        targetPointCards: 15
      })
    ).toMatchObject({
      previousTarget: 13,
      nextTarget: null
    });
  });

  it("does not find bid actions outside legalActions", () => {
    expect(
      findBidAction(legalBidActions, {
        suit: "diamonds",
        targetPointCards: 13
      })
    ).toBeUndefined();
  });
});
