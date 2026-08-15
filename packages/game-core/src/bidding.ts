import { suits } from "./deck.js";
import { GameRuleError } from "./errors.js";
import type { Bid, GameAction, PlayerId, Suit } from "./types.js";

export const minBidTargetPointCards = 10;
export const maxBidTargetPointCards = 19;
export const forcedAllPassTargetPointCards = 9;

export const biddingSuitPriority: Record<Suit, number> = {
  clubs: 0,
  diamonds: 1,
  hearts: 2,
  spades: 3
};

export const biddingSuitOrder: readonly Suit[] = ["clubs", "diamonds", "hearts", "spades"];

export function compareBids(left: Bid, right: Bid): number {
  if (left.targetPointCards !== right.targetPointCards) {
    return left.targetPointCards - right.targetPointCards;
  }

  return biddingSuitPriority[left.suit] - biddingSuitPriority[right.suit];
}

export function isBidHigher(candidate: Bid, current: Bid): boolean {
  return compareBids(candidate, current) > 0;
}

export function isSuit(value: unknown): value is Suit {
  return typeof value === "string" && (suits as readonly string[]).includes(value);
}

export function validateBidRange(targetPointCards: number): void {
  if (!Number.isInteger(targetPointCards)) {
    throw new GameRuleError("INVALID_BID", "Bid target must be an integer.");
  }

  if (
    targetPointCards < minBidTargetPointCards ||
    targetPointCards > maxBidTargetPointCards
  ) {
    throw new GameRuleError(
      "INVALID_BID",
      `Bid target must be between ${minBidTargetPointCards} and ${maxBidTargetPointCards}.`
    );
  }
}

export function getLegalBidActions(
  playerId: PlayerId,
  highestBid: Bid | null
): readonly GameAction[] {
  const bidActions = Array.from(
    { length: maxBidTargetPointCards - minBidTargetPointCards + 1 },
    (_, index) => index + minBidTargetPointCards
  ).flatMap((targetPointCards) =>
    biddingSuitOrder.map((suit) => ({
      type: "bid" as const,
      playerId,
      suit,
      targetPointCards
    }))
  );

  return [
    {
      type: "pass",
      playerId
    },
    ...bidActions.filter((action) =>
      highestBid === null ? true : isBidHigher(action, highestBid)
    )
  ];
}
