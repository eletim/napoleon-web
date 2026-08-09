import { describe, expect, it } from "vitest";
import type { PublicCard, PublicRank, PublicSuit } from "@napoleon/protocol";
import { compareCardsForRiipai, getDisplayedHandCards } from "./handSorting";

describe("handSorting", () => {
  it("sorts standard cards by suit, rank, then joker", () => {
    const hand = [
      standardCard("clubs", "2"),
      jokerCard,
      standardCard("spades", "2"),
      standardCard("hearts", "K"),
      standardCard("spades", "A"),
      standardCard("diamonds", "10"),
      standardCard("clubs", "A")
    ];

    expect(getDisplayedHandCards(hand, "riipai").map((card) => card.id)).toEqual([
      "spades-A",
      "spades-2",
      "hearts-K",
      "diamonds-10",
      "clubs-A",
      "clubs-2",
      "joker"
    ]);
  });

  it("keeps original order available without cloning or mutating the source hand", () => {
    const hand = [
      standardCard("clubs", "2"),
      standardCard("spades", "A"),
      jokerCard
    ];
    const originalIds = hand.map((card) => card.id);

    expect(getDisplayedHandCards(hand, "original")).toBe(hand);
    expect(getDisplayedHandCards(hand, "riipai")).not.toBe(hand);
    expect(hand.map((card) => card.id)).toEqual(originalIds);
  });

  it("defines deterministic positions for every suit and rank", () => {
    const cards = suits.flatMap((suit) => ranks.map((rank) => standardCard(suit, rank)));
    const sortedIds = [...cards, jokerCard].sort(compareCardsForRiipai).map((card) => card.id);

    expect(sortedIds).toEqual([
      ...ranks.map((rank) => `spades-${rank}`),
      ...ranks.map((rank) => `hearts-${rank}`),
      ...ranks.map((rank) => `diamonds-${rank}`),
      ...ranks.map((rank) => `clubs-${rank}`),
      "joker"
    ]);
  });
});

const suits: readonly PublicSuit[] = ["spades", "hearts", "diamonds", "clubs"];
const ranks: readonly PublicRank[] = [
  "A",
  "K",
  "Q",
  "J",
  "10",
  "9",
  "8",
  "7",
  "6",
  "5",
  "4",
  "3",
  "2"
];

const jokerCard: PublicCard = { type: "joker", id: "joker" };

function standardCard(suit: PublicSuit, rank: PublicRank): PublicCard {
  return {
    type: "standard",
    id: `${suit}-${rank}`,
    suit,
    rank
  };
}
