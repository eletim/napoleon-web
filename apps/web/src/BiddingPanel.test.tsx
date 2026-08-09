import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { PublicBidAction, PublicBiddingState } from "@napoleon/protocol";
import { BiddingPanel } from "./BiddingPanel";

describe("BiddingPanel", () => {
  it("keeps bidding controls compact while preserving accessible suit names", () => {
    const html = renderToStaticMarkup(
      <BiddingPanel
        bidding={createBidding()}
        canPass={true}
        currentPlayerId="player-0"
        formatPlayerLabel={(playerId) => (playerId === "player-0" ? "自分" : "左側AI")}
        isBusy={false}
        legalBidActions={legalBidActions}
        onBid={vi.fn()}
        onPass={vi.fn()}
        selfPlayerId="player-0"
      />
    );

    expect(html).not.toContain("競り履歴");
    expect(html).not.toContain("履歴はまだありません");
    expect(html).not.toContain("選択中");
    expect(html).not.toContain("スペード</small>");
    expect(html).toContain("aria-label=\"スペードを選択\"");
    expect(html).toContain("aria-label=\"ハートを選択\"");
    expect(html).toContain("♠");
    expect(html).toContain("♥");
  });
});

const legalBidActions: readonly PublicBidAction[] = [
  { type: "bid", suit: "spades", targetPointCards: 13 },
  { type: "bid", suit: "hearts", targetPointCards: 14 }
];

function createBidding(): PublicBiddingState {
  return {
    starterPlayerId: "player-0",
    highestBid: {
      playerId: "player-0",
      suit: "spades",
      targetPointCards: 13
    },
    consecutivePassCount: 1,
    history: [
      { type: "bid", playerId: "player-0", suit: "spades", targetPointCards: 13 },
      { type: "pass", playerId: "player-1" }
    ]
  };
}
