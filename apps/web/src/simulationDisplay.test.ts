import { describe, expect, it } from "vitest";
import type { PublicSimulationDecision } from "@napoleon/protocol";
import {
  createSimulationFilename,
  formatActualStateSummary,
  formatCompletedTrickStatus,
  formatHandCounts,
  formatSimulationBiddingStatus,
  formatSimulationAction,
  formatSimulationLegalAction,
  getSimulationPlayerRole
} from "./simulationDisplay";

describe("simulation display helpers", () => {
  it("formats simulation actions for timeline rows", () => {
    expect(
      formatSimulationAction("player-2", {
        type: "bid",
        suit: "spades",
        targetPointCards: 15
      })
    ).toBe("player-2: スペード 15枚で入札");
    expect(formatSimulationAction("player-4", { type: "pass" })).toBe("player-4: パス");
    expect(
      formatSimulationAction("player-2", {
        type: "choose-adjutant",
        cardId: "spades-J"
      })
    ).toBe("player-2: J♠を副官指定");
    expect(
      formatSimulationAction("player-1", {
        type: "play-card",
        cardId: "hearts-10"
      })
    ).toBe("player-1: 10♥を出した");
    expect(
      formatSimulationAction("player-2", {
        type: "discard-cards",
        cardIds: ["clubs-2", "diamonds-3", "joker"]
      })
    ).toBe("player-2: 3枚を捨てた");
  });

  it("formats legal actions for decision details", () => {
    expect(formatSimulationLegalAction({ type: "play-card", cardId: "hearts-10" })).toBe(
      "10♥を出す"
    );
    expect(
      formatSimulationLegalAction({
        type: "discard-cards",
        cardIds: ["clubs-2", "diamonds-3", "joker"]
      })
    ).toBe("2♣、3♦、ジョーカーを捨てる");
  });

  it("formats player roles from the final result", () => {
    const result = {
      winner: "napoleon-team",
      napoleonTeamPointCards: 13,
      alliancePointCards: 7,
      targetPointCards: 13,
      napoleonPlayerId: "player-2",
      adjutantPlayerId: "player-4"
    } as const;

    expect(getSimulationPlayerRole("player-2", result)).toBe("ナポレオン");
    expect(getSimulationPlayerRole("player-4", result)).toBe("副官");
    expect(getSimulationPlayerRole("player-1", result)).toBe("連合軍");
  });

  it("formats timeline hand counts and download filenames", () => {
    const decision = {
      handCounts: {
        "player-0": 10,
        "player-1": 9
      }
    } as unknown as PublicSimulationDecision;

    expect(formatHandCounts(decision)).toBe("player-0:10 / player-1:9");
    expect(createSimulationFilename(12345)).toBe("napoleon-simulation-seed-12345.json");
  });

  it("formats observation and complete-state summaries", () => {
    const decision = {
      observation: {
        bidding: {
          starterPlayerId: "player-0",
          highestBid: {
            playerId: "player-2",
            suit: "hearts",
            targetPointCards: 14
          },
          consecutivePassCount: 1,
          history: [
            {
              type: "bid",
              playerId: "player-2",
              suit: "hearts",
              targetPointCards: 14
            }
          ]
        },
        completedTrickCount: 2,
        completedTricks: [
          { trickNumber: 1, winnerId: "player-0", cards: [] },
          { trickNumber: 2, winnerId: "player-1", cards: [] }
        ]
      },
      actualState: {
        hands: {},
        unusedCardIds: ["clubs-2"],
        excludedCardIds: ["clubs-3"],
        awardedPointCardIds: {
          "player-0": ["hearts-A"]
        },
        currentTrickCardIds: ["spades-4"],
        completedTrickCardIds: ["diamonds-5", "diamonds-6"]
      }
    } as unknown as PublicSimulationDecision;

    expect(formatSimulationBiddingStatus(decision.observation)).toBe(
      "player-2 ハート 14枚 / 履歴 1件"
    );
    expect(formatCompletedTrickStatus(decision.observation)).toBe("2件完了 / 保存 2件");
    expect(formatActualStateSummary(decision.actualState)).toBe(
      "未使用 1枚 / 除外 1枚 / 埋札得点 1枚 / 現在トリック 1枚 / 完了トリック 2枚"
    );
  });
});
