import { describe, expect, it } from "vitest";
import { createDeck } from "@napoleon/game-core";
import type { Suit } from "@napoleon/game-core";
import { encodeBiddingAction } from "@napoleon/ai-observation";
import {
  selectStrongestSuit,
  summarizeBiddingSamples,
  summarizeEvaluationGames
} from "../src/nonPlayingBiddingDiagnostics.js";

const deck = createDeck();

describe("non-playing bidding diagnostics", () => {
  it("aggregates pass, target, suit, and target x suit action distributions", () => {
    const spadesHand = handInput([
      "spades-A",
      "spades-K",
      "spades-Q",
      "spades-J",
      "spades-10"
    ]);
    const heartsHand = handInput([
      "hearts-A",
      "hearts-K",
      "hearts-Q",
      "hearts-J",
      "hearts-10"
    ]);

    const summary = summarizeBiddingSamples([
      sample(spadesHand, 0),
      sample(spadesHand, bidIndex(13, "spades")),
      sample(heartsHand, bidIndex(16, "hearts"))
    ]);

    expect(summary.bidding.decisionCount).toBe(3);
    expect(summary.bidding.passCount).toBe(1);
    expect(summary.bidding.passRate).toBeCloseTo(1 / 3);
    expect(summary.bidding.bidCount).toBe(2);
    expect(summary.bidding.bidRate).toBeCloseTo(2 / 3);
    expect(summary.bidding.targets.PASS.count).toBe(1);
    expect(summary.bidding.targets["13"].count).toBe(1);
    expect(summary.bidding.targets["16"].count).toBe(1);
    expect(summary.bidding.suits.spades.count).toBe(1);
    expect(summary.bidding.suits.hearts.count).toBe(1);
    expect(summary.bidding.targetSuit["13-spades"].count).toBe(1);
    expect(summary.bidding.targetSuit["16-hearts"].count).toBe(1);
  });

  it("uses RuleBased strongest suit scoring and tie break order", () => {
    expect(selectStrongestSuit(cards(["spades-A", "spades-K"]))).toBe("spades");
    expect(selectStrongestSuit(cards(["hearts-A", "hearts-K", "hearts-Q"]))).toBe("hearts");
    expect(selectStrongestSuit(cards(["diamonds-A", "diamonds-K", "diamonds-Q"]))).toBe("diamonds");
    expect(selectStrongestSuit(cards(["clubs-A", "clubs-K", "clubs-Q"]))).toBe("clubs");
    expect(selectStrongestSuit([])).toBe("spades");
  });

  it("builds strongest x selected count and rate tables", () => {
    const spadesHand = handInput(["spades-A", "spades-K", "spades-Q"]);
    const clubsHand = handInput(["clubs-A", "clubs-K", "clubs-Q"]);
    const summary = summarizeBiddingSamples([
      sample(spadesHand, bidIndex(13, "spades")),
      sample(spadesHand, bidIndex(14, "hearts")),
      sample(spadesHand, 0),
      sample(clubsHand, bidIndex(15, "clubs"))
    ]);

    expect(summary.strongestSuit.strongestCounts.spades).toBe(3);
    expect(summary.strongestSuit.strongestCounts.clubs).toBe(1);
    expect(summary.strongestSuit.selectedByStrongestCounts.spades.spades).toBe(1);
    expect(summary.strongestSuit.selectedByStrongestCounts.spades.hearts).toBe(1);
    expect(summary.strongestSuit.selectedByStrongestCounts.spades.PASS).toBe(1);
    expect(summary.strongestSuit.selectedByStrongestCounts.clubs.clubs).toBe(1);
    expect(summary.strongestSuit.selectedByStrongestRates.spades.spades).toBeCloseTo(1 / 3);
    expect(summary.strongestSuit.passRateByStrongest.spades).toBeCloseTo(1 / 3);
    expect(summary.strongestSuit.matchCount).toBe(2);
    expect(summary.strongestSuit.matchRate).toBeCloseTo(2 / 3);
  });

  it("aggregates seat rotation and game result metrics", () => {
    const summary = summarizeEvaluationGames({
      configuration: { policyAgentName: "Candidate" },
      run: {
        games: [
          standardGame(0, "Candidate", "napoleon", "napoleon-team", 16, 18),
          standardGame(1, "Candidate", "adjutant", "alliance", 15, 11),
          standardGame(2, "Candidate", "alliance", "alliance", 14, 12),
          {
            status: "completed",
            resultType: "all-pass",
            seats: [{ agentName: "Candidate", role: "unknown" }]
          },
          { status: "failed", seats: [{ agentName: "Candidate", role: "unknown" }] }
        ]
      }
    });

    expect(summary.gameCount).toBe(5);
    expect(summary.completedGames).toBe(4);
    expect(summary.failedGames).toBe(1);
    expect(summary.allPassCount).toBe(1);
    expect(summary.allPassRate).toBeCloseTo(1 / 4);
    expect(summary.candidateNapoleonRate).toBeCloseTo(1 / 3);
    expect(summary.candidateAdjutantRate).toBeCloseTo(1 / 3);
    expect(summary.candidateCitizenRate).toBeCloseTo(1 / 3);
    expect(summary.napoleonContractSuccessRate).toBe(1);
    expect(summary.napoleonMeanTarget).toBe(16);
    expect(summary.napoleonMeanPointCards).toBe(18);
    expect(summary.candidateWinRate).toBeCloseTo(2 / 3);
  });
});

function sample(modelInput: readonly number[], selectedActionIndex: number) {
  return {
    actingPlayerId: "player-0",
    selectedActionIndex,
    modelInput
  };
}

function bidIndex(targetPointCards: number, suit: Suit): number {
  return encodeBiddingAction({ type: "bid", playerId: "player-0", targetPointCards, suit });
}

function handInput(cardIds: readonly string[]): readonly number[] {
  const ids = new Set(cardIds);
  return deck.map((card) => ids.has(card.id) ? 1 : 0);
}

function cards(cardIds: readonly string[]) {
  const ids = new Set(cardIds);
  return deck.filter((card) => ids.has(card.id));
}

function standardGame(
  seatIndex: number,
  agentName: string,
  role: "napoleon" | "adjutant" | "alliance",
  winner: "napoleon-team" | "alliance",
  targetPointCards: number,
  napoleonTeamPointCards: number
) {
  return {
    status: "completed" as const,
    resultType: "standard" as const,
    seats: [
      { agentName: seatIndex === 0 ? agentName : "Other", role: "alliance" as const },
      { agentName: seatIndex === 1 ? agentName : "Other", role: "alliance" as const },
      { agentName: seatIndex === 2 ? agentName : "Other", role: "alliance" as const },
      { agentName: seatIndex === 3 ? agentName : "Other", role: "alliance" as const },
      { agentName: seatIndex === 4 ? agentName : "Other", role: "alliance" as const }
    ].map((seat, index) => index === seatIndex ? { ...seat, role } : seat),
    winner,
    contractSucceeded: winner === "napoleon-team",
    contract: { targetPointCards },
    pointCards: { napoleonTeam: napoleonTeamPointCards, alliance: 20 - napoleonTeamPointCards }
  };
}
