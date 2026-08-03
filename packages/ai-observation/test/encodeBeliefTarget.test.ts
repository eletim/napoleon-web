import { describe, expect, it } from "vitest";
import { RuleBasedAgent, runAutomatedGame } from "@napoleon/ai";
import type { ActualCardState, DecisionRecord } from "@napoleon/ai";
import {
  CARD_COUNT,
  NOT_IN_HAND_CLASS_INDEX,
  createRelativePlayerOrder,
  encodeBeliefTarget,
  getRelativePlayerIndex,
  getCardIndex,
  validateEncodedBeliefTarget
} from "../src/index.js";

describe("encodeBeliefTarget", () => {
  it("encodes current hand ownership classes from complete card state", async () => {
    const { record, decision } = await getFirstPlayingDecision(12345);
    const target = encodeBeliefTarget(decision.observation, decision.actualState, record.playerIds);

    expect(target.ownerClassByCard).toHaveLength(CARD_COUNT);
    expect(target.hiddenOwnershipLossMask).toHaveLength(CARD_COUNT);
    validateEncodedBeliefTarget(target);

    const relativePlayerIds = createRelativePlayerOrder(record.playerIds, decision.playerId);

    for (const playerId of record.playerIds) {
      const relativeIndex = getRelativePlayerIndex(relativePlayerIds, playerId);
      const cards = decision.actualState.hands[playerId] ?? [];

      for (const cardId of cards) {
        expect(target.ownerClassByCard[getCardIndex(cardId)]).toBe(relativeIndex);
      }
    }

    for (const cardId of decision.actualState.excludedCardIds) {
      expect(target.ownerClassByCard[getCardIndex(cardId)]).toBe(NOT_IN_HAND_CLASS_INDEX);
    }

    for (const cardId of decision.actualState.completedTrickCardIds) {
      expect(target.ownerClassByCard[getCardIndex(cardId)]).toBe(NOT_IN_HAND_CLASS_INDEX);
    }
  });

  it("builds hidden ownership loss mask from public observation only", async () => {
    const { record, decision } = await getFirstPlayingDecision(12345);
    const target = encodeBeliefTarget(decision.observation, decision.actualState, record.playerIds);
    const self = decision.observation.view.players.find((player) => player.id === decision.playerId);

    if (self?.hand === undefined) {
      throw new Error("Expected self hand in observation.");
    }

    for (const card of self.hand) {
      expect(target.hiddenOwnershipLossMask[getCardIndex(card.id)]).toBe(0);
    }

    for (const playedCard of decision.observation.view.currentTrick) {
      expect(target.hiddenOwnershipLossMask[getCardIndex(playedCard.card.id)]).toBe(0);
    }

    for (const trick of decision.observation.view.completedTricks) {
      for (const playedCard of trick.cards) {
        expect(target.hiddenOwnershipLossMask[getCardIndex(playedCard.card.id)]).toBe(0);
      }
    }

    for (const player of decision.observation.view.players) {
      for (const card of player.capturedPointCards) {
        expect(target.hiddenOwnershipLossMask[getCardIndex(card.id)]).toBe(0);
      }
    }
  });

  it("rejects incomplete or duplicated complete-information labels", async () => {
    const { record, decision } = await getFirstPlayingDecision(12345);
    const firstSelfCard = decision.actualState.hands[decision.playerId]?.[0];

    if (firstSelfCard === undefined) {
      throw new Error("Expected self card in actual state.");
    }

    const duplicatedState: ActualCardState = {
      ...decision.actualState,
      excludedCardIds: [...decision.actualState.excludedCardIds, firstSelfCard]
    };

    expect(() => encodeBeliefTarget(decision.observation, duplicatedState, record.playerIds)).toThrow(
      "Actual card state must contain 53 card locations"
    );
  });
});

async function getFirstPlayingDecision(seed: number): Promise<{
  record: Awaited<ReturnType<typeof runAutomatedGame>>;
  decision: DecisionRecord;
}> {
  const record = await runAutomatedGame({
    seed,
    createAgent: ({ rng }) => new RuleBasedAgent(rng)
  });
  const decision = record.decisions.find((candidate) => candidate.phase === "playing");

  if (decision === undefined) {
    throw new Error("Expected at least one playing decision.");
  }

  return { record, decision };
}
