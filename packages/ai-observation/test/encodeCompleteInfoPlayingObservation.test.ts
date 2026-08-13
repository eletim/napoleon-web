import { describe, expect, it } from "vitest";
import { RuleBasedAgent, runAutomatedGame } from "@napoleon/ai";
import type { AutomatedGameRecord, DecisionRecord } from "@napoleon/ai";
import {
  CARD_COUNT,
  NOT_IN_HAND_CLASS_INDEX,
  createRelativePlayerOrder,
  encodeCompleteInfoPlayingObservation,
  getCardIndex,
  getRelativePlayerIndex,
  validateEncodedCompleteInfoPlayingObservation
} from "../src/index.js";

describe("encodeCompleteInfoPlayingObservation", () => {
  it("encodes complete current hand ownership classes", async () => {
    const { record, decision } = await getFirstPlayingDecision(12345);
    const observation = encodeCompleteInfoPlayingObservation(
      decision.observation,
      decision.actualState,
      record.playerIds
    );
    const relativePlayerIds = createRelativePlayerOrder(record.playerIds, decision.playerId);

    validateEncodedCompleteInfoPlayingObservation(observation);
    expect(observation.cardOwnerClassByCard).toHaveLength(CARD_COUNT);

    for (const playerId of record.playerIds) {
      const relativeIndex = getRelativePlayerIndex(relativePlayerIds, playerId);

      for (const cardId of decision.actualState.hands[playerId] ?? []) {
        expect(observation.cardOwnerClassByCard[getCardIndex(cardId)]).toBe(relativeIndex);
      }
    }

    for (const cardId of [
      ...decision.actualState.unusedCardIds,
      ...decision.actualState.excludedCardIds,
      ...Object.values(decision.actualState.awardedPointCardIds).flat(),
      ...decision.actualState.currentTrickCardIds,
      ...decision.actualState.completedTrickCardIds
    ]) {
      expect(observation.cardOwnerClassByCard[getCardIndex(cardId)]).toBe(
        NOT_IN_HAND_CLASS_INDEX
      );
    }
  });

  it("uses actual card state so swapped hidden hands change the complete-info observation", async () => {
    const record = await createRecord(12345);
    const decision = getPlayingDecisions(record).find((candidate) => {
      const opponentIds = record.playerIds.filter((playerId) => playerId !== candidate.playerId);

      return opponentIds.every(
        (playerId) => (candidate.actualState.hands[playerId]?.length ?? 0) > 0
      );
    });

    if (decision === undefined) {
      throw new Error("Expected a playing decision with opponent hands.");
    }

    const swappedDecision = swapTwoOpponentHands(record, decision);
    const original = encodeCompleteInfoPlayingObservation(
      decision.observation,
      decision.actualState,
      record.playerIds
    );
    const swapped = encodeCompleteInfoPlayingObservation(
      swappedDecision.observation,
      swappedDecision.actualState,
      record.playerIds
    );

    expect(original.cardOwnerClassByCard).not.toEqual(swapped.cardOwnerClassByCard);
    expect(original.legalPlayMask).toEqual(swapped.legalPlayMask);
    expect(original.trumpSuitOneHot).toEqual(swapped.trumpSuitOneHot);
  });

  it("produces deterministic complete-info observations for the same state", async () => {
    const { record, decision } = await getFirstPlayingDecision(777);
    const first = encodeCompleteInfoPlayingObservation(
      decision.observation,
      decision.actualState,
      record.playerIds
    );
    const second = encodeCompleteInfoPlayingObservation(
      decision.observation,
      decision.actualState,
      record.playerIds
    );

    expect(first).toEqual(second);
  });

  it("keeps bidding history and completed trick history out of the compact observation", async () => {
    const { record, decision } = await getFirstPlayingDecision(12345);
    const observation = encodeCompleteInfoPlayingObservation(
      decision.observation,
      decision.actualState,
      record.playerIds
    );

    expect(observation).not.toHaveProperty("biddingHistory");
    expect(observation).not.toHaveProperty("completedTrickCardIndices");
    expect(observation).not.toHaveProperty("completedTrickPlayerIndices");
    expect(observation).not.toHaveProperty("completedTrickWinnerIndices");
    expect(observation).toHaveProperty("completedTrickCount");
  });
});

async function createRecord(seed: number): Promise<AutomatedGameRecord> {
  return runAutomatedGame({
    seed,
    createAgent: ({ rng }) => new RuleBasedAgent(rng)
  });
}

async function getFirstPlayingDecision(seed: number): Promise<{
  record: AutomatedGameRecord;
  decision: DecisionRecord;
}> {
  const record = await createRecord(seed);
  const decision = getPlayingDecisions(record)[0];

  if (decision === undefined) {
    throw new Error("Expected at least one playing decision.");
  }

  return { record, decision };
}

function getPlayingDecisions(record: AutomatedGameRecord): readonly DecisionRecord[] {
  return record.decisions.filter((candidate) => candidate.phase === "playing");
}

function swapTwoOpponentHands(
  record: AutomatedGameRecord,
  decision: DecisionRecord
): DecisionRecord {
  const opponentIds = record.playerIds.filter((playerId) => playerId !== decision.playerId);
  const [firstOpponentId, secondOpponentId] = opponentIds;

  if (firstOpponentId === undefined || secondOpponentId === undefined) {
    throw new Error("Expected at least two opponents.");
  }

  return {
    ...decision,
    actualState: {
      ...decision.actualState,
      hands: {
        ...decision.actualState.hands,
        [firstOpponentId]: decision.actualState.hands[secondOpponentId] ?? [],
        [secondOpponentId]: decision.actualState.hands[firstOpponentId] ?? []
      }
    }
  };
}
