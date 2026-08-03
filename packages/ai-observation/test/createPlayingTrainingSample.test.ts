import { describe, expect, it } from "vitest";
import { RuleBasedAgent, runAutomatedGame } from "@napoleon/ai";
import type { ActualCardState, AutomatedGameRecord, DecisionRecord } from "@napoleon/ai";
import {
  BIDDING_ACTION_TYPE_BID,
  BIDDING_ACTION_TYPE_PASS,
  CARD_COUNT,
  EMPTY_BIDDING_ACTION_TYPE,
  EMPTY_BIDDING_SUIT_INDEX,
  EMPTY_PLAYER_INDEX,
  MAX_BIDDING_ACTION_COUNT,
  NOT_IN_HAND_CLASS_INDEX,
  createPlayingTrainingSample,
  createPlayingTrainingSamples,
  encodeBiddingHistory,
  encodeBeliefTarget,
  encodePlayingObservation,
  getCardIndex,
  getBiddingSuitIndex,
  validateEncodedBeliefTarget,
  validateEncodedBiddingHistory,
  validateEncodedPlayingObservation
} from "../src/index.js";

const integrationSeeds = [0, 1, 2, 12345, 0xffffffff] as const;

describe("createPlayingTrainingSample", () => {
  it("returns null for non-playing decisions", async () => {
    const record = await createRecord(12345);
    const biddingDecision = record.decisions.find((decision) => decision.phase === "bidding");

    if (biddingDecision === undefined) {
      throw new Error("Expected a bidding decision.");
    }

    expect(createPlayingTrainingSample(record, biddingDecision)).toBeNull();
  });

  it("creates one actor and belief sample for a playing decision", async () => {
    const record = await createRecord(12345);
    const decision = getPlayingDecisions(record)[0];
    const sample = createPlayingTrainingSample(record, decision);

    if (sample === null) {
      throw new Error("Expected a playing sample.");
    }

    expect(sample.seed).toBe(record.seed);
    expect(sample.step).toBe(decision.step);
    expect(sample.actingPlayerId).toBe(decision.playerId);
    expect(sample.relativePlayerIds).toEqual(sample.observation.relativePlayerIds);
    expect(sum(sample.observation.biddingHistory.actionMask)).toBe(
      record.decisions.filter(
        (candidate) => candidate.phase === "bidding" && candidate.step < decision.step
      ).length
    );
    expect(sample.actorTarget.selectedCardIndex).toBe(
      getCardIndex(decision.action.type === "play-card" ? decision.action.cardId : "")
    );
    validateEncodedPlayingObservation(sample.observation);
    validateEncodedBeliefTarget(sample.beliefTarget);
  });

  it("keeps actor observation independent from complete-information card ownership", async () => {
    const record = await createRecord(12345);
    const decision = getPlayingDecisions(record).find((candidate) => {
      const opponentIds = record.playerIds.filter((playerId) => playerId !== candidate.playerId);

      return opponentIds.every((playerId) => (candidate.actualState.hands[playerId]?.length ?? 0) > 0);
    });

    if (decision === undefined) {
      throw new Error("Expected a playing decision with opponent hands.");
    }

    const swappedDecision = swapTwoOpponentHands(record, decision);
    const sampleA = createPlayingTrainingSample(record, decision);
    const sampleB = createPlayingTrainingSample(record, swappedDecision);

    if (sampleA === null || sampleB === null) {
      throw new Error("Expected playing samples.");
    }

    expect(sampleA.observation).toEqual(sampleB.observation);
    expect(sampleA.observation.biddingHistory).toEqual(sampleB.observation.biddingHistory);
    expect(sampleA.actorTarget).toEqual(sampleB.actorTarget);
    expect(sampleA.beliefTarget).not.toEqual(sampleB.beliefTarget);
  });

  it("encodes public bidding decisions before the first playing decision in order", async () => {
    const record = await createRecord(12345);
    const decision = getPlayingDecisions(record)[0];
    const sample = createPlayingTrainingSample(record, decision);

    if (sample === null) {
      throw new Error("Expected a playing sample.");
    }

    const biddingDecisions = record.decisions.filter(
      (candidate) => candidate.phase === "bidding" && candidate.step < decision.step
    );
    const history = sample.observation.biddingHistory;

    expect(sum(history.actionMask)).toBe(biddingDecisions.length);

    biddingDecisions.forEach((biddingDecision, index) => {
      expect(history.actionMask[index]).toBe(1);
      expect(history.playerIndices[index]).toBe(
        sample.relativePlayerIds.indexOf(biddingDecision.playerId)
      );

      if (biddingDecision.action.type === "pass") {
        expect(history.actionTypeIndices[index]).toBe(BIDDING_ACTION_TYPE_PASS);
        expect(history.suitIndices[index]).toBe(EMPTY_BIDDING_SUIT_INDEX);
        expect(history.targetPointCards[index]).toBe(0);
      } else if (biddingDecision.action.type === "bid") {
        expect(history.actionTypeIndices[index]).toBe(BIDDING_ACTION_TYPE_BID);
        expect(history.suitIndices[index]).toBe(getBiddingSuitIndex(biddingDecision.action.suit));
        expect(history.targetPointCards[index]).toBe(biddingDecision.action.targetPointCards);
      } else {
        throw new Error(`Unexpected bidding action: ${biddingDecision.action.type}`);
      }
    });

    expect(history.actionMask[biddingDecisions.length]).toBe(0);
    expect(history.actionTypeIndices[biddingDecisions.length]).toBe(EMPTY_BIDDING_ACTION_TYPE);
    expect(history.playerIndices[biddingDecisions.length]).toBe(EMPTY_PLAYER_INDEX);
    expect(history.suitIndices[biddingDecisions.length]).toBe(EMPTY_BIDDING_SUIT_INDEX);
    expect(history.targetPointCards[biddingDecisions.length]).toBe(0);
  });

  it("represents all-pass contracts as real passes without a synthetic bid", async () => {
    const sourceRecord = await createRecord(12345);
    const sourceDecision = getPlayingDecisions(sourceRecord)[0];
    const passPlayerIds = sourceRecord.playerIds;
    const allPassDecisions = passPlayerIds.map((playerId, index): DecisionRecord => ({
      ...sourceRecord.decisions[index],
      step: index + 1,
      playerId,
      phase: "bidding",
      action: { type: "pass", playerId },
      observation: {
        ...sourceRecord.decisions[index].observation,
        playerId
      }
    }));
    const decision: DecisionRecord = {
      ...sourceDecision,
      step: allPassDecisions.length + 1,
      observation: {
        ...sourceDecision.observation,
        view: {
          ...sourceDecision.observation.view,
          trumpSuit: "spades",
          contract: {
            napoleonPlayerId: passPlayerIds[0],
            trumpSuit: "spades",
            targetPointCards: 12
          }
        }
      }
    };
    const record: AutomatedGameRecord = {
      ...sourceRecord,
      decisions: [...allPassDecisions, decision]
    };
    const sample = createPlayingTrainingSample(record, decision);

    if (sample === null) {
      throw new Error("Expected a playing sample.");
    }

    expect(sample.observation.contractTargetPointCards).toBe(12);
    expect(sample.observation.trumpSuitOneHot).toEqual([1, 0, 0, 0]);
    expect(sum(sample.observation.biddingHistory.actionMask)).toBe(5);
    expect(sample.observation.biddingHistory.actionTypeIndices.slice(0, 5)).toEqual(
      Array(5).fill(BIDDING_ACTION_TYPE_PASS)
    );
    expect(sample.observation.biddingHistory.targetPointCards.slice(0, 5)).toEqual(
      Array(5).fill(0)
    );
    expect(sample.observation.biddingHistory.actionTypeIndices).not.toContain(
      BIDDING_ACTION_TYPE_BID
    );
  });

  it("rejects bidding histories longer than the fixed maximum", async () => {
    const record = await createRecord(12345);
    const decision = getPlayingDecisions(record)[0];
    const tooManyBids = Array.from({ length: MAX_BIDDING_ACTION_COUNT + 1 }, (_, index) => ({
      ...record.decisions[0],
      step: index + 1,
      playerId: record.playerIds[index % record.playerIds.length],
      phase: "bidding" as const,
      action: {
        type: "pass" as const,
        playerId: record.playerIds[index % record.playerIds.length]
      }
    }));
    const shiftedDecision = {
      ...decision,
      step: tooManyBids.length + 1
    };
    const overflowingRecord = {
      ...record,
      decisions: [...tooManyBids, shiftedDecision]
    };

    expect(() => encodeBiddingHistory(
      overflowingRecord,
      shiftedDecision,
      record.playerIds
    )).toThrow(`Bidding history cannot exceed ${MAX_BIDDING_ACTION_COUNT} actions`);
  });

  it("does not include bidding decisions at or after the playing decision step", async () => {
    const record = await createRecord(12345);
    const decision = getPlayingDecisions(record)[0];
    const relativePlayerIds = [
      decision.playerId,
      ...record.playerIds.filter((id) => id !== decision.playerId)
    ];
    const futureBiddingDecision = {
      ...record.decisions[0],
      step: decision.step + 1,
      phase: "bidding" as const,
      playerId: record.playerIds[0],
      action: { type: "pass" as const, playerId: record.playerIds[0] }
    };
    const history = encodeBiddingHistory(
      {
        ...record,
        decisions: [...record.decisions, futureBiddingDecision]
      },
      decision,
      relativePlayerIds
    );

    expect(sum(history.actionMask)).toBe(
      record.decisions.filter(
        (candidate) => candidate.phase === "bidding" && candidate.step < decision.step
      ).length
    );
  });

  it("rejects mismatched action, observation, and view player identities", async () => {
    const record = await createRecord(12345);
    const decision = getPlayingDecisions(record)[0];
    const otherPlayerId = record.playerIds.find((playerId) => playerId !== decision.playerId);

    if (otherPlayerId === undefined || decision.action.type !== "play-card") {
      throw new Error("Expected a playing decision and another player.");
    }

    expect(() => createPlayingTrainingSample(record, {
      ...decision,
      action: {
        ...decision.action,
        playerId: otherPlayerId
      }
    })).toThrow("Decision action playerId must match decision playerId");

    expect(() => createPlayingTrainingSample(record, {
      ...decision,
      observation: {
        ...decision.observation,
        playerId: otherPlayerId
      }
    })).toThrow("Decision playerId must match observation playerId");

    expect(() => createPlayingTrainingSample(record, {
      ...decision,
      observation: {
        ...decision.observation,
        view: {
          ...decision.observation.view,
          selfId: otherPlayerId
        }
      }
    })).toThrow("Decision playerId must match observation view.selfId");
  });

  it("rejects decisions that do not match the source record", async () => {
    const record = await createRecord(12345);
    const decision = getPlayingDecisions(record)[0];
    const missingStepDecision = {
      ...decision,
      step: Math.max(...record.decisions.map((candidate) => candidate.step)) + 1
    };

    expect(() => createPlayingTrainingSample(record, missingStepDecision)).toThrow(
      "Decision step must exist in record.decisions"
    );

    const otherRecord = await createRecord(0);
    const differentRecordDecision = getPlayingDecisions(otherRecord).find((candidate) => {
      const sourceDecision = record.decisions.find((source) => source.step === candidate.step);

      return sourceDecision !== undefined && !actionsEqualForTest(sourceDecision, candidate);
    });

    if (differentRecordDecision === undefined) {
      throw new Error("Expected a different-record decision with a conflicting source step.");
    }

    expect(() => createPlayingTrainingSample(record, differentRecordDecision)).toThrow(
      "Decision must match record decision at step"
    );
  });

  it("converts all playing decisions in order", async () => {
    const record = await createRecord(12345);
    const playingDecisions = getPlayingDecisions(record);
    const samples = createPlayingTrainingSamples(record);

    expect(samples).toHaveLength(playingDecisions.length);
    expect(samples.map((sample) => sample.step)).toEqual(
      playingDecisions.map((decision) => decision.step)
    );
    expect(samples).toEqual(createPlayingTrainingSamples(record));
  });

  it("encodes multiple seeded automated games deterministically", async () => {
    for (const seed of integrationSeeds) {
      const firstRecord = await createRecord(seed);
      const secondRecord = await createRecord(seed);

      expect(secondRecord).toEqual(firstRecord);

      const samples = createPlayingTrainingSamples(firstRecord);

      expect(samples).toEqual(createPlayingTrainingSamples(firstRecord));
      expect(samples).toEqual(createPlayingTrainingSamples(secondRecord));
      expect(samples.length).toBeGreaterThan(0);
      expect(samples).toHaveLength(50);

      for (const sample of samples) {
        validateEncodedPlayingObservation(sample.observation);
        validateEncodedBeliefTarget(sample.beliefTarget);
        expect(sample.observation.legalPlayMask[sample.actorTarget.selectedCardIndex]).toBe(1);

        const sourceDecision = firstRecord.decisions.find(
          (decision) => decision.step === sample.step
        );

        if (sourceDecision === undefined) {
          throw new Error(`Missing source decision for step ${sample.step}.`);
        }

        expect(sample.observation).toEqual(
          encodePlayingObservation(
            sourceDecision.observation,
            firstRecord.playerIds,
            sample.observation.biddingHistory
          )
        );
        validateEncodedBiddingHistory(sample.observation.biddingHistory);
        expect(sample.beliefTarget).toEqual(
          encodeBeliefTarget(
            sourceDecision.observation,
            sourceDecision.actualState,
            firstRecord.playerIds
          )
        );
        expect(sample.beliefTarget.ownerClassByCard).toHaveLength(CARD_COUNT);

        for (const ownerClass of sample.beliefTarget.ownerClassByCard) {
          expect(ownerClass).toBeGreaterThanOrEqual(0);
          expect(ownerClass).toBeLessThanOrEqual(NOT_IN_HAND_CLASS_INDEX);
        }

        const self = sourceDecision.observation.view.players.find(
          (player) => player.id === sourceDecision.playerId
        );

        if (self?.hand === undefined) {
          throw new Error("Expected self hand in source observation.");
        }

        expect(sum(sample.observation.selfHandMask)).toBe(self.hand.length);

        for (const card of self.hand) {
          expect(sample.observation.selfHandMask[getCardIndex(card.id)]).toBe(1);
        }

        for (const [index, legalValue] of sample.observation.legalPlayMask.entries()) {
          if (legalValue === 1) {
            expect(sample.observation.selfHandMask[index]).toBe(1);
          }
        }

        for (const player of sourceDecision.observation.view.players) {
          if (player.id !== sourceDecision.playerId) {
            expect(player).not.toHaveProperty("hand");
          }
        }
      }
    }
  });
});

async function createRecord(seed: number): Promise<AutomatedGameRecord> {
  return runAutomatedGame({
    seed,
    createAgent: ({ rng }) => new RuleBasedAgent(rng)
  });
}

function getPlayingDecisions(record: AutomatedGameRecord): readonly DecisionRecord[] {
  return record.decisions.filter((decision) => decision.phase === "playing");
}

function swapTwoOpponentHands(
  record: AutomatedGameRecord,
  decision: DecisionRecord
): DecisionRecord {
  const opponentIds = record.playerIds.filter((playerId) => playerId !== decision.playerId);
  const firstOpponentId = opponentIds[0];
  const secondOpponentId = opponentIds[1];

  if (firstOpponentId === undefined || secondOpponentId === undefined) {
    throw new Error("Expected two opponents.");
  }

  const swappedActualState: ActualCardState = {
    ...decision.actualState,
    hands: {
      ...decision.actualState.hands,
      [firstOpponentId]: decision.actualState.hands[secondOpponentId] ?? [],
      [secondOpponentId]: decision.actualState.hands[firstOpponentId] ?? []
    }
  };

  return {
    ...decision,
    actualState: swappedActualState,
    actualHands: swappedActualState.hands
  };
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function actionsEqualForTest(left: DecisionRecord, right: DecisionRecord): boolean {
  if (
    left.playerId !== right.playerId ||
    left.phase !== right.phase ||
    left.action.type !== right.action.type ||
    left.action.playerId !== right.action.playerId
  ) {
    return false;
  }

  if (left.action.type !== "play-card" || right.action.type !== "play-card") {
    return false;
  }

  return left.action.cardId === right.action.cardId;
}
