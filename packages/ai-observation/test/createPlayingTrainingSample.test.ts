import { describe, expect, it } from "vitest";
import { RuleBasedAgent, runAutomatedGame } from "@napoleon/ai";
import type { ActualCardState, AutomatedGameRecord, DecisionRecord } from "@napoleon/ai";
import {
  CARD_COUNT,
  NOT_IN_HAND_CLASS_INDEX,
  createPlayingTrainingSample,
  createPlayingTrainingSamples,
  encodeBeliefTarget,
  encodePlayingObservation,
  getCardIndex,
  validateEncodedBeliefTarget,
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
    expect(sampleA.actorTarget).toEqual(sampleB.actorTarget);
    expect(sampleA.beliefTarget).not.toEqual(sampleB.beliefTarget);
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
          encodePlayingObservation(sourceDecision.observation, firstRecord.playerIds)
        );
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
