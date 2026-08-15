import { describe, expect, it } from "vitest";
import { RuleBasedAgent, runAutomatedGame } from "@napoleon/ai";
import type { ActualCardState, AutomatedGameRecord, DecisionRecord } from "@napoleon/ai";
import {
  BIDDING_ACTION_COUNT,
  BIDDING_ENCODER_SCHEMA_VERSION,
  BIDDING_HISTORY_SUIT_ORDER,
  CARD_COUNT,
  createBiddingTrainingSample,
  createBiddingTrainingSamples,
  encodeBiddingAction,
  encodeBiddingObservation,
  validateBiddingTrainingSample,
  validateEncodedBiddingObservation
} from "../src/index.js";

const smokeSeed = 12345;
const integrationSeeds = [0, 1, 2, 12345, 0xffffffff] as const;

describe("createBiddingTrainingSample", () => {
  it("returns null for non-bidding decisions", async () => {
    const record = await createRecord(smokeSeed);
    const playingDecision = record.decisions.find((decision) => decision.phase === "playing");

    if (playingDecision === undefined) {
      throw new Error("Expected a playing decision.");
    }

    expect(createBiddingTrainingSample(record, playingDecision)).toBeNull();
  });

  it("creates one actor sample for a bidding decision", async () => {
    const record = await createRecord(smokeSeed);
    const decision = getBiddingDecisions(record)[0];
    const sample = createBiddingTrainingSample(record, decision);

    if (sample === null) {
      throw new Error("Expected a bidding sample.");
    }

    expect(sample.sampleType).toBe("bidding-training-sample");
    expect(sample.schemaVersion).toBe(BIDDING_ENCODER_SCHEMA_VERSION);
    expect(sample.seed).toBe(record.seed);
    expect(sample.step).toBe(decision.step);
    expect(sample.actingPlayerId).toBe(decision.playerId);
    expect(sample.relativePlayerIds[0]).toBe(decision.playerId);
    expect(sample.relativePlayerIds).toEqual(sample.observation.relativePlayerIds);
    expect(sample.observation.legalBidMask).toHaveLength(BIDDING_ACTION_COUNT);
    expect(sample.observation.selfHandMask).toHaveLength(CARD_COUNT);
    expect(sample.actorTarget).toBe(encodeBiddingAction(decision.action));
    expect(sample.observation.legalBidMask[sample.actorTarget]).toBe(1);
    validateEncodedBiddingObservation(sample.observation);
    validateBiddingTrainingSample(sample);
  });

  it("encodes legalBidMask from decision.observation.legalActions without re-ranking suits", async () => {
    const record = await createRecord(smokeSeed);
    const decision = getBiddingDecisions(record)[0];
    const legalActions = [
      { type: "pass" as const, playerId: decision.playerId },
      { type: "bid" as const, playerId: decision.playerId, suit: "spades" as const, targetPointCards: 13 },
      { type: "bid" as const, playerId: decision.playerId, suit: "hearts" as const, targetPointCards: 13 }
    ];
    const encoded = encodeBiddingObservation({
      ...decision.observation,
      legalActions
    }, record.playerIds);

    expect(encoded.legalBidMask[0]).toBe(1);
    expect(encoded.legalBidMask[1]).toBe(1);
    expect(encoded.legalBidMask[2]).toBe(1);
    expect(encoded.legalBidMask[3]).toBe(0);
    expect(encoded.legalBidMask[4]).toBe(0);
    expect(sum(encoded.legalBidMask)).toBe(3);
  });

  it("keeps bidding observations independent from complete-information card state", async () => {
    const record = await createRecord(smokeSeed);
    const decision = getBiddingDecisions(record).find((candidate) => {
      const opponentIds = record.playerIds.filter((playerId) => playerId !== candidate.playerId);

      return opponentIds.every((playerId) => (candidate.actualState.hands[playerId]?.length ?? 0) > 0);
    });

    if (decision === undefined) {
      throw new Error("Expected a bidding decision with opponent hands.");
    }

    const swappedDecision = swapTwoOpponentHands(record, decision);
    const sampleA = createBiddingTrainingSample(record, decision);
    const sampleB = createBiddingTrainingSample(record, swappedDecision);

    if (sampleA === null || sampleB === null) {
      throw new Error("Expected bidding samples.");
    }

    expect(sampleA.observation).toEqual(sampleB.observation);
    expect(sampleA.actorTarget).toEqual(sampleB.actorTarget);
  });

  it("does not include future decisions in bidding history", async () => {
    const record = await createRecord(smokeSeed);
    const decision = getBiddingDecisions(record)[0];
    const futureDecision = {
      ...record.decisions.find((candidate) => candidate.phase === "playing"),
      step: decision.step + 1000
    } as DecisionRecord;
    const extendedRecord: AutomatedGameRecord = {
      ...record,
      decisions: [...record.decisions, futureDecision]
    };
    const sample = createBiddingTrainingSample(extendedRecord, decision);

    if (sample === null) {
      throw new Error("Expected a bidding sample.");
    }

    expect(sample.observation.biddingHistory).toEqual(
      encodeBiddingObservation(decision.observation, record.playerIds).biddingHistory
    );
  });

  it("falls back to public view bidding history when publicActionHistory is omitted", async () => {
    const record = await createRecord(smokeSeed);
    const decision = getBiddingDecisions(record).find(
      (candidate) => (candidate.observation.publicActionHistory?.length ?? 0) > 0
    );

    if (decision === undefined || decision.observation.view.bidding === null) {
      throw new Error("Expected a bidding decision with public history.");
    }

    const publicActionHistory = decision.observation.publicActionHistory ?? [];
    const encodedWithPublicActionHistory = encodeBiddingObservation(
      decision.observation,
      record.playerIds
    );
    const encodedFromViewHistory = encodeBiddingObservation({
      ...decision.observation,
      publicActionHistory: undefined,
      view: {
        ...decision.observation.view,
        bidding: {
          ...decision.observation.view.bidding,
          history: publicActionHistory.map((actionRecord) => actionRecord.action)
        }
      }
    }, record.playerIds);

    expect(encodedFromViewHistory.biddingHistory).toEqual(
      encodedWithPublicActionHistory.biddingHistory
    );
  });

  it("keeps the fifth all-pass teacher as pass index 0 and adds no spades-12 action", async () => {
    const record = await createSyntheticAllPassRecord(smokeSeed);
    const decision = record.decisions[4];
    const sample = createBiddingTrainingSample(record, decision);

    if (sample === null) {
      throw new Error("Expected a bidding sample.");
    }

    expect(decision.action).toEqual({ type: "pass", playerId: decision.playerId });
    expect(sample.actorTarget).toBe(0);
    expect(sample.observation.legalBidMask[0]).toBe(1);
    expect(sample.observation.legalBidMask).toHaveLength(29);
    expect(() => encodeBiddingAction({
      type: "bid",
      playerId: decision.playerId,
      suit: "spades",
      targetPointCards: 12
    })).toThrow("between 13 and 19");
  });

  it("converts all bidding decisions from fixed-seed runAutomatedGame and keeps teachers legal", async () => {
    const record = await createRecord(smokeSeed);
    const biddingDecisions = getBiddingDecisions(record);
    const samples = createBiddingTrainingSamples(record);

    expect(samples).toHaveLength(biddingDecisions.length);
    expect(samples.length).toBeGreaterThan(0);

    for (const [index, sample] of samples.entries()) {
      const sourceDecision = biddingDecisions[index];

      expect(sample.step).toBe(sourceDecision.step);
      expect(sample.actorTarget).toBe(encodeBiddingAction(sourceDecision.action));
      expect(sample.observation.legalBidMask[sample.actorTarget]).toBe(1);
      expect(sample.observation.legalBidMask).toEqual(
        createMaskFromBiddingActions(sourceDecision.legalActions)
      );
      expect(sample.relativePlayerIds[0]).toBe(sourceDecision.playerId);
      expect(sample.observation.relativePlayerIds[0]).toBe(sourceDecision.playerId);
      expect(sample.observation.legalBidMask).toEqual(
        encodeBiddingObservation(sourceDecision.observation, record.playerIds).legalBidMask
      );
      validateBiddingTrainingSample(sample);
    }
  });

  it("encodes multiple seeded automated games deterministically", async () => {
    for (const seed of integrationSeeds) {
      const firstRecord = await createRecord(seed);
      const secondRecord = await createRecord(seed);
      const firstSamples = createBiddingTrainingSamples(firstRecord);

      expect(secondRecord).toEqual(firstRecord);
      expect(firstSamples).toEqual(createBiddingTrainingSamples(firstRecord));
      expect(firstSamples).toEqual(createBiddingTrainingSamples(secondRecord));

      for (const sample of firstSamples) {
        expect(sample.observation.legalBidMask[sample.actorTarget]).toBe(1);
        expect(sample.relativePlayerIds[0]).toBe(sample.actingPlayerId);
        validateEncodedBiddingObservation(sample.observation);

        for (const player of firstRecord.decisions.find((decision) => decision.step === sample.step)?.observation.view.players ?? []) {
          if (player.id !== sample.actingPlayerId) {
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

async function createSyntheticAllPassRecord(seed: number): Promise<AutomatedGameRecord> {
  const source = await createRecord(seed);
  const firstFive = source.decisions.slice(0, 5);

  if (firstFive.length !== 5 || firstFive.some((decision) => decision.phase !== "bidding")) {
    throw new Error("Expected first five source decisions to be bidding decisions.");
  }

  const allPassDecisions = firstFive.map((decision, index) => {
    const playerId = source.playerIds[index];
    const history = source.playerIds.slice(0, index).map((historyPlayerId) => ({
      type: "pass" as const,
      playerId: historyPlayerId
    }));

    return {
      ...decision,
      playerId,
      action: { type: "pass" as const, playerId },
      legalActions: [
        { type: "pass" as const, playerId },
        ...BIDDING_HISTORY_SUIT_ORDER.map((suit) => ({
          type: "bid" as const,
          playerId,
          suit,
          targetPointCards: 13
        }))
      ],
      observation: {
        ...decision.observation,
        playerId,
        legalActions: [
          { type: "pass" as const, playerId },
          ...BIDDING_HISTORY_SUIT_ORDER.map((suit) => ({
            type: "bid" as const,
            playerId,
            suit,
            targetPointCards: 13
          }))
        ],
        publicActionHistory: history.map((action, historyIndex) => ({
          step: historyIndex + 1,
          playerId: action.playerId,
          phase: "bidding" as const,
          action
        })),
        view: {
          ...decision.observation.view,
          selfId: playerId,
          currentPlayerId: playerId,
          bidding: {
            starterPlayerId: source.playerIds[0],
            highestBid: null,
            consecutivePassCount: index,
            history
          },
          players: decision.observation.view.players.map((player) => {
            const { hand: _hand, ...publicPlayer } = player;

            return {
              ...publicPlayer,
              ...(player.id === playerId
                ? { hand: source.initialHands[playerId]?.map((cardId) => ({ id: cardId })) ?? [] }
                : {})
            };
          })
        }
      }
    };
  });

  return {
    ...source,
    decisions: [...allPassDecisions, ...source.decisions.slice(5)]
  };
}

function getBiddingDecisions(record: AutomatedGameRecord): readonly DecisionRecord[] {
  return record.decisions.filter((decision) => decision.phase === "bidding");
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

function createMaskFromBiddingActions(actions: DecisionRecord["legalActions"]): readonly number[] {
  const mask = Array(BIDDING_ACTION_COUNT).fill(0);

  for (const action of actions) {
    mask[encodeBiddingAction(action)] = 1;
  }

  return mask;
}
