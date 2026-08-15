import { describe, expect, it } from "vitest";
import { RuleBasedAgent, runAutomatedGame } from "@napoleon/ai";
import type { Agent, AutomatedGameRecord, DecisionRecord, PlayerObservation } from "@napoleon/ai";
import {
  ADJUTANT_ENCODER_SCHEMA_VERSION,
  CARD_COUNT,
  CARD_IDS,
  createAdjutantTrainingSample,
  createAdjutantTrainingSamples,
  decodeAdjutantAction,
  encodeAdjutantAction,
  encodeAdjutantObservation,
  getCardIndex,
  validateAdjutantTrainingSample,
  validateEncodedAdjutantObservation,
  validateEncodedBiddingHistory
} from "../src/index.js";

const smokeSeed = 12345;
const integrationSeeds = [0, 1, 2, 12345, 0xffffffff] as const;

describe("createAdjutantTrainingSample", () => {
  it("returns null for non-adjutant decisions", async () => {
    const record = await createRecord(smokeSeed);
    const biddingDecision = record.decisions.find((decision) => decision.phase === "bidding");

    if (biddingDecision === undefined) {
      throw new Error("Expected a bidding decision.");
    }

    expect(createAdjutantTrainingSample(record, biddingDecision)).toBeNull();
  });

  it("creates one fixed-shape adjutant sample from a fixed-seed runAutomatedGame", async () => {
    const record = await createRecord(smokeSeed);
    const adjutantDecisions = getAdjutantDecisions(record);
    const decision = adjutantDecisions[0];
    const sample = createAdjutantTrainingSample(record, decision);

    if (sample === null || decision.action.type !== "choose-adjutant") {
      throw new Error("Expected an adjutant sample.");
    }

    const self = decision.observation.view.players.find((player) => player.id === decision.playerId);

    if (self?.hand === undefined) {
      throw new Error("Expected Napoleon self hand.");
    }

    const selfCardIndex = getCardIndex(self.hand[0].id);
    const jokerIndex = getCardIndex("joker");

    expect(adjutantDecisions).toHaveLength(1);
    expect(createAdjutantTrainingSamples(record)).toHaveLength(1);
    expect(sample.sampleType).toBe("adjutant-training-sample");
    expect(sample.schemaVersion).toBe(ADJUTANT_ENCODER_SCHEMA_VERSION);
    expect(sample.seed).toBe(record.seed);
    expect(sample.step).toBe(decision.step);
    expect(sample.actingPlayerId).toBe(decision.playerId);
    expect(sample.relativePlayerIds[0]).toBe(decision.playerId);
    expect(sample.relativePlayerIds).toEqual(sample.observation.relativePlayerIds);
    expect(sample.actorTarget).toBe(encodeAdjutantAction(decision.action));
    expect(sample.actorTarget).toBeGreaterThanOrEqual(0);
    expect(sample.actorTarget).toBeLessThan(CARD_COUNT);
    expect(sample.observation.selfHandMask).toHaveLength(CARD_COUNT);
    expect(sum(sample.observation.selfHandMask)).toBe(10);
    expect(sample.observation.legalAdjutantMask).toHaveLength(CARD_COUNT);
    expect(sum(sample.observation.legalAdjutantMask)).toBe(CARD_COUNT);
    expect(sample.observation.legalAdjutantMask).toEqual(Array(CARD_COUNT).fill(1));
    expect(sample.observation.legalAdjutantMask[sample.actorTarget]).toBe(1);
    expect(sample.observation.legalAdjutantMask[jokerIndex]).toBe(1);
    expect(sample.observation.legalAdjutantMask[selfCardIndex]).toBe(1);
    expect(sample.observation.contractTargetPointCards).toBeGreaterThanOrEqual(12);
    expect(sample.observation.contractTargetPointCards).toBeLessThanOrEqual(19);
    expect(sample.observation).not.toHaveProperty("actualState");
    expect(sample.observation).not.toHaveProperty("actualHands");
    expect(sample.observation).not.toHaveProperty("unusedCards");
    expect(sample.observation).not.toHaveProperty("unusedCardIds");
    expect(sample.observation).not.toHaveProperty("adjutantPlayerId");
    expect(sample.observation).not.toHaveProperty("calledAdjutantCardMask");
    expect(sample.observation).not.toHaveProperty("teacherCardId");
    expect(sample.observation).not.toHaveProperty("handCountByPlayer");

    for (const player of decision.observation.view.players) {
      if (player.id !== decision.playerId) {
        expect(player).not.toHaveProperty("hand");
      }
    }

    expect(decodeAdjutantAction(sample.actorTarget, decision.playerId)).toEqual(decision.action);
    validateEncodedAdjutantObservation(sample.observation);
    validateEncodedBiddingHistory(sample.observation.biddingHistory);
    validateAdjutantTrainingSample(sample);
  });

  it("builds legalAdjutantMask from decision.observation.legalActions only", async () => {
    const record = await createRecord(smokeSeed);
    const decision = getAdjutantDecisions(record)[0];
    const legalActions = [
      { type: "choose-adjutant" as const, playerId: decision.playerId, cardId: "spades-A" },
      { type: "choose-adjutant" as const, playerId: decision.playerId, cardId: "joker" }
    ];
    const encoded = encodeAdjutantObservation({
      ...decision.observation,
      legalActions
    }, record.playerIds, createAdjutantTrainingSamples(record)[0].observation.biddingHistory);

    expect(sum(encoded.legalAdjutantMask)).toBe(2);
    expect(encoded.legalAdjutantMask[getCardIndex("spades-A")]).toBe(1);
    expect(encoded.legalAdjutantMask[getCardIndex("joker")]).toBe(1);
    expect(encoded.legalAdjutantMask[getCardIndex("hearts-A")]).toBe(0);
  });

  it("rejects an actorTarget outside the legal adjutant mask", async () => {
    const record = await createRecord(smokeSeed);
    const decision = getAdjutantDecisions(record)[0];

    if (decision.action.type !== "choose-adjutant") {
      throw new Error("Expected a choose-adjutant decision.");
    }

    const legalActions = decision.observation.legalActions.filter(
      (action) => action.type !== "choose-adjutant" || action.cardId !== decision.action.cardId
    );

    expect(() => createAdjutantTrainingSample(record, {
      ...decision,
      observation: {
        ...decision.observation,
        legalActions
      }
    })).toThrow("is not legal");
  });

  it("encodes all fixed-seed adjutant decisions deterministically", async () => {
    for (const seed of integrationSeeds) {
      const firstRecord = await createRecord(seed);
      const secondRecord = await createRecord(seed);
      const adjutantDecisions = getAdjutantDecisions(firstRecord);
      const samples = createAdjutantTrainingSamples(firstRecord);

      expect(secondRecord).toEqual(firstRecord);
      expect(samples).toEqual(createAdjutantTrainingSamples(firstRecord));
      expect(samples).toEqual(createAdjutantTrainingSamples(secondRecord));
      expect(samples).toHaveLength(adjutantDecisions.length);
      expect(samples).toHaveLength(1);

      for (const [index, sample] of samples.entries()) {
        const sourceDecision = adjutantDecisions[index];

        expect(sample.step).toBe(sourceDecision.step);
        expect(sample.relativePlayerIds[0]).toBe(sourceDecision.playerId);
        expect(sample.observation.relativePlayerIds[0]).toBe(sourceDecision.playerId);
        expect(sum(sample.observation.selfHandMask)).toBe(10);
        expect(sample.observation.legalAdjutantMask).toHaveLength(CARD_COUNT);
        expect(sample.observation.legalAdjutantMask).toEqual(
          createMaskFromAdjutantActions(sourceDecision.legalActions)
        );
        expect(sample.observation.legalAdjutantMask).toEqual(Array(CARD_COUNT).fill(1));
        expect(sample.observation.legalAdjutantMask[sample.actorTarget]).toBe(1);
        validateAdjutantTrainingSample(sample);
      }
    }
  });

  it("does not create adjutant samples for all-pass games", async () => {
    const record = await runAutomatedGame({
      seed: smokeSeed,
      createAgent: ({ rng }) => new PassThenRuleBasedAgent(rng)
    });

    expect(record.result.resultType).toBe("all-pass");
    expect(createAdjutantTrainingSamples(record)).toEqual([]);
  });

  it("keeps adjutant observations independent from complete-information card locations", async () => {
    const record = await createRecord(smokeSeed);
    const decision = getAdjutantDecisions(record)[0];
    const mutatedDecision: DecisionRecord = {
      ...decision,
      actualHands: Object.fromEntries(
        Object.entries(decision.actualHands).map(([playerId, hand]) => [
          playerId,
          playerId === decision.playerId ? hand : [...hand].reverse()
        ])
      ),
      actualState: {
        ...decision.actualState,
        hands: Object.fromEntries(
          Object.entries(decision.actualState.hands).map(([playerId, hand]) => [
            playerId,
            playerId === decision.playerId ? hand : [...hand].reverse()
          ])
        ),
        unusedCardIds: [...decision.actualState.unusedCardIds].reverse()
      }
    };
    const sampleA = createAdjutantTrainingSample(record, decision);
    const sampleB = createAdjutantTrainingSample(record, mutatedDecision);

    if (sampleA === null || sampleB === null) {
      throw new Error("Expected adjutant samples.");
    }

    expect(sampleA.observation).toEqual(sampleB.observation);
    expect(sampleA.actorTarget).toEqual(sampleB.actorTarget);
  });

  it("rejects malformed adjutant observations and actions", async () => {
    const record = await createRecord(smokeSeed);
    const decision = getAdjutantDecisions(record)[0];
    const otherPlayerId = record.playerIds.find((playerId) => playerId !== decision.playerId);
    const self = decision.observation.view.players.find((player) => player.id === decision.playerId);

    if (otherPlayerId === undefined || self?.hand === undefined) {
      throw new Error("Expected an adjutant decision and another player.");
    }

    expect(() => createAdjutantTrainingSample(record, {
      ...decision,
      observation: {
        ...decision.observation,
        view: {
          ...decision.observation.view,
          players: decision.observation.view.players.map((player) =>
            player.id === decision.playerId
              ? { ...player, hand: self.hand?.slice(1), handCount: 9 }
              : player
          )
        }
      }
    })).toThrow("10 cards");

    expect(() => createAdjutantTrainingSample(record, {
      ...decision,
      observation: {
        ...decision.observation,
        view: {
          ...decision.observation.view,
          currentPlayerId: otherPlayerId
        }
      }
    })).toThrow("current player");

    expect(() => createAdjutantTrainingSample(record, {
      ...decision,
      observation: {
        ...decision.observation,
        view: {
          ...decision.observation.view,
          adjutantChoiceRequirement: null
        }
      }
    })).toThrow("adjutant choice requirement");

    expect(() => encodeAdjutantAction(
      { type: "pass", playerId: decision.playerId },
      Array(CARD_COUNT).fill(1),
      decision.playerId
    )).toThrow("requires a choose-adjutant action");

    expect(() => encodeAdjutantAction(
      { type: "choose-adjutant", playerId: otherPlayerId, cardId: "joker" },
      Array(CARD_COUNT).fill(1),
      decision.playerId
    )).toThrow("playerId must match");
  });
});

async function createRecord(seed: number): Promise<AutomatedGameRecord> {
  return runAutomatedGame({
    seed,
    createAgent: ({ rng }) => new RuleBasedAgent(rng)
  });
}

function getAdjutantDecisions(record: AutomatedGameRecord): readonly DecisionRecord[] {
  return record.decisions.filter((decision) => decision.phase === "choosing-adjutant");
}

class PassThenRuleBasedAgent implements Agent {
  private readonly delegate: RuleBasedAgent;

  constructor(rng: () => number) {
    this.delegate = new RuleBasedAgent(rng);
  }

  async selectAction(observation: PlayerObservation) {
    if (observation.view.phase === "bidding") {
      return {
        type: "pass" as const,
        playerId: observation.playerId
      };
    }

    return this.delegate.selectAction(observation);
  }
}

function createMaskFromAdjutantActions(actions: readonly DecisionRecord["action"][]): readonly number[] {
  const mask = Array(CARD_COUNT).fill(0);

  for (const action of actions) {
    if (action.type !== "choose-adjutant") {
      throw new Error(`Expected choose-adjutant action, got ${action.type}.`);
    }

    mask[getCardIndex(action.cardId)] = 1;
  }

  return mask;
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}
