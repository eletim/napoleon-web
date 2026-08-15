import { describe, expect, it } from "vitest";
import { RuleBasedAgent, runAutomatedGame } from "@napoleon/ai";
import type { Agent, AutomatedGameRecord, DecisionRecord, PlayerObservation } from "@napoleon/ai";
import {
  CARD_COUNT,
  CARD_IDS,
  EXCHANGE_ENCODER_SCHEMA_VERSION,
  createExchangeTrainingSample,
  createExchangeTrainingSamples,
  encodeDiscardAction,
  getCardIndex,
  validateEncodedBiddingHistory,
  validateEncodedExchangeObservation,
  validateExchangeTrainingSample
} from "../src/index.js";

const smokeSeed = 12345;
const integrationSeeds = [0, 1, 2, 12345, 0xffffffff] as const;

describe("createExchangeTrainingSample", () => {
  it("returns null for non-exchange decisions", async () => {
    const record = await createRecord(smokeSeed);
    const biddingDecision = record.decisions.find((decision) => decision.phase === "bidding");

    if (biddingDecision === undefined) {
      throw new Error("Expected a bidding decision.");
    }

    expect(createExchangeTrainingSample(record, biddingDecision)).toBeNull();
  });

  it("creates one fixed-shape exchange sample from a fixed-seed runAutomatedGame", async () => {
    const record = await createRecord(smokeSeed);
    const exchangeDecision = getExchangeDecisions(record)[0];
    const sample = createExchangeTrainingSample(record, exchangeDecision);

    if (sample === null) {
      throw new Error("Expected an exchange sample.");
    }

    expect(sample.sampleType).toBe("exchange-training-sample");
    expect(sample.schemaVersion).toBe(EXCHANGE_ENCODER_SCHEMA_VERSION);
    expect(sample.seed).toBe(record.seed);
    expect(sample.step).toBe(exchangeDecision.step);
    expect(sample.actingPlayerId).toBe(exchangeDecision.playerId);
    expect(sample.relativePlayerIds[0]).toBe(exchangeDecision.playerId);
    expect(sample.relativePlayerIds).toEqual(sample.observation.relativePlayerIds);
    expect(sample.observation.calledAdjutantCardMask).toHaveLength(CARD_COUNT);
    expect(sample.observation.selfHandMask).toHaveLength(CARD_COUNT);
    expect(sample.observation.legalDiscardCardMask).toHaveLength(CARD_COUNT);
    expect(sample.actorTarget.discardTargetMask).toHaveLength(CARD_COUNT);
    expect(sum(sample.observation.selfHandMask)).toBe(13);
    expect(sample.observation.legalDiscardCardMask).toEqual(sample.observation.selfHandMask);
    expect(sum(sample.actorTarget.discardTargetMask)).toBe(3);
    expectMaskSubset(sample.actorTarget.discardTargetMask, sample.observation.legalDiscardCardMask);
    expect(sample.observation).not.toHaveProperty("discardTargetMask");
    expect(sample.observation).not.toHaveProperty("actualState");
    expect(sample.observation).not.toHaveProperty("adjutantPlayerId");
    expect(sample.observation).not.toHaveProperty("revealedAdjutantPlayerOneHot");
    validateEncodedExchangeObservation(sample.observation);
    validateEncodedBiddingHistory(sample.observation.biddingHistory);
    validateExchangeTrainingSample(sample);
  });

  it("builds legalDiscardCardMask from Napoleon's 13-card self hand, not legal action combinations", async () => {
    const record = await createRecord(smokeSeed);
    const exchangeDecision = getExchangeDecisions(record)[0];
    const sample = createExchangeTrainingSample(record, {
      ...exchangeDecision,
      observation: {
        ...exchangeDecision.observation,
        legalActions: []
      }
    });

    if (sample === null) {
      throw new Error("Expected an exchange sample.");
    }

    const self = exchangeDecision.observation.view.players.find(
      (player) => player.id === exchangeDecision.playerId
    );

    if (self?.hand === undefined) {
      throw new Error("Expected Napoleon self hand.");
    }

    for (const card of self.hand) {
      expect(sample.observation.selfHandMask[getCardIndex(card.id)]).toBe(1);
      expect(sample.observation.legalDiscardCardMask[getCardIndex(card.id)]).toBe(1);
    }

    for (const player of exchangeDecision.observation.view.players) {
      if (player.id !== exchangeDecision.playerId) {
        expect(player).not.toHaveProperty("hand");
      }
    }
  });

  it("encodes discard-cards.cardIds as an order-insensitive exactly 3-hot target mask", async () => {
    const record = await createRecord(smokeSeed);
    const decision = getExchangeDecisions(record)[0];
    const sample = createExchangeTrainingSample(record, decision);

    if (sample === null || decision.action.type !== "discard-cards") {
      throw new Error("Expected an exchange sample.");
    }

    const reversedAction = {
      ...decision.action,
      cardIds: [...decision.action.cardIds].reverse()
    };

    expect(encodeDiscardAction(
      reversedAction,
      sample.observation.legalDiscardCardMask,
      decision.playerId
    )).toEqual(sample.actorTarget);
    expect(sum(sample.actorTarget.discardTargetMask)).toBe(3);
  });

  it("encodes all exchange decisions in seeded automated games deterministically", async () => {
    for (const seed of integrationSeeds) {
      const firstRecord = await createRecord(seed);
      const secondRecord = await createRecord(seed);
      const exchangeDecisions = getExchangeDecisions(firstRecord);
      const samples = createExchangeTrainingSamples(firstRecord);

      expect(secondRecord).toEqual(firstRecord);
      expect(samples).toEqual(createExchangeTrainingSamples(firstRecord));
      expect(samples).toEqual(createExchangeTrainingSamples(secondRecord));
      expect(samples).toHaveLength(exchangeDecisions.length);
      expect(samples).toHaveLength(1);

      for (const [index, sample] of samples.entries()) {
        const sourceDecision = exchangeDecisions[index];

        expect(sample.step).toBe(sourceDecision.step);
        expect(sample.relativePlayerIds[0]).toBe(sourceDecision.playerId);
        expect(sample.observation.contractTargetPointCards).toBeGreaterThanOrEqual(12);
        expect(sample.observation.contractTargetPointCards).toBeLessThanOrEqual(19);
        expect(sum(sample.observation.selfHandMask)).toBe(13);
        expect(sample.observation.legalDiscardCardMask).toHaveLength(CARD_COUNT);
        expect(sample.observation.legalDiscardCardMask).toEqual(sample.observation.selfHandMask);
        expect(sum(sample.actorTarget.discardTargetMask)).toBe(3);
        expectMaskSubset(sample.actorTarget.discardTargetMask, sample.observation.legalDiscardCardMask);
        validateExchangeTrainingSample(sample);
      }
    }
  });

  it("does not create exchange samples for all-pass games", async () => {
    const record = await runAutomatedGame({
      seed: smokeSeed,
      createAgent: ({ rng }) => new PassThenRuleBasedAgent(rng)
    });

    expect(record.result.resultType).toBe("all-pass");
    expect(createExchangeTrainingSamples(record)).toEqual([]);
  });

  it("rejects exchange decisions without a post-adjutant 13-card Napoleon hand or discardCount 3", async () => {
    const record = await createRecord(smokeSeed);
    const decision = getExchangeDecisions(record)[0];
    const self = decision.observation.view.players.find((player) => player.id === decision.playerId);

    if (self?.hand === undefined) {
      throw new Error("Expected Napoleon self hand.");
    }

    expect(() => createExchangeTrainingSample(record, {
      ...decision,
      observation: {
        ...decision.observation,
        view: {
          ...decision.observation.view,
          players: decision.observation.view.players.map((player) =>
            player.id === decision.playerId
              ? { ...player, hand: self.hand?.slice(1), handCount: 12 }
              : player
          )
        }
      }
    })).toThrow("Napoleon must have 13 cards at exchange");

    expect(() => createExchangeTrainingSample(record, {
      ...decision,
      observation: {
        ...decision.observation,
        view: {
          ...decision.observation.view,
          exchangeRequirement: { discardCount: 2 }
        }
      }
    })).toThrow("discardCount 3");
  });

  it("rejects illegal discard targets outside Napoleon self hand", async () => {
    const record = await createRecord(smokeSeed);
    const decision = getExchangeDecisions(record)[0];
    const sample = createExchangeTrainingSample(record, decision);

    if (sample === null || decision.action.type !== "discard-cards") {
      throw new Error("Expected an exchange sample.");
    }

    const illegalCardId = findFirstCardOutsideMask(sample.observation.legalDiscardCardMask);

    expect(() => encodeDiscardAction(
      {
        ...decision.action,
        cardIds: [illegalCardId, ...decision.action.cardIds.slice(0, 2)]
      },
      sample.observation.legalDiscardCardMask,
      decision.playerId
    )).toThrow("discardTargetMask");
  });

  it("rejects malformed exchange observations and discard actions", async () => {
    const record = await createRecord(smokeSeed);
    const decision = getExchangeDecisions(record)[0];
    const sample = createExchangeTrainingSample(record, decision);
    const otherPlayerId = record.playerIds.find((playerId) => playerId !== decision.playerId);

    if (sample === null || otherPlayerId === undefined || decision.action.type !== "discard-cards") {
      throw new Error("Expected an exchange sample and another player.");
    }

    expect(() => createExchangeTrainingSample(record, {
      ...decision,
      observation: {
        ...decision.observation,
        view: {
          ...decision.observation.view,
          trumpSuit: null
        }
      }
    })).toThrow("resolved trump suit and contract");

    expect(() => createExchangeTrainingSample(record, {
      ...decision,
      observation: {
        ...decision.observation,
        view: {
          ...decision.observation.view,
          contract: null
        }
      }
    })).toThrow("resolved trump suit and contract");

    expect(() => createExchangeTrainingSample(record, {
      ...decision,
      observation: {
        ...decision.observation,
        view: {
          ...decision.observation.view,
          adjutant: null
        }
      }
    })).toThrow("called adjutant card");

    expect(() => createExchangeTrainingSample(record, {
      ...decision,
      observation: {
        ...decision.observation,
        view: {
          ...decision.observation.view,
          currentPlayerId: otherPlayerId
        }
      }
    })).toThrow("current player");

    expect(() => encodeDiscardAction(
      { type: "pass", playerId: decision.playerId },
      sample.observation.legalDiscardCardMask,
      decision.playerId
    )).toThrow("requires a discard-cards action");

    expect(() => encodeDiscardAction(
      { ...decision.action, playerId: otherPlayerId },
      sample.observation.legalDiscardCardMask,
      decision.playerId
    )).toThrow("playerId must match");

    expect(() => encodeDiscardAction(
      { ...decision.action, cardIds: decision.action.cardIds.slice(0, 2) },
      sample.observation.legalDiscardCardMask,
      decision.playerId
    )).toThrow("exactly 3");

    expect(() => encodeDiscardAction(
      { ...decision.action, cardIds: [decision.action.cardIds[0], ...decision.action.cardIds.slice(0, 2)] },
      sample.observation.legalDiscardCardMask,
      decision.playerId
    )).toThrow("distinct");
  });
});

async function createRecord(seed: number): Promise<AutomatedGameRecord> {
  return runAutomatedGame({
    seed,
    createAgent: ({ rng }) => new RuleBasedAgent(rng)
  });
}

function getExchangeDecisions(record: AutomatedGameRecord): readonly DecisionRecord[] {
  return record.decisions.filter((decision) => decision.phase === "exchanging");
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

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function expectMaskSubset(child: readonly number[], parent: readonly number[]): void {
  for (const [index, value] of child.entries()) {
    if (value === 1) {
      expect(parent[index]).toBe(1);
    }
  }
}

function findFirstCardOutsideMask(mask: readonly number[]): string {
  const index = mask.findIndex((value) => value === 0);

  if (index === -1) {
    throw new Error("Expected at least one card outside mask.");
  }

  return CARD_IDS[index];
}
