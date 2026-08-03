import { describe, expect, it } from "vitest";
import { RuleBasedAgent, runAutomatedGame } from "@napoleon/ai";
import {
  CARD_COUNT,
  CARDS_PER_TRICK,
  EMPTY_CARD_INDEX,
  EMPTY_PLAYER_INDEX,
  PLAYER_COUNT,
  PLAYING_ENCODER_SCHEMA_VERSION,
  TRICK_COUNT,
  encodePlayAction,
  encodePlayingObservation,
  getCardIndex,
  validateEncodedPlayingObservation
} from "../src/index.js";

describe("encodePlayingObservation", () => {
  it("encodes a playing observation into fixed shapes", async () => {
    const { record, decision } = await getFirstPlayingDecision(12345);
    const encoded = encodePlayingObservation(decision.observation, record.playerIds);

    expect(encoded.schemaVersion).toBe(PLAYING_ENCODER_SCHEMA_VERSION);
    expect(encoded.relativePlayerIds[0]).toBe(decision.playerId);
    expect(encoded.trumpSuitOneHot).toHaveLength(4);
    expect(encoded.napoleonPlayerOneHot).toHaveLength(PLAYER_COUNT);
    expect(encoded.revealedAdjutantPlayerOneHot).toHaveLength(PLAYER_COUNT + 1);
    expect(encoded.calledAdjutantCardMask).toHaveLength(CARD_COUNT);
    expect(encoded.selfHandMask).toHaveLength(CARD_COUNT);
    expect(encoded.legalPlayMask).toHaveLength(CARD_COUNT);
    expect(encoded.handCountByPlayer).toHaveLength(PLAYER_COUNT);
    expect(encoded.capturedPointCardMaskByPlayer).toHaveLength(PLAYER_COUNT);
    expect(encoded.currentTrickCardIndices).toHaveLength(CARDS_PER_TRICK);
    expect(encoded.currentTrickPlayerIndices).toHaveLength(CARDS_PER_TRICK);
    expect(encoded.currentTrickSlotMask).toHaveLength(CARDS_PER_TRICK);
    expect(encoded.completedTrickCardIndices).toHaveLength(TRICK_COUNT * CARDS_PER_TRICK);
    expect(encoded.completedTrickPlayerIndices).toHaveLength(TRICK_COUNT * CARDS_PER_TRICK);
    expect(encoded.completedTrickSlotMask).toHaveLength(TRICK_COUNT * CARDS_PER_TRICK);
    expect(encoded.completedTrickWinnerIndices).toHaveLength(TRICK_COUNT);
    expect(encoded.completedTrickMask).toHaveLength(TRICK_COUNT);
    expect(encoded.latestBuriedEventPointCardMask).toHaveLength(CARD_COUNT);
    validateEncodedPlayingObservation(encoded);
  });

  it("preserves current trick order and empty slots", async () => {
    const record = await runAutomatedGame({
      seed: 12345,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });
    const decision = record.decisions.find(
      (candidate) => candidate.phase === "playing" && candidate.observation.view.currentTrick.length > 0
    );

    if (decision === undefined) {
      throw new Error("Expected a playing decision with a non-empty current trick.");
    }

    const encoded = encodePlayingObservation(decision.observation, record.playerIds);
    const trick = decision.observation.view.currentTrick;

    expect(encoded.currentTrickSlotMask.slice(0, trick.length)).toEqual(
      Array(trick.length).fill(1)
    );
    expect(encoded.currentTrickSlotMask.slice(trick.length)).toEqual(
      Array(CARDS_PER_TRICK - trick.length).fill(0)
    );
    expect(encoded.currentTrickCardIndices.slice(0, trick.length)).toEqual(
      trick.map((playedCard) => getCardIndex(playedCard.card.id))
    );
    expect(encoded.currentTrickCardIndices.slice(trick.length)).toEqual(
      Array(CARDS_PER_TRICK - trick.length).fill(EMPTY_CARD_INDEX)
    );
    expect(encoded.currentTrickPlayerIndices.slice(trick.length)).toEqual(
      Array(CARDS_PER_TRICK - trick.length).fill(EMPTY_PLAYER_INDEX)
    );
  });

  it("encodes completed tricks as flattened fixed slots", async () => {
    const record = await runAutomatedGame({
      seed: 12345,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });
    const decision = record.decisions.find(
      (candidate) => candidate.phase === "playing" && candidate.observation.view.completedTricks.length > 0
    );

    if (decision === undefined) {
      throw new Error("Expected a playing decision with completed tricks.");
    }

    const encoded = encodePlayingObservation(decision.observation, record.playerIds);
    const completedSlotCount =
      decision.observation.view.completedTricks.length * CARDS_PER_TRICK;

    expect(encoded.completedTrickSlotMask.slice(0, completedSlotCount)).toEqual(
      Array(completedSlotCount).fill(1)
    );
    expect(encoded.completedTrickSlotMask.slice(completedSlotCount)).toEqual(
      Array(TRICK_COUNT * CARDS_PER_TRICK - completedSlotCount).fill(0)
    );
    expect(encoded.completedTrickMask.slice(0, decision.observation.view.completedTricks.length)).toEqual(
      Array(decision.observation.view.completedTricks.length).fill(1)
    );
  });

  it("encodes the latest buried-card event without hidden card ids", async () => {
    const record = await runAutomatedGame({
      seed: 12345,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });
    const decision = record.decisions.find(
      (candidate) =>
        candidate.phase === "playing" &&
        candidate.observation.view.latestEvent?.type === "buried-cards-resolved"
    );

    if (decision === undefined) {
      throw new Error("Expected a playing decision with a buried-card event.");
    }

    const encoded = encodePlayingObservation(decision.observation, record.playerIds);
    const event = decision.observation.view.latestEvent;

    if (event?.type !== "buried-cards-resolved") {
      throw new Error("Expected buried-card event.");
    }

    expect(encoded.latestBuriedEventPresent).toBe(1);
    expect(encoded.latestBuriedEventHiddenNonPointCount).toBe(event.hiddenNonPointCardCount);
    expect(sum(encoded.latestBuriedEventPointCardMask)).toBe(event.awardedPointCards.length);
  });

  it("rejects non-playing observations and non-play legal actions", async () => {
    const record = await runAutomatedGame({
      seed: 12345,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });
    const biddingDecision = record.decisions.find((candidate) => candidate.phase === "bidding");

    if (biddingDecision === undefined) {
      throw new Error("Expected a bidding decision.");
    }

    expect(() => encodePlayingObservation(biddingDecision.observation, record.playerIds)).toThrow(
      "requires a playing observation"
    );

    const { decision } = await getFirstPlayingDecision(12345);
    const invalidObservation = {
      ...decision.observation,
      legalActions: [
        ...decision.observation.legalActions,
        { type: "pass" as const, playerId: decision.playerId }
      ]
    };

    expect(() => encodePlayingObservation(invalidObservation, record.playerIds)).toThrow(
      "only play-card actions"
    );
  });

  it("encodes selected play-card actions and validates the legal mask", async () => {
    const { record, decision } = await getFirstPlayingDecision(12345);
    const observation = encodePlayingObservation(decision.observation, record.playerIds);

    expect(encodePlayAction(decision.action, observation.legalPlayMask)).toEqual({
      selectedCardIndex: getCardIndex(decision.action.type === "play-card" ? decision.action.cardId : "")
    });

    expect(() =>
      encodePlayAction(
        { type: "pass", playerId: decision.playerId },
        observation.legalPlayMask
      )
    ).toThrow("requires a play-card action");
    expect(() =>
      encodePlayAction(
        { type: "play-card", playerId: decision.playerId, cardId: "joker" },
        Array(CARD_COUNT).fill(0)
      )
    ).toThrow("Selected card is not legal");
  });
});

async function getFirstPlayingDecision(seed: number) {
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

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}
