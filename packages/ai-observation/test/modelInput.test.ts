import { describe, expect, it } from "vitest";
import { RuleBasedAgent, runAutomatedGame } from "@napoleon/ai";
import {
  CARD_COUNT,
  FLAT_OBSERVATION_FEATURE_COUNT,
  MODEL_INPUT_FEATURE_COUNT,
  createPlayingTrainingSample,
  encodePlayingModelInput,
  getCardIndex
} from "../src/index.js";

describe("encodePlayingModelInput", () => {
  it("builds the fixed 6242-feature model_input from an encoded playing observation", async () => {
    const record = await runAutomatedGame({
      seed: 12345,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });
    const decision = record.decisions.find((candidate) => candidate.phase === "playing");

    if (decision === undefined) {
      throw new Error("Expected a playing decision.");
    }

    const sample = createPlayingTrainingSample(record, decision);

    if (sample === null) {
      throw new Error("Expected a playing sample.");
    }

    const modelInput = encodePlayingModelInput(sample.observation);

    expect(modelInput).toBeInstanceOf(Float32Array);
    expect(modelInput).toHaveLength(MODEL_INPUT_FEATURE_COUNT);

    const legalMaskOffset = 4 + 5 + 6 + CARD_COUNT + CARD_COUNT;
    expect(Array.from(modelInput.slice(legalMaskOffset, legalMaskOffset + CARD_COUNT))).toEqual(
      sample.observation.legalPlayMask
    );

    const specialCardOneHot = modelInput.slice(
      FLAT_OBSERVATION_FEATURE_COUNT,
      FLAT_OBSERVATION_FEATURE_COUNT + 4 * CARD_COUNT
    );
    const specialCardIndices = [
      sample.observation.specialCardIndices.oruma,
      sample.observation.specialCardIndices.yoromeki,
      sample.observation.specialCardIndices.seiJack,
      sample.observation.specialCardIndices.uraJack
    ];

    specialCardIndices.forEach((cardIndex, slotIndex) => {
      const row = Array.from(
        specialCardOneHot.slice(slotIndex * CARD_COUNT, (slotIndex + 1) * CARD_COUNT)
      );

      if (cardIndex === -1) {
        expect(row.reduce((sum, value) => sum + value, 0)).toBe(0);
      } else {
        expect(row[cardIndex]).toBe(1);
        expect(row.reduce((sum, value) => sum + value, 0)).toBe(1);
      }
    });
  });

  it("produces byte-identical model_input for the same observation", async () => {
    const record = await runAutomatedGame({
      seed: 777,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });
    const decision = record.decisions.find((candidate) => candidate.phase === "playing");

    if (decision === undefined) {
      throw new Error("Expected a playing decision.");
    }

    const sample = createPlayingTrainingSample(record, decision);

    if (sample === null) {
      throw new Error("Expected a playing sample.");
    }

    const first = encodePlayingModelInput(sample.observation);
    const second = encodePlayingModelInput(sample.observation);

    expect(Buffer.from(first.buffer)).toEqual(Buffer.from(second.buffer));
  });

  it("keeps empty card index rows all-zero in appended one-hot regions", async () => {
    const record = await runAutomatedGame({
      seed: 12345,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });
    const decision = record.decisions.find(
      (candidate) => candidate.phase === "playing" && candidate.observation.view.currentTrick.length === 0
    );

    if (decision === undefined) {
      throw new Error("Expected a lead playing decision.");
    }

    const sample = createPlayingTrainingSample(record, decision);

    if (sample === null) {
      throw new Error("Expected a playing sample.");
    }

    const modelInput = encodePlayingModelInput(sample.observation);
    const currentTrickCardStart = FLAT_OBSERVATION_FEATURE_COUNT + 4 * CARD_COUNT;
    const currentTrickCardRows = modelInput.slice(
      currentTrickCardStart,
      currentTrickCardStart + 5 * CARD_COUNT
    );

    expect(sample.observation.currentTrickCardIndices.every((index) => index === -1)).toBe(true);
    expect(Array.from(currentTrickCardRows).reduce((sum, value) => sum + value, 0)).toBe(0);
  });

  it("places the selected action inside the legal mask encoded in model_input", async () => {
    const record = await runAutomatedGame({
      seed: 12345,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });
    const decision = record.decisions.find((candidate) => candidate.phase === "playing");

    if (decision === undefined || decision.action.type !== "play-card") {
      throw new Error("Expected a play-card decision.");
    }

    const sample = createPlayingTrainingSample(record, decision);

    if (sample === null) {
      throw new Error("Expected a playing sample.");
    }

    const modelInput = encodePlayingModelInput(sample.observation);
    const legalMaskOffset = 4 + 5 + 6 + CARD_COUNT + CARD_COUNT;
    const selectedCardIndex = getCardIndex(decision.action.cardId);

    expect(modelInput[legalMaskOffset + selectedCardIndex]).toBe(1);
  });
});
