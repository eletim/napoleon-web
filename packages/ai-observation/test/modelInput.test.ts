import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { RuleBasedAgent, runAutomatedGame } from "@napoleon/ai";
import {
  CARD_COUNT,
  FLAT_OBSERVATION_FEATURE_COUNT,
  FLAT_OBSERVATION_LAYOUT,
  MODEL_INPUT_LAYOUT,
  MODEL_INPUT_FEATURE_COUNT,
  MODEL_INPUT_ONEHOT_LAYOUT,
  createPlayingTrainingSample,
  encodePlayingModelInput,
  getCardIndex
} from "../src/index.js";
import type { PlayingTrainingSample } from "../src/index.js";

const pythonValidSampleUrl = new URL(
  "../../../python/tests/unit/fixtures/valid_sample.json",
  import.meta.url
);
const pythonValidSampleModelInputSha256 =
  "699a9fc67c6b93c5d866c73b8461f498d1183b3cefdfea693e31373b8d5380d8";

describe("encodePlayingModelInput", () => {
  it("matches the Python MODEL_INPUT_LAYOUT slice contract", () => {
    expect(FLAT_OBSERVATION_LAYOUT).toEqual([
      { name: "trumpSuitOneHot", start: 0, stop: 4, shape: [4], dtype: "float32" },
      { name: "napoleonPlayerOneHot", start: 4, stop: 9, shape: [5], dtype: "float32" },
      { name: "revealedAdjutantPlayerOneHot", start: 9, stop: 15, shape: [6], dtype: "float32" },
      { name: "calledAdjutantCardMask", start: 15, stop: 68, shape: [53], dtype: "float32" },
      { name: "selfHandMask", start: 68, stop: 121, shape: [53], dtype: "float32" },
      { name: "legalPlayMask", start: 121, stop: 174, shape: [53], dtype: "float32" },
      { name: "handCountByPlayer", start: 174, stop: 179, shape: [5], dtype: "float32" },
      { name: "capturedPointCardMaskByPlayer", start: 179, stop: 444, shape: [5, 53], dtype: "float32" },
      { name: "currentTrickSlotMask", start: 444, stop: 449, shape: [5], dtype: "float32" },
      { name: "completedTrickSlotMask", start: 449, stop: 499, shape: [50], dtype: "float32" },
      { name: "completedTrickMask", start: 499, stop: 509, shape: [10], dtype: "float32" },
      { name: "biddingHistoryActionMask", start: 509, stop: 626, shape: [117], dtype: "float32" },
      { name: "latestBuriedEventPointCardMask", start: 626, stop: 679, shape: [53], dtype: "float32" },
      { name: "trickNumber", start: 679, stop: 680, shape: [1], dtype: "float32" },
      { name: "completedTrickCount", start: 680, stop: 681, shape: [1], dtype: "float32" },
      { name: "contractTargetPointCards", start: 681, stop: 682, shape: [1], dtype: "float32" },
      { name: "latestBuriedEventHiddenNonPointCount", start: 682, stop: 683, shape: [1], dtype: "float32" },
      { name: "latestBuriedEventPresent", start: 683, stop: 684, shape: [1], dtype: "float32" }
    ]);
    expect(MODEL_INPUT_ONEHOT_LAYOUT).toEqual([
      { name: "specialCardIndicesOneHot", start: 684, stop: 896, shape: [4, 53], dtype: "float32" },
      { name: "currentTrickCardIndicesOneHot", start: 896, stop: 1161, shape: [5, 53], dtype: "float32" },
      { name: "completedTrickCardIndicesOneHot", start: 1161, stop: 3811, shape: [50, 53], dtype: "float32" },
      { name: "currentTrickPlayerIndicesOneHot", start: 3811, stop: 3836, shape: [5, 5], dtype: "float32" },
      { name: "completedTrickPlayerIndicesOneHot", start: 3836, stop: 4086, shape: [50, 5], dtype: "float32" },
      { name: "completedTrickWinnerIndicesOneHot", start: 4086, stop: 4136, shape: [10, 5], dtype: "float32" },
      { name: "biddingHistoryActionTypeIndicesOneHot", start: 4136, stop: 4370, shape: [117, 2], dtype: "float32" },
      { name: "biddingHistoryPlayerIndicesOneHot", start: 4370, stop: 4955, shape: [117, 5], dtype: "float32" },
      { name: "biddingHistorySuitIndicesOneHot", start: 4955, stop: 5423, shape: [117, 4], dtype: "float32" },
      { name: "biddingHistoryTargetPointCardsOneHot", start: 5423, stop: 6242, shape: [117, 7], dtype: "float32" }
    ]);
    expect(MODEL_INPUT_LAYOUT).toEqual([
      ...FLAT_OBSERVATION_LAYOUT,
      ...MODEL_INPUT_ONEHOT_LAYOUT
    ]);
  });

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

  it("matches Python tensorize_sample model_input bytes for the shared valid sample fixture", () => {
    const sample = JSON.parse(readFileSync(pythonValidSampleUrl, "utf8")) as PlayingTrainingSample;
    const modelInput = encodePlayingModelInput(sample.observation);
    const digest = createHash("sha256")
      .update(Buffer.from(modelInput.buffer, modelInput.byteOffset, modelInput.byteLength))
      .digest("hex");

    expect(digest).toBe(pythonValidSampleModelInputSha256);
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
