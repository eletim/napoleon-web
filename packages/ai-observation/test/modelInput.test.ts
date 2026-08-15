import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { RuleBasedAgent, runAutomatedGame } from "@napoleon/ai";
import {
  ADJUTANT_MODEL_INPUT_FEATURE_COUNT,
  ADJUTANT_MODEL_INPUT_LAYOUT,
  BIDDING_MODEL_INPUT_FEATURE_COUNT,
  BIDDING_MODEL_INPUT_LAYOUT,
  CARD_COUNT,
  COMPLETE_INFO_PLAYING_MODEL_INPUT_FEATURE_COUNT,
  COMPLETE_INFO_PLAYING_MODEL_INPUT_LAYOUT,
  EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
  EXCHANGE_MODEL_INPUT_LAYOUT,
  EXCHANGE_ENCODER_SCHEMA_VERSION,
  FLAT_OBSERVATION_FEATURE_COUNT,
  FLAT_OBSERVATION_LAYOUT,
  MAX_BIDDING_TARGET_POINT_CARDS,
  MIN_CONTRACT_TARGET_POINT_CARDS,
  MODEL_INPUT_LAYOUT,
  MODEL_INPUT_FEATURE_COUNT,
  MODEL_INPUT_ONEHOT_LAYOUT,
  createAdjutantTrainingSample,
  createBiddingTrainingSample,
  createCompleteInfoPlayingModelInput,
  createExchangeTrainingSample,
  createPlayingTrainingSample,
  encodeAdjutantModelInput,
  encodeBiddingModelInput,
  encodeCompleteInfoPlayingModelInput,
  encodeCompleteInfoPlayingObservation,
  encodeExchangeModelInput,
  encodePlayingModelInput,
  getCardIndex
} from "../src/index.js";
import type {
  AdjutantTrainingSample,
  BiddingTrainingSample,
  ExchangeTrainingSample,
  PlayingTrainingSample
} from "../src/index.js";

const pythonValidSampleUrl = new URL(
  "../../../python/tests/unit/fixtures/valid_sample.json",
  import.meta.url
);
const pythonNonplayingModelInputSamplesUrl = new URL(
  "../../../python/tests/unit/fixtures/nonplaying_model_input_samples.json",
  import.meta.url
);
const pythonValidSampleModelInputSha256 =
  "02507e208dcfbb1083f319bb693fac4b618d746fea22d9093433516fa11e9a56";
const pythonBiddingSampleModelInputSha256 =
  "8e3ddf90e5130c51cc5d81436f471ff82aaf87dd18baa43fb0417f24dd4689b9";
const pythonExchangeSampleModelInputSha256 =
  "46170cd80c99d66d174e11143821fb3ac07d1381dd8e24d4aab4295f9c673020";
const pythonAdjutantSampleModelInputSha256 =
  "94f30068c0f9bb274ba5b1d1869fee69fef3ccb116121fa51e7bb01787b5174b";
const pythonNonplayingModelInputSamples = JSON.parse(
  readFileSync(pythonNonplayingModelInputSamplesUrl, "utf8")
) as {
  bidding: BiddingTrainingSample;
  exchange: ExchangeTrainingSample;
  adjutant: AdjutantTrainingSample;
};

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
      { name: "biddingHistoryActionMask", start: 509, stop: 674, shape: [165], dtype: "float32" },
      { name: "latestBuriedEventPointCardMask", start: 674, stop: 727, shape: [53], dtype: "float32" },
      { name: "trickNumber", start: 727, stop: 728, shape: [1], dtype: "float32" },
      { name: "completedTrickCount", start: 728, stop: 729, shape: [1], dtype: "float32" },
      { name: "contractTargetPointCards", start: 729, stop: 730, shape: [1], dtype: "float32" },
      { name: "latestBuriedEventHiddenNonPointCount", start: 730, stop: 731, shape: [1], dtype: "float32" },
      { name: "latestBuriedEventPresent", start: 731, stop: 732, shape: [1], dtype: "float32" }
    ]);
    expect(MODEL_INPUT_ONEHOT_LAYOUT).toEqual([
      { name: "specialCardIndicesOneHot", start: 732, stop: 944, shape: [4, 53], dtype: "float32" },
      { name: "currentTrickCardIndicesOneHot", start: 944, stop: 1209, shape: [5, 53], dtype: "float32" },
      { name: "completedTrickCardIndicesOneHot", start: 1209, stop: 3859, shape: [50, 53], dtype: "float32" },
      { name: "currentTrickPlayerIndicesOneHot", start: 3859, stop: 3884, shape: [5, 5], dtype: "float32" },
      { name: "completedTrickPlayerIndicesOneHot", start: 3884, stop: 4134, shape: [50, 5], dtype: "float32" },
      { name: "completedTrickWinnerIndicesOneHot", start: 4134, stop: 4184, shape: [10, 5], dtype: "float32" },
      { name: "biddingHistoryActionTypeIndicesOneHot", start: 4184, stop: 4514, shape: [165, 2], dtype: "float32" },
      { name: "biddingHistoryPlayerIndicesOneHot", start: 4514, stop: 5339, shape: [165, 5], dtype: "float32" },
      { name: "biddingHistorySuitIndicesOneHot", start: 5339, stop: 5999, shape: [165, 4], dtype: "float32" },
      { name: "biddingHistoryTargetPointCardsOneHot", start: 5999, stop: 7649, shape: [165, 10], dtype: "float32" },
      { name: "selfRoleOneHot", start: 7649, stop: 7653, shape: [4], dtype: "float32" }
    ]);
    expect(MODEL_INPUT_LAYOUT).toEqual([
      ...FLAT_OBSERVATION_LAYOUT,
      ...MODEL_INPUT_ONEHOT_LAYOUT
    ]);
  });

  it("builds the fixed 7653-feature model_input from an encoded playing observation", async () => {
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
    expect(Array.from(modelInput.slice(MODEL_INPUT_FEATURE_COUNT - 4))).toEqual(
      sample.observation.selfRoleOneHot
    );

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

describe("encodeCompleteInfoPlayingModelInput", () => {
  it("fixes the compact complete-information playing layout at 385 features", () => {
    expect(COMPLETE_INFO_PLAYING_MODEL_INPUT_FEATURE_COUNT).toBe(385);
    expect(COMPLETE_INFO_PLAYING_MODEL_INPUT_LAYOUT).toEqual([
      { name: "cardOwnerClassByCardOneHot", start: 0, stop: 318, shape: [53, 6], dtype: "float32" },
      { name: "capturedPointCardCountByPlayer", start: 318, stop: 323, shape: [5], dtype: "float32" },
      { name: "trumpSuitOneHot", start: 323, stop: 327, shape: [4], dtype: "float32" },
      { name: "contractTargetPointCards", start: 327, stop: 328, shape: [1], dtype: "float32" },
      { name: "napoleonPlayerOneHot", start: 328, stop: 333, shape: [5], dtype: "float32" },
      { name: "revealedAdjutantPlayerOneHot", start: 333, stop: 339, shape: [6], dtype: "float32" },
      { name: "selfRoleOneHot", start: 339, stop: 343, shape: [4], dtype: "float32" },
      { name: "calledAdjutantCardIndex", start: 343, stop: 344, shape: [1], dtype: "float32" },
      { name: "specialCardIndices", start: 344, stop: 348, shape: [4], dtype: "float32" },
      { name: "currentTrickSlotMask", start: 348, stop: 353, shape: [5], dtype: "float32" },
      { name: "currentTrickCardIndices", start: 353, stop: 358, shape: [5], dtype: "float32" },
      { name: "currentTrickPlayerIndicesOneHot", start: 358, stop: 383, shape: [5, 5], dtype: "float32" },
      { name: "trickNumber", start: 383, stop: 384, shape: [1], dtype: "float32" },
      { name: "completedTrickCount", start: 384, stop: 385, shape: [1], dtype: "float32" }
    ]);
    expect(COMPLETE_INFO_PLAYING_MODEL_INPUT_LAYOUT.map((slice) => slice.name)).not.toContain(
      "legalPlayMask"
    );
  });

  it("builds deterministic compact input and returns legal play mask separately", async () => {
    const record = await runAutomatedGame({
      seed: 12345,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });
    const decision = record.decisions.find((candidate) => candidate.phase === "playing");

    if (decision === undefined) {
      throw new Error("Expected a playing decision.");
    }

    const observation = encodeCompleteInfoPlayingObservation(
      decision.observation,
      decision.actualState,
      record.playerIds
    );
    const wrapped = createCompleteInfoPlayingModelInput(observation);
    const direct = encodeCompleteInfoPlayingModelInput(observation);
    const repeated = encodeCompleteInfoPlayingModelInput(observation);

    expect(wrapped.modelInput).toBeInstanceOf(Float32Array);
    expect(wrapped.modelInput).toHaveLength(COMPLETE_INFO_PLAYING_MODEL_INPUT_FEATURE_COUNT);
    expect(wrapped.legalPlayMask).toEqual(observation.legalPlayMask);
    expect(Array.from(wrapped.modelInput)).toEqual(Array.from(direct));
    expect(Buffer.from(direct.buffer)).toEqual(Buffer.from(repeated.buffer));
  });

  it("one-hot encodes card ownership in the compact model input", async () => {
    const record = await runAutomatedGame({
      seed: 777,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });
    const decision = record.decisions.find((candidate) => candidate.phase === "playing");

    if (decision === undefined) {
      throw new Error("Expected a playing decision.");
    }

    const observation = encodeCompleteInfoPlayingObservation(
      decision.observation,
      decision.actualState,
      record.playerIds
    );
    const modelInput = encodeCompleteInfoPlayingModelInput(observation);
    const ownerRegion = modelInput.slice(0, CARD_COUNT * 6);

    observation.cardOwnerClassByCard.forEach((ownerClass, cardIndex) => {
      const row = Array.from(ownerRegion.slice(cardIndex * 6, (cardIndex + 1) * 6));

      expect(row[ownerClass]).toBe(1);
      expect(row.reduce((sum, value) => sum + value, 0)).toBe(1);
    });
  });

  it("tracks the public playing model input count", () => {
    expect(MODEL_INPUT_FEATURE_COUNT).toBe(7653);
  });
});

describe("encodeBiddingModelInput", () => {
  it("matches the Python BIDDING_MODEL_INPUT_LAYOUT slice contract", () => {
    expect(BIDDING_MODEL_INPUT_LAYOUT).toEqual([
      { name: "selfHandMask", start: 0, stop: 53, shape: [53], dtype: "float32" },
      { name: "legalBidMask", start: 53, stop: 94, shape: [41], dtype: "float32" },
      { name: "starterPlayerOneHot", start: 94, stop: 99, shape: [5], dtype: "float32" },
      { name: "highestBidPresent", start: 99, stop: 100, shape: [1], dtype: "float32" },
      { name: "highestBidPlayerOneHot", start: 100, stop: 105, shape: [5], dtype: "float32" },
      { name: "highestBidSuitOneHot", start: 105, stop: 109, shape: [4], dtype: "float32" },
      { name: "highestBidTargetPointCardsOneHot", start: 109, stop: 119, shape: [10], dtype: "float32" },
      { name: "consecutivePassCountOneHot", start: 119, stop: 125, shape: [6], dtype: "float32" },
      { name: "biddingHistoryActionMask", start: 125, stop: 290, shape: [165], dtype: "float32" },
      { name: "biddingHistoryActionTypeIndicesOneHot", start: 290, stop: 620, shape: [165, 2], dtype: "float32" },
      { name: "biddingHistoryPlayerIndicesOneHot", start: 620, stop: 1445, shape: [165, 5], dtype: "float32" },
      { name: "biddingHistorySuitIndicesOneHot", start: 1445, stop: 2105, shape: [165, 4], dtype: "float32" },
      { name: "biddingHistoryTargetPointCardsOneHot", start: 2105, stop: 3755, shape: [165, 10], dtype: "float32" }
    ]);
  });

  it("matches Python tensorize_sample model_input bytes for the shared bidding fixture", () => {
    const modelInput = encodeBiddingModelInput(createBiddingFixture().observation);

    expect(modelInput).toHaveLength(BIDDING_MODEL_INPUT_FEATURE_COUNT);
    expect(sha256Float32(modelInput)).toBe(pythonBiddingSampleModelInputSha256);
  });

  it("keeps absent highest bid and empty history one-hot rows all-zero", () => {
    const sample = createBiddingFixture();
    const modelInput = encodeBiddingModelInput(sample.observation);

    expect(sumModelInputSlice(modelInput, "highestBidPlayerOneHot", BIDDING_MODEL_INPUT_LAYOUT)).toBe(0);
    expect(sumModelInputSlice(modelInput, "highestBidSuitOneHot", BIDDING_MODEL_INPUT_LAYOUT)).toBe(0);
    expect(sumModelInputSlice(modelInput, "highestBidTargetPointCardsOneHot", BIDDING_MODEL_INPUT_LAYOUT)).toBe(0);
    expect(sumModelInputSlice(modelInput, "biddingHistoryActionTypeIndicesOneHot", BIDDING_MODEL_INPUT_LAYOUT)).toBe(0);
    expect(sumModelInputSlice(modelInput, "biddingHistoryPlayerIndicesOneHot", BIDDING_MODEL_INPUT_LAYOUT)).toBe(0);
    expect(sumModelInputSlice(modelInput, "biddingHistorySuitIndicesOneHot", BIDDING_MODEL_INPUT_LAYOUT)).toBe(0);
    expect(sumModelInputSlice(modelInput, "biddingHistoryTargetPointCardsOneHot", BIDDING_MODEL_INPUT_LAYOUT)).toBe(0);
  });
});

describe("encodeExchangeModelInput", () => {
  it("matches the Python EXCHANGE_MODEL_INPUT_LAYOUT slice contract", () => {
    expect(EXCHANGE_MODEL_INPUT_LAYOUT).toEqual([
      { name: "trumpSuitOneHot", start: 0, stop: 4, shape: [4], dtype: "float32" },
      { name: "selfHandMask", start: 4, stop: 57, shape: [53], dtype: "float32" },
      { name: "partialDiscardMask", start: 57, stop: 110, shape: [53], dtype: "float32" },
      { name: "legalDiscardCardMask", start: 110, stop: 163, shape: [53], dtype: "float32" },
      { name: "calledAdjutantCardMask", start: 163, stop: 216, shape: [53], dtype: "float32" },
      { name: "exchangeStepIndexOneHot", start: 216, stop: 219, shape: [3], dtype: "float32" },
      { name: "remainingDiscardCountOneHot", start: 219, stop: 223, shape: [4], dtype: "float32" },
      { name: "contractTargetPointCardsOneHot", start: 223, stop: 234, shape: [11], dtype: "float32" },
      { name: "handCountByPlayer", start: 234, stop: 239, shape: [5], dtype: "float32" },
      { name: "specialCardIndicesOneHot", start: 239, stop: 451, shape: [4, 53], dtype: "float32" },
      { name: "biddingHistoryActionMask", start: 451, stop: 616, shape: [165], dtype: "float32" },
      { name: "biddingHistoryActionTypeIndicesOneHot", start: 616, stop: 946, shape: [165, 2], dtype: "float32" },
      { name: "biddingHistoryPlayerIndicesOneHot", start: 946, stop: 1771, shape: [165, 5], dtype: "float32" },
      { name: "biddingHistorySuitIndicesOneHot", start: 1771, stop: 2431, shape: [165, 4], dtype: "float32" },
      { name: "biddingHistoryTargetPointCardsOneHot", start: 2431, stop: 4081, shape: [165, 10], dtype: "float32" }
    ]);
  });

  it("matches Python tensorize_sample model_input bytes for the shared exchange fixture", () => {
    const modelInput = encodeExchangeModelInput(createExchangeFixture().observation);

    expect(modelInput).toHaveLength(EXCHANGE_MODEL_INPUT_FEATURE_COUNT);
    expect(sha256Float32(modelInput)).toBe(pythonExchangeSampleModelInputSha256);
  });

  it("encodes exchange contract target min/max edges as 9..19 one-hot", () => {
    const minSample = createExchangeFixture();
    const minInput = encodeExchangeModelInput(minSample.observation);
    const maxSample = createExchangeFixture();
    const maxInput = encodeExchangeModelInput({
      ...maxSample.observation,
      contractTargetPointCards: MAX_BIDDING_TARGET_POINT_CARDS
    });

    expect(Array.from(modelInputSlice(minInput, "contractTargetPointCardsOneHot", EXCHANGE_MODEL_INPUT_LAYOUT))).toEqual(
      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    );
    expect(Array.from(modelInputSlice(maxInput, "contractTargetPointCardsOneHot", EXCHANGE_MODEL_INPUT_LAYOUT))).toEqual(
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    );
    expect(() => encodeExchangeModelInput({
      ...minSample.observation,
      contractTargetPointCards: MIN_CONTRACT_TARGET_POINT_CARDS - 1
    })).toThrow("contractTargetPointCards");
  });

  it("rejects empty special card sentinel values", () => {
    const sample = createExchangeFixture();

    expect(() => encodeExchangeModelInput({
      ...sample.observation,
      specialCardIndices: {
        ...sample.observation.specialCardIndices,
        seiJack: -1
      }
    })).toThrow("specialCardIndices.seiJack");
  });
});

describe("encodeAdjutantModelInput", () => {
  it("matches the Python ADJUTANT_MODEL_INPUT_LAYOUT slice contract", () => {
    expect(ADJUTANT_MODEL_INPUT_LAYOUT).toEqual([
      { name: "trumpSuitOneHot", start: 0, stop: 4, shape: [4], dtype: "float32" },
      { name: "selfHandMask", start: 4, stop: 57, shape: [53], dtype: "float32" },
      { name: "legalAdjutantMask", start: 57, stop: 110, shape: [53], dtype: "float32" },
      { name: "contractTargetPointCardsOneHot", start: 110, stop: 121, shape: [11], dtype: "float32" },
      { name: "specialCardIndicesOneHot", start: 121, stop: 333, shape: [4, 53], dtype: "float32" },
      { name: "biddingHistoryActionMask", start: 333, stop: 498, shape: [165], dtype: "float32" },
      { name: "biddingHistoryActionTypeIndicesOneHot", start: 498, stop: 828, shape: [165, 2], dtype: "float32" },
      { name: "biddingHistoryPlayerIndicesOneHot", start: 828, stop: 1653, shape: [165, 5], dtype: "float32" },
      { name: "biddingHistorySuitIndicesOneHot", start: 1653, stop: 2313, shape: [165, 4], dtype: "float32" },
      { name: "biddingHistoryTargetPointCardsOneHot", start: 2313, stop: 3963, shape: [165, 10], dtype: "float32" }
    ]);
  });

  it("matches Python tensorize_sample model_input bytes for the shared adjutant fixture", () => {
    const modelInput = encodeAdjutantModelInput(createAdjutantFixture().observation);

    expect(modelInput).toHaveLength(ADJUTANT_MODEL_INPUT_FEATURE_COUNT);
    expect(sha256Float32(modelInput)).toBe(pythonAdjutantSampleModelInputSha256);
  });

  it("encodes adjutant contract target min/max edges as 9..19 one-hot", () => {
    const minSample = createAdjutantFixture();
    const minInput = encodeAdjutantModelInput(minSample.observation);
    const maxSample = createAdjutantFixture();
    const maxInput = encodeAdjutantModelInput({
      ...maxSample.observation,
      contractTargetPointCards: MAX_BIDDING_TARGET_POINT_CARDS
    });

    expect(Array.from(modelInputSlice(minInput, "contractTargetPointCardsOneHot", ADJUTANT_MODEL_INPUT_LAYOUT))).toEqual(
      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    );
    expect(Array.from(modelInputSlice(maxInput, "contractTargetPointCardsOneHot", ADJUTANT_MODEL_INPUT_LAYOUT))).toEqual(
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    );
    expect(() => encodeAdjutantModelInput({
      ...minSample.observation,
      contractTargetPointCards: MIN_CONTRACT_TARGET_POINT_CARDS - 1
    })).toThrow("contractTargetPointCards");
  });

  it("rejects empty special card sentinel values", () => {
    const sample = createAdjutantFixture();

    expect(() => encodeAdjutantModelInput({
      ...sample.observation,
      specialCardIndices: {
        ...sample.observation.specialCardIndices,
        uraJack: -1
      }
    })).toThrow("specialCardIndices.uraJack");
  });
});

describe("non-playing model input smoke", () => {
  it("builds finite deterministic model_input from real runAutomatedGame decisions", async () => {
    const record = await runAutomatedGame({
      seed: 12345,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });
    const biddingDecision = record.decisions.find((decision) => decision.phase === "bidding");
    const adjutantDecision = record.decisions.find(
      (decision) => decision.phase === "choosing-adjutant"
    );
    const exchangeDecision = record.decisions.find((decision) => decision.phase === "exchanging");

    if (
      biddingDecision === undefined ||
      adjutantDecision === undefined ||
      exchangeDecision === undefined
    ) {
      throw new Error("Expected bidding, choosing-adjutant, and exchanging decisions.");
    }

    const biddingSample = createBiddingTrainingSample(record, biddingDecision);
    const adjutantSample = createAdjutantTrainingSample(record, adjutantDecision);
    const exchangeSample = createExchangeTrainingSample(record, exchangeDecision);

    if (biddingSample === null || adjutantSample === null || exchangeSample === null) {
      throw new Error("Expected all non-playing samples.");
    }

    assertRealBiddingModelInput(biddingSample);
    assertRealAdjutantModelInput(adjutantSample);
    assertRealExchangeModelInput(exchangeSample);
  });

  it("does not change model_input when only teacher targets change", () => {
    const firstBidding = createBiddingFixture();
    const secondBidding = {
      ...firstBidding,
      actorTarget: 5
    };
    const firstExchange = createExchangeFixture();
    const secondExchange = {
      ...firstExchange,
      actorTarget: {
        discardTargetMask: createMask([3, 4, 5])
      }
    };
    const firstAdjutant = createAdjutantFixture();
    const secondAdjutant = {
      ...firstAdjutant,
      actorTarget: 21
    };

    expect(sha256Float32(encodeBiddingModelInput(firstBidding.observation))).toBe(
      sha256Float32(encodeBiddingModelInput(secondBidding.observation))
    );
    expect(sha256Float32(encodeExchangeModelInput(firstExchange.observation))).toBe(
      sha256Float32(encodeExchangeModelInput(secondExchange.observation))
    );
    expect(sha256Float32(encodeAdjutantModelInput(firstAdjutant.observation))).toBe(
      sha256Float32(encodeAdjutantModelInput(secondAdjutant.observation))
    );
  });
});

function assertRealBiddingModelInput(sample: BiddingTrainingSample): void {
  const first = encodeBiddingModelInput(sample.observation);
  const second = encodeBiddingModelInput(sample.observation);

  expect(first).toHaveLength(BIDDING_MODEL_INPUT_FEATURE_COUNT);
  expect(Array.from(first).every(Number.isFinite)).toBe(true);
  expect(sha256Float32(first)).toBe(sha256Float32(second));
  expect(Array.from(modelInputSlice(first, "legalBidMask", BIDDING_MODEL_INPUT_LAYOUT))).toEqual(
    sample.observation.legalBidMask
  );
}

function assertRealExchangeModelInput(sample: ExchangeTrainingSample): void {
  const first = encodeExchangeModelInput(sample.observation);
  const second = encodeExchangeModelInput(sample.observation);

  expect(first).toHaveLength(EXCHANGE_MODEL_INPUT_FEATURE_COUNT);
  expect(Array.from(first).every(Number.isFinite)).toBe(true);
  expect(sha256Float32(first)).toBe(sha256Float32(second));
  expect(Array.from(modelInputSlice(first, "legalDiscardCardMask", EXCHANGE_MODEL_INPUT_LAYOUT))).toEqual(
    sample.observation.legalDiscardCardMask
  );
}

function assertRealAdjutantModelInput(sample: AdjutantTrainingSample): void {
  const first = encodeAdjutantModelInput(sample.observation);
  const second = encodeAdjutantModelInput(sample.observation);

  expect(first).toHaveLength(ADJUTANT_MODEL_INPUT_FEATURE_COUNT);
  expect(Array.from(first).every(Number.isFinite)).toBe(true);
  expect(sha256Float32(first)).toBe(sha256Float32(second));
  expect(Array.from(modelInputSlice(first, "legalAdjutantMask", ADJUTANT_MODEL_INPUT_LAYOUT))).toEqual(
    sample.observation.legalAdjutantMask
  );
}

function sha256Float32(values: Float32Array): string {
  return createHash("sha256")
    .update(Buffer.from(values.buffer, values.byteOffset, values.byteLength))
    .digest("hex");
}

function modelInputSlice(
  values: Float32Array,
  name: string,
  layout: readonly { name: string; start: number; stop: number }[]
): Float32Array {
  const feature = layout.find((candidate) => candidate.name === name);

  if (feature === undefined) {
    throw new Error(`Unknown model_input feature ${name}.`);
  }

  return values.slice(feature.start, feature.stop);
}

function sumModelInputSlice(
  values: Float32Array,
  name: string,
  layout: readonly { name: string; start: number; stop: number }[]
): number {
  return Array.from(modelInputSlice(values, name, layout)).reduce((total, value) => total + value, 0);
}

function createBiddingFixture(): BiddingTrainingSample {
  return structuredClone(pythonNonplayingModelInputSamples.bidding);
}

function createExchangeFixture(): ExchangeTrainingSample {
  const sample = structuredClone(pythonNonplayingModelInputSamples.exchange);
  sample.schemaVersion = EXCHANGE_ENCODER_SCHEMA_VERSION;
  sample.observation = {
    ...sample.observation,
    schemaVersion: EXCHANGE_ENCODER_SCHEMA_VERSION,
    partialDiscardMask: createMask([]),
    exchangeStepIndex: 0,
    remainingDiscardCount: 3
  };
  return sample;
}

function createAdjutantFixture(): AdjutantTrainingSample {
  return structuredClone(pythonNonplayingModelInputSamples.adjutant);
}

function createMask(indices: readonly number[], length = CARD_COUNT): number[] {
  const values = Array(length).fill(0);

  for (const index of indices) {
    values[index] = 1;
  }

  return values;
}
