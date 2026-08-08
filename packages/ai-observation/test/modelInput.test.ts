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
  EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
  EXCHANGE_MODEL_INPUT_LAYOUT,
  FLAT_OBSERVATION_FEATURE_COUNT,
  FLAT_OBSERVATION_LAYOUT,
  MAX_BIDDING_TARGET_POINT_CARDS,
  MIN_CONTRACT_TARGET_POINT_CARDS,
  MODEL_INPUT_LAYOUT,
  MODEL_INPUT_FEATURE_COUNT,
  MODEL_INPUT_ONEHOT_LAYOUT,
  createAdjutantTrainingSample,
  createBiddingTrainingSample,
  createExchangeTrainingSample,
  createPlayingTrainingSample,
  encodeAdjutantModelInput,
  encodeBiddingModelInput,
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
  "699a9fc67c6b93c5d866c73b8461f498d1183b3cefdfea693e31373b8d5380d8";
const pythonBiddingSampleModelInputSha256 =
  "e4f86e8b9dd5661301e701c71ad6fc167123acedb8e118192e8ccfe6bc6df877";
const pythonExchangeSampleModelInputSha256 =
  "f48558692dbd4bf825b0130e9940db4271c2f9653057e5f2186cb36d6d551233";
const pythonAdjutantSampleModelInputSha256 =
  "2f0c47b5a113059ed7d06a9966db1ee553163104215981928d6111917d603d52";
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

describe("encodeBiddingModelInput", () => {
  it("matches the Python BIDDING_MODEL_INPUT_LAYOUT slice contract", () => {
    expect(BIDDING_MODEL_INPUT_LAYOUT).toEqual([
      { name: "selfHandMask", start: 0, stop: 53, shape: [53], dtype: "float32" },
      { name: "legalBidMask", start: 53, stop: 82, shape: [29], dtype: "float32" },
      { name: "starterPlayerOneHot", start: 82, stop: 87, shape: [5], dtype: "float32" },
      { name: "highestBidPresent", start: 87, stop: 88, shape: [1], dtype: "float32" },
      { name: "highestBidPlayerOneHot", start: 88, stop: 93, shape: [5], dtype: "float32" },
      { name: "highestBidSuitOneHot", start: 93, stop: 97, shape: [4], dtype: "float32" },
      { name: "highestBidTargetPointCardsOneHot", start: 97, stop: 104, shape: [7], dtype: "float32" },
      { name: "consecutivePassCountOneHot", start: 104, stop: 110, shape: [6], dtype: "float32" },
      { name: "biddingHistoryActionMask", start: 110, stop: 227, shape: [117], dtype: "float32" },
      { name: "biddingHistoryActionTypeIndicesOneHot", start: 227, stop: 461, shape: [117, 2], dtype: "float32" },
      { name: "biddingHistoryPlayerIndicesOneHot", start: 461, stop: 1046, shape: [117, 5], dtype: "float32" },
      { name: "biddingHistorySuitIndicesOneHot", start: 1046, stop: 1514, shape: [117, 4], dtype: "float32" },
      { name: "biddingHistoryTargetPointCardsOneHot", start: 1514, stop: 2333, shape: [117, 7], dtype: "float32" }
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
      { name: "legalDiscardCardMask", start: 57, stop: 110, shape: [53], dtype: "float32" },
      { name: "calledAdjutantCardMask", start: 110, stop: 163, shape: [53], dtype: "float32" },
      { name: "contractTargetPointCardsOneHot", start: 163, stop: 171, shape: [8], dtype: "float32" },
      { name: "handCountByPlayer", start: 171, stop: 176, shape: [5], dtype: "float32" },
      { name: "specialCardIndicesOneHot", start: 176, stop: 388, shape: [4, 53], dtype: "float32" },
      { name: "biddingHistoryActionMask", start: 388, stop: 505, shape: [117], dtype: "float32" },
      { name: "biddingHistoryActionTypeIndicesOneHot", start: 505, stop: 739, shape: [117, 2], dtype: "float32" },
      { name: "biddingHistoryPlayerIndicesOneHot", start: 739, stop: 1324, shape: [117, 5], dtype: "float32" },
      { name: "biddingHistorySuitIndicesOneHot", start: 1324, stop: 1792, shape: [117, 4], dtype: "float32" },
      { name: "biddingHistoryTargetPointCardsOneHot", start: 1792, stop: 2611, shape: [117, 7], dtype: "float32" }
    ]);
  });

  it("matches Python tensorize_sample model_input bytes for the shared exchange fixture", () => {
    const modelInput = encodeExchangeModelInput(createExchangeFixture().observation);

    expect(modelInput).toHaveLength(EXCHANGE_MODEL_INPUT_FEATURE_COUNT);
    expect(sha256Float32(modelInput)).toBe(pythonExchangeSampleModelInputSha256);
  });

  it("encodes exchange contract target min/max edges as 12..19 one-hot", () => {
    const minSample = createExchangeFixture();
    const minInput = encodeExchangeModelInput(minSample.observation);
    const maxSample = createExchangeFixture();
    const maxInput = encodeExchangeModelInput({
      ...maxSample.observation,
      contractTargetPointCards: MAX_BIDDING_TARGET_POINT_CARDS
    });

    expect(Array.from(modelInputSlice(minInput, "contractTargetPointCardsOneHot", EXCHANGE_MODEL_INPUT_LAYOUT))).toEqual(
      [1, 0, 0, 0, 0, 0, 0, 0]
    );
    expect(Array.from(modelInputSlice(maxInput, "contractTargetPointCardsOneHot", EXCHANGE_MODEL_INPUT_LAYOUT))).toEqual(
      [0, 0, 0, 0, 0, 0, 0, 1]
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
      { name: "contractTargetPointCardsOneHot", start: 110, stop: 118, shape: [8], dtype: "float32" },
      { name: "specialCardIndicesOneHot", start: 118, stop: 330, shape: [4, 53], dtype: "float32" },
      { name: "biddingHistoryActionMask", start: 330, stop: 447, shape: [117], dtype: "float32" },
      { name: "biddingHistoryActionTypeIndicesOneHot", start: 447, stop: 681, shape: [117, 2], dtype: "float32" },
      { name: "biddingHistoryPlayerIndicesOneHot", start: 681, stop: 1266, shape: [117, 5], dtype: "float32" },
      { name: "biddingHistorySuitIndicesOneHot", start: 1266, stop: 1734, shape: [117, 4], dtype: "float32" },
      { name: "biddingHistoryTargetPointCardsOneHot", start: 1734, stop: 2553, shape: [117, 7], dtype: "float32" }
    ]);
  });

  it("matches Python tensorize_sample model_input bytes for the shared adjutant fixture", () => {
    const modelInput = encodeAdjutantModelInput(createAdjutantFixture().observation);

    expect(modelInput).toHaveLength(ADJUTANT_MODEL_INPUT_FEATURE_COUNT);
    expect(sha256Float32(modelInput)).toBe(pythonAdjutantSampleModelInputSha256);
  });

  it("encodes adjutant contract target min/max edges as 12..19 one-hot", () => {
    const minSample = createAdjutantFixture();
    const minInput = encodeAdjutantModelInput(minSample.observation);
    const maxSample = createAdjutantFixture();
    const maxInput = encodeAdjutantModelInput({
      ...maxSample.observation,
      contractTargetPointCards: MAX_BIDDING_TARGET_POINT_CARDS
    });

    expect(Array.from(modelInputSlice(minInput, "contractTargetPointCardsOneHot", ADJUTANT_MODEL_INPUT_LAYOUT))).toEqual(
      [1, 0, 0, 0, 0, 0, 0, 0]
    );
    expect(Array.from(modelInputSlice(maxInput, "contractTargetPointCardsOneHot", ADJUTANT_MODEL_INPUT_LAYOUT))).toEqual(
      [0, 0, 0, 0, 0, 0, 0, 1]
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
  return structuredClone(pythonNonplayingModelInputSamples.exchange);
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
