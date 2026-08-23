import { mkdir, mkdtemp, readdir, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { createDeck } from "@napoleon/game-core";
import { CARD_IDS, EXCHANGE_MODEL_INPUT_FEATURE_COUNT } from "@napoleon/ai-observation";
import {
  EXCHANGE_COUNTERFACTUAL_COMBINATION_COUNT,
  EXCHANGE_COUNTERFACTUAL_DATASET_SAMPLE_TYPE,
  enumerateExchangeDiscardCombinations,
  generateExchangeCounterfactualDataset,
  validateGenerateExchangeCounterfactualDatasetOptions
} from "../src/index.js";
import type { ExchangeCounterfactualSample } from "../src/index.js";

describe("generateExchangeCounterfactualDataset", () => {
  it("enumerates unordered 13C3 exchange discard combinations once", () => {
    const hand = [
      ...createDeck().slice(10, 13),
      ...createDeck().slice(0, 10)
    ];
    const combinations = enumerateExchangeDiscardCombinations(hand);

    expect(combinations).toHaveLength(EXCHANGE_COUNTERFACTUAL_COMBINATION_COUNT);
    expect(new Set(combinations.map((combo) => combo.join("|"))).size).toBe(
      EXCHANGE_COUNTERFACTUAL_COMBINATION_COUNT
    );
    expect(combinations.every((combo) => combo.length === 3)).toBe(true);
    expect(combinations.every((combo) => isCanonical(combo))).toBe(true);
    const reversedFirst = [...combinations[0]].reverse().join("|");
    expect(combinations.some((combo) => combo.join("|") === reversedFirst)).toBe(false);
  });

  it("generates real-flow exchange states with 286 counterfactual rollout samples", async () => {
    await withTempDir(async (directory) => {
      const outputDirectory = join(directory, "nested", "dataset");
      const result = await generateExchangeCounterfactualDataset({
        outputDirectory,
        sourceStateCount: 1,
        startSeed: 434000,
        statesPerShard: 1,
        maxDealAttempts: 50
      });

      expect(result.manifest.sampleType).toBe(EXCHANGE_COUNTERFACTUAL_DATASET_SAMPLE_TYPE);
      expect(result.manifest.sourceFlow).toEqual([
        "initial deal",
        "bidding",
        "contract established",
        "adjutant selection (rule-based-adjutant-v1)",
        "kitty 3-card pickup",
        "exchanging"
      ]);
      expect(result.manifest.discardCombinationCount).toBe(EXCHANGE_COUNTERFACTUAL_COMBINATION_COUNT);
      expect(result.manifest.sampleCount).toBe(EXCHANGE_COUNTERFACTUAL_COMBINATION_COUNT);
      expect(result.manifest.rolloutCount).toBe(EXCHANGE_COUNTERFACTUAL_COMBINATION_COUNT);
      expect(result.manifest.modelInput.featureCount).toBe(EXCHANGE_MODEL_INPUT_FEATURE_COUNT);
      expect(result.manifest.modelInput.hiddenOpponentHandsIncluded).toBe(false);
      expect(result.manifest.permutationActionsIncluded).toBe(false);
      expect(result.manifest.summary.invariantFailureCount).toBe(0);
      expect(result.manifest.summary.candidateCountPerState.min).toBe(
        EXCHANGE_COUNTERFACTUAL_COMBINATION_COUNT
      );
      expect(result.manifest.summary.candidateCountPerState.max).toBe(
        EXCHANGE_COUNTERFACTUAL_COMBINATION_COUNT
      );

      const manifest = JSON.parse(await readFile(join(outputDirectory, "manifest.json"), "utf8")) as typeof result.manifest;
      const rows = await readSamples(join(outputDirectory, manifest.shards[0].file));
      expect(rows).toHaveLength(EXCHANGE_COUNTERFACTUAL_COMBINATION_COUNT);
      expect(new Set(rows.map((row) => row.candidateIndex)).size).toBe(
        EXCHANGE_COUNTERFACTUAL_COMBINATION_COUNT
      );
      expect(new Set(rows.map((row) => row.candidateDiscardCardIds.join("|"))).size).toBe(
        EXCHANGE_COUNTERFACTUAL_COMBINATION_COUNT
      );
      expect(new Set(rows.map((row) => row.sourceStateKey)).size).toBe(1);
      expect(new Set(rows.map((row) => row.hiddenDealChecksum)).size).toBe(1);
      expect(rows.filter((row) => row.isRuleBasedAction)).toHaveLength(1);

      for (const row of rows) {
        expect(row.modelInput).toHaveLength(EXCHANGE_MODEL_INPUT_FEATURE_COUNT);
        expect(row.legalDiscardCardMask).toHaveLength(CARD_IDS.length);
        expect(sum(row.legalDiscardCardMask)).toBe(13);
        expect(row.pickupHandCardIds).toHaveLength(13);
        expect(row.candidateDiscardCardIds).toHaveLength(3);
        expect(new Set(row.candidateDiscardCardIds).size).toBe(3);
        expect(isCanonical(row.candidateDiscardCardIds)).toBe(true);
        expect(row.candidateDiscardMask).toHaveLength(CARD_IDS.length);
        expect(sum(row.candidateDiscardMask)).toBe(3);
        expect(row.candidateDiscardCardIds.every((cardId) =>
          row.pickupHandCardIds.includes(cardId)
        )).toBe(true);
        expect(Object.values(row.invariantChecks).every(Boolean)).toBe(true);
      }
    });
  }, 60_000);

  it("cleans same-filesystem staging data when generation fails", async () => {
    await withTempDir(async (directory) => {
      const outputDirectory = join(directory, "nested", "failed-dataset");
      await expect(generateExchangeCounterfactualDataset({
        outputDirectory,
        sourceStateCount: 1,
        startSeed: 434300,
        statesPerShard: 1,
        maxDealAttempts: 50,
        createPlayingAgent: () => ({
          selectAction: async () => {
            throw new Error("intentional rollout failure");
          }
        })
      })).rejects.toThrow("intentional rollout failure");

      await expect(readFile(join(outputDirectory, "manifest.json"), "utf8")).rejects.toThrow();
      await expect(readdir(join(directory, "nested"))).resolves.toEqual([]);
    });
  }, 60_000);

  it("records custom policy metadata as unknown unless explicitly supplied", () => {
    const validated = validateGenerateExchangeCounterfactualDatasetOptions({
      outputDirectory: "/tmp/custom-playing",
      sourceStateCount: 1,
      startSeed: 1,
      createBiddingAgent: () => ({
        selectAction: async () => {
          throw new Error("not used");
        }
      }),
      createAdjutantAgent: () => ({
        selectAction: async () => {
          throw new Error("not used");
        }
      }),
      createPlayingAgent: () => ({
        selectAction: async () => {
          throw new Error("not used");
        }
      })
    });

    expect(validated.biddingPolicy.id).toBe("custom-source-bidding-policy-unknown-v1");
    expect(validated.adjutantPolicy.id).toBe("custom-adjutant-policy-unknown-v1");
    expect(validated.playingPolicy.id).toBe("custom-playing-rollout-policy-unknown-v1");
  });

  it("rejects deal seed ranges that exceed uint32", () => {
    expect(() => validateGenerateExchangeCounterfactualDatasetOptions({
      outputDirectory: "/tmp/seed-overflow",
      sourceStateCount: 2,
      startSeed: 0xffffffff,
      maxDealAttempts: 2
    })).toThrow("uint32");
  });

  it("rejects an existing output directory before generating rollouts", async () => {
    await withTempDir(async (directory) => {
      const outputDirectory = join(directory, "existing");
      await mkdir(outputDirectory);

      await expect(generateExchangeCounterfactualDataset({
        outputDirectory,
        sourceStateCount: 1,
        startSeed: 434500,
        maxDealAttempts: 50
      })).rejects.toThrow("outputDirectory already exists");
    });
  });
});

async function readSamples(path: string): Promise<ExchangeCounterfactualSample[]> {
  const body = await readFile(path, "utf8");
  return body.trim().split("\n").filter(Boolean).map((line) =>
    JSON.parse(line) as ExchangeCounterfactualSample
  );
}

function isCanonical(cardIds: readonly string[]): boolean {
  const sorted = [...cardIds].sort((left, right) => CARD_IDS.indexOf(left) - CARD_IDS.indexOf(right));
  return cardIds.every((cardId, index) => cardId === sorted[index]);
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

async function withTempDir(callback: (directory: string) => Promise<void>): Promise<void> {
  const directory = await mkdtemp(join(tmpdir(), "napoleon-exchange-cf-test-"));
  try {
    await callback(directory);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}
