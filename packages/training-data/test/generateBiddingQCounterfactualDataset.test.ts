import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  BIDDING_ACTION_COUNT,
  BIDDING_MODEL_INPUT_FEATURE_COUNT,
  CARD_COUNT,
  CARD_IDS,
  decodeBiddingAction
} from "@napoleon/ai-observation";
import {
  BIDDING_Q_COUNTERFACTUAL_ACTION_MAPPING_ID,
  BIDDING_Q_COUNTERFACTUAL_ACTION_PLAN_ID,
  BIDDING_Q_COUNTERFACTUAL_DATASET_SAMPLE_TYPE,
  BIDDING_Q_COUNTERFACTUAL_DATASET_SCHEMA_VERSION,
  BIDDING_Q_COUNTERFACTUAL_REWARD_ID,
  BIDDING_Q_COUNTERFACTUAL_REWARD_TYPE,
  BIDDING_Q_COUNTERFACTUAL_REWARD_VERSION,
  BIDDING_Q_COUNTERFACTUAL_SAMPLE_SCHEMA_VERSION,
  calculateBiddingQCounterfactualTerminalReward,
  decodeBiddingQActionIndex,
  encodeBiddingQActionIndex,
  generateBiddingQCounterfactualDataset,
  serializeBiddingQCounterfactualSample,
  validateBiddingQCounterfactualDatasetManifest,
  validateBiddingQCounterfactualSample
} from "../src/index.js";
import type {
  BiddingQCounterfactualDatasetManifest,
  BiddingQCounterfactualPolicy,
  BiddingQCounterfactualSample,
  FixedPlayingPolicy
} from "../src/index.js";

describe("generateBiddingQCounterfactualDataset", () => {
  it("keeps the existing 29-action bidding mapping", () => {
    expect(decodeBiddingQActionIndex(0)).toEqual({ type: "pass" });
    for (let actionIndex = 0; actionIndex < BIDDING_ACTION_COUNT; actionIndex += 1) {
      const qAction = decodeBiddingQActionIndex(actionIndex);
      const existing = decodeBiddingAction(actionIndex, "player-0");
      if (existing.type === "pass") {
        expect(qAction).toEqual({ type: "pass" });
      } else {
        expect(qAction).toEqual({
          type: "bid",
          targetPointCards: existing.targetPointCards,
          suit: existing.suit
        });
        expect(encodeBiddingQActionIndex(existing.targetPointCards, existing.suit)).toBe(actionIndex);
      }
    }
  });

  it("fixes the v1 Q reward fixture", () => {
    expect(calculateBiddingQCounterfactualTerminalReward({
      actingPlayerId: "player-0",
      result: {
        resultType: "standard",
        winner: "napoleon-team",
        napoleonTeamPointCards: 13,
        alliancePointCards: 7,
        targetPointCards: 13,
        napoleonPlayerId: "player-0",
        adjutantPlayerId: "player-1"
      }
    }).reward).toBe(26);
    expect(calculateBiddingQCounterfactualTerminalReward({
      actingPlayerId: "player-0",
      result: {
        resultType: "standard",
        winner: "alliance",
        napoleonTeamPointCards: 12,
        alliancePointCards: 8,
        targetPointCards: 19,
        napoleonPlayerId: "player-0",
        adjutantPlayerId: "player-1"
      }
    }).reward).toBe(-1);
    expect(calculateBiddingQCounterfactualTerminalReward({
      actingPlayerId: "player-0",
      result: {
        resultType: "standard",
        winner: "napoleon-team",
        napoleonTeamPointCards: 19,
        alliancePointCards: 1,
        targetPointCards: 19,
        napoleonPlayerId: "player-0",
        adjutantPlayerId: "player-0"
      }
    }).reward).toBe(57);
    expect(calculateBiddingQCounterfactualTerminalReward({
      actingPlayerId: "player-2",
      result: {
        resultType: "standard",
        winner: "alliance",
        napoleonTeamPointCards: 12,
        alliancePointCards: 8,
        targetPointCards: 13,
        napoleonPlayerId: "player-0",
        adjutantPlayerId: "player-1"
      }
    }).reward).toBe(0);
    expect(calculateBiddingQCounterfactualTerminalReward({
      actingPlayerId: "player-2",
      result: {
        resultType: "all-pass",
        starterPlayerId: "player-0",
        payoffs: []
      }
    }).reward).toBe(0);
  });

  it("generates deterministic raw counterfactual samples with coverage summary", async () => {
    await withTempDir(async (directory) => {
      const artifacts = await createArtifactFiles(directory);
      const biddingPolicy = createBiddingPolicy();
      const playingPolicy = createPlayingPolicy();
      const firstOutput = join(directory, "first");
      const secondOutput = join(directory, "second");
      const options = {
        outputDirectory: firstOutput,
        biddingPolicy,
        biddingPolicyArtifact: artifacts.bidding,
        playingPolicy,
        playingPolicyArtifact: artifacts.playing,
        startSeed: 10,
        logicalSeedCount: 4,
        maxSourceStates: 8,
        repeats: 2,
        gamesPerShard: 20,
        randomSeed: 368,
        randomLegalBidCount: 2,
        sourceCommit: "test-commit"
      };

      const result = await generateBiddingQCounterfactualDataset(options);
      await generateBiddingQCounterfactualDataset({
        ...options,
        outputDirectory: secondOutput
      });
      await expectDirectoriesToBeByteIdentical(firstOutput, secondOutput);

      const manifest = await readManifest(firstOutput);
      const samples = (await readAllSamples(firstOutput, manifest)).map((line) =>
        JSON.parse(line) as BiddingQCounterfactualSample
      );
      validateBiddingQCounterfactualDatasetManifest(manifest);
      expect(result.manifest).toEqual(manifest);
      expect(manifest.datasetSchemaVersion).toBe(BIDDING_Q_COUNTERFACTUAL_DATASET_SCHEMA_VERSION);
      expect(manifest.sampleType).toBe(BIDDING_Q_COUNTERFACTUAL_DATASET_SAMPLE_TYPE);
      expect(manifest.sampleSchemaVersion).toBe(BIDDING_Q_COUNTERFACTUAL_SAMPLE_SCHEMA_VERSION);
      expect(manifest.actionMapping.id).toBe(BIDDING_Q_COUNTERFACTUAL_ACTION_MAPPING_ID);
      expect(manifest.actionMapping.actionCount).toBe(BIDDING_ACTION_COUNT);
      expect(manifest.actionPlan.id).toBe(BIDDING_Q_COUNTERFACTUAL_ACTION_PLAN_ID);
      expect(manifest.reward).toMatchObject({
        id: BIDDING_Q_COUNTERFACTUAL_REWARD_ID,
        type: BIDDING_Q_COUNTERFACTUAL_REWARD_TYPE,
        version: BIDDING_Q_COUNTERFACTUAL_REWARD_VERSION,
        contractLossReward: -1,
        nonContractReward: 0
      });
      expect(manifest.compactObservation.modelInputFeatureCount)
        .toBe(BIDDING_MODEL_INPUT_FEATURE_COUNT);
      expect(manifest.repeats).toBe(2);
      expect(manifest.sourceStates).toBe(8);
      expect(manifest.sampleCount).toBe(samples.length);
      expect(manifest.summary.totalSourceStates).toBe(8);
      expect(manifest.summary.totalForcedStateActionPairs * 2).toBe(samples.length);
      expect(manifest.summary.fallbackCount).toBe(0);
      expect(manifest.summary.illegalActionCount).toBe(0);
      expect(manifest.summary.passCount).toBeGreaterThan(0);
      for (const suit of ["spades", "hearts", "diamonds", "clubs"] as const) {
        expect(manifest.summary.suitCounts[suit]).toBeGreaterThan(0);
      }
      expect(manifest.summary.targetCounts["13"]).toBeGreaterThan(0);
      expect(
        Object.entries(manifest.summary.targetCounts)
          .filter(([target]) => target !== "13")
          .some(([, count]) => count > 0)
      ).toBe(true);
      expect(manifest.summary.actionIndexCounts["0"]).toBe(manifest.summary.passCount);
      expect(Object.values(manifest.summary.strongestSuitCounts).reduce((a, b) => a + b, 0))
        .toBe(manifest.sourceStates);
      expect(samples.every((sample) => sample.provenance.forcedOnce)).toBe(true);
      expect(samples.every((sample) => sample.modelInput.length === BIDDING_MODEL_INPUT_FEATURE_COUNT))
        .toBe(true);
      expect(samples.every((sample) => sample.legalBidMask.length === BIDDING_ACTION_COUNT))
        .toBe(true);
      expect(new Set(samples.map((sample) => sample.stateKey)).size).toBeGreaterThan(0);
      expect(
        samples.some((left) =>
          samples.some((right) =>
            left.stateKey === right.stateKey &&
            left.forcedActionIndex !== right.forcedActionIndex
          )
        )
      ).toBe(true);
      const repeatGroups = new Map<string, number[]>();
      for (const sample of samples) {
        validateBiddingQCounterfactualSample(sample);
        const key = `${sample.stateKey}:${sample.forcedActionIndex}`;
        repeatGroups.set(key, [...(repeatGroups.get(key) ?? []), sample.repeatIndex]);
        expect(sample.rawTerminalReward).toBe(sample.terminalReward);
        expect(sample.provenance.replayMatchedModelInput).toBe(true);
        expect(sample.provenance.replayMatchedLegalBidMask).toBe(true);
        expect(serializeBiddingQCounterfactualSample(sample)).toContain(sample.stateKey);
      }
      expect([...repeatGroups.values()].every((indexes) => indexes.sort().join(",") === "0,1"))
        .toBe(true);
    });
  });

  it("rejects illegal forced action samples and provenance mismatches", () => {
    const sample = createValidSample();
    expect(() => validateBiddingQCounterfactualSample({
      ...sample,
      forcedActionIndex: 1,
      legalBidMask: Array(BIDDING_ACTION_COUNT).fill(0).map((_, index) => index === 0 ? 1 : 0)
    })).toThrow("forced action must be legal");
    expect(() => validateBiddingQCounterfactualSample({
      ...sample,
      provenance: {
        ...sample.provenance,
        replayMatchedModelInput: false as true
      }
    })).toThrow("replay parity");
    const manifest = createValidManifest(sample);
    validateBiddingQCounterfactualDatasetManifest(manifest);
    expect(() => validateBiddingQCounterfactualDatasetManifest({
      ...manifest,
      reward: {
        ...manifest.reward,
        id: "wrong-reward" as typeof BIDDING_Q_COUNTERFACTUAL_REWARD_ID
      }
    })).toThrow("reward mismatch");
  });

  it("fails closed when output directory already exists", async () => {
    await withTempDir(async (directory) => {
      const artifacts = await createArtifactFiles(directory);
      const outputDirectory = join(directory, "existing");
      await mkdir(outputDirectory);
      await expect(generateBiddingQCounterfactualDataset({
        outputDirectory,
        biddingPolicy: createBiddingPolicy(),
        biddingPolicyArtifact: artifacts.bidding,
        playingPolicy: createPlayingPolicy(),
        playingPolicyArtifact: artifacts.playing,
        startSeed: 1,
        logicalSeedCount: 1,
        maxSourceStates: 1,
        repeats: 1,
        randomSeed: 1
      })).rejects.toThrow("Output directory already exists");
    });
  });
});

function createBiddingPolicy(): BiddingQCounterfactualPolicy {
  return {
    metadata: { fixture: "bidding" },
    runtime: {
      requestedInferenceDevice: "cpu",
      resolvedInferenceDevice: "cpu",
      executionProvider: "cpu"
    },
    async predictLogits() {
      const logits = new Float32Array(BIDDING_ACTION_COUNT);
      logits[0] = -1;
      for (let index = 1; index < logits.length; index += 1) {
        logits[index] = 1 / index;
      }
      return logits;
    }
  };
}

function createPlayingPolicy(): FixedPlayingPolicy {
  return {
    metadata: { fixture: "playing" },
    runtime: {
      requestedInferenceDevice: "cpu",
      resolvedInferenceDevice: "cpu",
      executionProvider: "cpu"
    },
    async predictLogits() {
      return new Float32Array(CARD_COUNT);
    }
  };
}

async function createArtifactFiles(directory: string): Promise<{
  bidding: { onnxPath: string; metadataPath: string; artifactId: string };
  playing: { onnxPath: string; metadataPath: string; artifactId: string };
}> {
  const biddingDir = join(directory, "bidding-artifact");
  const playingDir = join(directory, "playing-artifact");
  await mkdir(biddingDir);
  await mkdir(playingDir);
  const bidding = {
    onnxPath: join(biddingDir, "policy.onnx"),
    metadataPath: join(biddingDir, "policy.json"),
    artifactId: "test-bidding"
  };
  const playing = {
    onnxPath: join(playingDir, "policy.onnx"),
    metadataPath: join(playingDir, "policy.json"),
    artifactId: "test-playing"
  };
  await writeFile(bidding.onnxPath, "bidding");
  await writeFile(bidding.metadataPath, JSON.stringify({ policyType: "bidding" }));
  await writeFile(playing.onnxPath, "playing");
  await writeFile(playing.metadataPath, JSON.stringify({ policyType: "playing" }));
  return { bidding, playing };
}

function createValidSample(): BiddingQCounterfactualSample {
  const legalBidMask = Array(BIDDING_ACTION_COUNT).fill(0);
  legalBidMask[0] = 1;
  return {
    sampleType: BIDDING_Q_COUNTERFACTUAL_DATASET_SAMPLE_TYPE,
    schemaVersion: BIDDING_Q_COUNTERFACTUAL_SAMPLE_SCHEMA_VERSION,
    stateKey: "state",
    sourceSeed: 1,
    sourceGameSeed: 1,
    candidateSeatIndex: 0,
    actingPlayerId: "player-0",
    actingPlayerIndex: 0,
    biddingStep: 1,
    sourceSelectedActionIndex: 0,
    sourceSelectedAction: { type: "pass" },
    modelInput: Array(BIDDING_MODEL_INPUT_FEATURE_COUNT).fill(0),
    legalBidMask,
    forcedActionIndex: 0,
    forcedAction: { type: "pass" },
    strongestSuit: "spades",
    strongestSuitScore: 200,
    actionPlanId: BIDDING_Q_COUNTERFACTUAL_ACTION_PLAN_ID,
    repeatIndex: 0,
    rolloutSeed: 1,
    terminalReward: 0,
    rawTerminalReward: 0,
    terminalRole: "all-pass-starter",
    contractSuccess: false,
    resultType: "all-pass",
    result: { starterPlayerId: "player-0" },
    provenance: {
      sourceStateKey: "state",
      sourceSeed: 1,
      sourceGameSeed: 1,
      sourceBiddingStep: 1,
      replayMatchedModelInput: true,
      replayMatchedLegalBidMask: true,
      forcedOnce: true
    }
  };
}

function createValidManifest(
  sample: BiddingQCounterfactualSample
): BiddingQCounterfactualDatasetManifest {
  const summary = {
    totalSourceStates: 1,
    totalForcedStateActionPairs: 1,
    totalRolloutSamples: 1,
    fallbackCount: 0,
    illegalActionCount: 0,
    passCount: 1,
    bidCount: 0,
    suitCounts: { spades: 0, hearts: 0, diamonds: 0, clubs: 0 },
    targetCounts: { "13": 0, "14": 0, "15": 0, "16": 0, "17": 0, "18": 0, "19": 0 },
    actionIndexCounts: Object.fromEntries(Array.from({ length: BIDDING_ACTION_COUNT }, (_, index) => [
      String(index),
      index === 0 ? 1 : 0
    ])),
    strongestSuitCounts: { spades: 1, hearts: 0, diamonds: 0, clubs: 0 },
    strongestByForcedSuitCounts: {
      spades: { spades: 0, hearts: 0, diamonds: 0, clubs: 0 },
      hearts: { spades: 0, hearts: 0, diamonds: 0, clubs: 0 },
      diamonds: { spades: 0, hearts: 0, diamonds: 0, clubs: 0 },
      clubs: { spades: 0, hearts: 0, diamonds: 0, clubs: 0 }
    },
    legalButNeverSampledActionCount: 0,
    legalButNeverSampledActionIndexes: [],
    terminalReward: { mean: 0, std: 0, min: 0, max: 0 },
    contractSuccessBySuit: {
      spades: { count: 0, successCount: 0, successRate: null },
      hearts: { count: 0, successCount: 0, successRate: null },
      diamonds: { count: 0, successCount: 0, successRate: null },
      clubs: { count: 0, successCount: 0, successRate: null }
    },
    contractSuccessByTarget: {
      "13": { count: 0, successCount: 0, successRate: null },
      "14": { count: 0, successCount: 0, successRate: null },
      "15": { count: 0, successCount: 0, successRate: null },
      "16": { count: 0, successCount: 0, successRate: null },
      "17": { count: 0, successCount: 0, successRate: null },
      "18": { count: 0, successCount: 0, successRate: null },
      "19": { count: 0, successCount: 0, successRate: null }
    },
    resultTypeCounts: { standard: 0, "all-pass": 1 },
    terminalRoleCounts: {
      napoleon: 0,
      adjutant: 0,
      citizen: 0,
      "napoleon-adjutant": 0,
      "all-pass-starter": 1,
      "all-pass-other": 0
    }
  };
  return {
    datasetSchemaVersion: BIDDING_Q_COUNTERFACTUAL_DATASET_SCHEMA_VERSION,
    generatorVersion: 1,
    format: "jsonl",
    sampleType: BIDDING_Q_COUNTERFACTUAL_DATASET_SAMPLE_TYPE,
    sampleSchemaVersion: BIDDING_Q_COUNTERFACTUAL_SAMPLE_SCHEMA_VERSION,
    compactObservation: {
      phase: "bidding",
      encoderSchemaVersion: 1,
      modelInputSchemaVersion: 2,
      modelInputFeatureCount: BIDDING_MODEL_INPUT_FEATURE_COUNT
    },
    actionMapping: {
      id: BIDDING_Q_COUNTERFACTUAL_ACTION_MAPPING_ID,
      actionCount: BIDDING_ACTION_COUNT,
      passActionIndex: 0,
      bidTargets: [13, 14, 15, 16, 17, 18, 19],
      suitOrder: ["spades", "hearts", "diamonds", "clubs"]
    },
    reward: {
      id: BIDDING_Q_COUNTERFACTUAL_REWARD_ID,
      type: BIDDING_Q_COUNTERFACTUAL_REWARD_TYPE,
      version: BIDDING_Q_COUNTERFACTUAL_REWARD_VERSION,
      napoleonWinMultiplier: 2,
      napoleonAdjutantWinMultiplier: 3,
      contractLossReward: -1,
      nonContractReward: 0
    },
    terminalRewardTransform: {
      id: "bidding-q-contract-result-loss-minus-one-identity-v1",
      type: "identity",
      version: 1,
      formula: "terminal_reward = raw_bidding_q_reward"
    },
    actionPlan: {
      id: BIDDING_Q_COUNTERFACTUAL_ACTION_PLAN_ID,
      version: 1,
      randomLegalBidCount: 2
    },
    repeats: 1,
    sourceStates: 1,
    forcedStateActionPairs: 1,
    sampleCount: 1,
    startSeed: 1,
    endSeed: 1,
    logicalSeedCount: 1,
    actualSourceGameCount: 5,
    candidateSeatRotation: [0, 1, 2, 3, 4],
    gamesPerShard: 1,
    shardCount: 1,
    playerCount: 5,
    cardCount: CARD_COUNT,
    cardIds: CARD_IDS,
    cardIdsSha256: "7ea0fdb58078f835bc5f7e6307a2a0c869430db343dd9e50ed1226f0452aaf38",
    simulation: { backend: "typescript" },
    opponentMix: {
      type: "mixed-frozen-bidding",
      mixingRuleVersion: "per-seat-seeded-conservative-all-pass-50-50-v1",
      selectionUnit: "game-seat",
      conservativeWeight: 0.5,
      allPassWeight: 0.5,
      policies: {
        conservative: { type: "conservative-bidding", id: "conservative-bidding-v1" },
        allPass: { type: "all-pass-bidding", id: "all-pass-bidding-v1" }
      }
    },
    behaviorPolicy: {
      type: "bidding-onnx",
      artifactId: "bidding",
      onnxFileName: "policy.onnx",
      metadataFileName: "policy.json",
      onnxSha256: "0".repeat(64),
      metadataSha256: "0".repeat(64),
      metadata: {}
    },
    fixedPlayingPolicy: {
      type: "playing-onnx",
      artifactId: "playing",
      onnxFileName: "policy.onnx",
      metadataFileName: "policy.json",
      onnxSha256: "0".repeat(64),
      metadataSha256: "0".repeat(64),
      metadata: {}
    },
    sourceCommit: null,
    summary,
    shards: [{
      file: "shard-00000.jsonl",
      startSeed: sample.sourceSeed,
      endSeed: sample.sourceSeed,
      gameCount: 1,
      sampleCount: 1,
      byteLength: 1,
      sha256: "0".repeat(64)
    }]
  };
}

async function withTempDir(run: (directory: string) => Promise<void>): Promise<void> {
  const directory = await mkdtemp(join(tmpdir(), "napoleon-bidding-q-test-"));
  try {
    await run(directory);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}

async function readManifest(outputDirectory: string): Promise<BiddingQCounterfactualDatasetManifest> {
  return JSON.parse(await readFile(join(outputDirectory, "manifest.json"), "utf8")) as BiddingQCounterfactualDatasetManifest;
}

async function readAllSamples(
  outputDirectory: string,
  manifest: BiddingQCounterfactualDatasetManifest
): Promise<readonly string[]> {
  const chunks = await Promise.all(
    manifest.shards.map((shard) => readFile(join(outputDirectory, shard.file), "utf8"))
  );
  return chunks.flatMap((chunk) => chunk.split("\n").filter(Boolean));
}

async function expectDirectoriesToBeByteIdentical(first: string, second: string): Promise<void> {
  const firstFiles = (await readdir(first)).sort();
  const secondFiles = (await readdir(second)).sort();
  expect(firstFiles).toEqual(secondFiles);
  for (const file of firstFiles) {
    expect(await readFile(join(second, file), "utf8")).toBe(await readFile(join(first, file), "utf8"));
  }
}
