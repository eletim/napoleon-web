import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  BIDDING_ACTION_COUNT,
  BIDDING_MODEL_INPUT_FEATURE_COUNT,
  CARD_COUNT,
  MODEL_INPUT_FEATURE_COUNT
} from "@napoleon/ai-observation";
import {
  ONNX_INPUT_NAME,
  ONNX_OPSET_VERSION,
  ONNX_OUTPUT_NAME,
  calculateCardIdsSha256,
  loadNonPlayingPolicyOnnxModel,
  loadPolicyOnnxModel
} from "../../policy-onnx/src/index.js";
import type { NonPlayingPolicyOnnxMetadata, NonPlayingPolicyType } from "../../policy-onnx/src/index.js";
import { createConstantPolicyOnnx } from "../../policy-onnx/test/testOnnxFixture.js";
import {
  DATASET_FORMAT,
  NON_PLAYING_RL_DATASET_GENERATOR_VERSION,
  NON_PLAYING_RL_DATASET_SAMPLE_TYPE,
  NON_PLAYING_RL_PHASE_SCOPE,
  NON_PLAYING_RL_REWARD_ID,
  NON_PLAYING_RL_REWARD_TYPE,
  NON_PLAYING_RL_REWARD_VERSION,
  NON_PLAYING_RL_SAMPLING_ALGORITHM,
  calculateNonPlayingBiddingLogProbability,
  calculateNonPlayingTerminalRoleReward,
  generateNonPlayingBiddingRlDataset,
  validateNonPlayingBiddingRlSample,
  validateNonPlayingRlDatasetManifest
} from "../src/index.js";
import type {
  NonPlayingBiddingRlOutcome,
  NonPlayingBiddingRlSample,
  NonPlayingRlDatasetManifest
} from "../src/index.js";

describe("generateNonPlayingBiddingRlDataset", () => {
  it("generates deterministic bidding RL samples with role rewards and artifact metadata", async () => {
    await withTempDir(async (directory) => {
      const playingArtifact = await createPlayingPolicyFixture(join(directory, "playing"));
      const biddingArtifact = await createBiddingPolicyFixture(join(directory, "bidding"));
      const playingPolicy = await loadPolicyOnnxModel(playingArtifact);
      const biddingPolicy = await loadNonPlayingPolicyOnnxModel(biddingArtifact);
      const firstOutput = join(directory, "first");
      const secondOutput = join(directory, "second");
      const progressEvents: Array<{ completedGames: number; sampleCount: number; currentSeed: number }> = [];
      const options = {
        outputDirectory: firstOutput,
        biddingPolicy,
        biddingPolicyArtifact: biddingArtifact,
        playingPolicy,
        playingPolicyArtifact: playingArtifact,
        startSeed: 7,
        gameCount: 2,
        gamesPerShard: 1,
        temperature: 0.01,
        onProgress: (progress: {
          completedGames: number;
          sampleCount: number;
          currentSeed: number;
        }) => {
          progressEvents.push({
            completedGames: progress.completedGames,
            sampleCount: progress.sampleCount,
            currentSeed: progress.currentSeed
          });
        }
      };

      const result = await generateNonPlayingBiddingRlDataset(options);
      await generateNonPlayingBiddingRlDataset({
        ...options,
        outputDirectory: secondOutput,
        onProgress: undefined
      });
      await expectDirectoriesToBeByteIdentical(firstOutput, secondOutput);

      const manifest = await readManifest(firstOutput);
      const lines = await readAllShardLines(firstOutput, manifest);
      const samples = lines.map((line) => JSON.parse(line) as NonPlayingBiddingRlSample);

      expect(result.manifest).toEqual(manifest);
      validateNonPlayingRlDatasetManifest(manifest);
      expect(manifest.datasetSchemaVersion).toBe(1);
      expect(manifest.generatorVersion).toBe(NON_PLAYING_RL_DATASET_GENERATOR_VERSION);
      expect(manifest.format).toBe(DATASET_FORMAT);
      expect(manifest.sampleType).toBe(NON_PLAYING_RL_DATASET_SAMPLE_TYPE);
      expect(manifest.phaseScope).toBe(NON_PLAYING_RL_PHASE_SCOPE);
      expect(manifest.learnedPhases).toEqual(["bidding"]);
      expect(manifest.ruleBasedPhases).toEqual(["choosing-adjutant", "exchanging"]);
      expect(manifest.fixedPhases).toEqual(["playing"]);
      expect(manifest.samplingAlgorithm).toBe(NON_PLAYING_RL_SAMPLING_ALGORITHM);
      expect(manifest.temperature).toBe(0.01);
      expect(manifest.reward).toEqual({
        type: NON_PLAYING_RL_REWARD_TYPE,
        version: NON_PLAYING_RL_REWARD_VERSION,
        id: NON_PLAYING_RL_REWARD_ID
      });
      expect(manifest.behaviorPolicy).toMatchObject({
        type: "bidding-onnx",
        artifactId: "test-bidding-policy",
        requestedInferenceDevice: "cpu",
        resolvedInferenceDevice: "cpu",
        executionProvider: "cpu"
      });
      expect(manifest.fixedPlayingPolicy).toMatchObject({
        type: "playing-onnx",
        artifactId: "test-playing-policy",
        requestedInferenceDevice: "cpu",
        resolvedInferenceDevice: "cpu",
        executionProvider: "cpu"
      });
      expect(manifest.behaviorPolicy.onnxSha256).toBe(await sha256File(biddingArtifact.onnxPath));
      expect(manifest.fixedPlayingPolicy.onnxSha256).toBe(await sha256File(playingArtifact.onnxPath));
      expect(manifest.cardIdsSha256).toBe(calculateCardIdsSha256());
      expect(manifest.biddingModelInputFeatureCount).toBe(BIDDING_MODEL_INPUT_FEATURE_COUNT);
      expect(manifest.playingModelInputFeatureCount).toBe(MODEL_INPUT_FEATURE_COUNT);
      expect(manifest.actionCount).toBe(BIDDING_ACTION_COUNT);
      expect(manifest.sampleCount).toBe(lines.length);
      expect(manifest.sampleCount).toBeGreaterThan(0);
      expect(progressEvents.map((event) => event.completedGames)).toEqual([1, 2]);
      expect(progressEvents.map((event) => event.currentSeed)).toEqual([7, 8]);
      expect(progressEvents.at(-1)?.sampleCount).toBe(manifest.sampleCount);

      const playerDecisionCounts = new Map<string, number>();
      for (const sample of samples) {
        validateNonPlayingBiddingRlSample(sample, sample.seed);
        expect(sample.phase).toBe("bidding");
        expect(sample.modelInput).toHaveLength(BIDDING_MODEL_INPUT_FEATURE_COUNT);
        expect(sample.legalBidMask).toHaveLength(BIDDING_ACTION_COUNT);
        expect(sample.legalBidMask[sample.selectedActionIndex]).toBe(1);
        expect(sample.terminalReward).toBe(calculateNonPlayingTerminalRoleReward(sample.outcome));
        expect(sample.outcome.targetPointCards).toBeGreaterThanOrEqual(12);
        expect(sample.outcome.targetPointCards).toBeLessThanOrEqual(19);

        const logits = await biddingPolicy.predictLogits(sample.modelInput);
        const recomputedLogProbability = calculateNonPlayingBiddingLogProbability({
          logits,
          legalBidMask: sample.legalBidMask,
          selectedActionIndex: sample.selectedActionIndex,
          temperature: manifest.temperature
        });

        expect(sample.behaviorLogProbability).toBeCloseTo(recomputedLogProbability, 5);
        if (sum(sample.legalBidMask) === 1) {
          expect(sample.behaviorLogProbability).toBe(0);
        } else {
          expect(sample.behaviorLogProbability).toBeLessThanOrEqual(0);
        }

        const key = `${sample.seed}:${sample.actingPlayerId}`;
        playerDecisionCounts.set(key, (playerDecisionCounts.get(key) ?? 0) + 1);
      }

      expect([...playerDecisionCounts.values()].some((count) => count > 1)).toBe(true);
      expect(samples.some((sample) => sample.selectedActionIndex === 0)).toBe(true);
      expect(samples.some((sample) => sample.selectedActionIndex > 0)).toBe(true);

      for (const shard of manifest.shards) {
        const shardPath = join(firstOutput, shard.file);
        const file = await readFile(shardPath, "utf8");
        const fileStat = await stat(shardPath);

        expect(file.split("\n").filter(Boolean)).toHaveLength(shard.sampleCount);
        expect(fileStat.size).toBe(shard.byteLength);
        expect(sha256Utf8(file)).toBe(shard.sha256);
      }

      assertNoCompleteStateFields(lines);
    });
  });

  it("calculates terminal role rewards from the v1 reward table", () => {
    expectReward({ winner: "napoleon-team", targetPointCards: 18, actingPlayerRole: "napoleon" }, 18);
    expectReward({ winner: "napoleon-team", targetPointCards: 18, actingPlayerRole: "adjutant" }, 11);
    expectReward({ winner: "napoleon-team", targetPointCards: 18, actingPlayerRole: "citizen" }, 7);
    expectReward({
      winner: "napoleon-team",
      targetPointCards: 18,
      actingPlayerRole: "napoleon-adjutant"
    }, 29);
    expectReward({ winner: "alliance", targetPointCards: 18, actingPlayerRole: "napoleon" }, -3);
    expectReward({ winner: "alliance", targetPointCards: 18, actingPlayerRole: "adjutant" }, 0);
    expectReward({ winner: "alliance", targetPointCards: 18, actingPlayerRole: "citizen" }, 0);
    expectReward({
      winner: "alliance",
      targetPointCards: 18,
      actingPlayerRole: "napoleon-adjutant"
    }, -3);
  });

  it("rejects an existing output directory before writing non-playing RL data", async () => {
    await withTempDir(async (directory) => {
      const playingArtifact = await createPlayingPolicyFixture(join(directory, "playing"));
      const biddingArtifact = await createBiddingPolicyFixture(join(directory, "bidding"));
      const playingPolicy = await loadPolicyOnnxModel(playingArtifact);
      const biddingPolicy = await loadNonPlayingPolicyOnnxModel(biddingArtifact);
      const output = join(directory, "existing");
      const marker = join(output, "marker.txt");
      await mkdir(output);
      await writeFile(marker, "keep\n", "utf8");

      await expect(generateNonPlayingBiddingRlDataset({
        outputDirectory: output,
        biddingPolicy,
        biddingPolicyArtifact: biddingArtifact,
        playingPolicy,
        playingPolicyArtifact: playingArtifact,
        startSeed: 0,
        gameCount: 1,
        gamesPerShard: 1
      })).rejects.toThrow("Output directory already exists");

      expect(await readFile(marker, "utf8")).toBe("keep\n");
    });
  });
});

async function createPlayingPolicyFixture(directory: string): Promise<{
  onnxPath: string;
  metadataPath: string;
  artifactId: string;
}> {
  await mkdir(directory, { recursive: true });
  const onnxPath = join(directory, "playing-policy.onnx");
  const metadataPath = join(directory, "playing-policy.json");
  const logits = new Float32Array(CARD_COUNT);

  for (let index = 0; index < logits.length; index += 1) {
    logits[index] = (index % 13) / 5;
  }

  await writeFile(onnxPath, createConstantPolicyOnnx(logits));
  await writeFile(metadataPath, JSON.stringify(createPlayingMetadata()) + "\n", "utf8");

  return {
    onnxPath,
    metadataPath,
    artifactId: "test-playing-policy"
  };
}

async function createBiddingPolicyFixture(directory: string): Promise<{
  onnxPath: string;
  metadataPath: string;
  artifactId: string;
}> {
  await mkdir(directory, { recursive: true });
  const onnxPath = join(directory, "bidding-policy.onnx");
  const metadataPath = join(directory, "bidding-policy.json");
  const logits = new Float32Array(BIDDING_ACTION_COUNT);
  logits[0] = -100;
  for (let index = 1; index < logits.length; index += 1) {
    logits[index] = -index;
  }

  await writeFile(
    onnxPath,
    createConstantPolicyOnnx(logits, ONNX_OUTPUT_NAME, {
      inputFeatureCount: BIDDING_MODEL_INPUT_FEATURE_COUNT,
      outputCount: BIDDING_ACTION_COUNT
    })
  );
  await writeFile(
    metadataPath,
    JSON.stringify(createNonPlayingMetadata("bidding")) + "\n",
    "utf8"
  );

  return {
    onnxPath,
    metadataPath,
    artifactId: "test-bidding-policy"
  };
}

function createPlayingMetadata() {
  return {
    metadataSchemaVersion: 1,
    checkpointSchemaVersion: 1,
    datasetSchemaVersion: 1,
    playingEncoderSchemaVersion: 2,
    modelInputSchemaVersion: 2,
    cardIdsSha256: calculateCardIdsSha256(),
    onnx: {
      opsetVersion: ONNX_OPSET_VERSION,
      inputs: [
        {
          name: ONNX_INPUT_NAME,
          shape: ["batch", MODEL_INPUT_FEATURE_COUNT],
          dtype: "float32"
        }
      ],
      outputs: [
        {
          name: ONNX_OUTPUT_NAME,
          shape: ["batch", CARD_COUNT],
          dtype: "float32"
        }
      ]
    },
    policyModel: {
      input_dim: MODEL_INPUT_FEATURE_COUNT,
      hidden_dim: 8,
      hidden_layers: 1,
      dropout: 0
    }
  };
}

function createNonPlayingMetadata(policyType: NonPlayingPolicyType): NonPlayingPolicyOnnxMetadata {
  return {
    metadataSchemaVersion: 1,
    artifactType: "napoleon-bidding-policy-onnx",
    policyType,
    checkpointSchemaVersion: 1,
    datasetSchemaVersion: 2,
    encoderSchemaVersion: 1,
    modelInputSchemaVersion: 1,
    modelInputFeatureCount: BIDDING_MODEL_INPUT_FEATURE_COUNT,
    outputLogitCount: BIDDING_ACTION_COUNT,
    actionCount: BIDDING_ACTION_COUNT,
    cardIdsSha256: calculateCardIdsSha256(),
    inputName: ONNX_INPUT_NAME,
    outputName: ONNX_OUTPUT_NAME,
    inputShape: ["batch", BIDDING_MODEL_INPUT_FEATURE_COUNT],
    outputShape: ["batch", BIDDING_ACTION_COUNT],
    inputDtype: "float32",
    outputDtype: "float32",
    onnx: {
      opsetVersion: ONNX_OPSET_VERSION,
      inputs: [
        {
          name: ONNX_INPUT_NAME,
          shape: ["batch", BIDDING_MODEL_INPUT_FEATURE_COUNT],
          dtype: "float32"
        }
      ],
      outputs: [
        {
          name: ONNX_OUTPUT_NAME,
          shape: ["batch", BIDDING_ACTION_COUNT],
          dtype: "float32"
        }
      ]
    },
    modelConfig: {
      input_dim: BIDDING_MODEL_INPUT_FEATURE_COUNT,
      hidden_dim: 8,
      hidden_layers: 1,
      dropout: 0
    },
    checkpointSeed: 123,
    checkpointCompatibilityMetadata: {
      modelInputFeatureCount: BIDDING_MODEL_INPUT_FEATURE_COUNT,
      outputCount: BIDDING_ACTION_COUNT
    }
  };
}

function expectReward(
  outcome: Pick<NonPlayingBiddingRlOutcome, "winner" | "targetPointCards" | "actingPlayerRole">,
  reward: number
): void {
  expect(calculateNonPlayingTerminalRoleReward(outcome)).toBe(reward);
}

async function withTempDir(run: (directory: string) => Promise<void>): Promise<void> {
  const directory = await mkdtemp(join(tmpdir(), "napoleon-non-playing-rl-test-"));

  try {
    await run(directory);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}

async function readManifest(outputDirectory: string): Promise<NonPlayingRlDatasetManifest> {
  return JSON.parse(await readFile(join(outputDirectory, "manifest.json"), "utf8")) as NonPlayingRlDatasetManifest;
}

async function readAllShardLines(
  outputDirectory: string,
  manifest: NonPlayingRlDatasetManifest
): Promise<readonly string[]> {
  const chunks = await Promise.all(
    manifest.shards.map((shard) => readFile(join(outputDirectory, shard.file), "utf8"))
  );

  return chunks.flatMap((chunk) => chunk.split("\n").filter(Boolean));
}

async function expectDirectoriesToBeByteIdentical(
  firstOutput: string,
  secondOutput: string
): Promise<void> {
  const firstFiles = (await readdir(firstOutput)).sort();
  const secondFiles = (await readdir(secondOutput)).sort();

  expect(firstFiles).toEqual(secondFiles);

  for (const file of firstFiles) {
    expect(await readFile(join(secondOutput, file), "utf8")).toBe(
      await readFile(join(firstOutput, file), "utf8")
    );
  }
}

function assertNoCompleteStateFields(lines: readonly string[]): void {
  const forbiddenFields = [
    "actualHands",
    "actualState",
    "unusedCardIds",
    "excludedCardIds",
    "awardedPointCardIds",
    "currentTrickCardIds",
    "completedTrickCardIds",
    "beliefTarget",
    "teacherAction",
    "ruleBasedAction",
    "futureAction"
  ];

  for (const line of lines) {
    for (const field of forbiddenFields) {
      expect(line).not.toContain(`"${field}"`);
    }
  }
}

async function sha256File(path: string): Promise<string> {
  return createHash("sha256").update(await readFile(path)).digest("hex");
}

function sha256Utf8(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}
