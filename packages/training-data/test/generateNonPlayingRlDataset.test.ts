import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  ADJUTANT_MODEL_INPUT_FEATURE_COUNT,
  ADJUTANT_MODEL_INPUT_SCHEMA_VERSION,
  ADJUTANT_ENCODER_SCHEMA_VERSION,
  BIDDING_ACTION_COUNT,
  BIDDING_ENCODER_SCHEMA_VERSION,
  BIDDING_MODEL_INPUT_FEATURE_COUNT,
  BIDDING_MODEL_INPUT_SCHEMA_VERSION,
  CARD_COUNT,
  EXCHANGE_ENCODER_SCHEMA_VERSION,
  EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
  EXCHANGE_MODEL_INPUT_SCHEMA_VERSION,
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
  CONSERVATIVE_BIDDING_BASELINE_ID,
  FROZEN_BIDDING_OPPONENT_MIX_RULE_VERSION,
  NON_PLAYING_RL_DATASET_GENERATOR_VERSION,
  NON_PLAYING_RL_DATASET_SAMPLE_TYPE,
  NON_PLAYING_RL_PHASE_SCOPE,
  NON_PLAYING_RL_REWARD_ID,
  NON_PLAYING_RL_REWARD_TYPE,
  NON_PLAYING_RL_REWARD_VERSION,
  NON_PLAYING_RL_SAMPLING_ALGORITHM,
  PASSIVE_BIDDING_BASELINE_ID,
  calculateNonPlayingAdjutantLogProbability,
  calculateNonPlayingBiddingLogProbability,
  calculateNonPlayingExchangeLogProbability,
  calculateNonPlayingTerminalRoleReward,
  generateNonPlayingAdjutantRlDataset,
  generateNonPlayingBiddingRlDataset,
  generateNonPlayingExchangeRlDataset,
  selectFrozenBiddingOpponentPolicy,
  validateNonPlayingAdjutantRlDatasetManifest,
  validateNonPlayingAdjutantRlSample,
  validateNonPlayingBiddingRlSample,
  validateNonPlayingExchangeRlDatasetManifest,
  validateNonPlayingExchangeRlSample,
  validateNonPlayingRlDatasetManifest
} from "../src/index.js";
import type {
  NonPlayingAdjutantRlSample,
  NonPlayingBiddingRlOutcome,
  NonPlayingBiddingRlSample,
  NonPlayingExchangeRlSample,
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
      const progressEvents: Array<{
        completedGames: number;
        sampleCount: number;
        completedShards: number;
        currentSeed: number;
      }> = [];
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
          completedShards: number;
          currentSeed: number;
        }) => {
          progressEvents.push({
            completedGames: progress.completedGames,
            sampleCount: progress.sampleCount,
            completedShards: progress.completedShards,
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
      expect(manifest.datasetSchemaVersion).toBe(2);
      expect(manifest.sampleSchemaVersion).toBe(2);
      expect(manifest.generatorVersion).toBe(NON_PLAYING_RL_DATASET_GENERATOR_VERSION);
      expect(manifest.format).toBe(DATASET_FORMAT);
      expect(manifest.sampleType).toBe(NON_PLAYING_RL_DATASET_SAMPLE_TYPE);
      expect(manifest.phaseScope).toBe(NON_PLAYING_RL_PHASE_SCOPE);
      expect(manifest.learnedPhases).toEqual(["bidding"]);
      expect(manifest.ruleBasedPhases).toEqual(["choosing-adjutant", "exchanging"]);
      expect(manifest.fixedPhases).toEqual(["playing"]);
      expect(manifest.rolloutPolicyTopology).toBe("candidate-x1-frozen-x4-v1");
      expect(manifest.gameCountUnit).toBe("logical-seeds");
      expect(manifest.logicalSeedCount).toBe(2);
      expect(manifest.actualGameCount).toBe(10);
      expect(manifest.rotationOffsets).toEqual([0, 1, 2, 3, 4]);
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
      expect(manifest.gameRule).toEqual({
        id: "bidding-10-19-all-pass-9-v1",
        biddingMinTargetPointCards: 10,
        biddingMaxTargetPointCards: 19,
        allPassForcedTargetPointCards: 9
      });
      expect(manifest.biddingModelInputFeatureCount).toBe(BIDDING_MODEL_INPUT_FEATURE_COUNT);
      expect(manifest.playingModelInputFeatureCount).toBe(MODEL_INPUT_FEATURE_COUNT);
      expect(manifest.actionCount).toBe(BIDDING_ACTION_COUNT);
      expect(manifest.sampleCount).toBe(lines.length);
      expect(manifest.sampleCount).toBeGreaterThan(0);
      expect(progressEvents.map((event) => event.completedGames)).toEqual([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
      ]);
      expect(progressEvents.map((event) => event.currentSeed)).toEqual([
        7, 7, 7, 7, 7, 8, 8, 8, 8, 8
      ]);
      expect(progressEvents.at(-1)?.sampleCount).toBe(manifest.sampleCount);
      expect(progressEvents.at(-1)?.completedShards).toBe(manifest.shardCount);
      expect(manifest.nonLearningAgents.bidding).toMatchObject({
        type: "mixed-frozen-bidding",
        mixingRuleVersion: FROZEN_BIDDING_OPPONENT_MIX_RULE_VERSION,
        selectionUnit: "game-seat",
        policies: {
          conservative: {
            type: "conservative-bidding",
            id: CONSERVATIVE_BIDDING_BASELINE_ID
          },
          passive: {
            type: "passive-bidding",
            id: PASSIVE_BIDDING_BASELINE_ID
          }
        }
      });
      expect(manifest.diagnostics).toMatchObject({
        candidateSeatCount: 1,
        frozenSeatCount: 4,
        candidateRotationSeatCount: 5,
        actualGameCount: 10,
        logicalSeedCount: 2,
        rotationOffsets: [0, 1, 2, 3, 4]
      });
      expect(manifest.diagnostics?.bidding?.candidateBiddingDecisionCount).toBe(samples.length);
      expect(Object.keys(manifest.diagnostics?.bidding?.targetPointCardsDistribution ?? {})).toEqual([
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19"
      ]);
      expect(
        (manifest.diagnostics?.bidding?.passCount ?? 0) +
          (manifest.diagnostics?.bidding?.bidCount ?? 0)
      ).toBe(samples.length);
      const opponentMix = manifest.diagnostics?.frozenBiddingOpponentMix;
      expect(opponentMix).toMatchObject({
        type: "mixed-frozen-bidding",
        mixingRuleVersion: FROZEN_BIDDING_OPPONENT_MIX_RULE_VERSION
      });
      expect(opponentMix?.seatAssignments).toHaveLength(manifest.actualGameCount * 4);
      expect(
        (opponentMix?.conservativeSeatCount ?? 0) + (opponentMix?.passiveSeatCount ?? 0)
      ).toBe(opponentMix?.seatAssignments.length);
      expect(opponentMix?.conservativeSeatCount).toBeGreaterThan(0);
      expect(opponentMix?.passiveSeatCount).toBeGreaterThan(0);
      for (const assignment of opponentMix?.seatAssignments ?? []) {
        expect(assignment.playerIndex).not.toBe(assignment.candidateSeatIndex);
        expect(assignment.rotationOffset).toBe(assignment.candidateSeatIndex);
        expect(assignment.policy).toEqual(
          selectFrozenBiddingOpponentPolicy({
            seed: assignment.seed,
            candidateSeatIndex: assignment.candidateSeatIndex,
            playerIndex: assignment.playerIndex
          })
        );
      }

      const playerDecisionCounts = new Map<string, number>();
      for (const sample of samples) {
        validateNonPlayingBiddingRlSample(sample, sample.seed);
        expect(sample.phase).toBe("bidding");
        expect(sample.actingPlayerIndex).toBe(sample.candidateSeatIndex);
        expect(sample.candidateSeatIndex).toBe(sample.rotationOffset);
        expect(sample.modelInput).toHaveLength(BIDDING_MODEL_INPUT_FEATURE_COUNT);
        expect(sample.legalBidMask).toHaveLength(BIDDING_ACTION_COUNT);
        expect(sample.legalBidMask[sample.selectedActionIndex]).toBe(1);
        expect(sample.terminalReward).toBe(calculateNonPlayingTerminalRoleReward(sample.outcome));
        expect(sample.outcome.targetPointCards).toBeGreaterThanOrEqual(9);
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

      expect(new Set(samples.map((sample) => sample.candidateSeatIndex))).toEqual(
        new Set([0, 1, 2, 3, 4])
      );
      expect([...playerDecisionCounts.values()].every((count) => count === 1)).toBe(true);
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

  it("generates deterministic adjutant RL samples with 53-card legal masks", async () => {
    await withTempDir(async (directory) => {
      const playingArtifact = await createPlayingPolicyFixture(join(directory, "playing"));
      const adjutantArtifact = await createAdjutantPolicyFixture(join(directory, "adjutant"));
      const playingPolicy = await loadPolicyOnnxModel(playingArtifact);
      const adjutantPolicy = await loadNonPlayingPolicyOnnxModel(adjutantArtifact);
      const output = join(directory, "adjutant-rl");
      const result = await generateNonPlayingAdjutantRlDataset({
        outputDirectory: output,
        adjutantPolicy,
        adjutantPolicyArtifact: adjutantArtifact,
        playingPolicy,
        playingPolicyArtifact: playingArtifact,
        startSeed: 21,
        gameCount: 2,
        gamesPerShard: 1,
        temperature: 0.05
      });

      const manifest = await readManifest(output);
      const lines = await readAllShardLines(output, manifest);
      const samples = lines.map((line) => JSON.parse(line) as NonPlayingAdjutantRlSample);

      expect(result.manifest).toEqual(manifest);
      validateNonPlayingAdjutantRlDatasetManifest(manifest);
      expect(() =>
        validateNonPlayingAdjutantRlDatasetManifest({
          ...manifest,
          rolloutPolicyTopology: "legacy-self-play-x5"
        } as unknown as NonPlayingRlDatasetManifest)
      ).toThrow("rollout topology");
      expect(manifest.sampleType).toBe("non-playing-adjutant-rl-sample");
      expect(manifest.phaseScope).toBe("adjutant-only");
      expect(manifest.learnedPhases).toEqual(["choosing-adjutant"]);
      expect(manifest.ruleBasedPhases).toEqual(["bidding", "exchanging"]);
      expect(manifest.fixedPhases).toEqual(["playing"]);
      expect(manifest.adjutantModelInputFeatureCount).toBe(ADJUTANT_MODEL_INPUT_FEATURE_COUNT);
      expect(manifest.actionCount).toBe(CARD_COUNT);
      expect(manifest.behaviorPolicy).toMatchObject({
        type: "adjutant-onnx",
        artifactId: "test-adjutant-policy"
      });
      expect(manifest.nonLearningAgents.bidding).toMatchObject({
        type: "conservative-bidding",
        id: CONSERVATIVE_BIDDING_BASELINE_ID
      });
      expect(manifest.sampleCount).toBe(samples.length);
      expect(samples.length).toBeGreaterThan(0);

      for (const sample of samples) {
        validateNonPlayingAdjutantRlSample(sample, sample.seed);
        expect(sample.phase).toBe("choosing-adjutant");
        expect(sample.modelInput).toHaveLength(ADJUTANT_MODEL_INPUT_FEATURE_COUNT);
        expect(sample.legalAdjutantMask).toHaveLength(CARD_COUNT);
        expect(sum(sample.legalAdjutantMask)).toBe(CARD_COUNT);
        expect(sample.legalAdjutantMask[sample.selectedActionIndex]).toBe(1);
        expect(sample.terminalReward).toBe(calculateNonPlayingTerminalRoleReward(sample.outcome));

        const logits = await adjutantPolicy.predictLogits(sample.modelInput);
        const recomputedLogProbability = calculateNonPlayingAdjutantLogProbability({
          logits,
          legalAdjutantMask: sample.legalAdjutantMask,
          selectedActionIndex: sample.selectedActionIndex,
          temperature: manifest.temperature
        });

        expect(sample.behaviorLogProbability).toBeCloseTo(recomputedLogProbability, 5);
        expect(sample.behaviorLogProbability).toBeLessThanOrEqual(0);
      }

      assertNoCompleteStateFields(lines);
    });
  });

  it("calculates terminal role rewards from the v3 reward table", () => {
    expectReward({ winner: "napoleon-team", targetPointCards: 13, actingPlayerRole: "napoleon" }, 26);
    expectReward({ winner: "alliance", targetPointCards: 13, actingPlayerRole: "napoleon" }, -5);
    expectReward({ winner: "napoleon-team", targetPointCards: 13, actingPlayerRole: "adjutant" }, 13);
    expectReward({ winner: "alliance", targetPointCards: 13, actingPlayerRole: "adjutant" }, 0);
    expectReward({ winner: "napoleon-team", targetPointCards: 13, actingPlayerRole: "citizen" }, 13);
    expectReward({ winner: "alliance", targetPointCards: 13, actingPlayerRole: "citizen" }, 0);
    expectReward({
      winner: "napoleon-team",
      targetPointCards: 13,
      actingPlayerRole: "napoleon-adjutant"
    }, 39);
    expectReward({
      winner: "alliance",
      targetPointCards: 13,
      actingPlayerRole: "napoleon-adjutant"
    }, -5);

    expectReward({ winner: "napoleon-team", targetPointCards: 19, actingPlayerRole: "napoleon" }, 38);
    expectReward({ winner: "napoleon-team", targetPointCards: 19, actingPlayerRole: "adjutant" }, 19);
    expectReward({ winner: "napoleon-team", targetPointCards: 19, actingPlayerRole: "citizen" }, 19);
    expectReward({
      winner: "napoleon-team",
      targetPointCards: 19,
      actingPlayerRole: "napoleon-adjutant"
    }, 57);
    expectReward({ winner: "napoleon-team", targetPointCards: 9, actingPlayerRole: "napoleon" }, 18);
    expectReward({ winner: "alliance", targetPointCards: 9, actingPlayerRole: "napoleon" }, -5);
  });

  it("generates deterministic exchange RL samples with 3 sequential legal card steps", async () => {
    await withTempDir(async (directory) => {
      const playingArtifact = await createPlayingPolicyFixture(join(directory, "playing"));
      const exchangeArtifact = await createExchangePolicyFixture(join(directory, "exchange"));
      const playingPolicy = await loadPolicyOnnxModel(playingArtifact);
      const exchangePolicy = await loadNonPlayingPolicyOnnxModel(exchangeArtifact);
      const output = join(directory, "exchange-rl");
      const result = await generateNonPlayingExchangeRlDataset({
        outputDirectory: output,
        exchangePolicy,
        exchangePolicyArtifact: exchangeArtifact,
        playingPolicy,
        playingPolicyArtifact: playingArtifact,
        startSeed: 31,
        gameCount: 2,
        gamesPerShard: 1,
        temperature: 0.05
      });

      const manifest = await readManifest(output);
      const lines = await readAllShardLines(output, manifest);
      const samples = lines.map((line) => JSON.parse(line) as NonPlayingExchangeRlSample);

      expect(result.manifest).toEqual(manifest);
      validateNonPlayingExchangeRlDatasetManifest(manifest);
      expect(() =>
        validateNonPlayingExchangeRlDatasetManifest({
          ...manifest,
          actualGameCount: manifest.logicalSeedCount
        })
      ).toThrow("rotation metadata");
      expect(manifest.sampleType).toBe("non-playing-exchange-rl-sample");
      expect(manifest.phaseScope).toBe("exchange-only");
      expect(manifest.learnedPhases).toEqual(["exchanging"]);
      expect(manifest.ruleBasedPhases).toEqual(["bidding", "choosing-adjutant"]);
      expect(manifest.fixedPhases).toEqual(["playing"]);
      expect(manifest.exchangeModelInputFeatureCount).toBe(EXCHANGE_MODEL_INPUT_FEATURE_COUNT);
      expect(manifest.decisionMode).toBe("sequential-card-v1");
      expect(manifest.actionCount).toBe(CARD_COUNT);
      expect(manifest.behaviorPolicy).toMatchObject({
        type: "exchange-onnx",
        artifactId: "test-exchange-policy"
      });
      expect(samples.length).toBeGreaterThan(0);
      expect(samples.length % 3).toBe(0);

      for (let index = 0; index < samples.length; index += 3) {
        const group = samples.slice(index, index + 3);
        expect(group.map((sample) => sample.exchangeStepIndex)).toEqual([0, 1, 2]);
        expect(group.map((sample) => sample.remainingDiscardCount)).toEqual([3, 2, 1]);
        expect(new Set(group.map((sample) => sample.selectedActionIndex)).size).toBe(3);
        expect(new Set(group.map((sample) => sample.terminalReward)).size).toBe(1);
      }

      for (const sample of samples) {
        validateNonPlayingExchangeRlSample(sample, sample.seed);
        expect(sample.phase).toBe("exchanging");
        expect(sample.modelInput).toHaveLength(EXCHANGE_MODEL_INPUT_FEATURE_COUNT);
        expect(sample.legalDiscardCardMask).toHaveLength(CARD_COUNT);
        expect(sum(sample.legalDiscardCardMask)).toBe(13 - sample.exchangeStepIndex);
        expect(sample.legalDiscardCardMask[sample.selectedActionIndex]).toBe(1);
        expect(sample.terminalReward).toBe(calculateNonPlayingTerminalRoleReward(sample.outcome));

        const logits = await exchangePolicy.predictLogits(sample.modelInput);
        const recomputedLogProbability = calculateNonPlayingExchangeLogProbability({
          logits,
          legalDiscardCardMask: sample.legalDiscardCardMask,
          selectedActionIndex: sample.selectedActionIndex,
          temperature: manifest.temperature
        });

        expect(sample.behaviorLogProbability).toBeCloseTo(recomputedLogProbability, 5);
        expect(sample.behaviorLogProbability).toBeLessThanOrEqual(0);
      }

      assertNoCompleteStateFields(lines);
    });
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

async function createAdjutantPolicyFixture(directory: string): Promise<{
  onnxPath: string;
  metadataPath: string;
  artifactId: string;
}> {
  await mkdir(directory, { recursive: true });
  const onnxPath = join(directory, "adjutant-policy.onnx");
  const metadataPath = join(directory, "adjutant-policy.json");
  const logits = new Float32Array(CARD_COUNT);
  for (let index = 0; index < logits.length; index += 1) {
    logits[index] = index / 10;
  }

  await writeFile(
    onnxPath,
    createConstantPolicyOnnx(logits, ONNX_OUTPUT_NAME, {
      inputFeatureCount: ADJUTANT_MODEL_INPUT_FEATURE_COUNT,
      outputCount: CARD_COUNT
    })
  );
  await writeFile(
    metadataPath,
    JSON.stringify(createNonPlayingMetadata("adjutant")) + "\n",
    "utf8"
  );

  return {
    onnxPath,
    metadataPath,
    artifactId: "test-adjutant-policy"
  };
}

async function createExchangePolicyFixture(directory: string): Promise<{
  onnxPath: string;
  metadataPath: string;
  artifactId: string;
}> {
  await mkdir(directory, { recursive: true });
  const onnxPath = join(directory, "exchange-policy.onnx");
  const metadataPath = join(directory, "exchange-policy.json");
  const logits = new Float32Array(CARD_COUNT);
  for (let index = 0; index < logits.length; index += 1) {
    logits[index] = index / 10;
  }

  await writeFile(
    onnxPath,
    createConstantPolicyOnnx(logits, ONNX_OUTPUT_NAME, {
      inputFeatureCount: EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
      outputCount: CARD_COUNT
    })
  );
  await writeFile(
    metadataPath,
    JSON.stringify(createNonPlayingMetadata("exchange")) + "\n",
    "utf8"
  );

  return {
    onnxPath,
    metadataPath,
    artifactId: "test-exchange-policy"
  };
}

function createPlayingMetadata() {
  return {
    metadataSchemaVersion: 1,
    checkpointSchemaVersion: 1,
    datasetSchemaVersion: 1,
    playingEncoderSchemaVersion: 3,
    modelInputSchemaVersion: 3,
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
  const spec = nonPlayingSpec(policyType);
  const metadata: NonPlayingPolicyOnnxMetadata = {
    metadataSchemaVersion: 1,
    artifactType: spec.artifactType,
    policyType,
    checkpointSchemaVersion: 1,
    datasetSchemaVersion: 2,
    encoderSchemaVersion: spec.encoderSchemaVersion,
    modelInputSchemaVersion: spec.modelInputSchemaVersion,
    modelInputFeatureCount: spec.inputFeatureCount,
    outputLogitCount: spec.outputCount,
    actionCount: spec.outputCount,
    cardIdsSha256: calculateCardIdsSha256(),
    inputName: ONNX_INPUT_NAME,
    outputName: ONNX_OUTPUT_NAME,
    inputShape: ["batch", spec.inputFeatureCount],
    outputShape: ["batch", spec.outputCount],
    inputDtype: "float32",
    outputDtype: "float32",
    onnx: {
      opsetVersion: ONNX_OPSET_VERSION,
      inputs: [
        {
          name: ONNX_INPUT_NAME,
          shape: ["batch", spec.inputFeatureCount],
          dtype: "float32"
        }
      ],
      outputs: [
        {
          name: ONNX_OUTPUT_NAME,
          shape: ["batch", spec.outputCount],
          dtype: "float32"
        }
      ]
    },
    modelConfig: {
      input_dim: spec.inputFeatureCount,
      hidden_dim: 8,
      hidden_layers: 1,
      dropout: 0
    },
    checkpointSeed: 123,
    checkpointCompatibilityMetadata: {
      modelInputFeatureCount: spec.inputFeatureCount,
      outputCount: spec.outputCount
    }
  };
  if (policyType === "exchange") {
    metadata.discardCount = 3;
    metadata.decisionMode = "sequential-card-v1";
  }
  return metadata;
}

function nonPlayingSpec(policyType: NonPlayingPolicyType): {
  artifactType: string;
  inputFeatureCount: number;
  outputCount: number;
  encoderSchemaVersion: number;
  modelInputSchemaVersion: number;
} {
  if (policyType === "bidding") {
    return {
      artifactType: "napoleon-bidding-policy-onnx",
      inputFeatureCount: BIDDING_MODEL_INPUT_FEATURE_COUNT,
      outputCount: BIDDING_ACTION_COUNT,
      encoderSchemaVersion: BIDDING_ENCODER_SCHEMA_VERSION,
      modelInputSchemaVersion: BIDDING_MODEL_INPUT_SCHEMA_VERSION
    };
  }
  if (policyType === "exchange") {
    return {
      artifactType: "napoleon-exchange-policy-onnx",
      inputFeatureCount: EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
      outputCount: CARD_COUNT,
      encoderSchemaVersion: EXCHANGE_ENCODER_SCHEMA_VERSION,
      modelInputSchemaVersion: EXCHANGE_MODEL_INPUT_SCHEMA_VERSION
    };
  }
  return {
    artifactType: "napoleon-adjutant-policy-onnx",
    inputFeatureCount: ADJUTANT_MODEL_INPUT_FEATURE_COUNT,
    outputCount: CARD_COUNT,
    encoderSchemaVersion: ADJUTANT_ENCODER_SCHEMA_VERSION,
    modelInputSchemaVersion: ADJUTANT_MODEL_INPUT_SCHEMA_VERSION
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
