import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { describe, expect, it, vi } from "vitest";
import {
  CARD_COUNT,
  COMPLETE_INFO_PLAYING_MODEL_INPUT_FEATURE_COUNT,
  MODEL_INPUT_FEATURE_COUNT,
  createPlayingModelInput
} from "@napoleon/ai-observation";
import {
  ONNX_INPUT_NAME,
  ONNX_OPSET_VERSION,
  ONNX_OUTPUT_NAME,
  calculateCardIdsSha256,
  loadPolicyOnnxModel
} from "../../policy-onnx/src/index.js";
import {
  DATASET_FORMAT,
  PLAYING_SELF_PLAY_DATASET_GENERATOR_VERSION,
  PLAYING_SELF_PLAY_BINARY_DATASET_FORMAT,
  PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE,
  PLAYING_SELF_PLAY_ROSTER_ASSIGNMENT,
  PLAYING_SELF_PLAY_REWARD_TYPE,
  PLAYING_SELF_PLAY_REWARD_VERSION,
  PLAYING_SELF_PLAY_SAMPLING_ALGORITHM,
  assignRolloutRosterForSeed,
  calculatePlayingSelfPlayLogProbability,
  COMPLETE_INFO_COMPACT_PLAYING_OBSERVATION_VARIANT,
  generatePlayingSelfPlayDataset,
  runPlayingSelfPlayGame,
  runPlayingSelfPlayGameWithSamples,
  validatePlayingSelfPlayDatasetManifest,
  validatePlayingSelfPlaySample
} from "../src/index.js";
import type {
  PlayingSelfPlayDatasetManifest,
  PlayingSelfPlayGameRunner,
  PlayingSelfPlayPolicy,
  PlayingSelfPlayRolloutRosterOptions,
  PlayingSelfPlaySample,
  PlayingSelfPlayTensorSample
} from "../src/index.js";
import { createConstantPolicyOnnx } from "../../policy-onnx/test/testOnnxFixture.js";

describe("generatePlayingSelfPlayDataset", () => {
  it("generates deterministic playing self-play trajectories with valid rewards and hashes", async () => {
    await withTempDir(async (directory) => {
      const artifact = await createPlayingPolicyFixture(directory);
      const policy = await loadPolicyOnnxModel(artifact);
      const firstOutput = join(directory, "first");
      const secondOutput = join(directory, "second");
      const progressEvents: Array<{ completedGames: number; sampleCount: number; currentSeed: number }> = [];
      const options = {
        outputDirectory: firstOutput,
        playingPolicy: policy,
        playingPolicyArtifact: artifact,
        startSeed: 11,
        gameCount: 2,
        gamesPerShard: 1,
        format: DATASET_FORMAT,
        temperature: 1.25,
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

      const result = await generatePlayingSelfPlayDataset(options);
      await generatePlayingSelfPlayDataset({
        ...options,
        outputDirectory: secondOutput,
        onProgress: undefined
      });
      await expectDirectoriesToBeByteIdentical(firstOutput, secondOutput);

      const manifest = await readManifest(firstOutput);
      const lines = await readAllShardLines(firstOutput, manifest);
      const samples = lines.map((line) => JSON.parse(line) as PlayingSelfPlaySample);

      expect(result.manifest).toEqual(manifest);
      validatePlayingSelfPlayDatasetManifest(manifest);
      expect(manifest.datasetSchemaVersion).toBe(3);
      expect(manifest.generatorVersion).toBe(PLAYING_SELF_PLAY_DATASET_GENERATOR_VERSION);
      expect(manifest.sampleType).toBe(PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE);
      expect(manifest.sampleSchemaVersion).toBe(3);
      expect(manifest.samplingAlgorithm).toBe(PLAYING_SELF_PLAY_SAMPLING_ALGORITHM);
      expect(manifest.temperature).toBe(1.25);
      expect(manifest.reward).toEqual({
        type: PLAYING_SELF_PLAY_REWARD_TYPE,
        version: PLAYING_SELF_PLAY_REWARD_VERSION
      });
      expect(manifest.nonPlayingAgent).toEqual({
        type: "rule-based",
        version: 1
      });
      expect(manifest.rolloutRoster).toEqual({
        assignment: PLAYING_SELF_PLAY_ROSTER_ASSIGNMENT,
        seats: Array.from({ length: 5 }, () => ({ source: "current-policy" }))
      });
      expect(manifest.behaviorPolicy.onnxSha256).toBe(await sha256File(artifact.onnxPath));
      expect(manifest.behaviorPolicy.metadataSha256).toBe(await sha256File(artifact.metadataPath));
      expect(manifest.behaviorPolicy.requestedInferenceDevice).toBe("cpu");
      expect(manifest.behaviorPolicy.resolvedInferenceDevice).toBe("cpu");
      expect(manifest.behaviorPolicy.executionProvider).toBe("cpu");
      expect(manifest.cardIdsSha256).toBe(calculateCardIdsSha256());
      expect(manifest.gameCount).toBe(2);
      expect(manifest.shardCount).toBe(2);
      expect(manifest.sampleCount).toBe(lines.length);
      expect(manifest.sampleCount).toBeGreaterThan(0);
      expect(progressEvents.map((event) => event.completedGames)).toEqual([1, 2]);
      expect(progressEvents.map((event) => event.currentSeed)).toEqual([11, 12]);
      expect(progressEvents.at(-1)?.sampleCount).toBe(manifest.sampleCount);

      const rewards = new Set(samples.map((sample) => sample.terminalReward));
      const forcedSamples = samples.filter((sample) => sum(sample.observation.legalPlayMask) === 1);
      const nonForcedSamples = samples.filter((sample) => sum(sample.observation.legalPlayMask) > 1);

      expect(rewards).toEqual(new Set([1, -1]));
      expect(forcedSamples.length).toBeGreaterThan(0);
      expect(nonForcedSamples.length).toBeGreaterThan(0);

      for (const sample of samples) {
        validatePlayingSelfPlaySample(sample, sample.seed);
        expect(sample.actingSeatSource).toBe("current-policy");
        expect(sample.behaviorPolicyArtifactId).toBe("test-playing-policy");
        expect(sample.rolloutSeatSources).toEqual([
          "current-policy",
          "current-policy",
          "current-policy",
          "current-policy",
          "current-policy"
        ]);
        expect(sample.observation.legalPlayMask[sample.selectedCardIndex]).toBe(1);
        expect(sample.terminalReward).toBe(sample.outcome.actingPlayerTeam === sample.outcome.winner ? 1 : -1);

        const { modelInput, legalPlayMask } = createPlayingModelInput(sample.observation);
        const logits = await policy.predictLogits(modelInput);
        const recomputedLogProbability = calculatePlayingSelfPlayLogProbability({
          logits,
          legalPlayMask,
          selectedCardIndex: sample.selectedCardIndex,
          temperature: manifest.temperature
        });

        expect(sample.behaviorLogProbability).toBeCloseTo(recomputedLogProbability, 6);

        if (sum(sample.observation.legalPlayMask) === 1) {
          expect(sample.behaviorLogProbability).toBe(0);
        } else {
          expect(Number.isFinite(sample.behaviorLogProbability)).toBe(true);
          expect(sample.behaviorLogProbability).toBeLessThanOrEqual(0);
        }
      }

      for (const shard of manifest.shards) {
        const shardPath = join(firstOutput, shard.file);
        const file = await readFile(shardPath, "utf8");
        const fileStat = await stat(shardPath);

        expect(file.split("\n").filter(Boolean)).toHaveLength(shard.sampleCount);
        expect(fileStat.size).toBe(shard.byteLength);
        expect(sha256Utf8(file)).toBe(shard.sha256);
      }

      expect(lines.some((line) => line.includes("\"terminalReward\":1"))).toBe(true);
      expect(lines.some((line) => line.includes("\"terminalReward\":-1"))).toBe(true);
      assertNoHiddenStateFields(lines);
    });
  });

  it("writes binary tensor-ready shards by default", async () => {
    await withTempDir(async (directory) => {
      const artifact = await createPlayingPolicyFixture(directory);
      const policy = await loadPolicyOnnxModel(artifact);
      const output = join(directory, "binary");

      await generatePlayingSelfPlayDataset({
        outputDirectory: output,
        playingPolicy: policy,
        playingPolicyArtifact: artifact,
        startSeed: 50,
        gameCount: 2,
        gamesPerShard: 1
      });

      const manifest = await readManifest(output);
      validatePlayingSelfPlayDatasetManifest(manifest);
      expect(manifest.datasetSchemaVersion).toBe(4);
      expect(manifest.format).toBe(PLAYING_SELF_PLAY_BINARY_DATASET_FORMAT);
      expect(manifest.sampleSchemaVersion).toBe(4);
      expect(manifest.tensorSchema).toMatchObject({
        shardSchemaVersion: 1,
        byteOrder: "little-endian",
        compression: "none"
      });
      expect(manifest.tensorSchema?.fields.map((field) => field.name)).toEqual([
        "modelInput",
        "legalPlayMask",
        "selectedCardIndex",
        "behaviorLogProbability",
        "terminalReward",
        "seed",
        "step",
        "actingPlayerIndex",
        "selfRoleIndex"
      ]);
      expect(manifest.shards.every((shard) => shard.file.endsWith(".bin"))).toBe(true);

      const rawFloat32Bytes = manifest.sampleCount * MODEL_INPUT_FEATURE_COUNT * 4;
      const shardBytes = sum(manifest.shards.map((shard) => shard.byteLength));
      expect(shardBytes).toBeGreaterThan(rawFloat32Bytes);

      for (const shard of manifest.shards) {
        const shardPath = join(output, shard.file);
        const bytes = await readFile(shardPath);
        const fileStat = await stat(shardPath);
        expect(fileStat.size).toBe(shard.byteLength);
        expect(createHash("sha256").update(bytes).digest("hex")).toBe(shard.sha256);
      }
    });
  });

  it("wires compact complete-info model input into binary self-play rollout", async () => {
    await withTempDir(async (directory) => {
      const onnxPath = join(directory, "compact.onnx");
      const metadataPath = join(directory, "compact.json");
      await writeFile(onnxPath, "compact-policy");
      await writeFile(metadataPath, "{}\n");

      const observedInputLengths: number[] = [];
      const policy: PlayingSelfPlayPolicy = {
        metadata: { playingObservationVariant: COMPLETE_INFO_COMPACT_PLAYING_OBSERVATION_VARIANT },
        predictLogits: async (modelInput) => {
          observedInputLengths.push(modelInput.length);
          return new Float32Array(CARD_COUNT);
        }
      };
      const output = join(directory, "compact-binary");

      await generatePlayingSelfPlayDataset({
        outputDirectory: output,
        playingPolicy: policy,
        playingPolicyArtifact: {
          onnxPath,
          metadataPath,
          artifactId: "compact-policy"
        },
        startSeed: 61,
        gameCount: 1,
        gamesPerShard: 1,
        observationVariant: COMPLETE_INFO_COMPACT_PLAYING_OBSERVATION_VARIANT
      });

      const manifest = await readManifest(output);
      validatePlayingSelfPlayDatasetManifest(manifest);
      expect(manifest.playingObservationVariant).toBe(COMPLETE_INFO_COMPACT_PLAYING_OBSERVATION_VARIANT);
      expect(manifest.modelInputFeatureCount).toBe(COMPLETE_INFO_PLAYING_MODEL_INPUT_FEATURE_COUNT);
      expect(manifest.playingEncoderSchemaVersion).toBe(1);
      expect(manifest.playingModelInputSchemaVersion).toBe(1);
      expect(manifest.tensorSchema?.fields[0]).toEqual({
        name: "modelInput",
        dtype: "float32",
        shape: [COMPLETE_INFO_PLAYING_MODEL_INPUT_FEATURE_COUNT]
      });
      expect(manifest.shards.every((shard) => shard.file.endsWith(".bin"))).toBe(true);
      expect(observedInputLengths.length).toBeGreaterThan(0);
      expect(new Set(observedInputLengths)).toEqual(
        new Set([COMPLETE_INFO_PLAYING_MODEL_INPUT_FEATURE_COUNT])
      );
    });
  });

  it("can write legacy gzip binary shards when explicitly requested", async () => {
    await withTempDir(async (directory) => {
      const artifact = await createPlayingPolicyFixture(directory);
      const policy = await loadPolicyOnnxModel(artifact);
      const output = join(directory, "binary-gzip");

      await generatePlayingSelfPlayDataset({
        outputDirectory: output,
        playingPolicy: policy,
        playingPolicyArtifact: artifact,
        startSeed: 52,
        gameCount: 1,
        gamesPerShard: 1,
        binaryCompression: "gzip"
      });

      const manifest = await readManifest(output);
      validatePlayingSelfPlayDatasetManifest(manifest);
      expect(manifest.tensorSchema?.compression).toBe("gzip");
    });
  });

  it("samples only current-policy decisions with rule-based opponents and rotated seats", async () => {
    await withTempDir(async (directory) => {
      const artifact = await createPlayingPolicyFixture(directory);
      const policy = await loadPolicyOnnxModel(artifact);
      const output = join(directory, "current-vs-rule");
      const roster: PlayingSelfPlayRolloutRosterOptions = {
        seats: [
          { source: "current-policy" },
          { source: "rule-based" },
          { source: "rule-based" },
          { source: "rule-based" },
          { source: "rule-based" }
        ]
      };

      await generatePlayingSelfPlayDataset({
        outputDirectory: output,
        playingPolicy: policy,
        playingPolicyArtifact: artifact,
        rolloutRoster: roster,
        startSeed: 0,
        gameCount: 5,
        gamesPerShard: 5,
        format: DATASET_FORMAT
      });

      const manifest = await readManifest(output);
      const samples = (await readAllShardLines(output, manifest)).map((line) =>
        JSON.parse(line) as PlayingSelfPlaySample
      );

      expect(manifest.rolloutRoster.seats.map((seat) => seat.source)).toEqual([
        "current-policy",
        "rule-based",
        "rule-based",
        "rule-based",
        "rule-based"
      ]);
      expect(samples).toHaveLength(50);
      expect(new Set(samples.map((sample) => sample.actingPlayerId))).toEqual(
        new Set(["player-0", "player-1", "player-2", "player-3", "player-4"])
      );

      for (const sample of samples) {
        validatePlayingSelfPlaySample(sample, sample.seed);
        expect(sample.actingSeatSource).toBe("current-policy");
        expect(sample.rolloutSeatSources.filter((source) => source === "current-policy")).toHaveLength(1);
        expect(sample.rolloutSeatSources[Number(sample.actingPlayerId.at(-1))]).toBe("current-policy");
      }
    });
  });

  it("supports mixed current, rule-based, and frozen ONNX roster seats", async () => {
    await withTempDir(async (directory) => {
      const currentArtifact = await createPlayingPolicyFixture(join(directory, "current"));
      const frozenArtifact = await createPlayingPolicyFixture(join(directory, "frozen"));
      const currentPolicy = await loadPolicyOnnxModel(currentArtifact);
      const frozenPolicy = await loadPolicyOnnxModel(frozenArtifact);
      const output = join(directory, "mixed");

      await generatePlayingSelfPlayDataset({
        outputDirectory: output,
        playingPolicy: currentPolicy,
        playingPolicyArtifact: currentArtifact,
        rolloutRoster: {
          seats: [
            { source: "current-policy" },
            { source: "rule-based" },
            { source: "rule-based" },
            {
              source: "frozen-onnx",
              policy: frozenPolicy,
              artifact: frozenArtifact
            },
            {
              source: "frozen-onnx",
              policy: frozenPolicy,
              artifact: {
                ...frozenArtifact,
                artifactId: "frozen-v740-copy"
              }
            }
          ]
        },
        startSeed: 3,
        gameCount: 1,
        gamesPerShard: 1,
        format: DATASET_FORMAT
      });

      const manifest = await readManifest(output);
      const lines = await readAllShardLines(output, manifest);
      const samples = lines.map((line) => JSON.parse(line) as PlayingSelfPlaySample);

      expect(manifest.rolloutRoster.seats.map((seat) => seat.source)).toEqual([
        "current-policy",
        "rule-based",
        "rule-based",
        "frozen-onnx",
        "frozen-onnx"
      ]);
      expect(manifest.rolloutRoster.seats[3]).toMatchObject({
        source: "frozen-onnx",
        artifactId: "test-playing-policy",
        requestedInferenceDevice: "cpu",
        resolvedInferenceDevice: "cpu",
        executionProvider: "cpu"
      });
      expect(manifest.rolloutRoster.seats[4]).toMatchObject({
        source: "frozen-onnx",
        artifactId: "frozen-v740-copy",
        requestedInferenceDevice: "cpu",
        resolvedInferenceDevice: "cpu",
        executionProvider: "cpu"
      });
      expect(samples).toHaveLength(10);
      expect(samples.every((sample) => sample.actingSeatSource === "current-policy")).toBe(true);
      expect(samples.every((sample) => !sample.rolloutSeatSources.includes("frozen-onnx") ||
        sample.rolloutSeatSources.filter((source) => source === "current-policy").length === 1
      )).toBe(true);
      assertNoHiddenStateFields(lines);
    });
  });

  it("commits rollout results in seed order with bounded in-flight games", async () => {
    await withTempDir(async (directory) => {
      const currentArtifact = await createPlayingPolicyFixture(join(directory, "current"));
      const frozenArtifact = await createPlayingPolicyFixture(join(directory, "frozen"));
      const currentPolicy = await loadPolicyOnnxModel(currentArtifact);
      const frozenPolicy = await loadPolicyOnnxModel(frozenArtifact);
      const serialOutput = join(directory, "serial");
      const parallelOutput = join(directory, "parallel");
      const activeOffsets = new Set<number>();
      let maxInFlight = 0;
      const roster: PlayingSelfPlayRolloutRosterOptions = {
        seats: [
          { source: "current-policy" },
          { source: "rule-based" },
          { source: "frozen-onnx", policy: frozenPolicy, artifact: frozenArtifact },
          { source: "rule-based" },
          { source: "current-policy" }
        ]
      };
      const runner: PlayingSelfPlayGameRunner = {
        runGame: async (request) => {
          activeOffsets.add(request.gameOffset);
          maxInFlight = Math.max(maxInFlight, activeOffsets.size);
          if (request.gameOffset === 0) {
            await sleep(20);
          }
          try {
            return await runPlayingSelfPlayGame({
              seed: request.seed,
              currentPolicy: request.currentPolicy,
              rolloutRoster: request.rolloutRoster,
              temperature: request.temperature,
              maxDecisionSteps: request.maxDecisionSteps
            });
          } finally {
            activeOffsets.delete(request.gameOffset);
          }
        }
      };

      await generatePlayingSelfPlayDataset({
        outputDirectory: serialOutput,
        playingPolicy: currentPolicy,
        playingPolicyArtifact: currentArtifact,
        rolloutRoster: roster,
        startSeed: 21,
        gameCount: 4,
        gamesPerShard: 2,
        format: DATASET_FORMAT,
        temperature: 0.9
      });
      await generatePlayingSelfPlayDataset({
        outputDirectory: parallelOutput,
        playingPolicy: currentPolicy,
        playingPolicyArtifact: currentArtifact,
        rolloutRoster: roster,
        startSeed: 21,
        gameCount: 4,
        gamesPerShard: 2,
        rolloutWorkers: 3,
        temperature: 0.9,
        format: DATASET_FORMAT,
        gameRunner: runner
      });

      expect(maxInFlight).toBeGreaterThan(1);
      expect(maxInFlight).toBeLessThanOrEqual(3);
      await expectDirectoriesToBeByteIdentical(serialOutput, parallelOutput);

      const manifest = await readManifest(parallelOutput);
      const samples = (await readAllShardLines(parallelOutput, manifest)).map((line) =>
        JSON.parse(line) as PlayingSelfPlaySample
      );
      expect(new Set(samples.map((sample) => sample.seed))).toEqual(new Set([21, 22, 23, 24]));
      expect(samples.map((sample) => sample.seed)).toEqual(
        [...samples.map((sample) => sample.seed)].sort((left, right) => left - right)
      );
    });
  });

  it("separates logical rollout concurrency from worker count and writes rollout timing", async () => {
    await withTempDir(async (directory) => {
      const onnxPath = join(directory, "policy.onnx");
      const metadataPath = join(directory, "policy.json");
      const output = join(directory, "logical-concurrency");
      await writeFile(onnxPath, "fake-onnx");
      await writeFile(metadataPath, "fake-metadata");

      let requestCount = 0;
      let resetCount = 0;
      const policy: PlayingSelfPlayPolicy = {
        metadata: { policy: "fake" },
        predictLogits: async () => {
          requestCount += 1;
          return new Float32Array(CARD_COUNT);
        },
        getInferenceStats: () => ({
          requestCount,
          sessionRunCount: Math.ceil(requestCount / 3),
          meanBatchSize: requestCount / Math.ceil(requestCount / 3),
          maxObservedBatchSize: Math.min(3, requestCount),
          batchSizeHistogram: { "3": Math.floor(requestCount / 3) }
        }),
        resetInferenceStats: () => {
          resetCount += 1;
          requestCount = 0;
        }
      };
      const activeOffsets = new Set<number>();
      let maxInFlight = 0;
      const runner: PlayingSelfPlayGameRunner = {
        runGame: async (request) => {
          activeOffsets.add(request.gameOffset);
          maxInFlight = Math.max(maxInFlight, activeOffsets.size);
          if (request.gameOffset === 0) {
            await sleep(20);
          }
          try {
            return await runPlayingSelfPlayGameWithSamples({
              seed: request.seed,
              currentPolicy: request.currentPolicy,
              behaviorPolicyArtifactId: request.behaviorPolicyArtifactId,
              rolloutRoster: request.rolloutRoster,
              temperature: request.temperature,
              maxDecisionSteps: request.maxDecisionSteps
            });
          } finally {
            activeOffsets.delete(request.gameOffset);
          }
        }
      };

      const result = await generatePlayingSelfPlayDataset({
        outputDirectory: output,
        playingPolicy: policy,
        playingPolicyArtifact: { onnxPath, metadataPath },
        startSeed: 31,
        gameCount: 4,
        gamesPerShard: 2,
        rolloutWorkers: 2,
        rolloutConcurrency: 4,
        inferenceMaxBatchSize: 3,
        format: DATASET_FORMAT,
        gameRunner: runner
      });

      expect(resetCount).toBe(1);
      expect(maxInFlight).toBe(4);
      expect(result.rolloutTiming).toMatchObject({
        rolloutWorkers: 2,
        rolloutConcurrency: 4,
        inferenceMaxBatchSize: 3
      });
      expect(result.rolloutTiming.rolloutElapsedSeconds).toBeGreaterThanOrEqual(0);
      expect(result.rolloutTiming.inference.requestCount).toBeGreaterThan(0);
      expect(result.rolloutTiming.inference.sessionRunCount).toBeGreaterThan(0);
    });
  });

  it("propagates worker failures without publishing a completed dataset", async () => {
    await withTempDir(async (directory) => {
      const artifact = await createPlayingPolicyFixture(directory);
      const policy = await loadPolicyOnnxModel(artifact);
      const output = join(directory, "failed-parallel");
      const runner: PlayingSelfPlayGameRunner = {
        runGame: async (request) => {
          if (request.gameOffset === 1) {
            throw new Error("worker failed");
          }
          return runPlayingSelfPlayGame({
            seed: request.seed,
            currentPolicy: request.currentPolicy,
            rolloutRoster: request.rolloutRoster,
            temperature: request.temperature,
            maxDecisionSteps: request.maxDecisionSteps
          });
        }
      };

      await expect(generatePlayingSelfPlayDataset({
        outputDirectory: output,
        playingPolicy: policy,
        playingPolicyArtifact: artifact,
        startSeed: 30,
        gameCount: 3,
        gamesPerShard: 1,
        rolloutWorkers: 2,
        format: DATASET_FORMAT,
        gameRunner: runner
      })).rejects.toThrow("worker failed");

      await expect(stat(output)).rejects.toMatchObject({ code: "ENOENT" });
    });
  });

  it("uses runner-provided samples without parent policy inference", async () => {
    await withTempDir(async (directory) => {
      const artifact = await createPlayingPolicyFixture(directory);
      const workerPolicy = await loadPolicyOnnxModel(artifact);
      const output = join(directory, "worker-samples");
      const parentPredictLogits = vi.fn(async () => {
        throw new Error("parent policy inference should not be called");
      });
      const runner: PlayingSelfPlayGameRunner = {
        runGame: async (request) => runPlayingSelfPlayGameWithSamples({
          seed: request.seed,
          currentPolicy: workerPolicy,
          behaviorPolicyArtifactId: request.behaviorPolicyArtifactId,
          rolloutRoster: request.rolloutRoster,
          temperature: request.temperature,
          maxDecisionSteps: request.maxDecisionSteps
        })
      };

      await generatePlayingSelfPlayDataset({
        outputDirectory: output,
        playingPolicy: {
          metadata: workerPolicy.metadata,
          predictLogits: parentPredictLogits
        },
        playingPolicyArtifact: artifact,
        startSeed: 40,
        gameCount: 2,
        gamesPerShard: 2,
        rolloutWorkers: 2,
        format: DATASET_FORMAT,
        gameRunner: runner
      });

      expect(parentPredictLogits).not.toHaveBeenCalled();
      const manifest = await readManifest(output);
      const samples = (await readAllShardLines(output, manifest)).map((line) =>
        JSON.parse(line) as PlayingSelfPlaySample
      );
      expect(samples.length).toBeGreaterThan(0);
      for (const sample of samples) {
        validatePlayingSelfPlaySample(sample, sample.seed);
        expect(sample.behaviorPolicyArtifactId).toBe("test-playing-policy");
      }
    });
  });

  it("uses runner-provided tensor samples without parent policy inference", async () => {
    await withTempDir(async (directory) => {
      const artifact = await createPlayingPolicyFixture(directory);
      const output = join(directory, "worker-tensor-samples");
      const parentPredictLogits = vi.fn(async () => {
        throw new Error("parent policy inference should not be called");
      });
      const runner: PlayingSelfPlayGameRunner = {
        runGame: async (request) => ({
          seed: request.seed,
          tensorSamples: [createTensorSample(request.seed, 0)],
          rolloutInferenceStats: {
            requestCount: 1,
            sessionRunCount: 1,
            meanBatchSize: 1,
            maxObservedBatchSize: 1,
            batchSizeHistogram: { "1": 1 }
          }
        })
      };

      const result = await generatePlayingSelfPlayDataset({
        outputDirectory: output,
        playingPolicy: {
          metadata: {},
          predictLogits: parentPredictLogits
        },
        playingPolicyArtifact: artifact,
        startSeed: 45,
        gameCount: 1,
        gamesPerShard: 1,
        rolloutWorkers: 2,
        format: PLAYING_SELF_PLAY_BINARY_DATASET_FORMAT,
        gameRunner: runner
      });

      expect(parentPredictLogits).not.toHaveBeenCalled();
      expect(result.manifest.sampleCount).toBe(1);
      expect(result.rolloutTiming.inference).toMatchObject({
        requestCount: 1,
        sessionRunCount: 1,
        meanBatchSize: 1,
        maxObservedBatchSize: 1,
        batchSizeHistogram: { "1": 1 }
      });
      const manifest = await readManifest(output);
      expect(manifest.sampleCount).toBe(1);
      expect(manifest.tensorSchema?.compression).toBe("none");
      expect(manifest.shards[0]?.sampleCount).toBe(1);
    });
  });

  it("assigns rollout roster seats deterministically by seed rotation", () => {
    const roster: PlayingSelfPlayRolloutRosterOptions = {
      seats: [
        { source: "current-policy" },
        { source: "rule-based" },
        { source: "rule-based" },
        { source: "frozen-onnx", policy: fakePolicy(), artifact: fakeArtifact("a") },
        { source: "frozen-onnx", policy: fakePolicy(), artifact: fakeArtifact("b") }
      ]
    };

    expect(assignRolloutRosterForSeed(roster, 0).map((seat) => seat.source)).toEqual([
      "current-policy",
      "rule-based",
      "rule-based",
      "frozen-onnx",
      "frozen-onnx"
    ]);
    expect(assignRolloutRosterForSeed(roster, 1).map((seat) => seat.source)).toEqual([
      "frozen-onnx",
      "current-policy",
      "rule-based",
      "rule-based",
      "frozen-onnx"
    ]);
    expect(assignRolloutRosterForSeed(roster, 6).map((seat) => seat.source)).toEqual(
      assignRolloutRosterForSeed(roster, 1).map((seat) => seat.source)
    );
  });

  it("rejects an existing output directory before writing self-play data", async () => {
    await withTempDir(async (directory) => {
      const artifact = await createPlayingPolicyFixture(directory);
      const policy = await loadPolicyOnnxModel(artifact);
      const output = join(directory, "existing");
      const marker = join(output, "marker.txt");
      await mkdir(output);
      await writeFile(marker, "keep\n", "utf8");

      await expect(generatePlayingSelfPlayDataset({
        outputDirectory: output,
        playingPolicy: policy,
        playingPolicyArtifact: artifact,
        startSeed: 0,
        gameCount: 1,
        gamesPerShard: 1,
        format: DATASET_FORMAT
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
  await writeFile(metadataPath, JSON.stringify(createMetadata()) + "\n", "utf8");

  return {
    onnxPath,
    metadataPath,
    artifactId: "test-playing-policy"
  };
}

function fakePolicy() {
  return {
    metadata: {},
    predictLogits: async () => new Float32Array(CARD_COUNT)
  };
}

function fakeArtifact(artifactId: string) {
  return {
    onnxPath: `${artifactId}.onnx`,
    metadataPath: `${artifactId}.json`,
    artifactId
  };
}

function createTensorSample(seed: number, step: number): PlayingSelfPlayTensorSample {
  const selectedCardIndex = (seed + step) % CARD_COUNT;
  const legalPlayMask = new Uint8Array(CARD_COUNT);
  legalPlayMask[selectedCardIndex] = 1;
  const modelInput = new Float32Array(MODEL_INPUT_FEATURE_COUNT);
  modelInput[0] = seed;
  modelInput[1] = step;

  return {
    sampleType: PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE,
    schemaVersion: 4,
    seed,
    step,
    actingPlayerIndex: 0,
    selectedCardIndex,
    behaviorLogProbability: 0,
    terminalReward: 1,
    selfRoleIndex: 0,
    modelInput,
    legalPlayMask
  };
}

function createMetadata() {
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

async function withTempDir(run: (directory: string) => Promise<void>): Promise<void> {
  const directory = await mkdtemp(join(tmpdir(), "napoleon-playing-self-play-test-"));

  try {
    await run(directory);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}

async function readManifest(outputDirectory: string): Promise<PlayingSelfPlayDatasetManifest> {
  return JSON.parse(await readFile(join(outputDirectory, "manifest.json"), "utf8")) as PlayingSelfPlayDatasetManifest;
}

async function readAllShardLines(
  outputDirectory: string,
  manifest: PlayingSelfPlayDatasetManifest
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

function assertNoHiddenStateFields(lines: readonly string[]): void {
  const forbiddenFields = [
    "actualHands",
    "actualState",
    "unusedCardIds",
    "excludedCardIds",
    "awardedPointCardIds",
    "currentTrickCardIds",
    "completedTrickCardIds",
    "adjutantPlayerId",
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

function sleep(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
