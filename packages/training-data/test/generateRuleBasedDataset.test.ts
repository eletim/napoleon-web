import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { Writable } from "node:stream";
import { describe, expect, it } from "vitest";
import {
  ADJUTANT_ENCODER_SCHEMA_VERSION,
  BIDDING_ENCODER_SCHEMA_VERSION,
  CARD_COUNT,
  CARD_IDS,
  EXCHANGE_ENCODER_SCHEMA_VERSION,
  PLAYING_ENCODER_SCHEMA_VERSION
} from "@napoleon/ai-observation";
import {
  validateAdjutantTrainingSample,
  validateBiddingTrainingSample,
  validateEncodedBeliefTarget,
  validateEncodedPlayingObservation,
  validateExchangeTrainingSample
} from "@napoleon/ai-observation";
import type {
  AdjutantTrainingSample,
  BiddingTrainingSample,
  ExchangeTrainingSample,
  PlayingTrainingSample
} from "@napoleon/ai-observation";
import {
  ADJUTANT_DATASET_SAMPLE_TYPE,
  BIDDING_DATASET_SAMPLE_TYPE,
  calculateCardIdsSha256,
  generateRuleBasedDataset,
  DATASET_GENERATOR_VERSION,
  DATASET_SCHEMA_VERSION,
  DATASET_SAMPLE_TYPE,
  EXCHANGE_DATASET_SAMPLE_TYPE,
  MULTIPHASE_DATASET_GENERATOR_VERSION,
  MULTIPHASE_DATASET_SCHEMA_VERSION,
  PLAYING_DATASET_SAMPLE_TYPE,
  validateDatasetManifest
} from "../src/index.js";
import type { DatasetManifest, DatasetSampleType, TrainingSample } from "../src/index.js";
import { generateRuleBasedDatasetWithDependencies } from "../src/generateRuleBasedDataset.js";
import { createJsonlShardWriter } from "../src/shardWriter.js";

type ParsedTrainingSample = TrainingSample & { sampleType?: DatasetSampleType };

describe("generateRuleBasedDataset", () => {
  it("generates a small dataset with valid shards and manifest", async () => {
    await withTempDir(async (directory) => {
      const output = join(directory, "dataset");
      const result = await generateRuleBasedDataset({
        startSeed: 0,
        gameCount: 3,
        gamesPerShard: 2,
        outputDirectory: output
      });
      const manifest = await readManifest(output);
      const lines = await readAllShardLines(output, manifest);

      expect(result.manifest).toEqual(manifest);
      validateDatasetManifest(manifest);
      expect(manifest.datasetSchemaVersion).toBe(DATASET_SCHEMA_VERSION);
      expect(manifest.generatorVersion).toBe(DATASET_GENERATOR_VERSION);
      expect(manifest.sampleType).toBe(DATASET_SAMPLE_TYPE);
      expect(manifest.playingEncoderSchemaVersion).toBe(PLAYING_ENCODER_SCHEMA_VERSION);
      expect(manifest.gameCount).toBe(3);
      expect(manifest.sampleCount).toBe(lines.length);
      expect(manifest.shardCount).toBe(2);
      expect(manifest.shards[0]).toMatchObject({ startSeed: 0, endSeed: 1, gameCount: 2 });
      expect(manifest.shards[1]).toMatchObject({ startSeed: 2, endSeed: 2, gameCount: 1 });
      expect(manifest.cardIds).toEqual(CARD_IDS);
      expect(manifest.cardIds).toHaveLength(53);
      expect(manifest.cardIdsSha256).toBe(calculateCardIdsSha256());

      const parsedSamples = lines.map((line) => JSON.parse(line) as PlayingTrainingSample);
      expect(parsedSamples).toHaveLength(150);
      expect(parsedSamples.every((sample) => !("sampleType" in sample))).toBe(true);
      assertSampleOrderAndValidity(parsedSamples);

      for (const shard of manifest.shards) {
        const shardPath = join(output, shard.file);
        const file = await readFile(shardPath, "utf8");
        const fileStat = await stat(shardPath);

        expect(file.split("\n").filter(Boolean)).toHaveLength(shard.sampleCount);
        expect(fileStat.size).toBe(shard.byteLength);
        expect(sha256(file)).toBe(shard.sha256);
      }
    });
  });

  it.each([
    PLAYING_DATASET_SAMPLE_TYPE,
    BIDDING_DATASET_SAMPLE_TYPE,
    EXCHANGE_DATASET_SAMPLE_TYPE,
    ADJUTANT_DATASET_SAMPLE_TYPE
  ] as const)("smoke-generates a deterministic %s dataset", async (sampleType) => {
    await withTempDir(async (directory) => {
      const firstOutput = join(directory, "first");
      const secondOutput = join(directory, "second");
      const progressEvents: Array<{
        completedGames: number;
        sampleCount: number;
        currentSeed: number;
      }> = [];
      const options = {
        startSeed: 7,
        gameCount: 2,
        gamesPerShard: 1,
        sampleType,
        outputDirectory: firstOutput,
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

      await generateRuleBasedDataset(options);
      await generateRuleBasedDataset({
        startSeed: options.startSeed,
        gameCount: options.gameCount,
        gamesPerShard: options.gamesPerShard,
        sampleType: options.sampleType,
        outputDirectory: secondOutput
      });

      await expectDirectoriesToBeByteIdentical(firstOutput, secondOutput);

      const manifest = await readManifest(firstOutput);
      const lines = await readAllShardLines(firstOutput, manifest);

      validateDatasetManifest(manifest);
      assertManifestSampleType(manifest, sampleType);
      expect(manifest.startSeed).toBe(7);
      expect(manifest.endSeed).toBe(8);
      expect(manifest.gameCount).toBe(2);
      expect(manifest.shardCount).toBe(2);
      expect(manifest.sampleCount).toBe(lines.length);
      expect(manifest.sampleCount).toBeGreaterThan(0);
      expect(progressEvents).toHaveLength(2);
      expect(progressEvents.map((event) => event.completedGames)).toEqual([1, 2]);
      expect(progressEvents.map((event) => event.currentSeed)).toEqual([7, 8]);
      expect(progressEvents.at(-1)?.sampleCount).toBe(manifest.sampleCount);
      expect(manifest.shards.map((shard) => shard.sampleCount).reduce((a, b) => a + b, 0))
        .toBe(manifest.sampleCount);

      for (const shard of manifest.shards) {
        const shardPath = join(firstOutput, shard.file);
        const file = await readFile(shardPath, "utf8");
        const fileStat = await stat(shardPath);

        expect(file.split("\n").filter(Boolean)).toHaveLength(shard.sampleCount);
        expect(fileStat.size).toBe(shard.byteLength);
        expect(sha256(file)).toBe(shard.sha256);
      }

      const parsedSamples = lines.map((line) => JSON.parse(line) as ParsedTrainingSample);
      expect(new Set(parsedSamples.map((sample) => sample.sampleType ?? PLAYING_DATASET_SAMPLE_TYPE)))
        .toEqual(new Set([sampleType]));
      assertSamplesForType(sampleType, parsedSamples);
    });
  });

  it("generates byte-identical files for the same arguments in different directories", async () => {
    await withTempDir(async (directory) => {
      const firstOutput = join(directory, "first");
      const secondOutput = join(directory, "second");

      await generateRuleBasedDataset({
        startSeed: 0,
        gameCount: 2,
        gamesPerShard: 1,
        outputDirectory: firstOutput
      });
      await generateRuleBasedDataset({
        startSeed: 0,
        gameCount: 2,
        gamesPerShard: 1,
        outputDirectory: secondOutput
      });

      const firstFiles = (await readdir(firstOutput)).sort();
      const secondFiles = (await readdir(secondOutput)).sort();

      expect(firstFiles).toEqual(secondFiles);

      for (const file of firstFiles) {
        expect(await readFile(join(firstOutput, file), "utf8")).toBe(
          await readFile(join(secondOutput, file), "utf8")
        );
      }
    });
  });

  it("keeps shard boundaries at game boundaries", async () => {
    await withTempDir(async (directory) => {
      const output = join(directory, "dataset");

      await generateRuleBasedDataset({
        startSeed: 0,
        gameCount: 5,
        gamesPerShard: 2,
        outputDirectory: output
      });

      const manifest = await readManifest(output);

      expect(manifest.shards.map((shard) => shard.gameCount)).toEqual([2, 2, 1]);
      expect(manifest.shards.map((shard) => [shard.startSeed, shard.endSeed])).toEqual([
        [0, 1],
        [2, 3],
        [4, 4]
      ]);
    });
  });

  it("rejects existing output directories without modifying them", async () => {
    await withTempDir(async (directory) => {
      const output = join(directory, "existing");
      const marker = join(output, "marker.txt");
      await mkdir(output);
      await writeFile(marker, "keep\n", "utf8");

      await expect(generateRuleBasedDataset({
        startSeed: 0,
        gameCount: 1,
        gamesPerShard: 1,
        outputDirectory: output
      })).rejects.toThrow("Output directory already exists");

      expect(await readFile(marker, "utf8")).toBe("keep\n");
    });
  });

  it("removes temporary output and leaves no final output after failure", async () => {
    await withTempDir(async (directory) => {
      const output = join(directory, "dataset");

      await expect(generateRuleBasedDatasetWithDependencies({
        startSeed: 0,
        gameCount: 2,
        gamesPerShard: 1,
        outputDirectory: output,
        runGame: async () => {
          throw new Error("intentional game failure");
        }
      })).rejects.toThrow("intentional game failure");

      await expect(stat(output)).rejects.toMatchObject({ code: "ENOENT" });
      expect(await readdir(directory)).toEqual([]);
    });
  });

  it("propagates asynchronous shard write failures and removes temporary output", async () => {
    await withTempDir(async (directory) => {
      const output = join(directory, "dataset");
      const generation = generateRuleBasedDatasetWithDependencies({
        startSeed: 0,
        gameCount: 1,
        gamesPerShard: 1,
        outputDirectory: output,
        createShardWriter: (shardDirectory, shardIndex, startSeed) =>
          createJsonlShardWriterWithWritable(shardDirectory, shardIndex, startSeed, () => new FailingWriteWritable())
      });

      await expectRejectsWithoutHanging(generation, "intentional stream failure");
      await expect(stat(output)).rejects.toMatchObject({ code: "ENOENT" });
      await expect(stat(join(output, "manifest.json"))).rejects.toMatchObject({ code: "ENOENT" });
      expect(await readdir(directory)).toEqual([]);
    });
  });

  it("propagates shard close failures and removes temporary output", async () => {
    await withTempDir(async (directory) => {
      const output = join(directory, "dataset");
      const generation = generateRuleBasedDatasetWithDependencies({
        startSeed: 0,
        gameCount: 1,
        gamesPerShard: 1,
        outputDirectory: output,
        createShardWriter: (shardDirectory, shardIndex, startSeed) =>
          createJsonlShardWriterWithWritable(shardDirectory, shardIndex, startSeed, () => new FailingFinalWritable())
      });

      await expectRejectsWithoutHanging(generation, "intentional close failure");
      await expect(stat(output)).rejects.toMatchObject({ code: "ENOENT" });
      await expect(stat(join(output, "manifest.json"))).rejects.toMatchObject({ code: "ENOENT" });
      expect(await readdir(directory)).toEqual([]);
    });
  });

  it("rejects without hanging when a shard stream closes without error during backpressure", async () => {
    await withTempDir(async (directory) => {
      const output = join(directory, "dataset");
      const generation = generateRuleBasedDatasetWithDependencies({
        startSeed: 0,
        gameCount: 1,
        gamesPerShard: 1,
        outputDirectory: output,
        createShardWriter: (shardDirectory, shardIndex, startSeed) =>
          createJsonlShardWriterWithWritable(
            shardDirectory,
            shardIndex,
            startSeed,
            () => new PrematureCloseWritable()
          )
      });

      await expectRejectsWithoutHanging(generation, "closed before drain");
      await expect(stat(output)).rejects.toMatchObject({ code: "ENOENT" });
      await expect(stat(join(output, "manifest.json"))).rejects.toMatchObject({ code: "ENOENT" });
      expect(await readdir(directory)).toEqual([]);
    });
  });
});

async function withTempDir(run: (directory: string) => Promise<void>): Promise<void> {
  const directory = await mkdtemp(join(tmpdir(), "napoleon-training-data-test-"));

  try {
    await run(directory);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}

async function expectDirectoriesToBeByteIdentical(
  firstOutput: string,
  secondOutput: string
): Promise<void> {
  const firstFiles = (await readdir(firstOutput)).sort();
  const secondFiles = (await readdir(secondOutput)).sort();

  expect(firstFiles).toEqual(secondFiles);

  for (const file of firstFiles) {
    expect(await readFile(join(firstOutput, file), "utf8")).toBe(
      await readFile(join(secondOutput, file), "utf8")
    );
  }
}

async function readManifest(output: string): Promise<DatasetManifest> {
  return JSON.parse(await readFile(join(output, "manifest.json"), "utf8")) as DatasetManifest;
}

async function readAllShardLines(
  output: string,
  manifest: DatasetManifest
): Promise<readonly string[]> {
  const lines: string[] = [];

  for (const shard of manifest.shards) {
    const file = await readFile(join(output, shard.file), "utf8");
    lines.push(...file.split("\n").filter(Boolean));
  }

  return lines;
}

function assertSampleOrderAndValidity(samples: readonly PlayingTrainingSample[]): void {
  let previousSeed = -1;
  const lastStepBySeed = new Map<number, number>();

  for (const sample of samples) {
    expect(sample.seed).toBeGreaterThanOrEqual(previousSeed);
    previousSeed = sample.seed;

    const lastStep = lastStepBySeed.get(sample.seed) ?? 0;
    expect(sample.step).toBeGreaterThan(lastStep);
    lastStepBySeed.set(sample.seed, sample.step);
    validateEncodedPlayingObservation(sample.observation);
    validateEncodedBeliefTarget(sample.beliefTarget);
    expect(sample.observation.legalPlayMask[sample.actorTarget.selectedCardIndex]).toBe(1);
  }
}

function assertManifestSampleType(
  manifest: DatasetManifest,
  sampleType: DatasetSampleType
): void {
  expect(manifest.sampleType).toBe(sampleType);
  expect(manifest.cardIds).toEqual(CARD_IDS);
  expect(manifest.cardIdsSha256).toBe(calculateCardIdsSha256());

  if (sampleType === PLAYING_DATASET_SAMPLE_TYPE) {
    expect(manifest.datasetSchemaVersion).toBe(DATASET_SCHEMA_VERSION);
    expect(manifest.generatorVersion).toBe(DATASET_GENERATOR_VERSION);
    expect("playingEncoderSchemaVersion" in manifest && manifest.playingEncoderSchemaVersion)
      .toBe(PLAYING_ENCODER_SCHEMA_VERSION);
    return;
  }

  expect(manifest.datasetSchemaVersion).toBe(MULTIPHASE_DATASET_SCHEMA_VERSION);
  expect(manifest.generatorVersion).toBe(MULTIPHASE_DATASET_GENERATOR_VERSION);
  expect("encoderSchemaVersion" in manifest && manifest.encoderSchemaVersion)
    .toBe(expectedEncoderSchemaVersion(sampleType));
}

function assertSamplesForType(
  sampleType: DatasetSampleType,
  samples: readonly ParsedTrainingSample[]
): void {
  switch (sampleType) {
    case PLAYING_DATASET_SAMPLE_TYPE:
      assertSampleOrderAndValidity(samples as readonly PlayingTrainingSample[]);
      return;
    case BIDDING_DATASET_SAMPLE_TYPE:
      samples.forEach((sample) =>
        validateBiddingTrainingSample(sample as BiddingTrainingSample)
      );
      return;
    case EXCHANGE_DATASET_SAMPLE_TYPE:
      samples.forEach((sample) => {
        const exchangeSample = sample as ExchangeTrainingSample;

        validateExchangeTrainingSample(exchangeSample);
        expect(exchangeSample.observation.selfHandMask.reduce((a, b) => a + b, 0)).toBe(13);
        expect(exchangeSample.actorTarget.discardTargetMask.reduce((a, b) => a + b, 0))
          .toBe(3);
      });
      return;
    case ADJUTANT_DATASET_SAMPLE_TYPE:
      samples.forEach((sample) => {
        const adjutantSample = sample as AdjutantTrainingSample;

        validateAdjutantTrainingSample(adjutantSample);
        expect(adjutantSample.observation.legalAdjutantMask).toHaveLength(CARD_COUNT);
        expect(adjutantSample.observation.legalAdjutantMask[adjutantSample.actorTarget])
          .toBe(1);
      });
      return;
  }
}

function expectedEncoderSchemaVersion(sampleType: DatasetSampleType): number {
  switch (sampleType) {
    case PLAYING_DATASET_SAMPLE_TYPE:
      return PLAYING_ENCODER_SCHEMA_VERSION;
    case BIDDING_DATASET_SAMPLE_TYPE:
      return BIDDING_ENCODER_SCHEMA_VERSION;
    case EXCHANGE_DATASET_SAMPLE_TYPE:
      return EXCHANGE_ENCODER_SCHEMA_VERSION;
    case ADJUTANT_DATASET_SAMPLE_TYPE:
      return ADJUTANT_ENCODER_SCHEMA_VERSION;
  }
}

function sha256(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

async function expectRejectsWithoutHanging(
  promise: Promise<unknown>,
  message: string
): Promise<void> {
  let timeout: ReturnType<typeof setTimeout> | null = null;
  const timeoutPromise = new Promise<never>((_, reject) => {
    timeout = setTimeout(() => reject(new Error("dataset generation hung")), 1000);
  });

  try {
    await expect(Promise.race([promise, timeoutPromise])).rejects.toThrow(message);
  } finally {
    if (timeout !== null) {
      clearTimeout(timeout);
    }
  }
}

function createJsonlShardWriterWithWritable(
  directory: string,
  shardIndex: number,
  startSeed: number,
  createWritable: () => Writable
) {
  return createJsonlShardWriter(directory, shardIndex, startSeed, undefined, createWritable);
}

class FailingWriteWritable extends Writable {
  override _write(
    _chunk: Buffer | string,
    _encoding: BufferEncoding,
    callback: (error?: Error | null) => void
  ): void {
    queueMicrotask(() => callback(new Error("intentional stream failure")));
  }
}

class FailingFinalWritable extends Writable {
  override _write(
    _chunk: Buffer | string,
    _encoding: BufferEncoding,
    callback: (error?: Error | null) => void
  ): void {
    queueMicrotask(() => callback(null));
  }

  override _final(callback: (error?: Error | null) => void): void {
    queueMicrotask(() => callback(new Error("intentional close failure")));
  }
}

// Simulates a Writable that is force-destroyed (no error passed) while a
// write is still buffered, e.g. an upstream process killing the file
// descriptor. It never calls its _write callback, so the write stays
// pending and "drain" never fires; a tiny highWaterMark guarantees
// stream.write() reports backpressure on the very first chunk.
class PrematureCloseWritable extends Writable {
  constructor() {
    super({ highWaterMark: 1 });
  }

  override _write(
    _chunk: Buffer | string,
    _encoding: BufferEncoding,
    _callback: (error?: Error | null) => void
  ): void {
    queueMicrotask(() => {
      this.destroy();
    });
  }
}
