import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { Writable } from "node:stream";
import { describe, expect, it } from "vitest";
import { CARD_IDS } from "@napoleon/ai-observation";
import {
  validateEncodedBeliefTarget,
  validateEncodedPlayingObservation
} from "@napoleon/ai-observation";
import type { PlayingTrainingSample } from "@napoleon/ai-observation";
import {
  calculateCardIdsSha256,
  generateRuleBasedDataset,
  validateDatasetManifest
} from "../src/index.js";
import type { DatasetManifest } from "../src/index.js";
import { generateRuleBasedDatasetWithDependencies } from "../src/generateRuleBasedDataset.js";
import { createJsonlShardWriter } from "../src/shardWriter.js";

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
});

async function withTempDir(run: (directory: string) => Promise<void>): Promise<void> {
  const directory = await mkdtemp(join(tmpdir(), "napoleon-training-data-test-"));

  try {
    await run(directory);
  } finally {
    await rm(directory, { recursive: true, force: true });
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
  return createJsonlShardWriter(directory, shardIndex, startSeed, createWritable);
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
