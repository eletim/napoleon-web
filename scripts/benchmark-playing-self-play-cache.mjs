#!/usr/bin/env node
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { performance } from "node:perf_hooks";
import {
  generatePlayingSelfPlayDataset
} from "../packages/training-data/dist/index.js";
import {
  CARD_COUNT,
  MODEL_INPUT_FEATURE_COUNT
} from "../packages/ai-observation/dist/index.js";

const samplesPerGame = 50;
const defaultSampleCount = 60_000;

const sampleCount = parsePositiveInteger(
  process.env.NAPOLEON_CACHE_BENCH_SAMPLES ?? String(defaultSampleCount),
  "NAPOLEON_CACHE_BENCH_SAMPLES"
);
if (sampleCount % samplesPerGame !== 0) {
  throw new Error(`sample count must be divisible by ${samplesPerGame}.`);
}

const root = resolve(
  process.env.NAPOLEON_CACHE_BENCH_DIR ??
    await mkdtemp(join(tmpdir(), "napoleon-cache-bench-"))
);
const gameCount = sampleCount / samplesPerGame;
const gamesPerShard = parsePositiveInteger(
  process.env.NAPOLEON_CACHE_BENCH_GAMES_PER_SHARD ?? "20",
  "NAPOLEON_CACHE_BENCH_GAMES_PER_SHARD"
);

await mkdir(root, { recursive: true });
const artifact = await createArtifact(root);
const policy = {
  metadata: { benchmark: "synthetic-playing-self-play-cache" },
  runtime: {
    requestedInferenceDevice: "cpu",
    resolvedInferenceDevice: "cpu",
    executionProvider: "cpu"
  },
  predictLogits: async () => new Float32Array(CARD_COUNT)
};

const results = [];
for (const compression of ["none", "gzip"]) {
  const outputDirectory = join(root, compression);
  await rm(outputDirectory, { recursive: true, force: true });
  const started = performance.now();
  const result = await generatePlayingSelfPlayDataset({
    outputDirectory,
    playingPolicy: policy,
    playingPolicyArtifact: artifact,
    startSeed: 0,
    gameCount,
    gamesPerShard,
    binaryCompression: compression,
    gameRunner: createSyntheticRunner()
  });
  const elapsedSeconds = (performance.now() - started) / 1000;
  const shardByteLength = result.manifest.shards.reduce(
    (total, shard) => total + shard.byteLength,
    0
  );
  results.push({
    compression,
    outputDirectory,
    sampleCount: result.manifest.sampleCount,
    shardCount: result.manifest.shardCount,
    shardByteLength,
    elapsedSeconds
  });
}

console.log(JSON.stringify({ root, samplesPerGame, gameCount, gamesPerShard, results }, null, 2));

function createSyntheticRunner() {
  return {
    async runGame(request) {
      const tensorSamples = [];
      for (let step = 0; step < samplesPerGame; step += 1) {
        const selectedCardIndex = (request.seed + step) % CARD_COUNT;
        const legalPlayMask = new Uint8Array(CARD_COUNT);
        legalPlayMask[selectedCardIndex] = 1;
        const modelInput = new Float32Array(MODEL_INPUT_FEATURE_COUNT);
        modelInput[0] = request.seed;
        modelInput[1] = step;
        modelInput[2 + selectedCardIndex] = 1;
        tensorSamples.push({
          sampleType: "playing-self-play-sample",
          schemaVersion: 4,
          seed: request.seed,
          step,
          actingPlayerIndex: step % 5,
          selectedCardIndex,
          behaviorLogProbability: 0,
          terminalReward: step % 2 === 0 ? 1 : -1,
          selfRoleIndex: step % 4,
          modelInput,
          legalPlayMask
        });
      }
      return { seed: request.seed, tensorSamples };
    }
  };
}

async function createArtifact(rootDirectory) {
  const onnxPath = join(rootDirectory, "synthetic.onnx");
  const metadataPath = join(rootDirectory, "synthetic.json");
  await writeFile(onnxPath, "synthetic-onnx\n");
  await writeFile(metadataPath, JSON.stringify({ benchmark: "synthetic" }) + "\n");
  await readFile(onnxPath);
  return { onnxPath, metadataPath, artifactId: "synthetic-cache-benchmark" };
}

function parsePositiveInteger(value, name) {
  if (!/^[1-9][0-9]*$/.test(value)) {
    throw new Error(`${name} must be a positive integer.`);
  }
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed)) {
    throw new Error(`${name} must be a safe integer.`);
  }
  return parsed;
}
