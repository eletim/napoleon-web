import { describe, expect, it } from "vitest";
import { RuleBasedAgent, runAutomatedGame } from "@napoleon/ai";
import {
  CARD_IDS,
  PLAYING_ENCODER_SCHEMA_VERSION,
  createPlayingTrainingSamples
} from "@napoleon/ai-observation";
import {
  BIDDING_DATASET_SAMPLE_TYPE,
  calculateCardIdsSha256,
  DATASET_FORMAT,
  DATASET_GENERATOR_VERSION,
  DATASET_SAMPLE_TYPE,
  DATASET_SCHEMA_VERSION,
  MAX_SHARD_COUNT,
  MULTIPHASE_DATASET_GENERATOR_VERSION,
  MULTIPHASE_DATASET_SCHEMA_VERSION,
  RULE_BASED_AGENT_VERSION,
  shardFileName,
  validateDatasetManifest,
  validateGenerationOptions,
  validatePlayingTrainingSample
} from "../src/index.js";
import type {
  DatasetManifest,
  GenerateRuleBasedDatasetOptions,
  RuleBasedDatasetManifest
} from "../src/index.js";

describe("validation", () => {
  it("validates generation options and seed ranges", () => {
    expect(() => validateGenerationOptions({
      startSeed: 0,
      gameCount: 1,
      gamesPerShard: 1,
      outputDirectory: "out"
    })).not.toThrow();

    expect(() => validateGenerationOptions({
      startSeed: 4294967295,
      gameCount: 2,
      gamesPerShard: 1,
      outputDirectory: "out"
    })).toThrow("Seed range exceeds uint32");

    expect(() => validateGenerationOptions({
      startSeed: 0,
      gameCount: MAX_SHARD_COUNT + 1,
      gamesPerShard: 1,
      outputDirectory: "out"
    })).toThrow(`exceeding the maximum ${MAX_SHARD_COUNT}`);

    expect(() => validateGenerationOptions({
      startSeed: 0,
      gameCount: 1,
      gamesPerShard: 1,
      outputDirectory: "out",
      sampleType: "bogus-training-sample"
    } as unknown as GenerateRuleBasedDatasetOptions)).toThrow("Unsupported dataset sampleType");
  });

  it("rejects invalid shard file indexes", () => {
    expect(shardFileName(MAX_SHARD_COUNT - 1)).toBe("shard-99999.jsonl");
    expect(() => shardFileName(-1)).toThrow("shardIndex");
    expect(() => shardFileName(1.5)).toThrow("shardIndex");
    expect(() => shardFileName(Number.NaN)).toThrow("shardIndex");
    expect(() => shardFileName(Number.POSITIVE_INFINITY)).toThrow("shardIndex");
    expect(() => shardFileName(MAX_SHARD_COUNT)).toThrow("shardIndex");
    expect(() => shardFileName(100000)).toThrow("shardIndex");
  });

  it.each([
    ["gamesPerShard = 0", (manifest: DatasetManifest) => {
      manifest.gamesPerShard = 0;
    }],
    ["gamesPerShard = 1.5", (manifest: DatasetManifest) => {
      manifest.gamesPerShard = 1.5;
    }],
    ["startSeed = -1", (manifest: DatasetManifest) => {
      manifest.startSeed = -1;
    }],
    ["endSeed = 4294967296", (manifest: DatasetManifest) => {
      manifest.endSeed = 4294967296;
    }],
    ["shard startSeed = -1", (manifest: DatasetManifest) => {
      manifest.shards[0].startSeed = -1;
    }],
    ["shard endSeed = 4294967296", (manifest: DatasetManifest) => {
      manifest.shards[0].endSeed = 4294967296;
    }],
    ["sampleCount = 0", (manifest: DatasetManifest) => {
      manifest.sampleCount = 0;
    }],
    ["shardCount = 0", (manifest: DatasetManifest) => {
      manifest.shardCount = 0;
    }],
    ["shardCount !== shards.length", (manifest: DatasetManifest) => {
      manifest.shards = manifest.shards.slice(0, 1);
    }],
    ["shardCount !== ceil(gameCount / gamesPerShard)", (manifest: DatasetManifest) => {
      manifest.shardCount = 3;
    }],
    ["non-final shard gameCount is less than gamesPerShard", (manifest: DatasetManifest) => {
      manifest.shards = [
        { ...manifest.shards[0], endSeed: 0, gameCount: 1 },
        { ...manifest.shards[1], startSeed: 1, endSeed: 2, gameCount: 2 }
      ];
    }],
    ["non-final shard gameCount is greater than gamesPerShard", (manifest: DatasetManifest) => {
      manifest.gameCount = 4;
      manifest.endSeed = 3;
      manifest.gamesPerShard = 2;
      manifest.shards = [
        { ...manifest.shards[0], startSeed: 0, endSeed: 2, gameCount: 3 },
        { ...manifest.shards[1], startSeed: 3, endSeed: 3, gameCount: 1 }
      ];
    }],
    ["final shard gameCount is greater than gamesPerShard", (manifest: DatasetManifest) => {
      manifest.gameCount = 4;
      manifest.endSeed = 3;
      manifest.gamesPerShard = 2;
      manifest.shards = [
        { ...manifest.shards[0], startSeed: 0, endSeed: 1, gameCount: 2 },
        { ...manifest.shards[1], startSeed: 2, endSeed: 4, gameCount: 3 }
      ];
    }],
    ["seed-contiguous shards violate gamesPerShard layout", (manifest: DatasetManifest) => {
      manifest.shards = [
        { ...manifest.shards[0], startSeed: 0, endSeed: 0, gameCount: 1 },
        { ...manifest.shards[1], startSeed: 1, endSeed: 2, gameCount: 2 }
      ];
    }]
  ])("rejects invalid manifest: %s", (_label, mutate) => {
    const manifest = validManifest();
    mutate(manifest);

    expect(() => validateDatasetManifest(manifest)).toThrow();
  });

  it("validates a v2 non-playing manifest with an explicit encoder schema version", () => {
    const manifest = validV2Manifest();

    expect(() => validateDatasetManifest(manifest)).not.toThrow();
  });

  it.each([
    ["v1 manifest with a non-playing sampleType", () => ({
      ...validManifest(),
      sampleType: BIDDING_DATASET_SAMPLE_TYPE
    })],
    ["v2 manifest with a playing sampleType", () => ({
      ...validV2Manifest(),
      sampleType: DATASET_SAMPLE_TYPE
    })],
    ["v2 manifest with a mismatched encoderSchemaVersion", () => ({
      ...validV2Manifest(),
      encoderSchemaVersion: 2
    })]
  ])("rejects mixed manifest schema identity: %s", (_label, createManifest) => {
    expect(() =>
      validateDatasetManifest(createManifest() as unknown as RuleBasedDatasetManifest)
    ).toThrow();
  });

  it("rejects samples with non-json-safe values", async () => {
    const record = await runAutomatedGame({
      seed: 0,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });
    const sample = structuredClone(createPlayingTrainingSamples(record)[0]);

    sample.observation.trickNumber = Number.NaN;

    expect(() => validatePlayingTrainingSample(sample, 0)).toThrow("NaN or Infinity");
  });
});

function validManifest(): DatasetManifest {
  return {
    datasetSchemaVersion: DATASET_SCHEMA_VERSION,
    generatorVersion: DATASET_GENERATOR_VERSION,
    playingEncoderSchemaVersion: PLAYING_ENCODER_SCHEMA_VERSION,
    format: DATASET_FORMAT,
    sampleType: DATASET_SAMPLE_TYPE,
    agent: {
      type: "rule-based",
      version: RULE_BASED_AGENT_VERSION
    },
    startSeed: 0,
    endSeed: 2,
    gameCount: 3,
    sampleCount: 30,
    gamesPerShard: 2,
    shardCount: 2,
    playerCount: 5,
    cardCount: 53,
    cardIds: CARD_IDS,
    cardIdsSha256: calculateCardIdsSha256(),
    shards: [
      {
        file: "shard-00000.jsonl",
        startSeed: 0,
        endSeed: 1,
        gameCount: 2,
        sampleCount: 20,
        byteLength: 200,
        sha256: "a".repeat(64)
      },
      {
        file: "shard-00001.jsonl",
        startSeed: 2,
        endSeed: 2,
        gameCount: 1,
        sampleCount: 10,
        byteLength: 100,
        sha256: "b".repeat(64)
      }
    ]
  };
}

function validV2Manifest(): RuleBasedDatasetManifest {
  const manifest: RuleBasedDatasetManifest = {
    ...validManifest(),
    datasetSchemaVersion: MULTIPHASE_DATASET_SCHEMA_VERSION,
    generatorVersion: MULTIPHASE_DATASET_GENERATOR_VERSION,
    encoderSchemaVersion: 1,
    sampleType: BIDDING_DATASET_SAMPLE_TYPE
  };

  delete (manifest as { playingEncoderSchemaVersion?: number }).playingEncoderSchemaVersion;

  return manifest;
}
