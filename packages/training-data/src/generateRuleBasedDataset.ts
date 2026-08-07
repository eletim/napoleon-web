import { mkdtemp, mkdir, rename, rm, stat, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import {
  ADJUTANT_ENCODER_SCHEMA_VERSION,
  BIDDING_ENCODER_SCHEMA_VERSION,
  CARD_IDS,
  CARD_COUNT,
  EXCHANGE_ENCODER_SCHEMA_VERSION,
  PLAYER_COUNT,
  type PlayingTrainingSample,
  PLAYING_ENCODER_SCHEMA_VERSION
} from "@napoleon/ai-observation";
import { RuleBasedAgent, runAutomatedGame } from "@napoleon/ai";
import type { AutomatedGameRecord } from "@napoleon/ai";
import {
  createAdjutantTrainingSamples,
  createBiddingTrainingSamples,
  createExchangeTrainingSamples,
  createPlayingTrainingSamples
} from "@napoleon/ai-observation";
import {
  ADJUTANT_DATASET_SAMPLE_TYPE,
  BIDDING_DATASET_SAMPLE_TYPE,
  DATASET_FORMAT,
  DATASET_GENERATOR_VERSION,
  DATASET_SAMPLE_TYPE,
  DATASET_SCHEMA_VERSION,
  EXCHANGE_DATASET_SAMPLE_TYPE,
  MULTIPHASE_DATASET_GENERATOR_VERSION,
  MULTIPHASE_DATASET_SCHEMA_VERSION,
  PLAYING_DATASET_SAMPLE_TYPE,
  RULE_BASED_AGENT_VERSION
} from "./schema.js";
import type {
  DatasetShardManifest,
  DatasetSampleType,
  GenerateDatasetResult,
  GenerateNonPlayingRuleBasedDatasetOptions,
  GeneratePlayingRuleBasedDatasetOptions,
  GenerateRuleBasedDatasetInternalOptions,
  GenerateRuleBasedDatasetOptions,
  NonPlayingDatasetSampleType,
  RuleBasedDatasetManifest,
  SampleCreator,
  SampleSerializer,
  SampleValidator
} from "./types.js";
import { createJsonlShardWriter, type JsonlShardWriter } from "./shardWriter.js";
import {
  calculateCardIdsSha256,
  serializeManifest,
  serializePlayingTrainingSample,
  serializeTrainingSample
} from "./serialization.js";
import {
  validateDatasetManifest,
  validateGenerationOptions,
  validateTrainingSample
} from "./validation.js";

interface SampleGenerationSpec {
  sampleType: DatasetSampleType;
  encoderSchemaVersion: 1;
  createSamples: SampleCreator;
  validateSample: SampleValidator;
  serializeSample: SampleSerializer;
}

const sampleGenerationSpecs: Record<DatasetSampleType, SampleGenerationSpec> = {
  [PLAYING_DATASET_SAMPLE_TYPE]: {
    sampleType: PLAYING_DATASET_SAMPLE_TYPE,
    encoderSchemaVersion: PLAYING_ENCODER_SCHEMA_VERSION,
    createSamples: createPlayingTrainingSamples,
    validateSample: (sample, expectedSeed) =>
      validateTrainingSample(sample, expectedSeed, PLAYING_DATASET_SAMPLE_TYPE),
    serializeSample: (sample) => serializePlayingTrainingSample(sample as PlayingTrainingSample)
  },
  [BIDDING_DATASET_SAMPLE_TYPE]: {
    sampleType: BIDDING_DATASET_SAMPLE_TYPE,
    encoderSchemaVersion: BIDDING_ENCODER_SCHEMA_VERSION,
    createSamples: createBiddingTrainingSamples,
    validateSample: (sample, expectedSeed) =>
      validateTrainingSample(sample, expectedSeed, BIDDING_DATASET_SAMPLE_TYPE),
    serializeSample: serializeTrainingSample
  },
  [EXCHANGE_DATASET_SAMPLE_TYPE]: {
    sampleType: EXCHANGE_DATASET_SAMPLE_TYPE,
    encoderSchemaVersion: EXCHANGE_ENCODER_SCHEMA_VERSION,
    createSamples: createExchangeTrainingSamples,
    validateSample: (sample, expectedSeed) =>
      validateTrainingSample(sample, expectedSeed, EXCHANGE_DATASET_SAMPLE_TYPE),
    serializeSample: serializeTrainingSample
  },
  [ADJUTANT_DATASET_SAMPLE_TYPE]: {
    sampleType: ADJUTANT_DATASET_SAMPLE_TYPE,
    encoderSchemaVersion: ADJUTANT_ENCODER_SCHEMA_VERSION,
    createSamples: createAdjutantTrainingSamples,
    validateSample: (sample, expectedSeed) =>
      validateTrainingSample(sample, expectedSeed, ADJUTANT_DATASET_SAMPLE_TYPE),
    serializeSample: serializeTrainingSample
  }
};

export async function generateRuleBasedDataset(
  options: GeneratePlayingRuleBasedDatasetOptions
): Promise<GenerateDatasetResult<typeof DATASET_SAMPLE_TYPE>>;
export async function generateRuleBasedDataset<TSampleType extends NonPlayingDatasetSampleType>(
  options: GenerateNonPlayingRuleBasedDatasetOptions<TSampleType>
): Promise<GenerateDatasetResult<TSampleType>>;
export async function generateRuleBasedDataset(
  options: GenerateRuleBasedDatasetOptions
): Promise<GenerateDatasetResult<DatasetSampleType>> {
  return generateRuleBasedDatasetWithDependencies(options);
}

export async function generateRuleBasedDatasetWithDependencies(
  options: GenerateRuleBasedDatasetInternalOptions
): Promise<GenerateDatasetResult<DatasetSampleType>> {
  validateGenerationOptions(options);

  const outputDirectory = resolve(options.outputDirectory);
  await ensureOutputDoesNotExist(outputDirectory);
  await mkdir(dirname(outputDirectory), { recursive: true });

  const tempDirectory = await mkdtemp(
    join(dirname(outputDirectory), `.${basenameForTemp(outputDirectory)}.tmp-`)
  );
  const spec = sampleGenerationSpecs[options.sampleType ?? DATASET_SAMPLE_TYPE];
  const runGame = options.runGame ?? runRuleBasedGame;
  const createSamples = options.createSamples ?? spec.createSamples;
  const validateSample = options.validateSample ?? spec.validateSample;
  const serializeSample = options.serializeSample ?? spec.serializeSample;
  const createShardWriter = options.createShardWriter ?? createJsonlShardWriter;
  let activeShard: JsonlShardWriter | null = null;

  try {
    const shards: DatasetShardManifest[] = [];
    let totalSampleCount = 0;
    let shardGameCount = 0;

    for (let gameOffset = 0; gameOffset < options.gameCount; gameOffset += 1) {
      const seed = options.startSeed + gameOffset;

      if (activeShard === null) {
        activeShard = createShardWriter(tempDirectory, shards.length, seed, serializeSample);
        shardGameCount = 0;
      }

      const record = await runGame(seed);
      const samples = createSamples(record);

      for (const sample of samples) {
        validateSample(sample, seed);
        await activeShard.writeSample(sample);
      }

      totalSampleCount += samples.length;
      shardGameCount += 1;

      const shardIsComplete =
        shardGameCount === options.gamesPerShard || gameOffset === options.gameCount - 1;

      if (shardIsComplete) {
        const completedShard = await activeShard.close(seed, shardGameCount);
        shards.push(completedShard);
        activeShard = null;
      }

      options.onProgress?.({
        completedGames: gameOffset + 1,
        totalGames: options.gameCount,
        sampleCount: totalSampleCount,
        completedShards: shards.length,
        currentSeed: seed
      });
    }

    const manifest = createManifest({
      options,
      spec,
      sampleCount: totalSampleCount,
      shards
    });

    validateDatasetManifest(manifest);
    await writeFile(join(tempDirectory, "manifest.json"), serializeManifest(manifest), "utf8");
    await rename(tempDirectory, outputDirectory);

    return {
      outputDirectory: options.outputDirectory,
      manifest
    };
  } catch (error) {
    if (activeShard !== null) {
      await activeShard.abort().catch(() => undefined);
    }

    await rm(tempDirectory, { recursive: true, force: true }).catch((cleanupError: unknown) => {
      const message = cleanupError instanceof Error ? cleanupError.message : String(cleanupError);

      if (error instanceof Error) {
        error.message = `${error.message} Cleanup failed: ${message}`;
      }
    });

    throw error;
  }
}

async function runRuleBasedGame(seed: number): Promise<AutomatedGameRecord> {
  return runAutomatedGame({
    seed,
    createAgent: ({ rng }) => new RuleBasedAgent(rng)
  });
}

function createManifest(input: {
  options: GenerateRuleBasedDatasetOptions;
  spec: SampleGenerationSpec;
  sampleCount: number;
  shards: readonly DatasetShardManifest[];
}): RuleBasedDatasetManifest {
  const endSeed = input.options.startSeed + input.options.gameCount - 1;

  if (input.spec.sampleType === DATASET_SAMPLE_TYPE) {
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
      startSeed: input.options.startSeed,
      endSeed,
      gameCount: input.options.gameCount,
      sampleCount: input.sampleCount,
      gamesPerShard: input.options.gamesPerShard,
      shardCount: input.shards.length,
      playerCount: PLAYER_COUNT,
      cardCount: CARD_COUNT,
      cardIds: CARD_IDS,
      cardIdsSha256: calculateCardIdsSha256(),
      shards: input.shards
    };
  }

  return {
    datasetSchemaVersion: MULTIPHASE_DATASET_SCHEMA_VERSION,
    generatorVersion: MULTIPHASE_DATASET_GENERATOR_VERSION,
    encoderSchemaVersion: input.spec.encoderSchemaVersion,
    format: DATASET_FORMAT,
    sampleType: input.spec.sampleType,
    agent: {
      type: "rule-based",
      version: RULE_BASED_AGENT_VERSION
    },
    startSeed: input.options.startSeed,
    endSeed,
    gameCount: input.options.gameCount,
    sampleCount: input.sampleCount,
    gamesPerShard: input.options.gamesPerShard,
    shardCount: input.shards.length,
    playerCount: PLAYER_COUNT,
    cardCount: CARD_COUNT,
    cardIds: CARD_IDS,
    cardIdsSha256: calculateCardIdsSha256(),
    shards: input.shards
  };
}

async function ensureOutputDoesNotExist(outputDirectory: string): Promise<void> {
  try {
    await stat(outputDirectory);
  } catch (error) {
    if (isNotFoundError(error)) {
      return;
    }

    throw error;
  }

  throw new Error(`Output directory already exists: ${outputDirectory}`);
}

function isNotFoundError(error: unknown): boolean {
  return typeof error === "object" &&
    error !== null &&
    "code" in error &&
    error.code === "ENOENT";
}

function basenameForTemp(outputDirectory: string): string {
  const parts = outputDirectory.split(/[\\/]/);

  return parts.at(-1) ?? "dataset";
}
