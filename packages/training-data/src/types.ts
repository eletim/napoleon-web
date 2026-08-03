import type { AutomatedGameRecord } from "@napoleon/ai";
import type { PlayingTrainingSample } from "@napoleon/ai-observation";
import type {
  DATASET_FORMAT,
  DATASET_GENERATOR_VERSION,
  DATASET_SAMPLE_TYPE,
  DATASET_SCHEMA_VERSION,
  RULE_BASED_AGENT_VERSION
} from "./schema.js";

export interface DatasetShardManifest {
  file: string;
  startSeed: number;
  endSeed: number;
  gameCount: number;
  sampleCount: number;
  byteLength: number;
  sha256: string;
}

export interface DatasetManifest {
  datasetSchemaVersion: typeof DATASET_SCHEMA_VERSION;
  generatorVersion: typeof DATASET_GENERATOR_VERSION;
  playingEncoderSchemaVersion: 1;
  format: typeof DATASET_FORMAT;
  sampleType: typeof DATASET_SAMPLE_TYPE;
  agent: {
    type: "rule-based";
    version: typeof RULE_BASED_AGENT_VERSION;
  };
  startSeed: number;
  endSeed: number;
  gameCount: number;
  sampleCount: number;
  gamesPerShard: number;
  shardCount: number;
  playerCount: 5;
  cardCount: 53;
  cardIds: readonly string[];
  cardIdsSha256: string;
  shards: readonly DatasetShardManifest[];
}

export interface DatasetGenerationProgress {
  completedGames: number;
  totalGames: number;
  sampleCount: number;
  completedShards: number;
  currentSeed: number;
}

export interface GenerateRuleBasedDatasetOptions {
  startSeed: number;
  gameCount: number;
  gamesPerShard: number;
  outputDirectory: string;
  onProgress?: (progress: DatasetGenerationProgress) => void;
}

export interface GenerateDatasetResult {
  outputDirectory: string;
  manifest: DatasetManifest;
}

export type GameRunner = (seed: number) => Promise<AutomatedGameRecord>;
export type SampleCreator = (
  record: AutomatedGameRecord
) => readonly PlayingTrainingSample[];

export interface GenerateRuleBasedDatasetInternalOptions
  extends GenerateRuleBasedDatasetOptions {
  runGame?: GameRunner;
  createSamples?: SampleCreator;
}
