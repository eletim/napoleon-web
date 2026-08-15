import type { AutomatedGameRecord } from "@napoleon/ai";
import type {
  AdjutantTrainingSample,
  BiddingTrainingSample,
  ExchangeTrainingSample,
  PlayingTrainingSample,
  PLAYING_ENCODER_SCHEMA_VERSION
} from "@napoleon/ai-observation";
import type {
  DATASET_SAMPLE_TYPES,
  DATASET_FORMAT,
  DATASET_GENERATOR_VERSION,
  DATASET_SAMPLE_TYPE,
  DATASET_SCHEMA_VERSION,
  MULTIPHASE_DATASET_GENERATOR_VERSION,
  MULTIPHASE_DATASET_SCHEMA_VERSION,
  RULE_BASED_AGENT_VERSION
} from "./schema.js";

export type DatasetSampleType = typeof DATASET_SAMPLE_TYPES[number];
export type NonPlayingDatasetSampleType = Exclude<DatasetSampleType, typeof DATASET_SAMPLE_TYPE>;
export type TrainingSample =
  | PlayingTrainingSample
  | BiddingTrainingSample
  | ExchangeTrainingSample
  | AdjutantTrainingSample;

export interface DatasetShardManifest {
  file: string;
  startSeed: number;
  endSeed: number;
  gameCount: number;
  sampleCount: number;
  byteLength: number;
  sha256: string;
}

interface DatasetManifestBase {
  format: typeof DATASET_FORMAT;
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

export interface DatasetManifest extends DatasetManifestBase {
  datasetSchemaVersion: typeof DATASET_SCHEMA_VERSION;
  generatorVersion: typeof DATASET_GENERATOR_VERSION;
  playingEncoderSchemaVersion: typeof PLAYING_ENCODER_SCHEMA_VERSION;
  sampleType: typeof DATASET_SAMPLE_TYPE;
}

export type LegacyPlayingDatasetManifest = DatasetManifest;

export interface MultiphaseDatasetManifest extends DatasetManifestBase {
  datasetSchemaVersion: typeof MULTIPHASE_DATASET_SCHEMA_VERSION;
  generatorVersion: typeof MULTIPHASE_DATASET_GENERATOR_VERSION;
  encoderSchemaVersion: number;
  sampleType: NonPlayingDatasetSampleType;
}

export type RuleBasedDatasetManifest = DatasetManifest | MultiphaseDatasetManifest;

export interface DatasetGenerationProgress {
  completedGames: number;
  totalGames: number;
  sampleCount: number;
  completedShards: number;
  currentSeed: number;
}

interface GenerateRuleBasedDatasetBaseOptions {
  startSeed: number;
  gameCount: number;
  gamesPerShard: number;
  outputDirectory: string;
  onProgress?: (progress: DatasetGenerationProgress) => void;
}

export interface GeneratePlayingRuleBasedDatasetOptions
  extends GenerateRuleBasedDatasetBaseOptions {
  sampleType?: typeof DATASET_SAMPLE_TYPE;
}

export interface GenerateNonPlayingRuleBasedDatasetOptions<
  TSampleType extends NonPlayingDatasetSampleType = NonPlayingDatasetSampleType
> extends GenerateRuleBasedDatasetBaseOptions {
  sampleType: TSampleType;
}

export type GenerateRuleBasedDatasetOptions =
  | GeneratePlayingRuleBasedDatasetOptions
  | GenerateNonPlayingRuleBasedDatasetOptions;

export type DatasetManifestForSampleType<TSampleType extends DatasetSampleType> =
  TSampleType extends typeof DATASET_SAMPLE_TYPE
    ? DatasetManifest
    : MultiphaseDatasetManifest & { sampleType: TSampleType };

export interface GenerateDatasetResult<
  TSampleType extends DatasetSampleType = typeof DATASET_SAMPLE_TYPE
> {
  outputDirectory: string;
  manifest: DatasetManifestForSampleType<TSampleType>;
}

export type GameRunner = (seed: number) => Promise<AutomatedGameRecord>;
export type SampleCreator = (
  record: AutomatedGameRecord
) => readonly TrainingSample[];
export type SampleValidator = (sample: TrainingSample, expectedSeed: number) => void;
export type SampleSerializer<TSample = TrainingSample> = (sample: TSample) => string;
export type CreateShardWriter = (
  directory: string,
  shardIndex: number,
  startSeed: number,
  serializeSample: SampleSerializer
) => import("./shardWriter.js").JsonlShardWriter<TrainingSample>;

export type GenerateRuleBasedDatasetInternalOptions = GenerateRuleBasedDatasetOptions & {
  runGame?: GameRunner;
  createSamples?: SampleCreator;
  validateSample?: SampleValidator;
  serializeSample?: SampleSerializer;
  createShardWriter?: CreateShardWriter;
};
