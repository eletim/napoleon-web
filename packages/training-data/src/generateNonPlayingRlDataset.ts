import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { RuleBasedAgent, runAutomatedGame } from "@napoleon/ai";
import type {
  ActualCardState,
  Agent,
  AutomatedGameRecord,
  PlayerObservation
} from "@napoleon/ai";
import type { GameAction, GameResult, PlayerId, WinningTeam } from "@napoleon/game-core";
import {
  BIDDING_ACTION_COUNT,
  BIDDING_ENCODER_SCHEMA_VERSION,
  BIDDING_MODEL_INPUT_FEATURE_COUNT,
  BIDDING_MODEL_INPUT_SCHEMA_VERSION,
  ADJUTANT_ENCODER_SCHEMA_VERSION,
  ADJUTANT_MODEL_INPUT_FEATURE_COUNT,
  ADJUTANT_MODEL_INPUT_SCHEMA_VERSION,
  CARD_COUNT,
  CARD_IDS,
  EXCHANGE_ENCODER_SCHEMA_VERSION,
  EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
  EXCHANGE_MODEL_INPUT_SCHEMA_VERSION,
  MODEL_INPUT_FEATURE_COUNT,
  MODEL_INPUT_SCHEMA_VERSION,
  PLAYER_COUNT,
  createAdjutantModelInput,
  createBiddingModelInput,
  createExchangeModelInput,
  createPlayingModelInput,
  createRelativePlayerOrder,
  decodeAdjutantAction,
  decodeBiddingAction,
  encodeAdjutantObservation,
  encodeBiddingObservation,
  encodeBiddingHistoryFromPublicActions,
  encodeExchangeObservation,
  encodePlayingObservation,
  getCardId
} from "@napoleon/ai-observation";
import type { EncodedExchangeObservation } from "@napoleon/ai-observation";
import {
  DATASET_FORMAT,
  MAX_SHARD_COUNT,
  RULE_BASED_AGENT_VERSION,
  UINT32_MAX
} from "./schema.js";
import type { DatasetGenerationProgress, DatasetShardManifest } from "./types.js";
import { createJsonlShardWriter, type JsonlShardWriter } from "./shardWriter.js";
import { calculateCardIdsSha256, serializeManifest } from "./serialization.js";

export const NON_PLAYING_BIDDING_RL_DATASET_SAMPLE_TYPE = "non-playing-bidding-rl-sample" as const;
export const NON_PLAYING_ADJUTANT_RL_DATASET_SAMPLE_TYPE = "non-playing-adjutant-rl-sample" as const;
export const NON_PLAYING_EXCHANGE_RL_DATASET_SAMPLE_TYPE = "non-playing-exchange-rl-sample" as const;
export const NON_PLAYING_RL_DATASET_SAMPLE_TYPE = NON_PLAYING_BIDDING_RL_DATASET_SAMPLE_TYPE;
export const NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION = 1 as const;
export const NON_PLAYING_RL_DATASET_SCHEMA_VERSION = 1 as const;
export const NON_PLAYING_RL_DATASET_GENERATOR_VERSION = 1 as const;
export const NON_PLAYING_RL_REWARD_TYPE = "non-playing-terminal-role-reward" as const;
export const NON_PLAYING_RL_REWARD_VERSION = 1 as const;
export const NON_PLAYING_RL_REWARD_ID = "non-playing-terminal-role-reward-v1" as const;
export const NON_PLAYING_RL_SAMPLING_ALGORITHM = "masked-categorical" as const;
export const DEFAULT_NON_PLAYING_RL_TEMPERATURE = 1.0 as const;
export const NON_PLAYING_BIDDING_RL_PHASE_SCOPE = "bidding-only" as const;
export const NON_PLAYING_ADJUTANT_RL_PHASE_SCOPE = "adjutant-only" as const;
export const NON_PLAYING_EXCHANGE_RL_PHASE_SCOPE = "exchange-only" as const;
export const NON_PLAYING_RL_PHASE_SCOPE = NON_PLAYING_BIDDING_RL_PHASE_SCOPE;
export const NON_PLAYING_EXCHANGE_RL_DECISION_MODE = "sequential-card-v1" as const;

export type NonPlayingBiddingRlRole =
  | "napoleon"
  | "adjutant"
  | "citizen"
  | "napoleon-adjutant";

export interface NonPlayingBiddingRlPolicy {
  metadata: unknown;
  runtime?: PolicyRuntimeInfo;
  predictLogits: (modelInput: Float32Array | readonly number[]) => Promise<Float32Array>;
}

export interface NonPlayingAdjutantRlPolicy {
  metadata: unknown;
  runtime?: PolicyRuntimeInfo;
  predictLogits: (modelInput: Float32Array | readonly number[]) => Promise<Float32Array>;
}

export interface NonPlayingExchangeRlPolicy {
  metadata: unknown;
  runtime?: PolicyRuntimeInfo;
  predictLogits: (modelInput: Float32Array | readonly number[]) => Promise<Float32Array>;
}

export interface FixedPlayingPolicy {
  metadata: unknown;
  runtime?: PolicyRuntimeInfo;
  predictLogits: (modelInput: Float32Array | readonly number[]) => Promise<Float32Array>;
}

export interface PolicyRuntimeInfo {
  requestedInferenceDevice?: "cpu" | "auto" | "cuda";
  resolvedInferenceDevice?: "cpu" | "cuda";
  executionProvider?: "cpu" | "cuda";
}

export interface NonPlayingRlPolicyArtifactOptions {
  onnxPath: string;
  metadataPath: string;
  artifactId?: string;
}

export interface NonPlayingBiddingRlOutcome {
  winner: WinningTeam;
  targetPointCards: number;
  napoleonPlayerId: PlayerId;
  actingPlayerRole: NonPlayingBiddingRlRole;
}

export interface NonPlayingBiddingRlSample {
  sampleType: typeof NON_PLAYING_BIDDING_RL_DATASET_SAMPLE_TYPE;
  schemaVersion: typeof NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION;
  seed: number;
  step: number;
  phase: "bidding";
  actingPlayerId: PlayerId;
  actingPlayerIndex: number;
  relativePlayerIds: readonly PlayerId[];
  modelInput: readonly number[];
  legalBidMask: readonly number[];
  selectedActionIndex: number;
  behaviorLogProbability: number;
  terminalReward: number;
  outcome: NonPlayingBiddingRlOutcome;
}

export interface NonPlayingAdjutantRlSample {
  sampleType: typeof NON_PLAYING_ADJUTANT_RL_DATASET_SAMPLE_TYPE;
  schemaVersion: typeof NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION;
  seed: number;
  step: number;
  phase: "choosing-adjutant";
  actingPlayerId: PlayerId;
  actingPlayerIndex: number;
  relativePlayerIds: readonly PlayerId[];
  modelInput: readonly number[];
  legalAdjutantMask: readonly number[];
  selectedActionIndex: number;
  behaviorLogProbability: number;
  terminalReward: number;
  outcome: NonPlayingBiddingRlOutcome;
}

export interface NonPlayingExchangeRlSample {
  sampleType: typeof NON_PLAYING_EXCHANGE_RL_DATASET_SAMPLE_TYPE;
  schemaVersion: typeof NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION;
  seed: number;
  step: number;
  phase: "exchanging";
  actingPlayerId: PlayerId;
  actingPlayerIndex: number;
  relativePlayerIds: readonly PlayerId[];
  exchangeStepIndex: 0 | 1 | 2;
  remainingDiscardCount: 1 | 2 | 3;
  modelInput: readonly number[];
  legalDiscardCardMask: readonly number[];
  selectedActionIndex: number;
  behaviorLogProbability: number;
  terminalReward: number;
  outcome: NonPlayingBiddingRlOutcome;
}

export interface NonPlayingRlDatasetManifest {
  datasetSchemaVersion: typeof NON_PLAYING_RL_DATASET_SCHEMA_VERSION;
  generatorVersion: typeof NON_PLAYING_RL_DATASET_GENERATOR_VERSION;
  format: typeof DATASET_FORMAT;
  sampleType:
    | typeof NON_PLAYING_BIDDING_RL_DATASET_SAMPLE_TYPE
    | typeof NON_PLAYING_ADJUTANT_RL_DATASET_SAMPLE_TYPE
    | typeof NON_PLAYING_EXCHANGE_RL_DATASET_SAMPLE_TYPE;
  sampleSchemaVersion: typeof NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION;
  phaseScope:
    | typeof NON_PLAYING_BIDDING_RL_PHASE_SCOPE
    | typeof NON_PLAYING_ADJUTANT_RL_PHASE_SCOPE
    | typeof NON_PLAYING_EXCHANGE_RL_PHASE_SCOPE;
  learnedPhases: readonly ["bidding"] | readonly ["choosing-adjutant"] | readonly ["exchanging"];
  ruleBasedPhases:
    | readonly ["choosing-adjutant", "exchanging"]
    | readonly ["bidding", "exchanging"]
    | readonly ["bidding", "choosing-adjutant"];
  fixedPhases: readonly ["playing"];
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
  biddingEncoderSchemaVersion: typeof BIDDING_ENCODER_SCHEMA_VERSION;
  biddingModelInputSchemaVersion: typeof BIDDING_MODEL_INPUT_SCHEMA_VERSION;
  biddingModelInputFeatureCount: typeof BIDDING_MODEL_INPUT_FEATURE_COUNT;
  adjutantEncoderSchemaVersion?: typeof ADJUTANT_ENCODER_SCHEMA_VERSION;
  adjutantModelInputSchemaVersion?: typeof ADJUTANT_MODEL_INPUT_SCHEMA_VERSION;
  adjutantModelInputFeatureCount?: typeof ADJUTANT_MODEL_INPUT_FEATURE_COUNT;
  exchangeEncoderSchemaVersion?: typeof EXCHANGE_ENCODER_SCHEMA_VERSION;
  exchangeModelInputSchemaVersion?: typeof EXCHANGE_MODEL_INPUT_SCHEMA_VERSION;
  exchangeModelInputFeatureCount?: typeof EXCHANGE_MODEL_INPUT_FEATURE_COUNT;
  decisionMode?: typeof NON_PLAYING_EXCHANGE_RL_DECISION_MODE;
  playingModelInputSchemaVersion: typeof MODEL_INPUT_SCHEMA_VERSION;
  playingModelInputFeatureCount: typeof MODEL_INPUT_FEATURE_COUNT;
  actionCount: typeof BIDDING_ACTION_COUNT | typeof CARD_COUNT;
  behaviorPolicy: NonPlayingRlPolicyArtifactManifest;
  fixedPlayingPolicy: NonPlayingRlPolicyArtifactManifest;
  samplingAlgorithm: typeof NON_PLAYING_RL_SAMPLING_ALGORITHM;
  temperature: number;
  reward: {
    type: typeof NON_PLAYING_RL_REWARD_TYPE;
    version: typeof NON_PLAYING_RL_REWARD_VERSION;
    id: typeof NON_PLAYING_RL_REWARD_ID;
  };
  nonLearningAgents: {
    bidding?: {
      type: "rule-based";
      version: typeof RULE_BASED_AGENT_VERSION;
    };
    choosingAdjutant?: {
      type: "rule-based";
      version: typeof RULE_BASED_AGENT_VERSION;
    };
    exchanging?: {
      type: "rule-based";
      version: typeof RULE_BASED_AGENT_VERSION;
    };
  };
  shards: readonly DatasetShardManifest[];
}

export interface NonPlayingRlPolicyArtifactManifest {
  type: "bidding-onnx" | "adjutant-onnx" | "exchange-onnx" | "playing-onnx";
  artifactId: string;
  onnxFileName: string;
  metadataFileName: string;
  onnxSha256: string;
  metadataSha256: string;
  requestedInferenceDevice?: "cpu" | "auto" | "cuda";
  resolvedInferenceDevice?: "cpu" | "cuda";
  executionProvider?: "cpu" | "cuda";
  metadata: unknown;
}

export interface GenerateNonPlayingBiddingRlDatasetOptions {
  outputDirectory: string;
  biddingPolicy: NonPlayingBiddingRlPolicy;
  biddingPolicyArtifact: NonPlayingRlPolicyArtifactOptions;
  playingPolicy: FixedPlayingPolicy;
  playingPolicyArtifact: NonPlayingRlPolicyArtifactOptions;
  startSeed: number;
  gameCount: number;
  gamesPerShard: number;
  temperature?: number;
  maxDecisionSteps?: number;
  onProgress?: (progress: DatasetGenerationProgress) => void;
}

export interface GenerateNonPlayingBiddingRlDatasetResult {
  outputDirectory: string;
  manifest: NonPlayingRlDatasetManifest;
}

export interface GenerateNonPlayingAdjutantRlDatasetOptions {
  outputDirectory: string;
  adjutantPolicy: NonPlayingAdjutantRlPolicy;
  adjutantPolicyArtifact: NonPlayingRlPolicyArtifactOptions;
  playingPolicy: FixedPlayingPolicy;
  playingPolicyArtifact: NonPlayingRlPolicyArtifactOptions;
  startSeed: number;
  gameCount: number;
  gamesPerShard: number;
  temperature?: number;
  maxDecisionSteps?: number;
  onProgress?: (progress: DatasetGenerationProgress) => void;
}

export interface GenerateNonPlayingAdjutantRlDatasetResult {
  outputDirectory: string;
  manifest: NonPlayingRlDatasetManifest;
}

export interface GenerateNonPlayingExchangeRlDatasetOptions {
  outputDirectory: string;
  exchangePolicy: NonPlayingExchangeRlPolicy;
  exchangePolicyArtifact: NonPlayingRlPolicyArtifactOptions;
  playingPolicy: FixedPlayingPolicy;
  playingPolicyArtifact: NonPlayingRlPolicyArtifactOptions;
  startSeed: number;
  gameCount: number;
  gamesPerShard: number;
  temperature?: number;
  maxDecisionSteps?: number;
  onProgress?: (progress: DatasetGenerationProgress) => void;
}

export interface GenerateNonPlayingExchangeRlDatasetResult {
  outputDirectory: string;
  manifest: NonPlayingRlDatasetManifest;
}

export interface BiddingRlSampleDraft {
  step: number;
  playerId: PlayerId;
  relativePlayerIds: readonly PlayerId[];
  modelInput: Float32Array;
  legalBidMask: readonly number[];
  selectedActionIndex: number;
  behaviorLogProbability: number;
}

export interface BiddingRlGameRunResult {
  record: AutomatedGameRecord;
  drafts: readonly BiddingRlSampleDraft[];
}

export interface AdjutantRlSampleDraft {
  step: number;
  playerId: PlayerId;
  relativePlayerIds: readonly PlayerId[];
  modelInput: Float32Array;
  legalAdjutantMask: readonly number[];
  selectedActionIndex: number;
  behaviorLogProbability: number;
}

export interface AdjutantRlGameRunResult {
  record: AutomatedGameRecord;
  drafts: readonly AdjutantRlSampleDraft[];
}

export interface ExchangeRlSampleDraft {
  step: number;
  playerId: PlayerId;
  relativePlayerIds: readonly PlayerId[];
  exchangeStepIndex: 0 | 1 | 2;
  remainingDiscardCount: 1 | 2 | 3;
  modelInput: Float32Array;
  legalDiscardCardMask: readonly number[];
  selectedActionIndex: number;
  behaviorLogProbability: number;
}

export interface ExchangeRlGameRunResult {
  record: AutomatedGameRecord;
  drafts: readonly ExchangeRlSampleDraft[];
}

export async function generateNonPlayingBiddingRlDataset(
  options: GenerateNonPlayingBiddingRlDatasetOptions
): Promise<GenerateNonPlayingBiddingRlDatasetResult> {
  const temperature = options.temperature ?? DEFAULT_NON_PLAYING_RL_TEMPERATURE;
  validateNonPlayingBiddingRlGenerationOptions({ ...options, temperature });

  const outputDirectory = resolve(options.outputDirectory);
  await ensureOutputDoesNotExist(outputDirectory);
  await mkdir(dirname(outputDirectory), { recursive: true });

  const behaviorPolicy = await createPolicyArtifactManifest({
    type: "bidding-onnx",
    artifact: options.biddingPolicyArtifact,
    policy: options.biddingPolicy
  });
  const fixedPlayingPolicy = await createPolicyArtifactManifest({
    type: "playing-onnx",
    artifact: options.playingPolicyArtifact,
    policy: options.playingPolicy
  });
  const tempDirectory = await mkdtemp(
    join(dirname(outputDirectory), `.${basenameForTemp(outputDirectory)}.tmp-`)
  );
  let activeShard: JsonlShardWriter<NonPlayingBiddingRlSample> | null = null;

  try {
    const shards: DatasetShardManifest[] = [];
    let totalSampleCount = 0;
    let shardGameCount = 0;

    for (let gameOffset = 0; gameOffset < options.gameCount; gameOffset += 1) {
      const seed = options.startSeed + gameOffset;

      if (activeShard === null) {
        activeShard = createJsonlShardWriter(
          tempDirectory,
          shards.length,
          seed,
          serializeNonPlayingBiddingRlSample
        );
        shardGameCount = 0;
      }

      const result = await runNonPlayingBiddingRlGame({
        seed,
        biddingPolicy: options.biddingPolicy,
        playingPolicy: options.playingPolicy,
        temperature,
        maxDecisionSteps: options.maxDecisionSteps
      });
      const samples = completeNonPlayingBiddingRlSamples(result.record, result.drafts);

      for (const sample of samples) {
        validateNonPlayingBiddingRlSample(sample, seed);
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

    const manifest = createNonPlayingRlDatasetManifest({
      options,
      temperature,
      sampleCount: totalSampleCount,
      shards,
      behaviorPolicy,
      fixedPlayingPolicy
    });

    validateNonPlayingRlDatasetManifest(manifest);
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

export async function generateNonPlayingAdjutantRlDataset(
  options: GenerateNonPlayingAdjutantRlDatasetOptions
): Promise<GenerateNonPlayingAdjutantRlDatasetResult> {
  const temperature = options.temperature ?? DEFAULT_NON_PLAYING_RL_TEMPERATURE;
  validateNonPlayingAdjutantRlGenerationOptions({ ...options, temperature });

  const outputDirectory = resolve(options.outputDirectory);
  await ensureOutputDoesNotExist(outputDirectory);
  await mkdir(dirname(outputDirectory), { recursive: true });

  const behaviorPolicy = await createPolicyArtifactManifest({
    type: "adjutant-onnx",
    artifact: options.adjutantPolicyArtifact,
    policy: options.adjutantPolicy
  });
  const fixedPlayingPolicy = await createPolicyArtifactManifest({
    type: "playing-onnx",
    artifact: options.playingPolicyArtifact,
    policy: options.playingPolicy
  });
  const tempDirectory = await mkdtemp(
    join(dirname(outputDirectory), `.${basenameForTemp(outputDirectory)}.tmp-`)
  );
  let activeShard: JsonlShardWriter<NonPlayingAdjutantRlSample> | null = null;

  try {
    const shards: DatasetShardManifest[] = [];
    let totalSampleCount = 0;
    let shardGameCount = 0;

    for (let gameOffset = 0; gameOffset < options.gameCount; gameOffset += 1) {
      const seed = options.startSeed + gameOffset;

      if (activeShard === null) {
        activeShard = createJsonlShardWriter(
          tempDirectory,
          shards.length,
          seed,
          serializeNonPlayingAdjutantRlSample
        );
        shardGameCount = 0;
      }

      const result = await runNonPlayingAdjutantRlGame({
        seed,
        adjutantPolicy: options.adjutantPolicy,
        playingPolicy: options.playingPolicy,
        temperature,
        maxDecisionSteps: options.maxDecisionSteps
      });
      const samples = completeNonPlayingAdjutantRlSamples(result.record, result.drafts);

      for (const sample of samples) {
        validateNonPlayingAdjutantRlSample(sample, seed);
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

    const manifest = createNonPlayingAdjutantRlDatasetManifest({
      options,
      temperature,
      sampleCount: totalSampleCount,
      shards,
      behaviorPolicy,
      fixedPlayingPolicy
    });

    validateNonPlayingAdjutantRlDatasetManifest(manifest);
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

export async function generateNonPlayingExchangeRlDataset(
  options: GenerateNonPlayingExchangeRlDatasetOptions
): Promise<GenerateNonPlayingExchangeRlDatasetResult> {
  const temperature = options.temperature ?? DEFAULT_NON_PLAYING_RL_TEMPERATURE;
  validateNonPlayingExchangeRlGenerationOptions({ ...options, temperature });

  const outputDirectory = resolve(options.outputDirectory);
  await ensureOutputDoesNotExist(outputDirectory);
  await mkdir(dirname(outputDirectory), { recursive: true });

  const behaviorPolicy = await createPolicyArtifactManifest({
    type: "exchange-onnx",
    artifact: options.exchangePolicyArtifact,
    policy: options.exchangePolicy
  });
  const fixedPlayingPolicy = await createPolicyArtifactManifest({
    type: "playing-onnx",
    artifact: options.playingPolicyArtifact,
    policy: options.playingPolicy
  });
  const tempDirectory = await mkdtemp(
    join(dirname(outputDirectory), `.${basenameForTemp(outputDirectory)}.tmp-`)
  );
  let activeShard: JsonlShardWriter<NonPlayingExchangeRlSample> | null = null;

  try {
    const shards: DatasetShardManifest[] = [];
    let totalSampleCount = 0;
    let shardGameCount = 0;

    for (let gameOffset = 0; gameOffset < options.gameCount; gameOffset += 1) {
      const seed = options.startSeed + gameOffset;

      if (activeShard === null) {
        activeShard = createJsonlShardWriter(
          tempDirectory,
          shards.length,
          seed,
          serializeNonPlayingExchangeRlSample
        );
        shardGameCount = 0;
      }

      const result = await runNonPlayingExchangeRlGame({
        seed,
        exchangePolicy: options.exchangePolicy,
        playingPolicy: options.playingPolicy,
        temperature,
        maxDecisionSteps: options.maxDecisionSteps
      });
      const samples = completeNonPlayingExchangeRlSamples(result.record, result.drafts);

      for (const sample of samples) {
        validateNonPlayingExchangeRlSample(sample, seed);
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

    const manifest = createNonPlayingExchangeRlDatasetManifest({
      options,
      temperature,
      sampleCount: totalSampleCount,
      shards,
      behaviorPolicy,
      fixedPlayingPolicy
    });

    validateNonPlayingExchangeRlDatasetManifest(manifest);
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

export async function runNonPlayingBiddingRlGame(options: {
  seed: number;
  biddingPolicy: NonPlayingBiddingRlPolicy;
  playingPolicy: FixedPlayingPolicy;
  temperature?: number;
  maxDecisionSteps?: number;
}): Promise<BiddingRlGameRunResult> {
  const drafts: BiddingRlSampleDraft[] = [];
  const temperature = options.temperature ?? DEFAULT_NON_PLAYING_RL_TEMPERATURE;
  validateTemperature(temperature);

  const record = await runAutomatedGame({
    seed: options.seed,
    maxDecisionSteps: options.maxDecisionSteps,
    createAgent: ({ rng }) => new NonPlayingBiddingRlAgent({
      biddingPolicy: options.biddingPolicy,
      playingPolicy: options.playingPolicy,
      rng,
      temperature,
      recordSample: (sample) => {
        drafts.push(sample);
      }
    })
  });

  return { record, drafts };
}

export async function runNonPlayingAdjutantRlGame(options: {
  seed: number;
  adjutantPolicy: NonPlayingAdjutantRlPolicy;
  playingPolicy: FixedPlayingPolicy;
  temperature?: number;
  maxDecisionSteps?: number;
}): Promise<AdjutantRlGameRunResult> {
  const drafts: AdjutantRlSampleDraft[] = [];
  const temperature = options.temperature ?? DEFAULT_NON_PLAYING_RL_TEMPERATURE;
  validateTemperature(temperature);

  const record = await runAutomatedGame({
    seed: options.seed,
    maxDecisionSteps: options.maxDecisionSteps,
    createAgent: ({ rng }) => new NonPlayingAdjutantRlAgent({
      adjutantPolicy: options.adjutantPolicy,
      playingPolicy: options.playingPolicy,
      rng,
      temperature,
      recordSample: (sample) => {
        drafts.push(sample);
      }
    })
  });

  return { record, drafts };
}

export async function runNonPlayingExchangeRlGame(options: {
  seed: number;
  exchangePolicy: NonPlayingExchangeRlPolicy;
  playingPolicy: FixedPlayingPolicy;
  temperature?: number;
  maxDecisionSteps?: number;
}): Promise<ExchangeRlGameRunResult> {
  const drafts: ExchangeRlSampleDraft[] = [];
  const temperature = options.temperature ?? DEFAULT_NON_PLAYING_RL_TEMPERATURE;
  validateTemperature(temperature);

  const record = await runAutomatedGame({
    seed: options.seed,
    maxDecisionSteps: options.maxDecisionSteps,
    createAgent: ({ rng }) => new NonPlayingExchangeRlAgent({
      exchangePolicy: options.exchangePolicy,
      playingPolicy: options.playingPolicy,
      rng,
      temperature,
      recordSample: (sample) => {
        drafts.push(sample);
      }
    })
  });

  return { record, drafts };
}

export function completeNonPlayingBiddingRlSamples(
  record: AutomatedGameRecord,
  drafts: readonly BiddingRlSampleDraft[]
): readonly NonPlayingBiddingRlSample[] {
  const samples: NonPlayingBiddingRlSample[] = [];
  let draftIndex = 0;

  for (const decision of record.decisions) {
    if (decision.phase !== "bidding") {
      continue;
    }

    const draft = drafts[draftIndex];
    draftIndex += 1;

    if (draft === undefined) {
      throw new Error("Missing sampled bidding action for non-playing RL sample.");
    }
    if (draft.step !== decision.step || draft.playerId !== decision.playerId) {
      throw new Error("Sampled bidding action does not match automated decision order.");
    }
    if (decision.action.type !== "bid" && decision.action.type !== "pass") {
      throw new Error(`Bidding RL decision action must be bid/pass, got ${decision.action.type}.`);
    }

    const outcome = createBiddingRlOutcome(record.result, decision.playerId);
    const sample: NonPlayingBiddingRlSample = {
      sampleType: NON_PLAYING_RL_DATASET_SAMPLE_TYPE,
      schemaVersion: NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION,
      seed: record.seed,
      step: decision.step,
      phase: "bidding",
      actingPlayerId: decision.playerId,
      actingPlayerIndex: record.playerIds.indexOf(decision.playerId),
      relativePlayerIds: draft.relativePlayerIds,
      modelInput: Array.from(draft.modelInput),
      legalBidMask: [...draft.legalBidMask],
      selectedActionIndex: draft.selectedActionIndex,
      behaviorLogProbability: draft.behaviorLogProbability,
      terminalReward: calculateNonPlayingTerminalRoleReward(outcome),
      outcome
    };

    validateNonPlayingBiddingRlSample(sample, record.seed);
    samples.push(sample);
  }

  if (draftIndex !== drafts.length) {
    throw new Error("Unused sampled bidding actions remain after completing RL samples.");
  }

  return samples;
}

export function completeNonPlayingAdjutantRlSamples(
  record: AutomatedGameRecord,
  drafts: readonly AdjutantRlSampleDraft[]
): readonly NonPlayingAdjutantRlSample[] {
  const samples: NonPlayingAdjutantRlSample[] = [];
  let draftIndex = 0;

  for (const decision of record.decisions) {
    if (decision.phase !== "choosing-adjutant") {
      continue;
    }

    const draft = drafts[draftIndex];
    draftIndex += 1;

    if (draft === undefined) {
      throw new Error("Missing sampled adjutant action for non-playing RL sample.");
    }
    if (draft.step !== decision.step || draft.playerId !== decision.playerId) {
      throw new Error("Sampled adjutant action does not match automated decision order.");
    }
    if (decision.action.type !== "choose-adjutant") {
      throw new Error(
        `Adjutant RL decision action must be choose-adjutant, got ${decision.action.type}.`
      );
    }

    const outcome = createBiddingRlOutcome(record.result, decision.playerId);
    const sample: NonPlayingAdjutantRlSample = {
      sampleType: NON_PLAYING_ADJUTANT_RL_DATASET_SAMPLE_TYPE,
      schemaVersion: NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION,
      seed: record.seed,
      step: decision.step,
      phase: "choosing-adjutant",
      actingPlayerId: decision.playerId,
      actingPlayerIndex: record.playerIds.indexOf(decision.playerId),
      relativePlayerIds: draft.relativePlayerIds,
      modelInput: Array.from(draft.modelInput),
      legalAdjutantMask: [...draft.legalAdjutantMask],
      selectedActionIndex: draft.selectedActionIndex,
      behaviorLogProbability: draft.behaviorLogProbability,
      terminalReward: calculateNonPlayingTerminalRoleReward(outcome),
      outcome
    };

    validateNonPlayingAdjutantRlSample(sample, record.seed);
    samples.push(sample);
  }

  if (draftIndex !== drafts.length) {
    throw new Error("Unused sampled adjutant actions remain after completing RL samples.");
  }

  return samples;
}

export function completeNonPlayingExchangeRlSamples(
  record: AutomatedGameRecord,
  drafts: readonly ExchangeRlSampleDraft[]
): readonly NonPlayingExchangeRlSample[] {
  const samples: NonPlayingExchangeRlSample[] = [];
  let draftIndex = 0;

  for (const decision of record.decisions) {
    if (decision.phase !== "exchanging") {
      continue;
    }

    if (decision.action.type !== "discard-cards") {
      throw new Error(`Exchange RL decision action must be discard-cards, got ${decision.action.type}.`);
    }

    const selectedActionIndices = decision.action.cardIds.map((cardId) => CARD_IDS.indexOf(cardId));
    if (selectedActionIndices.includes(-1)) {
      throw new Error("Exchange RL decision contains unknown card ids.");
    }

    const outcome = createBiddingRlOutcome(record.result, decision.playerId);
    for (let exchangeStepIndex = 0; exchangeStepIndex < 3; exchangeStepIndex += 1) {
      const draft = drafts[draftIndex];
      draftIndex += 1;

      if (draft === undefined) {
        throw new Error("Missing sampled exchange action for non-playing RL sample.");
      }
      if (
        draft.playerId !== decision.playerId ||
        draft.exchangeStepIndex !== exchangeStepIndex
      ) {
        throw new Error("Sampled exchange action does not match automated decision order.");
      }
      if (draft.selectedActionIndex !== selectedActionIndices[exchangeStepIndex]) {
        throw new Error("Sampled exchange action does not match discarded card order.");
      }

      const sample: NonPlayingExchangeRlSample = {
        sampleType: NON_PLAYING_EXCHANGE_RL_DATASET_SAMPLE_TYPE,
        schemaVersion: NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION,
        seed: record.seed,
        step: decision.step,
        phase: "exchanging",
        actingPlayerId: decision.playerId,
        actingPlayerIndex: record.playerIds.indexOf(decision.playerId),
        relativePlayerIds: draft.relativePlayerIds,
        exchangeStepIndex: draft.exchangeStepIndex,
        remainingDiscardCount: draft.remainingDiscardCount,
        modelInput: Array.from(draft.modelInput),
        legalDiscardCardMask: [...draft.legalDiscardCardMask],
        selectedActionIndex: draft.selectedActionIndex,
        behaviorLogProbability: draft.behaviorLogProbability,
        terminalReward: calculateNonPlayingTerminalRoleReward(outcome),
        outcome
      };

      validateNonPlayingExchangeRlSample(sample, record.seed);
      samples.push(sample);
    }
  }

  if (draftIndex !== drafts.length) {
    throw new Error("Unused sampled exchange actions remain after completing RL samples.");
  }

  return samples;
}

export function calculateNonPlayingTerminalRoleReward(
  outcome: Pick<NonPlayingBiddingRlOutcome, "winner" | "targetPointCards" | "actingPlayerRole">
): number {
  const d = outcome.targetPointCards;
  const napoleonWon = outcome.winner === "napoleon-team";

  switch (outcome.actingPlayerRole) {
    case "napoleon":
      return napoleonWon ? d : -3;
    case "adjutant":
      return napoleonWon ? d - 7 : 0;
    case "citizen":
      return napoleonWon ? 7 : 0;
    case "napoleon-adjutant":
      return napoleonWon ? 2 * d - 7 : -3;
  }
}

export function calculateNonPlayingBiddingLogProbability(options: {
  logits: Float32Array | readonly number[];
  legalBidMask: ArrayLike<number | boolean>;
  selectedActionIndex: number;
  temperature?: number;
}): number {
  const distribution = createMaskedCategoricalDistribution(
    options.logits,
    options.legalBidMask,
    BIDDING_ACTION_COUNT,
    options.temperature ?? DEFAULT_NON_PLAYING_RL_TEMPERATURE
  );
  const index = distribution.legalIndices.indexOf(options.selectedActionIndex);

  if (index === -1) {
    throw new Error(
      `selectedActionIndex ${options.selectedActionIndex} is not legal under legalBidMask.`
    );
  }

  return distribution.logProbabilities[index];
}

export function calculateNonPlayingAdjutantLogProbability(options: {
  logits: Float32Array | readonly number[];
  legalAdjutantMask: ArrayLike<number | boolean>;
  selectedActionIndex: number;
  temperature?: number;
}): number {
  const distribution = createMaskedCategoricalDistribution(
    options.logits,
    options.legalAdjutantMask,
    CARD_COUNT,
    options.temperature ?? DEFAULT_NON_PLAYING_RL_TEMPERATURE
  );
  const index = distribution.legalIndices.indexOf(options.selectedActionIndex);

  if (index === -1) {
    throw new Error(
      `selectedActionIndex ${options.selectedActionIndex} is not legal under legalAdjutantMask.`
    );
  }

  return distribution.logProbabilities[index];
}

export function calculateNonPlayingExchangeLogProbability(options: {
  logits: Float32Array | readonly number[];
  legalDiscardCardMask: ArrayLike<number | boolean>;
  selectedActionIndex: number;
  temperature?: number;
}): number {
  const distribution = createMaskedCategoricalDistribution(
    options.logits,
    options.legalDiscardCardMask,
    CARD_COUNT,
    options.temperature ?? DEFAULT_NON_PLAYING_RL_TEMPERATURE
  );
  const index = distribution.legalIndices.indexOf(options.selectedActionIndex);

  if (index === -1) {
    throw new Error(
      `selectedActionIndex ${options.selectedActionIndex} is not legal under legalDiscardCardMask.`
    );
  }

  return distribution.logProbabilities[index];
}

export function serializeNonPlayingBiddingRlSample(
  sample: NonPlayingBiddingRlSample
): string {
  return `${JSON.stringify(sample)}\n`;
}

export function serializeNonPlayingAdjutantRlSample(
  sample: NonPlayingAdjutantRlSample
): string {
  return `${JSON.stringify(sample)}\n`;
}

export function serializeNonPlayingExchangeRlSample(
  sample: NonPlayingExchangeRlSample
): string {
  return `${JSON.stringify(sample)}\n`;
}

export function validateNonPlayingBiddingRlGenerationOptions(
  options: GenerateNonPlayingBiddingRlDatasetOptions & { temperature?: number }
): void {
  validateUint32("startSeed", options.startSeed);
  validatePositiveInteger("gameCount", options.gameCount);
  validatePositiveInteger("gamesPerShard", options.gamesPerShard);
  validateTemperature(options.temperature ?? DEFAULT_NON_PLAYING_RL_TEMPERATURE);

  if (options.outputDirectory.length === 0) {
    throw new Error("outputDirectory must be a non-empty path.");
  }

  const endSeed = options.startSeed + options.gameCount - 1;

  if (!Number.isSafeInteger(endSeed) || endSeed > UINT32_MAX) {
    throw new Error(`Seed range exceeds uint32: ${options.startSeed}..${endSeed}`);
  }

  const expectedShardCount = calculateExpectedShardCount(
    options.gameCount,
    options.gamesPerShard
  );

  if (expectedShardCount > MAX_SHARD_COUNT) {
    throw new Error(
      `Dataset would require ${expectedShardCount} shards, exceeding the maximum ${MAX_SHARD_COUNT}.`
    );
  }
}

export function validateNonPlayingAdjutantRlGenerationOptions(
  options: GenerateNonPlayingAdjutantRlDatasetOptions & { temperature?: number }
): void {
  validateUint32("startSeed", options.startSeed);
  validatePositiveInteger("gameCount", options.gameCount);
  validatePositiveInteger("gamesPerShard", options.gamesPerShard);
  validateTemperature(options.temperature ?? DEFAULT_NON_PLAYING_RL_TEMPERATURE);

  if (options.outputDirectory.length === 0) {
    throw new Error("outputDirectory must be a non-empty path.");
  }

  const endSeed = options.startSeed + options.gameCount - 1;

  if (!Number.isSafeInteger(endSeed) || endSeed > UINT32_MAX) {
    throw new Error(`Seed range exceeds uint32: ${options.startSeed}..${endSeed}`);
  }

  const expectedShardCount = calculateExpectedShardCount(
    options.gameCount,
    options.gamesPerShard
  );

  if (expectedShardCount > MAX_SHARD_COUNT) {
    throw new Error(
      `Dataset would require ${expectedShardCount} shards, exceeding the maximum ${MAX_SHARD_COUNT}.`
    );
  }
}

export function validateNonPlayingExchangeRlGenerationOptions(
  options: GenerateNonPlayingExchangeRlDatasetOptions & { temperature?: number }
): void {
  validateUint32("startSeed", options.startSeed);
  validatePositiveInteger("gameCount", options.gameCount);
  validatePositiveInteger("gamesPerShard", options.gamesPerShard);
  validateTemperature(options.temperature ?? DEFAULT_NON_PLAYING_RL_TEMPERATURE);

  if (options.outputDirectory.length === 0) {
    throw new Error("outputDirectory must be a non-empty path.");
  }

  const endSeed = options.startSeed + options.gameCount - 1;

  if (!Number.isSafeInteger(endSeed) || endSeed > UINT32_MAX) {
    throw new Error(`Seed range exceeds uint32: ${options.startSeed}..${endSeed}`);
  }

  const expectedShardCount = calculateExpectedShardCount(
    options.gameCount,
    options.gamesPerShard
  );

  if (expectedShardCount > MAX_SHARD_COUNT) {
    throw new Error(
      `Dataset would require ${expectedShardCount} shards, exceeding the maximum ${MAX_SHARD_COUNT}.`
    );
  }
}

export function validateNonPlayingBiddingRlSample(
  sample: NonPlayingBiddingRlSample,
  expectedSeed: number
): void {
  validateJsonSafeValue("sample", sample);

  if (sample.sampleType !== NON_PLAYING_RL_DATASET_SAMPLE_TYPE) {
    throw new Error(`Sample sampleType must be ${NON_PLAYING_RL_DATASET_SAMPLE_TYPE}.`);
  }
  if (sample.schemaVersion !== NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION) {
    throw new Error(`Unexpected non-playing RL sample schemaVersion: ${sample.schemaVersion}`);
  }
  if (sample.seed !== expectedSeed) {
    throw new Error(`Sample seed must match current seed: ${sample.seed} !== ${expectedSeed}`);
  }
  validateUint32("Sample seed", sample.seed);
  validatePositiveInteger("Sample step", sample.step);
  validatePlayerIndex("actingPlayerIndex", sample.actingPlayerIndex);
  expectLength("relativePlayerIds", sample.relativePlayerIds, PLAYER_COUNT);
  if (sample.relativePlayerIds[0] !== sample.actingPlayerId) {
    throw new Error("actingPlayerId must match relativePlayerIds[0].");
  }
  expectLength("modelInput", sample.modelInput, BIDDING_MODEL_INPUT_FEATURE_COUNT);
  for (const [index, value] of sample.modelInput.entries()) {
    if (!Number.isFinite(value)) {
      throw new Error(`modelInput[${index}] must be finite.`);
    }
  }
  validateLegalBidMask(sample.legalBidMask);
  validateBiddingActionIndex(sample.selectedActionIndex);
  if (sample.legalBidMask[sample.selectedActionIndex] !== 1) {
    throw new Error("selectedActionIndex must be legal in legalBidMask.");
  }
  if (
    !Number.isFinite(sample.behaviorLogProbability) ||
    sample.behaviorLogProbability > 1e-12
  ) {
    throw new Error("behaviorLogProbability must be finite and <= 0.");
  }
  if (!Number.isFinite(sample.terminalReward)) {
    throw new Error("terminalReward must be finite.");
  }
  validateOutcome(sample.outcome);
  if (sample.terminalReward !== calculateNonPlayingTerminalRoleReward(sample.outcome)) {
    throw new Error("terminalReward must match reward version formula.");
  }
}

export function validateNonPlayingAdjutantRlSample(
  sample: NonPlayingAdjutantRlSample,
  expectedSeed: number
): void {
  validateJsonSafeValue("sample", sample);

  if (sample.sampleType !== NON_PLAYING_ADJUTANT_RL_DATASET_SAMPLE_TYPE) {
    throw new Error(`Sample sampleType must be ${NON_PLAYING_ADJUTANT_RL_DATASET_SAMPLE_TYPE}.`);
  }
  if (sample.schemaVersion !== NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION) {
    throw new Error(`Unexpected non-playing RL sample schemaVersion: ${sample.schemaVersion}`);
  }
  if (sample.seed !== expectedSeed) {
    throw new Error(`Sample seed must match current seed: ${sample.seed} !== ${expectedSeed}`);
  }
  validateUint32("Sample seed", sample.seed);
  validatePositiveInteger("Sample step", sample.step);
  validatePlayerIndex("actingPlayerIndex", sample.actingPlayerIndex);
  expectLength("relativePlayerIds", sample.relativePlayerIds, PLAYER_COUNT);
  if (sample.relativePlayerIds[0] !== sample.actingPlayerId) {
    throw new Error("actingPlayerId must match relativePlayerIds[0].");
  }
  expectLength("modelInput", sample.modelInput, ADJUTANT_MODEL_INPUT_FEATURE_COUNT);
  for (const [index, value] of sample.modelInput.entries()) {
    if (!Number.isFinite(value)) {
      throw new Error(`modelInput[${index}] must be finite.`);
    }
  }
  validateLegalAdjutantMask(sample.legalAdjutantMask);
  validateCardActionIndex(sample.selectedActionIndex);
  if (sample.legalAdjutantMask[sample.selectedActionIndex] !== 1) {
    throw new Error("selectedActionIndex must be legal in legalAdjutantMask.");
  }
  if (
    !Number.isFinite(sample.behaviorLogProbability) ||
    sample.behaviorLogProbability > 1e-12
  ) {
    throw new Error("behaviorLogProbability must be finite and <= 0.");
  }
  if (!Number.isFinite(sample.terminalReward)) {
    throw new Error("terminalReward must be finite.");
  }
  validateOutcome(sample.outcome);
  if (sample.terminalReward !== calculateNonPlayingTerminalRoleReward(sample.outcome)) {
    throw new Error("terminalReward must match reward version formula.");
  }
}

export function validateNonPlayingExchangeRlSample(
  sample: NonPlayingExchangeRlSample,
  expectedSeed: number
): void {
  validateJsonSafeValue("sample", sample);

  if (sample.sampleType !== NON_PLAYING_EXCHANGE_RL_DATASET_SAMPLE_TYPE) {
    throw new Error(`Sample sampleType must be ${NON_PLAYING_EXCHANGE_RL_DATASET_SAMPLE_TYPE}.`);
  }
  if (sample.schemaVersion !== NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION) {
    throw new Error(`Unexpected non-playing RL sample schemaVersion: ${sample.schemaVersion}`);
  }
  if (sample.seed !== expectedSeed) {
    throw new Error(`Sample seed must match current seed: ${sample.seed} !== ${expectedSeed}`);
  }
  validateUint32("Sample seed", sample.seed);
  validatePositiveInteger("Sample step", sample.step);
  validatePlayerIndex("actingPlayerIndex", sample.actingPlayerIndex);
  expectLength("relativePlayerIds", sample.relativePlayerIds, PLAYER_COUNT);
  if (sample.relativePlayerIds[0] !== sample.actingPlayerId) {
    throw new Error("actingPlayerId must match relativePlayerIds[0].");
  }
  if (!Number.isSafeInteger(sample.exchangeStepIndex) || sample.exchangeStepIndex < 0 || sample.exchangeStepIndex > 2) {
    throw new Error("exchangeStepIndex must be 0, 1, or 2.");
  }
  if (sample.remainingDiscardCount !== 3 - sample.exchangeStepIndex) {
    throw new Error("remainingDiscardCount must equal 3 - exchangeStepIndex.");
  }
  expectLength("modelInput", sample.modelInput, EXCHANGE_MODEL_INPUT_FEATURE_COUNT);
  for (const [index, value] of sample.modelInput.entries()) {
    if (!Number.isFinite(value)) {
      throw new Error(`modelInput[${index}] must be finite.`);
    }
  }
  validateLegalDiscardMask(sample.legalDiscardCardMask);
  validateCardActionIndex(sample.selectedActionIndex);
  if (sample.legalDiscardCardMask[sample.selectedActionIndex] !== 1) {
    throw new Error("selectedActionIndex must be legal in legalDiscardCardMask.");
  }
  if (
    !Number.isFinite(sample.behaviorLogProbability) ||
    sample.behaviorLogProbability > 1e-12
  ) {
    throw new Error("behaviorLogProbability must be finite and <= 0.");
  }
  if (!Number.isFinite(sample.terminalReward)) {
    throw new Error("terminalReward must be finite.");
  }
  validateOutcome(sample.outcome);
  if (sample.terminalReward !== calculateNonPlayingTerminalRoleReward(sample.outcome)) {
    throw new Error("terminalReward must match reward version formula.");
  }
}

export function validateNonPlayingRlDatasetManifest(
  manifest: NonPlayingRlDatasetManifest
): void {
  if (manifest.datasetSchemaVersion !== NON_PLAYING_RL_DATASET_SCHEMA_VERSION) {
    throw new Error("Non-playing RL manifest datasetSchemaVersion mismatch.");
  }
  if (manifest.generatorVersion !== NON_PLAYING_RL_DATASET_GENERATOR_VERSION) {
    throw new Error("Non-playing RL manifest generatorVersion mismatch.");
  }
  if (
    manifest.format !== DATASET_FORMAT ||
    manifest.sampleType !== NON_PLAYING_RL_DATASET_SAMPLE_TYPE ||
    manifest.sampleSchemaVersion !== NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION
  ) {
    throw new Error("Non-playing RL manifest format or sample type mismatch.");
  }
  if (
    manifest.phaseScope !== NON_PLAYING_RL_PHASE_SCOPE ||
    !sameStringArray(manifest.learnedPhases, ["bidding"]) ||
    !sameStringArray(manifest.ruleBasedPhases, ["choosing-adjutant", "exchanging"]) ||
    !sameStringArray(manifest.fixedPhases, ["playing"])
  ) {
    throw new Error("Non-playing RL manifest phase scope mismatch.");
  }
  validateUint32("Manifest startSeed", manifest.startSeed);
  validateUint32("Manifest endSeed", manifest.endSeed);
  if (manifest.endSeed !== manifest.startSeed + manifest.gameCount - 1) {
    throw new Error("Non-playing RL manifest seed range mismatch.");
  }
  validatePositiveInteger("Manifest gameCount", manifest.gameCount);
  validatePositiveInteger("Manifest sampleCount", manifest.sampleCount);
  validatePositiveInteger("Manifest gamesPerShard", manifest.gamesPerShard);
  validatePositiveInteger("Manifest shardCount", manifest.shardCount);
  if (manifest.shardCount !== calculateExpectedShardCount(manifest.gameCount, manifest.gamesPerShard)) {
    throw new Error("Non-playing RL manifest shardCount mismatch.");
  }
  if (manifest.shardCount !== manifest.shards.length) {
    throw new Error("Non-playing RL manifest shardCount must match shards length.");
  }
  if (manifest.sampleCount !== sum(manifest.shards.map((shard) => shard.sampleCount))) {
    throw new Error("Non-playing RL manifest sampleCount must equal shard sample counts.");
  }
  if (manifest.playerCount !== PLAYER_COUNT || manifest.cardCount !== CARD_COUNT) {
    throw new Error("Non-playing RL manifest fixed dimensions mismatch.");
  }
  if (!sameStringArray(manifest.cardIds, CARD_IDS)) {
    throw new Error("Non-playing RL manifest cardIds mismatch.");
  }
  if (manifest.cardIdsSha256 !== calculateCardIdsSha256()) {
    throw new Error("Non-playing RL manifest cardIdsSha256 mismatch.");
  }
  if (
    manifest.biddingEncoderSchemaVersion !== BIDDING_ENCODER_SCHEMA_VERSION ||
    manifest.biddingModelInputSchemaVersion !== BIDDING_MODEL_INPUT_SCHEMA_VERSION ||
    manifest.biddingModelInputFeatureCount !== BIDDING_MODEL_INPUT_FEATURE_COUNT ||
    manifest.playingModelInputSchemaVersion !== MODEL_INPUT_SCHEMA_VERSION ||
    manifest.playingModelInputFeatureCount !== MODEL_INPUT_FEATURE_COUNT ||
    manifest.actionCount !== BIDDING_ACTION_COUNT
  ) {
    throw new Error("Non-playing RL manifest model schema metadata mismatch.");
  }
  validatePolicyArtifactManifest(manifest.behaviorPolicy, "bidding-onnx");
  validatePolicyArtifactManifest(manifest.fixedPlayingPolicy, "playing-onnx");
  if (manifest.samplingAlgorithm !== NON_PLAYING_RL_SAMPLING_ALGORITHM) {
    throw new Error("Non-playing RL manifest samplingAlgorithm mismatch.");
  }
  validateTemperature(manifest.temperature);
  if (
    manifest.reward.type !== NON_PLAYING_RL_REWARD_TYPE ||
    manifest.reward.version !== NON_PLAYING_RL_REWARD_VERSION ||
    manifest.reward.id !== NON_PLAYING_RL_REWARD_ID
  ) {
    throw new Error("Non-playing RL manifest reward metadata mismatch.");
  }
  if (
    manifest.nonLearningAgents.choosingAdjutant?.type !== "rule-based" ||
    manifest.nonLearningAgents.choosingAdjutant?.version !== RULE_BASED_AGENT_VERSION ||
    manifest.nonLearningAgents.exchanging?.type !== "rule-based" ||
    manifest.nonLearningAgents.exchanging?.version !== RULE_BASED_AGENT_VERSION
  ) {
    throw new Error("Non-playing RL manifest non-learning agent metadata mismatch.");
  }
  validateShards(manifest);
}

export function validateNonPlayingAdjutantRlDatasetManifest(
  manifest: NonPlayingRlDatasetManifest
): void {
  validateCommonNonPlayingRlDatasetManifest(manifest);
  if (
    manifest.sampleType !== NON_PLAYING_ADJUTANT_RL_DATASET_SAMPLE_TYPE ||
    manifest.sampleSchemaVersion !== NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION
  ) {
    throw new Error("Non-playing adjutant RL manifest sample type mismatch.");
  }
  if (
    manifest.phaseScope !== NON_PLAYING_ADJUTANT_RL_PHASE_SCOPE ||
    !sameStringArray(manifest.learnedPhases, ["choosing-adjutant"]) ||
    !sameStringArray(manifest.ruleBasedPhases, ["bidding", "exchanging"]) ||
    !sameStringArray(manifest.fixedPhases, ["playing"])
  ) {
    throw new Error("Non-playing adjutant RL manifest phase scope mismatch.");
  }
  if (
    manifest.adjutantEncoderSchemaVersion !== ADJUTANT_ENCODER_SCHEMA_VERSION ||
    manifest.adjutantModelInputSchemaVersion !== ADJUTANT_MODEL_INPUT_SCHEMA_VERSION ||
    manifest.adjutantModelInputFeatureCount !== ADJUTANT_MODEL_INPUT_FEATURE_COUNT ||
    manifest.playingModelInputSchemaVersion !== MODEL_INPUT_SCHEMA_VERSION ||
    manifest.playingModelInputFeatureCount !== MODEL_INPUT_FEATURE_COUNT ||
    manifest.actionCount !== CARD_COUNT
  ) {
    throw new Error("Non-playing adjutant RL manifest model schema metadata mismatch.");
  }
  validatePolicyArtifactManifest(manifest.behaviorPolicy, "adjutant-onnx");
  validatePolicyArtifactManifest(manifest.fixedPlayingPolicy, "playing-onnx");
  validateCommonNonPlayingRlPolicyMetadata(manifest);
  if (
    manifest.nonLearningAgents.bidding?.type !== "rule-based" ||
    manifest.nonLearningAgents.bidding.version !== RULE_BASED_AGENT_VERSION ||
    manifest.nonLearningAgents.exchanging?.type !== "rule-based" ||
    manifest.nonLearningAgents.exchanging?.version !== RULE_BASED_AGENT_VERSION
  ) {
    throw new Error("Non-playing adjutant RL manifest non-learning agent metadata mismatch.");
  }
  validateShards(manifest);
}

export function validateNonPlayingExchangeRlDatasetManifest(
  manifest: NonPlayingRlDatasetManifest
): void {
  validateCommonNonPlayingRlDatasetManifest(manifest);
  if (
    manifest.sampleType !== NON_PLAYING_EXCHANGE_RL_DATASET_SAMPLE_TYPE ||
    manifest.sampleSchemaVersion !== NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION
  ) {
    throw new Error("Non-playing exchange RL manifest sample type mismatch.");
  }
  if (
    manifest.phaseScope !== NON_PLAYING_EXCHANGE_RL_PHASE_SCOPE ||
    !sameStringArray(manifest.learnedPhases, ["exchanging"]) ||
    !sameStringArray(manifest.ruleBasedPhases, ["bidding", "choosing-adjutant"]) ||
    !sameStringArray(manifest.fixedPhases, ["playing"])
  ) {
    throw new Error("Non-playing exchange RL manifest phase scope mismatch.");
  }
  if (
    manifest.exchangeEncoderSchemaVersion !== EXCHANGE_ENCODER_SCHEMA_VERSION ||
    manifest.exchangeModelInputSchemaVersion !== EXCHANGE_MODEL_INPUT_SCHEMA_VERSION ||
    manifest.exchangeModelInputFeatureCount !== EXCHANGE_MODEL_INPUT_FEATURE_COUNT ||
    manifest.playingModelInputSchemaVersion !== MODEL_INPUT_SCHEMA_VERSION ||
    manifest.playingModelInputFeatureCount !== MODEL_INPUT_FEATURE_COUNT ||
    manifest.actionCount !== CARD_COUNT ||
    manifest.decisionMode !== NON_PLAYING_EXCHANGE_RL_DECISION_MODE
  ) {
    throw new Error("Non-playing exchange RL manifest model schema metadata mismatch.");
  }
  validatePolicyArtifactManifest(manifest.behaviorPolicy, "exchange-onnx");
  validatePolicyArtifactManifest(manifest.fixedPlayingPolicy, "playing-onnx");
  validateCommonNonPlayingRlPolicyMetadata(manifest);
  if (
    manifest.nonLearningAgents.bidding?.type !== "rule-based" ||
    manifest.nonLearningAgents.bidding.version !== RULE_BASED_AGENT_VERSION ||
    manifest.nonLearningAgents.choosingAdjutant?.type !== "rule-based" ||
    manifest.nonLearningAgents.choosingAdjutant?.version !== RULE_BASED_AGENT_VERSION
  ) {
    throw new Error("Non-playing exchange RL manifest non-learning agent metadata mismatch.");
  }
  validateShards(manifest);
}

function validateCommonNonPlayingRlDatasetManifest(manifest: NonPlayingRlDatasetManifest): void {
  if (manifest.datasetSchemaVersion !== NON_PLAYING_RL_DATASET_SCHEMA_VERSION) {
    throw new Error("Non-playing RL manifest datasetSchemaVersion mismatch.");
  }
  if (manifest.generatorVersion !== NON_PLAYING_RL_DATASET_GENERATOR_VERSION) {
    throw new Error("Non-playing RL manifest generatorVersion mismatch.");
  }
  if (manifest.format !== DATASET_FORMAT) {
    throw new Error("Non-playing RL manifest format mismatch.");
  }
  validateUint32("Manifest startSeed", manifest.startSeed);
  validateUint32("Manifest endSeed", manifest.endSeed);
  if (manifest.endSeed !== manifest.startSeed + manifest.gameCount - 1) {
    throw new Error("Non-playing RL manifest seed range mismatch.");
  }
  validatePositiveInteger("Manifest gameCount", manifest.gameCount);
  validatePositiveInteger("Manifest sampleCount", manifest.sampleCount);
  validatePositiveInteger("Manifest gamesPerShard", manifest.gamesPerShard);
  validatePositiveInteger("Manifest shardCount", manifest.shardCount);
  if (manifest.shardCount !== calculateExpectedShardCount(manifest.gameCount, manifest.gamesPerShard)) {
    throw new Error("Non-playing RL manifest shardCount mismatch.");
  }
  if (manifest.shardCount !== manifest.shards.length) {
    throw new Error("Non-playing RL manifest shardCount must match shards length.");
  }
  if (manifest.sampleCount !== sum(manifest.shards.map((shard) => shard.sampleCount))) {
    throw new Error("Non-playing RL manifest sampleCount must equal shard sample counts.");
  }
  if (manifest.playerCount !== PLAYER_COUNT || manifest.cardCount !== CARD_COUNT) {
    throw new Error("Non-playing RL manifest fixed dimensions mismatch.");
  }
  if (!sameStringArray(manifest.cardIds, CARD_IDS)) {
    throw new Error("Non-playing RL manifest cardIds mismatch.");
  }
  if (manifest.cardIdsSha256 !== calculateCardIdsSha256()) {
    throw new Error("Non-playing RL manifest cardIdsSha256 mismatch.");
  }
}

function validateCommonNonPlayingRlPolicyMetadata(manifest: NonPlayingRlDatasetManifest): void {
  if (manifest.samplingAlgorithm !== NON_PLAYING_RL_SAMPLING_ALGORITHM) {
    throw new Error("Non-playing RL manifest samplingAlgorithm mismatch.");
  }
  validateTemperature(manifest.temperature);
  if (
    manifest.reward.type !== NON_PLAYING_RL_REWARD_TYPE ||
    manifest.reward.version !== NON_PLAYING_RL_REWARD_VERSION ||
    manifest.reward.id !== NON_PLAYING_RL_REWARD_ID
  ) {
    throw new Error("Non-playing RL manifest reward metadata mismatch.");
  }
}

class NonPlayingBiddingRlAgent implements Agent {
  private readonly ruleBasedAgent: RuleBasedAgent;

  constructor(
    private readonly options: {
      biddingPolicy: NonPlayingBiddingRlPolicy;
      playingPolicy: FixedPlayingPolicy;
      rng: () => number;
      temperature: number;
      recordSample: (sample: BiddingRlSampleDraft) => void;
    }
  ) {
    this.ruleBasedAgent = new RuleBasedAgent(options.rng);
  }

  async selectAction(observation: PlayerObservation): Promise<GameAction> {
    return this.selectActionWithContext(observation);
  }

  async selectActionWithContext(
    observation: PlayerObservation,
    context?: { actualState: ActualCardState; playerIds: readonly PlayerId[] }
  ): Promise<GameAction> {
    switch (observation.view.phase) {
      case "bidding":
        return this.selectBiddingAction(observation);
      case "playing":
        return this.selectPlayingAction(observation, context);
      case "choosing-adjutant":
      case "exchanging":
      case "finished":
        return this.ruleBasedAgent.selectAction(observation);
    }
  }

  private async selectBiddingAction(observation: PlayerObservation): Promise<GameAction> {
    const absolutePlayerIds = observation.view.players.map((player) => player.id);
    const encoded = encodeBiddingObservation(observation, absolutePlayerIds);
    const { modelInput, legalBidMask } = createBiddingModelInput(encoded);
    const logits = await this.options.biddingPolicy.predictLogits(modelInput);
    const selection = sampleMaskedCategoricalAction({
      logits,
      legalMask: legalBidMask,
      actionCount: BIDDING_ACTION_COUNT,
      rng: this.options.rng,
      temperature: this.options.temperature
    });
    const selectedAction = decodeBiddingAction(selection.selectedIndex, observation.playerId);
    const legalAction = observation.legalActions.find((action) =>
      biddingActionsEqual(action, selectedAction)
    );

    if (legalAction === undefined) {
      throw new Error(
        `Bidding policy selected action index ${selection.selectedIndex} outside legal actions.`
      );
    }

    this.options.recordSample({
      step: nextDecisionStep(observation),
      playerId: observation.playerId,
      relativePlayerIds: encoded.relativePlayerIds,
      modelInput: Float32Array.from(modelInput),
      legalBidMask: [...legalBidMask],
      selectedActionIndex: selection.selectedIndex,
      behaviorLogProbability: selection.logProbability
    });

    return legalAction;
  }

  private async selectPlayingAction(
    observation: PlayerObservation,
    context: { actualState: ActualCardState; playerIds: readonly PlayerId[] } | undefined
  ): Promise<GameAction> {
    if (observation.publicActionHistory === undefined) {
      throw new Error("Fixed playing policy input requires publicActionHistory.");
    }
    const absolutePlayerIds = context?.playerIds ?? observation.view.players.map((player) => player.id);
    const relativePlayerIds = createRelativePlayerOrder(absolutePlayerIds, observation.playerId);
    const biddingHistory = encodeBiddingHistoryFromPublicActions(
      observation.publicActionHistory,
      relativePlayerIds
    );
    const encoded = encodePlayingObservation(observation, absolutePlayerIds, biddingHistory);
    const { modelInput, legalPlayMask } = createPlayingModelInput(encoded);
    const logits = await this.options.playingPolicy.predictLogits(modelInput);
    const selectedCardIndex = selectHighestLegalIndex(logits, legalPlayMask, CARD_COUNT);
    const selectedCardId = getCardId(selectedCardIndex);
    const selectedAction = observation.legalActions.find(
      (action) => action.type === "play-card" && action.cardId === selectedCardId
    );

    if (selectedAction === undefined) {
      throw new Error(
        `Fixed playing policy selected card index ${selectedCardIndex} (${selectedCardId}) outside legal actions.`
      );
    }

    return selectedAction;
  }
}

class NonPlayingAdjutantRlAgent implements Agent {
  private readonly ruleBasedAgent: RuleBasedAgent;

  constructor(
    private readonly options: {
      adjutantPolicy: NonPlayingAdjutantRlPolicy;
      playingPolicy: FixedPlayingPolicy;
      rng: () => number;
      temperature: number;
      recordSample: (sample: AdjutantRlSampleDraft) => void;
    }
  ) {
    this.ruleBasedAgent = new RuleBasedAgent(options.rng);
  }

  async selectAction(observation: PlayerObservation): Promise<GameAction> {
    return this.selectActionWithContext(observation);
  }

  async selectActionWithContext(
    observation: PlayerObservation,
    context?: { actualState: ActualCardState; playerIds: readonly PlayerId[] }
  ): Promise<GameAction> {
    switch (observation.view.phase) {
      case "choosing-adjutant":
        return this.selectAdjutantAction(observation);
      case "playing":
        return this.selectPlayingAction(observation, context);
      case "bidding":
      case "exchanging":
      case "finished":
        return this.ruleBasedAgent.selectAction(observation);
    }
  }

  private async selectAdjutantAction(observation: PlayerObservation): Promise<GameAction> {
    if (observation.publicActionHistory === undefined) {
      throw new Error("Adjutant policy input requires publicActionHistory.");
    }
    const absolutePlayerIds = observation.view.players.map((player) => player.id);
    const relativePlayerIds = createRelativePlayerOrder(absolutePlayerIds, observation.playerId);
    const biddingHistory = encodeBiddingHistoryFromPublicActions(
      observation.publicActionHistory,
      relativePlayerIds
    );
    const encoded = encodeAdjutantObservation(observation, absolutePlayerIds, biddingHistory);
    const { modelInput, legalAdjutantMask } = createAdjutantModelInput(encoded);
    const logits = await this.options.adjutantPolicy.predictLogits(modelInput);
    const selection = sampleMaskedCategoricalAction({
      logits,
      legalMask: legalAdjutantMask,
      actionCount: CARD_COUNT,
      rng: this.options.rng,
      temperature: this.options.temperature
    });
    const selectedAction = decodeAdjutantAction(selection.selectedIndex, observation.playerId);
    const legalAction = observation.legalActions.find((action) =>
      adjutantActionsEqual(action, selectedAction)
    );

    if (legalAction === undefined) {
      throw new Error(
        `Adjutant policy selected card index ${selection.selectedIndex} outside legal actions.`
      );
    }

    this.options.recordSample({
      step: nextDecisionStep(observation),
      playerId: observation.playerId,
      relativePlayerIds: encoded.relativePlayerIds,
      modelInput: Float32Array.from(modelInput),
      legalAdjutantMask: [...legalAdjutantMask],
      selectedActionIndex: selection.selectedIndex,
      behaviorLogProbability: selection.logProbability
    });

    return legalAction;
  }

  private async selectPlayingAction(
    observation: PlayerObservation,
    context: { actualState: ActualCardState; playerIds: readonly PlayerId[] } | undefined
  ): Promise<GameAction> {
    return selectFixedPlayingAction({
      observation,
      context,
      playingPolicy: this.options.playingPolicy
    });
  }
}

class NonPlayingExchangeRlAgent implements Agent {
  private readonly ruleBasedAgent: RuleBasedAgent;

  constructor(
    private readonly options: {
      exchangePolicy: NonPlayingExchangeRlPolicy;
      playingPolicy: FixedPlayingPolicy;
      rng: () => number;
      temperature: number;
      recordSample: (sample: ExchangeRlSampleDraft) => void;
    }
  ) {
    this.ruleBasedAgent = new RuleBasedAgent(options.rng);
  }

  async selectAction(observation: PlayerObservation): Promise<GameAction> {
    return this.selectActionWithContext(observation);
  }

  async selectActionWithContext(
    observation: PlayerObservation,
    context?: { actualState: ActualCardState; playerIds: readonly PlayerId[] }
  ): Promise<GameAction> {
    switch (observation.view.phase) {
      case "exchanging":
        return this.selectExchangeAction(observation);
      case "playing":
        return selectFixedPlayingAction({
          observation,
          context,
          playingPolicy: this.options.playingPolicy
        });
      case "bidding":
      case "choosing-adjutant":
      case "finished":
        return this.ruleBasedAgent.selectAction(observation);
    }
  }

  private async selectExchangeAction(observation: PlayerObservation): Promise<GameAction> {
    if (observation.publicActionHistory === undefined) {
      throw new Error("Exchange policy input requires publicActionHistory.");
    }
    const absolutePlayerIds = observation.view.players.map((player) => player.id);
    const relativePlayerIds = createRelativePlayerOrder(absolutePlayerIds, observation.playerId);
    const biddingHistory = encodeBiddingHistoryFromPublicActions(
      observation.publicActionHistory,
      relativePlayerIds
    );
    const baseObservation = encodeExchangeObservation(observation, absolutePlayerIds, biddingHistory);
    const selectedActionIndices: number[] = [];

    for (let exchangeStepIndex = 0; exchangeStepIndex < 3; exchangeStepIndex += 1) {
      const steppedObservation = createExchangeStepObservation(baseObservation, selectedActionIndices);
      const { modelInput, legalDiscardCardMask } = createExchangeModelInput(steppedObservation);
      const logits = await this.options.exchangePolicy.predictLogits(modelInput);
      const selection = sampleMaskedCategoricalAction({
        logits,
        legalMask: legalDiscardCardMask,
        actionCount: CARD_COUNT,
        rng: this.options.rng,
        temperature: this.options.temperature
      });

      this.options.recordSample({
        step: nextDecisionStep(observation),
        playerId: observation.playerId,
        relativePlayerIds: steppedObservation.relativePlayerIds,
        exchangeStepIndex: exchangeStepIndex as 0 | 1 | 2,
        remainingDiscardCount: (3 - exchangeStepIndex) as 1 | 2 | 3,
        modelInput: Float32Array.from(modelInput),
        legalDiscardCardMask: [...legalDiscardCardMask],
        selectedActionIndex: selection.selectedIndex,
        behaviorLogProbability: selection.logProbability
      });

      selectedActionIndices.push(selection.selectedIndex);
    }

    const selectedCardIds = selectedActionIndices.map((index) => getCardId(index));
    const legalAction = observation.legalActions.find(
      (action) =>
        action.type === "discard-cards" &&
        action.playerId === observation.playerId &&
        sameStringSet(action.cardIds, selectedCardIds)
    );

    if (legalAction === undefined) {
      throw new Error("Exchange policy selected cards outside legal discard action.");
    }

    return {
      type: "discard-cards",
      playerId: observation.playerId,
      cardIds: selectedCardIds
    };
  }
}

function createExchangeStepObservation(
  observation: EncodedExchangeObservation,
  selectedCardIndices: readonly number[]
): EncodedExchangeObservation {
  const partialDiscardMask = Array(CARD_COUNT).fill(0);
  const legalDiscardCardMask = observation.selfHandMask.map((value) => value);
  const seen = new Set<number>();

  for (const index of selectedCardIndices) {
    if (!Number.isSafeInteger(index) || index < 0 || index >= CARD_COUNT) {
      throw new Error(`Exchange selectedActionIndex is invalid: ${index}.`);
    }
    if (seen.has(index)) {
      throw new Error(`Exchange selectedActionIndex is duplicated: ${index}.`);
    }
    if (observation.selfHandMask[index] !== 1) {
      throw new Error(`Exchange selectedActionIndex is outside self hand: ${index}.`);
    }
    seen.add(index);
    partialDiscardMask[index] = 1;
    legalDiscardCardMask[index] = 0;
  }

  return {
    ...observation,
    partialDiscardMask,
    legalDiscardCardMask,
    exchangeStepIndex: selectedCardIndices.length,
    remainingDiscardCount: 3 - selectedCardIndices.length
  };
}

async function selectFixedPlayingAction(options: {
  observation: PlayerObservation;
  context: { actualState: ActualCardState; playerIds: readonly PlayerId[] } | undefined;
  playingPolicy: FixedPlayingPolicy;
}): Promise<GameAction> {
  const { observation, context } = options;
  if (observation.publicActionHistory === undefined) {
    throw new Error("Fixed playing policy input requires publicActionHistory.");
  }
  const absolutePlayerIds = context?.playerIds ?? observation.view.players.map((player) => player.id);
  const relativePlayerIds = createRelativePlayerOrder(absolutePlayerIds, observation.playerId);
  const biddingHistory = encodeBiddingHistoryFromPublicActions(
    observation.publicActionHistory,
    relativePlayerIds
  );
  const encoded = encodePlayingObservation(observation, absolutePlayerIds, biddingHistory);
  const { modelInput, legalPlayMask } = createPlayingModelInput(encoded);
  const logits = await options.playingPolicy.predictLogits(modelInput);
  const selectedCardIndex = selectHighestLegalIndex(logits, legalPlayMask, CARD_COUNT);
  const selectedCardId = getCardId(selectedCardIndex);
  const selectedAction = observation.legalActions.find(
    (action) => action.type === "play-card" && action.cardId === selectedCardId
  );

  if (selectedAction === undefined) {
    throw new Error(
      `Fixed playing policy selected card index ${selectedCardIndex} (${selectedCardId}) outside legal actions.`
    );
  }

  return selectedAction;
}

function createBiddingRlOutcome(
  result: GameResult,
  actingPlayerId: PlayerId
): NonPlayingBiddingRlOutcome {
  return {
    winner: result.winner,
    targetPointCards: result.targetPointCards,
    napoleonPlayerId: result.napoleonPlayerId,
    actingPlayerRole: getNonPlayingRole(result, actingPlayerId)
  };
}

function getNonPlayingRole(result: GameResult, playerId: PlayerId): NonPlayingBiddingRlRole {
  if (playerId === result.napoleonPlayerId) {
    return result.adjutantPlayerId === null || result.adjutantPlayerId === playerId
      ? "napoleon-adjutant"
      : "napoleon";
  }
  if (playerId === result.adjutantPlayerId) {
    return "adjutant";
  }
  return "citizen";
}

function nextDecisionStep(observation: PlayerObservation): number {
  const history = observation.publicActionHistory ?? [];
  if (history.length === 0) {
    return 1;
  }
  return Math.max(...history.map((record) => record.step)) + 1;
}

function sampleMaskedCategoricalAction(options: {
  logits: Float32Array | readonly number[];
  legalMask: ArrayLike<number | boolean>;
  actionCount: number;
  rng: () => number;
  temperature: number;
}): { selectedIndex: number; logProbability: number } {
  const distribution = createMaskedCategoricalDistribution(
    options.logits,
    options.legalMask,
    options.actionCount,
    options.temperature
  );

  if (distribution.legalIndices.length === 1) {
    return {
      selectedIndex: distribution.legalIndices[0],
      logProbability: 0
    };
  }

  const randomValue = options.rng();
  if (!Number.isFinite(randomValue) || randomValue < 0 || randomValue >= 1) {
    throw new Error("rng must return a finite value in [0, 1).");
  }

  let cumulativeProbability = 0;
  for (let index = 0; index < distribution.legalIndices.length; index += 1) {
    cumulativeProbability += distribution.probabilities[index];

    if (randomValue < cumulativeProbability) {
      return {
        selectedIndex: distribution.legalIndices[index],
        logProbability: distribution.logProbabilities[index]
      };
    }
  }

  const lastIndex = distribution.legalIndices.length - 1;
  return {
    selectedIndex: distribution.legalIndices[lastIndex],
    logProbability: distribution.logProbabilities[lastIndex]
  };
}

function createMaskedCategoricalDistribution(
  logits: Float32Array | readonly number[],
  legalMask: ArrayLike<number | boolean>,
  actionCount: number,
  temperature: number
): {
  legalIndices: readonly number[];
  probabilities: readonly number[];
  logProbabilities: readonly number[];
} {
  if (logits.length !== actionCount) {
    throw new Error(`logits must contain ${actionCount} values, got ${logits.length}.`);
  }
  validateTemperature(temperature);

  const legalIndices: number[] = [];
  const scaledLogits: number[] = [];

  for (let index = 0; index < actionCount; index += 1) {
    const logit = Number(logits[index]);

    if (!Number.isFinite(logit)) {
      throw new Error(`logits[${index}] must be finite.`);
    }

    if (isLegalMaskValue(legalMask[index])) {
      legalIndices.push(index);
      scaledLogits.push(logit / temperature);
    }
  }

  if (legalIndices.length === 0) {
    throw new Error("legal mask must contain at least one legal action.");
  }
  if (legalIndices.length === 1) {
    return {
      legalIndices,
      probabilities: [1],
      logProbabilities: [0]
    };
  }

  const maxScaledLogit = Math.max(...scaledLogits);
  const expValues = scaledLogits.map((logit) => Math.exp(logit - maxScaledLogit));
  const expSum = expValues.reduce((sumValue, value) => sumValue + value, 0);

  if (!Number.isFinite(expSum) || expSum <= 0) {
    throw new Error("masked categorical softmax normalization failed.");
  }

  const logDenominator = maxScaledLogit + Math.log(expSum);
  const probabilities = expValues.map((value) => value / expSum);
  const logProbabilities = scaledLogits.map((logit) => logit - logDenominator);

  return { legalIndices, probabilities, logProbabilities };
}

function selectHighestLegalIndex(
  logits: Float32Array | readonly number[],
  legalMask: ArrayLike<number | boolean>,
  actionCount: number
): number {
  if (logits.length !== actionCount) {
    throw new Error(`logits must contain ${actionCount} values, got ${logits.length}.`);
  }

  let selectedIndex = -1;
  let selectedLogit = Number.NEGATIVE_INFINITY;

  for (let index = 0; index < actionCount; index += 1) {
    const logit = Number(logits[index]);
    if (!Number.isFinite(logit)) {
      throw new Error(`logits[${index}] must be finite.`);
    }
    if (!isLegalMaskValue(legalMask[index])) {
      continue;
    }
    if (selectedIndex === -1 || logit > selectedLogit) {
      selectedIndex = index;
      selectedLogit = logit;
    }
  }

  if (selectedIndex === -1) {
    throw new Error("legal mask must contain at least one legal action.");
  }

  return selectedIndex;
}

function createNonPlayingRlDatasetManifest(input: {
  options: GenerateNonPlayingBiddingRlDatasetOptions;
  temperature: number;
  sampleCount: number;
  shards: readonly DatasetShardManifest[];
  behaviorPolicy: NonPlayingRlPolicyArtifactManifest;
  fixedPlayingPolicy: NonPlayingRlPolicyArtifactManifest;
}): NonPlayingRlDatasetManifest {
  return {
    datasetSchemaVersion: NON_PLAYING_RL_DATASET_SCHEMA_VERSION,
    generatorVersion: NON_PLAYING_RL_DATASET_GENERATOR_VERSION,
    format: DATASET_FORMAT,
    sampleType: NON_PLAYING_RL_DATASET_SAMPLE_TYPE,
    sampleSchemaVersion: NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION,
    phaseScope: NON_PLAYING_RL_PHASE_SCOPE,
    learnedPhases: ["bidding"],
    ruleBasedPhases: ["choosing-adjutant", "exchanging"],
    fixedPhases: ["playing"],
    startSeed: input.options.startSeed,
    endSeed: input.options.startSeed + input.options.gameCount - 1,
    gameCount: input.options.gameCount,
    sampleCount: input.sampleCount,
    gamesPerShard: input.options.gamesPerShard,
    shardCount: input.shards.length,
    playerCount: PLAYER_COUNT,
    cardCount: CARD_COUNT,
    cardIds: CARD_IDS,
    cardIdsSha256: calculateCardIdsSha256(),
    biddingEncoderSchemaVersion: BIDDING_ENCODER_SCHEMA_VERSION,
    biddingModelInputSchemaVersion: BIDDING_MODEL_INPUT_SCHEMA_VERSION,
    biddingModelInputFeatureCount: BIDDING_MODEL_INPUT_FEATURE_COUNT,
    playingModelInputSchemaVersion: MODEL_INPUT_SCHEMA_VERSION,
    playingModelInputFeatureCount: MODEL_INPUT_FEATURE_COUNT,
    actionCount: BIDDING_ACTION_COUNT,
    behaviorPolicy: input.behaviorPolicy,
    fixedPlayingPolicy: input.fixedPlayingPolicy,
    samplingAlgorithm: NON_PLAYING_RL_SAMPLING_ALGORITHM,
    temperature: input.temperature,
    reward: {
      type: NON_PLAYING_RL_REWARD_TYPE,
      version: NON_PLAYING_RL_REWARD_VERSION,
      id: NON_PLAYING_RL_REWARD_ID
    },
    nonLearningAgents: {
      choosingAdjutant: {
        type: "rule-based",
        version: RULE_BASED_AGENT_VERSION
      },
      exchanging: {
        type: "rule-based",
        version: RULE_BASED_AGENT_VERSION
      }
    },
    shards: input.shards
  };
}

function createNonPlayingAdjutantRlDatasetManifest(input: {
  options: GenerateNonPlayingAdjutantRlDatasetOptions;
  temperature: number;
  sampleCount: number;
  shards: readonly DatasetShardManifest[];
  behaviorPolicy: NonPlayingRlPolicyArtifactManifest;
  fixedPlayingPolicy: NonPlayingRlPolicyArtifactManifest;
}): NonPlayingRlDatasetManifest {
  return {
    datasetSchemaVersion: NON_PLAYING_RL_DATASET_SCHEMA_VERSION,
    generatorVersion: NON_PLAYING_RL_DATASET_GENERATOR_VERSION,
    format: DATASET_FORMAT,
    sampleType: NON_PLAYING_ADJUTANT_RL_DATASET_SAMPLE_TYPE,
    sampleSchemaVersion: NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION,
    phaseScope: NON_PLAYING_ADJUTANT_RL_PHASE_SCOPE,
    learnedPhases: ["choosing-adjutant"],
    ruleBasedPhases: ["bidding", "exchanging"],
    fixedPhases: ["playing"],
    startSeed: input.options.startSeed,
    endSeed: input.options.startSeed + input.options.gameCount - 1,
    gameCount: input.options.gameCount,
    sampleCount: input.sampleCount,
    gamesPerShard: input.options.gamesPerShard,
    shardCount: input.shards.length,
    playerCount: PLAYER_COUNT,
    cardCount: CARD_COUNT,
    cardIds: CARD_IDS,
    cardIdsSha256: calculateCardIdsSha256(),
    biddingEncoderSchemaVersion: BIDDING_ENCODER_SCHEMA_VERSION,
    biddingModelInputSchemaVersion: BIDDING_MODEL_INPUT_SCHEMA_VERSION,
    biddingModelInputFeatureCount: BIDDING_MODEL_INPUT_FEATURE_COUNT,
    adjutantEncoderSchemaVersion: ADJUTANT_ENCODER_SCHEMA_VERSION,
    adjutantModelInputSchemaVersion: ADJUTANT_MODEL_INPUT_SCHEMA_VERSION,
    adjutantModelInputFeatureCount: ADJUTANT_MODEL_INPUT_FEATURE_COUNT,
    playingModelInputSchemaVersion: MODEL_INPUT_SCHEMA_VERSION,
    playingModelInputFeatureCount: MODEL_INPUT_FEATURE_COUNT,
    actionCount: CARD_COUNT,
    behaviorPolicy: input.behaviorPolicy,
    fixedPlayingPolicy: input.fixedPlayingPolicy,
    samplingAlgorithm: NON_PLAYING_RL_SAMPLING_ALGORITHM,
    temperature: input.temperature,
    reward: {
      type: NON_PLAYING_RL_REWARD_TYPE,
      version: NON_PLAYING_RL_REWARD_VERSION,
      id: NON_PLAYING_RL_REWARD_ID
    },
    nonLearningAgents: {
      bidding: {
        type: "rule-based",
        version: RULE_BASED_AGENT_VERSION
      },
      choosingAdjutant: {
        type: "rule-based",
        version: RULE_BASED_AGENT_VERSION
      },
      exchanging: {
        type: "rule-based",
        version: RULE_BASED_AGENT_VERSION
      }
    },
    shards: input.shards
  };
}

function createNonPlayingExchangeRlDatasetManifest(input: {
  options: GenerateNonPlayingExchangeRlDatasetOptions;
  temperature: number;
  sampleCount: number;
  shards: readonly DatasetShardManifest[];
  behaviorPolicy: NonPlayingRlPolicyArtifactManifest;
  fixedPlayingPolicy: NonPlayingRlPolicyArtifactManifest;
}): NonPlayingRlDatasetManifest {
  return {
    datasetSchemaVersion: NON_PLAYING_RL_DATASET_SCHEMA_VERSION,
    generatorVersion: NON_PLAYING_RL_DATASET_GENERATOR_VERSION,
    format: DATASET_FORMAT,
    sampleType: NON_PLAYING_EXCHANGE_RL_DATASET_SAMPLE_TYPE,
    sampleSchemaVersion: NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION,
    phaseScope: NON_PLAYING_EXCHANGE_RL_PHASE_SCOPE,
    learnedPhases: ["exchanging"],
    ruleBasedPhases: ["bidding", "choosing-adjutant"],
    fixedPhases: ["playing"],
    startSeed: input.options.startSeed,
    endSeed: input.options.startSeed + input.options.gameCount - 1,
    gameCount: input.options.gameCount,
    sampleCount: input.sampleCount,
    gamesPerShard: input.options.gamesPerShard,
    shardCount: input.shards.length,
    playerCount: PLAYER_COUNT,
    cardCount: CARD_COUNT,
    cardIds: CARD_IDS,
    cardIdsSha256: calculateCardIdsSha256(),
    biddingEncoderSchemaVersion: BIDDING_ENCODER_SCHEMA_VERSION,
    biddingModelInputSchemaVersion: BIDDING_MODEL_INPUT_SCHEMA_VERSION,
    biddingModelInputFeatureCount: BIDDING_MODEL_INPUT_FEATURE_COUNT,
    exchangeEncoderSchemaVersion: EXCHANGE_ENCODER_SCHEMA_VERSION,
    exchangeModelInputSchemaVersion: EXCHANGE_MODEL_INPUT_SCHEMA_VERSION,
    exchangeModelInputFeatureCount: EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
    decisionMode: NON_PLAYING_EXCHANGE_RL_DECISION_MODE,
    playingModelInputSchemaVersion: MODEL_INPUT_SCHEMA_VERSION,
    playingModelInputFeatureCount: MODEL_INPUT_FEATURE_COUNT,
    actionCount: CARD_COUNT,
    behaviorPolicy: input.behaviorPolicy,
    fixedPlayingPolicy: input.fixedPlayingPolicy,
    samplingAlgorithm: NON_PLAYING_RL_SAMPLING_ALGORITHM,
    temperature: input.temperature,
    reward: {
      type: NON_PLAYING_RL_REWARD_TYPE,
      version: NON_PLAYING_RL_REWARD_VERSION,
      id: NON_PLAYING_RL_REWARD_ID
    },
    nonLearningAgents: {
      bidding: {
        type: "rule-based",
        version: RULE_BASED_AGENT_VERSION
      },
      choosingAdjutant: {
        type: "rule-based",
        version: RULE_BASED_AGENT_VERSION
      }
    },
    shards: input.shards
  };
}

async function createPolicyArtifactManifest(input: {
  type: NonPlayingRlPolicyArtifactManifest["type"];
  artifact: NonPlayingRlPolicyArtifactOptions;
  policy: { metadata: unknown; runtime?: PolicyRuntimeInfo };
}): Promise<NonPlayingRlPolicyArtifactManifest> {
  return {
    type: input.type,
    artifactId: input.artifact.artifactId ?? basename(input.artifact.metadataPath),
    onnxFileName: basename(input.artifact.onnxPath),
    metadataFileName: basename(input.artifact.metadataPath),
    onnxSha256: await sha256File(input.artifact.onnxPath),
    metadataSha256: await sha256File(input.artifact.metadataPath),
    ...runtimeInfoForPolicy(input.policy),
    metadata: input.policy.metadata
  };
}

function runtimeInfoForPolicy(policy: {
  runtime?: PolicyRuntimeInfo;
}): Pick<
  NonPlayingRlPolicyArtifactManifest,
  "requestedInferenceDevice" | "resolvedInferenceDevice" | "executionProvider"
> {
  return {
    requestedInferenceDevice: policy.runtime?.requestedInferenceDevice,
    resolvedInferenceDevice: policy.runtime?.resolvedInferenceDevice,
    executionProvider: policy.runtime?.executionProvider
  };
}

function validatePolicyArtifactManifest(
  artifact: NonPlayingRlPolicyArtifactManifest,
  expectedType: NonPlayingRlPolicyArtifactManifest["type"]
): void {
  if (
    artifact.type !== expectedType ||
    artifact.artifactId.length === 0 ||
    artifact.onnxFileName.length === 0 ||
    artifact.metadataFileName.length === 0 ||
    !sha256Pattern.test(artifact.onnxSha256) ||
    !sha256Pattern.test(artifact.metadataSha256)
  ) {
    throw new Error(`Non-playing RL ${expectedType} artifact metadata mismatch.`);
  }
}

function validateShards(manifest: NonPlayingRlDatasetManifest): void {
  let expectedStartSeed = manifest.startSeed;
  const seenFiles = new Set<string>();

  manifest.shards.forEach((shard, index) => {
    validateShard(shard);

    const expectedFile = `shard-${index.toString().padStart(5, "0")}.jsonl`;

    if (shard.file !== expectedFile) {
      throw new Error(`Shard file must be ${expectedFile}, got ${shard.file}.`);
    }
    if (seenFiles.has(shard.file)) {
      throw new Error(`Duplicate shard file: ${shard.file}`);
    }
    seenFiles.add(shard.file);
    if (shard.startSeed !== expectedStartSeed) {
      throw new Error(`Shard ${shard.file} has a seed gap or overlap.`);
    }
    if (shard.gameCount !== shard.endSeed - shard.startSeed + 1) {
      throw new Error(`Shard ${shard.file} gameCount must match seed range.`);
    }
    if (index !== manifest.shards.length - 1 && shard.gameCount !== manifest.gamesPerShard) {
      throw new Error(`Shard ${shard.file} gameCount must match gamesPerShard.`);
    }
    if (index === manifest.shards.length - 1 && shard.gameCount > manifest.gamesPerShard) {
      throw new Error(`Final shard ${shard.file} gameCount must not exceed gamesPerShard.`);
    }
    expectedStartSeed = shard.endSeed + 1;
  });

  if (manifest.shards[0]?.startSeed !== manifest.startSeed) {
    throw new Error("First shard must start at dataset startSeed.");
  }
  if (manifest.shards.at(-1)?.endSeed !== manifest.endSeed) {
    throw new Error("Last shard must end at dataset endSeed.");
  }
}

function validateShard(shard: DatasetShardManifest): void {
  validateUint32(`Shard ${shard.file} startSeed`, shard.startSeed);
  validateUint32(`Shard ${shard.file} endSeed`, shard.endSeed);
  validatePositiveInteger(`Shard ${shard.file} gameCount`, shard.gameCount);
  validatePositiveInteger(`Shard ${shard.file} sampleCount`, shard.sampleCount);
  validatePositiveInteger(`Shard ${shard.file} byteLength`, shard.byteLength);
  if (!sha256Pattern.test(shard.sha256)) {
    throw new Error(`Shard ${shard.file} sha256 must be lowercase hex.`);
  }
}

function validateOutcome(outcome: NonPlayingBiddingRlOutcome): void {
  if (outcome.winner !== "napoleon-team" && outcome.winner !== "alliance") {
    throw new Error("outcome.winner is invalid.");
  }
  validatePositiveInteger("outcome.targetPointCards", outcome.targetPointCards);
  if (!isNonPlayingBiddingRlRole(outcome.actingPlayerRole)) {
    throw new Error("outcome.actingPlayerRole is invalid.");
  }
  if (outcome.napoleonPlayerId.length === 0) {
    throw new Error("outcome.napoleonPlayerId must be non-empty.");
  }
}

function validateLegalBidMask(mask: readonly number[]): void {
  expectLength("legalBidMask", mask, BIDDING_ACTION_COUNT);
  let legalCount = 0;

  for (const value of mask) {
    if (!Number.isFinite(value) || !Number.isInteger(value) || (value !== 0 && value !== 1)) {
      throw new Error("legalBidMask must contain only 0/1 values.");
    }
    legalCount += value;
  }

  if (legalCount === 0) {
    throw new Error("legalBidMask must contain at least one legal action.");
  }
}

function validateLegalAdjutantMask(mask: readonly number[]): void {
  expectLength("legalAdjutantMask", mask, CARD_COUNT);
  let legalCount = 0;

  for (const value of mask) {
    if (!Number.isFinite(value) || !Number.isInteger(value) || (value !== 0 && value !== 1)) {
      throw new Error("legalAdjutantMask must contain only 0/1 values.");
    }
    legalCount += value;
  }

  if (legalCount === 0) {
    throw new Error("legalAdjutantMask must contain at least one legal action.");
  }
}

function validateLegalDiscardMask(mask: readonly number[]): void {
  expectLength("legalDiscardCardMask", mask, CARD_COUNT);
  let legalCount = 0;

  for (const value of mask) {
    if (!Number.isFinite(value) || !Number.isInteger(value) || (value !== 0 && value !== 1)) {
      throw new Error("legalDiscardCardMask must contain only 0/1 values.");
    }
    legalCount += value;
  }

  if (legalCount === 0) {
    throw new Error("legalDiscardCardMask must contain at least one legal action.");
  }
}

function validateBiddingActionIndex(value: number): void {
  if (!Number.isSafeInteger(value) || value < 0 || value >= BIDDING_ACTION_COUNT) {
    throw new Error(`selectedActionIndex must be between 0 and ${BIDDING_ACTION_COUNT - 1}.`);
  }
}

function validateCardActionIndex(value: number): void {
  if (!Number.isSafeInteger(value) || value < 0 || value >= CARD_COUNT) {
    throw new Error(`selectedActionIndex must be between 0 and ${CARD_COUNT - 1}.`);
  }
}

function validatePlayerIndex(name: string, value: number): void {
  if (!Number.isSafeInteger(value) || value < 0 || value >= PLAYER_COUNT) {
    throw new Error(`${name} must be between 0 and ${PLAYER_COUNT - 1}.`);
  }
}

function validateUint32(name: string, value: number): void {
  if (!Number.isSafeInteger(value) || value < 0 || value > UINT32_MAX) {
    throw new Error(`${name} must be an integer between 0 and ${UINT32_MAX}.`);
  }
}

function validatePositiveInteger(name: string, value: number): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
}

function validateTemperature(value: number): void {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`temperature must be finite and positive, got ${value}.`);
  }
}

function expectLength(name: string, value: readonly unknown[], expectedLength: number): void {
  if (value.length !== expectedLength) {
    throw new Error(`${name} must have length ${expectedLength}, got ${value.length}.`);
  }
}

function isLegalMaskValue(value: unknown): boolean {
  return value === 1 || value === true;
}

function isNonPlayingBiddingRlRole(value: string): value is NonPlayingBiddingRlRole {
  return (
    value === "napoleon" ||
    value === "adjutant" ||
    value === "citizen" ||
    value === "napoleon-adjutant"
  );
}

function biddingActionsEqual(left: GameAction, right: GameAction): boolean {
  if (left.type !== right.type || left.playerId !== right.playerId) {
    return false;
  }
  if (left.type === "pass") {
    return right.type === "pass";
  }
  if (left.type !== "bid" || right.type !== "bid") {
    return false;
  }
  return right.type === "bid" &&
    left.suit === right.suit &&
    left.targetPointCards === right.targetPointCards;
}

function adjutantActionsEqual(left: GameAction, right: GameAction): boolean {
  return left.type === "choose-adjutant" &&
    right.type === "choose-adjutant" &&
    left.playerId === right.playerId &&
    left.cardId === right.cardId;
}

function sameStringSet(left: readonly string[], right: readonly string[]): boolean {
  if (left.length !== right.length) {
    return false;
  }
  const rightSet = new Set(right);
  return left.every((value) => rightSet.has(value));
}

function calculateExpectedShardCount(gameCount: number, gamesPerShard: number): number {
  const expectedShardCount = Math.ceil(gameCount / gamesPerShard);

  if (!Number.isSafeInteger(expectedShardCount) || expectedShardCount < 1) {
    throw new Error("Expected shard count must be a positive safe integer.");
  }

  return expectedShardCount;
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

async function sha256File(path: string): Promise<string> {
  return createHash("sha256").update(await readFile(path)).digest("hex");
}

const sha256Pattern = /^[0-9a-f]{64}$/;

function validateJsonSafeValue(name: string, value: unknown, seen = new WeakSet<object>()): void {
  if (value === undefined) {
    throw new Error(`${name} must not contain undefined.`);
  }
  if (typeof value === "number" && !Number.isFinite(value)) {
    throw new Error(`${name} must not contain NaN or Infinity.`);
  }
  if (typeof value === "function" || typeof value === "symbol") {
    throw new Error(`${name} must not contain unserializable values.`);
  }
  if (value === null || typeof value !== "object") {
    return;
  }
  if (seen.has(value)) {
    throw new Error(`${name} must not contain circular references.`);
  }

  seen.add(value);
  if (Array.isArray(value)) {
    value.forEach((entry, index) => validateJsonSafeValue(`${name}[${index}]`, entry, seen));
    seen.delete(value);
    return;
  }
  Object.entries(value).forEach(([key, entry]) =>
    validateJsonSafeValue(`${name}.${key}`, entry, seen)
  );
  seen.delete(value);
}

function sameStringArray(left: readonly string[], right: readonly string[]): boolean {
  if (left.length !== right.length) {
    return false;
  }

  return left.every((value, index) => value === right[index]);
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}
