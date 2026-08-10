import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { gzipSync } from "node:zlib";
import { RuleBasedAgent, runAutomatedGame } from "@napoleon/ai";
import type { Agent, AutomatedGameRecord, DecisionRecord, PlayerObservation } from "@napoleon/ai";
import type { GameAction, PlayerId, WinningTeam } from "@napoleon/game-core";
import {
  CARD_COUNT,
  CARD_IDS,
  MODEL_INPUT_FEATURE_COUNT,
  MODEL_INPUT_SCHEMA_VERSION,
  PLAYER_COUNT,
  PLAYING_ENCODER_SCHEMA_VERSION,
  SELF_ROLE_ORDER,
  createPlayingModelInput,
  createRelativePlayerOrder,
  encodeBiddingHistory,
  encodeBiddingHistoryFromPublicActions,
  encodePlayingObservation,
  getCardId,
  getCardIndex,
  validateEncodedPlayingObservation
} from "@napoleon/ai-observation";
import {
  DATASET_FORMAT,
  MAX_SHARD_COUNT,
  PLAYING_SELF_PLAY_BINARY_DATASET_FORMAT,
  RULE_BASED_AGENT_VERSION,
  UINT32_MAX
} from "./schema.js";
import type { DatasetGenerationProgress, DatasetShardManifest } from "./types.js";
import { createJsonlShardWriter } from "./shardWriter.js";
import { calculateCardIdsSha256, serializeManifest } from "./serialization.js";

export const PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE = "playing-self-play-sample" as const;
export const PLAYING_SELF_PLAY_SAMPLE_SCHEMA_VERSION = 4 as const;
export const PLAYING_SELF_PLAY_LEGACY_SAMPLE_SCHEMA_VERSION = 3 as const;
export const PLAYING_SELF_PLAY_DATASET_SCHEMA_VERSION = 4 as const;
export const PLAYING_SELF_PLAY_LEGACY_DATASET_SCHEMA_VERSION = 3 as const;
export const PLAYING_SELF_PLAY_DATASET_GENERATOR_VERSION = 1 as const;
export const PLAYING_SELF_PLAY_BINARY_SHARD_SCHEMA_VERSION = 1 as const;
export const PLAYING_SELF_PLAY_REWARD_TYPE = "terminal-team-win" as const;
export const PLAYING_SELF_PLAY_REWARD_VERSION = 1 as const;
export const PLAYING_SELF_PLAY_SAMPLING_ALGORITHM = "masked-categorical" as const;
export const DEFAULT_PLAYING_SELF_PLAY_TEMPERATURE = 1.0 as const;
export const PLAYING_SELF_PLAY_ROSTER_ASSIGNMENT = "rotate-by-seed" as const;
export const CURRENT_POLICY_ROSTER_SOURCE = "current-policy" as const;
export const RULE_BASED_ROSTER_SOURCE = "rule-based" as const;
export const FROZEN_ONNX_ROSTER_SOURCE = "frozen-onnx" as const;

export type PlayingSelfPlayRole = typeof SELF_ROLE_ORDER[number];
export type PlayingSelfPlayRosterSource =
  | typeof CURRENT_POLICY_ROSTER_SOURCE
  | typeof RULE_BASED_ROSTER_SOURCE
  | typeof FROZEN_ONNX_ROSTER_SOURCE;

export interface CurrentPolicyRolloutRosterSeat {
  source: typeof CURRENT_POLICY_ROSTER_SOURCE;
}

export interface RuleBasedRolloutRosterSeat {
  source: typeof RULE_BASED_ROSTER_SOURCE;
}

export interface FrozenOnnxRolloutRosterSeat {
  source: typeof FROZEN_ONNX_ROSTER_SOURCE;
  policy: PlayingSelfPlayPolicy;
  artifact: PlayingSelfPlayPolicyArtifactOptions;
}

export type RolloutRosterSeatOptions =
  | CurrentPolicyRolloutRosterSeat
  | RuleBasedRolloutRosterSeat
  | FrozenOnnxRolloutRosterSeat;

export interface PlayingSelfPlayRolloutRosterOptions {
  seats: readonly RolloutRosterSeatOptions[];
}

export type PlayingSelfPlayRolloutRosterSeatManifest =
  | { source: typeof CURRENT_POLICY_ROSTER_SOURCE }
  | { source: typeof RULE_BASED_ROSTER_SOURCE; version: typeof RULE_BASED_AGENT_VERSION }
  | {
      source: typeof FROZEN_ONNX_ROSTER_SOURCE;
      artifactId: string;
      onnxFileName: string;
      metadataFileName: string;
      onnxSha256: string;
      metadataSha256: string;
      requestedInferenceDevice: PlayingSelfPlayPolicyRuntime["requestedInferenceDevice"];
      resolvedInferenceDevice: PlayingSelfPlayPolicyRuntime["resolvedInferenceDevice"];
      executionProvider: PlayingSelfPlayPolicyRuntime["executionProvider"];
      metadata: unknown;
    };

export interface PlayingSelfPlayRolloutRosterManifest {
  assignment: typeof PLAYING_SELF_PLAY_ROSTER_ASSIGNMENT;
  seats: readonly PlayingSelfPlayRolloutRosterSeatManifest[];
}

export interface PlayingSelfPlayOutcome {
  winner: WinningTeam;
  napoleonPlayerId: PlayerId;
  actingPlayerTeam: WinningTeam;
  actingPlayerRole: PlayingSelfPlayRole;
}

export interface PlayingSelfPlaySample {
  sampleType: typeof PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE;
  schemaVersion: typeof PLAYING_SELF_PLAY_LEGACY_SAMPLE_SCHEMA_VERSION;
  seed: number;
  step: number;
  actingPlayerId: PlayerId;
  actingSeatSource: typeof CURRENT_POLICY_ROSTER_SOURCE;
  behaviorPolicyArtifactId: string;
  rolloutSeatSources: readonly PlayingSelfPlayRosterSource[];
  relativePlayerIds: readonly PlayerId[];
  observation: ReturnType<typeof encodePlayingObservation>;
  selectedCardIndex: number;
  behaviorLogProbability: number;
  terminalReward: 1 | -1;
  outcome: PlayingSelfPlayOutcome;
}

export interface PlayingSelfPlayTensorSample {
  sampleType: typeof PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE;
  schemaVersion: typeof PLAYING_SELF_PLAY_SAMPLE_SCHEMA_VERSION;
  seed: number;
  step: number;
  actingPlayerIndex: number;
  selectedCardIndex: number;
  behaviorLogProbability: number;
  terminalReward: 1 | -1;
  selfRoleIndex: number;
  modelInput: Float32Array;
  legalPlayMask: Uint8Array;
}

export interface PlayingSelfPlayPolicy {
  metadata: unknown;
  runtime?: PlayingSelfPlayPolicyRuntime;
  predictLogits: (modelInput: Float32Array | readonly number[]) => Promise<Float32Array>;
}

export interface PlayingSelfPlayPolicyRuntime {
  requestedInferenceDevice: "cpu" | "auto" | "cuda";
  resolvedInferenceDevice: "cpu" | "cuda";
  executionProvider: "cpu" | "cuda";
}

export interface PlayingSelfPlayPolicyArtifactOptions {
  onnxPath: string;
  metadataPath: string;
  artifactId?: string;
}

export interface PlayingSelfPlaySampledAction {
  selectedCardIndex: number;
  logProbability: number;
}

export interface GeneratePlayingSelfPlayDatasetOptions {
  outputDirectory: string;
  playingPolicy: PlayingSelfPlayPolicy;
  playingPolicyArtifact: PlayingSelfPlayPolicyArtifactOptions;
  startSeed: number;
  gameCount: number;
  gamesPerShard: number;
  rolloutWorkers?: number;
  temperature?: number;
  rolloutRoster?: PlayingSelfPlayRolloutRosterOptions;
  onProgress?: (progress: DatasetGenerationProgress) => void;
  maxDecisionSteps?: number;
  gameRunner?: PlayingSelfPlayGameRunner;
  format?: typeof PLAYING_SELF_PLAY_BINARY_DATASET_FORMAT | typeof DATASET_FORMAT;
  binaryCompression?: PlayingSelfPlayBinaryCompression;
}

export interface PlayingSelfPlayGameRunRequest {
  gameOffset: number;
  seed: number;
  currentPolicy: PlayingSelfPlayPolicy;
  behaviorPolicyArtifactId: string;
  rolloutRoster: PlayingSelfPlayRolloutRosterOptions | undefined;
  temperature: number;
  maxDecisionSteps: number | undefined;
}

export interface PlayingSelfPlayGameRunResult {
  seed: number;
  record?: AutomatedGameRecord;
  samples?: readonly PlayingSelfPlaySample[];
  tensorSamples?: readonly PlayingSelfPlayTensorSample[];
}

export interface PlayingSelfPlayGameRunner {
  runGame: (
    request: PlayingSelfPlayGameRunRequest
  ) => Promise<AutomatedGameRecord | PlayingSelfPlayGameRunResult>;
  close?: () => Promise<void>;
}

export interface PlayingSelfPlayDatasetManifest {
  datasetSchemaVersion: typeof PLAYING_SELF_PLAY_DATASET_SCHEMA_VERSION | typeof PLAYING_SELF_PLAY_LEGACY_DATASET_SCHEMA_VERSION;
  generatorVersion: typeof PLAYING_SELF_PLAY_DATASET_GENERATOR_VERSION;
  format: typeof PLAYING_SELF_PLAY_BINARY_DATASET_FORMAT | typeof DATASET_FORMAT;
  sampleType: typeof PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE;
  sampleSchemaVersion: typeof PLAYING_SELF_PLAY_SAMPLE_SCHEMA_VERSION | typeof PLAYING_SELF_PLAY_LEGACY_SAMPLE_SCHEMA_VERSION;
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
  playingEncoderSchemaVersion: typeof PLAYING_ENCODER_SCHEMA_VERSION;
  playingModelInputSchemaVersion: typeof MODEL_INPUT_SCHEMA_VERSION;
  behaviorPolicy: {
    type: "playing-onnx";
    artifactId: string;
    onnxFileName: string;
    metadataFileName: string;
    onnxSha256: string;
    metadataSha256: string;
    requestedInferenceDevice: PlayingSelfPlayPolicyRuntime["requestedInferenceDevice"];
    resolvedInferenceDevice: PlayingSelfPlayPolicyRuntime["resolvedInferenceDevice"];
    executionProvider: PlayingSelfPlayPolicyRuntime["executionProvider"];
    metadata: unknown;
  };
  samplingAlgorithm: typeof PLAYING_SELF_PLAY_SAMPLING_ALGORITHM;
  temperature: number;
  reward: {
    type: typeof PLAYING_SELF_PLAY_REWARD_TYPE;
    version: typeof PLAYING_SELF_PLAY_REWARD_VERSION;
  };
  nonPlayingAgent: {
    type: "rule-based";
    version: typeof RULE_BASED_AGENT_VERSION;
  };
  rolloutRoster: PlayingSelfPlayRolloutRosterManifest;
  tensorSchema?: PlayingSelfPlayBinaryTensorSchema;
}

export interface PlayingSelfPlayBinaryTensorFieldSchema {
  name: keyof PlayingSelfPlayBinaryTensorFieldTypes;
  dtype: PlayingSelfPlayBinaryTensorFieldTypes[keyof PlayingSelfPlayBinaryTensorFieldTypes];
  shape: readonly number[];
}

export interface PlayingSelfPlayBinaryTensorSchema {
  shardSchemaVersion: typeof PLAYING_SELF_PLAY_BINARY_SHARD_SCHEMA_VERSION;
  byteOrder: "little-endian";
  compression: PlayingSelfPlayBinaryCompression;
  fields: readonly PlayingSelfPlayBinaryTensorFieldSchema[];
}

export type PlayingSelfPlayBinaryCompression = "none" | "gzip";

interface PlayingSelfPlayBinaryTensorFieldTypes {
  modelInput: "float32";
  legalPlayMask: "uint8";
  selectedCardIndex: "uint8";
  behaviorLogProbability: "float32";
  terminalReward: "int8";
  seed: "uint32";
  step: "uint16";
  actingPlayerIndex: "uint8";
  selfRoleIndex: "uint8";
}

export interface GeneratePlayingSelfPlayDatasetResult {
  outputDirectory: string;
  manifest: PlayingSelfPlayDatasetManifest;
}

type PlayingSelfPlayWritableSample = PlayingSelfPlaySample | PlayingSelfPlayTensorSample;

interface PlayingSelfPlayShardWriter {
  readonly sampleCount: number;
  writeSample: (sample: PlayingSelfPlayWritableSample) => Promise<void>;
  close: (endSeed: number, gameCount: number) => Promise<DatasetShardManifest>;
  abort: () => Promise<void>;
}

const BINARY_SHARD_MAGIC = Buffer.from("NPSPBD01", "ascii");

const PLAYING_SELF_PLAY_BINARY_TENSOR_SCHEMA: PlayingSelfPlayBinaryTensorSchema = {
  shardSchemaVersion: PLAYING_SELF_PLAY_BINARY_SHARD_SCHEMA_VERSION,
  byteOrder: "little-endian",
  compression: "none",
  fields: [
    { name: "modelInput", dtype: "float32", shape: [MODEL_INPUT_FEATURE_COUNT] },
    { name: "legalPlayMask", dtype: "uint8", shape: [CARD_COUNT] },
    { name: "selectedCardIndex", dtype: "uint8", shape: [] },
    { name: "behaviorLogProbability", dtype: "float32", shape: [] },
    { name: "terminalReward", dtype: "int8", shape: [] },
    { name: "seed", dtype: "uint32", shape: [] },
    { name: "step", dtype: "uint16", shape: [] },
    { name: "actingPlayerIndex", dtype: "uint8", shape: [] },
    { name: "selfRoleIndex", dtype: "uint8", shape: [] }
  ]
};

function createPlayingSelfPlayShardWriter(
  directory: string,
  shardIndex: number,
  startSeed: number,
  format: typeof PLAYING_SELF_PLAY_BINARY_DATASET_FORMAT | typeof DATASET_FORMAT,
  binaryCompression: PlayingSelfPlayBinaryCompression
): PlayingSelfPlayShardWriter {
  if (format === DATASET_FORMAT) {
    return createJsonlShardWriter(
      directory,
      shardIndex,
      startSeed,
      serializePlayingSelfPlaySample
    ) as PlayingSelfPlayShardWriter;
  }

  return createBinaryPlayingSelfPlayShardWriter(
    directory,
    shardIndex,
    startSeed,
    binaryCompression
  );
}

function createBinaryPlayingSelfPlayShardWriter(
  directory: string,
  shardIndex: number,
  startSeed: number,
  compression: PlayingSelfPlayBinaryCompression
): PlayingSelfPlayShardWriter {
  const fileName = binaryShardFileName(shardIndex);
  const samples: PlayingSelfPlayTensorSample[] = [];
  let closed = false;

  return {
    get sampleCount() {
      return samples.length;
    },
    writeSample: async (sample) => {
      if (closed) {
        throw new Error(`Cannot write to closed shard ${fileName}.`);
      }
      if (!isPlayingSelfPlayTensorSample(sample)) {
        throw new Error("Binary self-play shard writer requires tensor-ready samples.");
      }
      samples.push(sample);
    },
    close: async (endSeed, gameCount) => {
      if (closed) {
        throw new Error(`Cannot close shard ${fileName} more than once.`);
      }
      closed = true;
      const bytes = serializeBinaryPlayingSelfPlayShard(samples, compression);
      await writeFile(join(directory, fileName), bytes);
      return {
        file: fileName,
        startSeed,
        endSeed,
        gameCount,
        sampleCount: samples.length,
        byteLength: bytes.byteLength,
        sha256: createHash("sha256").update(bytes).digest("hex")
      };
    },
    abort: async () => {
      closed = true;
      samples.length = 0;
    }
  };
}

function serializeBinaryPlayingSelfPlayShard(
  samples: readonly PlayingSelfPlayTensorSample[],
  compression: PlayingSelfPlayBinaryCompression
): Buffer {
  const sampleCount = samples.length;
  const modelInput = new Float32Array(sampleCount * MODEL_INPUT_FEATURE_COUNT);
  const legalPlayMask = new Uint8Array(sampleCount * CARD_COUNT);
  const selectedCardIndex = new Uint8Array(sampleCount);
  const behaviorLogProbability = new Float32Array(sampleCount);
  const terminalReward = new Int8Array(sampleCount);
  const seed = new Uint32Array(sampleCount);
  const step = new Uint16Array(sampleCount);
  const actingPlayerIndex = new Uint8Array(sampleCount);
  const selfRoleIndex = new Uint8Array(sampleCount);

  samples.forEach((sample, index) => {
    modelInput.set(sample.modelInput, index * MODEL_INPUT_FEATURE_COUNT);
    legalPlayMask.set(sample.legalPlayMask, index * CARD_COUNT);
    selectedCardIndex[index] = sample.selectedCardIndex;
    behaviorLogProbability[index] = sample.behaviorLogProbability;
    terminalReward[index] = sample.terminalReward;
    seed[index] = sample.seed;
    step[index] = sample.step;
    actingPlayerIndex[index] = sample.actingPlayerIndex;
    selfRoleIndex[index] = sample.selfRoleIndex;
  });

  const buffers = [
    bufferFromTypedArray(modelInput),
    bufferFromTypedArray(legalPlayMask),
    bufferFromTypedArray(selectedCardIndex),
    bufferFromTypedArray(behaviorLogProbability),
    bufferFromTypedArray(terminalReward),
    bufferFromTypedArray(seed),
    bufferFromTypedArray(step),
    bufferFromTypedArray(actingPlayerIndex),
    bufferFromTypedArray(selfRoleIndex)
  ];
  let byteOffset = 0;
  const fields = PLAYING_SELF_PLAY_BINARY_TENSOR_SCHEMA.fields.map((field, index) => {
    const byteLength = buffers[index].byteLength;
    const record = { ...field, byteOffset, byteLength };
    byteOffset += byteLength;
    return record;
  });
  const rawPayload = Buffer.concat(buffers);
  const payload = compression === "gzip" ? gzipSync(rawPayload) : rawPayload;
  const header = Buffer.from(JSON.stringify({
    shardSchemaVersion: PLAYING_SELF_PLAY_BINARY_SHARD_SCHEMA_VERSION,
    sampleType: PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE,
    sampleSchemaVersion: PLAYING_SELF_PLAY_SAMPLE_SCHEMA_VERSION,
    sampleCount,
    modelInputFeatureCount: MODEL_INPUT_FEATURE_COUNT,
    cardCount: CARD_COUNT,
    byteOrder: "little-endian",
    compression,
    uncompressedByteLength: byteOffset,
    fields
  }), "utf8");
  const headerLength = Buffer.alloc(4);
  headerLength.writeUInt32LE(header.byteLength, 0);

  return Buffer.concat([BINARY_SHARD_MAGIC, headerLength, header, payload]);
}

function bufferFromTypedArray(array: ArrayBufferView): Buffer {
  return Buffer.from(array.buffer, array.byteOffset, array.byteLength);
}

export async function generatePlayingSelfPlayDataset(
  options: GeneratePlayingSelfPlayDatasetOptions
): Promise<GeneratePlayingSelfPlayDatasetResult> {
  const temperature = options.temperature ?? DEFAULT_PLAYING_SELF_PLAY_TEMPERATURE;
  validatePlayingSelfPlayGenerationOptions({ ...options, temperature });

  const outputDirectory = resolve(options.outputDirectory);
  const format = options.format ?? PLAYING_SELF_PLAY_BINARY_DATASET_FORMAT;
  const binaryCompression = options.binaryCompression ?? "none";
  await ensureOutputDoesNotExist(outputDirectory);
  await mkdir(dirname(outputDirectory), { recursive: true });

  const artifact = {
    artifactId: options.playingPolicyArtifact.artifactId ?? basename(options.playingPolicyArtifact.metadataPath),
    onnxFileName: basename(options.playingPolicyArtifact.onnxPath),
    metadataFileName: basename(options.playingPolicyArtifact.metadataPath),
    onnxSha256: await sha256File(options.playingPolicyArtifact.onnxPath),
    metadataSha256: await sha256File(options.playingPolicyArtifact.metadataPath),
    ...runtimeInfoForPolicy(options.playingPolicy),
    metadata: options.playingPolicy.metadata
  };
  const rolloutRoster = await createRolloutRosterManifest(options.rolloutRoster);
  const tempDirectory = await mkdtemp(
    join(dirname(outputDirectory), `.${basenameForTemp(outputDirectory)}.tmp-`)
  );
  let activeShard: PlayingSelfPlayShardWriter | null = null;
  const gameRunner = options.gameRunner ?? defaultPlayingSelfPlayGameRunner;
  const rolloutWorkers = options.rolloutWorkers ?? 1;

  try {
    const shards: DatasetShardManifest[] = [];
    let totalSampleCount = 0;
    let shardGameCount = 0;
    let nextGameOffsetToStart = 0;
    let nextGameOffsetToCommit = 0;
    let pendingError: unknown = null;
    const inFlight = new Set<Promise<void>>();
    const completedResults = new Map<number, AutomatedGameRecord | PlayingSelfPlayGameRunResult>();

    const startGame = (gameOffset: number): void => {
      const seed = options.startSeed + gameOffset;
      const task = gameRunner.runGame({
        gameOffset,
        seed,
        currentPolicy: options.playingPolicy,
        behaviorPolicyArtifactId: artifact.artifactId,
        rolloutRoster: options.rolloutRoster,
        temperature,
        maxDecisionSteps: options.maxDecisionSteps
      }).then(
        (result) => {
          const resultSeed = getGameRunSeed(result);
          if (resultSeed !== seed) {
            throw new Error(`Worker returned seed ${resultSeed} for expected seed ${seed}.`);
          }
          if (completedResults.has(gameOffset)) {
            throw new Error(`Duplicate completed game offset: ${gameOffset}`);
          }
          completedResults.set(gameOffset, result);
        },
        (error: unknown) => {
          pendingError = error;
        }
      ).catch((error: unknown) => {
        pendingError = error;
      }).finally(() => {
        inFlight.delete(task);
      });
      inFlight.add(task);
    };

    while (nextGameOffsetToCommit < options.gameCount) {
      while (
        pendingError === null &&
        nextGameOffsetToStart < options.gameCount &&
        inFlight.size < rolloutWorkers
      ) {
        startGame(nextGameOffsetToStart);
        nextGameOffsetToStart += 1;
      }

      if (pendingError !== null) {
        throw pendingError;
      }

      const gameOffset = nextGameOffsetToCommit;
      const seed = options.startSeed + gameOffset;
      const result = completedResults.get(gameOffset);

      if (result === undefined) {
        if (inFlight.size === 0) {
          throw new Error(`Missing completed game offset: ${gameOffset}`);
        }
        await Promise.race(inFlight);
        continue;
      }

      completedResults.delete(gameOffset);

      if (activeShard === null) {
        activeShard = createPlayingSelfPlayShardWriter(
          tempDirectory,
          shards.length,
          seed,
          format,
          binaryCompression
        );
        shardGameCount = 0;
      }

      const samples = await getSamplesForGameRunResult(result, options.playingPolicy, temperature, {
        behaviorPolicyArtifactId: artifact.artifactId,
        rolloutSeatSources: assignRolloutRosterForSeed(options.rolloutRoster, seed).map((seat) => seat.source)
      }, format);

      for (const sample of samples) {
        if (isPlayingSelfPlayTensorSample(sample)) {
          validatePlayingSelfPlayTensorSample(sample, seed);
        } else {
          validatePlayingSelfPlaySample(sample, seed);
        }
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

      nextGameOffsetToCommit += 1;
    }

    const manifest = createPlayingSelfPlayManifest({
      options,
      temperature,
      sampleCount: totalSampleCount,
      shards,
      artifact,
      rolloutRoster,
      format
    });

    validatePlayingSelfPlayDatasetManifest(manifest);
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
  } finally {
    await gameRunner.close?.().catch(() => undefined);
  }
}

export async function createPlayingSelfPlaySamples(
  record: AutomatedGameRecord,
  policy: PlayingSelfPlayPolicy,
  temperature: number = DEFAULT_PLAYING_SELF_PLAY_TEMPERATURE,
  metadata?: {
    behaviorPolicyArtifactId?: string;
    rolloutSeatSources?: readonly PlayingSelfPlayRosterSource[];
  }
): Promise<readonly PlayingSelfPlaySample[]> {
  validateTemperature(temperature);

  const samples: PlayingSelfPlaySample[] = [];
  const seatSources = metadata?.rolloutSeatSources ?? defaultRolloutSeatSources();

  for (const decision of record.decisions) {
    if (decision.phase !== "playing") {
      continue;
    }

    const seatIndex = record.playerIds.indexOf(decision.playerId);
    const source = seatSources[seatIndex];

    if (source !== CURRENT_POLICY_ROSTER_SOURCE) {
      continue;
    }

    samples.push(await createPlayingSelfPlaySample(
      record,
      decision,
      policy,
      temperature,
      {
        behaviorPolicyArtifactId: metadata?.behaviorPolicyArtifactId ?? "current-policy",
        rolloutSeatSources: seatSources
      }
    ));
  }

  return samples;
}

export function serializePlayingSelfPlaySample(sample: PlayingSelfPlaySample): string {
  return `${JSON.stringify(sample)}\n`;
}

function getGameRunSeed(result: AutomatedGameRecord | PlayingSelfPlayGameRunResult): number {
  return isPlayingSelfPlayGameRunResult(result) ? result.seed : result.seed;
}

async function getSamplesForGameRunResult(
  result: AutomatedGameRecord | PlayingSelfPlayGameRunResult,
  policy: PlayingSelfPlayPolicy,
  temperature: number,
  metadata: {
    behaviorPolicyArtifactId: string;
    rolloutSeatSources: readonly PlayingSelfPlayRosterSource[];
  },
  format: typeof PLAYING_SELF_PLAY_BINARY_DATASET_FORMAT | typeof DATASET_FORMAT
): Promise<readonly PlayingSelfPlayWritableSample[]> {
  if (isPlayingSelfPlayGameRunResult(result)) {
    if (format === PLAYING_SELF_PLAY_BINARY_DATASET_FORMAT && result.tensorSamples !== undefined) {
      return result.tensorSamples.map(normalizePlayingSelfPlayTensorSample);
    }
    if (result.samples !== undefined) {
      if (format === PLAYING_SELF_PLAY_BINARY_DATASET_FORMAT) {
        return result.samples.map(tensorSampleFromLegacySample);
      }
      return result.samples;
    }
    if (result.record !== undefined) {
      const samples = await createPlayingSelfPlaySamples(result.record, policy, temperature, metadata);
      return format === PLAYING_SELF_PLAY_BINARY_DATASET_FORMAT
        ? samples.map(tensorSampleFromLegacySample)
        : samples;
    }
    throw new Error("Game runner result must include samples or record.");
  }

  const samples = await createPlayingSelfPlaySamples(result, policy, temperature, metadata);
  return format === PLAYING_SELF_PLAY_BINARY_DATASET_FORMAT
    ? samples.map(tensorSampleFromLegacySample)
    : samples;
}

function isPlayingSelfPlayGameRunResult(
  result: AutomatedGameRecord | PlayingSelfPlayGameRunResult
): result is PlayingSelfPlayGameRunResult {
  return "samples" in result || "tensorSamples" in result || "record" in result;
}

export function calculatePlayingSelfPlayLogProbability(options: {
  logits: Float32Array | readonly number[];
  legalPlayMask: ArrayLike<number | boolean>;
  selectedCardIndex: number;
  temperature?: number;
}): number {
  const distribution = createMaskedCategoricalDistribution(
    options.logits,
    options.legalPlayMask,
    options.temperature ?? DEFAULT_PLAYING_SELF_PLAY_TEMPERATURE
  );
  const index = distribution.legalCardIndices.indexOf(options.selectedCardIndex);

  if (index === -1) {
    throw new Error(`selectedCardIndex ${options.selectedCardIndex} is not legal under legalPlayMask.`);
  }

  return distribution.logProbabilities[index];
}

export function validatePlayingSelfPlaySample(
  sample: PlayingSelfPlaySample,
  expectedSeed: number
): void {
  validateJsonSafeValue("sample", sample);

  if (sample.sampleType !== PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE) {
    throw new Error(`Sample sampleType must be ${PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE}.`);
  }
  if (sample.schemaVersion !== PLAYING_SELF_PLAY_LEGACY_SAMPLE_SCHEMA_VERSION) {
    throw new Error(`Unexpected self-play sample schemaVersion: ${sample.schemaVersion}`);
  }
  if (sample.seed !== expectedSeed) {
    throw new Error(`Sample seed must match current seed: ${sample.seed} !== ${expectedSeed}`);
  }
  if (sample.actingPlayerId !== sample.relativePlayerIds[0]) {
    throw new Error("actingPlayerId must match relativePlayerIds[0].");
  }
  if (sample.actingSeatSource !== CURRENT_POLICY_ROSTER_SOURCE) {
    throw new Error("actingSeatSource must be current-policy.");
  }
  if (sample.behaviorPolicyArtifactId.length === 0) {
    throw new Error("behaviorPolicyArtifactId must be non-empty.");
  }
  if (sample.rolloutSeatSources.length !== PLAYER_COUNT) {
    throw new Error(`rolloutSeatSources must contain ${PLAYER_COUNT} seats.`);
  }
  for (const source of sample.rolloutSeatSources) {
    if (!isRolloutRosterSource(source)) {
      throw new Error(`Invalid rolloutSeatSources entry: ${String(source)}`);
    }
  }
  if (!sameStringArray(sample.relativePlayerIds, sample.observation.relativePlayerIds)) {
    throw new Error("sample.relativePlayerIds must match observation.relativePlayerIds.");
  }

  validateEncodedPlayingObservation(sample.observation);
  validateCardIndex("selectedCardIndex", sample.selectedCardIndex);

  if (sample.observation.legalPlayMask[sample.selectedCardIndex] !== 1) {
    throw new Error("selectedCardIndex must be legal in observation.legalPlayMask.");
  }
  if (!Number.isFinite(sample.behaviorLogProbability) || sample.behaviorLogProbability > 1e-12) {
    throw new Error("behaviorLogProbability must be finite and <= 0.");
  }
  if (sample.terminalReward !== 1 && sample.terminalReward !== -1) {
    throw new Error("terminalReward must be +1 or -1.");
  }
  if (!SELF_ROLE_ORDER.includes(sample.outcome.actingPlayerRole)) {
    throw new Error("outcome.actingPlayerRole is invalid.");
  }
  const roleIndex = sample.observation.selfRoleOneHot.indexOf(1);
  if (sample.outcome.actingPlayerRole !== SELF_ROLE_ORDER[roleIndex]) {
    throw new Error("outcome.actingPlayerRole must match observation.selfRoleOneHot.");
  }
  if (sample.outcome.actingPlayerTeam === sample.outcome.winner && sample.terminalReward !== 1) {
    throw new Error("Winning team samples must have terminalReward +1.");
  }
  if (sample.outcome.actingPlayerTeam !== sample.outcome.winner && sample.terminalReward !== -1) {
    throw new Error("Losing team samples must have terminalReward -1.");
  }
}

export function validatePlayingSelfPlayTensorSample(
  sample: PlayingSelfPlayTensorSample,
  expectedSeed: number
): void {
  if (sample.sampleType !== PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE) {
    throw new Error(`Tensor sample sampleType must be ${PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE}.`);
  }
  if (sample.schemaVersion !== PLAYING_SELF_PLAY_SAMPLE_SCHEMA_VERSION) {
    throw new Error(`Unexpected tensor self-play sample schemaVersion: ${sample.schemaVersion}`);
  }
  if (sample.seed !== expectedSeed) {
    throw new Error(`Tensor sample seed must match current seed: ${sample.seed} !== ${expectedSeed}`);
  }
  validateUint32("Tensor sample seed", sample.seed);
  validateUint16("Tensor sample step", sample.step);
  validateCardIndex("selectedCardIndex", sample.selectedCardIndex);
  validatePlayerIndex("actingPlayerIndex", sample.actingPlayerIndex);
  validateSelfRoleIndex("selfRoleIndex", sample.selfRoleIndex);

  if (!isFloat32ArrayLike(sample.modelInput) || sample.modelInput.length !== MODEL_INPUT_FEATURE_COUNT) {
    throw new Error(`modelInput must be Float32Array-like(${MODEL_INPUT_FEATURE_COUNT}).`);
  }
  for (const [index, value] of sample.modelInput.entries()) {
    if (!Number.isFinite(value)) {
      throw new Error(`modelInput[${index}] must be finite.`);
    }
  }
  if (!(sample.legalPlayMask instanceof Uint8Array) || sample.legalPlayMask.length !== CARD_COUNT) {
    throw new Error(`legalPlayMask must be Uint8Array(${CARD_COUNT}).`);
  }
  let legalCount = 0;
  for (const value of sample.legalPlayMask) {
    if (value !== 0 && value !== 1) {
      throw new Error("legalPlayMask values must be 0 or 1.");
    }
    legalCount += value;
  }
  if (legalCount === 0) {
    throw new Error("legalPlayMask must contain at least one legal card.");
  }
  if (sample.legalPlayMask[sample.selectedCardIndex] !== 1) {
    throw new Error("selectedCardIndex must be legal in legalPlayMask.");
  }
  if (!Number.isFinite(sample.behaviorLogProbability) || sample.behaviorLogProbability > 1e-12) {
    throw new Error("behaviorLogProbability must be finite and <= 0.");
  }
  if (sample.terminalReward !== 1 && sample.terminalReward !== -1) {
    throw new Error("terminalReward must be +1 or -1.");
  }
}

export function validatePlayingSelfPlayDatasetManifest(
  manifest: PlayingSelfPlayDatasetManifest
): void {
  if (
    manifest.datasetSchemaVersion !== PLAYING_SELF_PLAY_DATASET_SCHEMA_VERSION &&
    manifest.datasetSchemaVersion !== PLAYING_SELF_PLAY_LEGACY_DATASET_SCHEMA_VERSION
  ) {
    throw new Error("Self-play manifest datasetSchemaVersion mismatch.");
  }
  if (manifest.generatorVersion !== PLAYING_SELF_PLAY_DATASET_GENERATOR_VERSION) {
    throw new Error("Self-play manifest generatorVersion mismatch.");
  }
  if (
    manifest.sampleType !== PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE ||
    !isExpectedSelfPlayFormatAndSchema(manifest)
  ) {
    throw new Error("Self-play manifest format or sample type mismatch.");
  }
  if (manifest.format === PLAYING_SELF_PLAY_BINARY_DATASET_FORMAT) {
    validatePlayingSelfPlayBinaryTensorSchema(manifest.tensorSchema);
  } else if (manifest.tensorSchema !== undefined) {
    throw new Error("Legacy JSONL self-play manifest must not include tensorSchema.");
  }
  validateUint32("Manifest startSeed", manifest.startSeed);
  validateUint32("Manifest endSeed", manifest.endSeed);
  validatePositiveInteger("Manifest gameCount", manifest.gameCount);
  validatePositiveInteger("Manifest sampleCount", manifest.sampleCount);
  validatePositiveInteger("Manifest gamesPerShard", manifest.gamesPerShard);
  validatePositiveInteger("Manifest shardCount", manifest.shardCount);

  if (manifest.endSeed !== manifest.startSeed + manifest.gameCount - 1) {
    throw new Error("Self-play manifest seed range mismatch.");
  }
  if (manifest.shardCount !== Math.ceil(manifest.gameCount / manifest.gamesPerShard)) {
    throw new Error("Self-play manifest shardCount mismatch.");
  }
  if (manifest.shardCount !== manifest.shards.length) {
    throw new Error("Self-play manifest shardCount must match shards length.");
  }
  if (manifest.sampleCount !== sum(manifest.shards.map((shard) => shard.sampleCount))) {
    throw new Error("Self-play manifest sampleCount must equal shard sample counts.");
  }
  if (manifest.playerCount !== PLAYER_COUNT || manifest.cardCount !== CARD_COUNT) {
    throw new Error("Self-play manifest fixed dimensions mismatch.");
  }
  if (!sameStringArray(manifest.cardIds, CARD_IDS)) {
    throw new Error("Self-play manifest cardIds mismatch.");
  }
  if (manifest.cardIdsSha256 !== calculateCardIdsSha256()) {
    throw new Error("Self-play manifest cardIdsSha256 mismatch.");
  }
  if (
    manifest.playingEncoderSchemaVersion !== PLAYING_ENCODER_SCHEMA_VERSION ||
    manifest.playingModelInputSchemaVersion !== MODEL_INPUT_SCHEMA_VERSION
  ) {
    throw new Error("Self-play manifest playing schema metadata mismatch.");
  }
  if (
    manifest.behaviorPolicy.type !== "playing-onnx" ||
    !sha256Pattern.test(manifest.behaviorPolicy.onnxSha256) ||
    !sha256Pattern.test(manifest.behaviorPolicy.metadataSha256) ||
    !isRequestedInferenceDevice(manifest.behaviorPolicy.requestedInferenceDevice) ||
    !isResolvedInferenceDevice(manifest.behaviorPolicy.resolvedInferenceDevice) ||
    !isResolvedInferenceDevice(manifest.behaviorPolicy.executionProvider)
  ) {
    throw new Error("Self-play manifest behavior policy metadata mismatch.");
  }
  if (manifest.samplingAlgorithm !== PLAYING_SELF_PLAY_SAMPLING_ALGORITHM) {
    throw new Error("Self-play manifest samplingAlgorithm mismatch.");
  }
  validateTemperature(manifest.temperature);
  if (
    manifest.reward.type !== PLAYING_SELF_PLAY_REWARD_TYPE ||
    manifest.reward.version !== PLAYING_SELF_PLAY_REWARD_VERSION
  ) {
    throw new Error("Self-play manifest reward metadata mismatch.");
  }
  if (
    manifest.nonPlayingAgent.type !== "rule-based" ||
    manifest.nonPlayingAgent.version !== RULE_BASED_AGENT_VERSION
  ) {
    throw new Error("Self-play manifest non-playing agent metadata mismatch.");
  }
  validateRolloutRosterManifest(manifest.rolloutRoster);

  validateShards(manifest);
}

function isExpectedSelfPlayFormatAndSchema(manifest: PlayingSelfPlayDatasetManifest): boolean {
  return (
    (
      manifest.datasetSchemaVersion === PLAYING_SELF_PLAY_DATASET_SCHEMA_VERSION &&
      manifest.format === PLAYING_SELF_PLAY_BINARY_DATASET_FORMAT &&
      manifest.sampleSchemaVersion === PLAYING_SELF_PLAY_SAMPLE_SCHEMA_VERSION
    ) ||
    (
      manifest.datasetSchemaVersion === PLAYING_SELF_PLAY_LEGACY_DATASET_SCHEMA_VERSION &&
      manifest.format === DATASET_FORMAT &&
      manifest.sampleSchemaVersion === PLAYING_SELF_PLAY_LEGACY_SAMPLE_SCHEMA_VERSION
    )
  );
}

function validatePlayingSelfPlayBinaryTensorSchema(
  schema: PlayingSelfPlayBinaryTensorSchema | undefined
): void {
  if (schema === undefined) {
    throw new Error("Binary self-play manifest tensorSchema is required.");
  }
  if (
    schema.shardSchemaVersion !== PLAYING_SELF_PLAY_BINARY_SHARD_SCHEMA_VERSION ||
    schema.byteOrder !== "little-endian" ||
    !isSupportedPlayingSelfPlayBinaryCompression(schema.compression)
  ) {
    throw new Error("Binary self-play manifest tensorSchema metadata mismatch.");
  }
  if (schema.fields.length !== PLAYING_SELF_PLAY_BINARY_TENSOR_SCHEMA.fields.length) {
    throw new Error("Binary self-play manifest tensorSchema field count mismatch.");
  }
  schema.fields.forEach((field, index) => {
    const expected = PLAYING_SELF_PLAY_BINARY_TENSOR_SCHEMA.fields[index];
    if (
      field.name !== expected.name ||
      field.dtype !== expected.dtype ||
      !sameNumberArray(field.shape, expected.shape)
    ) {
      throw new Error(`Binary self-play manifest tensorSchema field ${index} mismatch.`);
    }
  });
}

function isSupportedPlayingSelfPlayBinaryCompression(
  value: string
): value is PlayingSelfPlayBinaryCompression {
  return value === "none" || value === "gzip";
}

function validateRolloutRosterManifest(roster: PlayingSelfPlayRolloutRosterManifest): void {
  if (roster.assignment !== PLAYING_SELF_PLAY_ROSTER_ASSIGNMENT) {
    throw new Error("Self-play manifest rolloutRoster.assignment mismatch.");
  }
  if (!Array.isArray(roster.seats) || roster.seats.length !== PLAYER_COUNT) {
    throw new Error(`Self-play manifest rolloutRoster.seats must contain ${PLAYER_COUNT} seats.`);
  }

  let currentPolicyCount = 0;
  roster.seats.forEach((seat, index) => {
    if (!isRolloutRosterSource(seat.source)) {
      throw new Error(`Self-play manifest rolloutRoster.seats[${index}].source is invalid.`);
    }
    if (seat.source === CURRENT_POLICY_ROSTER_SOURCE) {
      currentPolicyCount += 1;
    }
    if (
      seat.source === RULE_BASED_ROSTER_SOURCE &&
      seat.version !== RULE_BASED_AGENT_VERSION
    ) {
      throw new Error(`Self-play manifest rolloutRoster.seats[${index}] rule-based version mismatch.`);
    }
    if (seat.source === FROZEN_ONNX_ROSTER_SOURCE) {
      if (
        seat.artifactId.length === 0 ||
        seat.onnxFileName.length === 0 ||
        seat.metadataFileName.length === 0 ||
        !sha256Pattern.test(seat.onnxSha256) ||
        !sha256Pattern.test(seat.metadataSha256) ||
        !isRequestedInferenceDevice(seat.requestedInferenceDevice) ||
        !isResolvedInferenceDevice(seat.resolvedInferenceDevice) ||
        !isResolvedInferenceDevice(seat.executionProvider)
      ) {
        throw new Error(`Self-play manifest rolloutRoster.seats[${index}] frozen-onnx metadata mismatch.`);
      }
    }
  });

  if (currentPolicyCount === 0) {
    throw new Error("Self-play manifest rolloutRoster must include current-policy.");
  }
}

export function assignRolloutRosterForSeed(
  roster: PlayingSelfPlayRolloutRosterOptions | undefined,
  seed: number
): readonly RolloutRosterSeatOptions[] {
  validateUint32("seed", seed);
  const seats = normalizeRolloutRosterOptions(roster).seats;
  const rotation = seed % PLAYER_COUNT;

  return seats.map((_, seatIndex) =>
    seats[(seatIndex - rotation + PLAYER_COUNT) % PLAYER_COUNT]
  );
}

export async function runPlayingSelfPlayGame(options: {
  seed: number;
  currentPolicy: PlayingSelfPlayPolicy;
  rolloutRoster: PlayingSelfPlayRolloutRosterOptions | undefined;
  temperature: number;
  maxDecisionSteps: number | undefined;
}): Promise<AutomatedGameRecord> {
  const assignedRoster = assignRolloutRosterForSeed(options.rolloutRoster, options.seed);

  return runAutomatedGame({
    seed: options.seed,
    maxDecisionSteps: options.maxDecisionSteps,
    createAgent: ({ rng, playerIndex }) => {
      const seat = assignedRoster[playerIndex];

      switch (seat.source) {
        case CURRENT_POLICY_ROSTER_SOURCE:
          return new PlayingSelfPlayAgent({
            policy: options.currentPolicy,
            rng,
            temperature: options.temperature
          });
        case RULE_BASED_ROSTER_SOURCE:
          return new RuleBasedAgent(rng);
        case FROZEN_ONNX_ROSTER_SOURCE:
          return new PlayingSelfPlayAgent({
            policy: seat.policy,
            rng,
            temperature: options.temperature
          });
      }
    }
  });
}

export async function runPlayingSelfPlayGameWithSamples(options: {
  seed: number;
  currentPolicy: PlayingSelfPlayPolicy;
  behaviorPolicyArtifactId: string;
  rolloutRoster: PlayingSelfPlayRolloutRosterOptions | undefined;
  temperature: number;
  maxDecisionSteps: number | undefined;
}): Promise<PlayingSelfPlayGameRunResult> {
  const assignedRoster = assignRolloutRosterForSeed(options.rolloutRoster, options.seed);
  const sampledDecisions: PlayingSelfPlaySampleDraft[] = [];
  const record = await runAutomatedGame({
    seed: options.seed,
    maxDecisionSteps: options.maxDecisionSteps,
    createAgent: ({ rng, playerIndex }) => {
      const seat = assignedRoster[playerIndex];

      switch (seat.source) {
        case CURRENT_POLICY_ROSTER_SOURCE:
          return new PlayingSelfPlayAgent({
            policy: options.currentPolicy,
            rng,
            temperature: options.temperature,
            recordSample: (sample) => {
              sampledDecisions.push(sample);
            }
          });
        case RULE_BASED_ROSTER_SOURCE:
          return new RuleBasedAgent(rng);
        case FROZEN_ONNX_ROSTER_SOURCE:
          return new PlayingSelfPlayAgent({
            policy: seat.policy,
            rng,
            temperature: options.temperature
          });
      }
    }
  });
  const samples = completePlayingSelfPlaySamples(record, sampledDecisions, {
    behaviorPolicyArtifactId: options.behaviorPolicyArtifactId,
    rolloutSeatSources: assignedRoster.map((seat) => seat.source)
  });
  const tensorSamples = completePlayingSelfPlayTensorSamples(record, sampledDecisions, {
    rolloutSeatSources: assignedRoster.map((seat) => seat.source)
  });

  return { seed: record.seed, samples, tensorSamples };
}

const defaultPlayingSelfPlayGameRunner: PlayingSelfPlayGameRunner = {
  runGame: async (request) => runPlayingSelfPlayGameWithSamples(request)
};

interface PlayingSelfPlaySampleDraft {
  playerId: PlayerId;
  relativePlayerIds: readonly PlayerId[];
  observation: ReturnType<typeof encodePlayingObservation>;
  modelInput: Float32Array;
  legalPlayMask: Uint8Array;
  selectedCardIndex: number;
  behaviorLogProbability: number;
}

class PlayingSelfPlayAgent implements Agent {
  private readonly ruleBasedAgent: RuleBasedAgent;

  constructor(
    private readonly options: {
      policy: PlayingSelfPlayPolicy;
      rng: () => number;
      temperature: number;
      recordSample?: (sample: PlayingSelfPlaySampleDraft) => void;
    }
  ) {
    this.ruleBasedAgent = new RuleBasedAgent(options.rng);
  }

  async selectAction(observation: PlayerObservation): Promise<GameAction> {
    if (observation.view.phase !== "playing") {
      return this.ruleBasedAgent.selectAction(observation);
    }

    const { modelInput, legalPlayMask, encodedObservation } = createPlayingSelfPlayPolicyInput(observation);
    const logits = await this.options.policy.predictLogits(modelInput);
    const selection = samplePlayingSelfPlayAction({
      logits,
      legalPlayMask,
      rng: this.options.rng,
      temperature: this.options.temperature
    });
    const selectedCardId = getCardId(selection.selectedCardIndex);
    const selectedAction = observation.legalActions.find(
      (action) => action.type === "play-card" && action.cardId === selectedCardId
    );

    if (selectedAction === undefined) {
      throw new Error(
        `Self-play ONNX policy selected card index ${selection.selectedCardIndex} (${selectedCardId}) outside legal actions.`
      );
    }

    this.options.recordSample?.({
      playerId: observation.playerId,
      relativePlayerIds: encodedObservation.relativePlayerIds,
      observation: encodedObservation,
      modelInput: Float32Array.from(modelInput),
      legalPlayMask: Uint8Array.from(legalPlayMask),
      selectedCardIndex: selection.selectedCardIndex,
      behaviorLogProbability: selection.logProbability
    });

    return selectedAction;
  }
}

function createPlayingSelfPlayPolicyInput(observation: PlayerObservation): {
  modelInput: Float32Array;
  legalPlayMask: readonly number[];
  encodedObservation: ReturnType<typeof encodePlayingObservation>;
} {
  if (observation.view.phase !== "playing") {
    throw new Error(
      `createPlayingSelfPlayPolicyInput requires a playing observation, got ${observation.view.phase}.`
    );
  }

  if (observation.publicActionHistory === undefined) {
    throw new Error("Playing self-play policy input requires publicActionHistory.");
  }

  const absolutePlayerIds = observation.view.players.map((player) => player.id);
  const relativePlayerIds = createRelativePlayerOrder(absolutePlayerIds, observation.playerId);
  const biddingHistory = encodeBiddingHistoryFromPublicActions(
    observation.publicActionHistory,
    relativePlayerIds
  );
  const encodedObservation = encodePlayingObservation(
    observation,
    absolutePlayerIds,
    biddingHistory
  );
  const { modelInput, legalPlayMask } = createPlayingModelInput(encodedObservation);

  return { modelInput, legalPlayMask, encodedObservation };
}

function samplePlayingSelfPlayAction(options: {
  logits: Float32Array | readonly number[];
  legalPlayMask: ArrayLike<number | boolean>;
  rng: () => number;
  temperature?: number;
}): PlayingSelfPlaySampledAction {
  const distribution = createMaskedCategoricalDistribution(
    options.logits,
    options.legalPlayMask,
    options.temperature ?? DEFAULT_PLAYING_SELF_PLAY_TEMPERATURE
  );

  if (distribution.legalCardIndices.length === 1) {
    return {
      selectedCardIndex: distribution.legalCardIndices[0],
      logProbability: 0
    };
  }

  const randomValue = options.rng();
  if (!Number.isFinite(randomValue) || randomValue < 0 || randomValue >= 1) {
    throw new Error("rng must return a finite value in [0, 1).");
  }

  let cumulativeProbability = 0;

  for (let index = 0; index < distribution.legalCardIndices.length; index += 1) {
    cumulativeProbability += distribution.probabilities[index];

    if (randomValue < cumulativeProbability) {
      return {
        selectedCardIndex: distribution.legalCardIndices[index],
        logProbability: distribution.logProbabilities[index]
      };
    }
  }

  const lastIndex = distribution.legalCardIndices.length - 1;

  return {
    selectedCardIndex: distribution.legalCardIndices[lastIndex],
    logProbability: distribution.logProbabilities[lastIndex]
  };
}

function createMaskedCategoricalDistribution(
  logits: Float32Array | readonly number[],
  legalPlayMask: ArrayLike<number | boolean>,
  temperature: number
): {
  legalCardIndices: readonly number[];
  probabilities: readonly number[];
  logProbabilities: readonly number[];
} {
  if (logits.length !== CARD_COUNT) {
    throw new Error(`logits must contain ${CARD_COUNT} values, got ${logits.length}.`);
  }

  validateLegalPlayMask(legalPlayMask);
  validateTemperature(temperature);

  const legalCardIndices: number[] = [];
  const scaledLogits: number[] = [];

  for (let index = 0; index < CARD_COUNT; index += 1) {
    const logit = Number(logits[index]);

    if (!Number.isFinite(logit)) {
      throw new Error(`logits[${index}] must be finite.`);
    }

    if (isLegalMaskValue(legalPlayMask[index])) {
      legalCardIndices.push(index);
      scaledLogits.push(logit / temperature);
    }
  }

  if (legalCardIndices.length === 1) {
    return {
      legalCardIndices,
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

  for (let index = 0; index < probabilities.length; index += 1) {
    if (
      !Number.isFinite(probabilities[index]) ||
      probabilities[index] < 0 ||
      !Number.isFinite(logProbabilities[index]) ||
      logProbabilities[index] > 1e-12
    ) {
      throw new Error("masked categorical distribution contains an invalid probability.");
    }
  }

  return {
    legalCardIndices,
    probabilities,
    logProbabilities
  };
}

function completePlayingSelfPlaySamples(
  record: AutomatedGameRecord,
  sampledDecisions: readonly PlayingSelfPlaySampleDraft[],
  metadata: {
    behaviorPolicyArtifactId: string;
    rolloutSeatSources: readonly PlayingSelfPlayRosterSource[];
  }
): readonly PlayingSelfPlaySample[] {
  const samples: PlayingSelfPlaySample[] = [];
  let sampledDecisionIndex = 0;

  for (const decision of record.decisions) {
    if (decision.phase !== "playing") {
      continue;
    }

    const seatIndex = record.playerIds.indexOf(decision.playerId);
    const source = metadata.rolloutSeatSources[seatIndex];

    if (source !== CURRENT_POLICY_ROSTER_SOURCE) {
      continue;
    }
    if (decision.action.type !== "play-card") {
      throw new Error(`Playing self-play decision action must be play-card, got ${decision.action.type}.`);
    }

    const sampledDecision = sampledDecisions[sampledDecisionIndex];
    sampledDecisionIndex += 1;

    if (sampledDecision === undefined) {
      throw new Error("Missing sampled action for current-policy playing decision.");
    }
    if (sampledDecision.playerId !== decision.playerId) {
      throw new Error(
        `Sampled action player mismatch: ${sampledDecision.playerId} !== ${decision.playerId}.`
      );
    }

    const selectedCardIndex = getCardIndex(decision.action.cardId);
    if (sampledDecision.selectedCardIndex !== selectedCardIndex) {
      throw new Error(
        `Sampled action card mismatch: ${sampledDecision.selectedCardIndex} !== ${selectedCardIndex}.`
      );
    }

    const relativePlayerIds = createRelativePlayerOrder(record.playerIds, decision.playerId);
    if (!sameStringArray(sampledDecision.relativePlayerIds, relativePlayerIds)) {
      throw new Error("Sampled action relativePlayerIds mismatch.");
    }
    if (!sameStringArray(sampledDecision.relativePlayerIds, sampledDecision.observation.relativePlayerIds)) {
      throw new Error("Sampled action observation relativePlayerIds mismatch.");
    }
    if (sampledDecision.observation.legalPlayMask[selectedCardIndex] !== 1) {
      throw new Error("Sampled action selectedCardIndex must be legal.");
    }
    if (
      !Number.isFinite(sampledDecision.behaviorLogProbability) ||
      sampledDecision.behaviorLogProbability > 1e-12
    ) {
      throw new Error("Sampled action behaviorLogProbability must be finite and <= 0.");
    }

    const outcome = createOutcome(record, decision.playerId);
    samples.push({
      sampleType: PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE,
      schemaVersion: PLAYING_SELF_PLAY_LEGACY_SAMPLE_SCHEMA_VERSION,
      seed: record.seed,
      step: decision.step,
      actingPlayerId: decision.playerId,
      actingSeatSource: CURRENT_POLICY_ROSTER_SOURCE,
      behaviorPolicyArtifactId: metadata.behaviorPolicyArtifactId,
      rolloutSeatSources: metadata.rolloutSeatSources,
      relativePlayerIds: sampledDecision.relativePlayerIds,
      observation: sampledDecision.observation,
      selectedCardIndex,
      behaviorLogProbability: sampledDecision.behaviorLogProbability,
      terminalReward: outcome.actingPlayerTeam === outcome.winner ? 1 : -1,
      outcome
    });
  }

  if (sampledDecisionIndex !== sampledDecisions.length) {
    throw new Error("Unused sampled actions remain after completing self-play samples.");
  }

  return samples;
}

function completePlayingSelfPlayTensorSamples(
  record: AutomatedGameRecord,
  sampledDecisions: readonly PlayingSelfPlaySampleDraft[],
  metadata: {
    rolloutSeatSources: readonly PlayingSelfPlayRosterSource[];
  }
): readonly PlayingSelfPlayTensorSample[] {
  const samples: PlayingSelfPlayTensorSample[] = [];
  let sampledDecisionIndex = 0;

  for (const decision of record.decisions) {
    if (decision.phase !== "playing") {
      continue;
    }

    const seatIndex = record.playerIds.indexOf(decision.playerId);
    const source = metadata.rolloutSeatSources[seatIndex];

    if (source !== CURRENT_POLICY_ROSTER_SOURCE) {
      continue;
    }

    const sampledDecision = sampledDecisions[sampledDecisionIndex];
    sampledDecisionIndex += 1;

    if (sampledDecision === undefined) {
      throw new Error("Missing sampled action for current-policy playing decision.");
    }
    if (sampledDecision.playerId !== decision.playerId) {
      throw new Error(
        `Sampled action player mismatch: ${sampledDecision.playerId} !== ${decision.playerId}.`
      );
    }
    if (decision.action.type !== "play-card") {
      throw new Error(`Playing self-play decision action must be play-card, got ${decision.action.type}.`);
    }

    const selectedCardIndex = getCardIndex(decision.action.cardId);
    const outcome = createOutcome(record, decision.playerId);
    const tensorSample: PlayingSelfPlayTensorSample = {
      sampleType: PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE,
      schemaVersion: PLAYING_SELF_PLAY_SAMPLE_SCHEMA_VERSION,
      seed: record.seed,
      step: decision.step,
      actingPlayerIndex: record.playerIds.indexOf(decision.playerId),
      selectedCardIndex,
      behaviorLogProbability: sampledDecision.behaviorLogProbability,
      terminalReward: outcome.actingPlayerTeam === outcome.winner ? 1 : -1,
      selfRoleIndex: SELF_ROLE_ORDER.indexOf(outcome.actingPlayerRole),
      modelInput: sampledDecision.modelInput,
      legalPlayMask: sampledDecision.legalPlayMask
    };
    validatePlayingSelfPlayTensorSample(tensorSample, record.seed);
    samples.push(tensorSample);
  }

  if (sampledDecisionIndex !== sampledDecisions.length) {
    throw new Error("Unused sampled actions remain after completing self-play samples.");
  }

  return samples;
}

function tensorSampleFromLegacySample(sample: PlayingSelfPlaySample): PlayingSelfPlayTensorSample {
  const { modelInput, legalPlayMask } = createPlayingModelInput(sample.observation);
  const selfRoleIndex = sample.observation.selfRoleOneHot.indexOf(1);
  const tensorSample: PlayingSelfPlayTensorSample = {
    sampleType: PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE,
    schemaVersion: PLAYING_SELF_PLAY_SAMPLE_SCHEMA_VERSION,
    seed: sample.seed,
    step: sample.step,
    actingPlayerIndex: sample.relativePlayerIds.indexOf(sample.actingPlayerId),
    selectedCardIndex: sample.selectedCardIndex,
    behaviorLogProbability: sample.behaviorLogProbability,
    terminalReward: sample.terminalReward,
    selfRoleIndex,
    modelInput: Float32Array.from(modelInput),
    legalPlayMask: Uint8Array.from(legalPlayMask)
  };
  validatePlayingSelfPlayTensorSample(tensorSample, sample.seed);
  return tensorSample;
}

function normalizePlayingSelfPlayTensorSample(
  sample: PlayingSelfPlayTensorSample
): PlayingSelfPlayTensorSample {
  const normalized = {
    ...sample,
    modelInput: Float32Array.from(sample.modelInput as ArrayLike<number>),
    legalPlayMask: Uint8Array.from(sample.legalPlayMask as ArrayLike<number>)
  };
  validatePlayingSelfPlayTensorSample(normalized, normalized.seed);
  return normalized;
}

async function createPlayingSelfPlaySample(
  record: AutomatedGameRecord,
  decision: DecisionRecord,
  policy: PlayingSelfPlayPolicy,
  temperature: number,
  metadata: {
    behaviorPolicyArtifactId: string;
    rolloutSeatSources: readonly PlayingSelfPlayRosterSource[];
  }
): Promise<PlayingSelfPlaySample> {
  if (decision.action.type !== "play-card") {
    throw new Error(`Playing self-play decision action must be play-card, got ${decision.action.type}.`);
  }

  const relativePlayerIds = createRelativePlayerOrder(record.playerIds, decision.playerId);
  const biddingHistory = encodeBiddingHistory(record, decision, relativePlayerIds);
  const observation = encodePlayingObservation(
    decision.observation,
    record.playerIds,
    biddingHistory
  );
  const selectedCardIndex = getCardIndex(decision.action.cardId);
  const { modelInput, legalPlayMask } = createPlayingModelInput(observation);
  const logits = await policy.predictLogits(modelInput);
  const behaviorLogProbability = calculatePlayingSelfPlayLogProbability({
    logits,
    legalPlayMask,
    selectedCardIndex,
    temperature
  });
  const outcome = createOutcome(record, decision.playerId);

  return {
    sampleType: PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE,
    schemaVersion: PLAYING_SELF_PLAY_LEGACY_SAMPLE_SCHEMA_VERSION,
    seed: record.seed,
    step: decision.step,
    actingPlayerId: decision.playerId,
    actingSeatSource: CURRENT_POLICY_ROSTER_SOURCE,
    behaviorPolicyArtifactId: metadata.behaviorPolicyArtifactId,
    rolloutSeatSources: metadata.rolloutSeatSources,
    relativePlayerIds: observation.relativePlayerIds,
    observation,
    selectedCardIndex,
    behaviorLogProbability,
    terminalReward: outcome.actingPlayerTeam === outcome.winner ? 1 : -1,
    outcome
  };
}

function createOutcome(record: AutomatedGameRecord, actingPlayerId: PlayerId): PlayingSelfPlayOutcome {
  const actingPlayerTeam = getPlayerTeam(actingPlayerId, record.result);

  return {
    winner: record.result.winner,
    napoleonPlayerId: record.result.napoleonPlayerId,
    actingPlayerTeam,
    actingPlayerRole: getPlayerRole(actingPlayerId, record.result)
  };
}

function getPlayerTeam(playerId: PlayerId, result: AutomatedGameRecord["result"]): WinningTeam {
  if (
    playerId === result.napoleonPlayerId ||
    (result.adjutantPlayerId !== null && playerId === result.adjutantPlayerId)
  ) {
    return "napoleon-team";
  }

  return "alliance";
}

function getPlayerRole(playerId: PlayerId, result: AutomatedGameRecord["result"]): PlayingSelfPlayRole {
  if (playerId === result.napoleonPlayerId) {
    return result.adjutantPlayerId === null ? "napoleon-solo" : "napoleon";
  }
  if (result.adjutantPlayerId !== null && playerId === result.adjutantPlayerId) {
    return "adjutant";
  }

  return "alliance";
}

function createPlayingSelfPlayManifest(input: {
  options: GeneratePlayingSelfPlayDatasetOptions;
  temperature: number;
  sampleCount: number;
  shards: readonly DatasetShardManifest[];
  artifact: {
    artifactId: string;
    onnxFileName: string;
    metadataFileName: string;
    onnxSha256: string;
    metadataSha256: string;
    requestedInferenceDevice: PlayingSelfPlayPolicyRuntime["requestedInferenceDevice"];
    resolvedInferenceDevice: PlayingSelfPlayPolicyRuntime["resolvedInferenceDevice"];
    executionProvider: PlayingSelfPlayPolicyRuntime["executionProvider"];
    metadata: unknown;
  };
  rolloutRoster: PlayingSelfPlayRolloutRosterManifest;
  format: typeof PLAYING_SELF_PLAY_BINARY_DATASET_FORMAT | typeof DATASET_FORMAT;
}): PlayingSelfPlayDatasetManifest {
  const manifest: PlayingSelfPlayDatasetManifest = {
    datasetSchemaVersion: input.format === DATASET_FORMAT
      ? PLAYING_SELF_PLAY_LEGACY_DATASET_SCHEMA_VERSION
      : PLAYING_SELF_PLAY_DATASET_SCHEMA_VERSION,
    generatorVersion: PLAYING_SELF_PLAY_DATASET_GENERATOR_VERSION,
    format: input.format,
    sampleType: PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE,
    sampleSchemaVersion: input.format === DATASET_FORMAT
      ? PLAYING_SELF_PLAY_LEGACY_SAMPLE_SCHEMA_VERSION
      : PLAYING_SELF_PLAY_SAMPLE_SCHEMA_VERSION,
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
    shards: input.shards,
    playingEncoderSchemaVersion: PLAYING_ENCODER_SCHEMA_VERSION,
    playingModelInputSchemaVersion: MODEL_INPUT_SCHEMA_VERSION,
    behaviorPolicy: {
      type: "playing-onnx",
      ...input.artifact
    },
    samplingAlgorithm: PLAYING_SELF_PLAY_SAMPLING_ALGORITHM,
    temperature: input.temperature,
    reward: {
      type: PLAYING_SELF_PLAY_REWARD_TYPE,
      version: PLAYING_SELF_PLAY_REWARD_VERSION
    },
    nonPlayingAgent: {
      type: "rule-based",
      version: RULE_BASED_AGENT_VERSION
    },
    rolloutRoster: input.rolloutRoster,
  };

  if (input.format === PLAYING_SELF_PLAY_BINARY_DATASET_FORMAT) {
    manifest.tensorSchema = {
      ...PLAYING_SELF_PLAY_BINARY_TENSOR_SCHEMA,
      compression: input.options.binaryCompression ?? "none"
    };
  }

  return manifest;
}

function validatePlayingSelfPlayGenerationOptions(
  options: GeneratePlayingSelfPlayDatasetOptions & { temperature: number }
): void {
  validateUint32("startSeed", options.startSeed);
  validatePositiveInteger("gameCount", options.gameCount);
  validatePositiveInteger("gamesPerShard", options.gamesPerShard);
  validatePositiveInteger("rolloutWorkers", options.rolloutWorkers ?? 1);
  validateTemperature(options.temperature);
  validateRolloutRosterOptions(options.rolloutRoster);

  if (
    options.format !== undefined &&
    options.format !== DATASET_FORMAT &&
    options.format !== PLAYING_SELF_PLAY_BINARY_DATASET_FORMAT
  ) {
    throw new Error(`format must be ${PLAYING_SELF_PLAY_BINARY_DATASET_FORMAT} or ${DATASET_FORMAT}.`);
  }
  if (
    options.binaryCompression !== undefined &&
    !isSupportedPlayingSelfPlayBinaryCompression(options.binaryCompression)
  ) {
    throw new Error("binaryCompression must be none or gzip.");
  }
  if (options.format === DATASET_FORMAT && options.binaryCompression !== undefined) {
    throw new Error("binaryCompression is only supported for binary self-play format.");
  }

  if (options.outputDirectory.length === 0) {
    throw new Error("outputDirectory must be a non-empty path.");
  }

  const endSeed = options.startSeed + options.gameCount - 1;
  if (!Number.isSafeInteger(endSeed) || endSeed > UINT32_MAX) {
    throw new Error(`Seed range exceeds uint32: ${options.startSeed}..${endSeed}`);
  }

  const expectedShardCount = Math.ceil(options.gameCount / options.gamesPerShard);
  if (expectedShardCount > MAX_SHARD_COUNT) {
    throw new Error(
      `Dataset would require ${expectedShardCount} shards, exceeding the maximum ${MAX_SHARD_COUNT}.`
    );
  }
}

async function createRolloutRosterManifest(
  roster: PlayingSelfPlayRolloutRosterOptions | undefined
): Promise<PlayingSelfPlayRolloutRosterManifest> {
  const normalized = normalizeRolloutRosterOptions(roster);

  return {
    assignment: PLAYING_SELF_PLAY_ROSTER_ASSIGNMENT,
    seats: await Promise.all(normalized.seats.map(async (seat) => {
      switch (seat.source) {
        case CURRENT_POLICY_ROSTER_SOURCE:
          return { source: CURRENT_POLICY_ROSTER_SOURCE };
        case RULE_BASED_ROSTER_SOURCE:
          return { source: RULE_BASED_ROSTER_SOURCE, version: RULE_BASED_AGENT_VERSION };
        case FROZEN_ONNX_ROSTER_SOURCE:
          return {
            source: FROZEN_ONNX_ROSTER_SOURCE,
            artifactId: seat.artifact.artifactId ?? basename(seat.artifact.metadataPath),
            onnxFileName: basename(seat.artifact.onnxPath),
            metadataFileName: basename(seat.artifact.metadataPath),
            onnxSha256: await sha256File(seat.artifact.onnxPath),
            metadataSha256: await sha256File(seat.artifact.metadataPath),
            ...runtimeInfoForPolicy(seat.policy),
            metadata: seat.policy.metadata
          };
      }
    }))
  };
}

function normalizeRolloutRosterOptions(
  roster: PlayingSelfPlayRolloutRosterOptions | undefined
): PlayingSelfPlayRolloutRosterOptions {
  return roster ?? {
    seats: Array.from({ length: PLAYER_COUNT }, () => ({ source: CURRENT_POLICY_ROSTER_SOURCE }))
  };
}

function defaultRolloutSeatSources(): readonly PlayingSelfPlayRosterSource[] {
  return normalizeRolloutRosterOptions(undefined).seats.map((seat) => seat.source);
}

function runtimeInfoForPolicy(policy: PlayingSelfPlayPolicy): PlayingSelfPlayPolicyRuntime {
  return policy.runtime ?? {
    requestedInferenceDevice: "cpu",
    resolvedInferenceDevice: "cpu",
    executionProvider: "cpu"
  };
}

function isRequestedInferenceDevice(
  value: unknown
): value is PlayingSelfPlayPolicyRuntime["requestedInferenceDevice"] {
  return value === "cpu" || value === "auto" || value === "cuda";
}

function isResolvedInferenceDevice(
  value: unknown
): value is PlayingSelfPlayPolicyRuntime["resolvedInferenceDevice"] {
  return value === "cpu" || value === "cuda";
}

function validateRolloutRosterOptions(
  roster: PlayingSelfPlayRolloutRosterOptions | undefined
): void {
  const normalized = normalizeRolloutRosterOptions(roster);

  if (!Array.isArray(normalized.seats) || normalized.seats.length !== PLAYER_COUNT) {
    throw new Error(`rolloutRoster.seats must contain exactly ${PLAYER_COUNT} seats.`);
  }

  let currentPolicyCount = 0;

  normalized.seats.forEach((seat, index) => {
    if (
      seat.source !== CURRENT_POLICY_ROSTER_SOURCE &&
      seat.source !== RULE_BASED_ROSTER_SOURCE &&
      seat.source !== FROZEN_ONNX_ROSTER_SOURCE
    ) {
      throw new Error(`rolloutRoster.seats[${index}].source is invalid.`);
    }

    if (seat.source === CURRENT_POLICY_ROSTER_SOURCE) {
      currentPolicyCount += 1;
    }

    if (seat.source === FROZEN_ONNX_ROSTER_SOURCE) {
      if (seat.policy === undefined) {
        throw new Error(`rolloutRoster.seats[${index}].policy is required for frozen-onnx.`);
      }
      if (seat.artifact === undefined) {
        throw new Error(`rolloutRoster.seats[${index}].artifact is required for frozen-onnx.`);
      }
      if (seat.artifact.onnxPath.length === 0 || seat.artifact.metadataPath.length === 0) {
        throw new Error(`rolloutRoster.seats[${index}] frozen-onnx artifact paths are required.`);
      }
    }
  });

  if (currentPolicyCount === 0) {
    throw new Error("rolloutRoster must include at least one current-policy seat.");
  }
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

function validateShards(manifest: PlayingSelfPlayDatasetManifest): void {
  let expectedStartSeed = manifest.startSeed;
  const seenFiles = new Set<string>();

  manifest.shards.forEach((shard, index) => {
    validateShard(shard);

    const expectedFile = manifest.format === PLAYING_SELF_PLAY_BINARY_DATASET_FORMAT
      ? binaryShardFileName(index)
      : `shard-${index.toString().padStart(5, "0")}.jsonl`;
    if (shard.file !== expectedFile) {
      throw new Error(`Shard file must be ${expectedFile}, got ${shard.file}.`);
    }
    if (seenFiles.has(shard.file)) {
      throw new Error(`Duplicate shard file: ${shard.file}`);
    }
    if (shard.startSeed !== expectedStartSeed) {
      throw new Error(`Shard ${shard.file} has a seed gap or overlap.`);
    }
    if (shard.gameCount !== shard.endSeed - shard.startSeed + 1) {
      throw new Error(`Shard ${shard.file} gameCount must match seed range.`);
    }

    const isLastShard = index === manifest.shards.length - 1;
    if (!isLastShard && shard.gameCount !== manifest.gamesPerShard) {
      throw new Error(`Shard ${shard.file} gameCount must match gamesPerShard.`);
    }
    if (isLastShard && shard.gameCount > manifest.gamesPerShard) {
      throw new Error(`Final shard ${shard.file} gameCount must not exceed gamesPerShard.`);
    }

    seenFiles.add(shard.file);
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

  if (shard.endSeed < shard.startSeed) {
    throw new Error(`Shard ${shard.file} endSeed must be greater than or equal to startSeed.`);
  }
  if (!sha256Pattern.test(shard.sha256)) {
    throw new Error(`Shard ${shard.file} sha256 must be lowercase hex.`);
  }
}

function validateUint32(name: string, value: number): void {
  if (!Number.isSafeInteger(value) || value < 0 || value > UINT32_MAX) {
    throw new Error(`${name} must be an integer between 0 and ${UINT32_MAX}.`);
  }
}

function validateUint16(name: string, value: number): void {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff) {
    throw new Error(`${name} must be an integer between 0 and ${0xffff}.`);
  }
}

function validatePositiveInteger(name: string, value: number): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
}

function validateCardIndex(name: string, value: number): void {
  if (!Number.isSafeInteger(value) || value < 0 || value >= CARD_COUNT) {
    throw new Error(`${name} must be an integer between 0 and ${CARD_COUNT - 1}.`);
  }
}

function validatePlayerIndex(name: string, value: number): void {
  if (!Number.isSafeInteger(value) || value < 0 || value >= PLAYER_COUNT) {
    throw new Error(`${name} must be an integer between 0 and ${PLAYER_COUNT - 1}.`);
  }
}

function validateSelfRoleIndex(name: string, value: number): void {
  if (!Number.isSafeInteger(value) || value < 0 || value >= SELF_ROLE_ORDER.length) {
    throw new Error(`${name} must be an integer between 0 and ${SELF_ROLE_ORDER.length - 1}.`);
  }
}

function validateTemperature(temperature: number): void {
  if (!Number.isFinite(temperature) || temperature <= 0) {
    throw new Error("temperature must be finite and greater than 0.");
  }
}

function validateLegalPlayMask(mask: ArrayLike<number | boolean>): void {
  if (mask.length !== CARD_COUNT) {
    throw new Error(`legalPlayMask must contain ${CARD_COUNT} values, got ${mask.length}.`);
  }

  let legalCount = 0;
  for (let index = 0; index < CARD_COUNT; index += 1) {
    const value = mask[index];
    if (value !== 0 && value !== 1 && value !== false && value !== true) {
      throw new Error(`legalPlayMask[${index}] must be 0/1 or boolean.`);
    }
    if (isLegalMaskValue(value)) {
      legalCount += 1;
    }
  }

  if (legalCount === 0) {
    throw new Error("legalPlayMask must contain at least one legal card.");
  }
}

function isLegalMaskValue(value: number | boolean): boolean {
  return value === 1 || value === true;
}

function isRolloutRosterSource(value: unknown): value is PlayingSelfPlayRosterSource {
  return value === CURRENT_POLICY_ROSTER_SOURCE ||
    value === RULE_BASED_ROSTER_SOURCE ||
    value === FROZEN_ONNX_ROSTER_SOURCE;
}

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

async function sha256File(path: string): Promise<string> {
  return createHash("sha256").update(await readFile(path)).digest("hex");
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

function binaryShardFileName(shardIndex: number): string {
  if (shardIndex < 0 || shardIndex >= MAX_SHARD_COUNT) {
    throw new Error(`shardIndex must be between 0 and ${MAX_SHARD_COUNT - 1}.`);
  }
  return `shard-${shardIndex.toString().padStart(5, "0")}.bin`;
}

function sameStringArray(left: readonly string[], right: readonly string[]): boolean {
  if (left.length !== right.length) {
    return false;
  }

  return left.every((value, index) => value === right[index]);
}

function sameNumberArray(left: readonly number[], right: readonly number[]): boolean {
  if (left.length !== right.length) {
    return false;
  }

  return left.every((value, index) => value === right[index]);
}

function isPlayingSelfPlayTensorSample(
  sample: PlayingSelfPlayWritableSample
): sample is PlayingSelfPlayTensorSample {
  return "modelInput" in sample && "legalPlayMask" in sample;
}

function isFloat32ArrayLike(value: unknown): value is Float32Array {
  return (
    value instanceof Float32Array ||
    (
      ArrayBuffer.isView(value) &&
      value.constructor.name === "Float32Array"
    )
  );
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

const sha256Pattern = /^[0-9a-f]{64}$/;
