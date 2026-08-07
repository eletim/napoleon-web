import { CARD_IDS } from "@napoleon/ai-observation";
import {
  ADJUTANT_ENCODER_SCHEMA_VERSION,
  BIDDING_ENCODER_SCHEMA_VERSION,
  CARD_COUNT,
  EXCHANGE_ENCODER_SCHEMA_VERSION,
  PLAYER_COUNT,
  PLAYING_ENCODER_SCHEMA_VERSION,
  validateAdjutantTrainingSample as validateAdjutantTrainingSampleSchema,
  validateBiddingTrainingSample as validateBiddingTrainingSampleSchema,
  validateEncodedBeliefTarget,
  validateEncodedPlayingObservation,
  validateExchangeTrainingSample as validateExchangeTrainingSampleSchema
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
  DATASET_FORMAT,
  DATASET_GENERATOR_VERSION,
  DATASET_SAMPLE_TYPE,
  DATASET_SAMPLE_TYPES,
  DATASET_SCHEMA_VERSION,
  EXCHANGE_DATASET_SAMPLE_TYPE,
  MULTIPHASE_DATASET_GENERATOR_VERSION,
  MULTIPHASE_DATASET_SCHEMA_VERSION,
  RULE_BASED_AGENT_VERSION,
  MAX_SHARD_COUNT,
  PLAYING_DATASET_SAMPLE_TYPE,
  SHARD_FILE_DIGITS,
  UINT32_MAX
} from "./schema.js";
import type {
  DatasetShardManifest,
  DatasetSampleType,
  GenerateRuleBasedDatasetOptions,
  RuleBasedDatasetManifest,
  TrainingSample
} from "./types.js";
import { calculateCardIdsSha256 } from "./serialization.js";

const sha256Pattern = /^[0-9a-f]{64}$/;

export function validateGenerationOptions(options: GenerateRuleBasedDatasetOptions): void {
  const sampleType = options.sampleType ?? DATASET_SAMPLE_TYPE;

  if (!isDatasetSampleType(sampleType)) {
    throw new Error(`Unsupported dataset sampleType: ${String(sampleType)}`);
  }

  validateUint32("startSeed", options.startSeed);
  validatePositiveInteger("gameCount", options.gameCount);
  validatePositiveInteger("gamesPerShard", options.gamesPerShard);

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

export function validatePlayingTrainingSample(
  sample: PlayingTrainingSample,
  expectedSeed: number
): void {
  validateJsonSafeValue("sample", sample);

  if (sample.schemaVersion !== PLAYING_ENCODER_SCHEMA_VERSION) {
    throw new Error(`Unexpected sample schemaVersion: ${sample.schemaVersion}`);
  }

  if (sample.observation.schemaVersion !== sample.schemaVersion) {
    throw new Error("Sample schemaVersion must match observation schemaVersion.");
  }

  if (sample.seed !== expectedSeed) {
    throw new Error(`Sample seed must match current seed: ${sample.seed} !== ${expectedSeed}`);
  }

  if (sample.actingPlayerId !== sample.relativePlayerIds[0]) {
    throw new Error("actingPlayerId must match relativePlayerIds[0].");
  }

  if (!sameStringArray(sample.relativePlayerIds, sample.observation.relativePlayerIds)) {
    throw new Error("sample.relativePlayerIds must match observation.relativePlayerIds.");
  }

  validateEncodedPlayingObservation(sample.observation);
  validateEncodedBeliefTarget(sample.beliefTarget);
  validateCardIndex("actorTarget.selectedCardIndex", sample.actorTarget.selectedCardIndex);

  if (sample.observation.legalPlayMask[sample.actorTarget.selectedCardIndex] !== 1) {
    throw new Error("actorTarget.selectedCardIndex must be legal in observation.legalPlayMask.");
  }

  sample.observation.legalPlayMask.forEach((value, index) => {
    if (value === 1 && sample.observation.selfHandMask[index] !== 1) {
      throw new Error(`legalPlayMask[${index}] must be within selfHandMask.`);
    }
  });

  JSON.stringify(sample);
}

export function validateTrainingSample(
  sample: TrainingSample,
  expectedSeed: number,
  expectedSampleType: DatasetSampleType
): void {
  switch (expectedSampleType) {
    case PLAYING_DATASET_SAMPLE_TYPE:
      validatePlayingTrainingSample(sample as PlayingTrainingSample, expectedSeed);
      return;
    case BIDDING_DATASET_SAMPLE_TYPE:
      validateSampleCommon(sample, expectedSeed, BIDDING_DATASET_SAMPLE_TYPE);
      validateBiddingTrainingSampleSchema(sample as BiddingTrainingSample);
      return;
    case EXCHANGE_DATASET_SAMPLE_TYPE:
      validateSampleCommon(sample, expectedSeed, EXCHANGE_DATASET_SAMPLE_TYPE);
      validateExchangeTrainingSampleSchema(sample as ExchangeTrainingSample);
      return;
    case ADJUTANT_DATASET_SAMPLE_TYPE:
      validateSampleCommon(sample, expectedSeed, ADJUTANT_DATASET_SAMPLE_TYPE);
      validateAdjutantTrainingSampleSchema(sample as AdjutantTrainingSample);
      return;
  }
}

export function validateDatasetManifest(manifest: RuleBasedDatasetManifest): void {
  if (
    manifest.datasetSchemaVersion !== DATASET_SCHEMA_VERSION &&
    manifest.datasetSchemaVersion !== MULTIPHASE_DATASET_SCHEMA_VERSION
  ) {
    throw new Error("Manifest datasetSchemaVersion mismatch.");
  }

  if (manifest.format !== DATASET_FORMAT || !isDatasetSampleType(manifest.sampleType)) {
    throw new Error("Manifest format or sampleType mismatch.");
  }

  validateManifestSchemaIdentity(manifest);

  if (
    manifest.agent.type !== "rule-based" ||
    manifest.agent.version !== RULE_BASED_AGENT_VERSION
  ) {
    throw new Error("Manifest agent metadata mismatch.");
  }

  validateManifestNumbers(manifest);

  if (manifest.gameCount !== manifest.endSeed - manifest.startSeed + 1) {
    throw new Error("Manifest gameCount must match seed range.");
  }

  const expectedShardCount = calculateExpectedShardCount(
    manifest.gameCount,
    manifest.gamesPerShard
  );

  if (manifest.shardCount !== expectedShardCount) {
    throw new Error("Manifest shardCount must match gameCount and gamesPerShard.");
  }

  if (manifest.shardCount !== manifest.shards.length) {
    throw new Error("Manifest shardCount must match shards length.");
  }

  if (manifest.sampleCount !== sum(manifest.shards.map((shard) => shard.sampleCount))) {
    throw new Error("Manifest sampleCount must equal shard sample counts.");
  }

  if (manifest.playerCount !== PLAYER_COUNT || manifest.cardCount !== CARD_COUNT) {
    throw new Error("Manifest fixed dimensions mismatch.");
  }

  if (!sameStringArray(manifest.cardIds, CARD_IDS)) {
    throw new Error("Manifest cardIds must match encoder CARD_IDS.");
  }

  if (manifest.cardIdsSha256 !== calculateCardIdsSha256()) {
    throw new Error("Manifest cardIdsSha256 mismatch.");
  }

  validateShards(manifest);
}

export function getEncoderSchemaVersion(sampleType: DatasetSampleType): number {
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

export function isDatasetSampleType(value: unknown): value is DatasetSampleType {
  return typeof value === "string" &&
    (DATASET_SAMPLE_TYPES as readonly string[]).includes(value);
}

export function shardFileName(shardIndex: number): string {
  if (
    !Number.isSafeInteger(shardIndex) ||
    shardIndex < 0 ||
    shardIndex >= MAX_SHARD_COUNT
  ) {
    throw new Error(`shardIndex must be an integer between 0 and ${MAX_SHARD_COUNT - 1}.`);
  }

  return `shard-${shardIndex.toString().padStart(SHARD_FILE_DIGITS, "0")}.jsonl`;
}

function validateShards(manifest: RuleBasedDatasetManifest): void {
  let expectedStartSeed = manifest.startSeed;
  const seenFiles = new Set<string>();

  manifest.shards.forEach((shard, index) => {
    validateShard(shard);

    const expectedFile = shardFileName(index);

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

    const isLastShard = index === manifest.shards.length - 1;

    if (!isLastShard && shard.gameCount !== manifest.gamesPerShard) {
      throw new Error(`Shard ${shard.file} gameCount must match gamesPerShard.`);
    }

    if (isLastShard && shard.gameCount > manifest.gamesPerShard) {
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

  if (expectedStartSeed !== manifest.endSeed + 1) {
    throw new Error("Shard seed ranges must cover the dataset range.");
  }
}

function validateManifestNumbers(manifest: RuleBasedDatasetManifest): void {
  validateUint32("Manifest startSeed", manifest.startSeed);
  validateUint32("Manifest endSeed", manifest.endSeed);

  if (manifest.endSeed < manifest.startSeed) {
    throw new Error("Manifest endSeed must be greater than or equal to startSeed.");
  }

  validatePositiveInteger("Manifest gameCount", manifest.gameCount);
  validatePositiveInteger("Manifest sampleCount", manifest.sampleCount);
  validatePositiveInteger("Manifest gamesPerShard", manifest.gamesPerShard);
  validatePositiveInteger("Manifest shardCount", manifest.shardCount);

  if (manifest.shardCount > MAX_SHARD_COUNT) {
    throw new Error(`Manifest shardCount must not exceed ${MAX_SHARD_COUNT}.`);
  }

  if (manifest.playerCount !== PLAYER_COUNT) {
    throw new Error(`Manifest playerCount must be ${PLAYER_COUNT}.`);
  }

  if (manifest.cardCount !== CARD_COUNT) {
    throw new Error(`Manifest cardCount must be ${CARD_COUNT}.`);
  }
}

function validateManifestSchemaIdentity(manifest: RuleBasedDatasetManifest): void {
  if (manifest.datasetSchemaVersion === DATASET_SCHEMA_VERSION) {
    if (manifest.generatorVersion !== DATASET_GENERATOR_VERSION) {
      throw new Error("Manifest generatorVersion mismatch.");
    }

    if (manifest.sampleType !== DATASET_SAMPLE_TYPE) {
      throw new Error("Manifest v1 sampleType must be playing-training-sample.");
    }

    if (
      !("playingEncoderSchemaVersion" in manifest) ||
      manifest.playingEncoderSchemaVersion !== PLAYING_ENCODER_SCHEMA_VERSION
    ) {
      throw new Error("Manifest playingEncoderSchemaVersion mismatch.");
    }

    return;
  }

  if (manifest.generatorVersion !== MULTIPHASE_DATASET_GENERATOR_VERSION) {
    throw new Error("Manifest generatorVersion mismatch.");
  }

  const sampleType = manifest.sampleType as DatasetSampleType;

  if (sampleType === DATASET_SAMPLE_TYPE) {
    throw new Error("Manifest v2 sampleType must use a non-playing sample type.");
  }

  if (
    !("encoderSchemaVersion" in manifest) ||
    manifest.encoderSchemaVersion !== getEncoderSchemaVersion(sampleType)
  ) {
    throw new Error("Manifest encoderSchemaVersion mismatch.");
  }
}

function validateSampleCommon(
  sample: TrainingSample,
  expectedSeed: number,
  expectedSampleType: DatasetSampleType
): void {
  validateJsonSafeValue("sample", sample);

  if (!("sampleType" in sample) || sample.sampleType !== expectedSampleType) {
    throw new Error(`Sample sampleType must be ${expectedSampleType}.`);
  }

  if (sample.seed !== expectedSeed) {
    throw new Error(`Sample seed must match current seed: ${sample.seed} !== ${expectedSeed}`);
  }
}

function validateShard(shard: DatasetShardManifest): void {
  validateUint32(`Shard ${shard.file} startSeed`, shard.startSeed);
  validateUint32(`Shard ${shard.file} endSeed`, shard.endSeed);

  if (shard.endSeed < shard.startSeed) {
    throw new Error(`Shard ${shard.file} endSeed must be greater than or equal to startSeed.`);
  }

  validatePositiveInteger(`Shard ${shard.file} gameCount`, shard.gameCount);
  validatePositiveInteger(`Shard ${shard.file} sampleCount`, shard.sampleCount);
  validatePositiveInteger(`Shard ${shard.file} byteLength`, shard.byteLength);

  if (!sha256Pattern.test(shard.sha256)) {
    throw new Error(`Shard ${shard.file} sha256 must be lowercase hex.`);
  }
}

function calculateExpectedShardCount(gameCount: number, gamesPerShard: number): number {
  const expectedShardCount = Math.ceil(gameCount / gamesPerShard);

  if (!Number.isSafeInteger(expectedShardCount) || expectedShardCount < 1) {
    throw new Error("Expected shard count must be a positive safe integer.");
  }

  return expectedShardCount;
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

function validateCardIndex(name: string, value: number): void {
  if (!Number.isSafeInteger(value) || value < 0 || value >= CARD_COUNT) {
    throw new Error(`${name} must be an integer between 0 and ${CARD_COUNT - 1}.`);
  }
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

function sameStringArray(left: readonly string[], right: readonly string[]): boolean {
  if (left.length !== right.length) {
    return false;
  }

  return left.every((value, index) => value === right[index]);
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}
