import { CARD_IDS } from "@napoleon/ai-observation";
import {
  CARD_COUNT,
  PLAYER_COUNT,
  PLAYING_ENCODER_SCHEMA_VERSION,
  validateEncodedBeliefTarget,
  validateEncodedPlayingObservation
} from "@napoleon/ai-observation";
import type { PlayingTrainingSample } from "@napoleon/ai-observation";
import {
  DATASET_FORMAT,
  DATASET_GENERATOR_VERSION,
  DATASET_SAMPLE_TYPE,
  DATASET_SCHEMA_VERSION,
  RULE_BASED_AGENT_VERSION,
  SHARD_FILE_DIGITS,
  UINT32_MAX
} from "./schema.js";
import type {
  DatasetManifest,
  DatasetShardManifest,
  GenerateRuleBasedDatasetOptions
} from "./types.js";
import { calculateCardIdsSha256 } from "./serialization.js";

const sha256Pattern = /^[0-9a-f]{64}$/;

export function validateGenerationOptions(options: GenerateRuleBasedDatasetOptions): void {
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

export function validateDatasetManifest(manifest: DatasetManifest): void {
  if (manifest.datasetSchemaVersion !== DATASET_SCHEMA_VERSION) {
    throw new Error("Manifest datasetSchemaVersion mismatch.");
  }

  if (manifest.generatorVersion !== DATASET_GENERATOR_VERSION) {
    throw new Error("Manifest generatorVersion mismatch.");
  }

  if (manifest.playingEncoderSchemaVersion !== PLAYING_ENCODER_SCHEMA_VERSION) {
    throw new Error("Manifest playingEncoderSchemaVersion mismatch.");
  }

  if (manifest.format !== DATASET_FORMAT || manifest.sampleType !== DATASET_SAMPLE_TYPE) {
    throw new Error("Manifest format or sampleType mismatch.");
  }

  if (
    manifest.agent.type !== "rule-based" ||
    manifest.agent.version !== RULE_BASED_AGENT_VERSION
  ) {
    throw new Error("Manifest agent metadata mismatch.");
  }

  if (manifest.gameCount !== manifest.endSeed - manifest.startSeed + 1) {
    throw new Error("Manifest gameCount must match seed range.");
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

export function shardFileName(shardIndex: number): string {
  return `shard-${shardIndex.toString().padStart(SHARD_FILE_DIGITS, "0")}.jsonl`;
}

function validateShards(manifest: DatasetManifest): void {
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

function validateShard(shard: DatasetShardManifest): void {
  if (shard.gameCount <= 0 || !Number.isSafeInteger(shard.gameCount)) {
    throw new Error(`Shard ${shard.file} gameCount must be a positive integer.`);
  }

  if (shard.sampleCount <= 0 || !Number.isSafeInteger(shard.sampleCount)) {
    throw new Error(`Shard ${shard.file} sampleCount must be a positive integer.`);
  }

  if (shard.byteLength <= 0 || !Number.isSafeInteger(shard.byteLength)) {
    throw new Error(`Shard ${shard.file} byteLength must be a positive integer.`);
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
    return;
  }

  Object.entries(value).forEach(([key, entry]) =>
    validateJsonSafeValue(`${name}.${key}`, entry, seen)
  );
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
