import { createHash } from "node:crypto";
import type { PlayingTrainingSample } from "@napoleon/ai-observation";
import { CARD_IDS } from "@napoleon/ai-observation";
import { DATASET_SAMPLE_TYPE } from "./schema.js";
import type { DatasetSampleType, TrainingSample } from "./types.js";

export function sha256Utf8(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

export function calculateCardIdsSha256(): string {
  return sha256Utf8(JSON.stringify(CARD_IDS));
}

export function serializePlayingTrainingSample(sample: PlayingTrainingSample): string {
  return `${JSON.stringify({
    schemaVersion: sample.schemaVersion,
    seed: sample.seed,
    step: sample.step,
    actingPlayerId: sample.actingPlayerId,
    relativePlayerIds: sample.relativePlayerIds,
    observation: sample.observation,
    actorTarget: sample.actorTarget,
    beliefTarget: sample.beliefTarget
  })}\n`;
}

export function serializeTrainingSample(
  sample: TrainingSample,
  sampleType: DatasetSampleType = inferSampleType(sample)
): string {
  if (sampleType === DATASET_SAMPLE_TYPE) {
    return serializePlayingTrainingSample(sample as PlayingTrainingSample);
  }

  if (!("sampleType" in sample) || sample.sampleType !== sampleType) {
    throw new Error(`Sample sampleType must be ${sampleType}.`);
  }

  return `${JSON.stringify(sample)}\n`;
}

function inferSampleType(sample: TrainingSample): DatasetSampleType {
  return "sampleType" in sample ? sample.sampleType : DATASET_SAMPLE_TYPE;
}

export function serializeManifest(manifest: unknown): string {
  return `${JSON.stringify(manifest, null, 2)}\n`;
}
