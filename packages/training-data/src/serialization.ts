import { createHash } from "node:crypto";
import type { PlayingTrainingSample } from "@napoleon/ai-observation";
import { CARD_IDS } from "@napoleon/ai-observation";

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

export function serializeManifest(manifest: unknown): string {
  return `${JSON.stringify(manifest, null, 2)}\n`;
}
