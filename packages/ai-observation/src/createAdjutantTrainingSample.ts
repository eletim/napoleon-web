import type { AutomatedGameRecord, DecisionRecord } from "@napoleon/ai";
import {
  encodeAdjutantAction,
  validateLegalAdjutantMask
} from "./encodeAdjutantAction.js";
import {
  encodeAdjutantObservation,
  validateEncodedAdjutantObservation
} from "./encodeAdjutantObservation.js";
import type { EncodedAdjutantObservation } from "./encodeAdjutantObservation.js";
import { encodeBiddingHistoryBeforeDecision } from "./encodeBiddingHistory.js";
import { createRelativePlayerOrder } from "./playerIndex.js";
import { ADJUTANT_ENCODER_SCHEMA_VERSION, CARD_COUNT } from "./schema.js";

export interface AdjutantTrainingSample {
  sampleType: "adjutant-training-sample";
  schemaVersion: typeof ADJUTANT_ENCODER_SCHEMA_VERSION;
  seed: number;
  step: number;
  actingPlayerId: string;
  relativePlayerIds: readonly string[];
  observation: EncodedAdjutantObservation;
  actorTarget: number;
}

export function createAdjutantTrainingSample(
  record: AutomatedGameRecord,
  decision: DecisionRecord
): AdjutantTrainingSample | null {
  if (decision.phase !== "choosing-adjutant") {
    return null;
  }

  if (decision.action.type !== "choose-adjutant") {
    throw new Error(
      `Adjutant decision action must be choose-adjutant, got ${decision.action.type}.`
    );
  }

  validateAdjutantDecisionConsistency(record, decision);

  const relativePlayerIds = createRelativePlayerOrder(record.playerIds, decision.playerId);
  const biddingHistory = encodeBiddingHistoryBeforeDecision(record, decision, relativePlayerIds);
  const observation = encodeAdjutantObservation(
    decision.observation,
    record.playerIds,
    biddingHistory
  );

  if (observation.relativePlayerIds[0] !== decision.playerId) {
    throw new Error(
      `Observation relative player index 0 must be Napoleon: ${observation.relativePlayerIds[0]} !== ${decision.playerId}`
    );
  }

  if (!samePlayerOrder(observation.relativePlayerIds, relativePlayerIds)) {
    throw new Error("Observation relative player order must match the sample player order.");
  }

  const actorTarget = encodeAdjutantAction(
    decision.action,
    observation.legalAdjutantMask,
    decision.playerId
  );
  const sample: AdjutantTrainingSample = {
    sampleType: "adjutant-training-sample",
    schemaVersion: ADJUTANT_ENCODER_SCHEMA_VERSION,
    seed: record.seed,
    step: decision.step,
    actingPlayerId: decision.playerId,
    relativePlayerIds: observation.relativePlayerIds,
    observation,
    actorTarget
  };

  validateAdjutantTrainingSample(sample);

  return sample;
}

export function createAdjutantTrainingSamples(
  record: AutomatedGameRecord
): readonly AdjutantTrainingSample[] {
  return record.decisions.flatMap((decision) => {
    const sample = createAdjutantTrainingSample(record, decision);

    return sample === null ? [] : [sample];
  });
}

export function validateAdjutantTrainingSample(sample: AdjutantTrainingSample): void {
  if (sample.sampleType !== "adjutant-training-sample") {
    throw new Error(`Unsupported adjutant sample type: ${sample.sampleType}`);
  }

  if (sample.schemaVersion !== ADJUTANT_ENCODER_SCHEMA_VERSION) {
    throw new Error(`Unsupported adjutant encoder schema version: ${sample.schemaVersion}`);
  }

  validateEncodedAdjutantObservation(sample.observation);
  validateLegalAdjutantMask(sample.observation.legalAdjutantMask);

  if (!Number.isFinite(sample.seed) || !Number.isInteger(sample.seed)) {
    throw new Error("Adjutant sample seed must be a finite integer.");
  }

  expectIntegerInRange("Adjutant sample step", sample.step, 1, Number.MAX_SAFE_INTEGER);
  expectIntegerInRange("Adjutant actorTarget", sample.actorTarget, 0, CARD_COUNT - 1);

  if (sample.relativePlayerIds[0] !== sample.actingPlayerId) {
    throw new Error("Adjutant sample relative player index 0 must be Napoleon.");
  }

  if (!samePlayerOrder(sample.relativePlayerIds, sample.observation.relativePlayerIds)) {
    throw new Error("Adjutant sample relativePlayerIds must match observation.relativePlayerIds.");
  }

  if (sample.observation.legalAdjutantMask[sample.actorTarget] !== 1) {
    throw new Error(`Adjutant actorTarget ${sample.actorTarget} must be inside legalAdjutantMask.`);
  }
}

function validateAdjutantDecisionConsistency(
  record: AutomatedGameRecord,
  decision: DecisionRecord
): void {
  if (!record.playerIds.includes(decision.playerId)) {
    throw new Error(`Decision playerId must exist in record.playerIds: ${decision.playerId}`);
  }

  if (decision.playerId !== decision.observation.playerId) {
    throw new Error(
      `Decision playerId must match observation playerId: ${decision.playerId} !== ${decision.observation.playerId}`
    );
  }

  if (decision.observation.view.selfId !== decision.playerId) {
    throw new Error(
      `Decision playerId must match observation view.selfId: ${decision.playerId} !== ${decision.observation.view.selfId}`
    );
  }

  if (decision.action.playerId !== decision.playerId) {
    throw new Error(
      `Decision action playerId must match decision playerId: ${decision.action.playerId} !== ${decision.playerId}`
    );
  }

  const sourceDecision = record.decisions.find(
    (candidate) => candidate.step === decision.step
  );

  if (sourceDecision === undefined) {
    throw new Error(`Decision step must exist in record.decisions: ${decision.step}`);
  }

  if (
    sourceDecision.playerId !== decision.playerId ||
    sourceDecision.phase !== decision.phase ||
    !actionsEqual(sourceDecision.action, decision.action)
  ) {
    throw new Error(`Decision must match record decision at step ${decision.step}.`);
  }
}

function actionsEqual(left: DecisionRecord["action"], right: DecisionRecord["action"]): boolean {
  if (left.type !== right.type || left.playerId !== right.playerId) {
    return false;
  }

  switch (left.type) {
    case "bid":
      return right.type === "bid" &&
        left.suit === right.suit &&
        left.targetPointCards === right.targetPointCards;
    case "pass":
      return right.type === "pass";
    case "choose-adjutant":
      return right.type === "choose-adjutant" && left.cardId === right.cardId;
    case "discard-cards":
      return right.type === "discard-cards" && sameCardIds(left.cardIds, right.cardIds);
    case "play-card":
      return right.type === "play-card" && left.cardId === right.cardId;
  }
}

function sameCardIds(left: readonly string[], right: readonly string[]): boolean {
  if (left.length !== right.length) {
    return false;
  }

  const leftIds = new Set(left);
  const rightIds = new Set(right);

  if (leftIds.size !== left.length || rightIds.size !== right.length) {
    return false;
  }

  return left.every((cardId) => rightIds.has(cardId));
}

function samePlayerOrder(left: readonly string[], right: readonly string[]): boolean {
  if (left.length !== right.length) {
    return false;
  }

  return left.every((playerId, index) => playerId === right[index]);
}

function expectIntegerInRange(name: string, value: number, min: number, max: number): void {
  if (!Number.isFinite(value) || !Number.isInteger(value)) {
    throw new Error(`${name} must be a finite integer.`);
  }

  if (value < min || value > max) {
    throw new Error(`${name} must be between ${min} and ${max}, got ${value}.`);
  }
}
