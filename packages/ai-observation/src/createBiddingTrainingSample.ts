import type { AutomatedGameRecord, DecisionRecord } from "@napoleon/ai";
import { encodeBiddingAction } from "./encodeBiddingAction.js";
import {
  encodeBiddingObservation,
  validateEncodedBiddingObservation,
  type EncodedBiddingObservation
} from "./encodeBiddingObservation.js";
import { createRelativePlayerOrder } from "./playerIndex.js";
import { BIDDING_ACTION_COUNT, BIDDING_ENCODER_SCHEMA_VERSION } from "./schema.js";

export interface BiddingTrainingSample {
  sampleType: "bidding-training-sample";
  schemaVersion: typeof BIDDING_ENCODER_SCHEMA_VERSION;
  seed: number;
  step: number;
  actingPlayerId: string;
  relativePlayerIds: readonly string[];
  observation: EncodedBiddingObservation;
  actorTarget: number;
}

export function createBiddingTrainingSample(
  record: AutomatedGameRecord,
  decision: DecisionRecord
): BiddingTrainingSample | null {
  if (decision.phase !== "bidding") {
    return null;
  }

  if (decision.action.type !== "pass" && decision.action.type !== "bid") {
    throw new Error(`Bidding decision action must be pass or bid, got ${decision.action.type}.`);
  }

  validateBiddingDecisionConsistency(record, decision);

  const relativePlayerIds = createRelativePlayerOrder(record.playerIds, decision.playerId);
  const observation = encodeBiddingObservation(decision.observation, record.playerIds);

  if (observation.relativePlayerIds[0] !== decision.playerId) {
    throw new Error(
      `Observation relative player index 0 must be the acting player: ${observation.relativePlayerIds[0]} !== ${decision.playerId}`
    );
  }

  if (!samePlayerOrder(observation.relativePlayerIds, relativePlayerIds)) {
    throw new Error("Observation relative player order must match the sample player order.");
  }

  const actorTarget = encodeBiddingAction(
    decision.action,
    observation.legalBidMask,
    decision.playerId
  );
  const sample: BiddingTrainingSample = {
    sampleType: "bidding-training-sample",
    schemaVersion: BIDDING_ENCODER_SCHEMA_VERSION,
    seed: record.seed,
    step: decision.step,
    actingPlayerId: decision.playerId,
    relativePlayerIds: observation.relativePlayerIds,
    observation,
    actorTarget
  };

  validateBiddingTrainingSample(sample);

  return sample;
}

export function createBiddingTrainingSamples(
  record: AutomatedGameRecord
): readonly BiddingTrainingSample[] {
  return record.decisions.flatMap((decision) => {
    const sample = createBiddingTrainingSample(record, decision);

    return sample === null ? [] : [sample];
  });
}

export function validateBiddingTrainingSample(sample: BiddingTrainingSample): void {
  if (sample.sampleType !== "bidding-training-sample") {
    throw new Error(`Unsupported bidding sample type: ${sample.sampleType}`);
  }

  if (sample.schemaVersion !== BIDDING_ENCODER_SCHEMA_VERSION) {
    throw new Error(`Unsupported bidding encoder schema version: ${sample.schemaVersion}`);
  }

  validateEncodedBiddingObservation(sample.observation);

  if (!Number.isFinite(sample.seed) || !Number.isInteger(sample.seed)) {
    throw new Error("Bidding sample seed must be a finite integer.");
  }

  expectIntegerInRange("Bidding sample step", sample.step, 1, Number.MAX_SAFE_INTEGER);
  expectIntegerInRange("Bidding actorTarget", sample.actorTarget, 0, BIDDING_ACTION_COUNT - 1);

  if (sample.relativePlayerIds[0] !== sample.actingPlayerId) {
    throw new Error("Bidding sample relative player index 0 must be the acting player.");
  }

  if (!samePlayerOrder(sample.relativePlayerIds, sample.observation.relativePlayerIds)) {
    throw new Error("Bidding sample relativePlayerIds must match observation.relativePlayerIds.");
  }

  if (sample.observation.legalBidMask[sample.actorTarget] !== 1) {
    throw new Error(`Bidding actorTarget ${sample.actorTarget} must be inside legalBidMask.`);
  }
}

function validateBiddingDecisionConsistency(
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
