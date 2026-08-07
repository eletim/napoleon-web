import type { AutomatedGameRecord, DecisionRecord, PublicActionRecord } from "@napoleon/ai";
import type { GameAction } from "@napoleon/game-core";
import {
  encodeBiddingHistoryFromPublicActions
} from "./encodeBiddingHistory.js";
import { encodeDiscardAction, validateEncodedExchangeAction } from "./encodeExchangeAction.js";
import type { EncodedExchangeAction } from "./encodeExchangeAction.js";
import {
  encodeExchangeObservation,
  validateEncodedExchangeObservation
} from "./encodeExchangeObservation.js";
import type { EncodedExchangeObservation } from "./encodeExchangeObservation.js";
import { createRelativePlayerOrder } from "./playerIndex.js";
import { EXCHANGE_ENCODER_SCHEMA_VERSION } from "./schema.js";

export interface ExchangeTrainingSample {
  sampleType: "exchange-training-sample";
  schemaVersion: typeof EXCHANGE_ENCODER_SCHEMA_VERSION;
  seed: number;
  step: number;
  actingPlayerId: string;
  relativePlayerIds: readonly string[];
  observation: EncodedExchangeObservation;
  actorTarget: EncodedExchangeAction;
}

export function createExchangeTrainingSample(
  record: AutomatedGameRecord,
  decision: DecisionRecord
): ExchangeTrainingSample | null {
  if (decision.phase !== "exchanging") {
    return null;
  }

  if (decision.action.type !== "discard-cards") {
    throw new Error(`Exchange decision action must be discard-cards, got ${decision.action.type}.`);
  }

  validateExchangeDecisionConsistency(record, decision);

  const relativePlayerIds = createRelativePlayerOrder(record.playerIds, decision.playerId);
  const biddingHistory = encodeBiddingHistoryFromPublicActions(
    record.decisions
      .filter((candidate) => candidate.step < decision.step)
      .flatMap(toPublicBiddingRecord),
    relativePlayerIds
  );
  const observation = encodeExchangeObservation(
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

  const actorTarget = encodeDiscardAction(
    decision.action,
    observation.legalDiscardCardMask,
    decision.playerId
  );
  const sample: ExchangeTrainingSample = {
    sampleType: "exchange-training-sample",
    schemaVersion: EXCHANGE_ENCODER_SCHEMA_VERSION,
    seed: record.seed,
    step: decision.step,
    actingPlayerId: decision.playerId,
    relativePlayerIds: observation.relativePlayerIds,
    observation,
    actorTarget
  };

  validateExchangeTrainingSample(sample);

  return sample;
}

export function createExchangeTrainingSamples(
  record: AutomatedGameRecord
): readonly ExchangeTrainingSample[] {
  return record.decisions.flatMap((decision) => {
    const sample = createExchangeTrainingSample(record, decision);

    return sample === null ? [] : [sample];
  });
}

export function validateExchangeTrainingSample(sample: ExchangeTrainingSample): void {
  if (sample.sampleType !== "exchange-training-sample") {
    throw new Error(`Unsupported exchange sample type: ${sample.sampleType}`);
  }

  if (sample.schemaVersion !== EXCHANGE_ENCODER_SCHEMA_VERSION) {
    throw new Error(`Unsupported exchange encoder schema version: ${sample.schemaVersion}`);
  }

  validateEncodedExchangeObservation(sample.observation);
  validateEncodedExchangeAction(sample.actorTarget, sample.observation.legalDiscardCardMask);

  if (!Number.isFinite(sample.seed) || !Number.isInteger(sample.seed)) {
    throw new Error("Exchange sample seed must be a finite integer.");
  }

  expectIntegerInRange("Exchange sample step", sample.step, 1, Number.MAX_SAFE_INTEGER);

  if (sample.relativePlayerIds[0] !== sample.actingPlayerId) {
    throw new Error("Exchange sample relative player index 0 must be Napoleon.");
  }

  if (!samePlayerOrder(sample.relativePlayerIds, sample.observation.relativePlayerIds)) {
    throw new Error("Exchange sample relativePlayerIds must match observation.relativePlayerIds.");
  }
}

function validateExchangeDecisionConsistency(
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

function toPublicBiddingRecord(decision: DecisionRecord): readonly PublicActionRecord[] {
  if (decision.phase !== "bidding") {
    return [];
  }

  if (decision.action.type !== "bid" && decision.action.type !== "pass") {
    throw new Error(
      `Bidding history supports only pass and bid actions, got ${(decision.action as GameAction).type}.`
    );
  }

  return [{
    step: decision.step,
    playerId: decision.playerId,
    phase: "bidding",
    action: decision.action
  }];
}
