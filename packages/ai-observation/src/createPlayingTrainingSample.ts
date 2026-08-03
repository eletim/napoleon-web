import type { AutomatedGameRecord, DecisionRecord } from "@napoleon/ai";
import { encodeBeliefTarget } from "./encodeBeliefTarget.js";
import type { EncodedBeliefTarget } from "./encodeBeliefTarget.js";
import { encodeBiddingHistory } from "./encodeBiddingHistory.js";
import { encodePlayAction } from "./encodePlayAction.js";
import type { EncodedPlayAction } from "./encodePlayAction.js";
import { encodePlayingObservation } from "./encodePlayingObservation.js";
import type { EncodedPlayingObservation } from "./encodePlayingObservation.js";
import { createRelativePlayerOrder } from "./playerIndex.js";
import { PLAYING_ENCODER_SCHEMA_VERSION } from "./schema.js";

export interface PlayingTrainingSample {
  schemaVersion: typeof PLAYING_ENCODER_SCHEMA_VERSION;
  seed: number;
  step: number;
  actingPlayerId: string;
  relativePlayerIds: readonly string[];
  observation: EncodedPlayingObservation;
  actorTarget: EncodedPlayAction;
  beliefTarget: EncodedBeliefTarget;
}

export function createPlayingTrainingSample(
  record: AutomatedGameRecord,
  decision: DecisionRecord
): PlayingTrainingSample | null {
  if (decision.phase !== "playing") {
    return null;
  }

  if (decision.action.type !== "play-card") {
    throw new Error(`Playing decision action must be play-card, got ${decision.action.type}.`);
  }

  validatePlayingDecisionConsistency(record, decision);

  const relativePlayerIds = createRelativePlayerOrder(record.playerIds, decision.playerId);
  const biddingHistory = encodeBiddingHistory(
    record,
    decision,
    relativePlayerIds
  );
  const observation = encodePlayingObservation(
    decision.observation,
    record.playerIds,
    biddingHistory
  );

  if (observation.relativePlayerIds[0] !== decision.playerId) {
    throw new Error(
      `Observation relative player index 0 must be the acting player: ${observation.relativePlayerIds[0]} !== ${decision.playerId}`
    );
  }

  if (!samePlayerOrder(observation.relativePlayerIds, relativePlayerIds)) {
    throw new Error("Observation relative player order must match the sample player order.");
  }

  const actorTarget = encodePlayAction(
    decision.action,
    observation.legalPlayMask,
    decision.playerId
  );
  const beliefTarget = encodeBeliefTarget(
    decision.observation,
    decision.actualState,
    record.playerIds
  );

  return {
    schemaVersion: PLAYING_ENCODER_SCHEMA_VERSION,
    seed: record.seed,
    step: decision.step,
    actingPlayerId: decision.playerId,
    relativePlayerIds: observation.relativePlayerIds,
    observation,
    actorTarget,
    beliefTarget
  };
}

export function createPlayingTrainingSamples(
  record: AutomatedGameRecord
): readonly PlayingTrainingSample[] {
  return record.decisions.flatMap((decision) => {
    const sample = createPlayingTrainingSample(record, decision);

    return sample === null ? [] : [sample];
  });
}

function validatePlayingDecisionConsistency(
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
