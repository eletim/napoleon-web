import type { AutomatedGameRecord, DecisionRecord } from "@napoleon/ai";
import { encodeBeliefTarget } from "./encodeBeliefTarget.js";
import type { EncodedBeliefTarget } from "./encodeBeliefTarget.js";
import { encodePlayAction } from "./encodePlayAction.js";
import type { EncodedPlayAction } from "./encodePlayAction.js";
import { encodePlayingObservation } from "./encodePlayingObservation.js";
import type { EncodedPlayingObservation } from "./encodePlayingObservation.js";
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

  if (decision.playerId !== decision.observation.playerId) {
    throw new Error(
      `Decision playerId must match observation playerId: ${decision.playerId} !== ${decision.observation.playerId}`
    );
  }

  const observation = encodePlayingObservation(decision.observation, record.playerIds);
  const actorTarget = encodePlayAction(decision.action, observation.legalPlayMask);
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
