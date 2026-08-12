import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { loadPolicyOnnxModel } from "@napoleon/policy-onnx";
import {
  CURRENT_POLICY_ROSTER_SOURCE,
  FROZEN_ONNX_ROSTER_SOURCE,
  RULE_BASED_ROSTER_SOURCE,
  type PlayingSelfPlayInferenceStats,
  runPlayingSelfPlayGameWithSamples,
  type PlayingSelfPlayPolicy,
  type PlayingSelfPlayRolloutRosterOptions
} from "@napoleon/training-data";
import type {
  PlayingSelfPlayWorkerMessage,
  PlayingSelfPlayWorkerResponse,
  WorkerPolicyFingerprint,
  WorkerRolloutRosterFingerprint,
  WorkerRolloutRosterSeat
} from "./playingSelfPlayWorkerProtocol.js";

let currentPolicy: PlayingSelfPlayPolicy | null = null;
let rolloutRoster: PlayingSelfPlayRolloutRosterOptions | undefined;
let temperature = 1;
let maxDecisionSteps: number | undefined;
let observationVariant: "public" | "complete-info-compact" = "public";

process.on("message", (message: PlayingSelfPlayWorkerMessage) => {
  void handleMessage(message).catch((error: unknown) => {
    sendError(undefined, error);
  });
});

async function handleMessage(message: PlayingSelfPlayWorkerMessage): Promise<void> {
  switch (message.type) {
    case "init":
      currentPolicy = await loadPolicyOnnxModel(message.currentPolicy);
      rolloutRoster = await loadWorkerRolloutRoster(
        message.rolloutRoster,
        message.currentPolicy.inferenceDevice,
        message.currentPolicy.inferenceMaxBatchSize
      );
      temperature = message.temperature;
      maxDecisionSteps = message.maxDecisionSteps;
      observationVariant = message.observationVariant;
      send({
        type: "ready",
        currentPolicy: await createPolicyFingerprint(
          message.currentPolicy.onnxPath,
          message.currentPolicy.metadataPath
        ),
        rolloutRoster: await createRolloutRosterFingerprint(message.rolloutRoster)
      });
      return;
    case "run-game":
      if (currentPolicy === null) {
        throw new Error("Worker received run-game before init.");
      }
      try {
        resetWorkerInferenceStats();
        const result = await runPlayingSelfPlayGameWithSamples({
          seed: message.seed,
          currentPolicy,
          behaviorPolicyArtifactId: message.currentPolicyArtifactId,
          rolloutRoster,
          temperature,
          maxDecisionSteps,
          observationVariant
        });
        const resultWithStats = {
          ...result,
          rolloutInferenceStats: collectWorkerInferenceStats()
        };
        send({
          type: "game-complete",
          requestId: message.requestId,
          gameOffset: message.gameOffset,
          seed: message.seed,
          result: resultWithStats
        });
      } catch (error: unknown) {
        sendError(message.requestId, error);
      }
      return;
    case "shutdown":
      process.disconnect?.();
      return;
  }
}

function resetWorkerInferenceStats(): void {
  for (const policy of collectWorkerPolicies()) {
    policy.resetInferenceStats?.();
  }
}

function collectWorkerInferenceStats(): PlayingSelfPlayInferenceStats {
  return combineInferenceStats(
    collectWorkerPolicies().map((policy) => policy.getInferenceStats?.() ?? emptyInferenceStats())
  );
}

function collectWorkerPolicies(): readonly PlayingSelfPlayPolicy[] {
  const policies: PlayingSelfPlayPolicy[] = [];
  if (currentPolicy !== null) {
    policies.push(currentPolicy);
  }
  const seen = new Set<PlayingSelfPlayPolicy>(policies);

  for (const seat of rolloutRoster?.seats ?? []) {
    if (seat.source !== FROZEN_ONNX_ROSTER_SOURCE || seen.has(seat.policy)) {
      continue;
    }
    seen.add(seat.policy);
    policies.push(seat.policy);
  }

  return policies;
}

function emptyInferenceStats(): PlayingSelfPlayInferenceStats {
  return {
    requestCount: 0,
    sessionRunCount: 0,
    meanBatchSize: 0,
    maxObservedBatchSize: 0,
    batchSizeHistogram: {}
  };
}

function combineInferenceStats(
  statsList: readonly PlayingSelfPlayInferenceStats[]
): PlayingSelfPlayInferenceStats {
  const combined = emptyInferenceStats();

  for (const stats of statsList) {
    combined.requestCount += stats.requestCount;
    combined.sessionRunCount += stats.sessionRunCount;
    combined.maxObservedBatchSize = Math.max(combined.maxObservedBatchSize, stats.maxObservedBatchSize);
    for (const [batchSize, count] of Object.entries(stats.batchSizeHistogram)) {
      combined.batchSizeHistogram = {
        ...combined.batchSizeHistogram,
        [batchSize]: (combined.batchSizeHistogram[batchSize] ?? 0) + count
      };
    }
  }

  combined.meanBatchSize =
    combined.sessionRunCount === 0 ? 0 : combined.requestCount / combined.sessionRunCount;
  return combined;
}

async function createPolicyFingerprint(
  onnxPath: string,
  metadataPath: string
): Promise<WorkerPolicyFingerprint> {
  return {
    onnxSha256: await sha256File(onnxPath),
    metadataSha256: await sha256File(metadataPath)
  };
}

async function createRolloutRosterFingerprint(
  seats: readonly WorkerRolloutRosterSeat[] | undefined
): Promise<WorkerRolloutRosterFingerprint | undefined> {
  if (seats === undefined) {
    return undefined;
  }

  return {
    seats: await Promise.all(seats.map(async (seat) => {
      if (seat.source !== FROZEN_ONNX_ROSTER_SOURCE) {
        return null;
      }

      return createPolicyFingerprint(seat.onnxPath, seat.metadataPath);
    }))
  };
}

async function loadWorkerRolloutRoster(
  seats: readonly WorkerRolloutRosterSeat[] | undefined,
  inferenceDevice: "cpu" | "auto" | "cuda",
  inferenceMaxBatchSize: number | undefined
): Promise<PlayingSelfPlayRolloutRosterOptions | undefined> {
  if (seats === undefined) {
    return undefined;
  }

  return {
    seats: await Promise.all(seats.map(async (seat) => {
      switch (seat.source) {
        case CURRENT_POLICY_ROSTER_SOURCE:
          return { source: CURRENT_POLICY_ROSTER_SOURCE };
        case RULE_BASED_ROSTER_SOURCE:
          return { source: RULE_BASED_ROSTER_SOURCE };
        case FROZEN_ONNX_ROSTER_SOURCE:
          return {
            source: FROZEN_ONNX_ROSTER_SOURCE,
            policy: await loadPolicyOnnxModel({
              onnxPath: seat.onnxPath,
              metadataPath: seat.metadataPath,
              inferenceDevice,
              inferenceMaxBatchSize
            }),
            artifact: {
              onnxPath: seat.onnxPath,
              metadataPath: seat.metadataPath,
              artifactId: seat.artifactId
            }
          };
      }
    }))
  };
}

function send(response: PlayingSelfPlayWorkerResponse): void {
  if (!process.send?.(response)) {
    process.exitCode = 1;
  }
}

function sendError(requestId: number | undefined, error: unknown): void {
  const message = error instanceof Error ? error.message : String(error);
  const stack = error instanceof Error ? error.stack : undefined;
  send({
    type: "error",
    requestId,
    message,
    stack
  });
}

async function sha256File(path: string): Promise<string> {
  return createHash("sha256").update(await readFile(path)).digest("hex");
}
