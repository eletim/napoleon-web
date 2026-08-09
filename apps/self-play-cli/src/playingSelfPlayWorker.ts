import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { loadPolicyOnnxModel } from "@napoleon/policy-onnx";
import {
  CURRENT_POLICY_ROSTER_SOURCE,
  FROZEN_ONNX_ROSTER_SOURCE,
  RULE_BASED_ROSTER_SOURCE,
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

process.on("message", (message: PlayingSelfPlayWorkerMessage) => {
  void handleMessage(message).catch((error: unknown) => {
    sendError(undefined, error);
  });
});

async function handleMessage(message: PlayingSelfPlayWorkerMessage): Promise<void> {
  switch (message.type) {
    case "init":
      currentPolicy = await loadPolicyOnnxModel(message.currentPolicy);
      rolloutRoster = await loadWorkerRolloutRoster(message.rolloutRoster);
      temperature = message.temperature;
      maxDecisionSteps = message.maxDecisionSteps;
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
        const result = await runPlayingSelfPlayGameWithSamples({
          seed: message.seed,
          currentPolicy,
          behaviorPolicyArtifactId: message.currentPolicyArtifactId,
          rolloutRoster,
          temperature,
          maxDecisionSteps
        });
        send({
          type: "game-complete",
          requestId: message.requestId,
          gameOffset: message.gameOffset,
          seed: message.seed,
          result
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
  seats: readonly WorkerRolloutRosterSeat[] | undefined
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
              metadataPath: seat.metadataPath
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
