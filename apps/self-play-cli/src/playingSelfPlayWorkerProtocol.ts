export type WorkerRolloutRosterSeat =
  | { source: "current-policy" }
  | { source: "rule-based" }
  | {
      source: "frozen-onnx";
      onnxPath: string;
      metadataPath: string;
      onnxSha256: string;
      metadataSha256: string;
      artifactId?: string;
    };

export type WorkerInferenceDevice = "cpu" | "auto" | "cuda";

export interface PlayingSelfPlayWorkerInitMessage {
  type: "init";
  currentPolicy: {
    onnxPath: string;
    metadataPath: string;
    inferenceDevice: WorkerInferenceDevice;
    inferenceMaxBatchSize?: number;
  };
  rolloutRoster: readonly WorkerRolloutRosterSeat[] | undefined;
  temperature: number;
  maxDecisionSteps: number | undefined;
}

export interface WorkerPolicyFingerprint {
  onnxSha256: string;
  metadataSha256: string;
}

export interface WorkerRolloutRosterFingerprint {
  seats: readonly (WorkerPolicyFingerprint | null)[];
}

export interface PlayingSelfPlayWorkerRunMessage {
  type: "run-game";
  requestId: number;
  gameOffset: number;
  seed: number;
  currentPolicyArtifactId: string;
}

export interface PlayingSelfPlayWorkerShutdownMessage {
  type: "shutdown";
}

export type PlayingSelfPlayWorkerMessage =
  | PlayingSelfPlayWorkerInitMessage
  | PlayingSelfPlayWorkerRunMessage
  | PlayingSelfPlayWorkerShutdownMessage;

export type PlayingSelfPlayWorkerResponse =
  | {
      type: "ready";
      currentPolicy: WorkerPolicyFingerprint;
      rolloutRoster: WorkerRolloutRosterFingerprint | undefined;
    }
  | { type: "game-complete"; requestId: number; gameOffset: number; seed: number; result: unknown }
  | { type: "error"; requestId?: number; message: string; stack?: string };
