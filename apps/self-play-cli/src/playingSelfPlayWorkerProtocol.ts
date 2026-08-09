export type WorkerRolloutRosterSeat =
  | { source: "current-policy" }
  | { source: "rule-based" }
  | {
      source: "frozen-onnx";
      onnxPath: string;
      metadataPath: string;
      artifactId?: string;
    };

export interface PlayingSelfPlayWorkerInitMessage {
  type: "init";
  currentPolicy: {
    onnxPath: string;
    metadataPath: string;
  };
  rolloutRoster: readonly WorkerRolloutRosterSeat[] | undefined;
  temperature: number;
  maxDecisionSteps: number | undefined;
}

export interface PlayingSelfPlayWorkerRunMessage {
  type: "run-game";
  requestId: number;
  gameOffset: number;
  seed: number;
}

export interface PlayingSelfPlayWorkerShutdownMessage {
  type: "shutdown";
}

export type PlayingSelfPlayWorkerMessage =
  | PlayingSelfPlayWorkerInitMessage
  | PlayingSelfPlayWorkerRunMessage
  | PlayingSelfPlayWorkerShutdownMessage;

export type PlayingSelfPlayWorkerResponse =
  | { type: "ready" }
  | { type: "game-complete"; requestId: number; gameOffset: number; seed: number; record: unknown }
  | { type: "error"; requestId?: number; message: string; stack?: string };
