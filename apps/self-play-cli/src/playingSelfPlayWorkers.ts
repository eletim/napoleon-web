import { fork, type ChildProcess } from "node:child_process";
import { fileURLToPath } from "node:url";
import type {
  PlayingSelfPlayGameRunner,
  PlayingSelfPlayGameRunRequest,
  PlayingSelfPlayGameRunResult
} from "@napoleon/training-data";
import type {
  PlayingSelfPlayWorkerMessage,
  PlayingSelfPlayWorkerResponse,
  WorkerRolloutRosterFingerprint,
  WorkerRolloutRosterSeat
} from "./playingSelfPlayWorkerProtocol.js";

type GameRunResult = Awaited<ReturnType<PlayingSelfPlayGameRunner["runGame"]>>;

export interface ChildProcessPlayingSelfPlayGameRunnerOptions {
  workerCount: number;
  currentPolicy: {
    onnxPath: string;
    metadataPath: string;
    inferenceDevice: "cpu" | "auto" | "cuda";
    onnxSha256: string;
    metadataSha256: string;
  };
  rolloutRoster: readonly WorkerRolloutRosterSeat[] | undefined;
  temperature: number;
  maxDecisionSteps?: number;
}

interface WorkerSlot {
  child: ChildProcess;
  ready: boolean;
  busy: boolean;
  stderr: string;
}

interface PendingRequest {
  requestId: number;
  request: PlayingSelfPlayGameRunRequest;
  resolve: (result: GameRunResult) => void;
  reject: (error: Error) => void;
}

export class ChildProcessPlayingSelfPlayGameRunner implements PlayingSelfPlayGameRunner {
  private readonly workers: WorkerSlot[];
  private readonly pending: PendingRequest[] = [];
  private readonly activeRequests = new Map<number, PendingRequest>();
  private nextRequestId = 1;
  private closed = false;
  private failed: Error | null = null;

  constructor(private readonly options: ChildProcessPlayingSelfPlayGameRunnerOptions) {
    this.workers = Array.from({ length: options.workerCount }, () => this.spawnWorker());
  }

  runGame(request: PlayingSelfPlayGameRunRequest): Promise<GameRunResult> {
    if (this.closed) {
      return Promise.reject(new Error("Playing self-play worker pool is closed."));
    }
    if (this.failed !== null) {
      return Promise.reject(this.failed);
    }

    return new Promise((resolve, reject) => {
      this.pending.push({
        requestId: this.nextRequestId,
        request,
        resolve,
        reject
      });
      this.nextRequestId += 1;
      this.dispatch();
    });
  }

  async close(): Promise<void> {
    this.closed = true;

    for (const worker of this.workers) {
      if (worker.child.connected) {
        sendToWorker(worker, { type: "shutdown" });
      }
      worker.child.kill();
    }
  }

  private spawnWorker(): WorkerSlot {
    const child = fork(fileURLToPath(new URL("./playingSelfPlayWorker.js", import.meta.url)), [], {
      stdio: ["ignore", "ignore", "pipe", "ipc"],
      serialization: "advanced"
    });
    const worker: WorkerSlot = {
      child,
      ready: false,
      busy: false,
      stderr: ""
    };

    child.stderr?.on("data", (chunk: Buffer) => {
      worker.stderr = `${worker.stderr}${chunk.toString("utf8")}`.slice(-4000);
    });
    child.on("message", (message: PlayingSelfPlayWorkerResponse) => {
      this.handleWorkerMessage(worker, message);
    });
    child.on("error", (error) => {
      this.fail(new Error(`self-play worker process error: ${error.message}`));
    });
    child.on("exit", (code, signal) => {
      if (this.closed) {
        return;
      }
      this.fail(new Error(
        `self-play worker exited unexpectedly: code=${String(code)} signal=${String(signal)} stderr=${worker.stderr}`
      ));
    });

    sendToWorker(worker, {
      type: "init",
      currentPolicy: this.options.currentPolicy,
      rolloutRoster: this.options.rolloutRoster,
      temperature: this.options.temperature,
      maxDecisionSteps: this.options.maxDecisionSteps
    });

    return worker;
  }

  private handleWorkerMessage(
    worker: WorkerSlot,
    message: PlayingSelfPlayWorkerResponse
  ): void {
    switch (message.type) {
      case "ready":
        if (!workerPolicyFingerprintsMatch(message.currentPolicy, this.options.currentPolicy)) {
          this.fail(new Error("self-play worker current policy hash mismatch."));
          return;
        }
        if (!rolloutRosterFingerprintsMatch(message.rolloutRoster, this.options.rolloutRoster)) {
          this.fail(new Error("self-play worker rollout roster policy hash mismatch."));
          return;
        }
        worker.ready = true;
        this.dispatch();
        return;
      case "game-complete": {
        const pending = this.activeRequests.get(message.requestId);
        this.activeRequests.delete(message.requestId);
        worker.busy = false;
        if (pending === undefined) {
          this.fail(new Error(`self-play worker returned unknown requestId ${message.requestId}.`));
          return;
        }
        const result = message.result as PlayingSelfPlayGameRunResult;
        if (
          message.gameOffset !== pending.request.gameOffset ||
          message.seed !== pending.request.seed ||
          result.seed !== pending.request.seed
        ) {
          this.fail(new Error(
            `self-play worker returned mismatched game: ` +
              `offset ${message.gameOffset}/${pending.request.gameOffset}, ` +
              `seed ${message.seed}/${pending.request.seed}.`
          ));
          return;
        }
        pending.resolve(result);
        this.dispatch();
        return;
      }
      case "error": {
        const error = new Error(message.stack ?? message.message);
        if (message.requestId !== undefined) {
          const pending = this.activeRequests.get(message.requestId);
          this.activeRequests.delete(message.requestId);
          worker.busy = false;
          pending?.reject(error);
        }
        this.fail(error);
        return;
      }
    }
  }

  private dispatch(): void {
    if (this.failed !== null) {
      return;
    }

    for (const worker of this.workers) {
      if (!worker.ready || worker.busy || this.pending.length === 0) {
        continue;
      }

      const pending = this.pending.shift();
      if (pending === undefined) {
        return;
      }

      worker.busy = true;
      this.activeRequests.set(pending.requestId, pending);
      try {
        sendToWorker(worker, {
          type: "run-game",
          requestId: pending.requestId,
          gameOffset: pending.request.gameOffset,
          seed: pending.request.seed,
          currentPolicyArtifactId: pending.request.behaviorPolicyArtifactId
        });
      } catch (error: unknown) {
        const message = error instanceof Error ? error.message : String(error);
        this.fail(new Error(message));
      }
    }
  }

  private fail(error: Error): void {
    if (this.failed !== null) {
      return;
    }

    this.failed = error;
    for (const request of this.pending.splice(0)) {
      request.reject(error);
    }
    for (const request of this.activeRequests.values()) {
      request.reject(error);
    }
    this.activeRequests.clear();
    void this.close();
  }
}

function sendToWorker(worker: WorkerSlot, message: PlayingSelfPlayWorkerMessage): void {
  if (!worker.child.connected || !worker.child.send(message)) {
    throw new Error("self-play worker IPC channel is closed.");
  }
}

function workerPolicyFingerprintsMatch(
  actual: { onnxSha256: string; metadataSha256: string },
  expected: { onnxSha256: string; metadataSha256: string }
): boolean {
  return actual.onnxSha256 === expected.onnxSha256 &&
    actual.metadataSha256 === expected.metadataSha256;
}

function rolloutRosterFingerprintsMatch(
  actual: WorkerRolloutRosterFingerprint | undefined,
  expectedSeats: readonly WorkerRolloutRosterSeat[] | undefined
): boolean {
  if (expectedSeats === undefined) {
    return actual === undefined;
  }
  if (actual === undefined || actual.seats.length !== expectedSeats.length) {
    return false;
  }

  return expectedSeats.every((expected, index) => {
    const actualSeat = actual.seats[index];
    if (expected.source !== "frozen-onnx") {
      return actualSeat === null;
    }
    return actualSeat !== null &&
      actualSeat.onnxSha256 === expected.onnxSha256 &&
      actualSeat.metadataSha256 === expected.metadataSha256;
  });
}
