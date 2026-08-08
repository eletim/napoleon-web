import { createHash } from "node:crypto";
import { createWriteStream } from "node:fs";
import { join } from "node:path";
import { finished } from "node:stream/promises";
import type { Writable } from "node:stream";
import type { DatasetShardManifest, SampleSerializer, TrainingSample } from "./types.js";
import { serializeTrainingSample } from "./serialization.js";
import { shardFileName } from "./validation.js";

export interface JsonlShardWriter<TSample = TrainingSample> {
  readonly fileName: string;
  readonly sampleCount: number;
  writeSample: (sample: TSample) => Promise<void>;
  close: (endSeed: number, gameCount: number) => Promise<DatasetShardManifest>;
  abort: () => Promise<void>;
}

export type ShardWriterState = "open" | "closing" | "closed" | "aborted" | "failed";
export type WritableFactory = (path: string) => Writable;

export function createJsonlShardWriter<TSample = TrainingSample>(
  directory: string,
  shardIndex: number,
  startSeed: number,
  serializeSample: SampleSerializer<TSample> = serializeTrainingSample as SampleSerializer<TSample>,
  writableFactory: WritableFactory = (path) => createWriteStream(path, {
    encoding: "utf8",
    flags: "wx"
  })
): JsonlShardWriter<TSample> {
  const fileName = shardFileName(shardIndex);
  const stream = writableFactory(join(directory, fileName));
  const hash = createHash("sha256");
  let byteLength = 0;
  let sampleCount = 0;
  let state: ShardWriterState = "open";
  let streamFailure: unknown;
  const completion = finished(stream).catch((error: unknown) => {
    streamFailure ??= error;
    if (state !== "aborted") {
      state = "failed";
    }
  });

  async function writeChunk(chunk: string): Promise<void> {
    assertOpenForWrite();

    let accepted: boolean;

    try {
      accepted = stream.write(chunk);
    } catch (error) {
      streamFailure ??= error;
      state = "failed";
      throw error;
    }

    hash.update(chunk, "utf8");
    byteLength += Buffer.byteLength(chunk, "utf8");
    sampleCount += 1;

    if (!accepted) {
      await waitForDrain();
      throwIfStreamFailed();
    }
  }

  async function close(endSeed: number, gameCount: number): Promise<DatasetShardManifest> {
    if (state !== "open") {
      throw new Error(`Cannot close shard ${fileName} while ${state}.`);
    }

    throwIfStreamFailed();
    state = "closing";

    try {
      stream.end();
    } catch (error) {
      streamFailure ??= error;
      state = "failed";
      throw error;
    }

    await completion;
    throwIfStreamFailed();
    state = "closed";

    return {
      file: fileName,
      startSeed,
      endSeed,
      gameCount,
      sampleCount,
      byteLength,
      sha256: hash.digest("hex")
    };
  }

  async function abort(): Promise<void> {
    if (state === "closed" || state === "aborted") {
      return;
    }

    state = "aborted";
    stream.destroy();
    await completion;
  }

  return {
    fileName,
    get sampleCount() {
      return sampleCount;
    },
    writeSample: (sample) => writeChunk(serializeSample(sample)),
    close,
    abort
  };

  function assertOpenForWrite(): void {
    throwIfStreamFailed();

    if (state !== "open") {
      throw new Error(`Cannot write to shard ${fileName} while ${state}.`);
    }
  }

  function throwIfStreamFailed(): void {
    if (streamFailure !== undefined) {
      throw streamFailure;
    }
  }

  async function waitForDrain(): Promise<void> {
    let drained = false;
    let cleanup = () => undefined;

    const drainOrError = new Promise<void>((resolve, reject) => {
      const onDrain = () => {
        drained = true;
        cleanup();
        resolve();
      };
      const onError = (error: Error) => {
        cleanup();
        streamFailure ??= error;
        state = "failed";
        reject(error);
      };

      cleanup = () => {
        stream.off("drain", onDrain);
        stream.off("error", onError);
      };

      stream.once("drain", onDrain);
      stream.once("error", onError);
    });

    try {
      // A stream can be destroyed without emitting "error" (e.g. a bare
      // `stream.destroy()` during backpressure), which never fires "drain"
      // either. Racing against overall stream completion ensures we stop
      // waiting once the stream is no longer writable, instead of hanging
      // forever on events that will never come.
      await Promise.race([drainOrError, completion]);
    } finally {
      cleanup();
    }

    if (drained) {
      throwIfStreamFailed();
      return;
    }

    // The stream completed (successfully or not) before "drain" ever fired.
    // Wrap whatever caused that in a stable, stream-writer-owned error so
    // callers never depend on Node's internal "Premature close" wording.
    const error = new Error(`Shard stream closed before drain: ${fileName}`, {
      cause: streamFailure
    });

    streamFailure = error;
    state = "failed";
    throw error;
  }
}
