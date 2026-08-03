import { createHash } from "node:crypto";
import { createWriteStream } from "node:fs";
import { once } from "node:events";
import { join } from "node:path";
import type { PlayingTrainingSample } from "@napoleon/ai-observation";
import type { DatasetShardManifest } from "./types.js";
import { serializePlayingTrainingSample } from "./serialization.js";
import { shardFileName } from "./validation.js";

export interface JsonlShardWriter {
  readonly fileName: string;
  readonly sampleCount: number;
  writeSample: (sample: PlayingTrainingSample) => Promise<void>;
  close: (endSeed: number, gameCount: number) => Promise<DatasetShardManifest>;
  abort: () => Promise<void>;
}

export function createJsonlShardWriter(
  directory: string,
  shardIndex: number,
  startSeed: number
): JsonlShardWriter {
  const fileName = shardFileName(shardIndex);
  const stream = createWriteStream(join(directory, fileName), {
    encoding: "utf8",
    flags: "wx"
  });
  const hash = createHash("sha256");
  let byteLength = 0;
  let sampleCount = 0;
  let closed = false;

  async function writeChunk(chunk: string): Promise<void> {
    if (closed) {
      throw new Error(`Cannot write to closed shard: ${fileName}`);
    }

    hash.update(chunk, "utf8");
    byteLength += Buffer.byteLength(chunk, "utf8");
    sampleCount += 1;

    if (!stream.write(chunk)) {
      await once(stream, "drain");
    }
  }

  async function close(endSeed: number, gameCount: number): Promise<DatasetShardManifest> {
    if (closed) {
      throw new Error(`Shard already closed: ${fileName}`);
    }

    closed = true;
    stream.end();
    await once(stream, "finish");

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
    if (closed) {
      return;
    }

    closed = true;
    stream.destroy();
    await once(stream, "close");
  }

  stream.on("error", () => {
    // The pending stream operation will reject through Node's event emitter.
  });

  return {
    fileName,
    get sampleCount() {
      return sampleCount;
    },
    writeSample: (sample) => writeChunk(serializePlayingTrainingSample(sample)),
    close,
    abort
  };
}
