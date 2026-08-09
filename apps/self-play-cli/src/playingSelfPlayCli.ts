import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { loadPolicyOnnxModel } from "@napoleon/policy-onnx";
import {
  CURRENT_POLICY_ROSTER_SOURCE,
  FROZEN_ONNX_ROSTER_SOURCE,
  RULE_BASED_ROSTER_SOURCE,
  UINT32_MAX,
  generatePlayingSelfPlayDataset,
  type PlayingSelfPlayPolicy,
  type PlayingSelfPlayRolloutRosterOptions,
  type RolloutRosterSeatOptions
} from "@napoleon/training-data";
import { createProgressReporter } from "./formatProgress.js";
import { ChildProcessPlayingSelfPlayGameRunner } from "./playingSelfPlayWorkers.js";
import type { WorkerRolloutRosterSeat } from "./playingSelfPlayWorkerProtocol.js";
import {
  optionalValue,
  parseOptionMap,
  parsePositiveInteger,
  parsePositiveNumber,
  parseUnsignedInteger,
  requireValue
} from "./cliArgs.js";

interface ParsedArgs {
  onnx: string;
  metadata: string;
  output: string;
  startSeed: number;
  games: number;
  gamesPerShard: number;
  rolloutWorkers: number;
  temperature: number;
  artifactId: string | undefined;
  rolloutRoster: string | undefined;
  progressPrefix: string;
}

const optionNames = new Set([
  "--onnx",
  "--metadata",
  "--output",
  "--start-seed",
  "--games",
  "--games-per-shard",
  "--rollout-workers",
  "--temperature",
  "--artifact-id",
  "--rollout-roster",
  "--progress-prefix"
]);

export async function runPlayingSelfPlayCli(
  argv: readonly string[],
  io: {
    stdout: { write: (chunk: string) => void };
    stderr: { write: (chunk: string) => void };
  }
): Promise<number> {
  let gameRunner: ChildProcessPlayingSelfPlayGameRunner | undefined;
  try {
    const args = parseArgs(argv);
    const policy = await loadPolicyOnnxModel({
      onnxPath: args.onnx,
      metadataPath: args.metadata
    });
    const rolloutRoster = await loadRolloutRoster(args.rolloutRoster);
    gameRunner = args.rolloutWorkers === 1
      ? undefined
      : new ChildProcessPlayingSelfPlayGameRunner({
          workerCount: args.rolloutWorkers,
          currentPolicy: {
            onnxPath: args.onnx,
            metadataPath: args.metadata,
            ...await createPolicyFingerprint(args.onnx, args.metadata)
          },
          rolloutRoster: rolloutRoster.workerSeats,
          temperature: args.temperature
        });
    const result = await generatePlayingSelfPlayDataset({
      outputDirectory: args.output,
      playingPolicy: policy,
      playingPolicyArtifact: {
        onnxPath: args.onnx,
        metadataPath: args.metadata,
        artifactId: args.artifactId
      },
      startSeed: args.startSeed,
      gameCount: args.games,
      gamesPerShard: args.gamesPerShard,
      rolloutWorkers: args.rolloutWorkers,
      temperature: args.temperature,
      rolloutRoster: rolloutRoster.options,
      onProgress: createProgressReporter(
        args.games,
        (text) => io.stderr.write(`${args.progressPrefix}${text}`),
        { rolloutWorkers: args.rolloutWorkers }
      ),
      gameRunner
    });

    io.stdout.write(`${JSON.stringify({
      outputDirectory: result.outputDirectory,
      gameCount: result.manifest.gameCount,
      sampleCount: result.manifest.sampleCount,
      shardCount: result.manifest.shardCount,
      startSeed: result.manifest.startSeed,
      endSeed: result.manifest.endSeed,
      rolloutWorkers: args.rolloutWorkers,
      behaviorOnnxSha256: result.manifest.behaviorPolicy.onnxSha256,
      behaviorMetadataSha256: result.manifest.behaviorPolicy.metadataSha256,
      rolloutRoster: result.manifest.rolloutRoster
    })}\n`);
    return 0;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    io.stderr.write(`${message}\n`);
    return 1;
  } finally {
    await Promise.resolve(gameRunner?.close()).catch(() => undefined);
  }
}

function parseArgs(argv: readonly string[]): ParsedArgs {
  const values = parseOptionMap(argv, optionNames);
  const startSeed = parseUnsignedInteger("--start-seed", requireValue(values, "--start-seed"));
  const games = parsePositiveInteger("--games", requireValue(values, "--games"));
  const endSeed = startSeed + games - 1;

  if (!Number.isSafeInteger(endSeed) || endSeed > UINT32_MAX) {
    throw new Error(`Seed range exceeds uint32: ${startSeed}..${endSeed}`);
  }

  return {
    onnx: requireValue(values, "--onnx"),
    metadata: requireValue(values, "--metadata"),
    output: requireValue(values, "--output"),
    startSeed,
    games,
    gamesPerShard: parsePositiveInteger(
      "--games-per-shard",
      requireValue(values, "--games-per-shard")
    ),
    rolloutWorkers: parsePositiveInteger(
      "--rollout-workers",
      optionalValue(values, "--rollout-workers") ?? "1"
    ),
    temperature: parsePositiveNumber(
      "--temperature",
      optionalValue(values, "--temperature") ?? "1"
    ),
    artifactId: optionalValue(values, "--artifact-id"),
    rolloutRoster: optionalValue(values, "--rollout-roster"),
    progressPrefix: optionalValue(values, "--progress-prefix") ?? ""
  };
}

interface LoadedRolloutRoster {
  options: PlayingSelfPlayRolloutRosterOptions | undefined;
  workerSeats: readonly WorkerRolloutRosterSeat[] | undefined;
}

async function loadRolloutRoster(
  value: string | undefined
): Promise<LoadedRolloutRoster> {
  if (value === undefined) {
    return {
      options: undefined,
      workerSeats: undefined
    };
  }

  const rawSeats = parseRolloutRosterValue(value);
  const seats = await Promise.all(rawSeats.map(loadRolloutRosterSeat));

  return {
    options: { seats: seats.map((seat) => seat.option) },
    workerSeats: seats.map((seat) => seat.workerSeat)
  };
}

function parseRolloutRosterValue(value: string): readonly unknown[] {
  const trimmed = value.trim();

  if (trimmed.length === 0) {
    throw new Error("--rollout-roster must be non-empty.");
  }

  if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
    const parsed = JSON.parse(trimmed) as unknown;

    if (Array.isArray(parsed)) {
      return parsed;
    }

    if (isRecord(parsed) && Array.isArray(parsed.seats)) {
      return parsed.seats;
    }

    throw new Error("--rollout-roster JSON must be an array or an object with seats.");
  }

  return trimmed.split(",").map((entry) => entry.trim());
}

async function loadRolloutRosterSeat(raw: unknown): Promise<{
  option: RolloutRosterSeatOptions;
  workerSeat: WorkerRolloutRosterSeat;
}> {
  if (raw === CURRENT_POLICY_ROSTER_SOURCE) {
    return {
      option: { source: CURRENT_POLICY_ROSTER_SOURCE },
      workerSeat: { source: CURRENT_POLICY_ROSTER_SOURCE }
    };
  }
  if (raw === RULE_BASED_ROSTER_SOURCE) {
    return {
      option: { source: RULE_BASED_ROSTER_SOURCE },
      workerSeat: { source: RULE_BASED_ROSTER_SOURCE }
    };
  }
  if (!isRecord(raw)) {
    throw new Error(`Invalid rollout roster seat: ${String(raw)}`);
  }

  const source = raw.source;
  if (source === CURRENT_POLICY_ROSTER_SOURCE) {
    return {
      option: { source: CURRENT_POLICY_ROSTER_SOURCE },
      workerSeat: { source: CURRENT_POLICY_ROSTER_SOURCE }
    };
  }
  if (source === RULE_BASED_ROSTER_SOURCE) {
    return {
      option: { source: RULE_BASED_ROSTER_SOURCE },
      workerSeat: { source: RULE_BASED_ROSTER_SOURCE }
    };
  }
  if (source !== FROZEN_ONNX_ROSTER_SOURCE) {
    throw new Error(`Invalid rollout roster seat source: ${String(source)}`);
  }

  const onnxPath = requiredString(raw.onnxPath, "rolloutRoster.seats[].onnxPath");
  const metadataPath = requiredString(raw.metadataPath, "rolloutRoster.seats[].metadataPath");
  const policy: PlayingSelfPlayPolicy = await loadPolicyOnnxModel({ onnxPath, metadataPath });
  const artifactId = typeof raw.artifactId === "string" ? raw.artifactId : undefined;
  const fingerprint = await createPolicyFingerprint(onnxPath, metadataPath);

  return {
    option: {
      source: FROZEN_ONNX_ROSTER_SOURCE,
      policy,
      artifact: {
        onnxPath,
        metadataPath,
        artifactId
      }
    },
    workerSeat: {
      source: FROZEN_ONNX_ROSTER_SOURCE,
      onnxPath,
      metadataPath,
      ...fingerprint,
      artifactId
    }
  };
}

async function createPolicyFingerprint(
  onnxPath: string,
  metadataPath: string
): Promise<{ onnxSha256: string; metadataSha256: string }> {
  return {
    onnxSha256: await sha256File(onnxPath),
    metadataSha256: await sha256File(metadataPath)
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requiredString(value: unknown, name: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${name} must be a non-empty string.`);
  }

  return value;
}

async function sha256File(path: string): Promise<string> {
  return createHash("sha256").update(await readFile(path)).digest("hex");
}

if (import.meta.url === `file://${process.argv[1]}`) {
  raiseSystemExit(runPlayingSelfPlayCli(process.argv.slice(2), process));
}

function raiseSystemExit(result: Promise<number>): void {
  result.then(
    (code) => {
      process.exitCode = code;
    },
    (error: unknown) => {
      const message = error instanceof Error ? error.stack ?? error.message : String(error);
      process.stderr.write(`${message}\n`);
      process.exitCode = 1;
    }
  );
}
