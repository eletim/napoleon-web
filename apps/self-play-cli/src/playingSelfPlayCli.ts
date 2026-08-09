import { loadPolicyOnnxModel } from "@napoleon/policy-onnx";
import {
  UINT32_MAX,
  generatePlayingSelfPlayDataset
} from "@napoleon/training-data";
import { createProgressReporter } from "./formatProgress.js";
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
  temperature: number;
  artifactId: string | undefined;
  progressPrefix: string;
}

const optionNames = new Set([
  "--onnx",
  "--metadata",
  "--output",
  "--start-seed",
  "--games",
  "--games-per-shard",
  "--temperature",
  "--artifact-id",
  "--progress-prefix"
]);

export async function runPlayingSelfPlayCli(
  argv: readonly string[],
  io: {
    stdout: { write: (chunk: string) => void };
    stderr: { write: (chunk: string) => void };
  }
): Promise<number> {
  try {
    const args = parseArgs(argv);
    const policy = await loadPolicyOnnxModel({
      onnxPath: args.onnx,
      metadataPath: args.metadata
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
      temperature: args.temperature,
      onProgress: createProgressReporter(args.games, (text) =>
        io.stderr.write(`${args.progressPrefix}${text}`)
      )
    });

    io.stdout.write(`${JSON.stringify({
      outputDirectory: result.outputDirectory,
      gameCount: result.manifest.gameCount,
      sampleCount: result.manifest.sampleCount,
      shardCount: result.manifest.shardCount,
      startSeed: result.manifest.startSeed,
      endSeed: result.manifest.endSeed,
      behaviorOnnxSha256: result.manifest.behaviorPolicy.onnxSha256,
      behaviorMetadataSha256: result.manifest.behaviorPolicy.metadataSha256
    })}\n`);
    return 0;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    io.stderr.write(`${message}\n`);
    return 1;
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
    temperature: parsePositiveNumber(
      "--temperature",
      optionalValue(values, "--temperature") ?? "1"
    ),
    artifactId: optionalValue(values, "--artifact-id"),
    progressPrefix: optionalValue(values, "--progress-prefix") ?? ""
  };
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
