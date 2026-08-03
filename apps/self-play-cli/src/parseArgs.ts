import { UINT32_MAX } from "@napoleon/training-data";

export interface ParsedCliArgs {
  startSeed: number;
  games: number;
  output: string;
  gamesPerShard: number;
}

export type ParseArgsResult =
  | {
      type: "help";
      helpText: string;
    }
  | {
      type: "options";
      options: ParsedCliArgs;
    };

const optionNames = new Set([
  "--start-seed",
  "--games",
  "--output",
  "--games-per-shard"
]);
const defaultGamesPerShard = 100;

export function parseArgs(argv: readonly string[]): ParseArgsResult {
  if (argv.length === 1 && argv[0] === "--help") {
    return { type: "help", helpText: createHelpText() };
  }

  if (argv.includes("--help")) {
    throw new Error("--help cannot be combined with generation arguments.");
  }

  const values = new Map<string, string>();

  for (let index = 0; index < argv.length; index += 1) {
    const name = argv[index];

    if (name === undefined) {
      throw new Error("Unexpected argument parser state.");
    }

    if (!name.startsWith("--")) {
      throw new Error(`Unknown argument: ${name}`);
    }

    if (!optionNames.has(name)) {
      throw new Error(`Unknown argument: ${name}`);
    }

    if (values.has(name)) {
      throw new Error(`Duplicate argument: ${name}`);
    }

    const value = argv[index + 1];

    if (value === undefined || value.startsWith("--")) {
      throw new Error(`${name} requires a value.`);
    }

    values.set(name, value);
    index += 1;
  }

  const startSeedText = requireValue(values, "--start-seed");
  const gamesText = requireValue(values, "--games");
  const output = requireValue(values, "--output");
  const gamesPerShardText = values.get("--games-per-shard") ?? String(defaultGamesPerShard);

  if (output.length === 0) {
    throw new Error("--output must be a non-empty path.");
  }

  const startSeed = parseUnsignedInteger("--start-seed", startSeedText);
  const games = parsePositiveInteger("--games", gamesText);
  const gamesPerShard = parsePositiveInteger("--games-per-shard", gamesPerShardText);

  if (startSeed > UINT32_MAX) {
    throw new Error(`--start-seed must be an integer between 0 and ${UINT32_MAX}.`);
  }

  const endSeed = startSeed + games - 1;

  if (!Number.isSafeInteger(endSeed) || endSeed > UINT32_MAX) {
    throw new Error(`Seed range exceeds uint32: ${startSeed}..${endSeed}`);
  }

  return {
    type: "options",
    options: {
      startSeed,
      games,
      output,
      gamesPerShard
    }
  };
}

export function createHelpText(): string {
  return [
    "Usage: pnpm self-play:generate -- --start-seed <uint32> --games <count> --output <path> [--games-per-shard <count>]",
    "",
    "Options:",
    "  --start-seed <uint32>        First seed to generate.",
    "  --games <positive integer>  Number of consecutive games.",
    "  --output <path>             New output dataset directory.",
    "  --games-per-shard <count>   Games per JSONL shard. Default: 100.",
    "  --help                      Show this help text."
  ].join("\n");
}

function requireValue(values: ReadonlyMap<string, string>, name: string): string {
  const value = values.get(name);

  if (value === undefined) {
    throw new Error(`${name} is required.`);
  }

  return value;
}

function parseUnsignedInteger(name: string, value: string): number {
  if (!/^(0|[1-9][0-9]*)$/.test(value)) {
    throw new Error(`${name} must be an integer.`);
  }

  const parsed = Number(value);

  if (!Number.isSafeInteger(parsed)) {
    throw new Error(`${name} must be a safe integer.`);
  }

  return parsed;
}

function parsePositiveInteger(name: string, value: string): number {
  const parsed = parseUnsignedInteger(name, value);

  if (parsed <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }

  return parsed;
}
