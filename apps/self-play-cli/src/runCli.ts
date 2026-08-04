import { generateRuleBasedDataset } from "@napoleon/training-data";
import { createProgressReporter } from "./formatProgress.js";
import { parseArgs } from "./parseArgs.js";

export interface CliIo {
  stdout: {
    write: (chunk: string) => void;
  };
  stderr: {
    write: (chunk: string) => void;
  };
}

export async function runCli(argv: readonly string[], io: CliIo): Promise<number> {
  try {
    const parsed = parseArgs(argv);

    if (parsed.type === "help") {
      io.stdout.write(`${parsed.helpText}\n`);
      return 0;
    }

    const result = await generateRuleBasedDataset({
      startSeed: parsed.options.startSeed,
      gameCount: parsed.options.games,
      gamesPerShard: parsed.options.gamesPerShard,
      outputDirectory: parsed.options.output,
      onProgress: createProgressReporter(parsed.options.games, (text) => io.stderr.write(text))
    });

    io.stdout.write([
      "Generated self-play dataset",
      `  games: ${result.manifest.gameCount}`,
      `  samples: ${result.manifest.sampleCount}`,
      `  shards: ${result.manifest.shardCount}`,
      `  seeds: ${result.manifest.startSeed}..${result.manifest.endSeed}`,
      `  output: ${parsed.options.output}`,
      ""
    ].join("\n"));
    return 0;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    io.stderr.write(`${message}\n`);
    return 1;
  }
}
