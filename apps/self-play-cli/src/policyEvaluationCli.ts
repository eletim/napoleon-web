import { writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { mkdtemp, rename, rm } from "node:fs/promises";
import {
  loadPolicyOnnxModel,
  runPolicyVsRuleBasedEvaluation,
  runStandardPlayingPolicyBenchmarks
} from "@napoleon/policy-onnx";
import type { StandardPlayingPolicyBenchmarkId } from "@napoleon/policy-onnx";
import {
  optionalValue,
  parseOptionMap,
  parsePositiveInteger,
  parseUnsignedInteger,
  requireValue
} from "./cliArgs.js";

interface ParsedArgs {
  onnx: string;
  metadata: string;
  output: string;
  startSeed: number;
  seedCount: number;
  benchmark: PolicyEvaluationBenchmarkArgument;
  inferenceDevice: "cpu" | "auto" | "cuda";
  progressPrefix: string;
}

type PolicyEvaluationBenchmarkArgument = StandardPlayingPolicyBenchmarkId | "standard";

const optionNames = new Set([
  "--onnx",
  "--metadata",
  "--output",
  "--start-seed",
  "--seed-count",
  "--benchmark",
  "--inference-device",
  "--progress-prefix"
]);

export async function runPolicyEvaluationCli(
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
      metadataPath: args.metadata,
      inferenceDevice: args.inferenceDevice
    });
    if (args.benchmark === "rule-based-x4") {
      const result = await runPolicyVsRuleBasedEvaluation({
        policy,
        startSeed: args.startSeed,
        gameCount: args.seedCount
      });
      await writeJsonAtomic(args.output, result);
      const scheduledGames = result.run.gameCount * result.run.rotationOffsets.length;
      io.stderr.write(`${args.progressPrefix}completed ${result.run.completedCount}/${scheduledGames}\n`);
      io.stdout.write(`${JSON.stringify({
        scheduledGames,
        completedGames: result.run.completedCount,
        failedGames: result.run.failedCount,
        illegalActionCount: result.comparison.illegalActionCount,
        policy: result.comparison.policy,
        ruleBased: result.comparison.ruleBased
      })}\n`);
      return 0;
    }

    const suite = await runStandardPlayingPolicyBenchmarks({
      candidatePolicy: policy,
      benchmarks: args.benchmark === "standard" ? undefined : [args.benchmark],
      startSeed: args.startSeed,
      gameCount: args.seedCount,
      inferenceDevice: args.inferenceDevice
    });
    await writeJsonAtomic(args.output, suite);
    const summaries = suite.benchmarks.map((benchmark) => ({
      benchmarkId: benchmark.benchmarkId,
      scheduledGames: benchmark.result.run.games.length,
      completedGames: benchmark.result.run.completedCount,
      failedGames: benchmark.result.run.failedCount,
      illegalActionCount: benchmark.result.comparison.illegalActionCount
    }));
    io.stderr.write(`${args.progressPrefix}completed ${summaries.reduce(
      (sum, summary) => sum + summary.completedGames,
      0
    )}/${summaries.reduce((sum, summary) => sum + summary.scheduledGames, 0)}\n`);
    io.stdout.write(`${JSON.stringify({ benchmarks: summaries })}\n`);
    return 0;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    io.stderr.write(`${message}\n`);
    return 1;
  }
}

function parseArgs(argv: readonly string[]): ParsedArgs {
  const values = parseOptionMap(argv, optionNames);

  return {
    onnx: requireValue(values, "--onnx"),
    metadata: requireValue(values, "--metadata"),
    output: requireValue(values, "--output"),
    startSeed: parseUnsignedInteger("--start-seed", requireValue(values, "--start-seed")),
    seedCount: parsePositiveInteger("--seed-count", requireValue(values, "--seed-count")),
    benchmark: parseBenchmark(optionalValue(values, "--benchmark") ?? "rule-based-x4"),
    inferenceDevice: parseInferenceDevice(optionalValue(values, "--inference-device") ?? "cpu"),
    progressPrefix: optionalValue(values, "--progress-prefix") ?? ""
  };
}

function parseInferenceDevice(value: string): "cpu" | "auto" | "cuda" {
  if (value === "cpu" || value === "auto" || value === "cuda") {
    return value;
  }
  throw new Error("--inference-device must be one of cpu, auto, cuda.");
}

function parseBenchmark(value: string): PolicyEvaluationBenchmarkArgument {
  if (
    value === "rule-based-x4" ||
    value === "rl-v740-x4" ||
    value === "rule-based-x2-rl-v740-x2" ||
    value === "standard"
  ) {
    return value;
  }

  throw new Error(
    "--benchmark must be one of rule-based-x4, rl-v740-x4, rule-based-x2-rl-v740-x2, standard."
  );
}

async function writeJsonAtomic(path: string, value: unknown): Promise<void> {
  const directory = dirname(path);
  const tempDirectory = await mkdtemp(join(directory, ".policy-evaluation.tmp-"));
  const tempPath = join(tempDirectory, "evaluation.json");

  try {
    await writeFile(tempPath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
    await rename(tempPath, path);
  } catch (error) {
    await rm(tempDirectory, { recursive: true, force: true }).catch(() => undefined);
    throw error;
  }

  await rm(tempDirectory, { recursive: true, force: true });
}

if (import.meta.url === `file://${process.argv[1]}`) {
  raiseSystemExit(runPolicyEvaluationCli(process.argv.slice(2), process));
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
