import { writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { mkdtemp, rename, rm } from "node:fs/promises";
import {
  loadPolicyOnnxModel,
  runPolicyVsRuleBasedEvaluation,
  runStandardPlayingPolicyBenchmarks
} from "@napoleon/policy-onnx";
import type {
  PolicyOnnxModel,
  StandardPlayingPolicyBenchmarkId,
  StandardPlayingPolicyBenchmarkSuiteResult,
  PolicyVsRuleBasedEvaluationResult
} from "@napoleon/policy-onnx";
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
  policyLabel: string;
  baselineOnnx?: string;
  baselineMetadata?: string;
  baselineLabel: string;
  output: string;
  startSeed: number;
  seedCount: number;
  benchmark: PolicyEvaluationBenchmarkArgument;
  inferenceDevice: "cpu" | "auto" | "cuda";
  progressPrefix: string;
}

type PolicyEvaluationBenchmarkArgument = StandardPlayingPolicyBenchmarkId | "standard";
type SinglePolicyEvaluationResult =
  | PolicyVsRuleBasedEvaluationResult
  | StandardPlayingPolicyBenchmarkSuiteResult;

interface PolicyEvaluationArtifactSummary {
  label: string;
  onnxPath: string;
  metadataPath: string;
  playingObservationVariant: "public" | "complete-info-compact";
  modelInputFeatureCount: number | null;
}

interface PolicyEvaluationComparisonEntry {
  artifact: PolicyEvaluationArtifactSummary;
  runtime: PolicyOnnxModel["runtime"];
  result: SinglePolicyEvaluationResult;
}

interface PolicyEvaluationComparisonResult {
  schemaVersion: 1;
  evaluationType: "playing-policy-comparison";
  benchmark: PolicyEvaluationBenchmarkArgument;
  startSeed: number;
  endSeed: number;
  seedCount: number;
  inferenceDevice: "cpu" | "auto" | "cuda";
  policies: readonly PolicyEvaluationComparisonEntry[];
}

const optionNames = new Set([
  "--onnx",
  "--metadata",
  "--policy-label",
  "--baseline-onnx",
  "--baseline-metadata",
  "--baseline-label",
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

    if (args.baselineOnnx !== undefined && args.baselineMetadata !== undefined) {
      const policies = [
        await runNamedPolicyEvaluation({
          label: args.policyLabel,
          onnxPath: args.onnx,
          metadataPath: args.metadata,
          args
        }),
        await runNamedPolicyEvaluation({
          label: args.baselineLabel,
          onnxPath: args.baselineOnnx,
          metadataPath: args.baselineMetadata,
          args
        })
      ];
      const result: PolicyEvaluationComparisonResult = {
        schemaVersion: 1,
        evaluationType: "playing-policy-comparison",
        benchmark: args.benchmark,
        startSeed: args.startSeed,
        endSeed: args.startSeed + args.seedCount - 1,
        seedCount: args.seedCount,
        inferenceDevice: args.inferenceDevice,
        policies
      };

      await writeJsonAtomic(args.output, result);
      const summaries = policies.map((entry) => ({
        label: entry.artifact.label,
        variant: entry.artifact.playingObservationVariant,
        modelInputFeatureCount: entry.artifact.modelInputFeatureCount,
        runtime: entry.runtime,
        ...summarizeEvaluationResult(entry.result)
      }));
      const totals = summaries.reduce(
        (sum, summary) => ({
          completedGames: sum.completedGames + summary.completedGames,
          scheduledGames: sum.scheduledGames + summary.scheduledGames
        }),
        { completedGames: 0, scheduledGames: 0 }
      );
      io.stderr.write(`${args.progressPrefix}completed ${totals.completedGames}/${totals.scheduledGames}\n`);
      io.stdout.write(`${JSON.stringify({ policies: summaries })}\n`);
      return 0;
    }

    const result = await runPolicyEvaluation({
      policy: await loadPolicyOnnxModel({
        onnxPath: args.onnx,
        metadataPath: args.metadata,
        inferenceDevice: args.inferenceDevice
      }),
      args
    });
    await writeJsonAtomic(args.output, result);
    writeSingleEvaluationSummary(result, args.progressPrefix, io);
    return 0;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    io.stderr.write(`${message}\n`);
    return 1;
  }
}

function parseArgs(argv: readonly string[]): ParsedArgs {
  const values = parseOptionMap(argv, optionNames);

  const parsed = {
    onnx: requireValue(values, "--onnx"),
    metadata: requireValue(values, "--metadata"),
    policyLabel: optionalValue(values, "--policy-label") ?? "candidate",
    baselineOnnx: optionalValue(values, "--baseline-onnx"),
    baselineMetadata: optionalValue(values, "--baseline-metadata"),
    baselineLabel: optionalValue(values, "--baseline-label") ?? "baseline",
    output: requireValue(values, "--output"),
    startSeed: parseUnsignedInteger("--start-seed", requireValue(values, "--start-seed")),
    seedCount: parsePositiveInteger("--seed-count", requireValue(values, "--seed-count")),
    benchmark: parseBenchmark(optionalValue(values, "--benchmark") ?? "rule-based-x4"),
    inferenceDevice: parseInferenceDevice(optionalValue(values, "--inference-device") ?? "cpu"),
    progressPrefix: optionalValue(values, "--progress-prefix") ?? ""
  };

  if ((parsed.baselineOnnx === undefined) !== (parsed.baselineMetadata === undefined)) {
    throw new Error("--baseline-onnx and --baseline-metadata must be provided together.");
  }

  return parsed;
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

async function runNamedPolicyEvaluation(args: {
  label: string;
  onnxPath: string;
  metadataPath: string;
  args: ParsedArgs;
}): Promise<PolicyEvaluationComparisonEntry> {
  const policy = await loadPolicyOnnxModel({
    onnxPath: args.onnxPath,
    metadataPath: args.metadataPath,
    inferenceDevice: args.args.inferenceDevice
  });

  return {
    artifact: {
      label: args.label,
      onnxPath: args.onnxPath,
      metadataPath: args.metadataPath,
      playingObservationVariant: policy.metadata.playingObservationVariant ?? "public",
      modelInputFeatureCount: getModelInputFeatureCount(policy)
    },
    runtime: policy.runtime,
    result: await runPolicyEvaluation({
      policy,
      args: args.args
    })
  };
}

async function runPolicyEvaluation(args: {
  policy: PolicyOnnxModel;
  args: ParsedArgs;
}): Promise<SinglePolicyEvaluationResult> {
  if (args.args.benchmark === "rule-based-x4") {
    return runPolicyVsRuleBasedEvaluation({
      policy: args.policy,
      startSeed: args.args.startSeed,
      gameCount: args.args.seedCount
    });
  }

  return runStandardPlayingPolicyBenchmarks({
    candidatePolicy: args.policy,
    benchmarks: args.args.benchmark === "standard" ? undefined : [args.args.benchmark],
    startSeed: args.args.startSeed,
    gameCount: args.args.seedCount,
    inferenceDevice: args.args.inferenceDevice
  });
}

function writeSingleEvaluationSummary(
  result: SinglePolicyEvaluationResult,
  progressPrefix: string,
  io: {
    stdout: { write: (chunk: string) => void };
    stderr: { write: (chunk: string) => void };
  }
): void {
  if (isPolicyVsRuleBasedEvaluationResult(result)) {
    const scheduledGames = result.run.gameCount * result.run.rotationOffsets.length;
    io.stderr.write(`${progressPrefix}completed ${result.run.completedCount}/${scheduledGames}\n`);
    io.stdout.write(`${JSON.stringify({
      scheduledGames,
      completedGames: result.run.completedCount,
      failedGames: result.run.failedCount,
      illegalActionCount: result.comparison.illegalActionCount,
      policy: result.comparison.policy,
      ruleBased: result.comparison.ruleBased
    })}\n`);
    return;
  }

  const summaries = result.benchmarks.map((benchmark) => ({
    benchmarkId: benchmark.benchmarkId,
    scheduledGames: benchmark.result.run.games.length,
    completedGames: benchmark.result.run.completedCount,
    failedGames: benchmark.result.run.failedCount,
    illegalActionCount: benchmark.result.comparison.illegalActionCount
  }));
  io.stderr.write(`${progressPrefix}completed ${summaries.reduce(
    (sum, summary) => sum + summary.completedGames,
    0
  )}/${summaries.reduce((sum, summary) => sum + summary.scheduledGames, 0)}\n`);
  io.stdout.write(`${JSON.stringify({ benchmarks: summaries })}\n`);
}

function summarizeEvaluationResult(result: SinglePolicyEvaluationResult): {
  scheduledGames: number;
  completedGames: number;
  failedGames: number;
  benchmarks?: readonly {
    benchmarkId: StandardPlayingPolicyBenchmarkId;
    scheduledGames: number;
    completedGames: number;
    failedGames: number;
    illegalActionCount: number;
  }[];
  illegalActionCount?: number;
} {
  if (isPolicyVsRuleBasedEvaluationResult(result)) {
    return {
      scheduledGames: result.run.gameCount * result.run.rotationOffsets.length,
      completedGames: result.run.completedCount,
      failedGames: result.run.failedCount,
      illegalActionCount: result.comparison.illegalActionCount
    };
  }

  const benchmarks = result.benchmarks.map((benchmark) => ({
    benchmarkId: benchmark.benchmarkId,
    scheduledGames: benchmark.result.run.games.length,
    completedGames: benchmark.result.run.completedCount,
    failedGames: benchmark.result.run.failedCount,
    illegalActionCount: benchmark.result.comparison.illegalActionCount
  }));

  return {
    scheduledGames: benchmarks.reduce((sum, summary) => sum + summary.scheduledGames, 0),
    completedGames: benchmarks.reduce((sum, summary) => sum + summary.completedGames, 0),
    failedGames: benchmarks.reduce((sum, summary) => sum + summary.failedGames, 0),
    benchmarks
  };
}

function isPolicyVsRuleBasedEvaluationResult(
  result: SinglePolicyEvaluationResult
): result is PolicyVsRuleBasedEvaluationResult {
  return "comparison" in result;
}

function getModelInputFeatureCount(policy: PolicyOnnxModel): number | null {
  const metadataFeatureCount = policy.metadata.modelInputFeatureCount;
  if (metadataFeatureCount !== undefined) {
    return metadataFeatureCount;
  }

  const inputFeatureCount = policy.metadata.onnx.inputs[0]?.shape[1];
  return typeof inputFeatureCount === "number" ? inputFeatureCount : null;
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
