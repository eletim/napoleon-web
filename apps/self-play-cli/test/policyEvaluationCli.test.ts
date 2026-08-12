import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { beforeEach, describe, expect, it, vi } from "vitest";

const loadPolicyOnnxModel = vi.hoisted(() => vi.fn());
const runPolicyVsRuleBasedEvaluation = vi.hoisted(() => vi.fn());
const runStandardPlayingPolicyBenchmarks = vi.hoisted(() => vi.fn());

vi.mock("@napoleon/policy-onnx", () => ({
  loadPolicyOnnxModel,
  runPolicyVsRuleBasedEvaluation,
  runStandardPlayingPolicyBenchmarks
}));

const { runPolicyEvaluationCli } = await import("../src/policyEvaluationCli.js");

function createIo() {
  return {
    stdout: { write: vi.fn() },
    stderr: { write: vi.fn() }
  };
}

describe("runPolicyEvaluationCli", () => {
  beforeEach(() => {
    loadPolicyOnnxModel.mockReset();
    runPolicyVsRuleBasedEvaluation.mockReset();
    runStandardPlayingPolicyBenchmarks.mockReset();
  });

  it("preserves the legacy RuleBased x4 output by default", async () => {
    const directory = await mkdtemp(join(tmpdir(), "policy-eval-cli-"));
    const output = join(directory, "evaluation.json");
    const io = createIo();
    const policy = {};
    const result = {
      schemaVersion: 1,
      run: {
        gameCount: 1,
        rotationOffsets: [0, 1, 2, 3, 4],
        completedCount: 5,
        failedCount: 0
      },
      comparison: {
        illegalActionCount: 0,
        policy: { agentName: "PolicyOnnxAgent" },
        ruleBased: { agentName: "RuleBasedAgent" }
      }
    };
    loadPolicyOnnxModel.mockResolvedValueOnce(policy);
    runPolicyVsRuleBasedEvaluation.mockResolvedValueOnce(result);

    const code = await runPolicyEvaluationCli([
      "--onnx",
      "/models/candidate.onnx",
      "--metadata",
      "/models/candidate.json",
      "--output",
      output,
      "--start-seed",
      "10",
      "--seed-count",
      "1"
    ], io);

    expect(code).toBe(0);
    expect(runPolicyVsRuleBasedEvaluation).toHaveBeenCalledWith({
      policy,
      startSeed: 10,
      gameCount: 1
    });
    expect(loadPolicyOnnxModel).toHaveBeenCalledWith({
      onnxPath: "/models/candidate.onnx",
      metadataPath: "/models/candidate.json",
      inferenceDevice: "cpu"
    });
    expect(runStandardPlayingPolicyBenchmarks).not.toHaveBeenCalled();
    expect(JSON.parse(await readFile(output, "utf8"))).toEqual(result);
    expect(JSON.parse(io.stdout.write.mock.calls[0][0])).toMatchObject({
      scheduledGames: 5,
      completedGames: 5,
      failedGames: 0,
      illegalActionCount: 0
    });

    await rm(directory, { recursive: true, force: true });
  });

  it("runs the standard benchmark suite when requested", async () => {
    const directory = await mkdtemp(join(tmpdir(), "policy-eval-cli-"));
    const output = join(directory, "evaluation.json");
    const io = createIo();
    const policy = {};
    const suite = {
      schemaVersion: 1,
      candidateMetadata: {},
      benchmarks: [
        {
          benchmarkId: "rule-based-x4",
          result: {
            run: {
              games: [{ gameIndex: 0 }],
              completedCount: 1,
              failedCount: 0
            },
            comparison: {
              illegalActionCount: 0
            }
          }
        }
      ]
    };
    loadPolicyOnnxModel.mockResolvedValueOnce(policy);
    runStandardPlayingPolicyBenchmarks.mockResolvedValueOnce(suite);

    const code = await runPolicyEvaluationCli([
      "--onnx",
      "/models/candidate.onnx",
      "--metadata",
      "/models/candidate.json",
      "--output",
      output,
      "--start-seed",
      "10",
      "--seed-count",
      "1",
      "--benchmark",
      "standard"
    ], io);

    expect(code).toBe(0);
    expect(runStandardPlayingPolicyBenchmarks).toHaveBeenCalledWith({
      candidatePolicy: policy,
      benchmarks: undefined,
      startSeed: 10,
      gameCount: 1,
      inferenceDevice: "cpu"
    });
    expect(JSON.parse(await readFile(output, "utf8"))).toEqual(suite);
    expect(JSON.parse(io.stdout.write.mock.calls[0][0])).toEqual({
      benchmarks: [
        {
          benchmarkId: "rule-based-x4",
          scheduledGames: 1,
          completedGames: 1,
          failedGames: 0,
          illegalActionCount: 0
        }
      ]
    });

    await rm(directory, { recursive: true, force: true });
  });

  it("runs a single named standard benchmark when requested", async () => {
    const directory = await mkdtemp(join(tmpdir(), "policy-eval-cli-"));
    const output = join(directory, "evaluation.json");
    const io = createIo();
    const policy = {};
    const suite = {
      schemaVersion: 1,
      candidateMetadata: {},
      benchmarks: [
        {
          benchmarkId: "rl-v740-x4",
          result: {
            run: {
              games: [{ gameIndex: 0 }],
              completedCount: 1,
              failedCount: 0
            },
            comparison: {
              illegalActionCount: 0
            }
          }
        }
      ]
    };
    loadPolicyOnnxModel.mockResolvedValueOnce(policy);
    runStandardPlayingPolicyBenchmarks.mockResolvedValueOnce(suite);

    const code = await runPolicyEvaluationCli([
      "--onnx",
      "/models/candidate.onnx",
      "--metadata",
      "/models/candidate.json",
      "--output",
      output,
      "--start-seed",
      "10",
      "--seed-count",
      "1",
      "--benchmark",
      "rl-v740-x4"
    ], io);

    expect(code).toBe(0);
    expect(runStandardPlayingPolicyBenchmarks).toHaveBeenCalledWith({
      candidatePolicy: policy,
      benchmarks: ["rl-v740-x4"],
      startSeed: 10,
      gameCount: 1,
      inferenceDevice: "cpu"
    });
    expect(JSON.parse(await readFile(output, "utf8"))).toEqual(suite);

    await rm(directory, { recursive: true, force: true });
  });

  it("rejects unknown benchmark names before loading a policy", async () => {
    const io = createIo();
    const code = await runPolicyEvaluationCli([
      "--onnx",
      "/models/candidate.onnx",
      "--metadata",
      "/models/candidate.json",
      "--output",
      "/tmp/evaluation.json",
      "--start-seed",
      "10",
      "--seed-count",
      "1",
      "--benchmark",
      "missing"
    ], io);

    expect(code).toBe(1);
    expect(io.stderr.write.mock.calls.join("\n")).toContain(
      "--benchmark must be one of rule-based-x4"
    );
    expect(loadPolicyOnnxModel).not.toHaveBeenCalled();
  });

  it("passes explicit inference device to policy loading and standard benchmarks", async () => {
    const directory = await mkdtemp(join(tmpdir(), "policy-eval-cli-"));
    const output = join(directory, "evaluation.json");
    const io = createIo();
    const policy = {};
    const suite = {
      schemaVersion: 1,
      candidateMetadata: {},
      benchmarks: []
    };
    loadPolicyOnnxModel.mockResolvedValueOnce(policy);
    runStandardPlayingPolicyBenchmarks.mockResolvedValueOnce(suite);

    const code = await runPolicyEvaluationCli([
      "--onnx",
      "/models/candidate.onnx",
      "--metadata",
      "/models/candidate.json",
      "--output",
      output,
      "--start-seed",
      "10",
      "--seed-count",
      "1",
      "--benchmark",
      "standard",
      "--inference-device",
      "cuda"
    ], io);

    expect(code).toBe(0);
    expect(loadPolicyOnnxModel).toHaveBeenCalledWith({
      onnxPath: "/models/candidate.onnx",
      metadataPath: "/models/candidate.json",
      inferenceDevice: "cuda"
    });
    expect(runStandardPlayingPolicyBenchmarks).toHaveBeenCalledWith(expect.objectContaining({
      inferenceDevice: "cuda"
    }));

    await rm(directory, { recursive: true, force: true });
  });

  it("evaluates candidate and baseline policies with the same RuleBased conditions", async () => {
    const directory = await mkdtemp(join(tmpdir(), "policy-eval-cli-"));
    const output = join(directory, "evaluation.json");
    const io = createIo();
    const completeInfoPolicy = {
      metadata: {
        playingObservationVariant: "complete-info-compact",
        modelInputFeatureCount: 385
      },
      runtime: {
        requestedInferenceDevice: "cpu",
        resolvedInferenceDevice: "cpu",
        executionProvider: "cpu"
      }
    };
    const publicPolicy = {
      metadata: {
        playingObservationVariant: "public",
        onnx: {
          inputs: [
            {
              shape: ["batch", 6246]
            }
          ]
        }
      },
      runtime: {
        requestedInferenceDevice: "cpu",
        resolvedInferenceDevice: "cpu",
        executionProvider: "cpu"
      }
    };
    const completeInfoResult = {
      schemaVersion: 1,
      run: {
        gameCount: 2,
        rotationOffsets: [0, 1, 2, 3, 4],
        completedCount: 10,
        failedCount: 0
      },
      comparison: {
        illegalActionCount: 0
      }
    };
    const publicResult = {
      schemaVersion: 1,
      run: {
        gameCount: 2,
        rotationOffsets: [0, 1, 2, 3, 4],
        completedCount: 9,
        failedCount: 1
      },
      comparison: {
        illegalActionCount: 1
      }
    };
    loadPolicyOnnxModel
      .mockResolvedValueOnce(completeInfoPolicy)
      .mockResolvedValueOnce(publicPolicy);
    runPolicyVsRuleBasedEvaluation
      .mockResolvedValueOnce(completeInfoResult)
      .mockResolvedValueOnce(publicResult);

    const code = await runPolicyEvaluationCli([
      "--onnx",
      "/models/compact.onnx",
      "--metadata",
      "/models/compact.json",
      "--policy-label",
      "complete-info-compact",
      "--baseline-onnx",
      "/models/public.onnx",
      "--baseline-metadata",
      "/models/public.json",
      "--baseline-label",
      "public",
      "--output",
      output,
      "--start-seed",
      "200",
      "--seed-count",
      "2"
    ], io);

    expect(code).toBe(0);
    expect(loadPolicyOnnxModel).toHaveBeenNthCalledWith(1, {
      onnxPath: "/models/compact.onnx",
      metadataPath: "/models/compact.json",
      inferenceDevice: "cpu"
    });
    expect(loadPolicyOnnxModel).toHaveBeenNthCalledWith(2, {
      onnxPath: "/models/public.onnx",
      metadataPath: "/models/public.json",
      inferenceDevice: "cpu"
    });
    expect(runPolicyVsRuleBasedEvaluation).toHaveBeenNthCalledWith(1, {
      policy: completeInfoPolicy,
      startSeed: 200,
      gameCount: 2
    });
    expect(runPolicyVsRuleBasedEvaluation).toHaveBeenNthCalledWith(2, {
      policy: publicPolicy,
      startSeed: 200,
      gameCount: 2
    });
    expect(runStandardPlayingPolicyBenchmarks).not.toHaveBeenCalled();

    const written = JSON.parse(await readFile(output, "utf8"));
    expect(written).toMatchObject({
      schemaVersion: 1,
      evaluationType: "playing-policy-comparison",
      benchmark: "rule-based-x4",
      startSeed: 200,
      endSeed: 201,
      seedCount: 2,
      inferenceDevice: "cpu",
      policies: [
        {
          artifact: {
            label: "complete-info-compact",
            onnxPath: "/models/compact.onnx",
            metadataPath: "/models/compact.json",
            playingObservationVariant: "complete-info-compact",
            modelInputFeatureCount: 385
          },
          result: completeInfoResult
        },
        {
          artifact: {
            label: "public",
            onnxPath: "/models/public.onnx",
            metadataPath: "/models/public.json",
            playingObservationVariant: "public",
            modelInputFeatureCount: 6246
          },
          result: publicResult
        }
      ]
    });
    expect(JSON.parse(io.stdout.write.mock.calls[0][0])).toEqual({
      policies: [
        {
          label: "complete-info-compact",
          variant: "complete-info-compact",
          modelInputFeatureCount: 385,
          runtime: completeInfoPolicy.runtime,
          scheduledGames: 10,
          completedGames: 10,
          failedGames: 0,
          illegalActionCount: 0
        },
        {
          label: "public",
          variant: "public",
          modelInputFeatureCount: 6246,
          runtime: publicPolicy.runtime,
          scheduledGames: 10,
          completedGames: 9,
          failedGames: 1,
          illegalActionCount: 1
        }
      ]
    });

    await rm(directory, { recursive: true, force: true });
  });

  it("rejects incomplete baseline artifact arguments before loading policies", async () => {
    const io = createIo();
    const code = await runPolicyEvaluationCli([
      "--onnx",
      "/models/compact.onnx",
      "--metadata",
      "/models/compact.json",
      "--baseline-onnx",
      "/models/public.onnx",
      "--output",
      "/tmp/evaluation.json",
      "--start-seed",
      "200",
      "--seed-count",
      "2"
    ], io);

    expect(code).toBe(1);
    expect(io.stderr.write.mock.calls.join("\n")).toContain(
      "--baseline-onnx and --baseline-metadata must be provided together."
    );
    expect(loadPolicyOnnxModel).not.toHaveBeenCalled();
  });
});
