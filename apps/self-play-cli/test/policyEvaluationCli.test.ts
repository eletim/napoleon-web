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
      gameCount: 1
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
});
