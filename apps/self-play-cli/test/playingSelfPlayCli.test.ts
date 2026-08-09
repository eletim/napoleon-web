import { beforeEach, describe, expect, it, vi } from "vitest";

const loadPolicyOnnxModel = vi.hoisted(() => vi.fn());
const generatePlayingSelfPlayDataset = vi.hoisted(() => vi.fn());
const childRunnerConstructor = vi.hoisted(() => vi.fn());

vi.mock("@napoleon/policy-onnx", () => ({
  loadPolicyOnnxModel
}));

vi.mock("@napoleon/training-data", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@napoleon/training-data")>()),
  generatePlayingSelfPlayDataset
}));

vi.mock("../src/playingSelfPlayWorkers.js", () => ({
  ChildProcessPlayingSelfPlayGameRunner: childRunnerConstructor
}));

const { runPlayingSelfPlayCli } = await import("../src/playingSelfPlayCli.js");

function createIo() {
  return {
    stdout: { write: vi.fn() },
    stderr: { write: vi.fn() }
  };
}

describe("runPlayingSelfPlayCli", () => {
  beforeEach(() => {
    loadPolicyOnnxModel.mockReset();
    generatePlayingSelfPlayDataset.mockReset();
    childRunnerConstructor.mockReset();
  });

  it("passes rollout worker count and worker-safe roster descriptors", async () => {
    const io = createIo();
    const currentPolicy = { metadata: { policy: "current" } };
    const frozenPolicy = { metadata: { policy: "frozen" } };
    const workerRunner = { runGame: vi.fn(), close: vi.fn() };
    childRunnerConstructor.mockReturnValueOnce(workerRunner);
    loadPolicyOnnxModel
      .mockResolvedValueOnce(currentPolicy)
      .mockResolvedValueOnce(frozenPolicy);
    generatePlayingSelfPlayDataset.mockResolvedValueOnce({
      outputDirectory: "/out",
      manifest: {
        gameCount: 2,
        sampleCount: 20,
        shardCount: 1,
        startSeed: 7,
        endSeed: 8,
        behaviorPolicy: {
          onnxSha256: "a".repeat(64),
          metadataSha256: "b".repeat(64)
        },
        rolloutRoster: { seats: [] }
      }
    });

    const code = await runPlayingSelfPlayCli([
      "--onnx",
      "/models/current.onnx",
      "--metadata",
      "/models/current.json",
      "--output",
      "/out",
      "--start-seed",
      "7",
      "--games",
      "2",
      "--games-per-shard",
      "2",
      "--rollout-workers",
      "2",
      "--rollout-roster",
      JSON.stringify([
        "current-policy",
        "rule-based",
        {
          source: "frozen-onnx",
          onnxPath: "/models/frozen.onnx",
          metadataPath: "/models/frozen.json",
          artifactId: "frozen-v1"
        },
        "rule-based",
        "current-policy"
      ])
    ], io);

    expect(code).toBe(0);
    expect(childRunnerConstructor).toHaveBeenCalledWith({
      workerCount: 2,
      currentPolicy: {
        onnxPath: "/models/current.onnx",
        metadataPath: "/models/current.json"
      },
      rolloutRoster: [
        { source: "current-policy" },
        { source: "rule-based" },
        {
          source: "frozen-onnx",
          onnxPath: "/models/frozen.onnx",
          metadataPath: "/models/frozen.json",
          artifactId: "frozen-v1"
        },
        { source: "rule-based" },
        { source: "current-policy" }
      ],
      temperature: 1
    });
    expect(generatePlayingSelfPlayDataset).toHaveBeenCalledWith(expect.objectContaining({
      rolloutWorkers: 2,
      gameRunner: workerRunner
    }));
    expect(JSON.parse(io.stdout.write.mock.calls[0][0])).toMatchObject({
      rolloutWorkers: 2
    });
  });

  it("keeps workers=1 on the serial generator path", async () => {
    const io = createIo();
    const currentPolicy = { metadata: { policy: "current" } };
    loadPolicyOnnxModel.mockResolvedValueOnce(currentPolicy);
    generatePlayingSelfPlayDataset.mockResolvedValueOnce({
      outputDirectory: "/out",
      manifest: {
        gameCount: 1,
        sampleCount: 10,
        shardCount: 1,
        startSeed: 1,
        endSeed: 1,
        behaviorPolicy: {
          onnxSha256: "a".repeat(64),
          metadataSha256: "b".repeat(64)
        },
        rolloutRoster: { seats: [] }
      }
    });

    const code = await runPlayingSelfPlayCli([
      "--onnx",
      "/models/current.onnx",
      "--metadata",
      "/models/current.json",
      "--output",
      "/out",
      "--start-seed",
      "1",
      "--games",
      "1",
      "--games-per-shard",
      "1"
    ], io);

    expect(code).toBe(0);
    expect(childRunnerConstructor).not.toHaveBeenCalled();
    expect(generatePlayingSelfPlayDataset).toHaveBeenCalledWith(expect.objectContaining({
      rolloutWorkers: 1,
      gameRunner: undefined
    }));
  });

  it("rejects non-positive rollout workers", async () => {
    const io = createIo();
    const code = await runPlayingSelfPlayCli([
      "--onnx",
      "/models/current.onnx",
      "--metadata",
      "/models/current.json",
      "--output",
      "/out",
      "--start-seed",
      "1",
      "--games",
      "1",
      "--games-per-shard",
      "1",
      "--rollout-workers",
      "0"
    ], io);

    expect(code).toBe(1);
    expect(io.stderr.write.mock.calls.join("\n")).toContain("--rollout-workers must be a positive integer");
    expect(generatePlayingSelfPlayDataset).not.toHaveBeenCalled();
  });
});
