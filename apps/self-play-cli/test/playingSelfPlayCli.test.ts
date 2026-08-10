import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
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

function createDatasetResult(overrides: Record<string, unknown> = {}) {
  return {
    outputDirectory: "/out",
    manifest: {
      gameCount: 1,
      sampleCount: 10,
      shardCount: 1,
      startSeed: 1,
      endSeed: 1,
      behaviorPolicy: {
        requestedInferenceDevice: "cpu",
        resolvedInferenceDevice: "cpu",
        executionProvider: "cpu",
        onnxSha256: "a".repeat(64),
        metadataSha256: "b".repeat(64)
      },
      rolloutRoster: { seats: [] },
      ...overrides
    },
    rolloutTiming: {
      rolloutElapsedSeconds: 0.25,
      inference: {
        requestCount: 4,
        sessionRunCount: 2,
        meanBatchSize: 2,
        maxObservedBatchSize: 3,
        batchSizeHistogram: { "1": 1, "3": 1 }
      }
    }
  };
}

describe("runPlayingSelfPlayCli", () => {
  beforeEach(() => {
    loadPolicyOnnxModel.mockReset();
    generatePlayingSelfPlayDataset.mockReset();
    childRunnerConstructor.mockReset();
  });

  it("passes rollout worker count and worker-safe roster descriptors", async () => {
    const directory = await mkdtemp(join(tmpdir(), "playing-cli-test-"));
    const currentOnnx = join(directory, "current.onnx");
    const currentMetadata = join(directory, "current.json");
    const frozenOnnx = join(directory, "frozen.onnx");
    const frozenMetadata = join(directory, "frozen.json");
    try {
      const io = createIo();
      const currentPolicy = { metadata: { policy: "current" } };
      const frozenPolicy = { metadata: { policy: "frozen" } };
      const workerRunner = { runGame: vi.fn(), close: vi.fn() };
      childRunnerConstructor.mockReturnValueOnce(workerRunner);
      loadPolicyOnnxModel
        .mockResolvedValueOnce(currentPolicy)
        .mockResolvedValueOnce(frozenPolicy);
      generatePlayingSelfPlayDataset.mockResolvedValueOnce(createDatasetResult({
          gameCount: 2,
          sampleCount: 20,
          startSeed: 7,
          endSeed: 8,
      }));
      await writeFile(currentOnnx, "current-onnx");
      await writeFile(currentMetadata, "current-metadata");
      await writeFile(frozenOnnx, "frozen-onnx");
      await writeFile(frozenMetadata, "frozen-metadata");

      const code = await runPlayingSelfPlayCli([
        "--onnx",
        currentOnnx,
        "--metadata",
        currentMetadata,
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
            onnxPath: frozenOnnx,
            metadataPath: frozenMetadata,
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
          onnxPath: currentOnnx,
          metadataPath: currentMetadata,
          inferenceDevice: "cpu",
          inferenceMaxBatchSize: 256,
          onnxSha256: "f1b90777ed3270f25bcb35d754aceefd946660ca3d50362e02135704c60cd051",
          metadataSha256: "ff8e755f6a6c8c7a4b36a040afca64b79c479684f18b10477eff43c2db2600fb"
        },
        rolloutRoster: [
          { source: "current-policy" },
          { source: "rule-based" },
          {
            source: "frozen-onnx",
            onnxPath: frozenOnnx,
            metadataPath: frozenMetadata,
            onnxSha256: "4aacdae2154a3246d2159fe1d0dc3be49a8a24c26b9ad748efcfbf2db9e92e37",
            metadataSha256: "94d3ea5e4dc8f8dcc8f93e90536bfdd686144acc5c4bc1ee03710b412a52c529",
            artifactId: "frozen-v1"
          },
          { source: "rule-based" },
          { source: "current-policy" }
        ],
        temperature: 1
      });
      expect(generatePlayingSelfPlayDataset).toHaveBeenCalledWith(expect.objectContaining({
        rolloutWorkers: 2,
        rolloutConcurrency: 2,
        inferenceMaxBatchSize: 256,
        gameRunner: workerRunner
      }));
      expect(JSON.parse(io.stdout.write.mock.calls[0][0])).toMatchObject({
        rolloutWorkers: 2,
        rolloutConcurrency: 2,
        inferenceMaxBatchSize: 256,
        inferenceSessionRunCount: 2
      });
      expect(loadPolicyOnnxModel).toHaveBeenNthCalledWith(1, {
        onnxPath: currentOnnx,
        metadataPath: currentMetadata,
        inferenceDevice: "cpu",
        inferenceMaxBatchSize: 256
      });
      expect(loadPolicyOnnxModel).toHaveBeenNthCalledWith(2, {
        onnxPath: frozenOnnx,
        metadataPath: frozenMetadata,
        inferenceDevice: "cpu",
        inferenceMaxBatchSize: 256
      });
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  it("keeps workers=1 on the serial generator path", async () => {
    const io = createIo();
    const currentPolicy = { metadata: { policy: "current" } };
    loadPolicyOnnxModel.mockResolvedValueOnce(currentPolicy);
    generatePlayingSelfPlayDataset.mockResolvedValueOnce(createDatasetResult());

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
      rolloutConcurrency: 1,
      inferenceMaxBatchSize: 256,
      gameRunner: undefined
    }));
  });

  it("passes explicit CUDA inference device to current, frozen, and worker policies", async () => {
    const directory = await mkdtemp(join(tmpdir(), "playing-cli-test-"));
    const currentOnnx = join(directory, "current.onnx");
    const currentMetadata = join(directory, "current.json");
    const frozenOnnx = join(directory, "frozen.onnx");
    const frozenMetadata = join(directory, "frozen.json");
    try {
      const io = createIo();
      const currentPolicy = {
        metadata: { policy: "current" },
        runtime: {
          requestedInferenceDevice: "cuda",
          resolvedInferenceDevice: "cuda",
          executionProvider: "cuda"
        }
      };
      const frozenPolicy = {
        metadata: { policy: "frozen" },
        runtime: {
          requestedInferenceDevice: "cuda",
          resolvedInferenceDevice: "cuda",
          executionProvider: "cuda"
        }
      };
      loadPolicyOnnxModel
        .mockResolvedValueOnce(currentPolicy)
        .mockResolvedValueOnce(frozenPolicy);
      generatePlayingSelfPlayDataset.mockResolvedValueOnce(createDatasetResult({
          behaviorPolicy: {
            requestedInferenceDevice: "cuda",
            resolvedInferenceDevice: "cuda",
            executionProvider: "cuda",
            onnxSha256: "a".repeat(64),
            metadataSha256: "b".repeat(64)
          }
      }));
      await writeFile(currentOnnx, "current-onnx");
      await writeFile(currentMetadata, "current-metadata");
      await writeFile(frozenOnnx, "frozen-onnx");
      await writeFile(frozenMetadata, "frozen-metadata");

      const code = await runPlayingSelfPlayCli([
        "--onnx",
        currentOnnx,
        "--metadata",
        currentMetadata,
        "--output",
        "/out",
        "--start-seed",
        "1",
        "--games",
        "1",
        "--games-per-shard",
        "1",
        "--rollout-workers",
        "2",
        "--inference-device",
        "cuda",
        "--rollout-roster",
        JSON.stringify([
          "current-policy",
          "rule-based",
          {
            source: "frozen-onnx",
            onnxPath: frozenOnnx,
            metadataPath: frozenMetadata
          },
          "rule-based",
          "current-policy"
        ])
      ], io);

      expect(code).toBe(0);
      expect(loadPolicyOnnxModel).toHaveBeenNthCalledWith(1, {
        onnxPath: currentOnnx,
        metadataPath: currentMetadata,
        inferenceDevice: "cuda",
        inferenceMaxBatchSize: 256
      });
      expect(loadPolicyOnnxModel).toHaveBeenNthCalledWith(2, {
        onnxPath: frozenOnnx,
        metadataPath: frozenMetadata,
        inferenceDevice: "cuda",
        inferenceMaxBatchSize: 256
      });
      expect(childRunnerConstructor).not.toHaveBeenCalled();
      expect(generatePlayingSelfPlayDataset).toHaveBeenCalledWith(expect.objectContaining({
        rolloutWorkers: 2,
        rolloutConcurrency: 2,
        gameRunner: undefined
      }));
      expect(JSON.parse(io.stdout.write.mock.calls[0][0])).toMatchObject({
        requestedInferenceDevice: "cuda",
        resolvedInferenceDevice: "cuda",
        executionProvider: "cuda"
      });
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
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
