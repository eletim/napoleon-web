import { mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  CARD_COUNT,
  MODEL_INPUT_FEATURE_COUNT,
  ONNX_INPUT_NAME,
  ONNX_OPSET_VERSION,
  ONNX_OUTPUT_NAME,
  PolicyOnnxCompatibilityError,
  calculateCardIdsSha256,
  loadPolicyOnnxModel,
  runPolicyVsRuleBasedEvaluation
} from "../src/index.js";
import type { PolicyOnnxModel, SelectLegalPlayInput } from "../src/index.js";
import { createConstantPolicyOnnx } from "./testOnnxFixture.js";

const playerIds = ["player-0", "player-1", "player-2", "player-3", "player-4"] as const;

describe("runPolicyVsRuleBasedEvaluation", () => {
  it("runs a reproducible ONNX policy versus RuleBasedAgent comparison across all seats", async () => {
    const policy = await createIncreasingLogitPolicy();
    const options = {
      policy,
      startSeed: 700,
      gameCount: 2,
      playerIds
    };
    const first = await runPolicyVsRuleBasedEvaluation(options);
    const second = await runPolicyVsRuleBasedEvaluation(options);

    expect(second.run).toEqual(first.run);
    expect(second.comparison).toEqual(first.comparison);
    expect(first.configuration).toMatchObject({
      startSeed: 700,
      endSeed: 701,
      gameCount: 2,
      rotationOffsets: [0, 1, 2, 3, 4],
      playerIds,
      policyAgentName: "PolicyOnnxAgent",
      ruleBasedAgentName: "RuleBasedAgent"
    });
    expect(first.run.games).toHaveLength(10);
    expect(first.run.completedCount).toBe(10);
    expect(first.run.failedCount).toBe(0);
    expect(first.comparison.illegalActionCount).toBe(0);
    expect(first.comparison.failedGames).toEqual([]);

    expect(first.comparison.policy).toMatchObject({
      agentGroup: "policy",
      agentName: "PolicyOnnxAgent",
      sourceAgentIndices: [0],
      games: {
        total: 10,
        completed: 10,
        failed: 0
      },
      sampleCount: 10
    });
    expect(first.comparison.ruleBased).toMatchObject({
      agentGroup: "rule-based",
      agentName: "RuleBasedAgent",
      sourceAgentIndices: [1, 2, 3, 4],
      games: {
        total: 40,
        completed: 40,
        failed: 0
      },
      sampleCount: 40
    });
    expect(first.comparison.policy.seatResults.map((seat) => [
      seat.seatIndex,
      seat.sampleCount
    ])).toEqual([
      [0, 2],
      [1, 2],
      [2, 2],
      [3, 2],
      [4, 2]
    ]);
    expect(first.comparison.ruleBased.seatResults.map((seat) => [
      seat.seatIndex,
      seat.sampleCount
    ])).toEqual([
      [0, 8],
      [1, 8],
      [2, 8],
      [3, 8],
      [4, 8]
    ]);
    expect(first.comparison.policy.roleResults.map((role) => role.role)).toEqual([
      "napoleon",
      "adjutant",
      "alliance"
    ]);
    expect(first.comparison.policy.roleResults.reduce(
      (sum, role) => sum + role.sampleCount,
      0
    )).toBe(10);
    expect(first.comparison.ruleBased.roleResults.reduce(
      (sum, role) => sum + role.sampleCount,
      0
    )).toBe(40);
    expect(first.comparison.policy.winRate.confidenceInterval.method).toBe("wilson");
    expect(first.comparison.policy.contractSuccessRate.confidenceInterval.method).toBe("wilson");
    expect(first.comparison.policy.comparison.winRateDeltaConfidenceInterval.method)
      .toBe("newcombe-wilson");
    expect(first.comparison.policy.comparison.contractSuccessRateDeltaConfidenceInterval.method)
      .toBe("newcombe-wilson");
    expect(first.comparison.policy.comparison.averagePointCardsDeltaConfidenceInterval.method)
      .toBe("normal");
    expect(first.comparison.policy.comparison.winRateDelta).toBe(
      -first.comparison.ruleBased.comparison.winRateDelta!
    );
    expectReportHasNoNonFiniteNumbers(first.comparison);

    for (const game of first.run.games) {
      expect(game.seats.filter((seat) => seat.sourceAgentIndex === 0)).toHaveLength(1);
      expect(game.seats.filter((seat) => seat.agentName === "RuleBasedAgent")).toHaveLength(4);
    }
  });

  it("records failed games instead of excluding them from the comparison", async () => {
    const failingPolicy = {
      metadata: createMetadata(),
      async selectLegalPlay(): Promise<never> {
        throw new PolicyOnnxCompatibilityError("forced policy failure");
      }
    } as unknown as PolicyOnnxModel;
    const result = await runPolicyVsRuleBasedEvaluation({
      policy: failingPolicy,
      startSeed: 800,
      gameCount: 1,
      playerIds
    });

    expect(result.run.games).toHaveLength(5);
    expect(result.run.completedCount).toBe(0);
    expect(result.run.failedCount).toBe(5);
    expect(result.comparison.failedGames).toHaveLength(5);
    expect(result.comparison.illegalActionCount).toBe(0);
    expect(result.comparison.policy.failures).toEqual({
      total: 5,
      byReason: {
        "forced policy failure": 5
      }
    });
    expect(result.comparison.ruleBased.failures).toEqual({
      total: 20,
      byReason: {
        "forced policy failure": 20
      }
    });
  });

  it("counts ONNX selections outside legal play actions as illegal action failures", async () => {
    const illegalSelectionPolicy = {
      metadata: createMetadata(),
      async selectLegalPlay(input: SelectLegalPlayInput) {
        const illegalCardIndex = Array.from(
          { length: input.legalPlayMask.length },
          (_, index) => index
        ).find((index) => input.legalPlayMask[index] === 0);

        if (illegalCardIndex === undefined) {
          throw new Error("Expected at least one illegal card index.");
        }

        return {
          selectedCardIndex: illegalCardIndex,
          logits: new Float32Array(CARD_COUNT)
        };
      }
    } as unknown as PolicyOnnxModel;
    const result = await runPolicyVsRuleBasedEvaluation({
      policy: illegalSelectionPolicy,
      startSeed: 810,
      gameCount: 1,
      playerIds
    });

    expect(result.run.games).toHaveLength(5);
    expect(result.run.completedCount).toBe(0);
    expect(result.run.failedCount).toBe(5);
    expect(result.comparison.failedGames).toHaveLength(5);
    expect(result.comparison.illegalActionCount).toBe(5);
    expect(result.comparison.failedGames.every((game) =>
      game.failureReason.includes("outside legal actions")
    )).toBe(true);
  });
});

const externalOnnxPath = process.env.NAPOLEON_POLICY_ONNX_PATH;
const externalMetadataPath = process.env.NAPOLEON_POLICY_METADATA_PATH;
const maybeExternalIt = externalOnnxPath !== undefined && externalMetadataPath !== undefined
  ? it
  : it.skip;

describe("Policy vs RuleBased external artifact smoke", () => {
  maybeExternalIt("runs complete deterministic mixed-agent comparison games", async () => {
    if (externalOnnxPath === undefined || externalMetadataPath === undefined) {
      throw new Error("Expected external policy artifact paths.");
    }

    const policy = await loadPolicyOnnxModel({
      onnxPath: externalOnnxPath,
      metadataPath: externalMetadataPath
    });
    const first = await runPolicyVsRuleBasedEvaluation({
      policy,
      startSeed: 900,
      gameCount: 1,
      playerIds
    });
    const second = await runPolicyVsRuleBasedEvaluation({
      policy,
      startSeed: 900,
      gameCount: 1,
      playerIds
    });

    expect(first.run.games).toHaveLength(5);
    expect(first.run.completedCount).toBe(5);
    expect(first.run.failedCount).toBe(0);
    expect(first.comparison.illegalActionCount).toBe(0);
    expect(first.comparison.failedGames).toEqual([]);
    expect(second.run).toEqual(first.run);
    expect(second.comparison).toEqual(first.comparison);
    expectReportHasNoNonFiniteNumbers(first.comparison);
  });
});

async function createIncreasingLogitPolicy(): Promise<PolicyOnnxModel> {
  const logits = new Float32Array(CARD_COUNT);
  for (let index = 0; index < CARD_COUNT; index += 1) {
    logits[index] = index;
  }

  const directory = await temporaryDirectory();
  const onnxPath = join(directory, "policy.onnx");
  const metadataPath = join(directory, "policy.json");
  await writeFile(onnxPath, createConstantPolicyOnnx(logits));
  await writeFile(metadataPath, JSON.stringify(createMetadata()) + "\n", "utf8");

  return loadPolicyOnnxModel({ onnxPath, metadataPath });
}

function createMetadata() {
  return {
    metadataSchemaVersion: 1,
    checkpointSchemaVersion: 1,
    datasetSchemaVersion: 1,
    playingEncoderSchemaVersion: 1,
    modelInputSchemaVersion: 1,
    cardIdsSha256: calculateCardIdsSha256(),
    onnx: {
      opsetVersion: ONNX_OPSET_VERSION,
      inputs: [
        {
          name: ONNX_INPUT_NAME,
          shape: ["batch", MODEL_INPUT_FEATURE_COUNT],
          dtype: "float32"
        }
      ],
      outputs: [
        {
          name: ONNX_OUTPUT_NAME,
          shape: ["batch", CARD_COUNT],
          dtype: "float32"
        }
      ]
    },
    policyModel: {
      input_dim: MODEL_INPUT_FEATURE_COUNT,
      hidden_dim: 8,
      hidden_layers: 1,
      dropout: 0
    }
  };
}

function expectReportHasNoNonFiniteNumbers(value: unknown): void {
  if (typeof value === "number") {
    expect(Number.isFinite(value)).toBe(true);
    return;
  }

  if (Array.isArray(value)) {
    for (const item of value) {
      expectReportHasNoNonFiniteNumbers(item);
    }
    return;
  }

  if (value !== null && typeof value === "object") {
    for (const item of Object.values(value)) {
      expectReportHasNoNonFiniteNumbers(item);
    }
  }
}

async function temporaryDirectory(): Promise<string> {
  const directory = join(tmpdir(), `policy-vs-rulebased-${process.pid}-${Date.now()}-${Math.random()}`);
  await mkdir(directory, { recursive: true });
  return directory;
}
