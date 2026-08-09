import { mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";
import {
  ADJUTANT_MODEL_INPUT_FEATURE_COUNT,
  BIDDING_ACTION_COUNT,
  BIDDING_MODEL_INPUT_FEATURE_COUNT,
  CARD_COUNT,
  EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
  MODEL_INPUT_FEATURE_COUNT,
  MULTIPHASE_DATASET_SCHEMA_VERSION,
  NONPLAYING_ONNX_METADATA_SCHEMA_VERSION,
  ONNX_INPUT_NAME,
  ONNX_OPSET_VERSION,
  ONNX_OUTPUT_NAME,
  PolicyOnnxCompatibilityError,
  calculateCardIdsSha256,
  loadNonPlayingPolicyOnnxModel,
  loadPolicyOnnxModel,
  runFullPolicyVsRuleBasedEvaluation,
  runPolicyVsRuleBasedEvaluation
} from "../src/index.js";
import type {
  NonPlayingPolicyOnnxMetadata,
  NonPlayingPolicyOnnxModel,
  NonPlayingPolicyType,
  PolicyOnnxModel,
  SelectLegalPlayInput
} from "../src/index.js";
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

describe("runFullPolicyVsRuleBasedEvaluation", () => {
  it("runs a deterministic full-policy comparison across all five seats", async () => {
    const {
      playingPolicy,
      biddingPolicy,
      adjutantPolicy,
      exchangePolicy
    } = await createFullPolicyFixture();
    const biddingSpy = vi.spyOn(biddingPolicy, "selectBidding");
    const adjutantSpy = vi.spyOn(adjutantPolicy, "selectAdjutant");
    const exchangeSpy = vi.spyOn(exchangePolicy, "selectExchange");
    const playingSpy = vi.spyOn(playingPolicy, "selectLegalPlay");
    const options = {
      playingPolicy,
      biddingPolicy,
      adjutantPolicy,
      exchangePolicy,
      startSeed: 1000,
      gameCount: 1,
      playerIds
    };

    try {
      const first = await runFullPolicyVsRuleBasedEvaluation(options);
      const firstPhaseCalls = {
        bidding: biddingSpy.mock.calls.length,
        adjutant: adjutantSpy.mock.calls.length,
        exchange: exchangeSpy.mock.calls.length,
        playing: playingSpy.mock.calls.length
      };
      const second = await runFullPolicyVsRuleBasedEvaluation(options);

      expect(first.configuration).toMatchObject({
        startSeed: 1000,
        endSeed: 1000,
        gameCount: 1,
        rotationOffsets: [0, 1, 2, 3, 4],
        playerIds,
        policyAgentName: "FullPolicyOnnxAgent",
        ruleBasedAgentName: "RuleBasedAgent",
        policyMetadata: {
          bidding: { policyType: "bidding" },
          adjutant: { policyType: "adjutant" },
          exchange: { policyType: "exchange" }
        }
      });
      expect(first.configuration.policyMetadata.playing.playingEncoderSchemaVersion).toBe(2);
      expect(first.run.games).toHaveLength(5);
      expect(first.run.completedCount).toBe(5);
      expect(first.run.failedCount).toBe(0);
      expect(first.comparison.illegalActionCount).toBe(0);
      expect(first.comparison.failedGames).toEqual([]);

      expect(first.comparison.policy).toMatchObject({
        agentGroup: "policy",
        agentName: "FullPolicyOnnxAgent",
        sourceAgentIndices: [0],
        games: {
          total: 5,
          completed: 5,
          failed: 0
        },
        sampleCount: 5
      });
      expect(first.comparison.ruleBased).toMatchObject({
        agentGroup: "rule-based",
        agentName: "RuleBasedAgent",
        sourceAgentIndices: [1, 2, 3, 4],
        games: {
          total: 20,
          completed: 20,
          failed: 0
        },
        sampleCount: 20
      });
      expect(first.comparison.policy.seatResults.map((seat) => [
        seat.seatIndex,
        seat.sampleCount
      ])).toEqual([
        [0, 1],
        [1, 1],
        [2, 1],
        [3, 1],
        [4, 1]
      ]);
      expect(first.run.games.map((game) =>
        game.seats.find((seat) => seat.sourceAgentIndex === 0)?.seatIndex
      )).toEqual([0, 1, 2, 3, 4]);
      expect(first.run.games.every((game) =>
        game.seats.filter((seat) => seat.sourceAgentIndex === 0).length === 1 &&
        game.seats.filter((seat) => seat.agentName === "RuleBasedAgent").length === 4
      )).toBe(true);
      expect(first.comparison.policy.roleResults.map((role) => role.role)).toEqual([
        "napoleon",
        "adjutant",
        "alliance"
      ]);
      expect(first.comparison.policy.roleResults.reduce(
        (sum, role) => sum + role.sampleCount,
        0
      )).toBe(5);
      expect(first.comparison.policy.comparison.winRateDeltaConfidenceInterval.method)
        .toBe("newcombe-wilson");
      expect(first.comparison.policy.comparison.contractSuccessRateDeltaConfidenceInterval.method)
        .toBe("newcombe-wilson");
      expect(first.comparison.policy.comparison.averagePointCardsDeltaConfidenceInterval.method)
        .toBe("normal");
      expectReportHasNoNonFiniteNumbers(first.comparison);

      expect(firstPhaseCalls.bidding).toBeGreaterThan(0);
      expect(firstPhaseCalls.adjutant).toBeGreaterThan(0);
      expect(firstPhaseCalls.exchange).toBeGreaterThan(0);
      expect(firstPhaseCalls.playing).toBeGreaterThan(0);
      expect(first.diagnostics.policyAgentDecisionCounts).toMatchObject({
        ruleBasedFallbackDecisionCount: 0
      });
      expect(first.diagnostics.policyAgentDecisionCounts.biddingOnnxCallCount).toBeGreaterThan(0);
      expect(first.diagnostics.policyAgentDecisionCounts.adjutantOnnxCallCount).toBeGreaterThan(0);
      expect(first.diagnostics.policyAgentDecisionCounts.exchangeOnnxCallCount).toBeGreaterThan(0);
      expect(first.diagnostics.policyAgentDecisionCounts.playingOnnxCallCount).toBeGreaterThan(0);

      expect(second.run).toEqual(first.run);
      expect(second.comparison).toEqual(first.comparison);
      expect(second.diagnostics).toEqual(first.diagnostics);
    } finally {
      playingSpy.mockRestore();
      exchangeSpy.mockRestore();
      adjutantSpy.mockRestore();
      biddingSpy.mockRestore();
    }
  });

  it("rejects full-policy artifacts assigned to the wrong phase slot before running games", async () => {
    const playingPolicy = await createIncreasingLogitPolicy();
    const exchangePolicy = await createNonPlayingPolicy("exchange");
    const adjutantPolicy = await createNonPlayingPolicy("adjutant");

    await expect(runFullPolicyVsRuleBasedEvaluation({
      playingPolicy,
      biddingPolicy: exchangePolicy,
      adjutantPolicy,
      exchangePolicy,
      startSeed: 1100,
      gameCount: 1,
      playerIds
    })).rejects.toThrow(/biddingPolicy policy type mismatch/);
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

const externalBiddingOnnxPath = process.env.NAPOLEON_BIDDING_POLICY_ONNX_PATH;
const externalBiddingMetadataPath = process.env.NAPOLEON_BIDDING_POLICY_METADATA_PATH;
const externalAdjutantOnnxPath = process.env.NAPOLEON_ADJUTANT_POLICY_ONNX_PATH;
const externalAdjutantMetadataPath = process.env.NAPOLEON_ADJUTANT_POLICY_METADATA_PATH;
const externalExchangeOnnxPath = process.env.NAPOLEON_EXCHANGE_POLICY_ONNX_PATH;
const externalExchangeMetadataPath = process.env.NAPOLEON_EXCHANGE_POLICY_METADATA_PATH;
const maybeExternalFullIt =
  externalOnnxPath !== undefined &&
  externalMetadataPath !== undefined &&
  externalBiddingOnnxPath !== undefined &&
  externalBiddingMetadataPath !== undefined &&
  externalAdjutantOnnxPath !== undefined &&
  externalAdjutantMetadataPath !== undefined &&
  externalExchangeOnnxPath !== undefined &&
  externalExchangeMetadataPath !== undefined
    ? it
    : it.skip;

describe("Full policy vs RuleBased external artifact smoke", () => {
  maybeExternalFullIt("runs complete deterministic full-policy comparison games", async () => {
    if (
      externalOnnxPath === undefined ||
      externalMetadataPath === undefined ||
      externalBiddingOnnxPath === undefined ||
      externalBiddingMetadataPath === undefined ||
      externalAdjutantOnnxPath === undefined ||
      externalAdjutantMetadataPath === undefined ||
      externalExchangeOnnxPath === undefined ||
      externalExchangeMetadataPath === undefined
    ) {
      throw new Error("Expected all external full-policy artifact paths.");
    }

    const first = await runFullPolicyVsRuleBasedEvaluation({
      playingPolicy: await loadPolicyOnnxModel({
        onnxPath: externalOnnxPath,
        metadataPath: externalMetadataPath
      }),
      biddingPolicy: await loadNonPlayingPolicyOnnxModel({
        onnxPath: externalBiddingOnnxPath,
        metadataPath: externalBiddingMetadataPath
      }),
      adjutantPolicy: await loadNonPlayingPolicyOnnxModel({
        onnxPath: externalAdjutantOnnxPath,
        metadataPath: externalAdjutantMetadataPath
      }),
      exchangePolicy: await loadNonPlayingPolicyOnnxModel({
        onnxPath: externalExchangeOnnxPath,
        metadataPath: externalExchangeMetadataPath
      }),
      startSeed: 910,
      gameCount: 1,
      playerIds
    });
    const second = await runFullPolicyVsRuleBasedEvaluation({
      playingPolicy: await loadPolicyOnnxModel({
        onnxPath: externalOnnxPath,
        metadataPath: externalMetadataPath
      }),
      biddingPolicy: await loadNonPlayingPolicyOnnxModel({
        onnxPath: externalBiddingOnnxPath,
        metadataPath: externalBiddingMetadataPath
      }),
      adjutantPolicy: await loadNonPlayingPolicyOnnxModel({
        onnxPath: externalAdjutantOnnxPath,
        metadataPath: externalAdjutantMetadataPath
      }),
      exchangePolicy: await loadNonPlayingPolicyOnnxModel({
        onnxPath: externalExchangeOnnxPath,
        metadataPath: externalExchangeMetadataPath
      }),
      startSeed: 910,
      gameCount: 1,
      playerIds
    });

    expect(first.run.games).toHaveLength(5);
    expect(first.run.completedCount).toBe(5);
    expect(first.run.failedCount).toBe(0);
    expect(first.comparison.illegalActionCount).toBe(0);
    expect(first.comparison.failedGames).toEqual([]);
    expect(first.diagnostics.policyAgentDecisionCounts.ruleBasedFallbackDecisionCount).toBe(0);
    expect(first.diagnostics.policyAgentDecisionCounts.biddingOnnxCallCount).toBeGreaterThan(0);
    expect(first.diagnostics.policyAgentDecisionCounts.adjutantOnnxCallCount).toBeGreaterThan(0);
    expect(first.diagnostics.policyAgentDecisionCounts.exchangeOnnxCallCount).toBeGreaterThan(0);
    expect(first.diagnostics.policyAgentDecisionCounts.playingOnnxCallCount).toBeGreaterThan(0);
    expect(first.run.games.map((game) =>
      game.seats.find((seat) => seat.sourceAgentIndex === 0)?.seatIndex
    )).toEqual([0, 1, 2, 3, 4]);
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

async function createFullPolicyFixture(): Promise<{
  playingPolicy: PolicyOnnxModel;
  biddingPolicy: NonPlayingPolicyOnnxModel;
  adjutantPolicy: NonPlayingPolicyOnnxModel;
  exchangePolicy: NonPlayingPolicyOnnxModel;
}> {
  return {
    playingPolicy: await createIncreasingLogitPolicy(),
    biddingPolicy: await createNonPlayingPolicy("bidding"),
    adjutantPolicy: await createNonPlayingPolicy("adjutant"),
    exchangePolicy: await createNonPlayingPolicy("exchange")
  };
}

async function createNonPlayingPolicy(
  policyType: NonPlayingPolicyType
): Promise<NonPlayingPolicyOnnxModel> {
  const spec = nonPlayingSpec(policyType);
  const logits = new Float32Array(spec.outputCount);
  for (let index = 0; index < logits.length; index += 1) {
    logits[index] = index;
  }

  const directory = await temporaryDirectory();
  const onnxPath = join(directory, `${policyType}.onnx`);
  const metadataPath = join(directory, `${policyType}.json`);
  await writeFile(
    onnxPath,
    createConstantPolicyOnnx(logits, ONNX_OUTPUT_NAME, {
      inputFeatureCount: spec.inputFeatureCount,
      outputCount: spec.outputCount
    })
  );
  await writeFile(metadataPath, JSON.stringify(createNonPlayingMetadata(policyType)) + "\n", "utf8");

  return loadNonPlayingPolicyOnnxModel({ onnxPath, metadataPath });
}

function createMetadata() {
  return {
    metadataSchemaVersion: 1,
    checkpointSchemaVersion: 1,
    datasetSchemaVersion: 1,
    playingEncoderSchemaVersion: 2,
    modelInputSchemaVersion: 2,
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

function createNonPlayingMetadata(policyType: NonPlayingPolicyType): NonPlayingPolicyOnnxMetadata {
  const spec = nonPlayingSpec(policyType);
  const metadata: NonPlayingPolicyOnnxMetadata = {
    metadataSchemaVersion: NONPLAYING_ONNX_METADATA_SCHEMA_VERSION,
    artifactType: spec.artifactType,
    policyType,
    checkpointSchemaVersion: 1,
    datasetSchemaVersion: MULTIPHASE_DATASET_SCHEMA_VERSION,
    encoderSchemaVersion: 1,
    modelInputSchemaVersion: 1,
    modelInputFeatureCount: spec.inputFeatureCount,
    outputLogitCount: spec.outputCount,
    actionCount: spec.outputCount,
    cardIdsSha256: calculateCardIdsSha256(),
    inputName: ONNX_INPUT_NAME,
    outputName: ONNX_OUTPUT_NAME,
    inputShape: ["batch", spec.inputFeatureCount],
    outputShape: ["batch", spec.outputCount],
    inputDtype: "float32",
    outputDtype: "float32",
    onnx: {
      opsetVersion: ONNX_OPSET_VERSION,
      inputs: [
        {
          name: ONNX_INPUT_NAME,
          shape: ["batch", spec.inputFeatureCount],
          dtype: "float32"
        }
      ],
      outputs: [
        {
          name: ONNX_OUTPUT_NAME,
          shape: ["batch", spec.outputCount],
          dtype: "float32"
        }
      ]
    },
    modelConfig: {
      input_dim: spec.inputFeatureCount,
      hidden_dim: 8,
      hidden_layers: 1,
      dropout: 0
    },
    checkpointSeed: 123,
    checkpointCompatibilityMetadata: {
      modelInputFeatureCount: spec.inputFeatureCount,
      outputCount: spec.outputCount
    }
  };

  if (policyType === "exchange") {
    metadata.discardCount = 3;
  }

  return metadata;
}

function nonPlayingSpec(policyType: NonPlayingPolicyType): {
  artifactType: string;
  inputFeatureCount: number;
  outputCount: number;
} {
  if (policyType === "bidding") {
    return {
      artifactType: "napoleon-bidding-policy-onnx",
      inputFeatureCount: BIDDING_MODEL_INPUT_FEATURE_COUNT,
      outputCount: BIDDING_ACTION_COUNT
    };
  }
  if (policyType === "exchange") {
    return {
      artifactType: "napoleon-exchange-policy-onnx",
      inputFeatureCount: EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
      outputCount: CARD_COUNT
    };
  }
  return {
    artifactType: "napoleon-adjutant-policy-onnx",
    inputFeatureCount: ADJUTANT_MODEL_INPUT_FEATURE_COUNT,
    outputCount: CARD_COUNT
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
