import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { describe, expect, it } from "vitest";
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
  maskIllegalPolicyLogits,
  parseNonPlayingPolicyOnnxMetadata,
  parsePolicyOnnxMetadata,
  selectLegalAdjutantCard,
  selectLegalBiddingAction,
  selectLegalExchangeDiscards,
  selectLegalPolicyAction
} from "../src/index.js";
import type { NonPlayingPolicyOnnxMetadata, NonPlayingPolicyType } from "../src/index.js";
import { createConstantPolicyOnnx } from "./testOnnxFixture.js";

describe("policy ONNX metadata", () => {
  it("accepts #29 metadata contract and rejects drift before inference", () => {
    const metadata = createMetadata();

    expect(parsePolicyOnnxMetadata(JSON.stringify(metadata))).toEqual(metadata);

    expect(() =>
      parsePolicyOnnxMetadata(JSON.stringify({ ...metadata, modelInputSchemaVersion: 2 }))
    ).toThrow(/modelInputSchemaVersion/);
    expect(() =>
      parsePolicyOnnxMetadata(
        JSON.stringify({
          ...metadata,
          onnx: {
            ...metadata.onnx,
            inputs: [{ name: "input", shape: ["batch", MODEL_INPUT_FEATURE_COUNT], dtype: "float32" }]
          }
        })
      )
    ).toThrow(/input\.name/);
    expect(() =>
      parsePolicyOnnxMetadata(JSON.stringify({ ...metadata, cardIdsSha256: "0".repeat(64) }))
    ).toThrow(/cardIdsSha256/);
  });
});

describe("legal policy selection", () => {
  it("masks illegal logits and never selects an illegal card", () => {
    const logits = new Float32Array(CARD_COUNT);
    logits[0] = 100;
    logits[3] = 7;
    logits[7] = 9;
    const legalPlayMask = new Uint8Array(CARD_COUNT);
    legalPlayMask[3] = 1;
    legalPlayMask[7] = 1;

    const masked = maskIllegalPolicyLogits(logits, legalPlayMask);

    expect(masked[0]).toBeLessThan(-1e30);
    expect(selectLegalPolicyAction(logits, legalPlayMask)).toBe(7);
  });

  it("rejects masks with no legal card", () => {
    expect(() =>
      selectLegalPolicyAction(new Float32Array(CARD_COUNT), new Uint8Array(CARD_COUNT))
    ).toThrow(/at least one legal card/);
  });
});

describe("non-playing policy ONNX metadata", () => {
  it("accepts #62 metadata contracts and rejects incompatible metadata before inference", () => {
    const bidding = createNonPlayingMetadata("bidding");
    const exchange = createNonPlayingMetadata("exchange");

    expect(parseNonPlayingPolicyOnnxMetadata(JSON.stringify(bidding))).toEqual(bidding);
    expect(parseNonPlayingPolicyOnnxMetadata(JSON.stringify(exchange))).toEqual(exchange);

    const cases: readonly [string, unknown][] = [
      ["metadataSchemaVersion", { ...bidding, metadataSchemaVersion: NONPLAYING_ONNX_METADATA_SCHEMA_VERSION + 1 }],
      ["policyType", { ...bidding, policyType: "playing" }],
      ["datasetSchemaVersion", { ...bidding, datasetSchemaVersion: MULTIPHASE_DATASET_SCHEMA_VERSION + 1 }],
      ["encoderSchemaVersion", { ...bidding, encoderSchemaVersion: 2 }],
      ["modelInputSchemaVersion", { ...bidding, modelInputSchemaVersion: 2 }],
      ["modelInputFeatureCount", { ...bidding, modelInputFeatureCount: BIDDING_MODEL_INPUT_FEATURE_COUNT + 1 }],
      ["outputLogitCount", { ...bidding, outputLogitCount: BIDDING_ACTION_COUNT + 1 }],
      ["cardIdsSha256", { ...bidding, cardIdsSha256: "0".repeat(64) }],
      ["inputName", { ...bidding, inputName: "input" }],
      ["outputShape", { ...bidding, outputShape: ["batch", BIDDING_ACTION_COUNT + 1] }],
      ["inputDtype", { ...bidding, inputDtype: "float64" }],
      ["discardCount", { ...exchange, discardCount: 2 }]
    ];

    for (const [label, metadata] of cases) {
      expect(() => parseNonPlayingPolicyOnnxMetadata(JSON.stringify(metadata)), label).toThrow();
    }
  });
});

describe("non-playing legal selection", () => {
  it("selects deterministic masked bidding and adjutant argmax values", () => {
    const biddingLogits = new Float32Array(BIDDING_ACTION_COUNT);
    biddingLogits[20] = 1000;
    biddingLogits[5] = 9;
    biddingLogits[7] = 9;
    const legalBidMask = Array(BIDDING_ACTION_COUNT).fill(0);
    legalBidMask[3] = 1;
    legalBidMask[5] = true;
    legalBidMask[7] = 1;

    expect(selectLegalBiddingAction(biddingLogits, legalBidMask)).toBe(5);

    const adjutantLogits = new Float32Array(CARD_COUNT);
    adjutantLogits[52] = 1000;
    adjutantLogits[4] = 12;
    adjutantLogits[9] = 12;
    const legalAdjutantMask = new Uint8Array(CARD_COUNT);
    legalAdjutantMask[4] = 1;
    legalAdjutantMask[9] = 1;

    expect(selectLegalAdjutantCard(adjutantLogits, legalAdjutantMask)).toBe(4);
  });

  it("selects a deterministic legal exchange top-3 set", () => {
    const logits = new Float32Array(CARD_COUNT);
    logits[52] = 1000;
    logits[2] = 7;
    logits[6] = 5;
    logits[8] = 5;
    logits[10] = 1;
    const legalDiscardMask = new Uint8Array(CARD_COUNT);
    for (const index of [2, 6, 8, 10]) {
      legalDiscardMask[index] = 1;
    }

    const selected = selectLegalExchangeDiscards(logits, legalDiscardMask);

    expect(selected).toEqual([2, 6, 8]);
    expect(new Set(selected).size).toBe(3);
    expect(selected.every((index) => legalDiscardMask[index] === 1)).toBe(true);
  });

  it("rejects invalid masks and non-finite selected logits", () => {
    expect(() => selectLegalBiddingAction(new Float32Array(BIDDING_ACTION_COUNT), [])).toThrow(/legalBidMask/);
    expect(() => selectLegalAdjutantCard(new Float32Array(CARD_COUNT), new Uint8Array(CARD_COUNT))).toThrow(
      /at least one legal card/
    );
    expect(() => selectLegalExchangeDiscards(new Float32Array(CARD_COUNT), new Uint8Array(CARD_COUNT))).toThrow(
      /at least 3 legal cards/
    );

    const mask = new Uint8Array(BIDDING_ACTION_COUNT);
    mask[0] = 1;
    const logits = new Float32Array(BIDDING_ACTION_COUNT);
    logits[0] = Number.NaN;
    expect(() => selectLegalBiddingAction(logits, mask)).toThrow(/finite/);
  });
});

describe("policy ONNX Runtime smoke", () => {
  it("loads ONNX and metadata, then matches fixed ONNX-side logits and masked selection", async () => {
    const expectedOnnxLogits = new Float32Array(CARD_COUNT);
    for (let index = 0; index < expectedOnnxLogits.length; index += 1) {
      expectedOnnxLogits[index] = index / 10;
    }
    expectedOnnxLogits[52] = 100;
    expectedOnnxLogits[4] = 8;
    expectedOnnxLogits[9] = 9.5;

    const directory = await temporaryDirectory();
    const onnxPath = join(directory, "policy.onnx");
    const metadataPath = join(directory, "policy.json");
    await writeFile(onnxPath, createConstantPolicyOnnx(expectedOnnxLogits));
    await writeFile(metadataPath, JSON.stringify(createMetadata()) + "\n", "utf8");

    const modelInput = new Float32Array(MODEL_INPUT_FEATURE_COUNT);
    modelInput[0] = 1;
    modelInput[MODEL_INPUT_FEATURE_COUNT - 1] = 1;
    const legalPlayMask = new Uint8Array(CARD_COUNT);
    legalPlayMask[4] = 1;
    legalPlayMask[9] = 1;

    const model = await loadPolicyOnnxModel({ onnxPath, metadataPath });
    const actualLogits = await model.predictLogits(modelInput);
    const selection = await model.selectLegalPlay({ modelInput, legalPlayMask });

    expectMaxAbsDiff(actualLogits, expectedOnnxLogits, 1e-6);
    expect(selection.selectedCardIndex).toBe(9);
    expect(legalPlayMask[selection.selectedCardIndex]).toBe(1);
  });

  it("rejects ONNX graph output drift before running inference", async () => {
    const directory = await temporaryDirectory();
    const onnxPath = join(directory, "policy.onnx");
    const metadataPath = join(directory, "policy.json");
    await writeFile(onnxPath, createConstantPolicyOnnx(new Float32Array(CARD_COUNT), "bad_logits"));
    await writeFile(metadataPath, JSON.stringify(createMetadata()) + "\n", "utf8");

    await expect(loadPolicyOnnxModel({ onnxPath, metadataPath })).rejects.toBeInstanceOf(
      PolicyOnnxCompatibilityError
    );
    await expect(loadPolicyOnnxModel({ onnxPath, metadataPath })).rejects.toThrow(/output name/);
  });
});

describe("non-playing policy ONNX Runtime smoke", () => {
  it.each([
    {
      policyType: "bidding" as const,
      inputFeatureCount: BIDDING_MODEL_INPUT_FEATURE_COUNT,
      outputCount: BIDDING_ACTION_COUNT,
      logits: createBiddingGoldenLogits(),
      expectedSelection: 5
    },
    {
      policyType: "exchange" as const,
      inputFeatureCount: EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
      outputCount: CARD_COUNT,
      logits: createExchangeGoldenLogits(),
      expectedSelection: [2, 6, 8]
    },
    {
      policyType: "adjutant" as const,
      inputFeatureCount: ADJUTANT_MODEL_INPUT_FEATURE_COUNT,
      outputCount: CARD_COUNT,
      logits: createAdjutantGoldenLogits(),
      expectedSelection: 4
    }
  ])(
    "loads $policyType ONNX and metadata, matches raw logits and masked parity",
    async ({ policyType, inputFeatureCount, outputCount, logits, expectedSelection }) => {
      const directory = await temporaryDirectory();
      const onnxPath = join(directory, `${policyType}.onnx`);
      const metadataPath = join(directory, `${policyType}.json`);
      await writeFile(
        onnxPath,
        createConstantPolicyOnnx(logits, ONNX_OUTPUT_NAME, {
          inputFeatureCount,
          outputCount
        })
      );
      await writeFile(metadataPath, JSON.stringify(createNonPlayingMetadata(policyType)) + "\n", "utf8");

      const modelInput = new Float32Array(inputFeatureCount);
      modelInput[0] = 1;
      modelInput[inputFeatureCount - 1] = 1;
      const model = await loadNonPlayingPolicyOnnxModel({ onnxPath, metadataPath });
      const actualLogits = await model.predictLogits(modelInput);

      expect(model.policyType).toBe(policyType);
      expectMaxAbsDiff(actualLogits, logits, 1e-6);

      if (policyType === "bidding") {
        const legalBidMask = Array(BIDDING_ACTION_COUNT).fill(0);
        legalBidMask[3] = 1;
        legalBidMask[5] = 1;
        legalBidMask[7] = 1;
        const selection = await model.selectBidding({ modelInput, legalBidMask });
        expect(selection.selectedIndex).toBe(expectedSelection);
        expect(legalBidMask[selection.selectedIndex]).toBe(1);
      } else if (policyType === "exchange") {
        const legalDiscardMask = new Uint8Array(CARD_COUNT);
        for (const index of [2, 6, 8, 10]) {
          legalDiscardMask[index] = 1;
        }
        const selection = await model.selectExchange({ modelInput, legalDiscardMask });
        expect(selection.selectedCardIndices).toEqual(expectedSelection);
        expect(selection.selectedCardIndices).toHaveLength(3);
        expect(new Set(selection.selectedCardIndices).size).toBe(3);
        expect(selection.selectedCardIndices.every((index) => legalDiscardMask[index] === 1)).toBe(true);
      } else {
        const legalAdjutantMask = new Uint8Array(CARD_COUNT);
        legalAdjutantMask[4] = 1;
        legalAdjutantMask[9] = 1;
        const selection = await model.selectAdjutant({ modelInput, legalAdjutantMask });
        expect(selection.selectedIndex).toBe(expectedSelection);
        expect(legalAdjutantMask[selection.selectedIndex]).toBe(1);
      }
    }
  );

  it("rejects non-playing input length and non-finite values before inference", async () => {
    const directory = await temporaryDirectory();
    const onnxPath = join(directory, "bidding.onnx");
    const metadataPath = join(directory, "bidding.json");
    await writeFile(
      onnxPath,
      createConstantPolicyOnnx(createBiddingGoldenLogits(), ONNX_OUTPUT_NAME, {
        inputFeatureCount: BIDDING_MODEL_INPUT_FEATURE_COUNT,
        outputCount: BIDDING_ACTION_COUNT
      })
    );
    await writeFile(metadataPath, JSON.stringify(createNonPlayingMetadata("bidding")) + "\n", "utf8");

    const model = await loadNonPlayingPolicyOnnxModel({ onnxPath, metadataPath });
    await expect(model.predictLogits(new Float32Array(BIDDING_MODEL_INPUT_FEATURE_COUNT - 1))).rejects.toThrow(
      /2333/
    );

    const input = new Float32Array(BIDDING_MODEL_INPUT_FEATURE_COUNT);
    input[12] = Number.POSITIVE_INFINITY;
    await expect(model.predictLogits(input)).rejects.toThrow(/modelInput\[12\] must be finite/);
  });

  it("rejects non-playing ONNX graph drift before running inference", async () => {
    const directory = await temporaryDirectory();
    const onnxPath = join(directory, "exchange.onnx");
    const metadataPath = join(directory, "exchange.json");
    await writeFile(
      onnxPath,
      createConstantPolicyOnnx(createExchangeGoldenLogits(), ONNX_OUTPUT_NAME, {
        inputFeatureCount: EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
        outputCount: CARD_COUNT - 1
      })
    );
    await writeFile(metadataPath, JSON.stringify(createNonPlayingMetadata("exchange")) + "\n", "utf8");

    await expect(loadNonPlayingPolicyOnnxModel({ onnxPath, metadataPath })).rejects.toBeInstanceOf(
      PolicyOnnxCompatibilityError
    );
    await expect(loadNonPlayingPolicyOnnxModel({ onnxPath, metadataPath })).rejects.toThrow(/output shape/);
  });
});

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

function createBiddingGoldenLogits(): Float32Array {
  const logits = new Float32Array(BIDDING_ACTION_COUNT);
  logits[20] = 1000;
  logits[3] = 1;
  logits[5] = 9;
  logits[7] = 9;
  return logits;
}

function createExchangeGoldenLogits(): Float32Array {
  const logits = new Float32Array(CARD_COUNT);
  logits[52] = 1000;
  logits[2] = 7;
  logits[6] = 5;
  logits[8] = 5;
  logits[10] = 1;
  return logits;
}

function createAdjutantGoldenLogits(): Float32Array {
  const logits = new Float32Array(CARD_COUNT);
  logits[52] = 1000;
  logits[4] = 12;
  logits[9] = 12;
  return logits;
}

function expectMaxAbsDiff(actual: Float32Array, expected: Float32Array, tolerance: number): void {
  expect(actual).toHaveLength(expected.length);
  let maxAbsDiff = 0;

  for (let index = 0; index < expected.length; index += 1) {
    maxAbsDiff = Math.max(maxAbsDiff, Math.abs(actual[index] - expected[index]));
  }

  expect(maxAbsDiff).toBeLessThanOrEqual(tolerance);
}

async function temporaryDirectory(): Promise<string> {
  const directory = join(tmpdir(), `policy-onnx-${process.pid}-${Date.now()}-${Math.random()}`);
  await mkdir(directory, { recursive: true });
  return directory;
}
