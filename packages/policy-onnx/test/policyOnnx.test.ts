import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
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
  maskIllegalPolicyLogits,
  parsePolicyOnnxMetadata,
  selectLegalPolicyAction
} from "../src/index.js";
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
