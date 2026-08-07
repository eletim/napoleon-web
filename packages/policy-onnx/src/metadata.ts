import {
  CARD_COUNT,
  DATASET_SCHEMA_VERSION,
  MODEL_INPUT_FEATURE_COUNT,
  MODEL_INPUT_SCHEMA_VERSION,
  ONNX_BATCH_DIMENSION,
  ONNX_DTYPE,
  ONNX_INPUT_NAME,
  ONNX_OPSET_VERSION,
  ONNX_OUTPUT_NAME,
  PLAYING_ENCODER_SCHEMA_VERSION,
  POLICY_CHECKPOINT_SCHEMA_VERSION,
  POLICY_ONNX_METADATA_SCHEMA_VERSION
} from "./constants.js";
import { calculateCardIdsSha256 } from "./cardIdsHash.js";
import { PolicyOnnxCompatibilityError } from "./errors.js";
import type { PolicyOnnxIoMetadata, PolicyOnnxMetadata } from "./types.js";

export function parsePolicyOnnxMetadata(text: string): PolicyOnnxMetadata {
  let raw: unknown;

  try {
    raw = JSON.parse(text);
  } catch (error) {
    throw new PolicyOnnxCompatibilityError(`metadata JSON cannot be parsed: ${String(error)}`);
  }

  validatePolicyOnnxMetadata(raw);
  return raw;
}

export function validatePolicyOnnxMetadata(value: unknown): asserts value is PolicyOnnxMetadata {
  if (!isRecord(value)) {
    throw new PolicyOnnxCompatibilityError("metadata must be a JSON object.");
  }

  expectEqual("metadataSchemaVersion", value.metadataSchemaVersion, POLICY_ONNX_METADATA_SCHEMA_VERSION);
  expectEqual("checkpointSchemaVersion", value.checkpointSchemaVersion, POLICY_CHECKPOINT_SCHEMA_VERSION);
  expectEqual("datasetSchemaVersion", value.datasetSchemaVersion, DATASET_SCHEMA_VERSION);
  expectEqual("playingEncoderSchemaVersion", value.playingEncoderSchemaVersion, PLAYING_ENCODER_SCHEMA_VERSION);
  expectEqual("modelInputSchemaVersion", value.modelInputSchemaVersion, MODEL_INPUT_SCHEMA_VERSION);
  expectEqual("cardIdsSha256", value.cardIdsSha256, calculateCardIdsSha256());

  if (!isRecord(value.onnx)) {
    throw new PolicyOnnxCompatibilityError("metadata onnx must be an object.");
  }

  expectEqual("onnx.opsetVersion", value.onnx.opsetVersion, ONNX_OPSET_VERSION);
  validateIoList(value.onnx.inputs, {
    label: "input",
    expected: {
      name: ONNX_INPUT_NAME,
      shape: [ONNX_BATCH_DIMENSION, MODEL_INPUT_FEATURE_COUNT],
      dtype: ONNX_DTYPE
    }
  });
  validateIoList(value.onnx.outputs, {
    label: "output",
    expected: {
      name: ONNX_OUTPUT_NAME,
      shape: [ONNX_BATCH_DIMENSION, CARD_COUNT],
      dtype: ONNX_DTYPE
    }
  });
}

function validateIoList(
  value: unknown,
  {
    label,
    expected
  }: {
    label: string;
    expected: PolicyOnnxIoMetadata;
  }
): void {
  if (!Array.isArray(value) || value.length !== 1) {
    throw new PolicyOnnxCompatibilityError(`metadata onnx ${label}s must contain one item.`);
  }

  const item = value[0];
  if (!isRecord(item)) {
    throw new PolicyOnnxCompatibilityError(`metadata onnx ${label} item must be an object.`);
  }

  expectEqual(`onnx.${label}.name`, item.name, expected.name);
  if (!sameShape(item.shape, expected.shape)) {
    throw new PolicyOnnxCompatibilityError(
      `metadata onnx ${label} shape mismatch: expected ${JSON.stringify(expected.shape)}, got ${JSON.stringify(item.shape)}.`
    );
  }
  expectEqual(`onnx.${label}.dtype`, item.dtype, expected.dtype);
}

function expectEqual(label: string, actual: unknown, expected: unknown): void {
  if (actual !== expected) {
    throw new PolicyOnnxCompatibilityError(
      `metadata ${label} mismatch: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}.`
    );
  }
}

function sameShape(actual: unknown, expected: readonly unknown[]): boolean {
  return (
    Array.isArray(actual) &&
    actual.length === expected.length &&
    actual.every((value, index) => value === expected[index])
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
