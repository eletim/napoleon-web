import {
  ONNX_OPSET_VERSION,
  ONNX_DTYPE
} from "./constants.js";
import { calculateCardIdsSha256 } from "./cardIdsHash.js";
import { PolicyOnnxCompatibilityError } from "./errors.js";
import {
  getNonPlayingPolicyOnnxSpec,
  ioMetadataForSpec,
  PLAYING_POLICY_ONNX_SPEC,
  type RuntimePolicyOnnxSpec
} from "./policySpecs.js";
import type {
  NonPlayingPolicyOnnxMetadata,
  NonPlayingPolicyType,
  PolicyOnnxIoMetadata,
  PolicyOnnxMetadata
} from "./types.js";

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

export function parseNonPlayingPolicyOnnxMetadata(text: string): NonPlayingPolicyOnnxMetadata {
  let raw: unknown;

  try {
    raw = JSON.parse(text);
  } catch (error) {
    throw new PolicyOnnxCompatibilityError(`metadata JSON cannot be parsed: ${String(error)}`);
  }

  validateNonPlayingPolicyOnnxMetadata(raw);
  return raw;
}

export function validatePolicyOnnxMetadata(value: unknown): asserts value is PolicyOnnxMetadata {
  if (!isRecord(value)) {
    throw new PolicyOnnxCompatibilityError("metadata must be a JSON object.");
  }

  const spec = PLAYING_POLICY_ONNX_SPEC;
  expectEqual("metadataSchemaVersion", value.metadataSchemaVersion, spec.metadataSchemaVersion);
  expectEqual("checkpointSchemaVersion", value.checkpointSchemaVersion, spec.checkpointSchemaVersion);
  expectEqual("datasetSchemaVersion", value.datasetSchemaVersion, spec.datasetSchemaVersion);
  expectEqual("playingEncoderSchemaVersion", value.playingEncoderSchemaVersion, spec.encoderSchemaVersion);
  expectEqual("modelInputSchemaVersion", value.modelInputSchemaVersion, spec.modelInputSchemaVersion);
  expectEqual("cardIdsSha256", value.cardIdsSha256, calculateCardIdsSha256());

  validateOnnxMetadata(value.onnx, spec);
}

export function validateNonPlayingPolicyOnnxMetadata(
  value: unknown
): asserts value is NonPlayingPolicyOnnxMetadata {
  if (!isRecord(value)) {
    throw new PolicyOnnxCompatibilityError("metadata must be a JSON object.");
  }

  const policyType = requireNonPlayingPolicyType(value.policyType);
  const spec = getNonPlayingPolicyOnnxSpec(policyType);

  expectEqual("metadataSchemaVersion", value.metadataSchemaVersion, spec.metadataSchemaVersion);
  expectEqual("artifactType", value.artifactType, spec.artifactType);
  expectEqual("policyType", value.policyType, spec.policyType);
  expectEqual("checkpointSchemaVersion", value.checkpointSchemaVersion, spec.checkpointSchemaVersion);
  expectEqual("datasetSchemaVersion", value.datasetSchemaVersion, spec.datasetSchemaVersion);
  expectEqual("encoderSchemaVersion", value.encoderSchemaVersion, spec.encoderSchemaVersion);
  expectEqual("modelInputSchemaVersion", value.modelInputSchemaVersion, spec.modelInputSchemaVersion);
  expectEqual("modelInputFeatureCount", value.modelInputFeatureCount, spec.modelInputFeatureCount);
  expectEqual("outputLogitCount", value.outputLogitCount, spec.outputLogitCount);
  expectEqual("actionCount", value.actionCount, spec.actionCount);
  expectEqual("cardIdsSha256", value.cardIdsSha256, calculateCardIdsSha256());
  expectEqual("inputName", value.inputName, spec.inputName);
  expectEqual("outputName", value.outputName, spec.outputName);
  expectShape("inputShape", value.inputShape, spec.inputShape);
  expectShape("outputShape", value.outputShape, spec.outputShape);
  expectEqual("inputDtype", value.inputDtype, ONNX_DTYPE);
  expectEqual("outputDtype", value.outputDtype, ONNX_DTYPE);

  if (spec.discardCount === undefined) {
    if ("discardCount" in value) {
      throw new PolicyOnnxCompatibilityError("metadata discardCount is only valid for exchange policy.");
    }
  } else {
    expectEqual("discardCount", value.discardCount, spec.discardCount);
  }

  validateOnnxMetadata(value.onnx, spec);
}

function validateOnnxMetadata(value: unknown, spec: RuntimePolicyOnnxSpec): void {
  if (!isRecord(value)) {
    throw new PolicyOnnxCompatibilityError("metadata onnx must be an object.");
  }

  expectEqual("onnx.opsetVersion", value.opsetVersion, ONNX_OPSET_VERSION);
  validateIoList(value.inputs, {
    label: "input",
    expected: ioMetadataForSpec(spec, "input")
  });
  validateIoList(value.outputs, {
    label: "output",
    expected: ioMetadataForSpec(spec, "output")
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
  expectShape(`onnx.${label}.shape`, item.shape, expected.shape);
  expectEqual(`onnx.${label}.dtype`, item.dtype, expected.dtype);
}

function expectEqual(label: string, actual: unknown, expected: unknown): void {
  if (actual !== expected) {
    throw new PolicyOnnxCompatibilityError(
      `metadata ${label} mismatch: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}.`
    );
  }
}

function expectShape(label: string, actual: unknown, expected: readonly unknown[]): void {
  if (!sameShape(actual, expected)) {
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

function requireNonPlayingPolicyType(value: unknown): NonPlayingPolicyType {
  if (value === "bidding" || value === "exchange" || value === "adjutant") {
    return value;
  }

  throw new PolicyOnnxCompatibilityError(`metadata policyType is unsupported: ${JSON.stringify(value)}.`);
}
