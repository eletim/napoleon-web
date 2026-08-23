import {
  BIDDING_MARGIN_ONNX_METADATA_SCHEMA_VERSION,
  EXCHANGE_DECISION_MODE_TOP3_SET,
  EXCHANGE_DECISION_MODE_SEQUENTIAL_CARD,
  BIDDING_ACTION_COUNT,
  BIDDING_MODEL_INPUT_FEATURE_COUNT,
  ONNX_OPSET_VERSION,
  ONNX_DTYPE
} from "./constants.js";
import { calculateCardIdsSha256 } from "./cardIdsHash.js";
import { PolicyOnnxCompatibilityError } from "./errors.js";
import {
  getNonPlayingPolicyOnnxSpec,
  ioMetadataForSpec,
  getPlayingPolicyOnnxSpec,
  PLAYING_CRITIC_ONNX_SPEC,
  type RuntimeOnnxIoSpec
} from "./policySpecs.js";
import type {
  NonPlayingPolicyOnnxMetadata,
  BiddingMarginOnnxMetadata,
  NonPlayingPolicyType,
  PolicyCriticOnnxMetadata,
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

export function parsePolicyCriticOnnxMetadata(text: string): PolicyCriticOnnxMetadata {
  let raw: unknown;

  try {
    raw = JSON.parse(text);
  } catch (error) {
    throw new PolicyOnnxCompatibilityError(`metadata JSON cannot be parsed: ${String(error)}`);
  }

  validatePolicyCriticOnnxMetadata(raw);
  return raw;
}

export function parseBiddingMarginOnnxMetadata(text: string): BiddingMarginOnnxMetadata {
  let raw: unknown;

  try {
    raw = JSON.parse(text);
  } catch (error) {
    throw new PolicyOnnxCompatibilityError(`metadata JSON cannot be parsed: ${String(error)}`);
  }

  validateBiddingMarginOnnxMetadata(raw);
  return raw;
}

export function validatePolicyOnnxMetadata(value: unknown): asserts value is PolicyOnnxMetadata {
  if (!isRecord(value)) {
    throw new PolicyOnnxCompatibilityError("metadata must be a JSON object.");
  }

  const variant = requirePlayingObservationVariant(value.playingObservationVariant);
  const spec = getPlayingPolicyOnnxSpec(variant);
  expectEqual("metadataSchemaVersion", value.metadataSchemaVersion, spec.metadataSchemaVersion);
  expectEqual("checkpointSchemaVersion", value.checkpointSchemaVersion, spec.checkpointSchemaVersion);
  expectEqual("datasetSchemaVersion", value.datasetSchemaVersion, spec.datasetSchemaVersion);
  if ("playingObservationVariant" in value || variant !== "public") {
    expectEqual("playingObservationVariant", value.playingObservationVariant, variant);
  }
  expectEqual("playingEncoderSchemaVersion", value.playingEncoderSchemaVersion, spec.encoderSchemaVersion);
  expectEqual("modelInputSchemaVersion", value.modelInputSchemaVersion, spec.modelInputSchemaVersion);
  if ("modelInputFeatureCount" in value || variant !== "public") {
    expectEqual("modelInputFeatureCount", value.modelInputFeatureCount, spec.modelInputFeatureCount);
  }
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
  validateDecisionMode(value, spec);

  validateOnnxMetadata(value.onnx, spec);
}

function validateDecisionMode(value: Record<string, unknown>, spec: RuntimeOnnxIoSpec): void {
  if (spec.policyType !== "exchange") {
    if ("decisionMode" in value) {
      throw new PolicyOnnxCompatibilityError("metadata decisionMode is only valid for exchange policy.");
    }
    return;
  }

  const mode = value.decisionMode ?? EXCHANGE_DECISION_MODE_TOP3_SET;
  if (
    mode !== EXCHANGE_DECISION_MODE_TOP3_SET &&
    mode !== EXCHANGE_DECISION_MODE_SEQUENTIAL_CARD
  ) {
    throw new PolicyOnnxCompatibilityError(
      `metadata decisionMode is unsupported: ${JSON.stringify(mode)}.`
    );
  }
}

export function validatePolicyCriticOnnxMetadata(
  value: unknown
): asserts value is PolicyCriticOnnxMetadata {
  if (!isRecord(value)) {
    throw new PolicyOnnxCompatibilityError("metadata must be a JSON object.");
  }

  const spec = PLAYING_CRITIC_ONNX_SPEC;
  expectEqual("metadataSchemaVersion", value.metadataSchemaVersion, spec.metadataSchemaVersion);
  expectEqual("artifactType", value.artifactType, spec.artifactType);
  expectEqual("checkpointSchemaVersion", value.checkpointSchemaVersion, spec.checkpointSchemaVersion);
  expectEqual("datasetSchemaVersion", value.datasetSchemaVersion, spec.datasetSchemaVersion);
  expectEqual("playingEncoderSchemaVersion", value.playingEncoderSchemaVersion, spec.encoderSchemaVersion);
  expectEqual("modelInputSchemaVersion", value.modelInputSchemaVersion, spec.modelInputSchemaVersion);
  expectEqual("modelInputFeatureCount", value.modelInputFeatureCount, spec.modelInputFeatureCount);
  expectEqual("outputValueCount", value.outputValueCount, spec.outputValueCount);
  expectEqual("cardIdsSha256", value.cardIdsSha256, calculateCardIdsSha256());
  expectEqual("inputName", value.inputName, spec.inputName);
  expectEqual("outputName", value.outputName, spec.outputName);
  expectShape("inputShape", value.inputShape, spec.inputShape);
  expectShape("outputShape", value.outputShape, spec.outputShape);
  expectEqual("inputDtype", value.inputDtype, ONNX_DTYPE);
  expectEqual("outputDtype", value.outputDtype, ONNX_DTYPE);

  validateOnnxMetadata(value.onnx, spec);
}

export function validateBiddingMarginOnnxMetadata(
  value: unknown
): asserts value is BiddingMarginOnnxMetadata {
  if (!isRecord(value)) {
    throw new PolicyOnnxCompatibilityError("metadata must be a JSON object.");
  }

  expectEqual("metadataSchemaVersion", value.metadataSchemaVersion, BIDDING_MARGIN_ONNX_METADATA_SCHEMA_VERSION);
  expectEqual("artifactType", value.artifactType, "napoleon-bidding-margin-heteroscedastic-onnx");
  if (
    value.modelType !== "fixed-hand-bidding-margin" &&
    value.modelType !== "bidding-margin-heteroscedastic"
  ) {
    throw new PolicyOnnxCompatibilityError(
      `metadata modelType is unsupported: ${JSON.stringify(value.modelType)}.`
    );
  }
  if (value.variant !== "M1" && value.variant !== "M2") {
    throw new PolicyOnnxCompatibilityError(`metadata variant is unsupported: ${JSON.stringify(value.variant)}.`);
  }
  validateStandardization(value.targetStandardization);
  if (!Number.isFinite(value.constantSigma) || typeof value.constantSigma !== "number" || value.constantSigma < 0) {
    throw new PolicyOnnxCompatibilityError("metadata constantSigma must be a non-negative finite number.");
  }
  expectEqual("inputName", value.inputName, "model_input");
  if (
    !Array.isArray(value.outputNames) ||
    value.outputNames.length !== 2 ||
    value.outputNames[0] !== "mean" ||
    value.outputNames[1] !== "log_variance"
  ) {
    throw new PolicyOnnxCompatibilityError("metadata outputNames must be [\"mean\", \"log_variance\"].");
  }
  expectEqual("outputValueType", value.outputValueType, "standardized-margin-mean-and-log-variance");
  if (!isRecord(value.onnx)) {
    throw new PolicyOnnxCompatibilityError("metadata onnx must be an object.");
  }
  expectEqual("onnx.opsetVersion", value.onnx.opsetVersion, ONNX_OPSET_VERSION);
  validateIoList(value.onnx.inputs, {
    label: "input",
    expected: {
      name: "model_input",
      dtype: ONNX_DTYPE,
      shape: ["batch", BIDDING_MODEL_INPUT_FEATURE_COUNT]
    }
  });
  if (!Array.isArray(value.onnx.outputs) || value.onnx.outputs.length !== 2) {
    throw new PolicyOnnxCompatibilityError("metadata onnx outputs must contain mean and log_variance.");
  }
  validateIoList([value.onnx.outputs[0]], {
    label: "output",
    expected: {
      name: "mean",
      dtype: ONNX_DTYPE,
      shape: ["batch", BIDDING_ACTION_COUNT]
    }
  });
  validateIoList([value.onnx.outputs[1]], {
    label: "output",
    expected: {
      name: "log_variance",
      dtype: ONNX_DTYPE,
      shape: ["batch", BIDDING_ACTION_COUNT]
    }
  });
}

function validateStandardization(value: unknown): void {
  if (!isRecord(value)) {
    throw new PolicyOnnxCompatibilityError("metadata targetStandardization must be an object.");
  }
  if (typeof value.enabled !== "boolean") {
    throw new PolicyOnnxCompatibilityError("metadata targetStandardization.enabled must be boolean.");
  }
  if (typeof value.mean !== "number" || !Number.isFinite(value.mean)) {
    throw new PolicyOnnxCompatibilityError("metadata targetStandardization.mean must be finite.");
  }
  if (typeof value.std !== "number" || !Number.isFinite(value.std) || value.std <= 0) {
    throw new PolicyOnnxCompatibilityError("metadata targetStandardization.std must be positive.");
  }
}

function validateOnnxMetadata(value: unknown, spec: RuntimeOnnxIoSpec): void {
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

function requirePlayingObservationVariant(value: unknown): "public" | "complete-info-compact" {
  if (value === undefined || value === "public") {
    return "public";
  }
  if (value === "complete-info-compact") {
    return value;
  }

  throw new PolicyOnnxCompatibilityError(
    `metadata playingObservationVariant is unsupported: ${JSON.stringify(value)}.`
  );
}
