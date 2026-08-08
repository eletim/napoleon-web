import {
  ADJUTANT_ENCODER_SCHEMA_VERSION,
  ADJUTANT_MODEL_INPUT_FEATURE_COUNT,
  BIDDING_ACTION_COUNT,
  BIDDING_ENCODER_SCHEMA_VERSION,
  BIDDING_MODEL_INPUT_FEATURE_COUNT,
  CARD_COUNT,
  DATASET_SCHEMA_VERSION,
  EXCHANGE_DISCARD_COUNT,
  EXCHANGE_ENCODER_SCHEMA_VERSION,
  EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
  MODEL_INPUT_FEATURE_COUNT,
  MODEL_INPUT_SCHEMA_VERSION,
  MULTIPHASE_DATASET_SCHEMA_VERSION,
  NONPLAYING_ONNX_METADATA_SCHEMA_VERSION,
  ONNX_BATCH_DIMENSION,
  ONNX_DTYPE,
  ONNX_INPUT_NAME,
  ONNX_OUTPUT_NAME,
  PLAYING_ENCODER_SCHEMA_VERSION,
  POLICY_CHECKPOINT_SCHEMA_VERSION,
  POLICY_ONNX_METADATA_SCHEMA_VERSION
} from "./constants.js";
import type { NonPlayingPolicyType, PolicyOnnxIoMetadata } from "./types.js";

export interface RuntimePolicyOnnxSpec {
  policyType: "playing" | NonPlayingPolicyType;
  artifactType?: string;
  metadataSchemaVersion: number;
  checkpointSchemaVersion: number;
  datasetSchemaVersion: number;
  encoderSchemaVersion: number;
  modelInputSchemaVersion: number;
  modelInputFeatureCount: number;
  outputLogitCount: number;
  actionCount: number;
  inputName: string;
  outputName: string;
  inputShape: readonly [string, number];
  outputShape: readonly [string, number];
  inputDtype: string;
  outputDtype: string;
  discardCount?: number;
}

export const PLAYING_POLICY_ONNX_SPEC: RuntimePolicyOnnxSpec = {
  policyType: "playing",
  metadataSchemaVersion: POLICY_ONNX_METADATA_SCHEMA_VERSION,
  checkpointSchemaVersion: POLICY_CHECKPOINT_SCHEMA_VERSION,
  datasetSchemaVersion: DATASET_SCHEMA_VERSION,
  encoderSchemaVersion: PLAYING_ENCODER_SCHEMA_VERSION,
  modelInputSchemaVersion: MODEL_INPUT_SCHEMA_VERSION,
  modelInputFeatureCount: MODEL_INPUT_FEATURE_COUNT,
  outputLogitCount: CARD_COUNT,
  actionCount: CARD_COUNT,
  inputName: ONNX_INPUT_NAME,
  outputName: ONNX_OUTPUT_NAME,
  inputShape: [ONNX_BATCH_DIMENSION, MODEL_INPUT_FEATURE_COUNT],
  outputShape: [ONNX_BATCH_DIMENSION, CARD_COUNT],
  inputDtype: ONNX_DTYPE,
  outputDtype: ONNX_DTYPE
};

export const NONPLAYING_POLICY_ONNX_SPECS = {
  bidding: {
    policyType: "bidding",
    artifactType: "napoleon-bidding-policy-onnx",
    metadataSchemaVersion: NONPLAYING_ONNX_METADATA_SCHEMA_VERSION,
    checkpointSchemaVersion: POLICY_CHECKPOINT_SCHEMA_VERSION,
    datasetSchemaVersion: MULTIPHASE_DATASET_SCHEMA_VERSION,
    encoderSchemaVersion: BIDDING_ENCODER_SCHEMA_VERSION,
    modelInputSchemaVersion: MODEL_INPUT_SCHEMA_VERSION,
    modelInputFeatureCount: BIDDING_MODEL_INPUT_FEATURE_COUNT,
    outputLogitCount: BIDDING_ACTION_COUNT,
    actionCount: BIDDING_ACTION_COUNT,
    inputName: ONNX_INPUT_NAME,
    outputName: ONNX_OUTPUT_NAME,
    inputShape: [ONNX_BATCH_DIMENSION, BIDDING_MODEL_INPUT_FEATURE_COUNT],
    outputShape: [ONNX_BATCH_DIMENSION, BIDDING_ACTION_COUNT],
    inputDtype: ONNX_DTYPE,
    outputDtype: ONNX_DTYPE
  },
  exchange: {
    policyType: "exchange",
    artifactType: "napoleon-exchange-policy-onnx",
    metadataSchemaVersion: NONPLAYING_ONNX_METADATA_SCHEMA_VERSION,
    checkpointSchemaVersion: POLICY_CHECKPOINT_SCHEMA_VERSION,
    datasetSchemaVersion: MULTIPHASE_DATASET_SCHEMA_VERSION,
    encoderSchemaVersion: EXCHANGE_ENCODER_SCHEMA_VERSION,
    modelInputSchemaVersion: MODEL_INPUT_SCHEMA_VERSION,
    modelInputFeatureCount: EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
    outputLogitCount: CARD_COUNT,
    actionCount: CARD_COUNT,
    inputName: ONNX_INPUT_NAME,
    outputName: ONNX_OUTPUT_NAME,
    inputShape: [ONNX_BATCH_DIMENSION, EXCHANGE_MODEL_INPUT_FEATURE_COUNT],
    outputShape: [ONNX_BATCH_DIMENSION, CARD_COUNT],
    inputDtype: ONNX_DTYPE,
    outputDtype: ONNX_DTYPE,
    discardCount: EXCHANGE_DISCARD_COUNT
  },
  adjutant: {
    policyType: "adjutant",
    artifactType: "napoleon-adjutant-policy-onnx",
    metadataSchemaVersion: NONPLAYING_ONNX_METADATA_SCHEMA_VERSION,
    checkpointSchemaVersion: POLICY_CHECKPOINT_SCHEMA_VERSION,
    datasetSchemaVersion: MULTIPHASE_DATASET_SCHEMA_VERSION,
    encoderSchemaVersion: ADJUTANT_ENCODER_SCHEMA_VERSION,
    modelInputSchemaVersion: MODEL_INPUT_SCHEMA_VERSION,
    modelInputFeatureCount: ADJUTANT_MODEL_INPUT_FEATURE_COUNT,
    outputLogitCount: CARD_COUNT,
    actionCount: CARD_COUNT,
    inputName: ONNX_INPUT_NAME,
    outputName: ONNX_OUTPUT_NAME,
    inputShape: [ONNX_BATCH_DIMENSION, ADJUTANT_MODEL_INPUT_FEATURE_COUNT],
    outputShape: [ONNX_BATCH_DIMENSION, CARD_COUNT],
    inputDtype: ONNX_DTYPE,
    outputDtype: ONNX_DTYPE
  }
} as const satisfies Record<NonPlayingPolicyType, RuntimePolicyOnnxSpec>;

export function getNonPlayingPolicyOnnxSpec(policyType: NonPlayingPolicyType): RuntimePolicyOnnxSpec {
  return NONPLAYING_POLICY_ONNX_SPECS[policyType];
}

export function ioMetadataForSpec(
  spec: RuntimePolicyOnnxSpec,
  kind: "input" | "output"
): PolicyOnnxIoMetadata {
  return {
    name: kind === "input" ? spec.inputName : spec.outputName,
    shape: kind === "input" ? spec.inputShape : spec.outputShape,
    dtype: kind === "input" ? spec.inputDtype : spec.outputDtype
  };
}
