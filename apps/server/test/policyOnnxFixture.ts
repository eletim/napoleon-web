import {
  CARD_COUNT,
  DATASET_SCHEMA_VERSION,
  MODEL_INPUT_FEATURE_COUNT,
  MODEL_INPUT_SCHEMA_VERSION,
  ONNX_INPUT_NAME,
  ONNX_OPSET_VERSION,
  ONNX_OUTPUT_NAME,
  PLAYING_ENCODER_SCHEMA_VERSION,
  POLICY_CHECKPOINT_SCHEMA_VERSION,
  POLICY_ONNX_METADATA_SCHEMA_VERSION,
  calculateCardIdsSha256
} from "@napoleon/policy-onnx";
import { createConstantPolicyOnnx } from "@napoleon/policy-onnx/test-fixtures";

export function createIncreasingLogitPolicyOnnx(): Uint8Array {
  const logits = new Float32Array(CARD_COUNT);

  for (let index = 0; index < CARD_COUNT; index += 1) {
    logits[index] = index;
  }

  return createConstantPolicyOnnx(logits);
}

export function createPlayingPolicyMetadata(): object {
  return {
    metadataSchemaVersion: POLICY_ONNX_METADATA_SCHEMA_VERSION,
    checkpointSchemaVersion: POLICY_CHECKPOINT_SCHEMA_VERSION,
    datasetSchemaVersion: DATASET_SCHEMA_VERSION,
    playingEncoderSchemaVersion: PLAYING_ENCODER_SCHEMA_VERSION,
    modelInputSchemaVersion: MODEL_INPUT_SCHEMA_VERSION,
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
    }
  };
}
