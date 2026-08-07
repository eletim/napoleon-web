import { CARD_COUNT, PLAYING_ENCODER_SCHEMA_VERSION } from "@napoleon/ai-observation";
import { DATASET_SCHEMA_VERSION } from "@napoleon/training-data";

export const POLICY_ONNX_METADATA_SCHEMA_VERSION = 1 as const;
export const POLICY_CHECKPOINT_SCHEMA_VERSION = 1 as const;
export const MODEL_INPUT_SCHEMA_VERSION = 1 as const;
export const MODEL_INPUT_FEATURE_COUNT = 6242 as const;
export const ONNX_OPSET_VERSION = 18 as const;
export const ONNX_INPUT_NAME = "model_input" as const;
export const ONNX_OUTPUT_NAME = "logits" as const;
export const ONNX_DTYPE = "float32" as const;
export const ONNX_TENSOR_FLOAT_TYPE = "tensor(float)" as const;
export const ONNX_BATCH_DIMENSION = "batch" as const;

export {
  CARD_COUNT,
  DATASET_SCHEMA_VERSION,
  PLAYING_ENCODER_SCHEMA_VERSION
};
