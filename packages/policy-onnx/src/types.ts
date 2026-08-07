export type PolicyOnnxDimension = string | number;

export interface PolicyOnnxIoMetadata {
  name: string;
  shape: readonly PolicyOnnxDimension[];
  dtype: string;
}

export interface PolicyOnnxMetadata {
  metadataSchemaVersion: number;
  checkpointSchemaVersion: number;
  datasetSchemaVersion: number;
  playingEncoderSchemaVersion: number;
  modelInputSchemaVersion: number;
  cardIdsSha256: string;
  onnx: {
    opsetVersion: number;
    inputs: readonly PolicyOnnxIoMetadata[];
    outputs: readonly PolicyOnnxIoMetadata[];
  };
  policyModel?: unknown;
}

export interface PolicyOnnxLoadOptions {
  onnxPath: string;
  metadataPath: string;
}

export interface SelectLegalPlayInput {
  modelInput: Float32Array | readonly number[];
  legalPlayMask: ArrayLike<number | boolean>;
}

export interface PolicyOnnxSelection {
  selectedCardIndex: number;
  logits: Float32Array;
}
