export type PolicyOnnxDimension = string | number;
export type NonPlayingPolicyType = "bidding" | "exchange" | "adjutant";

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

export interface NonPlayingPolicyOnnxMetadata {
  metadataSchemaVersion: number;
  artifactType: string;
  policyType: NonPlayingPolicyType;
  checkpointSchemaVersion: number;
  datasetSchemaVersion: number;
  encoderSchemaVersion: number;
  modelInputSchemaVersion: number;
  modelInputFeatureCount: number;
  outputLogitCount: number;
  actionCount: number;
  cardIdsSha256: string;
  inputName: string;
  outputName: string;
  inputShape: readonly PolicyOnnxDimension[];
  outputShape: readonly PolicyOnnxDimension[];
  inputDtype: string;
  outputDtype: string;
  discardCount?: number;
  onnx: {
    opsetVersion: number;
    inputs: readonly PolicyOnnxIoMetadata[];
    outputs: readonly PolicyOnnxIoMetadata[];
  };
  modelConfig?: unknown;
  checkpointSeed?: unknown;
  checkpointCompatibilityMetadata?: unknown;
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

export interface SelectLegalBiddingInput {
  modelInput: Float32Array | readonly number[];
  legalBidMask: ArrayLike<number | boolean>;
}

export interface SelectLegalExchangeInput {
  modelInput: Float32Array | readonly number[];
  legalDiscardMask: ArrayLike<number | boolean>;
}

export interface SelectLegalAdjutantInput {
  modelInput: Float32Array | readonly number[];
  legalAdjutantMask: ArrayLike<number | boolean>;
}

export interface NonPlayingPolicyOnnxSingleSelection {
  selectedIndex: number;
  logits: Float32Array;
}

export interface NonPlayingPolicyOnnxExchangeSelection {
  selectedCardIndices: readonly [number, number, number];
  logits: Float32Array;
}
