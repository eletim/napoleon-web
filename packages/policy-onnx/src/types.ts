export type PolicyOnnxDimension = string | number;
export type NonPlayingPolicyType = "bidding" | "exchange" | "adjutant";
export type PolicyOnnxInferenceDevice = "cpu" | "auto" | "cuda";
export type PolicyOnnxExecutionProvider = "cpu" | "cuda";

export interface PolicyOnnxRuntimeInfo {
  requestedInferenceDevice: PolicyOnnxInferenceDevice;
  resolvedInferenceDevice: PolicyOnnxExecutionProvider;
  executionProvider: PolicyOnnxExecutionProvider;
}

export interface PolicyOnnxInferenceStats {
  requestCount: number;
  sessionRunCount: number;
  meanBatchSize: number;
  maxObservedBatchSize: number;
  batchSizeHistogram: Readonly<Record<string, number>>;
}

export interface PolicyOnnxIoMetadata {
  name: string;
  shape: readonly PolicyOnnxDimension[];
  dtype: string;
}

export interface PolicyOnnxMetadata {
  metadataSchemaVersion: number;
  checkpointSchemaVersion: number;
  datasetSchemaVersion: number;
  playingObservationVariant?: "public" | "complete-info-compact";
  playingEncoderSchemaVersion: number;
  modelInputSchemaVersion: number;
  modelInputFeatureCount?: number;
  cardIdsSha256: string;
  onnx: {
    opsetVersion: number;
    inputs: readonly PolicyOnnxIoMetadata[];
    outputs: readonly PolicyOnnxIoMetadata[];
  };
  policyModel?: unknown;
  sourceCheckpointSha256?: string;
}

export interface PolicyCriticOnnxMetadata {
  metadataSchemaVersion: number;
  artifactType: string;
  checkpointSchemaVersion: number;
  datasetSchemaVersion: number;
  playingEncoderSchemaVersion: number;
  modelInputSchemaVersion: number;
  modelInputFeatureCount: number;
  outputValueCount: 1;
  cardIdsSha256: string;
  inputName: string;
  outputName: string;
  inputShape: readonly PolicyOnnxDimension[];
  outputShape: readonly PolicyOnnxDimension[];
  inputDtype: string;
  outputDtype: string;
  onnx: {
    opsetVersion: number;
    inputs: readonly PolicyOnnxIoMetadata[];
    outputs: readonly PolicyOnnxIoMetadata[];
  };
  criticModel?: unknown;
  modelArchitecture?: unknown;
  sourceCheckpointSha256?: string;
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
  inferenceDevice?: PolicyOnnxInferenceDevice;
  inferenceMaxBatchSize?: number;
  sessionFactory?: PolicyOnnxSessionFactory;
}

export interface PolicyCriticOnnxSelection {
  value: number;
  winRateEquivalent: number;
}

export type PolicyOnnxSessionFactory = (
  onnxPath: string,
  options: { executionProviders: readonly PolicyOnnxExecutionProvider[] }
) => Promise<unknown>;

export interface SelectLegalPlayInput {
  modelInput: Float32Array | readonly number[];
  legalPlayMask: ArrayLike<number | boolean>;
}

export interface PolicyOnnxSelection {
  selectedCardIndex: number;
  logits: Float32Array;
}

export interface PolicyOnnxSampledSelection extends PolicyOnnxSelection {
  logProbability: number;
}

export interface SampleLegalPolicyActionOptions {
  logits: Float32Array | readonly number[];
  legalPlayMask: ArrayLike<number | boolean>;
  rng: () => number;
  temperature?: number;
}

export interface CalculateLegalPolicyLogProbabilityOptions {
  logits: Float32Array | readonly number[];
  legalPlayMask: ArrayLike<number | boolean>;
  selectedCardIndex: number;
  temperature?: number;
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
