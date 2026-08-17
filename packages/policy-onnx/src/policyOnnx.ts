import { readFile } from "node:fs/promises";
import * as ort from "onnxruntime-node";
import {
  BIDDING_ACTION_COUNT,
  CARD_COUNT,
  EXCHANGE_DISCARD_COUNT
} from "./constants.js";
import { PolicyOnnxCompatibilityError } from "./errors.js";
import {
  parseNonPlayingPolicyOnnxMetadata,
  parsePolicyCriticOnnxMetadata,
  parsePolicyOnnxMetadata
} from "./metadata.js";
import { validateOnnxModelIo } from "./onnxProto.js";
import {
  getNonPlayingPolicyOnnxSpec,
  getPlayingPolicyOnnxSpec,
  PLAYING_CRITIC_ONNX_SPEC,
  type RuntimeCriticOnnxSpec,
  type RuntimeOnnxIoSpec,
  type RuntimePolicyOnnxSpec
} from "./policySpecs.js";
import type {
  NonPlayingPolicyOnnxExchangeSelection,
  NonPlayingPolicyOnnxMetadata,
  NonPlayingPolicyOnnxSingleSelection,
  CalculateLegalPolicyLogProbabilityOptions,
  PolicyCriticOnnxMetadata,
  PolicyCriticOnnxSelection,
  PolicyOnnxExecutionProvider,
  PolicyOnnxInferenceDevice,
  PolicyOnnxInferenceStats,
  PolicyOnnxLoadOptions,
  PolicyOnnxMetadata,
  PolicyOnnxRuntimeInfo,
  PolicyOnnxSampledSelection,
  PolicyOnnxSelection,
  PolicyOnnxSessionFactory,
  SampleLegalPolicyActionOptions,
  SelectLegalAdjutantInput,
  SelectLegalBiddingInput,
  SelectLegalExchangeInput,
  SelectLegalPlayInput
} from "./types.js";

const FLOAT32_MIN = -3.4028234663852886e38;

export class PolicyOnnxModel {
  private readonly inferenceQueue: PolicyOnnxInferenceQueue;
  private readonly spec: RuntimePolicyOnnxSpec;

  constructor(
    readonly metadata: PolicyOnnxMetadata,
    session: ort.InferenceSession,
    readonly runtime: PolicyOnnxRuntimeInfo,
    options: { inferenceMaxBatchSize?: number } = {}
  ) {
    this.spec = getPlayingPolicyOnnxSpec(metadata.playingObservationVariant ?? "public");
    this.inferenceQueue = new PolicyOnnxInferenceQueue(
      session,
      this.spec,
      options.inferenceMaxBatchSize ?? 1
    );
  }

  async predictLogits(modelInput: Float32Array | readonly number[]): Promise<Float32Array> {
    return this.inferenceQueue.predict(modelInput);
  }

  async predictLogitsBatch(
    modelInputs: readonly (Float32Array | readonly number[])[]
  ): Promise<readonly Float32Array[]> {
    return this.inferenceQueue.predictBatch(modelInputs);
  }

  getInferenceStats(): PolicyOnnxInferenceStats {
    return this.inferenceQueue.getStats();
  }

  resetInferenceStats(): void {
    this.inferenceQueue.resetStats();
  }

  async selectLegalPlay(input: SelectLegalPlayInput): Promise<PolicyOnnxSelection> {
    const logits = await this.predictLogits(input.modelInput);
    const selectedCardIndex = selectLegalPolicyAction(logits, input.legalPlayMask);

    return {
      selectedCardIndex,
      logits
    };
  }

  async sampleLegalPlay(
    input: SelectLegalPlayInput & { rng: () => number; temperature?: number }
  ): Promise<PolicyOnnxSampledSelection> {
    const logits = await this.predictLogits(input.modelInput);
    const selection = sampleLegalPolicyAction({
      logits,
      legalPlayMask: input.legalPlayMask,
      rng: input.rng,
      temperature: input.temperature
    });

    return {
      ...selection,
      logits
    };
  }
}

export class NonPlayingPolicyOnnxModel {
  readonly policyType: NonPlayingPolicyOnnxMetadata["policyType"];
  private readonly spec: RuntimePolicyOnnxSpec;

  constructor(
    readonly metadata: NonPlayingPolicyOnnxMetadata,
    private readonly session: ort.InferenceSession,
    readonly runtime: PolicyOnnxRuntimeInfo
  ) {
    this.policyType = metadata.policyType;
    this.spec = getNonPlayingPolicyOnnxSpec(metadata.policyType);
  }

  async predictLogits(modelInput: Float32Array | readonly number[]): Promise<Float32Array> {
    return runPolicyOnnxLogitsBatch(
      this.session,
      this.spec,
      [normalizeModelInputForSpec(modelInput, this.spec)]
    ).then((outputs) => outputs[0]);
  }

  async selectBidding(input: SelectLegalBiddingInput): Promise<NonPlayingPolicyOnnxSingleSelection> {
    this.assertPolicyType("bidding");
    const logits = await this.predictLogits(input.modelInput);
    return {
      selectedIndex: selectLegalBiddingAction(logits, input.legalBidMask),
      logits
    };
  }

  async selectExchange(input: SelectLegalExchangeInput): Promise<NonPlayingPolicyOnnxExchangeSelection> {
    this.assertPolicyType("exchange");
    const logits = await this.predictLogits(input.modelInput);
    return {
      selectedCardIndices: selectLegalExchangeDiscards(logits, input.legalDiscardMask),
      logits
    };
  }

  async selectExchangeCard(input: SelectLegalExchangeInput): Promise<NonPlayingPolicyOnnxSingleSelection> {
    this.assertPolicyType("exchange");
    const logits = await this.predictLogits(input.modelInput);
    return {
      selectedIndex: selectLegalExchangeCard(logits, input.legalDiscardMask),
      logits
    };
  }

  async selectAdjutant(input: SelectLegalAdjutantInput): Promise<NonPlayingPolicyOnnxSingleSelection> {
    this.assertPolicyType("adjutant");
    const logits = await this.predictLogits(input.modelInput);
    return {
      selectedIndex: selectLegalAdjutantCard(logits, input.legalAdjutantMask),
      logits
    };
  }

  private assertPolicyType(policyType: NonPlayingPolicyOnnxMetadata["policyType"]): void {
    if (this.metadata.policyType !== policyType) {
      throw new PolicyOnnxCompatibilityError(
        `ONNX policy type mismatch: expected ${policyType}, got ${this.metadata.policyType}.`
      );
    }
  }
}

export class PolicyCriticOnnxModel {
  private readonly inferenceQueue: PolicyCriticOnnxInferenceQueue;

  constructor(
    readonly metadata: PolicyCriticOnnxMetadata,
    session: ort.InferenceSession,
    readonly runtime: PolicyOnnxRuntimeInfo,
    options: { inferenceMaxBatchSize?: number } = {}
  ) {
    this.inferenceQueue = new PolicyCriticOnnxInferenceQueue(
      session,
      PLAYING_CRITIC_ONNX_SPEC,
      options.inferenceMaxBatchSize ?? 1
    );
  }

  async predictValue(modelInput: Float32Array | readonly number[]): Promise<number> {
    return this.inferenceQueue.predict(modelInput);
  }

  async predictValuesBatch(
    modelInputs: readonly (Float32Array | readonly number[])[]
  ): Promise<readonly number[]> {
    return this.inferenceQueue.predictBatch(modelInputs);
  }

  async predictWinRateEquivalent(
    modelInput: Float32Array | readonly number[]
  ): Promise<PolicyCriticOnnxSelection> {
    const value = await this.predictValue(modelInput);
    return {
      value,
      winRateEquivalent: criticValueToWinRateEquivalent(value)
    };
  }

  getInferenceStats(): PolicyOnnxInferenceStats {
    return this.inferenceQueue.getStats();
  }

  resetInferenceStats(): void {
    this.inferenceQueue.resetStats();
  }
}

export async function loadPolicyOnnxModel(options: PolicyOnnxLoadOptions): Promise<PolicyOnnxModel> {
  const metadata = parsePolicyOnnxMetadata(await readFile(options.metadataPath, "utf8"));
  const spec = getPlayingPolicyOnnxSpec(metadata.playingObservationVariant ?? "public");
  await validateOnnxModelIo(options.onnxPath, metadata, spec);

  const { session, runtime } = await createPolicyOnnxSession(options);

  validateSessionNames(session, spec);

  return new PolicyOnnxModel(metadata, session, runtime, {
    inferenceMaxBatchSize: options.inferenceMaxBatchSize
  });
}

export async function loadNonPlayingPolicyOnnxModel(
  options: PolicyOnnxLoadOptions
): Promise<NonPlayingPolicyOnnxModel> {
  const metadata = parseNonPlayingPolicyOnnxMetadata(await readFile(options.metadataPath, "utf8"));
  const spec = getNonPlayingPolicyOnnxSpec(metadata.policyType);
  await validateOnnxModelIo(options.onnxPath, metadata, spec);

  const { session, runtime } = await createPolicyOnnxSession(options);

  validateSessionNames(session, spec);

  return new NonPlayingPolicyOnnxModel(metadata, session, runtime);
}

export async function loadPolicyCriticOnnxModel(options: PolicyOnnxLoadOptions): Promise<PolicyCriticOnnxModel> {
  const metadata = parsePolicyCriticOnnxMetadata(await readFile(options.metadataPath, "utf8"));
  await validateOnnxModelIo(options.onnxPath, metadata, PLAYING_CRITIC_ONNX_SPEC);

  const { session, runtime } = await createPolicyOnnxSession(options);

  validateSessionNames(session, PLAYING_CRITIC_ONNX_SPEC);

  return new PolicyCriticOnnxModel(metadata, session, runtime, {
    inferenceMaxBatchSize: options.inferenceMaxBatchSize
  });
}

export function criticValueToWinRateEquivalent(value: number): number {
  if (!Number.isFinite(value)) {
    throw new PolicyOnnxCompatibilityError(`critic value must be finite, got ${value}.`);
  }

  return Math.min(1, Math.max(0, (value + 1) / 2));
}

async function createPolicyOnnxSession(options: PolicyOnnxLoadOptions): Promise<{
  session: ort.InferenceSession;
  runtime: PolicyOnnxRuntimeInfo;
}> {
  const requestedInferenceDevice = options.inferenceDevice ?? "cpu";
  validateInferenceDevice(requestedInferenceDevice);
  const sessionFactory = options.sessionFactory ?? defaultSessionFactory;

  if (requestedInferenceDevice === "cpu") {
    return {
      session: await createSession(sessionFactory, options.onnxPath, "cpu", requestedInferenceDevice),
      runtime: {
        requestedInferenceDevice,
        resolvedInferenceDevice: "cpu",
        executionProvider: "cpu"
      }
    };
  }

  try {
    return {
      session: await createSession(sessionFactory, options.onnxPath, "cuda", requestedInferenceDevice),
      runtime: {
        requestedInferenceDevice,
        resolvedInferenceDevice: "cuda",
        executionProvider: "cuda"
      }
    };
  } catch (error) {
    if (requestedInferenceDevice === "cuda") {
      throw new PolicyOnnxCompatibilityError(
        "CUDA ONNX Runtime execution provider was requested but could not create a CUDA session. " +
        "Install the CUDA 12 ONNX Runtime Node artifact and ensure CUDA/cuDNN shared libraries are visible. " +
        formatErrorCause(error)
      );
    }
  }

  return {
    session: await createSession(sessionFactory, options.onnxPath, "cpu", requestedInferenceDevice),
    runtime: {
      requestedInferenceDevice,
      resolvedInferenceDevice: "cpu",
      executionProvider: "cpu"
    }
  };
}

async function createSession(
  sessionFactory: PolicyOnnxSessionFactory,
  onnxPath: string,
  executionProvider: PolicyOnnxExecutionProvider,
  requestedInferenceDevice: PolicyOnnxInferenceDevice
): Promise<ort.InferenceSession> {
  try {
    return await sessionFactory(onnxPath, { executionProviders: [executionProvider] }) as ort.InferenceSession;
  } catch (error) {
    throw new PolicyOnnxCompatibilityError(
      `Failed to create ONNX Runtime session with executionProvider=${executionProvider} ` +
      `for requestedInferenceDevice=${requestedInferenceDevice}. ${formatErrorCause(error)}`
    );
  }
}

const defaultSessionFactory: PolicyOnnxSessionFactory = async (onnxPath, options) =>
  ort.InferenceSession.create(onnxPath, {
    executionProviders: [...options.executionProviders]
  });

function validateInferenceDevice(value: PolicyOnnxInferenceDevice): void {
  if (value !== "cpu" && value !== "auto" && value !== "cuda") {
    throw new PolicyOnnxCompatibilityError(
      `inferenceDevice must be one of cpu, auto, cuda; got ${String(value)}.`
    );
  }
}

function formatErrorCause(error: unknown): string {
  if (error instanceof Error) {
    return `Cause: ${error.message}`;
  }
  return `Cause: ${String(error)}`;
}

export function selectLegalPolicyAction(
  logits: Float32Array | readonly number[],
  legalPlayMask: ArrayLike<number | boolean>
): number {
  if (logits.length !== CARD_COUNT) {
    throw new PolicyOnnxCompatibilityError(`logits must contain ${CARD_COUNT} values, got ${logits.length}.`);
  }
  validateLegalPlayMask(legalPlayMask);

  let selectedIndex = -1;
  let selectedLogit = -Infinity;

  for (let index = 0; index < CARD_COUNT; index += 1) {
    if (legalPlayMask[index] === 1 || legalPlayMask[index] === true) {
      const logit = Number(logits[index]);
      if (!Number.isFinite(logit)) {
        throw new PolicyOnnxCompatibilityError(`logits[${index}] must be finite.`);
      }
      if (selectedIndex === -1 || logit > selectedLogit) {
        selectedIndex = index;
        selectedLogit = logit;
      }
    }
  }

  return selectedIndex;
}

export function sampleLegalPolicyAction(
  options: SampleLegalPolicyActionOptions
): Omit<PolicyOnnxSampledSelection, "logits"> {
  const distribution = createMaskedCategoricalDistribution(
    options.logits,
    options.legalPlayMask,
    options.temperature ?? 1
  );

  if (distribution.legalCardIndices.length === 1) {
    return {
      selectedCardIndex: distribution.legalCardIndices[0],
      logProbability: 0
    };
  }

  const randomValue = options.rng();
  if (!Number.isFinite(randomValue) || randomValue < 0 || randomValue >= 1) {
    throw new PolicyOnnxCompatibilityError("rng must return a finite value in [0, 1).");
  }

  let cumulativeProbability = 0;

  for (let index = 0; index < distribution.legalCardIndices.length; index += 1) {
    cumulativeProbability += distribution.probabilities[index];

    if (randomValue < cumulativeProbability) {
      return {
        selectedCardIndex: distribution.legalCardIndices[index],
        logProbability: distribution.logProbabilities[index]
      };
    }
  }

  const lastIndex = distribution.legalCardIndices.length - 1;

  return {
    selectedCardIndex: distribution.legalCardIndices[lastIndex],
    logProbability: distribution.logProbabilities[lastIndex]
  };
}

export function calculateLegalPolicyLogProbability(
  options: CalculateLegalPolicyLogProbabilityOptions
): number {
  const distribution = createMaskedCategoricalDistribution(
    options.logits,
    options.legalPlayMask,
    options.temperature ?? 1
  );
  const index = distribution.legalCardIndices.indexOf(options.selectedCardIndex);

  if (index === -1) {
    throw new PolicyOnnxCompatibilityError(
      `selectedCardIndex ${options.selectedCardIndex} is not legal under legalPlayMask.`
    );
  }

  return distribution.logProbabilities[index];
}

export function selectLegalBiddingAction(
  logits: Float32Array | readonly number[],
  legalBidMask: ArrayLike<number | boolean>
): number {
  return selectLegalIndex(logits, legalBidMask, BIDDING_ACTION_COUNT, {
    logitsLabel: "bidding logits",
    maskLabel: "legalBidMask",
    emptyMessage: "legalBidMask must contain at least one legal action."
  });
}

export function selectLegalAdjutantCard(
  logits: Float32Array | readonly number[],
  legalAdjutantMask: ArrayLike<number | boolean>
): number {
  return selectLegalIndex(logits, legalAdjutantMask, CARD_COUNT, {
    logitsLabel: "adjutant logits",
    maskLabel: "legalAdjutantMask",
    emptyMessage: "legalAdjutantMask must contain at least one legal card."
  });
}

export function selectLegalExchangeCard(
  logits: Float32Array | readonly number[],
  legalDiscardMask: ArrayLike<number | boolean>
): number {
  return selectLegalIndex(logits, legalDiscardMask, CARD_COUNT, {
    logitsLabel: "exchange logits",
    maskLabel: "legalDiscardMask",
    emptyMessage: "legalDiscardMask must contain at least one legal card."
  });
}

export function selectLegalExchangeDiscards(
  logits: Float32Array | readonly number[],
  legalDiscardMask: ArrayLike<number | boolean>
): readonly [number, number, number] {
  if (logits.length !== CARD_COUNT) {
    throw new PolicyOnnxCompatibilityError(`exchange logits must contain ${CARD_COUNT} values, got ${logits.length}.`);
  }
  const legalCount = validateMask(legalDiscardMask, CARD_COUNT, "legalDiscardMask");
  if (legalCount < EXCHANGE_DISCARD_COUNT) {
    throw new PolicyOnnxCompatibilityError(
      `legalDiscardMask must contain at least ${EXCHANGE_DISCARD_COUNT} legal cards.`
    );
  }

  const legalCandidates: { index: number; logit: number }[] = [];
  for (let index = 0; index < CARD_COUNT; index += 1) {
    if (isLegalMaskValue(legalDiscardMask[index])) {
      const logit = Number(logits[index]);
      if (!Number.isFinite(logit)) {
        throw new PolicyOnnxCompatibilityError(`exchange logits[${index}] must be finite.`);
      }
      legalCandidates.push({ index, logit });
    }
  }

  const selected = legalCandidates
    .sort((left, right) => right.logit - left.logit || left.index - right.index)
    .slice(0, EXCHANGE_DISCARD_COUNT)
    .map((candidate) => candidate.index)
    .sort((left, right) => left - right);

  if (selected.length !== EXCHANGE_DISCARD_COUNT || new Set(selected).size !== EXCHANGE_DISCARD_COUNT) {
    throw new PolicyOnnxCompatibilityError("exchange selection must contain exactly 3 distinct card indices.");
  }

  return [selected[0], selected[1], selected[2]];
}

export function maskIllegalPolicyLogits(
  logits: Float32Array | readonly number[],
  legalPlayMask: ArrayLike<number | boolean>
): Float32Array {
  if (logits.length !== CARD_COUNT) {
    throw new PolicyOnnxCompatibilityError(`logits must contain ${CARD_COUNT} values, got ${logits.length}.`);
  }
  validateLegalPlayMask(legalPlayMask);

  const masked = new Float32Array(CARD_COUNT);
  for (let index = 0; index < CARD_COUNT; index += 1) {
    if (legalPlayMask[index] === 1 || legalPlayMask[index] === true) {
      const logit = Number(logits[index]);
      if (!Number.isFinite(logit)) {
        throw new PolicyOnnxCompatibilityError(`logits[${index}] must be finite.`);
      }
      masked[index] = logit;
    } else {
      masked[index] = FLOAT32_MIN;
    }
  }

  return masked;
}

function normalizeModelInputForSpec(
  modelInput: Float32Array | readonly number[],
  spec: RuntimeOnnxIoSpec
): Float32Array {
  if (modelInput.length !== spec.modelInputFeatureCount) {
    throw new PolicyOnnxCompatibilityError(
      `modelInput must contain ${spec.modelInputFeatureCount} values for ${spec.policyType} policy, got ${modelInput.length}.`
    );
  }

  const input = modelInput instanceof Float32Array ? new Float32Array(modelInput) : Float32Array.from(modelInput);
  for (let index = 0; index < input.length; index += 1) {
    if (!Number.isFinite(input[index])) {
      throw new PolicyOnnxCompatibilityError(`modelInput[${index}] must be finite.`);
    }
  }

  return input;
}

function validateLegalPlayMask(mask: ArrayLike<number | boolean>): void {
  const legalCount = validateMask(mask, CARD_COUNT, "legalPlayMask");

  if (legalCount === 0) {
    throw new PolicyOnnxCompatibilityError("legalPlayMask must contain at least one legal card.");
  }
}

interface QueuedPolicyOnnxInference {
  input: Float32Array;
  resolve: (logits: Float32Array) => void;
  reject: (error: unknown) => void;
}

class PolicyOnnxInferenceQueue {
  private readonly maxBatchSize: number;
  private readonly queue: QueuedPolicyOnnxInference[] = [];
  private flushScheduled = false;
  private draining = false;
  private stats = createEmptyInferenceStats();

  constructor(
    private readonly session: ort.InferenceSession,
    private readonly spec: RuntimePolicyOnnxSpec,
    maxBatchSize: number
  ) {
    validateInferenceMaxBatchSize(maxBatchSize);
    this.maxBatchSize = maxBatchSize;
  }

  predict(modelInput: Float32Array | readonly number[]): Promise<Float32Array> {
    const input = normalizeModelInputForSpec(modelInput, this.spec);

    if (this.maxBatchSize === 1) {
      return this.runBatch([input]).then((outputs) => outputs[0]);
    }

    return new Promise((resolve, reject) => {
      this.queue.push({ input, resolve, reject });

      if (this.queue.length >= this.maxBatchSize) {
        this.triggerFlush();
      } else {
        this.scheduleFlush();
      }
    });
  }

  predictBatch(
    modelInputs: readonly (Float32Array | readonly number[])[]
  ): Promise<readonly Float32Array[]> {
    const inputs = normalizeModelInputBatchForSpec(modelInputs, this.spec);
    return this.runBatch(inputs);
  }

  getStats(): PolicyOnnxInferenceStats {
    return {
      requestCount: this.stats.requestCount,
      sessionRunCount: this.stats.sessionRunCount,
      meanBatchSize: this.stats.meanBatchSize,
      maxObservedBatchSize: this.stats.maxObservedBatchSize,
      batchSizeHistogram: { ...this.stats.batchSizeHistogram }
    };
  }

  resetStats(): void {
    this.stats = createEmptyInferenceStats();
  }

  private scheduleFlush(): void {
    if (this.flushScheduled || this.draining) {
      return;
    }

    this.flushScheduled = true;
    setImmediate(() => {
      this.flushScheduled = false;
      this.triggerFlush();
    });
  }

  private triggerFlush(): void {
    if (this.draining) {
      return;
    }

    this.draining = true;
    void this.drain().finally(() => {
      this.draining = false;
      if (this.queue.length > 0) {
        this.scheduleFlush();
      }
    });
  }

  private async drain(): Promise<void> {
    while (this.queue.length > 0) {
      const batch = this.queue.splice(0, this.maxBatchSize);
      try {
        const outputs = await this.runBatch(batch.map((item) => item.input));
        outputs.forEach((output, index) => {
          batch[index].resolve(output);
        });
      } catch (error: unknown) {
        batch.forEach((item) => item.reject(error));
      }
    }
  }

  private async runBatch(inputs: readonly Float32Array[]): Promise<readonly Float32Array[]> {
    const outputs = await runPolicyOnnxLogitsBatch(this.session, this.spec, inputs);
    recordInferenceBatch(this.stats, inputs.length);
    return outputs;
  }
}

interface QueuedPolicyCriticOnnxInference {
  input: Float32Array;
  resolve: (value: number) => void;
  reject: (error: unknown) => void;
}

class PolicyCriticOnnxInferenceQueue {
  private readonly maxBatchSize: number;
  private readonly queue: QueuedPolicyCriticOnnxInference[] = [];
  private flushScheduled = false;
  private draining = false;
  private stats = createEmptyInferenceStats();

  constructor(
    private readonly session: ort.InferenceSession,
    private readonly spec: RuntimeCriticOnnxSpec,
    maxBatchSize: number
  ) {
    validateInferenceMaxBatchSize(maxBatchSize);
    this.maxBatchSize = maxBatchSize;
  }

  predict(modelInput: Float32Array | readonly number[]): Promise<number> {
    const input = normalizeModelInputForSpec(modelInput, this.spec);

    if (this.maxBatchSize === 1) {
      return this.runBatch([input]).then((outputs) => outputs[0]);
    }

    return new Promise((resolve, reject) => {
      this.queue.push({ input, resolve, reject });

      if (this.queue.length >= this.maxBatchSize) {
        this.triggerFlush();
      } else {
        this.scheduleFlush();
      }
    });
  }

  predictBatch(modelInputs: readonly (Float32Array | readonly number[])[]): Promise<readonly number[]> {
    const inputs = normalizeModelInputBatchForSpec(modelInputs, this.spec);
    return this.runBatch(inputs);
  }

  getStats(): PolicyOnnxInferenceStats {
    return {
      requestCount: this.stats.requestCount,
      sessionRunCount: this.stats.sessionRunCount,
      meanBatchSize: this.stats.meanBatchSize,
      maxObservedBatchSize: this.stats.maxObservedBatchSize,
      batchSizeHistogram: { ...this.stats.batchSizeHistogram }
    };
  }

  resetStats(): void {
    this.stats = createEmptyInferenceStats();
  }

  private scheduleFlush(): void {
    if (this.flushScheduled || this.draining) {
      return;
    }

    this.flushScheduled = true;
    setImmediate(() => {
      this.flushScheduled = false;
      this.triggerFlush();
    });
  }

  private triggerFlush(): void {
    if (this.draining) {
      return;
    }

    this.draining = true;
    void this.drain().finally(() => {
      this.draining = false;
      if (this.queue.length > 0) {
        this.scheduleFlush();
      }
    });
  }

  private async drain(): Promise<void> {
    while (this.queue.length > 0) {
      const batch = this.queue.splice(0, this.maxBatchSize);
      try {
        const outputs = await this.runBatch(batch.map((item) => item.input));
        outputs.forEach((output, index) => {
          batch[index].resolve(output);
        });
      } catch (error: unknown) {
        batch.forEach((item) => item.reject(error));
      }
    }
  }

  private async runBatch(inputs: readonly Float32Array[]): Promise<readonly number[]> {
    const outputs = await runPolicyCriticOnnxValuesBatch(this.session, this.spec, inputs);
    recordInferenceBatch(this.stats, inputs.length);
    return outputs;
  }
}

function validateInferenceMaxBatchSize(value: number): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new PolicyOnnxCompatibilityError(
      `inferenceMaxBatchSize must be a positive integer, got ${String(value)}.`
    );
  }
}

function createEmptyInferenceStats(): PolicyOnnxInferenceStats {
  return {
    requestCount: 0,
    sessionRunCount: 0,
    meanBatchSize: 0,
    maxObservedBatchSize: 0,
    batchSizeHistogram: {}
  };
}

function recordInferenceBatch(stats: PolicyOnnxInferenceStats, batchSize: number): void {
  stats.requestCount += batchSize;
  stats.sessionRunCount += 1;
  stats.meanBatchSize = stats.requestCount / stats.sessionRunCount;
  stats.maxObservedBatchSize = Math.max(stats.maxObservedBatchSize, batchSize);
  const key = String(batchSize);
  stats.batchSizeHistogram = {
    ...stats.batchSizeHistogram,
    [key]: (stats.batchSizeHistogram[key] ?? 0) + 1
  };
}

function normalizeModelInputBatchForSpec(
  modelInputs: readonly (Float32Array | readonly number[])[],
  spec: RuntimeOnnxIoSpec
): readonly Float32Array[] {
  if (modelInputs.length === 0) {
    throw new PolicyOnnxCompatibilityError("modelInputs batch must contain at least one input.");
  }

  return modelInputs.map((modelInput) => normalizeModelInputForSpec(modelInput, spec));
}

async function runPolicyOnnxLogitsBatch(
  session: ort.InferenceSession,
  spec: RuntimePolicyOnnxSpec,
  inputs: readonly Float32Array[]
): Promise<readonly Float32Array[]> {
  if (inputs.length === 0) {
    throw new PolicyOnnxCompatibilityError("modelInputs batch must contain at least one input.");
  }

  const batchInput = new Float32Array(inputs.length * spec.modelInputFeatureCount);
  inputs.forEach((input, index) => {
    batchInput.set(input, index * spec.modelInputFeatureCount);
  });
  const tensor = new ort.Tensor("float32", batchInput, [inputs.length, spec.modelInputFeatureCount]);

  return session.run({ [spec.inputName]: tensor }, [spec.outputName]).then((outputs) => {
    const outputNames = Object.keys(outputs);
    if (outputNames.length !== 1) {
      throw new PolicyOnnxCompatibilityError(`ONNX Runtime must return one output, got ${outputNames.length}.`);
    }

    const logits = outputs[spec.outputName];
    if (logits === undefined) {
      throw new PolicyOnnxCompatibilityError(`ONNX output ${spec.outputName} is missing.`);
    }
    if (logits.type !== "float32") {
      throw new PolicyOnnxCompatibilityError(`ONNX output dtype mismatch: expected float32, got ${logits.type}.`);
    }
    if (logits.dims.length !== 2 || logits.dims[0] !== inputs.length || logits.dims[1] !== spec.outputLogitCount) {
      throw new PolicyOnnxCompatibilityError(
        `ONNX output shape mismatch: expected [${inputs.length}, ${spec.outputLogitCount}], got ${JSON.stringify(logits.dims)}.`
      );
    }
    const outputData = logits.data;
    if (!(outputData instanceof Float32Array)) {
      throw new PolicyOnnxCompatibilityError("ONNX output data must be Float32Array.");
    }
    if (outputData.length !== inputs.length * spec.outputLogitCount) {
      throw new PolicyOnnxCompatibilityError(
        `ONNX output must contain ${inputs.length * spec.outputLogitCount} values, got ${outputData.length}.`
      );
    }

    return inputs.map((_, index) =>
      new Float32Array(outputData.slice(
        index * spec.outputLogitCount,
        (index + 1) * spec.outputLogitCount
      ))
    );
  });
}

async function runPolicyCriticOnnxValuesBatch(
  session: ort.InferenceSession,
  spec: RuntimeCriticOnnxSpec,
  inputs: readonly Float32Array[]
): Promise<readonly number[]> {
  if (inputs.length === 0) {
    throw new PolicyOnnxCompatibilityError("modelInputs batch must contain at least one input.");
  }

  const batchInput = new Float32Array(inputs.length * spec.modelInputFeatureCount);
  inputs.forEach((input, index) => {
    batchInput.set(input, index * spec.modelInputFeatureCount);
  });
  const tensor = new ort.Tensor("float32", batchInput, [inputs.length, spec.modelInputFeatureCount]);

  return session.run({ [spec.inputName]: tensor }, [spec.outputName]).then((outputs) => {
    const outputNames = Object.keys(outputs);
    if (outputNames.length !== 1) {
      throw new PolicyOnnxCompatibilityError(`ONNX Runtime must return one output, got ${outputNames.length}.`);
    }

    const value = outputs[spec.outputName];
    if (value === undefined) {
      throw new PolicyOnnxCompatibilityError(`ONNX output ${spec.outputName} is missing.`);
    }
    if (value.type !== "float32") {
      throw new PolicyOnnxCompatibilityError(`ONNX output dtype mismatch: expected float32, got ${value.type}.`);
    }
    if (value.dims.length !== 1 || value.dims[0] !== inputs.length) {
      throw new PolicyOnnxCompatibilityError(
        `ONNX output shape mismatch: expected [${inputs.length}], got ${JSON.stringify(value.dims)}.`
      );
    }
    const outputData = value.data;
    if (!(outputData instanceof Float32Array)) {
      throw new PolicyOnnxCompatibilityError("ONNX output data must be Float32Array.");
    }
    if (outputData.length !== inputs.length) {
      throw new PolicyOnnxCompatibilityError(
        `ONNX output must contain ${inputs.length} values, got ${outputData.length}.`
      );
    }

    return Array.from(outputData);
  });
}

function selectLegalIndex(
  logits: Float32Array | readonly number[],
  mask: ArrayLike<number | boolean>,
  count: number,
  labels: { logitsLabel: string; maskLabel: string; emptyMessage: string }
): number {
  if (logits.length !== count) {
    throw new PolicyOnnxCompatibilityError(`${labels.logitsLabel} must contain ${count} values, got ${logits.length}.`);
  }
  const legalCount = validateMask(mask, count, labels.maskLabel);
  if (legalCount === 0) {
    throw new PolicyOnnxCompatibilityError(labels.emptyMessage);
  }

  let selectedIndex = -1;
  let selectedLogit = -Infinity;

  for (let index = 0; index < count; index += 1) {
    if (isLegalMaskValue(mask[index])) {
      const logit = Number(logits[index]);
      if (!Number.isFinite(logit)) {
        throw new PolicyOnnxCompatibilityError(`${labels.logitsLabel}[${index}] must be finite.`);
      }
      if (selectedIndex === -1 || logit > selectedLogit) {
        selectedIndex = index;
        selectedLogit = logit;
      }
    }
  }

  return selectedIndex;
}

function createMaskedCategoricalDistribution(
  logits: Float32Array | readonly number[],
  legalPlayMask: ArrayLike<number | boolean>,
  temperature: number
): {
  legalCardIndices: readonly number[];
  probabilities: readonly number[];
  logProbabilities: readonly number[];
} {
  if (logits.length !== CARD_COUNT) {
    throw new PolicyOnnxCompatibilityError(`logits must contain ${CARD_COUNT} values, got ${logits.length}.`);
  }

  validateLegalPlayMask(legalPlayMask);
  validateTemperature(temperature);

  const legalCardIndices: number[] = [];
  const scaledLogits: number[] = [];

  for (let index = 0; index < CARD_COUNT; index += 1) {
    const logit = Number(logits[index]);

    if (!Number.isFinite(logit)) {
      throw new PolicyOnnxCompatibilityError(`logits[${index}] must be finite.`);
    }

    if (isLegalMaskValue(legalPlayMask[index])) {
      legalCardIndices.push(index);
      scaledLogits.push(logit / temperature);
    }
  }

  if (legalCardIndices.length === 1) {
    return {
      legalCardIndices,
      probabilities: [1],
      logProbabilities: [0]
    };
  }

  const maxScaledLogit = Math.max(...scaledLogits);
  const expValues = scaledLogits.map((logit) => Math.exp(logit - maxScaledLogit));
  const expSum = expValues.reduce((sum, value) => sum + value, 0);

  if (!Number.isFinite(expSum) || expSum <= 0) {
    throw new PolicyOnnxCompatibilityError("masked categorical softmax normalization failed.");
  }

  const logDenominator = maxScaledLogit + Math.log(expSum);
  const probabilities = expValues.map((value) => value / expSum);
  const logProbabilities = scaledLogits.map((logit) => logit - logDenominator);

  for (let index = 0; index < probabilities.length; index += 1) {
    if (
      !Number.isFinite(probabilities[index]) ||
      probabilities[index] < 0 ||
      !Number.isFinite(logProbabilities[index]) ||
      logProbabilities[index] > 1e-12
    ) {
      throw new PolicyOnnxCompatibilityError("masked categorical distribution contains an invalid probability.");
    }
  }

  return {
    legalCardIndices,
    probabilities,
    logProbabilities
  };
}

function validateTemperature(temperature: number): void {
  if (!Number.isFinite(temperature) || temperature <= 0) {
    throw new PolicyOnnxCompatibilityError("temperature must be finite and greater than 0.");
  }
}

function validateMask(mask: ArrayLike<number | boolean>, count: number, label: string): number {
  if (mask.length !== count) {
    throw new PolicyOnnxCompatibilityError(`${label} must contain ${count} values, got ${mask.length}.`);
  }

  let legalCount = 0;
  for (let index = 0; index < count; index += 1) {
    const value = mask[index];
    if (value !== 0 && value !== 1 && value !== false && value !== true) {
      throw new PolicyOnnxCompatibilityError(`${label}[${index}] must be 0/1 or boolean.`);
    }
    if (isLegalMaskValue(value)) {
      legalCount += 1;
    }
  }

  return legalCount;
}

function isLegalMaskValue(value: number | boolean): boolean {
  return value === 1 || value === true;
}

function validateSessionNames(session: ort.InferenceSession, spec: RuntimeOnnxIoSpec): void {
  if (!sameNames(session.inputNames, [spec.inputName])) {
    throw new PolicyOnnxCompatibilityError(
      `ONNX Runtime input names mismatch: expected ${spec.inputName}, got ${session.inputNames.join(", ")}.`
    );
  }
  if (!sameNames(session.outputNames, [spec.outputName])) {
    throw new PolicyOnnxCompatibilityError(
      `ONNX Runtime output names mismatch: expected ${spec.outputName}, got ${session.outputNames.join(", ")}.`
    );
  }
}

function sameNames(actual: readonly string[], expected: readonly string[]): boolean {
  return actual.length === expected.length && actual.every((name, index) => name === expected[index]);
}
