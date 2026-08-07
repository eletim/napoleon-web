import { readFile } from "node:fs/promises";
import * as ort from "onnxruntime-node";
import {
  CARD_COUNT,
  MODEL_INPUT_FEATURE_COUNT,
  ONNX_INPUT_NAME,
  ONNX_OUTPUT_NAME
} from "./constants.js";
import { PolicyOnnxCompatibilityError } from "./errors.js";
import { parsePolicyOnnxMetadata } from "./metadata.js";
import { validateOnnxModelIo } from "./onnxProto.js";
import type {
  PolicyOnnxLoadOptions,
  PolicyOnnxMetadata,
  PolicyOnnxSelection,
  SelectLegalPlayInput
} from "./types.js";

const FLOAT32_MIN = -3.4028234663852886e38;

export class PolicyOnnxModel {
  constructor(
    readonly metadata: PolicyOnnxMetadata,
    private readonly session: ort.InferenceSession
  ) {}

  async predictLogits(modelInput: Float32Array | readonly number[]): Promise<Float32Array> {
    const input = normalizeModelInput(modelInput);
    const tensor = new ort.Tensor("float32", input, [1, MODEL_INPUT_FEATURE_COUNT]);
    const outputs = await this.session.run({ [ONNX_INPUT_NAME]: tensor }, [ONNX_OUTPUT_NAME]);
    const logits = outputs[ONNX_OUTPUT_NAME];

    if (logits === undefined) {
      throw new PolicyOnnxCompatibilityError(`ONNX output ${ONNX_OUTPUT_NAME} is missing.`);
    }
    if (logits.type !== "float32") {
      throw new PolicyOnnxCompatibilityError(`ONNX output dtype mismatch: expected float32, got ${logits.type}.`);
    }
    if (logits.dims.length !== 2 || logits.dims[0] !== 1 || logits.dims[1] !== CARD_COUNT) {
      throw new PolicyOnnxCompatibilityError(
        `ONNX output shape mismatch: expected [1, ${CARD_COUNT}], got ${JSON.stringify(logits.dims)}.`
      );
    }
    if (!(logits.data instanceof Float32Array)) {
      throw new PolicyOnnxCompatibilityError("ONNX output data must be Float32Array.");
    }

    return new Float32Array(logits.data);
  }

  async selectLegalPlay(input: SelectLegalPlayInput): Promise<PolicyOnnxSelection> {
    const logits = await this.predictLogits(input.modelInput);
    const selectedCardIndex = selectLegalPolicyAction(logits, input.legalPlayMask);

    return {
      selectedCardIndex,
      logits
    };
  }
}

export async function loadPolicyOnnxModel(options: PolicyOnnxLoadOptions): Promise<PolicyOnnxModel> {
  const metadata = parsePolicyOnnxMetadata(await readFile(options.metadataPath, "utf8"));
  await validateOnnxModelIo(options.onnxPath, metadata);

  const session = await ort.InferenceSession.create(options.onnxPath, {
    executionProviders: ["cpu"]
  });

  if (!sameNames(session.inputNames, [ONNX_INPUT_NAME])) {
    throw new PolicyOnnxCompatibilityError(
      `ONNX Runtime input names mismatch: expected ${ONNX_INPUT_NAME}, got ${session.inputNames.join(", ")}.`
    );
  }
  if (!sameNames(session.outputNames, [ONNX_OUTPUT_NAME])) {
    throw new PolicyOnnxCompatibilityError(
      `ONNX Runtime output names mismatch: expected ${ONNX_OUTPUT_NAME}, got ${session.outputNames.join(", ")}.`
    );
  }

  return new PolicyOnnxModel(metadata, session);
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

function normalizeModelInput(modelInput: Float32Array | readonly number[]): Float32Array {
  if (modelInput.length !== MODEL_INPUT_FEATURE_COUNT) {
    throw new PolicyOnnxCompatibilityError(
      `modelInput must contain ${MODEL_INPUT_FEATURE_COUNT} values, got ${modelInput.length}.`
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
  if (mask.length !== CARD_COUNT) {
    throw new PolicyOnnxCompatibilityError(`legalPlayMask must contain ${CARD_COUNT} values, got ${mask.length}.`);
  }

  let legalCount = 0;
  for (let index = 0; index < CARD_COUNT; index += 1) {
    const value = mask[index];
    if (value !== 0 && value !== 1 && value !== false && value !== true) {
      throw new PolicyOnnxCompatibilityError(`legalPlayMask[${index}] must be 0/1 or boolean.`);
    }
    if (value === 1 || value === true) {
      legalCount += 1;
    }
  }

  if (legalCount === 0) {
    throw new PolicyOnnxCompatibilityError("legalPlayMask must contain at least one legal card.");
  }
}

function sameNames(actual: readonly string[], expected: readonly string[]): boolean {
  return actual.length === expected.length && actual.every((name, index) => name === expected[index]);
}
