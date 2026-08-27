import { readFile } from "node:fs/promises";
import {
  ONNX_TENSOR_FLOAT_TYPE
} from "./constants.js";
import { PolicyOnnxCompatibilityError } from "./errors.js";
import type { PolicyOnnxIoMetadata, PolicyOnnxMetadata } from "./types.js";
import { PLAYING_POLICY_ONNX_SPEC, type RuntimeOnnxIoSpec } from "./policySpecs.js";

const MODEL_GRAPH_FIELD = 7;
const GRAPH_INPUT_FIELD = 11;
const GRAPH_OUTPUT_FIELD = 12;
const VALUE_INFO_NAME_FIELD = 1;
const VALUE_INFO_TYPE_FIELD = 2;
const TYPE_PROTO_TENSOR_FIELD = 1;
const TENSOR_TYPE_ELEM_TYPE_FIELD = 1;
const TENSOR_TYPE_SHAPE_FIELD = 2;
const TENSOR_SHAPE_DIM_FIELD = 1;
const DIM_VALUE_FIELD = 1;
const DIM_PARAM_FIELD = 2;
const ONNX_FLOAT_ELEM_TYPE = 1;

export async function validateOnnxModelIo(
  onnxPath: string,
  metadata: Pick<PolicyOnnxMetadata, "onnx">,
  spec: RuntimeOnnxIoSpec = PLAYING_POLICY_ONNX_SPEC
): Promise<void> {
  const bytes = await readFile(onnxPath);
  const modelIo = parseOnnxModelIo(bytes);

  validateSingleIo("input", modelIo.inputs, metadata.onnx.inputs[0], {
    name: spec.inputName,
    shape: spec.inputShape,
    dtype: ONNX_TENSOR_FLOAT_TYPE
  });
  validateSingleIo("output", modelIo.outputs, metadata.onnx.outputs[0], {
    name: spec.outputName,
    shape: spec.outputShape,
    dtype: ONNX_TENSOR_FLOAT_TYPE
  });
}

export async function validateOnnxModelExactIo(
  onnxPath: string,
  expected: {
    inputs: readonly ExpectedRuntimeIo[];
    outputs: readonly ExpectedRuntimeIo[];
  }
): Promise<void> {
  const bytes = await readFile(onnxPath);
  const modelIo = parseOnnxModelIo(bytes);
  validateExactIoList("input", modelIo.inputs, expected.inputs);
  validateExactIoList("output", modelIo.outputs, expected.outputs);
}

export interface ParsedOnnxModelIo {
  inputs: readonly PolicyOnnxIoMetadata[];
  outputs: readonly PolicyOnnxIoMetadata[];
}

export function parseOnnxModelIo(bytes: Uint8Array): ParsedOnnxModelIo {
  const modelFields = readFields(bytes);
  const graphBytes = modelFields.find((field) => field.fieldNumber === MODEL_GRAPH_FIELD)?.bytes;

  if (graphBytes === undefined) {
    throw new PolicyOnnxCompatibilityError("ONNX model graph is missing.");
  }

  const graphFields = readFields(graphBytes);

  return {
    inputs: graphFields
      .filter((field) => field.fieldNumber === GRAPH_INPUT_FIELD && field.bytes !== undefined)
      .map((field) => parseValueInfo(requireBytes(field, "ONNX graph input"))),
    outputs: graphFields
      .filter((field) => field.fieldNumber === GRAPH_OUTPUT_FIELD && field.bytes !== undefined)
      .map((field) => parseValueInfo(requireBytes(field, "ONNX graph output")))
  };
}

interface ExpectedRuntimeIo {
  name: string;
  shape: readonly (string | number)[];
  dtype: string;
}

function validateSingleIo(
  label: string,
  actualList: readonly PolicyOnnxIoMetadata[],
  metadata: PolicyOnnxIoMetadata | undefined,
  expected: ExpectedRuntimeIo
): void {
  if (metadata === undefined) {
    throw new PolicyOnnxCompatibilityError(`metadata ONNX ${label} is missing.`);
  }
  if (actualList.length !== 1) {
    throw new PolicyOnnxCompatibilityError(`ONNX model must have one ${label}, got ${actualList.length}.`);
  }

  const actual = actualList[0];
  if (actual.name !== metadata.name || actual.name !== expected.name) {
    throw new PolicyOnnxCompatibilityError(
      `ONNX ${label} name mismatch: expected ${expected.name}, metadata has ${metadata.name}, model has ${actual.name}.`
    );
  }
  if (actual.dtype !== expected.dtype) {
    throw new PolicyOnnxCompatibilityError(
      `ONNX ${label} dtype mismatch: expected ${expected.dtype}, got ${actual.dtype}.`
    );
  }
  if (!sameRuntimeShape(actual.shape, expected.shape)) {
    throw new PolicyOnnxCompatibilityError(
      `ONNX ${label} shape mismatch: expected ${JSON.stringify(expected.shape)}, got ${JSON.stringify(actual.shape)}.`
    );
  }
}

function validateExactIoList(
  label: string,
  actualList: readonly PolicyOnnxIoMetadata[],
  expectedList: readonly ExpectedRuntimeIo[]
): void {
  if (actualList.length !== expectedList.length) {
    throw new PolicyOnnxCompatibilityError(
      `ONNX model must have ${expectedList.length} ${label}s, got ${actualList.length}.`
    );
  }

  for (let index = 0; index < expectedList.length; index += 1) {
    const actual = actualList[index];
    const expected = expectedList[index];
    if (actual.name !== expected.name) {
      throw new PolicyOnnxCompatibilityError(
        `ONNX ${label} ${index} name mismatch: expected ${expected.name}, got ${actual.name}.`
      );
    }
    if (actual.dtype !== expected.dtype) {
      throw new PolicyOnnxCompatibilityError(
        `ONNX ${label} ${index} dtype mismatch: expected ${expected.dtype}, got ${actual.dtype}.`
      );
    }
    if (!sameRuntimeShape(actual.shape, expected.shape)) {
      throw new PolicyOnnxCompatibilityError(
        `ONNX ${label} ${index} shape mismatch: expected ${JSON.stringify(expected.shape)}, got ${JSON.stringify(actual.shape)}.`
      );
    }
  }
}

function parseValueInfo(bytes: Uint8Array): PolicyOnnxIoMetadata {
  const fields = readFields(bytes);
  const name = fields.find((field) => field.fieldNumber === VALUE_INFO_NAME_FIELD)?.text;
  const typeBytes = fields.find((field) => field.fieldNumber === VALUE_INFO_TYPE_FIELD)?.bytes;

  if (name === undefined || typeBytes === undefined) {
    throw new PolicyOnnxCompatibilityError("ONNX value info is missing name or type.");
  }

  const tensorBytes = readFields(typeBytes).find(
    (field) => field.fieldNumber === TYPE_PROTO_TENSOR_FIELD
  )?.bytes;
  if (tensorBytes === undefined) {
    throw new PolicyOnnxCompatibilityError(`ONNX value ${name} must be a tensor.`);
  }

  const tensorFields = readFields(tensorBytes);
  const elemType = tensorFields.find((field) => field.fieldNumber === TENSOR_TYPE_ELEM_TYPE_FIELD)?.number;
  const shapeBytes = tensorFields.find((field) => field.fieldNumber === TENSOR_TYPE_SHAPE_FIELD)?.bytes;

  if (elemType !== ONNX_FLOAT_ELEM_TYPE) {
    throw new PolicyOnnxCompatibilityError(`ONNX value ${name} dtype mismatch: expected tensor(float).`);
  }
  if (shapeBytes === undefined) {
    throw new PolicyOnnxCompatibilityError(`ONNX value ${name} shape is missing.`);
  }

  return {
    name,
    dtype: ONNX_TENSOR_FLOAT_TYPE,
    shape: readFields(shapeBytes)
      .filter((field) => field.fieldNumber === TENSOR_SHAPE_DIM_FIELD && field.bytes !== undefined)
      .map((field) => parseDimension(requireBytes(field, `ONNX value ${name} dimension`)))
  };
}

function parseDimension(bytes: Uint8Array): string | number {
  const fields = readFields(bytes);
  const value = fields.find((field) => field.fieldNumber === DIM_VALUE_FIELD)?.number;
  const param = fields.find((field) => field.fieldNumber === DIM_PARAM_FIELD)?.text;

  if (param !== undefined) {
    return param;
  }
  if (value !== undefined) {
    return value;
  }

  throw new PolicyOnnxCompatibilityError("ONNX dimension must use dim_value or dim_param.");
}

function sameRuntimeShape(actual: readonly (string | number)[], expected: readonly (string | number)[]): boolean {
  if (actual.length !== expected.length) {
    return false;
  }

  return actual.every((value, index) => {
    const expectedValue = expected[index];
    if (typeof expectedValue === "string") {
      return typeof value === "string" && value.length > 0;
    }

    return value === expectedValue;
  });
}

interface ProtoField {
  fieldNumber: number;
  wireType: number;
  number?: number;
  bytes?: Uint8Array;
  text?: string;
}

function readFields(bytes: Uint8Array): ProtoField[] {
  const reader = new ProtoReader(bytes);
  const fields: ProtoField[] = [];

  while (!reader.done()) {
    const tag = reader.varint();
    const fieldNumber = Math.floor(tag / 8);
    const wireType = tag % 8;

    if (wireType === 0) {
      fields.push({ fieldNumber, wireType, number: reader.varint() });
    } else if (wireType === 1) {
      reader.skip(8);
      fields.push({ fieldNumber, wireType });
    } else if (wireType === 2) {
      const fieldBytes = reader.readBytes();
      fields.push({
        fieldNumber,
        wireType,
        bytes: fieldBytes,
        text: utf8IfValid(fieldBytes)
      });
    } else if (wireType === 5) {
      reader.skip(4);
      fields.push({ fieldNumber, wireType });
    } else {
      throw new PolicyOnnxCompatibilityError(`Unsupported ONNX protobuf wire type: ${wireType}.`);
    }
  }

  return fields;
}

function utf8IfValid(bytes: Uint8Array): string | undefined {
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    return undefined;
  }
}

class ProtoReader {
  private offset = 0;

  constructor(private readonly buffer: Uint8Array) {}

  done(): boolean {
    return this.offset >= this.buffer.length;
  }

  varint(): number {
    let shift = 0;
    let value = 0;

    while (this.offset < this.buffer.length) {
      const byte = this.buffer[this.offset];
      this.offset += 1;
      value += (byte & 0x7f) * 2 ** shift;
      if ((byte & 0x80) === 0) {
        if (!Number.isSafeInteger(value)) {
          throw new PolicyOnnxCompatibilityError("ONNX protobuf integer exceeds safe range.");
        }
        return value;
      }
      shift += 7;
    }

    throw new PolicyOnnxCompatibilityError("ONNX protobuf varint is truncated.");
  }

  readBytes(): Uint8Array {
    const length = this.varint();
    if (this.offset + length > this.buffer.length) {
      throw new PolicyOnnxCompatibilityError("ONNX protobuf length-delimited field is truncated.");
    }

    const start = this.offset;
    this.offset += length;
    return this.buffer.subarray(start, this.offset);
  }

  skip(length: number): void {
    if (this.offset + length > this.buffer.length) {
      throw new PolicyOnnxCompatibilityError("ONNX protobuf field is truncated.");
    }
    this.offset += length;
  }
}

function requireBytes(field: ProtoField, label: string): Uint8Array {
  if (field.bytes === undefined) {
    throw new PolicyOnnxCompatibilityError(`${label} is missing protobuf payload.`);
  }

  return field.bytes;
}
