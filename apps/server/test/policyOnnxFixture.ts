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

function createConstantPolicyOnnx(logits: Float32Array): Uint8Array {
  const tensor = message(
    fieldVarint(1, 1),
    fieldVarint(1, CARD_COUNT),
    fieldVarint(2, 1),
    fieldString(8, "constant_logits"),
    fieldBytes(9, new Uint8Array(logits.buffer, logits.byteOffset, logits.byteLength))
  );
  const attribute = message(
    fieldString(1, "value"),
    fieldBytes(5, tensor),
    fieldVarint(20, 4)
  );
  const node = message(
    fieldString(2, ONNX_OUTPUT_NAME),
    fieldString(3, "constant_logits_node"),
    fieldString(4, "Constant"),
    fieldBytes(5, attribute)
  );
  const graph = message(
    fieldBytes(1, node),
    fieldString(2, "policy_graph"),
    fieldBytes(11, valueInfo(ONNX_INPUT_NAME, ["batch", MODEL_INPUT_FEATURE_COUNT])),
    fieldBytes(12, valueInfo(ONNX_OUTPUT_NAME, ["batch", CARD_COUNT]))
  );
  const opset = message(fieldVarint(2, ONNX_OPSET_VERSION));

  return message(
    fieldVarint(1, 8),
    fieldString(2, "policy-onnx-test"),
    fieldBytes(7, graph),
    fieldBytes(8, opset)
  );
}

function valueInfo(name: string, shape: readonly (string | number)[]): Uint8Array {
  const tensorShape = message(...shape.map((dimension) => fieldBytes(1, dimensionMessage(dimension))));
  const tensorType = message(fieldVarint(1, 1), fieldBytes(2, tensorShape));
  const type = message(fieldBytes(1, tensorType));

  return message(fieldString(1, name), fieldBytes(2, type));
}

function dimensionMessage(dimension: string | number): Uint8Array {
  if (typeof dimension === "string") {
    return message(fieldString(2, dimension));
  }

  return message(fieldVarint(1, dimension));
}

function fieldVarint(fieldNumber: number, value: number): Uint8Array {
  return concat(varint(fieldNumber * 8), varint(value));
}

function fieldString(fieldNumber: number, value: string): Uint8Array {
  return fieldBytes(fieldNumber, new TextEncoder().encode(value));
}

function fieldBytes(fieldNumber: number, value: Uint8Array): Uint8Array {
  return concat(varint(fieldNumber * 8 + 2), varint(value.length), value);
}

function message(...fields: Uint8Array[]): Uint8Array {
  return concat(...fields);
}

function varint(value: number): Uint8Array {
  const bytes: number[] = [];
  let remaining = value;

  while (remaining >= 0x80) {
    bytes.push((remaining & 0x7f) | 0x80);
    remaining = Math.floor(remaining / 0x80);
  }
  bytes.push(remaining);

  return Uint8Array.from(bytes);
}

function concat(...arrays: Uint8Array[]): Uint8Array {
  const length = arrays.reduce((sum, array) => sum + array.length, 0);
  const result = new Uint8Array(length);
  let offset = 0;

  for (const array of arrays) {
    result.set(array, offset);
    offset += array.length;
  }

  return result;
}
