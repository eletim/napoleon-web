# @napoleon/policy-onnx

TypeScript inference wrapper for the policy ONNX artifact exported by
`python/src/napoleon_ml/policy/onnx_export.py`.

The package loads a `.onnx` file plus its `.json` metadata, validates the
runtime contract before inference, then runs CPU ONNX Runtime from Node.js.
It deliberately does not connect the model to the game Agent layer.

```ts
import { loadPolicyOnnxModel } from "@napoleon/policy-onnx";

const policy = await loadPolicyOnnxModel({
  onnxPath: "./artifacts/policy.onnx",
  metadataPath: "./artifacts/policy.json"
});

const logits = await policy.predictLogits(modelInput);
const action = await policy.selectLegalPlay({
  modelInput,
  legalPlayMask
});
```

`modelInput` must be a 6242-element `float32` feature vector using model input
schema version 1. `legalPlayMask` must contain 53 entries and at least one
legal card. `selectLegalPlay` always applies the mask before choosing the
highest-logit card index.

The loader rejects artifacts whose metadata or ONNX graph disagrees with the
expected contract:

- metadata schema version 1
- dataset schema version 1
- playing encoder schema version 1
- model input schema version 1
- current `CARD_IDS` SHA-256
- ONNX input `model_input`, shape `["batch", 6242]`, dtype `float32`
- ONNX output `logits`, shape `["batch", 53]`, dtype `float32`

The tests create a temporary ONNX model at runtime and compare a fixed sample's
logits and masked selection against expected ONNX-side values without committing
an ONNX model file.
