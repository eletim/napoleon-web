# @napoleon/policy-onnx

TypeScript inference wrapper for the policy ONNX artifact exported by
`python/src/napoleon_ml/policy/onnx_export.py`.

The package loads a `.onnx` file plus its `.json` metadata, validates the
runtime contract before inference, then runs CPU ONNX Runtime from Node.js.

```ts
import { PolicyOnnxAgent, loadPolicyOnnxModel } from "@napoleon/policy-onnx";
import { runAutomatedGame } from "@napoleon/ai";

const policy = await loadPolicyOnnxModel({
  onnxPath: "./artifacts/policy.onnx",
  metadataPath: "./artifacts/policy.json"
});

const logits = await policy.predictLogits(modelInput);
const action = await policy.selectLegalPlay({
  modelInput,
  legalPlayMask
});

const record = await runAutomatedGame({
  seed: 12345,
  createAgent: ({ rng }) => new PolicyOnnxAgent({ policy, rng })
});
```

`modelInput` must be a 6242-element `float32` feature vector using model input
schema version 1. `legalPlayMask` must contain 53 entries and at least one
legal card. `selectLegalPlay` always applies the mask before choosing the
highest-logit card index.

`PolicyOnnxAgent` uses the ONNX policy only during the `playing` phase. Bidding,
card exchange, and adjutant selection are delegated to the existing
`RuleBasedAgent`. The agent builds the existing encoded playing observation,
`model_input`, and `legalPlayMask` from the current player observation plus
`runAutomatedGame`'s public bidding action history. During play, missing bidding
history, schema/hash drift, shape mismatch, or inference failure is treated as an
error and is not silently converted into a RuleBased play.

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

To smoke-test an externally trained artifact without committing it, set both
paths before running this package's tests:

```sh
NAPOLEON_POLICY_ONNX_PATH=/path/to/policy.onnx \
NAPOLEON_POLICY_METADATA_PATH=/path/to/policy.json \
pnpm --filter @napoleon/policy-onnx test
```

## RuleBased comparison evaluation

`runPolicyVsRuleBasedEvaluation` connects the loaded ONNX playing policy to the
existing `@napoleon/ai` evaluation runner. The scheduled matchup uses one
`PolicyOnnxAgent` and four `RuleBasedAgent` seats, then rotates the policy
through all five seats by default with fixed seeds and fixed options.

```ts
import {
  loadPolicyOnnxModel,
  runPolicyVsRuleBasedEvaluation
} from "@napoleon/policy-onnx";

const policy = await loadPolicyOnnxModel({
  onnxPath: "/path/to/policy.onnx",
  metadataPath: "/path/to/policy.json"
});

const result = await runPolicyVsRuleBasedEvaluation({
  policy,
  startSeed: 900,
  gameCount: 10
});

console.log(result.comparison.illegalActionCount);
console.log(result.comparison.failedGames);
console.log(result.comparison.policy.comparison.winRateDeltaConfidenceInterval);
```

The result keeps the raw evaluation run, the existing detailed evaluation
report, and a policy-vs-rulebased grouped comparison. The comparison includes
win rate, contract success rate, role summaries, seat summaries, failed games,
illegal-action failure count, and 95% confidence intervals for rate and
point-card deltas. Failed games remain in the output instead of being excluded
from denominators.
