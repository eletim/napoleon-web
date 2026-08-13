# @napoleon/policy-onnx

TypeScript inference wrapper for the policy ONNX artifact exported by
`python/src/napoleon_ml/policy/onnx_export.py`.

The package loads a `.onnx` file plus its `.json` metadata, validates the
runtime contract before inference, then runs ONNX Runtime from Node.js.

```ts
import {
  PolicyOnnxAgent,
  loadNonPlayingPolicyOnnxModel,
  loadPolicyOnnxModel
} from "@napoleon/policy-onnx";
import { runAutomatedGame } from "@napoleon/ai";

const policy = await loadPolicyOnnxModel({
  onnxPath: "./artifacts/policy.onnx",
  metadataPath: "./artifacts/policy.json",
  inferenceDevice: "cpu"
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

`inferenceDevice` accepts:

- `cpu`: create a CPU Execution Provider session only.
- `auto`: try CUDA first, then fall back to CPU if CUDA session creation fails.
- `cuda`: require CUDA Execution Provider and fail fast if it cannot be used.

The default is `cpu` for existing CI and development compatibility. Loaded
models expose `policy.runtime.requestedInferenceDevice`,
`policy.runtime.resolvedInferenceDevice`, and `policy.runtime.executionProvider`
so callers can verify that an explicit CUDA request did not silently run on CPU.

`modelInput` must be a 6246-element `float32` feature vector using model input
schema version 2. `legalPlayMask` must contain 53 entries and at least one
legal card. `selectLegalPlay` always applies the mask before choosing the
highest-logit card index.

`new PolicyOnnxAgent({ policy })` is the backward-compatible playing-only
configuration. It uses ONNX for `playing`, while `bidding`,
`choosing-adjutant`, and `exchanging` fall back to the existing
`RuleBasedAgent`.

To run every phase through ONNX, load each phase artifact and assign it to the
matching slot:

```ts
const biddingPolicy = await loadNonPlayingPolicyOnnxModel({
  onnxPath: "./artifacts/bidding.onnx",
  metadataPath: "./artifacts/bidding.json"
});
const adjutantPolicy = await loadNonPlayingPolicyOnnxModel({
  onnxPath: "./artifacts/adjutant.onnx",
  metadataPath: "./artifacts/adjutant.json"
});
const exchangePolicy = await loadNonPlayingPolicyOnnxModel({
  onnxPath: "./artifacts/exchange.onnx",
  metadataPath: "./artifacts/exchange.json"
});

const fullPolicyAgent = new PolicyOnnxAgent({
  policy,
  biddingPolicy,
  adjutantPolicy,
  exchangePolicy
});
```

Each non-playing slot is optional. If, for example, only `biddingPolicy` is
provided, bidding uses ONNX and the other non-playing phases fall back to
`RuleBasedAgent`; playing continues to use `policy`. The constructor rejects
non-playing artifacts assigned to the wrong slot, such as an exchange artifact in
`biddingPolicy`.

The phase-specific live input helpers are exported for smoke tests and parity
checks:

```ts
import {
  createPolicyOnnxBiddingInput,
  createPolicyOnnxAdjutantInput,
  createPolicyOnnxExchangeInput,
  createPolicyOnnxPlayInput
} from "@napoleon/policy-onnx";
```

These helpers build model input only from `PlayerObservation` and the public
bidding action history. Missing `publicActionHistory`, schema/hash drift, shape
mismatch, illegal ONNX selections, or inference failure is treated as an error
and is not silently converted into a RuleBased decision for that phase.

The loader rejects artifacts whose metadata or ONNX graph disagrees with the
expected contract:

- metadata schema version 1
- dataset schema version 1
- playing encoder schema version 2
- model input schema version 2
- current `CARD_IDS` SHA-256
- ONNX input `model_input`, shape `["batch", 6246]`, dtype `float32`
- ONNX output `logits`, shape `["batch", 53]`, dtype `float32`

The tests create a temporary ONNX model at runtime and compare a fixed sample's
logits and masked selection against expected ONNX-side values without committing
an ONNX model file.

## CUDA 12 Node runtime

`onnxruntime-node` is pinned to `1.26.0` so fresh installs stay on the CUDA 12
Node artifact line. Do not upgrade this package to a CUDA 13-only release while
the Python training environment is pinned to CUDA 12.

The npm package install script downloads the Linux x64 CUDA EP artifact into
`node_modules`; no manual binary copying is required. On CUDA machines, make
sure CUDA and cuDNN shared libraries are visible to Node, usually through
`LD_LIBRARY_PATH`. If `inferenceDevice: "cuda"` cannot load CUDA/cuDNN libraries,
the loader fails with the underlying ONNX Runtime error included in the message.

CUDA 12 smoke commands:

```sh
pnpm install
pnpm --filter @napoleon/self-play-cli... build

node apps/self-play-cli/dist/policyEvaluationCli.js \
  --onnx /path/to/policy.onnx \
  --metadata /path/to/policy.json \
  --output /tmp/napoleon-policy-eval.json \
  --start-seed 900 \
  --seed-count 2 \
  --benchmark standard \
  --inference-device cuda

node apps/self-play-cli/dist/playingSelfPlayCli.js \
  --onnx /path/to/policy.onnx \
  --metadata /path/to/policy.json \
  --output /tmp/napoleon-selfplay-cuda \
  --start-seed 1000 \
  --games 2 \
  --games-per-shard 1 \
  --rollout-workers 2 \
  --inference-device cuda
```

For RL orchestration:

```sh
napoleon-run-playing-rl \
  --run-directory /tmp/napoleon-rl-cuda \
  --initial-checkpoint /path/to/initial-playing.pt \
  --supervised-dataset /path/to/supervised-dataset \
  --iterations 1 \
  --games-per-iteration 2 \
  --games-per-shard 1 \
  --device cuda \
  --inference-device cuda
```

Confirm the JSON outputs contain `resolvedInferenceDevice: "cuda"` and
`executionProvider: "cuda"`, and watch `nvidia-smi` while the Node commands are
running.

To smoke-test an externally trained artifact without committing it, set both
paths before running this package's tests:

```sh
NAPOLEON_POLICY_ONNX_PATH=/path/to/policy.onnx \
NAPOLEON_POLICY_METADATA_PATH=/path/to/policy.json \
pnpm --filter @napoleon/policy-onnx test
```

These package-level smoke-test variables are separate from the normal web/server
AI selection configuration. For local app startup, copy the repository root
`.env.sample` to `.env` and configure `NAPOLEON_POLICY_1_DISPLAY_NAME`,
`NAPOLEON_POLICY_1_ONNX_PATH`, and `NAPOLEON_POLICY_1_METADATA_PATH` through slot
5 as needed.

To smoke-test supplied artifacts for all four phases, also set:

```sh
NAPOLEON_BIDDING_POLICY_ONNX_PATH=/path/to/bidding.onnx \
NAPOLEON_BIDDING_POLICY_METADATA_PATH=/path/to/bidding.json \
NAPOLEON_ADJUTANT_POLICY_ONNX_PATH=/path/to/adjutant.onnx \
NAPOLEON_ADJUTANT_POLICY_METADATA_PATH=/path/to/adjutant.json \
NAPOLEON_EXCHANGE_POLICY_ONNX_PATH=/path/to/exchange.onnx \
NAPOLEON_EXCHANGE_POLICY_METADATA_PATH=/path/to/exchange.json \
NAPOLEON_POLICY_ONNX_PATH=/path/to/playing.onnx \
NAPOLEON_POLICY_METADATA_PATH=/path/to/playing.json \
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

`runPolicyVsRuleBasedEvaluation` is intentionally playing-only: the
`PolicyOnnxAgent` uses ONNX for playing decisions and keeps bidding,
adjutant selection, and exchange on the `RuleBasedAgent` fallback path.

For fixed-opponent benchmark evaluation, use
`runPlayingPolicyRosterEvaluation` or `runStandardPlayingPolicyBenchmarks`.
The candidate remains source agent index 0 and is rotated through all five
seats. Mixed opponent rosters use deterministic opponent source-agent orders so
each opponent type is balanced across candidate-relative seats for the same
seed set.

```ts
import {
  PPO_SEPARATED_V1000_BENCHMARK_POLICY_ID,
  RL_V740_BENCHMARK_POLICY_ID,
  loadRepoManagedPlayingPolicyBenchmark,
  runStandardPlayingPolicyBenchmarks
} from "@napoleon/policy-onnx";

const rlV740 = await loadRepoManagedPlayingPolicyBenchmark(RL_V740_BENCHMARK_POLICY_ID);
console.log(rlV740.artifact.onnxSha256);

const ppoV1000 = await loadRepoManagedPlayingPolicyBenchmark(PPO_SEPARATED_V1000_BENCHMARK_POLICY_ID);
console.log(ppoV1000.artifact.checkpointSha256);

const suite = await runStandardPlayingPolicyBenchmarks({
  candidatePolicy: policy,
  startSeed: 900,
  gameCount: 10
});
```

The built-in minimum standard suite evaluates:

- `rule-based-x4`
- `rl-v740-x4`
- `rule-based-x2-rl-v740-x2`

Repo-managed frozen playing policy artifacts live under
`benchmarks/playing-policies`:

- `rl-v740` records the source v1 artifact, v1-to-v2 logit-preserving
  migration, committed hashes, and parity results.
- `ppo-separated-v1000` records the completed PPO separated v1000 run,
  Actor-only ONNX runtime artifact, Critic-only ONNX runtime artifact, and
  full Actor/Critic checkpoint.

The loader validates committed ONNX and metadata SHA-256 values before runtime
loading. The Critic runtime returns a raw playing value, not a calibrated
probability; the simple EV bidding agent converts it with
`clamp((value + 1) / 2, 0, 1)` and then applies the Issue #193 heuristic reward
rules.

For a full-policy comparison, load all four phase artifacts and call
`runFullPolicyVsRuleBasedEvaluation`:

```ts
import {
  loadNonPlayingPolicyOnnxModel,
  loadPolicyOnnxModel,
  runFullPolicyVsRuleBasedEvaluation
} from "@napoleon/policy-onnx";

const playingPolicy = await loadPolicyOnnxModel({
  onnxPath: "/models/playing.onnx",
  metadataPath: "/models/playing.json"
});
const biddingPolicy = await loadNonPlayingPolicyOnnxModel({
  onnxPath: "/models/bidding.onnx",
  metadataPath: "/models/bidding.json"
});
const adjutantPolicy = await loadNonPlayingPolicyOnnxModel({
  onnxPath: "/models/adjutant.onnx",
  metadataPath: "/models/adjutant.json"
});
const exchangePolicy = await loadNonPlayingPolicyOnnxModel({
  onnxPath: "/models/exchange.onnx",
  metadataPath: "/models/exchange.json"
});

const result = await runFullPolicyVsRuleBasedEvaluation({
  playingPolicy,
  biddingPolicy,
  adjutantPolicy,
  exchangePolicy,
  startSeed: 900,
  gameCount: 10
});
```

The full-policy evaluation uses source agent index 0 for one full
`PolicyOnnxAgent` and source agent indices 1 through 4 for `RuleBasedAgent`.
The default rotation offsets are `[0, 1, 2, 3, 4]`, so the full-policy agent is
scheduled once in each seat for every seed. The result has its own
configuration schema with the four policy metadata objects under
`configuration.policyMetadata.{playing,bidding,adjutant,exchange}` and keeps the
existing grouped comparison shape for policy versus rule-based metrics.

The grouped comparison reports scheduled/completed/failed games, illegal-action
failure count, policy and rule-based sample counts, wins/losses, win rate,
contract success rate, average point cards, failures, role breakdowns
(`napoleon`, `adjutant`, `alliance`), seat breakdowns (`0` through `4`), and
95% confidence intervals for win-rate delta, contract-success-rate delta, and
average-point-card delta. `diagnostics.policyAgentDecisionCounts` records the
full-policy agent's phase ONNX decision counts and RuleBased fallback decision
count for smoke validation.

The non-playing loader rejects artifacts whose metadata `policyType` does not
match the supplied slot (`bidding`, `adjutant`, or `exchange`). The full-policy
evaluation also checks the four slots before scheduling games; normal metadata,
hash, ONNX graph, input, and runtime compatibility checks still come from the
ONNX loaders and runtime.
