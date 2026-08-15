# @napoleon/cpp-core

AI training and evaluation only C++ simulation core scaffold for Napoleon.

The TypeScript `@napoleon/game-core` package remains the rules source of truth.
This package exists to grow a fast C++ mirror behind differential tests, not to
enter the browser, web UI, or normal server runtime.

## Seed Contract

Seeds are unsigned 32-bit integers. The native core uses the same Mulberry32
stream as the existing TypeScript simulation helpers:

1. normalize the seed into `[0, 0xffffffff]`
2. advance state with `state = (state + 0x6d2b79f5) >>> 0`
3. produce a floating value in `[0, 1)` from the same integer mixing steps
4. use Fisher-Yates shuffle with `floor(rng() * (index + 1))`

Differential tests pass the same seed and action sequence to TypeScript and
C++, then compare canonical snapshots.

## Simulation Runtime

`napoleon_simulation_runtime.hpp` provides the deterministic multi-game runtime
for AI training and evaluation applications. A `SimulationRuntime` owns all
logical games and exposes value-only queues:

```text
add_games(N)
advance_runnable_games()
collect_agent_requests()
submit_agent_results()
collect_finished_games()
```

Game seeds are `base_seed + game_index` modulo `uint32_t`; roster sampling uses
`roster_seed` plus the same stable `game_index`. Runnable games are advanced in
ascending `game_id` order. `RuleBased` seats use the C++ RuleBased selector for
playing decisions and retain the deterministic CPU first-legal selector for
non-playing setup phases that are outside that agent's current scope. Policy
seats emit stable `AgentRequest` records and resume only through
`submit_agent_results()`.

The runtime is intentionally single-owner: callers exchange request/result
values instead of holding pointers into game state. This keeps the API ready for
worker-thread scheduling and policy-specific batched inference without making
Node IPC part of the per-decision hot path.

`RuntimeMetrics` records added/finished games, request/result counts, internal
CPU transitions, elapsed CPU nanoseconds, and throughput estimates.

## C++ ONNX Policy Batching

`napoleon_onnx_policy.hpp` connects playing-phase `AgentRequest` values from
`SimulationRuntime` to policy-specific batched inference. The runtime still owns
game state and scheduling; the ONNX module provides an opt-in request payload
builder for playing model input v2 plus an executor that owns policy sessions,
per-policy request queues, action sampling, and inference metrics.

Policy agents are keyed by `agent_type:id`, for example
`current-policy:current` and `frozen-policy:rl-v740`. Each key has its own
`PolicySession`, request queue, `session.run()` count, batch histogram, and
elapsed inference time. `max_batch_size` belongs to the executor and is separate
from `SimulationRuntimeConfig::max_concurrent_games`, so logical game
concurrency and GPU batch size can be tuned independently.

When `SimulationRuntimeConfig::build_agent_request_payload` is set to
`onnx_policy::attach_playing_model_input`, playing requests carry value-only
`playing_model_input` `[6246]` and `legal_play_mask` `[53]` fields. Non-playing
requests leave those fields empty; the ONNX executor rejects them instead of
falling back silently. Selected actions are sampled from the same masked
categorical softmax contract as the TypeScript policy path, including
deterministic request-seeded sampling and `behavior_log_probability` on
`AgentResult`. If a legal mask contains one card, the selected log probability
is `0`.

`BatchedPolicyStats` reports:

- request count
- session.run count
- mean batch size
- max observed batch size
- batch histogram
- policy-specific stats
- inference elapsed nanoseconds

The default build includes `DeterministicPolicySession` for batching, ordering,
masking, and behavior parity tests without requiring ONNX Runtime. Real ONNX
Runtime C++ support is opt-in:

```sh
cmake -S packages/cpp-core/native -B packages/cpp-core/build-ort \
  -DNAPOLEON_ENABLE_ONNXRUNTIME=ON
cmake --build packages/cpp-core/build-ort
```

If `NAPOLEON_ENABLE_ONNXRUNTIME=ON` cannot find `onnxruntime_cxx_api.h` or
`libonnxruntime`, configuration fails. If `--provider cuda` is requested and
CUDAExecutionProvider is unavailable, session creation fails; the executor does
not silently fall back to CPU.

CUDA smoke example:

```sh
./packages/cpp-core/build-ort/napoleon_onnx_policy_smoke \
  --current-onnx /path/to/current/policy.onnx \
  --frozen-onnx /path/to/frozen/policy.onnx \
  --provider cuda \
  --games 8 \
  --max-batch-size 16
```

The smoke command runs C++ simulation only, with first-legal setup actions for
non-playing phases and ONNX policy decisions for playing. It prints runtime and
per-policy inference metrics as JSON. This executable is for AI development
smoke testing only and is not used by the browser, web UI, or normal server
runtime.

## Evaluation / Benchmark / Tournament CLI

`napoleon_eval_cli` runs the native simulation runtime as an AI-development
application. It is intentionally outside the browser and normal Web runtime.
The first backend consumes policy `AgentRequest` batches deterministically so
candidate, frozen-policy, opponent-pool, and tournament schedules can be tested
without depending on RL dataset generation.

```sh
./build/napoleon_eval_cli \
  --scenario candidate-vs-rule-based \
  --start-seed 0 \
  --seed-count 400 \
  --max-concurrent-games 256 \
  --inference-max-batch-size 32 \
  --output cpp-evaluation.json
```

The JSON artifact includes completed game records, candidate and per-policy
stats, contract success, average point cards, total/simulation/inference
elapsed time, games/sec, decisions/sec, request count, session-run count,
mean/max batch size, policy-specific batch stats, and the known TypeScript
2000-game baselines:

```text
TS CUDA batch=1 / workers=4: 11.20 sec / 2000 games
TS batched path:            18-22 sec / 2000 games
```

## RL Tensor Dataset CLI

`napoleon_rl_dataset_cli` generates tensor-ready binary shards for policy
training. It consumes the C++ runtime, attributes samples only to the current
policy seat, and writes raw shard output plus a manifest with policy artifact
provenance and roster-seat attribution.

## Local Commands

```sh
pnpm --filter @napoleon/cpp-core build
pnpm --filter @napoleon/cpp-core test
```
