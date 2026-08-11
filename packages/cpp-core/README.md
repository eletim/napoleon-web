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
ascending `game_id` order. `RuleBased` seats currently use the runtime's
deterministic CPU first-legal selector as the minimal in-process backend; policy
seats emit stable `AgentRequest` records and resume only through
`submit_agent_results()`.

The runtime is intentionally single-owner: callers exchange request/result
values instead of holding pointers into game state. This keeps the API ready for
worker-thread scheduling and policy-specific batched inference without making
Node IPC part of the per-decision hot path.

`RuntimeMetrics` records added/finished games, request/result counts, internal
CPU transitions, elapsed CPU nanoseconds, and throughput estimates.

## Local Commands

```sh
pnpm --filter @napoleon/cpp-core build
pnpm --filter @napoleon/cpp-core test
```
