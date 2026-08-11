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

## Local Commands

```sh
pnpm --filter @napoleon/cpp-core build
pnpm --filter @napoleon/cpp-core test
```
