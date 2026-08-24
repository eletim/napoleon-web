# Issue #444 joint adjutant/exchange teacher diagnostic

## Scope

This diagnostic keeps the runtime order unchanged:

```text
choose adjutant -> kitty 3-card pickup -> choose 3 discards -> playing
```

It does not introduce a runtime `53 x 286` joint action, Frozen artifact, reward change,
runtime wiring, large NN training, exchange retraining, or playing-policy retraining.

The implementation adds a C++ diagnostic evaluator and CLI:

- `packages/cpp-core/native/include/napoleon_joint_teacher.hpp`
- `packages/cpp-core/native/src/napoleon_joint_teacher.cpp`
- `packages/cpp-core/native/src/napoleon_joint_teacher_cli.cpp`

Raw output:

- `diagnostics/issue444-joint-teacher.json`

## Joint Teacher Definition

Exchange remains the post-kitty value problem:

```text
Q_exchange(exchangeVisibleStateCompact343, discardMask53)
  -> downstream value after discard then playing
```

Adjutant becomes the upstream expected downstream value problem:

```text
Q_adjutant(adjutantVisibleStateCompact237, adjutantCard53)
  = E[ value after
      adjutant fixed
      -> kitty revealed
      -> best/high-quality exchange selected
      -> playing ]
```

The joint part is teacher/value composition only. Runtime still chooses adjutant first,
then observes kitty, then chooses exchange.

## compact290 Audit

The proposed adjutant value input is valid for the adjutant-choice information set:

| Feature | Dims | Known at adjutant choice |
| --- | ---: | --- |
| original hand mask | 53 | yes |
| contract suit | 4 | yes |
| contract target 13..19 | 7 | yes |
| bidding starter relative to Napoleon | 5 | yes |
| compact bid-owner table | 168 | yes |
| candidate adjutant card | 53 | yes |

Total: `290`.

Intentionally excluded:

- kitty pickup
- pickup hand 13
- discard candidate
- adjutant owner after the call

No required known information is missing for the compact state proposed in the issue.

## compact396 Reuse

The exchange side reuses the #438 compact396 shape:

- state: `343`
- candidate discard mask: `53`
- value input: `396`

The C++ evaluator reconstructs the original 10-card hand, actual kitty pickup, called
adjutant, contract, starter-relative feature, bid-owner table, and discard mask for
invariant checking.

## Gold Exhaustive Cost

Command:

```bash
packages/cpp-core/build-issue444/napoleon_joint_teacher_cli \
  --states 5 \
  --exhaustive-states 5 \
  --max-deal-attempts 125 \
  --heuristic-top-k 8 \
  --start-seed 444000000 \
  --agent-seed 444 \
  --output diagnostics/issue444-joint-teacher.json
```

Measured:

| Metric | Value |
| --- | ---: |
| source states | 5 |
| adjutant candidates/state | 53 |
| discard candidates/adjutant | 286 |
| exhaustive gold rollouts | 75,790 |
| total terminal rollouts including approximations | 78,191 |
| elapsed | 127.079 s |
| throughput | 615.294 rollouts/s |

Full `53 x 286` is feasible for small diagnostics, but too expensive as the default
teacher path at large scale.

## Approximation Result

Approximation tested here:

```text
RuleBased exchange candidate
+ compact396-proxy heuristic top-k discard candidates
-> terminal rollouts only for that small candidate set
```

The proxy uses the same candidate-value slot intended for compact396, but it does not
load the PyTorch compact396 checkpoint. Production teacher generation should replace
this proxy scorer with batched compact396 checkpoint inference.

Gold match against 5 exhaustive states:

| Metric | Value |
| --- | ---: |
| best-adjutant exchange gold contained in top-k | 0.20 |
| joint top-1 exact match | 0.00 |
| RuleBased exchange equals gold | 0.00 |

Conclusion: RuleBased + simple heuristic top-k is not enough. The practical teacher
should use batched compact396 scoring for all 286 exchange candidates per adjutant,
then terminal-rollout `top-k + RuleBased + a small diversity/random tail`. Use this
diagnostic shape to measure gold containment before scaling.

## Same-State Value Decomposition

All three values below are measured on the same source states:

| Policy composition | Mean contract margin |
| --- | ---: |
| RuleBased adjutant + RuleBased exchange | -2.0 |
| RuleBased adjutant + optimized exchange | 2.4 |
| optimized adjutant + optimized exchange | 3.2 |

Gain decomposition:

| Gain | Min | Max | Mean |
| --- | ---: | ---: | ---: |
| optimized exchange with RuleBased adjutant | 2 | 6 | 4.4 |
| optimized adjutant after optimized exchange | 0 | 1 | 0.8 |

Most improvement in this small sample comes from exchange optimization, but adjutant
selection still contributes residual value after exchange is optimized.

Per-source summary:

| Seed | Contract | RB adj | Best adj | Approx best adj | RB/RB | RB/best exch | Best/best |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| 444000000 | spades 16 | clubs-J | clubs-10 | hearts-J | -4 | 2 | 3 |
| 444000001 | hearts 15 | spades-A | clubs-9 | hearts-J | -3 | 1 | 2 |
| 444000002 | diamonds 15 | spades-A | diamonds-10 | diamonds-Q | 0 | 2 | 3 |
| 444000003 | spades 16 | spades-A | clubs-5 | spades-2 | -1 | 4 | 4 |
| 444000004 | diamonds 14 | spades-A | clubs-2 | clubs-3 | -2 | 3 | 4 |

## Adjutant Diagnostics

| Metric | Value |
| --- | ---: |
| RuleBased adjutant match rate | 0.00 |
| RuleBased adjutant regret mean | 0.8 |
| RuleBased adjutant regret range | 0..1 |
| adjutant candidate value spread mean | 6.2 |
| adjutant candidate value spread range | 5..8 |
| top1-top3 gap mean | 0.0 |
| unique best adjutants across 5 states | 5 |

The top of the adjutant ranking is often tied or near-tied under this deterministic
single-hidden-deal rollout, but candidate spread is non-trivial.

## Exchange Diagnostics

For the gold best adjutant, exchange candidate value spread averaged `11.4`
contract-margin points (`11..12` range). RuleBased exchange did not match gold in
this 5-state sample.

## Hidden Deal Variation

This diagnostic preserves each accepted deal's hidden hands and bidding history and
does not reshuffle after bidding. It does not yet build repeated fixed-hand,
history-consistent accepted-deal groups, so hidden-deal best-adjutant variation is
reported only as cross-source variation (`5` unique best adjutants across `5`
accepted deals).

Production teacher generation should add pseudo-fixed groups:

```text
candidate original hand fixed
-> remaining cards shuffled
-> bidding rerun
-> accept only candidate-Napoleon contract states
-> evaluate adjutant + optimized exchange downstream value
```

That gives the real hidden-deal variation estimate without violating bidding-history
consistency.

## Next Training Recommendation

Dataset scale for the next training issue:

- Start with `10k` accepted choosing-adjutant states for the first adjutant-value dataset.
- For each state, score all `53` adjutant candidates.
- For each adjutant, use compact396 batched scoring over all `286` exchange candidates.
- Terminal-rollout `top 16 + RuleBased + 8 diversity/random` exchange candidates per adjutant.
- Keep a held-out `200` state gold set with full `53 x 286` exhaustive rollout for teacher
  approximation audits.
- Scale to `50k..100k` accepted states only after gold containment and regret are stable.

Recommended adjutant model:

```text
compact290 candidate-value MLP -> scalar downstream-optimized contract margin
```

This matches exchange's candidate-value form and avoids a giant runtime joint action.
A state-only trunk with 53 outputs is reasonable later, but candidate-value is the
cleaner first model for dataset/debug parity.

Exchange model retraining:

- Do not retrain exchange from this issue's data alone.
- First use the existing compact396 checkpoint as a teacher proposal scorer.
- Retrain exchange only after the joint teacher produces a materially cleaner target:
  expected downstream value over history-consistent hidden deals, not one-hidden-deal
  terminal margin.

## Verification

```bash
cmake -S packages/cpp-core/native -B packages/cpp-core/build-issue444
cmake --build packages/cpp-core/build-issue444 --target napoleon_joint_teacher_cli napoleon_core_self_test
packages/cpp-core/build-issue444/napoleon_core_self_test
packages/cpp-core/build-issue444/napoleon_joint_teacher_cli --states 5 --exhaustive-states 5 --max-deal-attempts 125 --heuristic-top-k 8 --start-seed 444000000 --agent-seed 444 --output diagnostics/issue444-joint-teacher.json
```

`napoleon_core_self_test` includes a deterministic 1-state joint teacher fixture with:

- choosing-adjutant source generation
- 53 adjutant candidates
- adjutant fixed -> kitty pickup -> exchanging
- 13-card pickup hand
- 286 exchange candidates
- discard -> playing -> terminal
- compact290 feature count
- compact396 reuse invariant
- RuleBased vs optimized decomposition report shape
- deterministic seed replay
