# Issue #452 parameterized adjutant/exchange policy

## Scope and fixed policies

- Learning target: adjutant choice and three-card exchange only.
- Bidding: `frozen-raise-v1`; the candidate cannot pass while a legal bid exists.
- Opponent bidding topology: deterministic per-seat Frozen / strong RuleBased /
  Conservative 1:1:1 assignment.
- Playing: `ppo-separated-v1000` for every seat.
- Fitness: candidate Napoleon mean relative reward, unchanged from Reward v3.
- Source commit: `9751afc5e8ee41b2748bcc1994bd3bbd8ec6f761`.
- No runtime wiring, Frozen promotion, NN training, reward change, or hidden-hand
  inference is included.

## Feature and parameter design

Schema v1 has 95 weights: 35 adjutant and 60 exchange. Every feature is computed
from the contract, public bidding history, Napoleon's visible hand, called card,
and the three visibly picked-up kitty cards. Other hands and the actual adjutant
owner are never read by the extractor.

The adjutant block contains the existing oruma / regular jack / reverse jack /
trump / rank structure plus candidate suit length and high cards, own point/high
card counts, void/singleton/doubleton structure, target interactions, and public
bidding counts.

The exchange block contains buried point/trump/rank and special-card features,
remaining trump, retained point/high cards, all four retained suit lengths and
void/singleton/doubleton flags, longest/shortest suits, same-suit discards, kitty
origin, called-card burial, target interactions, and the exact legacy RuleBased
regular-card score terms. The complete names, descriptions, and scales are in
`main/feature-schema.json`; every learned and initial weight is in
`main/best-parameters.json`.

The initial weights reproduce the existing RuleBased adjutant action and the
RuleBased-optimal exchange value on 100 deterministic fixtures. On the 3,000-game
main validation pool, the initial parameterized policy and existing RuleBased had
mean relative reward 0.21058 and 0.20725 respectively (difference +0.00333).

## Optimizer and CRN

The optimizer is the maintained `pycma` 4.4.4
`cma.CMAEvolutionStrategy`, not a custom CMA implementation. It optimizes bounded
deltas in [-40, 40] around the RuleBased initial vector. State is checkpointed in
`optimizer-state.pkl` and was verified to resume identically in a unit test.

Within every generation, the complete candidate population uses the exact same
deal-seed vector. The stochastic playing policy request sequence is also reset for
every candidate, so its sampled action stream is common as well. Generation seed
batches change deterministically. Smoke, main train, main validation, diagnostics,
and final manifests contain 57,100 unique seeds with zero overlap.

For the smoke A/B pair (10 batches, 200 games/parameter/batch), difference variance
fell from 1.06463 with independent seeds to 0.36091 with common seeds: ratio 0.3390,
or a 66.1% reduction. For the more behaviorally separated final A/B pair (12
batches, 500 games/parameter/batch), it fell from 0.90989 to 0.86396: ratio 0.9495,
or a 5.0% reduction. Both raw batch vectors and all manifests are retained.

## Optimization

Smoke used population 12, 8 generations, and 200 games/candidate (19,200 games;
83.5 seconds measured generation runtime). Validation fitness improved to 0.57375,
with no illegal action or fallback.

Main used population 16 and 500 fresh games/candidate/generation. It ran 45
generations (360,000 candidate-games; 1,554 seconds measured generation runtime)
and stopped before the requested maximum 50 after 12 generations without a 0.02
validation improvement. The fixed 3,000-game validation incumbent progressed:

```text
warm start  0.44808
generation 1  0.51308
generation 9  0.69725
generation 14 1.01225
generation 20 1.35142
generation 26 1.57308
generation 32 1.63225  (final incumbent)
generation 44 plateau stop
```

All 45 generations had illegal count 0 and fallback count 0. The final sigma was
1.3411 and the incumbent delta-vector norm was 45.7291.

## Final 5,000-game holdout

The final pool starts at seed 652,000,000 and was never used by smoke, main,
validation, or diagnostics. Both policies used the identical final seeds and
identical playing sampling stream.

| Metric | Learned | Existing RuleBased |
| --- | ---: | ---: |
| Mean relative reward | 1.14595 | -0.06705 |
| Standard error | 0.23633 | 0.22974 |
| 95% CI | [0.68275, 1.60915] | [-0.51734, 0.38324] |
| Contract success | 34.12% | 30.58% |
| Mean contract margin | -2.0702 | -2.5266 |
| Napoleon-side point cards | 12.1972 | 11.7408 |
| Illegal / fallback | 0 / 0 | 0 / 0 |

The paired mean difference is **+1.21300**, SE 0.25256, 95% CI
**[0.71799, 1.70801]**. Per-game win/tie/loss is 737 / 3,703 / 560. The declared
target distribution is identical by construction because adjutant/exchange occur
after bidding.

## Interpretation

Largest absolute weight changes (initial -> learned, with normalized feature scale):

| Block / feature | Initial | Learned | Delta | Scale |
| --- | ---: | ---: | ---: | ---: |
| exchange / buried_rank_sum | 0 | 14.886 | +14.886 | 60 |
| exchange / buried_point_count | 0 | 13.737 | +13.737 | 3 |
| exchange / retained_void_count | 0 | 11.904 | +11.904 | 4 |
| exchange / retained_spades_singleton | 0 | -10.864 | -10.864 | 1 |
| exchange / buried_ten_count | 0 | 10.526 | +10.526 | 3 |
| exchange / buried_called_adjutant | 0 | 8.060 | +8.060 | 1 |
| adjutant / is_yoromeki | 0 | 7.731 | +7.731 | 1 |
| adjutant / hand_high_count | 0 | 7.716 | +7.716 | 10 |
| adjutant / target_level | 0 | -7.580 | -7.580 | 6 |
| exchange / retained_diamonds_singleton | 0 | -7.540 | -7.540 | 1 |

Selection means show that the learned adjutant calls oruma less often (81.24% ->
74.46%), the regular jack more often (10.18% -> 18.76%), and the reverse jack less
often (7.78% -> 3.34%). It calls a trump-suit card more often (44.72% -> 55.94%)
and favors candidate suits in which its own hand has more length and high-card
support.

The dominant exchange change is deliberate point-card burial. Mean buried point
count rises from 0.077 to 1.230 cards (normalized 0.0258 -> 0.4101), especially
tens, while kitty-origin burial rises from 0.818 to 1.329 cards. This is consistent
with game semantics: buried point cards are immediately awarded to Napoleon's
side. The learned policy also creates more voids (normalized count 0.0683 ->
0.1127), particularly non-trump voids, and changes suit-shape preferences rather
than simply discarding the three lowest ranks.

## Artifacts

- Best logical artifact SHA-256: `d364aef0c48a1832bd6602d254d0440f6cb2e2cb50492cfb53934e0378a84d69`
- Best parameter file SHA-256: `6f069616e138a589c1aef4a6ca337def23f8d9da535a59dcd8825749bf5aa03d`
- Feature schema file SHA-256: `fe217987adac8d4671e849da795f0028516b709291f340a30ad45dd9d7aee098`
- Full learning curve, optimizer resume state, initial vector, seed manifests,
  variance reports, seed-overlap audit, byte-identical CRN replay check, per-seed
  final results, and selection tendencies are stored under this directory.

The holdout improvement is statistically clear and all invariants are clean, so
this artifact is suitable for a separate runtime-wiring/Frozen-candidate issue.
