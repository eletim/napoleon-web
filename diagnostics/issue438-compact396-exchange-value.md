# Issue #438 compact396 exchange value report

## Audit

The requested 396-dim input is fully available at exchange source-state generation time:

- original 10-card Napoleon hand before kitty pickup
- 3-card kitty pickup from `state.unusedCards`
- called adjutant card
- final contract suit and target
- bidding starter relative player
- compact278-style bid-owner table from the encoded public bidding history
- candidate discard mask

The existing #436 JSONL rows cannot reconstruct compact396 because they only store the post-pickup 13-card hand and 2671-dim exchange observation. The dataset was regenerated with the same seed range and policies after adding these fields.

## Dataset

- Path: `/tmp/issue438-exchange-cf-compact1000`
- Source states: 1,000
- Samples: 286,000
- Invariant failures: 0
- Split guard: passed
- Split hashes match #436:
  - train `599d74342d18ec3ca5b8627a6a0f7896e407fe3f72eaa6ab5b48d86b6115df23`
  - validation `01c26f1b6beb791ed7e4f01527896cc442c33c363658f426ba982c8d2b05e00b`
  - final `069ea5f954f1209d312809c9b2a1ed565a68ef97e54864b3894baa7259524d15`

## Matched Final Held-Out

| Metric | #436 2724 | compact396 pointwise | compact396 pairwise | RuleBased |
| --- | ---: | ---: | ---: | ---: |
| MAE | 2.657 | 2.497 | 2.530 | n/a |
| RMSE | 3.310 | 3.118 | 3.166 | n/a |
| Pearson | 0.317 | 0.409 | 0.407 | n/a |
| bias | -0.120 | 0.021 | -0.119 | n/a |
| pairwise accuracy | 0.588 | 0.592 | 0.586 | n/a |
| exact top-action | 0.01 | 0.02 | 0.00 | n/a |
| top3 hit | 0.03 | 0.03 | 0.02 | n/a |
| top5 hit | 0.07 | 0.04 | 0.04 | n/a |
| selected rank percentile mean | 0.675 | 0.650 | 0.656 | 0.636 |
| margin regret mean | 4.39 | 4.60 | 4.43 | 4.59 |
| relative reward regret mean | 12.975 | 12.300 | 11.875 | 14.200 |
| model vs RuleBased margin regret | 39 / 16 / 45 | 43 / 13 / 44 | 45 / 18 / 37 | n/a |

## Learning Curve

| Train states | MAE | Pearson | pairwise | exact | top3 | top5 | rank pct | margin regret | reward regret |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 250 | 2.526 | 0.405 | 0.581 | 0.02 | 0.05 | 0.07 | 0.624 | 4.68 | 14.650 |
| 500 | 2.532 | 0.380 | 0.588 | 0.01 | 0.05 | 0.05 | 0.625 | 4.76 | 14.325 |
| 800 | 2.497 | 0.409 | 0.592 | 0.02 | 0.03 | 0.04 | 0.650 | 4.60 | 12.300 |

The scalar metrics improve with the full train split, but 286-candidate ranking and top-k do not show a useful monotonic improvement.

## Bury Content

| Selection | point-card counts 0/1/2/3 | trump mean | any special |
| --- | ---: | ---: | ---: |
| compact396 pointwise | 12 / 33 / 40 / 15 | 0.08 | 0.15 |
| compact396 pairwise | 9 / 43 / 30 / 18 | 0.21 | 0.15 |
| RuleBased | 77 / 20 / 3 / 0 | 0.00 | 0.00 |
| teacher-best | 13 / 39 / 38 / 10 | 0.70 | 0.26 |

There is no collapse to RuleBased zero-point burying. The model still under-buries trump and special cards relative to teacher-best.

## Checkpoints

- Selected pointwise checkpoint: `benchmarks/exchange-values/exchange-compact396-value-v1/checkpoint.pt`
- Selected checkpoint SHA-256: `50176357a6678de71889701965603d1dddc8c4101aabe467fa8d55c4fac38e6d`
- Pairwise comparison checkpoint: `benchmarks/exchange-values/exchange-compact396-value-v1/pairwise800/checkpoint.pt`
- Pairwise checkpoint SHA-256: `d6faeefe810314833ca0f12a86e3ff287a1e14f4c231ead96a9c8050e858e55c`

## Conclusion

Do not formally adopt compact396 as a runtime/Frozen replacement from this run. It is a cleaner representation and improves scalar regression, but it does not clearly improve the final held-out ranking objective over #436 2724 or RuleBased. The likely blocker is the one-hidden-deal teacher / hidden-information dependence rather than redundant input representation. The next step should be an expected-value teacher over sampled hidden deals, not runtime wiring, Frozen promotion, reward changes, or playing-policy retraining.
