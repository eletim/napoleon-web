# exchange-compact396-value-v1

Issue #438 exchange discard-combination compact396 comparison.

- Dataset: `/tmp/issue438-exchange-cf-compact1000`
- Samples: 1,000 source exchange states x 286 candidates = 286,000 samples
- Split: source-state grouped 800 / 100 / 100 train / validation / final
- Split hashes match `exchange-combination-value-v1` from Issue #436:
  - train `599d74342d18ec3ca5b8627a6a0f7896e407fe3f72eaa6ab5b48d86b6115df23`
  - validation `01c26f1b6beb791ed7e4f01527896cc442c33c363658f426ba982c8d2b05e00b`
  - final `069ea5f954f1209d312809c9b2a1ed565a68ef97e54864b3894baa7259524d15`
- Input: compact exchange state 343 dims + candidate discard mask 53 dims = 396 dims
- Model: MLP `[512, 512, 256, 256] -> scalar contractMargin`
- Selected checkpoint: pointwise Huber regression, delta 1.0
- Checkpoint SHA-256: `50176357a6678de71889701965603d1dddc8c4101aabe467fa8d55c4fac38e6d`
- Pairwise comparison run: `pairwise800/`, checkpoint SHA-256 `d6faeefe810314833ca0f12a86e3ff287a1e14f4c231ead96a9c8050e858e55c`

Final held-out summary:

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

Learning curve on the same final held-out split:

| Train states | MAE | Pearson | pairwise | exact | top3 | top5 | rank pct | margin regret | reward regret |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 250 | 2.526 | 0.405 | 0.581 | 0.02 | 0.05 | 0.07 | 0.624 | 4.68 | 14.650 |
| 500 | 2.532 | 0.380 | 0.588 | 0.01 | 0.05 | 0.05 | 0.625 | 4.76 | 14.325 |
| 800 | 2.497 | 0.409 | 0.592 | 0.02 | 0.03 | 0.04 | 0.650 | 4.60 | 12.300 |

Bury content on final held-out is not collapsed to RuleBased zero-point-card burying, but compact396 selected actions still do not track teacher-best special-card/trump burying closely:

| Selection | point-card counts 0/1/2/3 | trump mean | any special |
| --- | ---: | ---: | ---: |
| compact396 pointwise | 12 / 33 / 40 / 15 | 0.08 | 0.15 |
| compact396 pairwise | 9 / 43 / 30 / 18 | 0.21 | 0.15 |
| RuleBased | 77 / 20 / 3 / 0 | 0.00 | 0.00 |
| teacher-best | 13 / 39 / 38 / 10 | 0.70 | 0.26 |

Conclusion: compact396 improves scalar regression versus the 2724 baseline, but it does not clearly improve the 286-candidate ranking objective. Pointwise compact396 is also not clearly above RuleBased on margin regret, while pairwise compact396 is only marginally above RuleBased and still below the #436 top-k/rank behavior. Treat this as evidence that input representation is not the main blocker; the next experiment should move to an expected-value teacher over hidden deals rather than runtime wiring or Frozen promotion.
