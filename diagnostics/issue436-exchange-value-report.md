# Issue #436 Exchange Combination Value Report

## Dataset

- Path: `/tmp/issue436-exchange-cf-1000states`
- Size: 1,000 unique source exchange states, 286,000 samples
- Candidates/state: 286 min / 286 max / 286 mean
- Invariant failures: 0
- Generator source commit: `1c873cc701f70f99f44ba92bf35dec958d2e6718`
- Policies: `frozen-raise-v1` bidding, RuleBased adjutant, `ppo-separated-v1000` fixed playing

## Split

- Train: 800 states, 228,800 samples
- Validation: 100 states, 28,600 samples
- Final: 100 states, 28,600 samples
- Train state hash: `599d74342d18ec3ca5b8627a6a0f7896e407fe3f72eaa6ab5b48d86b6115df23`
- Validation state hash: `01c26f1b6beb791ed7e4f01527896cc442c33c363658f426ba982c8d2b05e00b`
- Final state hash: `069ea5f954f1209d312809c9b2a1ed565a68ef97e54864b3894baa7259524d15`
- Leakage guard: passed for `sourceStateKey`, `dealSeed`, `hiddenDealChecksum`, `fixedHandId`, and pickup hand.

## Learning Curve

Same final held-out split. The 250/500 rows limit train source states while keeping the same validation/final groups.

| Run | Best epoch | MAE | Pearson | Pairwise | Exact | Top5 | Model margin regret | RuleBased margin regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R0 MSE 250 train states | 3 | 2.641 | 0.328 | 0.569 | 0.00 | 0.06 | 4.63 | 4.59 |
| R0 MSE 500 train states | 2 | 2.669 | 0.321 | 0.583 | 0.01 | 0.05 | 4.75 | 4.59 |
| R0 MSE 1000 source states | 8 | 2.680 | 0.296 | 0.574 | 0.01 | 0.07 | 4.50 | 4.59 |

No strong learning-curve improvement appeared from 250 to 1,000 states.

## Loss Comparison

| Run | MAE | Pearson | Pairwise | Top5 | Model margin regret | Model reward regret | RuleBased margin regret | RuleBased reward regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R0 Huber | 2.657 | 0.317 | 0.588 | 0.07 | 4.39 | 12.975 | 4.59 | 14.20 |
| R1 pairwise 0.05 | 2.613 | 0.339 | 0.558 | 0.05 | 4.88 | 13.925 | 4.59 | 14.20 |
| R1 pairwise 0.5 | 2.582 | 0.333 | 0.562 | 0.04 | 4.98 | 15.325 | 4.59 | 14.20 |

Pairwise auxiliary improved neither final ranking nor regret. The best checkpoint is the R0 Huber pointwise model.

## Best Checkpoint

- Path: `benchmarks/exchange-values/exchange-combination-value-v1/checkpoint.pt`
- SHA-256: `f11e6a8d4638e477b603491d5ecef722457b73aad377a25bfe48f9763e9b3f24`
- Config: `2724 -> [512, 512, 256, 256] -> scalar contractMargin`
- Loss: pointwise Huber regression, target standardized
- Metadata/report/split: `benchmarks/exchange-values/exchange-combination-value-v1/`

## Bury Content

Final held-out selected actions:

| Selector | Point-card count 0/1/2/3 | Trump mean | Any special |
| --- | --- | ---: | ---: |
| model-selected | 13 / 35 / 35 / 17 | 0.25 | 0.16 |
| RuleBased | 77 / 20 / 3 / 0 | 0.00 | 0.00 |
| teacher-best | 13 / 39 / 38 / 10 | 0.70 | 0.26 |

The model moves toward teacher-best point-card burying and does not show a single-action collapse, but the trump/special distribution remains conservative relative to teacher-best.

## Conclusion

The implementation, dataset, training, and evaluation pipeline are in place. The best model is slightly better than RuleBased on final mean regret, but the gap is small and not a clear runtime/Frozen candidate yet. I would not wire this checkpoint into runtime in the next issue without addressing label noise/input sufficiency or collecting stronger evidence from a larger or repeated-rollout dataset.
