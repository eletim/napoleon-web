# Issue #442 pseudo-fixed exchange repeat ablation

Base: `feature/issue-202-non-playing-ai`

## Final held-out comparison

| layout | Pearson | pairwise | margin regret | relative reward regret | RuleBased margin regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| #438 compact396 baseline | 0.409 | 0.592 | 4.60 | 12.30 | 4.59 |
| 1000x1 | 0.262 | 0.572 | 5.34 | 23.52 | 5.28 |
| 200x5 | 0.080 | 0.561 | 5.01 | 20.88 | 5.33 |
| 100x10 | 0.250 | 0.565 | 4.59 | 18.98 | 4.64 |

## Layout details

### 1000x1

- Dataset: `/tmp/issue442-pseudo-fixed-exchange-1000x1`
- Manifest SHA-256: `e0ddcf2954bb3eebbc4b1fd696077aee68be66ab44858a2d65601fb1df5f2cec`
- Checkpoint: `benchmarks/exchange-values/issue442-pseudo-fixed-exchange-repeat-ablation/1000x1/checkpoint.pt`
- Checkpoint SHA-256: `c329b86de6978ec4d5225ef3d9fb29b413c3cb16d814a65746701ce3b2517b2a`
- Accepted/rejected: 1000 / 0
- Rejection reasons: `{}`
- Target distribution: `{"13": 256, "14": 340, "15": 271, "16": 106, "17": 22, "18": 5}`
- Suit distribution: `{"clubs": 137, "diamonds": 279, "hearts": 187, "spades": 397}`
- Unique bidding histories: 638
- Opponent policy ratios: `{"conservative-bidding-v1": 0.3355, "frozen-raise-v1": 0.3425, "strong-rule-based-bidding-v1": 0.322}`
- Split: `{"final": {"fixedThirteenGroupCount": 100, "sampleCount": 28600, "stateCount": 100}, "train": {"fixedThirteenGroupCount": 800, "sampleCount": 228800, "stateCount": 800}, "validation": {"fixedThirteenGroupCount": 100, "sampleCount": 28600, "stateCount": 100}}`
- Leakage guard: `{"dealSeed": 0, "fixedThirteenGroupId": 0, "hiddenDealChecksum": 0, "pickupHand": 0, "sourceStateKey": 0}`
- Scalar: MAE 2.754, RMSE 3.376, bias -0.038
- Ranking: exact 0.000, top3 0.020, top5 0.020, selected rank percentile 0.569
- RuleBased reward regret: 21.52
- Model vs RuleBased win/tie/loss: 46 / 11 / 43
- Same-13 repeat metrics: n/a

### 200x5

- Dataset: `/tmp/issue442-pseudo-fixed-exchange-200x5-full`
- Manifest SHA-256: `4cd2a62887b821976d68276adfa7d8e63c42184baabead10966417288b41389c`
- Checkpoint: `benchmarks/exchange-values/issue442-pseudo-fixed-exchange-repeat-ablation/200x5/checkpoint.pt`
- Checkpoint SHA-256: `44eecdbb548eb867d0ae2c9f84e1c54f71f4bdada9276241bf9539f652bf7712`
- Accepted/rejected: 1000 / 0
- Rejection reasons: `{}`
- Target distribution: `{"13": 237, "14": 327, "15": 272, "16": 120, "17": 36, "18": 8}`
- Suit distribution: `{"clubs": 138, "diamonds": 247, "hearts": 215, "spades": 400}`
- Unique bidding histories: 700
- Opponent policy ratios: `{"conservative-bidding-v1": 0.32175, "frozen-raise-v1": 0.34025, "strong-rule-based-bidding-v1": 0.338}`
- Split: `{"final": {"fixedThirteenGroupCount": 20, "sampleCount": 28600, "stateCount": 100}, "train": {"fixedThirteenGroupCount": 160, "sampleCount": 228800, "stateCount": 800}, "validation": {"fixedThirteenGroupCount": 20, "sampleCount": 28600, "stateCount": 100}}`
- Leakage guard: `{"dealSeed": 0, "fixedThirteenGroupId": 0, "hiddenDealChecksum": 0, "pickupHand": 0, "sourceStateKey": 0}`
- Scalar: MAE 3.091, RMSE 3.776, bias -0.856
- Ranking: exact 0.000, top3 0.010, top5 0.010, selected rank percentile 0.605
- RuleBased reward regret: 21.18
- Model vs RuleBased win/tie/loss: 48 / 15 / 37
- Same-13 group mean margin regret: 5.01
- Same-13 group mean relative reward regret: 20.88
- Teacher-best discard unique count mean: 4.95
- Model-selected discard unique count mean: 3.95
- Predicted value consistency stddev mean: 0.977

### 100x10

- Dataset: `/tmp/issue442-pseudo-fixed-exchange-100x10-full`
- Manifest SHA-256: `2611b660fe2e7a8ed4e172bede95f0c883631ce2b08849c735e0a6221f9e2594`
- Checkpoint: `benchmarks/exchange-values/issue442-pseudo-fixed-exchange-repeat-ablation/100x10/checkpoint.pt`
- Checkpoint SHA-256: `032592ee50bbf8c4a1f98b1be58c71019e2d82e2bb64ab4ce078fb375023cb79`
- Accepted/rejected: 1000 / 0
- Rejection reasons: `{}`
- Target distribution: `{"13": 244, "14": 323, "15": 278, "16": 116, "17": 29, "18": 10}`
- Suit distribution: `{"clubs": 174, "diamonds": 243, "hearts": 156, "spades": 427}`
- Unique bidding histories: 659
- Opponent policy ratios: `{"conservative-bidding-v1": 0.32525, "frozen-raise-v1": 0.33625, "strong-rule-based-bidding-v1": 0.3385}`
- Split: `{"final": {"fixedThirteenGroupCount": 10, "sampleCount": 28600, "stateCount": 100}, "train": {"fixedThirteenGroupCount": 80, "sampleCount": 228800, "stateCount": 800}, "validation": {"fixedThirteenGroupCount": 10, "sampleCount": 28600, "stateCount": 100}}`
- Leakage guard: `{"dealSeed": 0, "fixedThirteenGroupId": 0, "hiddenDealChecksum": 0, "pickupHand": 0, "sourceStateKey": 0}`
- Scalar: MAE 3.037, RMSE 3.705, bias -1.923
- Ranking: exact 0.020, top3 0.040, top5 0.040, selected rank percentile 0.611
- RuleBased reward regret: 17.07
- Model vs RuleBased win/tie/loss: 41 / 22 / 37
- Same-13 group mean margin regret: 4.59
- Same-13 group mean relative reward regret: 18.98
- Teacher-best discard unique count mean: 9.50
- Model-selected discard unique count mean: 6.20
- Predicted value consistency stddev mean: 1.036

## Conclusion

- Pseudo-fixed repeats helped: `True`
- Best layout: `100x10`
- Adopt pseudo-fixed as formal exchange teacher candidate: `False`
- Move to adjutant+kitty joint design if no improvement: `True`
- Summary: 100x10 is best within the pseudo-fixed layouts, but it only ties the #438 margin-regret baseline after rounding while pairwise and relative reward regret are worse. Treat pseudo-fixed repeats as useful for hidden-deal variation, not as a formal exchange teacher candidate yet.
