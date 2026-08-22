# Issue #378 Multi-head Bidding Teacher Decomposition

Base branch: `feature/issue-202-non-playing-ai`

Runtime bidding-agent integration was intentionally not changed.

## Dataset

- Reused dataset: `/tmp/napoleon-issue377-qdataset-22000x1`
- Source states: `22,000`
- Raw samples / forced state-action pairs: `122,765`
- Dataset schema / sample schema: `2 / 2`
- Manifest SHA256: `e83730b05cb97c4f107c9fcb2b068ff9bea6cf6b0859a4e83f8a9b46c590eec4`
- Fixed validation state file: `/tmp/napoleon-issue377-learning-curve/validation-state-keys.json`
- Fixed validation hash: `9ff43c57867712f5352467b3d56e397046b02c3c11b22a579ab0298a9777d57a`
- Train / validation states: `20,000 / 2,000`
- Train / validation raw samples: `111,563 / 11,202`
- No rollout was regenerated.

## Model

- Architecture: `bidding-multi-head-q-mlp-v1`
- Input: compact278 bidding observation only
- Shared MLP hidden dims: `[512,512,256,256]`
- Heads:
  - `P(finalRole | s,a)`: `29 x 5` role logits
  - `E(napoleonSidePointCards | s,a)`: `29` regression outputs
  - `E(coalitionSidePointCards | s,a)`: deterministic `20 - napoleonSidePointCards`
  - `P(contractSuccess | s,a)`: `29` binary logits
  - `E(contractMargin | s,a)`: `29` regression outputs
- Loss weights: role CE `1.0`, Napoleon points MSE `1.0`, success BCE `1.0`, margin MSE `1.0`
- Regression targets were standardized for loss scale control.
- Weight decay: `1e-4`
- Early stopping: monitor validation total loss, patience `10`
- Best epoch: `6` of `16` executed epochs
- Output: `/tmp/napoleon-issue378-multi-head-20k`

## Teacher Signal

Coalition-side point cards are fully determined:

- Checked samples: `121,883`
- Observed total point cards: `20`
- Mismatches: `0`

Action signal:

| teacher | same-target suit gap | exact tie | strongest teacher-best | target gap |
|---|---:|---:|---:|---:|
| Napoleon-side cards | 4.608 | 7.24% | 38.64% | 2.018 |
| Contract margin | 4.893 | 5.74% | 37.85% | 3.064 |

Contract success:

- Same-state action changes success: `23.19%`
- Suit-change success changes: `10.95%`
- Target-change success changes: `13.39%`

Conclusion: contract margin has the largest fine-grained numeric action signal. Success has meaningful ranking signal but the positive class is sparse.

## Fixed Validation Metrics

Role head:

- Accuracy: `96.02%`
- Macro F1: `0.606`
- Cross entropy: `0.123`
- Majority role baseline accuracy: `87.22%`
- Action-index role-frequency baseline accuracy: `90.59%`

Point-card heads:

| target | MAE | RMSE | Pearson |
|---|---:|---:|---:|
| Napoleon-side | 2.858 | 3.527 | 0.425 |
| Coalition-side deterministic | 2.858 | 3.527 | 0.425 |

Napoleon-side baselines:

| baseline | MAE | RMSE | Pearson |
|---|---:|---:|---:|
| global mean | 3.189 | 3.884 | 0.000 |
| action-index mean | 3.149 | 3.837 | 0.155 |
| role mean | 3.130 | 3.808 | 0.197 |
| role x action mean | 3.121 | 3.803 | 0.203 |
| suit x target mean | 3.149 | 3.837 | 0.155 |

Contract success:

- Accuracy: `94.52%`
- Precision / recall / F1 at threshold 0.5: `None / 0.000 / None`
- ROC-AUC: `0.788`
- PR-AUC: `0.172`
- Brier: `0.048`
- Positive rate: `5.48%`

Success baselines:

| baseline | ROC-AUC | PR-AUC |
|---|---:|---:|
| global success rate | 0.500 | 0.055 |
| target success rate | 0.762 | 0.130 |
| action-index success rate | 0.715 | 0.108 |
| suit x target success rate | 0.766 | 0.140 |

Contract margin:

- MAE: `3.074`
- RMSE: `3.832`
- Pearson: `0.532`
- Sign accuracy: `94.53%`

Margin baselines:

| baseline | MAE | RMSE | Pearson |
|---|---:|---:|---:|
| global mean | 3.692 | 4.511 | 0.000 |
| target mean | 3.402 | 4.185 | 0.373 |
| action-index mean | 3.402 | 4.185 | 0.373 |
| role x action mean | 3.343 | 4.112 | 0.411 |
| suit x target mean | 3.402 | 4.185 | 0.373 |

## Suit / Target Breakdown

Napoleon-side by suit:

| suit | MAE | RMSE | Pearson |
|---|---:|---:|---:|
| clubs | 2.822 | 3.483 | 0.399 |
| diamonds | 2.753 | 3.395 | 0.424 |
| hearts | 2.874 | 3.533 | 0.417 |
| spades | 2.862 | 3.542 | 0.419 |

Napoleon-side by target:

| target | MAE | RMSE | Pearson |
|---:|---:|---:|---:|
| 13 | 2.856 | 3.478 | 0.429 |
| 14 | 2.818 | 3.472 | 0.414 |
| 15 | 2.853 | 3.516 | 0.414 |
| 16 | 2.733 | 3.409 | 0.443 |
| 17 | 2.806 | 3.469 | 0.393 |
| 18 | 2.815 | 3.550 | 0.415 |
| 19 | 2.909 | 3.636 | 0.402 |

Margin by target Pearson stayed positive for every target (`0.386` to `0.452`).

## Offline Score Ranking

Teacher-covered fixed validation actions:

| score | teacher best | best hit | top3 hit | pairwise | PASS vs best-bid | strongest match |
|---|---|---:|---:|---:|---:|---:|
| Napoleon-side cards | Napoleon-side cards | 41.95% | 74.85% | 58.77% | 62.29% | 37.82% |
| Contract success | Contract success | 91.15% | 97.60% | 77.24% | 74.00% | 32.95% |
| Contract margin | Contract margin | 49.50% | 80.55% | 66.56% | 62.97% | 32.82% |
| Role-aware | candidate team cards | 41.35% | 75.10% | 58.17% | 53.52% | 38.15% |

Predicted best target distributions:

- Napoleon-side score: 13:`321`, 14:`97`, 15:`134`, 16:`57`, 17:`120`, 18:`49`, 19:`92`
- Contract success score: 13:`439`, 14:`122`, 15:`37`, 16:`6`
- Contract margin score: 13:`504`, 14:`86`, 15:`57`, 16:`8`
- Role-aware score: 13:`240`, 14:`46`, 15:`70`, 16:`32`, 17:`73`, 18:`29`, 19:`50`

## Comparison

Previous fixed-validation references:

- #375 candidateTeamPointCards: MAE `3.226`, RMSE `4.079`, Pearson `0.358`
- #377 20k candidateTeamPointCards: MAE `3.239`, RMSE `4.053`, Pearson `0.342`, pairwise `55.36%`, strongest suit match `32.0%`

This run:

- Napoleon-side point cards: MAE `2.858`, RMSE `3.527`, Pearson `0.425`
- Napoleon-side ranking pairwise: `58.77%`
- Napoleon-side strongest suit match: `37.82%`
- Contract margin pairwise: `66.56%`
- Contract success ROC-AUC: `0.788`

## Conclusion

Teacher decomposition is promising. Napoleon-side point-card prediction is clearly easier than candidateTeamPointCards, contract success beats action/target/suit-target baselines by ROC-AUC and PR-AUC, margin has a strong positive Pearson, and ranking exceeds the previous 55% ceiling.

The remaining weakness is thresholded success classification: with only `5.48%` positives, the default 0.5 threshold predicts no positives despite good AUC/calibration. Runtime score design should use calibrated probability/ranking, not a hard 0.5 classifier.

Next step: proceed to runtime score design using Napoleon-side cards and margin/success as score components, while separately investigating suit-relative architecture if strongest-suit relation needs further improvement.
