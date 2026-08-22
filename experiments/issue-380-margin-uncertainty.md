# Issue #380 Margin Mean and Uncertainty

Base branch: `feature/issue-202-non-playing-ai`

Runtime bidding-agent integration was intentionally not changed.

## Dataset

- Reused dataset: `/tmp/napoleon-issue377-qdataset-22000x1`
- Source states: `22,000`
- Raw samples / forced state-action pairs: `122,765`
- Manifest SHA256: `e83730b05cb97c4f107c9fcb2b068ff9bea6cf6b0859a4e83f8a9b46c590eec4`
- Fixed validation state file: `/tmp/napoleon-issue377-learning-curve/validation-state-keys.json`
- Fixed validation hash: `9ff43c57867712f5352467b3d56e397046b02c3c11b22a579ab0298a9777d57a`
- Train / validation states: `20,000 / 2,000`
- Train / validation raw samples: `111,563 / 11,202`
- Train / validation contract samples: `110,777 / 11,106`
- No rollout was regenerated.

## Model

- Architecture: `bidding-margin-heteroscedastic-mlp-v1`
- Input: compact278 bidding observation only
- Dedicated MLP hidden dims: `[512,512,256,256]`
- Outputs:
  - `mean[29]`: standardized contract-margin mean
  - `log_variance[29]`: clamped standardized log variance
- Sigma parameterization: `sigma = exp(0.5 * log_variance)`
- Log variance clamp: `[-6.0, 5.0]`
- Target standardization: enabled with train-only mean `-7.806467`, std `4.499564`
- Raw-scale reports decode mean and multiply sigma by the train target std.
- Loss: Gaussian heteroscedastic NLL, selected `forcedActionIndex` only, NoContract / AllPass masked
- Optimizer: `AdamW`, lr `1e-3`, weight decay `1e-4`
- Batch size: `128`
- Device: `cpu`
- Early stopping: validation NLL, patience `10`
- Best epoch: `5`
- Stopped epoch: `15`
- Output: `/tmp/napoleon-issue380-margin-20k`
- ONNX export: `/tmp/napoleon-issue380-margin-20k/model.onnx`
- ONNX parity: max abs diff `1.49e-7`, within tolerance

## Mean Metrics

Fixed validation contract-margin prediction:

| metric | value |
|---|---:|
| MAE | 3.086 |
| RMSE | 3.830 |
| Pearson | 0.529 |
| Sign accuracy | 94.52% |
| Pairwise ranking | 66.76% |
| Best-action hit | 49.70% |
| Top-3 hit | 81.35% |
| PASS vs best-bid | 63.45% |
| Strongest suit match | 31.23% |

Compared with #378 contract margin head:

| run | MAE | RMSE | Pearson | pairwise |
|---|---:|---:|---:|---:|
| #378 multi-head margin | 3.074 | 3.832 | 0.532 | 66.56% |
| #380 dedicated mean + sigma | 3.086 | 3.830 | 0.529 | 66.76% |

The dedicated mean head is effectively at parity with #378. RMSE and pairwise are slightly better, while MAE and Pearson are slightly worse.

## Suit / Target Breakdown

Mean prediction by suit:

| suit | MAE | RMSE | Pearson | sign |
|---|---:|---:|---:|---:|
| clubs | 2.967 | 3.678 | 0.539 | 95.08% |
| diamonds | 2.931 | 3.636 | 0.547 | 96.73% |
| hearts | 3.039 | 3.771 | 0.504 | 96.04% |
| spades | 3.158 | 3.897 | 0.484 | 95.65% |

Mean prediction by target:

| target | MAE | RMSE | Pearson | sign |
|---:|---:|---:|---:|---:|
| 13 | 3.253 | 4.017 | 0.451 | 92.01% |
| 14 | 3.152 | 3.885 | 0.409 | 96.15% |
| 15 | 2.964 | 3.674 | 0.407 | 96.29% |
| 16 | 2.808 | 3.483 | 0.420 | 97.85% |
| 17 | 2.812 | 3.459 | 0.389 | 98.90% |
| 18 | 2.776 | 3.487 | 0.429 | 99.32% |
| 19 | 2.910 | 3.592 | 0.412 | 99.63% |

## Sigma Diagnostics

Sharpness:

| statistic | sigma |
|---|---:|
| mean | 3.806 |
| std | 0.379 |
| min | 2.191 |
| p10 | 3.356 |
| p25 | 3.540 |
| p50 | 3.803 |
| p75 | 4.089 |
| p90 | 4.253 |
| max | 5.278 |

Sigma vs absolute residual:

- Pearson: `0.134`
- Spearman: `0.109`

Sigma buckets:

| bucket | count | mean sigma | MAE | RMSE |
|---|---:|---:|---:|---:|
| low | 3,702 | 3.393 | 2.813 | 3.503 |
| mid | 3,702 | 3.806 | 3.031 | 3.745 |
| high | 3,702 | 4.219 | 3.415 | 4.208 |

Standardized residual:

| statistic | value |
|---|---:|
| mean | 0.038 |
| std | 1.002 |
| median | 0.007 |
| p5 | -1.545 |
| p50 | 0.007 |
| p95 | 1.735 |

Prediction interval coverage:

| interval | coverage |
|---|---:|
| `mu +/- 1 sigma` | 67.35% |
| `mu +/- 2 sigma` | 95.79% |

Sigma by suit mean:

| suit | mean sigma | std |
|---|---:|---:|
| clubs | 3.697 | 0.348 |
| diamonds | 3.676 | 0.305 |
| hearts | 3.679 | 0.375 |
| spades | 3.924 | 0.342 |

Sigma by target mean:

| target | mean sigma | std |
|---:|---:|---:|
| 13 | 4.066 | 0.214 |
| 14 | 3.913 | 0.291 |
| 15 | 3.658 | 0.307 |
| 16 | 3.532 | 0.205 |
| 17 | 3.374 | 0.261 |
| 18 | 3.387 | 0.224 |
| 19 | 3.502 | 0.233 |

Sigma by hand-strength bucket:

| bucket | mean sigma | std |
|---|---:|---:|
| low | 3.854 | 0.419 |
| mid | 3.819 | 0.354 |
| high | 3.743 | 0.352 |

## Sigma Baselines

Baselines use the same NN mean and replace sigma with a train-split residual std bucket.

| sigma source | NLL | mean sigma | sigma std | 1 sigma | 2 sigma |
|---|---:|---:|---:|---:|---:|
| NN state/action sigma | 1.834 | 3.806 | 0.379 | 67.35% | 95.79% |
| global residual std | 1.866 | 4.500 | 0.000 | 75.79% | 98.25% |
| action-index residual std | 1.847 | 4.181 | 0.248 | 72.09% | 97.35% |
| suit x target residual std | 1.847 | 4.181 | 0.248 | 72.09% | 97.35% |

The NN sigma has better NLL and much sharper intervals than the constant/bucket baselines. Its 1 sigma and 2 sigma coverage are also close to Gaussian reference levels.

## Risk-aware Ranking

Teacher best: actual `contractMargin`.

| score | best hit | top3 | pairwise | PASS vs best-bid | strongest suit |
|---|---:|---:|---:|---:|---:|
| `mu - 0.0 sigma` | 49.70% | 81.35% | 66.76% | 63.45% | 31.23% |
| `mu - 0.25 sigma` | 50.20% | 81.35% | 66.82% | 63.45% | 31.29% |
| `mu - 0.5 sigma` | 50.20% | 81.45% | 66.79% | 63.66% | 29.97% |
| `mu - 1.0 sigma` | 49.80% | 81.45% | 66.54% | 63.55% | 28.74% |

`lambda=0.25` is the best pairwise score in this fixed set, but the gain over mean-only is small. Risk-aware scoring is plausible but not strong enough to justify runtime integration by itself.

Predicted best suit distribution:

| lambda | clubs | diamonds | hearts | spades |
|---|---:|---:|---:|---:|
| 0.0 | 370 | 40 | 75 | 165 |
| 0.25 | 384 | 46 | 78 | 144 |
| 0.5 | 400 | 45 | 83 | 126 |
| 1.0 | 409 | 50 | 112 | 97 |

Predicted best target distribution:

| lambda | 13 | 14 | 15 | 16 |
|---|---:|---:|---:|---:|
| 0.0 | 474 | 113 | 54 | 9 |
| 0.25 | 465 | 113 | 61 | 13 |
| 0.5 | 454 | 119 | 66 | 15 |
| 1.0 | 399 | 131 | 101 | 37 |

## Gaussian-derived Success

`P(margin >= 0)` from `mu, sigma`:

| metric | #378 explicit success head | #380 Gaussian-derived |
|---|---:|---:|
| ROC-AUC | 0.788 | 0.782 |
| PR-AUC | 0.172 | 0.162 |
| Pairwise | 77.24% | 74.80% |

Additional fixed-validation values:

- Positive rate: `5.48%`
- Brier: `0.049`

Calibration bins:

| probability bin | count | mean prediction | observed rate |
|---|---:|---:|---:|
| 0.0-0.1 | 9,796 | 0.027 | 0.040 |
| 0.1-0.2 | 1,239 | 0.128 | 0.160 |
| 0.2-0.3 | 57 | 0.231 | 0.246 |
| 0.3-0.4 | 12 | 0.330 | 0.500 |
| 0.4-0.5 | 2 | 0.467 | 0.500 |

The Gaussian-derived success probability is useful, but it remains below the explicit #378 success head. Keep an explicit success head if success probability is a first-class runtime score.

## Conclusion

The mean + sigma approach is promising for aleatoric outcome uncertainty:

- Mean prediction stayed at #378 parity.
- Sigma did not collapse to a constant.
- Sigma is positively related to absolute residual.
- Low-sigma samples are substantially easier than high-sigma samples.
- Interval coverage is well behaved.
- NN sigma improves NLL over global/action/suit-target std baselines.

The runtime score should not move directly to sigma-only risk adjustment yet. A small `lambda=0.25` risk penalty improved pairwise ranking by only `0.06pp`, and stronger penalties hurt strongest-suit relation. For runtime bidding score design, prefer mean margin plus the explicit success head from #378, with sigma as an optional risk diagnostic.
