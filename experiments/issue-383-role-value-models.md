# Issue #383 Citizen / Adjutant Bidding Value Models

Base branch: `feature/issue-202-non-playing-ai`

Runtime bidding-agent integration was intentionally not changed.

## Dataset

- Reused dataset: `/tmp/napoleon-issue377-qdataset-22000x1`
- Source states: `22,000`
- Raw samples: `122,765`
- Manifest SHA256: `e83730b05cb97c4f107c9fcb2b068ff9bea6cf6b0859a4e83f8a9b46c590eec4`
- Fixed validation file: `/tmp/napoleon-issue377-learning-curve/validation-state-keys.json`
- Fixed validation hash: `9ff43c57867712f5352467b3d56e397046b02c3c11b22a579ab0298a9777d57a`
- Train / validation states: `20,000 / 2,000`
- No rollout was regenerated for the main experiment.

## Existing Coverage

Full pool role coverage:

| role | samples | states | pass | bid | action coverage |
|---|---:|---:|---:|---:|---:|
| Citizen | 12,217 | 11,911 | 11,585 | 632 | 22 / 29 |
| Adjutant | 2,994 | 2,939 | 2,825 | 169 | 18 / 29 |

Teacher coverage in the full pool:

| role | teacher | samples | states | ranking states | pairs | diff pairs | tie rate |
|---|---|---:|---:|---:|---:|---:|---:|
| Citizen | `coalition-side-point-cards` | 12,217 | 11,911 | 297 | 315 | 253 | 19.68% |
| Citizen | `negative-contract-margin` | 12,217 | 11,911 | 297 | 315 | 278 | 11.75% |
| Citizen | `citizen-margin` | 12,217 | 11,911 | 297 | 315 | 253 | 19.68% |
| Citizen | `contract-failure-binary` | 12,217 | 11,911 | 297 | 315 | 78 | 75.24% |
| Adjutant | `contract-margin` | 2,994 | 2,939 | 54 | 56 | 51 | 8.93% |
| Adjutant | `napoleon-side-point-cards` | 2,994 | 2,939 | 54 | 56 | 44 | 21.43% |
| Adjutant | `contract-success` | 2,994 | 2,939 | 54 | 56 | 17 | 69.64% |

Conclusion before training:

- Citizen has enough samples for mean regression, but only limited same-state ranking coverage.
- Adjutant has too few pairwise comparisons for reliable ranking evaluation.
- Citizen primary teacher: `negative-contract-margin = finalDeclaredTarget - napoleonSidePointCards`.
- Adjutant primary teacher: `contract-margin`.
- Binary teachers have high tie rates and were not chosen as primary mean-value teachers.

## Dataset Generation Extension

The TypeScript counterfactual generator now supports:

`all-legal-actions-role-value-v1`

This action plan forces every legal bidding action for each collected source state. It does not modify game rules or role outcomes; it only increases naturally generated same-state action coverage. Use it when generating a role-value dataset to improve Citizen/Adjutant pair coverage.

CLI:

```text
node scripts/generate-bidding-q-counterfactual-dataset.mjs \
  --action-plan-id all-legal-actions-role-value-v1 \
  ...
```

## Model

Separate models:

- Citizen architecture: `bidding-citizen-value-mlp-v1`
- Adjutant architecture: `bidding-adjutant-value-mlp-v1`

Shared shape:

```text
compact278 -> MLP [512,512,256,256] -> value[29]
```

Training:

- selected-action-only MSE
- role mask: only matching finalRole samples
- NoContract / AllPass masked
- target standardization from train split only
- lr `1e-3`
- batch size `128`
- weight decay `1e-4`
- early stopping on validation MSE, patience `10`
- device `cpu`

## Citizen Result

- Output: `/tmp/napoleon-issue383-citizen-20k`
- Teacher: `negative-contract-margin`
- Target standardization: mean `4.489367`, std `3.574033`
- Best epoch: `7`
- Stopped epoch: `17`
- Fixed validation teacher samples / states: `1,072 / 1,041`

Regression:

| metric | NN | global mean | action mean | suit x target mean |
|---|---:|---:|---:|---:|
| MAE | 2.868 | 2.978 | 2.981 | 2.981 |
| RMSE | 3.536 | 3.649 | 3.649 | 3.649 |
| Pearson | 0.248 | n/a | 0.011 | 0.011 |

Ranking:

| metric | value |
|---|---:|
| ranking states | 30 |
| pair count | 32 |
| different pair count | 28 |
| teacher tie rate | 12.50% |
| pairwise accuracy | 64.29% |
| best-action hit | 70.00% |
| top3 hit | 100.00% |
| strongest suit match | 21.74% |

By action type:

| action type | samples | MAE | RMSE | Pearson |
|---|---:|---:|---:|---:|
| pass | 1,014 | 2.883 | 3.559 | 0.251 |
| bid | 58 | 2.620 | 3.101 | 0.155 |

Citizen verdict: promising for mean value. The NN beats simple regression baselines and pairwise is above random, but the fixed validation ranking set has only 28 different pairs. Generate an all-legal role-value dataset before using this in runtime scoring.

## Adjutant Result

- Output: `/tmp/napoleon-issue383-adjutant-20k`
- Teacher: `contract-margin`
- Target standardization: mean `-3.303297`, std `3.335176`
- Best epoch: `6`
- Stopped epoch: `16`
- Fixed validation teacher samples / states: `264 / 261`

Regression:

| metric | NN | global mean | action mean | suit x target mean |
|---|---:|---:|---:|---:|
| MAE | 2.596 | 2.700 | 2.700 | 2.700 |
| RMSE | 3.209 | 3.345 | 3.345 | 3.345 |
| Pearson | 0.283 | 0.000 | 0.025 | 0.025 |

Ranking:

| metric | value |
|---|---:|
| ranking states | 3 |
| pair count | 3 |
| different pair count | 3 |
| teacher tie rate | 0.00% |
| pairwise accuracy | 66.67% |
| best-action hit | 66.67% |
| top3 hit | 100.00% |
| strongest suit match | 100.00% |

By action type:

| action type | samples | MAE | RMSE | Pearson |
|---|---:|---:|---:|---:|
| pass | 254 | 2.583 | 3.194 | 0.281 |
| bid | 10 | 2.919 | 3.570 | 0.158 |

Adjutant verdict: not enough existing data for a reliable ranking conclusion. Mean regression is above baseline, but fixed validation has only 264 samples and 3 different comparison pairs. Treat this as a low-confidence smoke result, not a successful Adjutant value model.

## Conclusion

Citizen value learning is directionally useful but needs a role-value dataset with more same-state action comparisons. Adjutant should not proceed to runtime scoring from the existing #377 dataset. The next step is generating a dataset with `all-legal-actions-role-value-v1`, then re-running Citizen/Adjutant training with fixed and role-stratified validation coverage.
