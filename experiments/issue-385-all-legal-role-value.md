# Issue #385 All-Legal Role Value Experiment

Base commit: `c84f9a449c82060fb4353b1d8e67ed75ec8f8e93`

## Dataset

- Path: `/tmp/napoleon-issue385-alllegal-1000`
- Manifest SHA-256: `adef4dee5674866c05a81900179266e517555ac9cfbfe678b7418050047b2057`
- Summary SHA-256: `687b99d9a7012a5dcb52231f721a64f7d26162455569388f93842e0553f8518e`
- Generation mode: `all-legal-actions-role-value-v1`
- Source states: `1,000`
- Total rollouts / forced state-action pairs: `25,621`
- Repeats: `1`
- Bidding policy: `bidding-compact278-reward-attribution-iter029`
- Playing policy: `ppo-separated-v1000`
- Inference device: `cpu`
- Fallback / illegal forced action count: `0 / 0`

Full-pool role coverage:

| role | samples | states | pass | bid | action-index coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Citizen | 2,592 | 709 | 705 | 1,887 | 21 / 29 |
| Adjutant | 951 | 245 | 241 | 710 | 17 / 29 |

Full-pool primary teacher coverage:

| role | teacher | samples | states | ranking states | pairs | diff pairs | tie rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Citizen | `negative-contract-margin` | 2,592 | 709 | 363 | 8,044 | 5,546 | 31.05% |
| Adjutant | `contract-margin` | 951 | 245 | 139 | 2,969 | 2,119 | 28.63% |

Adjutant coverage was sufficient with natural all-legal collection, so no rule changes or role-forcing were used.

## Training

Shared config:

- Architecture: `compact278 -> MLP [512,512,256,256] -> value[29]`
- Loss: selected-action-only MSE
- Target standardization: train split statistics only
- Optimizer: `AdamW`, lr `1e-3`, weight decay `1e-4`
- Batch size: `128`
- Max epochs: `80`
- Patience: `10`
- Seed: `385`
- Device: `cpu`

Role-stratified validation used stateKey-disjoint splits and selected role-matching ranking states first.

| role | train states | validation states | validation hash | best epoch |
| --- | ---: | ---: | --- | ---: |
| Citizen | 750 | 250 | `e6907ee78107496b2c94bb92a44a0b77ac9f0c1dcab3f4869b9a73b31b9ab54c` | 2 |
| Adjutant | 900 | 100 | `3e33e1a73fd572ba82a402ad239d240e2012b2e875443ccdacf2824e2bd227c2` | 12 |

## Validation Results

| role | samples | states | MAE | RMSE | Pearson | pairwise | pairs | diff pairs | ranking states | best | top3 | tie rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Citizen | 1,669 | 250 | 2.572 | 3.108 | 0.083 | 60.71% | 6,337 | 4,457 | 250 | 46.40% | 78.40% | 29.67% |
| Adjutant | 650 | 100 | 2.796 | 3.411 | 0.259 | 62.16% | 2,392 | 1,739 | 100 | 43.00% | 74.00% | 27.30% |

Baseline comparison:

| role | NN MAE/RMSE | global mean MAE/RMSE | action-index mean MAE/RMSE | suit x target mean MAE/RMSE |
| --- | --- | --- | --- | --- |
| Citizen | 2.572 / 3.108 | 2.589 / 3.123 | 2.598 / 3.129 | 2.598 / 3.129 |
| Adjutant | 2.796 / 3.411 | 2.886 / 3.535 | 3.013 / 3.622 | 3.013 / 3.622 |

Validation action coverage:

| role | pass | bid | suits | targets | role-matching action count distribution |
| --- | ---: | ---: | --- | --- | --- |
| Citizen | 247 | 1,422 | spades 423, hearts 386, diamonds 338, clubs 275 | 13:535, 14:529, 15:284, 16:55, 17:19 | 2:22, 3:27, 4:26, 5:52, 6:12, 7:15, 8:24, 9:30, 10:5, 11:6, 12:8, 13:14, 15:2, 16:5, 17:1, 20:1 |
| Adjutant | 98 | 552 | spades 162, hearts 151, diamonds 130, clubs 109 | 13:203, 14:215, 15:114, 16:20 | 2:8, 3:10, 4:14, 5:24, 6:2, 7:8, 8:5, 9:13, 10:2, 11:4, 12:2, 13:5, 16:1, 17:2 |

## Verdict

Citizen value learning is established for this dataset: coverage exceeds the configured `1,000` diff-pair / `200` ranking-state threshold, regression beats global/action/suit-target baselines, and pairwise ranking is above random on `4,457` diff pairs.

Adjutant value learning is established for this dataset: coverage exceeds the configured `200` diff-pair / `100` ranking-state threshold, regression beats global/action/suit-target baselines, and pairwise ranking is above random on `1,739` diff pairs. This is no longer the low-pair #383 condition.

Both roles have sufficient coverage and positive state-conditioned signal to proceed to the next role-probability integration issue. This issue does not implement the integration.

## Artifacts

- Citizen output: `/tmp/napoleon-issue385-citizen-alllegal-1000`
- Citizen checkpoint SHA-256: `9ee1239345b2d14a769ed08eea85c5c7461e899c8fa6869646cebfb88ac0a71c`
- Adjutant output: `/tmp/napoleon-issue385-adjutant-alllegal-1000-v100`
- Adjutant checkpoint SHA-256: `17cb938d6bff794d2af4b99c21f56b525bd0828972c3daa965bb47234a555b84`

## Tests

- `pnpm --filter @napoleon/training-data test -- generateBiddingQCounterfactualDataset`: pass
- `pnpm -r typecheck`: pass
- `UV_CACHE_DIR=/tmp/uv-cache UV_PROJECT_ENVIRONMENT=/tmp/napoleon-issue385-venv uv run --project python --extra dev mypy python/src/napoleon_ml/bidding_q/role_value_training.py python/src/napoleon_ml/cli/train_bidding_role_value_q.py python/tests/unit/test_bidding_role_value_q.py`: pass
- `UV_CACHE_DIR=/tmp/uv-cache UV_PROJECT_ENVIRONMENT=/tmp/napoleon-issue385-venv PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --project python --extra dev python -m pytest python/tests/unit/test_bidding_role_value_q.py`: pass
- `pnpm -r test`: fails in existing `apps/server/test/agentRegistry.test.ts` full-policy fixture expectation; targeted training-data tests pass.
- `UV_CACHE_DIR=/tmp/uv-cache UV_PROJECT_ENVIRONMENT=/tmp/napoleon-issue385-venv PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --project python --extra dev python -m pytest python/tests/unit`: fails in existing exchange multiphase fixture/schema tests; role-value unit tests pass.
