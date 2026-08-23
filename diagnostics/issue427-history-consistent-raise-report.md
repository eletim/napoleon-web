# Issue 427 History-Consistent Raise Teacher Report

Base: `feature/issue-202-non-playing-ai` at `58282fc`.

Runtime wiring, reward changes, Citizen/Adjutant retraining, PPO, and opening/raise head separation were not changed.

## Generator Semantics

New dataset sample type: `history-consistent-raise-margin-sample`.

Raise state generation:

1. Fix candidate hand.
2. Shuffle remaining deck by `dealSeed` into the other four hands and kitty.
3. Start bidding from the true initial state: starter/current player is `player-0`, highest bid is null, history is empty.
4. Run all bidding actions naturally with the existing RuleBased/Conservative 50:50 seat policy mix. The candidate uses the same source-bidding policy mix while the source state is being generated.
5. Collect only states where it is candidate's turn and a current highest bid exists.
6. Encode the collected visible bidding observation/history as model input.
7. Evaluate legal raise actions from that exact source state by fixing candidate=Napo, contract owner=candidate, and target/suit=evaluated raise.
8. Do not run downstream bidding. Continue normal adjutant selection, kitty pickup, exchange/bury, and play.

No hidden hands are reshuffled after the raise state is collected. Each raw rollout stores `dealSeed`, `sourceStateKey`, `biddingHistorySummary`, `hiddenDealChecksum`, and invariant checks.

## Dataset

Primary balanced raise dataset:

- path: `/tmp/issue427-raise-main-5k`
- fixed hands: 1,250
- unique deal seeds: 1,250
- unique raise states: 1,250
- evaluated action pairs: 5,000
- rollout count: 5,000
- evaluated suit counts: S/H/D/C = 1,250 each

Strong-hand raise dataset used for the final T1 candidate:

- path: `/tmp/issue427-raise-strong-5k`
- fixed hands: 1,250
- unique deal seeds: 1,250
- unique raise states: 1,250
- evaluated action pairs: 5,000
- rollout count: 5,000
- evaluated suit counts: S/H/D/C = 1,250 each
- empirical margin mean: -4.284
- empirical win-rate mean: 0.188

Mixed dataset for T2:

- path: `/tmp/issue427-mixed-opening5k-strongraise5k`
- inputs: `/tmp/issue423-main-5k-n50-v2`, `/tmp/issue427-raise-strong-5k`
- samples: 10,000
- fixed hands: 2,500
- unique raise states: 1,250
- rollout count: 255,000

Split remains by `fixedHandId`; raise samples additionally expose `sourceStateKey` and `dealSeed` for leakage audits. No #421/#422 representative states were added to training.

## Models

All models use the #423 single-head M2 architecture:

`compact bidding input + selected action -> MLP [512,512,256,256] -> margin mean/std`.

- B0: `/tmp/issue423-model-5k-M2/checkpoint.pt`
- T1: `/tmp/issue427-t1-strong-raise-ft/checkpoint.pt`
  - initialized from B0
  - raise-only fine-tune on `/tmp/issue427-raise-strong-5k`
  - best epoch 3
- T2: `/tmp/issue427-t2-strong-mixed-retrain/checkpoint.pt`
  - scratch mixed retrain on opening 5k + strong raise 5k
  - best epoch 11
- T2init was checked only as a guardrail: `/tmp/issue427-t2init-strong-mixed-ft/checkpoint.pt`

Single-rollout raise samples have empirical std 0. The M2 std loss ignores zero-std single-rollout targets, so the sigma head is not trained to collapse on deterministic one-deal labels.

## Held-Out

T1 strong raise held-out:

- final samples: 500
- margin MAE/RMSE/Pearson/bias: 2.738 / 3.419 / 0.511 / -0.021
- pWin MAE/RMSE/Pearson/Brier: 0.274 / 0.363 / 0.306 / 0.132
- same-state pairwise/top-action: 0.724 / 0.528

T2 strong mixed held-out:

- final samples: 1,000
- margin MAE/RMSE/Pearson/bias: 1.716 / 2.487 / 0.704 / 0.160
- pWin MAE/RMSE/Pearson/Brier: 0.138 / 0.251 / 0.525 / 0.063
- same-state pairwise/top-action: 0.835 / 0.752

T2 has better mixed held-out ranking, but it does not transfer to #421/#422 diagnostics.

## #421 Raise Diagnostic

40 current-bid representative states. Same representative raise actions and PASS/Citizen EV baseline as #423.

| model | sign accuracy | false PASS | false raise | raise rate | EV MAE | EV bias |
|---|---:|---:|---:|---:|---:|---:|
| B0 #423 M2 | 67.5% | 12 | 1 | 45.0% | 9.37 | -8.75 |
| T1 raise-only FT | 80.0% | 7 | 1 | 57.5% | 7.34 | -5.55 |
| T2 mixed scratch | 47.5% | 21 | 0 | 20.0% | 11.01 | -11.00 |
| T2init mixed FT | 65.0% | 13 | 1 | 42.5% | 9.35 | -8.87 |

T1 clearly improves the #421 target: sign accuracy +12.5pp, false PASS -5, EV bias +3.20 toward zero.

## #422 Opening Diagnostic

45 no-current-bid representative states.

| model | sign accuracy | false PASS | false BID | BID rate | EV MAE | EV bias |
|---|---:|---:|---:|---:|---:|---:|
| B0 #423 M2 | 88.9% | 5 | 0 | 77.8% | 4.41 | -3.53 |
| T1 raise-only FT | 88.9% | 0 | 5 | 100.0% | 3.94 | -0.13 |
| T2 mixed scratch | 73.3% | 12 | 0 | 62.2% | 5.42 | -4.99 |
| T2init mixed FT | 80.0% | 9 | 0 | 68.9% | 4.72 | -3.98 |

T1 preserves sign accuracy and removes false PASS, but it becomes more aggressive on opening with 5 false BID cases. T2 variants are not acceptable because they degrade both #421 and #422.

## Old vs New Teacher Check

Small comparison on 80 fixed hand/action pairs from the strong raise dataset:

- old Napoleon-fixed hidden-reshuffle teacher: margin mean -3.516, win-rate mean 0.256
- new history-consistent one-deal teacher: margin mean -3.750, win-rate mean 0.213
- new minus old: margin -0.234, win-rate -0.043

The average is not simply higher. The important difference is semantic: the new label is conditioned on the same hidden deal that produced the visible bidding history, while the old teacher breaks that relation. The #421 improvement comes with the same single-head model and no runtime/reward changes, so the improvement is attributable to teacher-data semantics and stronger raise-state coverage, not model structure.

## Judgment

Adopt T1 as the best offline candidate from this issue. It fixes the main #421 under-raise symptom materially while keeping #422 sign accuracy, but the new opening false BID rate means runtime wiring should not proceed directly without an opening guardrail/calibration pass.

## Tests

Passed:

- `pnpm --filter @napoleon/training-data typecheck`
- `pnpm --filter @napoleon/training-data test -- generateFixedHandBiddingMarginDataset.test.ts`
- `pnpm --filter @napoleon/training-data... build`
- `UV_CACHE_DIR=/tmp/uv-cache PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --project python --extra dev python -m pytest python/tests/unit/test_fixed_hand_margin_training.py`
- `UV_CACHE_DIR=/tmp/uv-cache PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --project python --extra dev mypy python/src/napoleon_ml/bidding_q/fixed_hand_margin_training.py python/src/napoleon_ml/cli/train_fixed_hand_bidding_margin.py`
