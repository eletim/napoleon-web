# Issue 423 Napoleon-Fixed Contract Value Report

Base: `feature/issue-202-non-playing-ai` at `3939710`

## State Construction

Napoleon-fixed rollouts use `createContractEstablishedState(initialState, contract)` in `@napoleon/game-core`. The generator first creates a legal fixed-hand deal, then establishes a final contract with:

- candidate role/contract owner: candidate player
- declared target/suit: requested pair target/suit
- phase: `choosing-adjutant`
- bidding state: `null`

No BID/PASS action is applied during rollout. The normal engine then handles adjutant selection, kitty pickup, exchange/bury, and play.

## Raw Invariants

Every raw rollout verifies:

- candidate role is Napoleon
- contract owner is candidate
- target/suit match the requested contract
- downstream bidding action count is 0
- candidate hand is fixed at rollout start
- deck conservation has 53 unique cards

Smoke: 20 pairs x N=20 completed 400 raw rollouts with no invariant failures.

## N Stability

Same 8 fixed-hand x contract pairs:

| repeats | rollouts | mean margin mean | win-rate mean |
|---:|---:|---:|---:|
| 50 | 400 | -4.725 | 0.140 |
| 100 | 800 | -4.556 | 0.159 |
| 200 | 1600 | -4.686 | 0.148 |

N=50 vs N=200 average absolute pair delta: margin `0.2725`, win rate `0.0400`. Main teacher used N=50.

## Dataset Scale

| dataset | fixed hands | pairs | rollouts | repeats |
|---|---:|---:|---:|---:|
| 1k | 250 | 1000 | 50000 | 50 |
| 2k | 500 | 2000 | 100000 | 50 |
| 5k | 1250 | 5000 | 250000 | 50 |

5k summary: target 13-19 covered, suits S/H/D/C covered, empirical margin mean `-4.995`, empirical win-rate mean `0.1459`.

Split for 5k final model seed 4236: train 1000 hands / 4000 pairs, validation 125 hands / 500 pairs, final 125 hands / 500 pairs. Splits are by `fixedHandId`; no hand crosses splits.

## Models

Architecture: compact bidding observation + selected action head, MLP `[512,512,256,256]`.

- M0: #411 old-semantics M2 baseline trained on forced-BID-then-continue-bidding dataset.
- M1: Napoleon-fixed mean-only.
- M2: Napoleon-fixed mean + empirical std direct regression.

Teachers:

- primary: `empiricalMarginMean`
- sigma: `empiricalMarginStd`
- pWin reference: `empiricalWinRate`

## Same Napoleon-Fixed Held-Out

500 held-out pairs / 125 fixed hands / 25000 raw rollouts.

| model | mean RMSE | mean MAE | pWin MAE | pWin RMSE | pWin Pearson | Brier | pairwise | top action |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 old #411 M2 | 1.175 | 0.902 | 0.067 | 0.100 | 0.771 | 0.0100 | 0.924 | 0.840 |
| M1 new 5k | 0.583 | 0.448 | 0.047 | 0.0666 | 0.947 | 0.00443 | 0.948 | 0.944 |
| M2 new 5k | 0.649 | 0.503 | 0.045 | 0.0651 | 0.938 | 0.00424 | 0.929 | 0.888 |

M2 has the best pWin/Brier and explicit sigma head; M1 has better same-hand top-action on this split. New Napoleon-fixed semantics clearly improve calibration over M0.

## #421 Raise States

40 current-bid representative states from #421. PASS/Citizen EV remains the #421 value; only Napoleon raise EV is replaced by new M2.

| scorer | sign accuracy | false PASS | false raise | positive decision rate | EV MAE | EV bias |
|---|---:|---:|---:|---:|---:|---:|
| old #420/#411 | 30.0% | 28 | 0 | 2.5% | 10.01 | -9.95 |
| new 5k M2 | 67.5% | 12 | 1 | 45.0% | 9.37 | -8.75 |

Raise underestimation is materially reduced, but not eliminated.

## #422 Opening States

45 no-current-bid representative states from #422. All-Pass baseline is 0.

| scorer | sign accuracy | false PASS | false BID | BID rate | EV MAE | EV bias |
|---|---:|---:|---:|---:|---:|---:|
| old #420/#411 | 71.1% | 13 | 0 | 60.0% | 5.94 | -5.84 |
| new 5k M2 | 88.9% | 5 | 0 | 77.8% | 4.41 | -3.53 |

Opening underestimation is substantially improved.

## Tests

Passed:

- `pnpm --filter @napoleon/training-data typecheck`
- `pnpm --filter @napoleon/game-core test -- game.test.ts`
- `pnpm --filter @napoleon/training-data test -- generateFixedHandBiddingMarginDataset.test.ts`
- `pnpm --filter @napoleon/training-data... build`
- `cd python && UV_CACHE_DIR=/tmp/uv-cache PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/unit/test_fixed_hand_margin_training.py`

Additional validation:

- 5k Napoleon-fixed generation: 250000 raw rollouts, invariant failures: 0
- checkpoint save/load fixture passed in Python unit tests

## Judgment

The new Napoleon-fixed teacher has the intended semantics and improves held-out pWin calibration, same-hand ranking, and representative opening/raise false PASS rates. Runtime wiring was intentionally not changed in this issue. It is reasonable to proceed to a runtime-wiring issue using M2 as the calibrated pWin source, while keeping a guardrail for current-bid raise states because 12/40 false PASS remain in #421 diagnostics.
