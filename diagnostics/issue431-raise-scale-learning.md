# Issue #431 raise scale learning curve

Frozen selection: `benchmarks/bidding-margin-policies/frozen-raise-v1` (`50k`).

Fixed conditions: #423 single-head M2, #427 history-consistent raise teacher, #423 checkpoint fine-tune, unchanged reward/EV/runtime decision semantics, no head split, no Citizen/Adjutant/PPO retraining.

## Dataset scale

| scale | action pairs | fixed hands | dealSeeds | raise states | hand bucket | current target | suit |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 5k | 5000 | 1250 | 1250 | 1250 | strong=5000 | {'14': 4356, '15': 644} | {'clubs': 1250, 'diamonds': 1250, 'hearts': 1250, 'spades': 1250} |
| 10k | 10000 | 2500 | 2500 | 2500 | strong=10000 | {'14': 8824, '15': 1176} | {'clubs': 2500, 'diamonds': 2500, 'hearts': 2500, 'spades': 2500} |
| 20k | 20000 | 5000 | 5000 | 5000 | strong=20000 | {'14': 17480, '15': 2500, '16': 20} | {'clubs': 5000, 'diamonds': 5000, 'hearts': 5000, 'spades': 5000} |
| 50k | 50000 | 12500 | 12500 | 12500 | strong=50000 | {'14': 44132, '15': 5816, '16': 52} | {'clubs': 12500, 'diamonds': 12500, 'hearts': 12500, 'spades': 12500} |

Leakage/invariant checks: all scales have `invariantViolationCount=0`, `maxDealSeedsPerFixedHand=1`, `maxRaiseStatesPerFixedHand=1`, and `maxHiddenDealsPerRaiseState=1`. #421/#422 representative modelInput hash intersections are 0 for every scale.

## Held-out learning curve

| scale | margin MAE | RMSE | Pearson | bias | pWin MAE | pWin RMSE | pWin Pearson | Brier | pairwise | top-action |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5k | 2.7377 | 3.4190 | 0.5111 | -0.0212 | 0.2736 | 0.3633 | 0.3059 | 0.1320 | 0.7238 | 0.5280 |
| 10k | 2.7040 | 3.2643 | 0.5787 | 0.0654 | 0.2583 | 0.3421 | 0.4621 | 0.1171 | 0.7141 | 0.5800 |
| 20k | 2.7695 | 3.4195 | 0.5591 | 0.1434 | 0.2739 | 0.3579 | 0.4300 | 0.1281 | 0.7294 | 0.5720 |
| 50k | 2.7309 | 3.3689 | 0.5692 | -0.1095 | 0.2620 | 0.3527 | 0.4438 | 0.1244 | 0.7263 | 0.6120 |

## #421 raise diagnostic

| scale | sign | false PASS | false raise | raise rate | EV MAE | EV bias | Pearson | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5k | 0.800 | 7 | 1 | 0.575 | 7.340 | -5.554 | 0.555 | 0.559 |
| 10k | 0.725 | 10 | 1 | 0.500 | 7.617 | -6.304 | 0.523 | 0.551 |
| 20k | 0.750 | 7 | 3 | 0.625 | 7.817 | -5.552 | 0.491 | 0.504 |
| 50k | 0.800 | 6 | 2 | 0.625 | 7.345 | -5.456 | 0.510 | 0.503 |

## #422 opening diagnostic

| scale | sign | false PASS | false BID | BID rate | EV MAE | EV bias | Pearson | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5k | 0.889 | 0 | 5 | 1.000 | 3.940 | -0.127 | 0.508 | 0.555 |
| 10k | 0.867 | 1 | 5 | 0.978 | 3.795 | -0.996 | 0.567 | 0.613 |
| 20k | 0.889 | 0 | 5 | 1.000 | 4.315 | -0.610 | 0.359 | 0.382 |
| 50k | 0.889 | 0 | 5 | 1.000 | 3.850 | -0.019 | 0.482 | 0.494 |

## Runtime 1,000 games

| model | opening BID | raise opp | raise rate | final Napoleon | success | candidate RR | safety |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A-current | 1.0000 | 0 | None | 1.0000 | 0.0010 | -3.9808 | illegal=0, fallback=0, inference=0, crash=0 |
| B-5k | 0.7340 | 1089 | 0.0790 | 0.0920 | 0.3380 | 0.0054 | illegal=0, fallback=0, inference=0, crash=0 |
| B-10k | 0.6450 | 1108 | 0.0505 | 0.0620 | 0.3370 | 0.0238 | illegal=0, fallback=0, inference=0, crash=0 |
| B-20k | 0.6730 | 1098 | 0.0838 | 0.0950 | 0.3390 | -0.0018 | illegal=0, fallback=0, inference=0, crash=0 |
| B-50k | 0.7290 | 1088 | 0.0809 | 0.0900 | 0.3370 | 0.0858 | illegal=0, fallback=0, inference=0, crash=0 |

## Runtime 5,000 games top candidates

| model | opening BID | raise opp | raise rate | final Napoleon | success | candidate RR | target 13-19 | suits | safety |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| A-current | 1.0000 | 0 | None | 1.0000 | 0.0032 | -3.9386 | {'13': 0, '14': 0, '15': 0, '16': 0, '17': 1, '18': 3, '19': 4996} | {'clubs': 40, 'diamonds': 798, 'hearts': 10, 'spades': 4152} | illegal=0, fallback=0, inference=0, crash=0 |
| B-50k | 0.7296 | 5404 | 0.0703 | 0.0822 | 0.3322 | -0.0204 | {'13': 228, '14': 2608, '15': 1857, '16': 293, '17': 14, '18': 0, '19': 0} | {'clubs': 1247, 'diamonds': 1438, 'hearts': 1272, 'spades': 1043} | illegal=0, fallback=0, inference=0, crash=0 |
| B-10k | 0.6502 | 5476 | 0.0495 | 0.0656 | 0.3296 | -0.1104 | {'13': 248, '14': 2650, '15': 1799, '16': 289, '17': 14, '18': 0, '19': 0} | {'clubs': 1253, 'diamonds': 1403, 'hearts': 1313, 'spades': 1031} | illegal=0, fallback=0, inference=0, crash=0 |
| B-5k | 0.7234 | 5402 | 0.0698 | 0.0844 | 0.3296 | -0.0854 | {'13': 229, '14': 2620, '15': 1791, '16': 344, '17': 15, '18': 1, '19': 0} | {'clubs': 1262, 'diamonds': 1417, 'hearts': 1313, 'spades': 1008} | illegal=0, fallback=0, inference=0, crash=0 |

## Decision

50k had the best 5,000-game candidate relative reward among promoted candidates, no safety failures, #421 sign tied with 5k while reducing false PASS from 7 to 6, and #422 opening remained at 88.9% sign / 0 false PASS / 5 false BID. Held-out metrics plateaued rather than improved monotonically, so this is a modest data-scale gain, not an architecture/reward/runtime change.

The curve is mostly plateaued in held-out metrics, but 50k is the best runtime candidate after the 5,000-game panel and improves #421 false PASS versus the 5k T1 baseline without degrading #422. Adopt `frozen-raise-v1` and treat bidding AI as frozen unless a new clear issue appears.
