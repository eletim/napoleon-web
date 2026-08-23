# Issue 425 Raise Head Ablation

Base: `feature/issue-202-non-playing-ai` after #423.

## Input Audit

Current #423 compact bidding input (`BIDDING_MODEL_INPUT_FEATURE_COUNT=278`, schema v2) includes:

- candidate hand: `selfHandMask`
- legal mask: `legalBidMask`
- current bid presence: `highestBidPresent`
- current bid owner: `highestBidPlayerOneHot` relative to acting candidate
- current bid suit/target: `highestBidSuitOneHot`, `highestBidTargetPointCardsOneHot`
- pass count: `consecutivePassCountOneHot`
- bid ownership table by bid position: `biddingBidOwnerTableOneHot`

It does not include an explicit scalar bidding step or chronological PASS/BID sequence in the bidding model input. Candidate is always relative seat 0 in the compact observation; absolute `candidateSeatIndex` is sample metadata.

## Dataset

All rollout teachers preserve #423 semantics: candidate is the contract owner/Napoleon for the requested target/suit before adjutant/exchange/play; no downstream bidding is executed.

Generated datasets:

| dataset | pairs | fixed hands | repeats | rollouts | opening | raise |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `/tmp/issue425-mixed-1k-n50` | 1,000 | 250 | 50 | 50,000 | 490 | 510 |
| `/tmp/issue425-mixed-2k-n50` | 2,000 | 500 | 50 | 100,000 | 987 | 1,013 |
| `/tmp/issue425-mixed-5k-n50` | 5,000 | 1,250 | 50 | 250,000 | 2,490 | 2,510 |

N-repeat stability on 40 matched pairs:

| compare | margin mean MAE | margin std MAE | win-rate MAE |
| --- | ---: | ---: | ---: |
| N50 vs N100 | 0.3095 | 0.1874 | 0.0280 |
| N100 vs N200 | 0.2022 | 0.1133 | 0.0184 |
| N50 vs N200 | 0.3978 | 0.2291 | 0.0366 |

## Held-Out Training Metrics

5k mixed final split, fixedHandId/sourceStateKey separated:

| model | final MAE | final pWin MAE | opening MAE | raise MAE | same-hand pairwise |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 M1 | 0.6792 | 0.0533 | 0.5961 | 0.7656 | 0.9227 |
| B1 M2 | 0.5557 | 0.0498 | 0.5039 | 0.6067 | 0.9453 |

## #421/#422 Final Diagnostics

Representative #421/#422 states were not mixed into training.

| model | #422 sign | #422 false PASS | #422 EV bias | #421 sign | #421 false PASS | #421 EV bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 #423 M2 | 88.9% | 5/45 | -3.53 | 67.5% | 12/40 | -8.75 |
| B1 M1 | 71.1% | 13/45 | -4.65 | 70.0% | 11/40 | -8.25 |
| B1 M2 | 86.7% | 6/45 | -3.72 | 60.0% | 15/40 | -9.19 |
| B2 M2 minimal context | 82.2% | 6/45 | -3.79 | 65.0% | 13/40 | -8.90 |

Conclusion: head separation alone gives only a small raise improvement in M1 and hurts opening calibration. Minimal context features did not explain the #421 raise underestimation. The #425 model should not be wired into runtime yet.
