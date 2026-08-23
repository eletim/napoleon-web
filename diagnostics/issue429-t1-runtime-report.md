# Issue #429 T1 Runtime Wiring Evaluation

Date: 2026-08-23

## Runtime wiring

- Candidate model: Issue #427 T1, exported as a repo-managed ONNX margin artifact at `benchmarks/bidding-margin-policies/issue427-t1-strong-raise-ft/`.
- Architecture: #423 single-head M2 fixed-hand bidding margin model. No opening/raise head split, no minimal-context path, no reward changes.
- Runtime loader: `loadRepoManagedBiddingMarginPolicyBenchmark()` validates `margin.onnx`, `margin.onnx.data`, `margin.json`, and `export-report.json` with fixed SHA-256 values before inference.
- Runtime dependency note: the runtime loads only the repo-managed artifact paths above. The original `/tmp` checkpoint was used once for export and is not read by runtime code or evaluation scripts.

## Decision rule

- Every legal BID/raise action allowed by `legalBidMask` is evaluated.
- T1 mean margin and sigma are converted to `pWin = P(margin > 0)`.
- Napoleon relative EV is `EV_N = pWin * (7 / 4) + (1 - pWin) * (-3 / 4)`.
- Opening: choose the legal BID with max `EV_N` only when it is positive; otherwise PASS with all-pass EV 0.
- Current bid exists: choose the legal raise with max `EV_N` only when it beats the existing CriticEv Citizen PASS EV; otherwise PASS.
- Adjutant, Napoleon-after-PASS, NoContract, reward redesign, PPO, and playing AI retraining were not changed.

## Evaluation setup

- Same seed panel for both conditions.
- Seeds: `429000000..429004999`.
- Games: 5,000 per condition after a 20-game smoke run.
- A: current runtime bidding AI (`CriticEvBiddingAgent`).
- B: T1 runtime (`T1NapoleonEvBiddingAgent`).
- Playing policy, exchange policy, adjutant policy, opponents, and game seeds were shared between A/B.
- Smoke command:
  `node scripts/evaluate-issue429-t1-runtime.mjs --games 20 --start-seed 429000000 --output /tmp/issue429-smoke-20.json --progress`
- Main command:
  `node scripts/evaluate-issue429-t1-runtime.mjs --games 5000 --start-seed 429000000 --output /tmp/issue429-main-5000.json --progress`

## 20-game smoke

| Metric | A current | B T1 |
| --- | ---: | ---: |
| completed / crashed | 20 / 0 | 20 / 0 |
| opening BID rate | 100.00% | 85.00% |
| raise rate | n/a | 14.29% |
| contract success rate | 0.00% | 30.00% |
| candidate relative reward | -4.0000 | -0.9000 |
| illegal / fallback / inference failure | 0 / 0 / 0 | 0 / 0 / 0 |

## 5,000-game A/B results

| Metric | A current | B T1 |
| --- | ---: | ---: |
| completed / crashed | 5000 / 0 | 5000 / 0 |
| candidate PASS rate | 0.00% | 62.41% |
| candidate BID rate | 100.00% | 37.59% |
| opening BID rate | 100.00% | 71.68% |
| raise opportunities | 0 | 5444 |
| raise rate | n/a | 6.28% |
| all-pass rate | 0.00% | 0.00% |
| final candidate Napoleon rate | 100.00% | 7.12% |
| contract success rate | 0.34% | 33.72% |
| candidate relative reward | -3.9347 | -0.0906 |
| B minus A candidate relative reward | n/a | +3.8441 |
| illegal / fallback / inference failure | 0 / 0 / 0 | 0 / 0 / 0 |

## Final target distribution

| Target | A current | B T1 |
| --- | ---: | ---: |
| 13 | 0 | 210 |
| 14 | 0 | 2644 |
| 15 | 0 | 1812 |
| 16 | 0 | 325 |
| 17 | 0 | 9 |
| 18 | 0 | 0 |
| 19 | 5000 | 0 |

## Suit distribution

| Suit | A current | B T1 |
| --- | ---: | ---: |
| spades | 4178 | 1022 |
| hearts | 12 | 1327 |
| diamonds | 786 | 1411 |
| clubs | 24 | 1240 |

## Strength buckets

| Bucket | Games | A opening BID | A Napoleon | A success | B opening BID | B raise | B Napoleon | B success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low | 772 | 100.00% | 100.00% | 0.13% | 13.86% | 0.00% | 0.00% | n/a |
| medium | 3725 | 100.00% | 100.00% | 0.30% | 79.84% | 2.88% | 4.16% | 32.90% |
| strong | 471 | 100.00% | 100.00% | 0.85% | 100.00% | 40.28% | 38.00% | 47.49% |
| veryStrong | 32 | 100.00% | 100.00% | 3.13% | 100.00% | 69.70% | 68.75% | 72.73% |

## Relative rewards by role

| Role | A mean | B mean |
| --- | ---: | ---: |
| napoleon | -3.9296 | +1.5142 |
| adjutant | +0.9824 | -0.3786 |
| citizen | +0.9841 | -0.3531 |
| napoleon-adjutant | -4.0000 | +0.6857 |

## Decision diagnostics

| Diagnostic | B T1 |
| --- | ---: |
| opening EV near zero decisions | 553 / 5000 = 11.06% |
| raise EV delta near zero decisions | 129 / 5444 = 2.37% |
| low pWin BID (`pWin < 0.10`) | 0 / 3926 = 0.00% |
| high pWin PASS (`pWin > 0.70`) | 0 / 6518 = 0.00% |

## Judgment

T1 materially improves over the current runtime bidding baseline in whole-game play: contract success rises from 0.34% to 33.72%, candidate relative reward improves by +3.8441, low hands mostly pass, strong hands raise/Napoleon more often, and no illegal actions, fallbacks, inference failures, or crashes were observed.

The improvement is attributable to wiring the #427 history-consistent single-head T1 value model into runtime EV selection. This PR does not add model-structure changes such as opening/raise heads or minimal context, and does not change downstream rewards or playing policies.

Recommendation: use T1 as the next final bidding AI candidate for runtime wiring. Remaining caveat: candidate relative reward is still slightly negative over this seed panel, so deployment should keep diagnostics enabled while broader panels are run.
