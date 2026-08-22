# Issue #390 role probability x role-specific value offline score

Date: 2026-08-23

Base branch: `feature/issue-202-non-playing-ai`

Implementation:

- Added offline score evaluator `napoleon-evaluate-bidding-role-score`.
- Runtime bidding agent is not connected in this issue.
- `S0`: Napoleon-only margin score.
- `S1`: raw `P(role | state, action) * V_role`.
- `S2`: S1 with role-wise affine calibration fitted only on calibration states.
- `S3`: S2 with Napoleon `mu - 0.25 sigma`.
- `no-contract` role mass contributes zero value. `napoleon-adjutant` role probability is counted with Napoleon value.
- Teacher utility is `contractMargin` for Napoleon/Adjutant, `-contractMargin` for Citizen, and `0` for NoContract. Citizen margin is reconstructed from final target and Napoleon-side point cards when the row does not carry `contractMargin`.

Artifacts:

- Heldout dataset: `/tmp/napoleon-issue390-heldout-alllegal-1000`
- Manifest SHA-256: `ac1ab03c985fcccf0bfad9a8206b3069754c5fb6a0978eef2a64f1d7e5e3152a`
- Source states / samples / forced state-action pairs: `1000 / 25460 / 25460`
- Role probability model: `/tmp/napoleon-issue378-multi-head-20k/checkpoint.pt`
- Napoleon margin model: `/tmp/napoleon-issue380-margin-20k/checkpoint.pt`
- Citizen value model: `/tmp/napoleon-issue385-citizen-alllegal-1000/checkpoint.pt`
- Adjutant value model: `/tmp/napoleon-issue385-adjutant-alllegal-1000-v100/checkpoint.pt`
- Full report: `/tmp/napoleon-issue390-role-score-report.json`

Split:

| split | states | samples | state key hash |
| --- | ---: | ---: | --- |
| calibration | 250 | 6407 | `06164328d20848fd2828211f9cbfd091100d819b5d11bcfb9a8872d435e12199` |
| validation | 250 | 6412 | `fa20aaaec23bc11af2e48fa81f51c693cb5db1d1929f25c1927835f4af8767cf` |
| final | 500 | 12641 | `258e66d47462b993f6dbef0bfcf9854d068d460ec45f0596f23ad66cbb3d6901` |

Calibration fit used only the calibration split:

| role | samples | slope | intercept |
| --- | ---: | ---: | ---: |
| Napoleon | 5498 | 0.7413 | 0.7702 |
| Adjutant | 216 | 0.0487 | -0.7663 |
| Citizen | 679 | 0.5657 | 0.2472 |

Dataset coverage:

| item | count |
| --- | ---: |
| PASS samples | 1000 |
| BID samples | 24460 |
| Napoleon terminal role | 21902 |
| Adjutant terminal role | 876 |
| Citizen terminal role | 2637 |
| all-pass starter/other | 9 / 36 |

Final heldout role probability metrics:

| metric | value |
| --- | ---: |
| accuracy | 0.8752 |
| macro F1 | 0.3401 |
| Brier | 0.2368 |
| Napoleon recall / F1 | 1.0000 / 0.9372 |
| Adjutant recall / F1 | 0.0255 / 0.0453 |
| Citizen recall / F1 | 0.1410 / 0.2353 |

Final heldout score variants:

| variant | pairwise | diff pairs | best | top3 | mean selected utility | regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S0 Napoleon-only | 0.7345 | 145231 | 0.358 | 0.616 | -0.266 | 2.742 |
| S1 raw integration | 0.7401 | 145231 | 0.426 | 0.668 | 0.380 | 2.096 |
| S2 calibrated integration | 0.7398 | 145231 | 0.412 | 0.664 | 0.256 | 2.220 |
| S3 calibrated risk-aware | 0.7377 | 145231 | 0.422 | 0.650 | 0.318 | 2.158 |

Final heldout baselines:

| baseline | pairwise | best | top3 | mean selected utility | regret | PASS rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| behavior source policy | 0.5292 | 0.452 | 0.668 | 0.510 | 1.966 | 0.994 |
| pass-only | 0.5293 | 0.454 | 0.666 | 0.534 | 1.942 | 1.000 |
| rule-based-offline | 0.5290 | 0.400 | 0.676 | 0.286 | 2.190 | 0.498 |
| conservative-offline | 0.5294 | 0.442 | 0.666 | 0.516 | 1.960 | 0.960 |
| random legal | 0.5001 | 0.068 | 0.618 | -5.336 | 7.812 | 0.058 |

Final heldout selected action distribution:

| variant | PASS rate | BID rate | strongest-suit match among bids | bid suit distribution | bid target distribution |
| --- | ---: | ---: | ---: | --- | --- |
| S0 | 0.626 | 0.374 | 0.3155 | clubs 122, diamonds 5, hearts 33, spades 27 | 13:51, 14:86, 15:49, 16:1 |
| S1 | 0.896 | 0.104 | 0.3077 | clubs 34, diamonds 11, hearts 1, spades 6 | 13:11, 14:26, 15:15 |
| S2 | 0.866 | 0.134 | 0.3433 | clubs 47, diamonds 10, hearts 2, spades 8 | 13:16, 14:27, 15:24 |
| S3 | 0.884 | 0.116 | 0.3793 | clubs 38, diamonds 6, hearts 9, spades 5 | 13:12, 14:22, 15:23, 16:1 |

Hand-strength PASS/BID:

| variant | low `<200` bid | medium `200-279` bid | strong `280-329` bid | very strong `330+` bid |
| --- | ---: | ---: | ---: | ---: |
| S0 | 0.2055 | 0.3875 | 0.4912 | 1.0000 |
| S1 | 0.0411 | 0.1192 | 0.0877 | 0.0000 |
| S2 | 0.0959 | 0.1463 | 0.1053 | 0.0000 |
| S3 | 0.1096 | 0.1220 | 0.0877 | 0.0000 |

Collapse:

- S0/S1/S2/S3 collapse flags are all false under the configured offline thresholds.
- However S1/S2/S3 are PASS-heavy (`86.6%` to `89.6%`) and do not beat pass-only on selected utility/regret.

Conclusion:

- S1/S2/S3 improve the primary S0 Napoleon-only baseline on pairwise, best/top3, mean selected teacher utility, and regret.
- S1 is the best S0 improvement: `+0.646` selected utility and `-0.646` regret versus S0.
- Calibration does not improve over raw S1 on this heldout; S2 remains better than S0 but worse than S1.
- Runtime connection should not proceed from this issue. The next step should keep the integration offline or add policy-side guardrails, because the integrated scores remain worse than pass-only/conservative behavior on selected utility/regret.

Verification:

- `env UV_CACHE_DIR=/tmp/uv-cache PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --project python --extra dev python -m pytest python/tests/unit/test_bidding_role_score_integration.py python/tests/unit/test_bidding_q.py -q`
- `env UV_CACHE_DIR=/tmp/uv-cache uv run --project python --extra dev ruff check python/src/napoleon_ml/bidding_q/role_score_integration.py python/src/napoleon_ml/cli/evaluate_bidding_role_score.py python/tests/unit/test_bidding_role_score_integration.py python/src/napoleon_ml/bidding_q/dataset.py python/src/napoleon_ml/bidding_q/__init__.py`
