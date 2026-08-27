# Issue #454 independent parameterized-policy verification

## Decision

The frozen Issue #452 parameterized adjutant + exchange policy is independently
verified and adopted as a formal artifact candidate. No optimization, retraining,
parameter selection, feature change, reward change, or runtime wiring was performed.

The exact Issue #452 logical parameter SHA-256 is
`d364aef0c48a1832bd6602d254d0440f6cb2e2cb50492cfb53934e0378a84d69`.
Its 95 weights remain split into 35 adjutant and 60 exchange weights under feature
schema v1.

## Fixed evaluation

- Games: 10,000 paired games.
- Bidding: `frozen-raise-v1`.
- Playing: `ppo-separated-v1000` for every seat.
- Reward: unchanged current relative reward.
- Phase order: bidding -> adjutant -> kitty -> exchange -> playing.
- Pairing: each policy used the same ordered deal seeds and the playing request
  sequence was reset for each policy evaluation.
- Confidence interval: paired normal interval, mean difference +/- 1.96 SE.
- Win/tie/loss: learned reward greater than/equal to/less than RuleBased reward on
  the same deal seed.

## Independent seed audit

The verification pool starts at 954,000,000 and contains the contiguous accepted
candidate-Napoleon seeds 954,000,000 through 954,009,999. Its logical seed-list
SHA-256 is
`0408442b7a3fa7dbb0521e4d15755262dd3af21aaf7478e1f5201c71312bff56`.

All 122 Issue #452 manifests were loaded, checksum-validated, and jointly audited.
They contain 57,100 unique seeds. Overlap between those seeds and the 10,000 new
verification seeds is exactly zero.

## Main result and Issue #452 comparison

| Evaluation | Games | Learned mean | RuleBased mean | Paired difference | SE | Paired 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| #452 final holdout | 5,000 | 1.14595 | -0.06705 | +1.21300 | 0.25256 | [0.71799, 1.70801] |
| #454 independent verification | 10,000 | 1.21330 | -0.00470 | +1.21800 | 0.17891 | [0.86734, 1.56866] |

The independent estimate is nearly identical to #452. Its confidence interval is
narrower and its lower bound remains clearly above zero. Paired win/tie/loss is
1,470 / 7,408 / 1,122.

| Secondary metric | Learned | Existing RuleBased |
| --- | ---: | ---: |
| Contract success | 34.30% | 30.82% |
| Mean contract margin | -2.0715 | -2.5001 |
| Napoleon-side point cards | 12.1991 | 11.7705 |
| Mean buried point cards | 1.2415 | 0.0887 |
| Illegal / fallback / invariant failure | 0 / 0 / 0 | 0 / 0 / 0 |

The evaluator executed 120,000 explicit phase/contract/adjutant/turn/kitty/hand-size
invariant checks for each policy. All passed. The instrumented verification
evaluator SHA and the original Issue #452 evaluator SHA are both retained in the
dependency provenance; bidding and playing artifact hashes are unchanged.

Declared-target distributions are byte-for-byte equal between policies because
bidding is fixed before the evaluated decisions. Full declared-target, adjutant
choice, buried-card/rank/suit, per-seed reward, margin, success, and point-card
distributions are retained in `verification-report.json`.

## Block diagnostics

Blocks were fixed at 1,000 paired games before evaluation.

| Block | Seed range | Paired difference | SE | 95% CI |
| ---: | --- | ---: | ---: | ---: |
| 0 | 954000000-954000999 | +1.4375 | 0.54996 | [0.35958, 2.51542] |
| 1 | 954001000-954001999 | +1.4900 | 0.59296 | [0.32779, 2.65221] |
| 2 | 954002000-954002999 | +1.1050 | 0.53867 | [0.04920, 2.16080] |
| 3 | 954003000-954003999 | +1.5075 | 0.55687 | [0.41603, 2.59897] |
| 4 | 954004000-954004999 | +0.0000 | 0.56808 | [-1.11343, 1.11343] |
| 5 | 954005000-954005999 | +1.6975 | 0.57293 | [0.57455, 2.82045] |
| 6 | 954006000-954006999 | +0.5775 | 0.54908 | [-0.49869, 1.65369] |
| 7 | 954007000-954007999 | +0.8575 | 0.56642 | [-0.25268, 1.96768] |
| 8 | 954008000-954008999 | +2.1475 | 0.59171 | [0.98775, 3.30725] |
| 9 | 954009000-954009999 | +1.3600 | 0.56804 | [0.24664, 2.47336] |

Nine blocks are positive and one is exactly zero; none is negative. The result is
therefore directionally stable at the requested block scale.

## Provenance and artifacts

- Verification report file SHA-256:
  `9503ec68de9bc342a08ad949e38cefb645576cb8667270ac81b132f8338538bf`.
- Verification manifest file SHA-256:
  `8d72f2fea934d65b421375666abf9ecc7f831cd4d12f5a65e157b89123e7455b`.
- Formal candidate logical artifact SHA-256:
  `a6e97b72160338d3f0ce831f5b1422f86dafb419ff8e458b79741440b2433faa`.
- Formal candidate policy file SHA-256:
  `71b417b9011907ee02e5b3f7521e5479ca260101937dab05ee5cb91a8634c9fa`.
- Weight-vector SHA-256:
  `c84fbf7012ee043b2f3451213ecf30cdf596eaec41c6aeff7381ff7fccc9a4f0`.

The human-readable source-of-truth candidate is
`benchmarks/non-playing-policies/parameterized-adjutant-exchange-v1/policy.json`.
It embeds all 95 weights, Issue #452 optimizer/run provenance, verification
manifest/report references, and exact bidding/playing/evaluator dependency hashes.
Runtime wiring is explicitly excluded and should be handled by the next issue.
