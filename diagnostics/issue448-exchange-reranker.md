# Issue #448 exchange proposal scorer / reranker

## Decision

Do not regenerate the adjutant dataset and do not wire or freeze a new exchange scorer.
The best formal candidate is the 1,988-state compact396 listwise reranker, but its gain at
the current practical proposal budget is only +0.0219 containment and -0.0258 mean margin
regret. This is not a large improvement over the Issue #446 baseline.

No production proposal K is adopted. K=64 is recorded only as a diagnostic operating
point: it reaches 0.5720 practical containment and 0.4101 regret, but the baseline at the
same budget is already 0.5324 / 0.4258. The next work should revisit the exchange teacher
target, hidden expectation design, or a sequential exchange policy rather than scaling
this architecture or regenerating the adjutant dataset.

## Fixed audit identity

The audit reuses the exact Issue #446 200 choosing-adjutant state holdout:

- 200 source states
- 10,600 adjutant groups
- full 53 x 286 candidates
- 3,031,600 terminal rollouts
- fixed manifest SHA-256:
  `4f38f95314aa6922c549c54c50532ae0b7aa95dd8fc2706a235d3506467d3da7`
- regenerated detailed-audit manifest SHA-256:
  `37b991703d8a772d4523ad15e5a673ac6113ca2f0006f14d65e671813d15a340`

The regenerated detailed audit was accepted only after the fixed `features.f32`,
`contract-margin.f32`, `relative-reward.f32`, and `candidate-card.u8` streams matched the
Issue #446 artifacts byte-for-byte. This is not a replacement holdout.

## compact396 failure analysis

The old scorer puts the gold-best discard at mean rank 81.31, median 62, and p90 189.
Only 1.78% are rank 1; 24.02% are below rank 128.

| contract slice | mean rank | median | p90 |
| --- | ---: | ---: | ---: |
| spades | 78.22 | 58 | 185.6 |
| hearts | 81.26 | 64 | 182.7 |
| diamonds | 79.30 | 60 | 187.4 |
| clubs | 103.24 | 89.5 | 209 |
| target 13 | 80.18 | 60 | 187 |
| target 14 | 85.00 | 68 | 193.2 |
| target 15 | 95.89 | 77 | 225 |
| target 16 | 52.13 | 26 | 131.8 |

The candidate content exposes a concrete ranking error:

| selection | buried point cards mean | buried trump mean | joker | oruma | seiJack | uraJack |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gold-best | 1.156 | 0.589 | 0.0441 | 0.0398 | 0.0254 | 0.0419 |
| compact396 top1 | 1.575 | 0.108 | 0.0016 | 0.0000 | 0.0000 | 0.0016 |
| RuleBased | 0.238 | 0.000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Thus compact396 over-selects point-card burial relative to gold, strongly under-selects
trump burial, and almost never surfaces the special-card burials that sometimes win under
the full teacher. RuleBased is the opposite extreme and never buries trump in this audit.
Gold / RuleBased exact agreement is 0.0225. Gold-best rank is also worse when the called
card is in an opponent hand (mean 85.71) than in the original hand (66.87) or kitty
(70.85). Old scorer top1 margin regret is 4.884 versus 5.220 for RuleBased.

## Training and leakage

The formal comparison combines the existing Issue #438 1,000-state dataset and Issue
#442 1,000x1 pseudo-fixed dataset. Their manifest hashes are:

- `657ce74cef0540b5c8fabda54dbb57a182aecba14955d1a049daca01dc3d4164`
- `e0ddcf2954bb3eebbc4b1fd696077aee68be66ab44858a2d65601fb1df5f2cec`

Before splitting or training, 12 states whose three-card kitty identity happened to match
the fixed audit were removed as whole 286-candidate groups. The effective dataset is
1,988 states / 568,568 samples. The final guard reports zero train/audit overlap for deal
seed, hidden deal checksum, original hand, kitty, bidding history, and encoded visible
source identity. No new exchange holdout or adjutant dataset was generated.

## Formal scorer comparison

All models use one `[512, 512, 256, 256]` architecture and the same fixed audit.

| scorer | K1 | K4 | K8 | K16 | K24 | K32 | K48 | K64 | K96 | K128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old compact396 | 0.0178 | 0.0654 | 0.1139 | 0.1981 | 0.2678 | 0.3258 | 0.4256 | 0.5110 | 0.6504 | 0.7598 |
| pointwise + pairwise | 0.0129 | 0.0439 | 0.0807 | 0.1461 | 0.2048 | 0.2528 | 0.3487 | 0.4315 | 0.5753 | 0.6966 |
| pairwise-only | 0.0218 | 0.0731 | 0.1282 | 0.2167 | 0.2931 | 0.3551 | 0.4583 | 0.5407 | 0.6828 | 0.7858 |
| state-wise listwise | 0.0228 | 0.0758 | 0.1346 | 0.2249 | 0.2993 | 0.3623 | 0.4642 | 0.5534 | 0.6884 | 0.7910 |
| listwise + visible tactical | 0.0222 | 0.0752 | 0.1324 | 0.2235 | 0.2936 | 0.3542 | 0.4568 | 0.5478 | 0.6856 | 0.7856 |

| scorer | rank mean | median | p90 | pairwise accuracy | practical K16 containment | practical K16 regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| old compact396 | 81.31 | 62 | 189 | 0.6056 | 0.2358 | 1.1235 |
| pointwise + pairwise | 94.89 | 78 | 208 | 0.5796 | 0.1939 | 1.3189 |
| pairwise-only | 76.22 | 56 | 182 | 0.6135 | 0.2538 | 1.0875 |
| state-wise listwise | 74.72 | 54 | 179 | 0.6081 | 0.2576 | 1.0977 |
| listwise + visible tactical | 75.91 | 56 | 181 | 0.6085 | 0.2549 | 1.0787 |

The tactical input appends only visible candidate-local aggregates: buried point/trump,
remaining trump, retained points, joker/oruma/seiJack/uraJack/yoromeki, and called-card
burial. It does not use hidden opponent information. It lowers practical regret slightly
but does not improve containment over plain listwise.

## Oracle and practical proposal budgets

| K | baseline oracle margin / reward regret | listwise oracle margin / reward regret | baseline practical containment / margin / reward | listwise practical containment / margin / reward |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 1.3266 / 3.3722 | 1.2786 / 3.3057 | 0.2358 / 1.1235 / 2.6594 | 0.2576 / 1.0977 / 2.6182 |
| 24 | 1.0253 / 2.4021 | 0.9842 / 2.2745 | 0.3003 / 0.8979 / 1.9948 | 0.3270 / 0.8777 / 1.9469 |
| 32 | 0.8387 / 1.9042 | 0.8081 / 1.7575 | 0.3550 / 0.7496 / 1.5979 | 0.3868 / 0.7336 / 1.5370 |
| 48 | 0.6079 / 1.2889 | 0.5808 / 1.1575 | 0.4501 / 0.5543 / 1.1054 | 0.4854 / 0.5366 / 1.0269 |
| 64 | 0.4610 / 0.9111 | 0.4418 / 0.8401 | 0.5324 / 0.4258 / 0.7896 | 0.5720 / 0.4101 / 0.7495 |

The full report also records oracle K=1/4/8/96/128. None of the primary targets are met:
top16 0.2249 < 0.35, top32 0.3623 < 0.50, top64 0.5534 < 0.70, and practical K16
0.2576 / 1.0977 remains far from 0.50 / 0.75.

## Artifact and scope

The best experimental checkpoint is
`benchmarks/exchange-values/issue448-listwise-reranker-experimental/checkpoint.pt`, SHA-256
`5c859933a798935a1bc2a92115b4263f3ab29ee50e3c535cbf5558e33d968e3a`.

Issue #448 does not perform runtime wiring, Frozen promotion, adjutant 50k/100k training,
reward changes, playing retraining, bidding retraining, or adjutant dataset regeneration.
