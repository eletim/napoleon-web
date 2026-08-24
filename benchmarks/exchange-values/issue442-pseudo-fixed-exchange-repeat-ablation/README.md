# issue442-pseudo-fixed-exchange-repeat-ablation

Issue #442 pseudo-fixed exchange teacher repeat ablation.

Datasets were generated under `/tmp` with the #440 pseudo-fixed generator semantics unchanged:

- `1000x1`: `/tmp/issue442-pseudo-fixed-exchange-1000x1`
- `200x5`: `/tmp/issue442-pseudo-fixed-exchange-200x5-full`
- `100x10`: `/tmp/issue442-pseudo-fixed-exchange-100x10-full`

All layouts have 1,000 exchange states and 286,000 candidate samples.

| layout | Pearson | pairwise | margin regret | relative reward regret | RuleBased margin regret | checkpoint SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| #438 compact396 baseline | 0.409 | 0.592 | 4.60 | 12.30 | 4.59 | n/a |
| 1000x1 | 0.262 | 0.572 | 5.34 | 23.52 | 5.28 | `c329b86de6978ec4d5225ef3d9fb29b413c3cb16d814a65746701ce3b2517b2a` |
| 200x5 | 0.080 | 0.561 | 5.01 | 20.88 | 5.33 | `44eecdbb548eb867d0ae2c9f84e1c54f71f4bdada9276241bf9539f652bf7712` |
| 100x10 | 0.250 | 0.565 | 4.59 | 18.98 | 4.64 | `032592ee50bbf8c4a1f98b1be58c71019e2d82e2bb64ab4ce078fb375023cb79` |

Conclusion: `100x10` is best within this pseudo-fixed ablation, and repeats/group clearly increased same-13 discard variation. It does not clearly beat #438 overall: pairwise and relative reward regret are worse, and margin regret only ties #438 after rounding. Do not adopt pseudo-fixed as the formal exchange teacher candidate from this run; proceed to adjutant+kitty joint design.

Full report: `diagnostics/issue442-pseudo-fixed-exchange-repeat-ablation.md`.
