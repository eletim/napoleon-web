# exchange-combination-value-v1

Issue #436 exchange discard-combination scalar value baseline.

- Dataset: `/tmp/issue436-exchange-cf-1000states`
- Samples: 1,000 source exchange states x 286 candidates = 286,000 samples
- Split: source-state grouped 800 / 100 / 100 train / validation / final
- Input: exchange observation 2671 dims + candidate discard mask 53 dims = 2724 dims
- Model: MLP `[512, 512, 256, 256] -> scalar contractMargin`
- Selected loss: pointwise Huber regression, delta 1.0
- Checkpoint SHA-256: `f11e6a8d4638e477b603491d5ecef722457b73aad377a25bfe48f9763e9b3f24`

Final held-out summary:

| Metric | Model | RuleBased |
| --- | ---: | ---: |
| margin regret mean | 4.39 | 4.59 |
| relative reward regret mean | 12.975 | 14.2 |
| rank percentile mean | 0.6746 | 0.6359 |

The model is only a small improvement over RuleBased on final held-out and is
not strong enough to freeze for runtime wiring without more work.
