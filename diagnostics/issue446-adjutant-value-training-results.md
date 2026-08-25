# Issue #446 Adjutant Value Training Results

Generated on 2026-08-25 with the formal policy path:

- Bidding: `frozen-raise-v1`
- Playing: `ppo-separated-v1000`
- Playing critic for `frozen-raise-v1` pass EV: `ppo-separated-v1000/critic.onnx`
- Runtime order: adjutant choice -> kitty pickup -> exchange -> playing

## Datasets

- Proposal teacher dataset: `/tmp/issue446-policy-production/proposal-merged`
- Source states: 10,000
- Compact290 samples: 530,000
- Proposal terminal rollouts: 13,062,912
- Manifest SHA256: `1d610a653428a9ceed7fdba9c6c38059d31b5979c19020c92ee482883cb12342`
- Full-gold holdout: `/tmp/issue446-policy-production/fullgold-merged`
- Full-gold states: 200
- Full-gold samples: 10,600
- Full-gold terminal rollouts: 3,031,600
- Manifest SHA256: `4f38f95314aa6922c549c54c50532ae0b7aa95dd8fc2706a235d3506467d3da7`

Both datasets used `compact396` scoring over all 286 exchange candidates per adjutant,
then proposal rollout over top16 + RuleBased exchange + 8 deterministic diversity candidates.
Full-gold evaluated all 53 adjutants x 286 exchange candidates.

Production merged manifests preserve the generated source seeds in `sourceDiagnostics`.
The committed generator/merge path now also emits explicit `hiddenDealChecksum`,
`originalHandIdentity`, `biddingHistoryHash`, seed span, and source distribution fields
for regenerated shards/merged datasets.

### Source Distribution

Proposal train states:

- Contract suit: spades 4,516 / hearts 2,032 / diamonds 2,520 / clubs 932
- Contract target: 13 -> 7,186 / 14 -> 2,615 / 15 -> 188 / 16 -> 11
- Napoleon seat: 0 -> 3,706 / 1 -> 1,921 / 2 -> 1,565 / 3 -> 1,515 / 4 -> 1,293

Full-gold states:

- Contract suit: spades 95 / hearts 38 / diamonds 49 / clubs 18
- Contract target: 13 -> 153 / 14 -> 43 / 15 -> 3 / 16 -> 1
- Napoleon seat: 0 -> 76 / 1 -> 44 / 2 -> 32 / 3 -> 32 / 4 -> 16

## Best Checkpoint

- Path: `/tmp/issue446-policy-production/adjutant-value-mlp-10k/checkpoint.pt`
- SHA256: `6b8ba104e5bd1d4350f148c2fb741e258ccac953e9da230569cccf6bc39d0eae`
- Report SHA256: `1c567079fca05b7a81ba2462ece68a7a9bbdf75670d0dd5bd0ff3c4bc296bcb6`
- Best epoch: 2
- Best validation Huber: 1.4097167967850308
- Architecture: compact290 -> `[512,512,256,256]` -> scalar contract margin
- Training split: train 424,000 / validation 53,000 / final 53,000 samples
- Device: CUDA, NVIDIA GeForce RTX 4060

## Proposal Accuracy On Full Gold

Candidate count: 10,600.

- compact396 top4 containment: 0.0654
- top8 containment: 0.1139
- top16 containment: 0.1981
- top32 containment: 0.3258
- top64 containment: 0.5110
- top16 + RuleBased containment: 0.2091
- full proposal containment: 0.2358
- RuleBased exchange alone gold match: 0.0225
- proposal-best regret mean:
  - top16: 1.3266
  - top16 + RuleBased: 1.2792
  - top16 + RuleBased + diversity: 1.1235
- RuleBased additive effect: +0.0109 containment, -0.0474 regret
- Diversity additive effect: +0.0267 containment, -0.1558 regret

## Compact290 Model On Full Gold

- MAE: 1.9789
- RMSE: 2.4446
- Bias: -1.2891
- Pearson: 0.4866
- Pairwise accuracy: 0.6636
- Model-selected adjutant exact gold match: 0.0900
- Model-selected adjutant top3 gold match: 0.1700
- Model-selected adjutant top5 gold match: 0.2300
- Model-selected full-gold regret mean: 1.5400

## Decomposition On Same Full-Gold States

Mean contract margin over 200 states:

- RuleBased adjutant + RuleBased exchange: -0.320
- RuleBased adjutant + optimized exchange: 4.860
- Model adjutant + optimized exchange: 4.645
- Gold adjutant + gold exchange: 6.185
- RuleBased adjutant regret vs gold: 1.325

This decomposition shows most of the available gain comes from exchange optimization.
The learned adjutant selector does not yet beat RuleBased adjutant selection on the
full-gold holdout when both use optimized exchange.

## Scale-Up Decision

Do not proceed to 50k/100k by simple scale-up from this teacher/model setup.
The full-gold proposal containment is only 23.6% for the actual proposal set and
top64 containment is 51.1%, so the 10k training labels are still too approximation-biased.
The compact290 model also underperforms RuleBased adjutant selection on full-gold
decomposition.

Recommended next step is to improve the exchange proposal teacher before larger
adjutant training. The current compact396 exchange model should be retrained or
replaced with a better proposal/reranker, then the adjutant value dataset should be
regenerated. Runtime wiring and Frozen conversion should wait until full-gold regret
improves.

## Verification

- `cmake --build /tmp/napoleon-issue446-verify-build --target napoleon_adjutant_value_stream_cli napoleon_core_self_test`
- `/tmp/napoleon-issue446-verify-build/napoleon_core_self_test`
- `python3 -m compileall python/src/napoleon_ml/adjutant_value python/src/napoleon_ml/cli/generate_adjutant_value_dataset.py python/src/napoleon_ml/cli/merge_adjutant_value_datasets.py python/src/napoleon_ml/cli/train_adjutant_value_mlp.py`
- `env UV_CACHE_DIR=/tmp/uv-cache-issue446 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --project python --extra dev python -m pytest python/tests/unit/test_adjutant_value.py python/tests/unit/test_exchange_value.py`

`pnpm --filter @napoleon/cpp-core test` was also attempted, but the `/tmp`
worktree has no `node_modules`; it stopped at `tsc: not found` after C++ build
and CTest had already passed.
