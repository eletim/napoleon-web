# Issue 354: latest bidding-only PPO policy real-game diagnostics

Issue: https://github.com/eletim/napoleon-web/issues/354

## Experiment setup

- Source run: `/home/eletim/napoleon_runs/non-playing-ppo-compact278-bidding-only-stratified-suit32-e01-lr1e4-100x2000`
- Evaluated iteration: `bidding-iter-000100`, provenance `iterations/iter-000099/bidding`
- Bidding ONNX: `/home/eletim/napoleon_runs/non-playing-ppo-compact278-bidding-only-stratified-suit32-e01-lr1e4-100x2000/iterations/iter-000099/bidding/policy.onnx`
- Bidding metadata: `/home/eletim/napoleon_runs/non-playing-ppo-compact278-bidding-only-stratified-suit32-e01-lr1e4-100x2000/iterations/iter-000099/bidding/policy.json`
- Bidding ONNX SHA-256: `4a4df2eb80287c464c2bbe246602f0b85b37c988c2ec7bb431a9ed512a7f75a5`
- Bidding metadata SHA-256: `69bb14107a3e5ea1f0fee9ba7588d330e1c4e40cb6228463a893953e0cf6a7ca`
- Bidding checkpoint SHA-256: `3464976be3181ae4634e5d506a00f2d0726fff90359d6ff5ab427189c7e07e02`
- Playing artifact: `benchmarks/playing-policies/ppo-separated-v1000/policy.onnx`
- Playing ONNX SHA-256: `54d7ba29222a12e99a91ab61ee7aa253fe3fab73200d78167d64bf9e7bb8887e`
- Adjutant artifact: `adjutant-bootstrap-seed-202`
- Adjutant ONNX SHA-256: `9010861f013d5fca57034f8f1e89cbb1ad0df9c2c6b03e35f41fb21c46fb088c`
- Exchange artifact: `exchange-bootstrap-seed-202`
- Exchange ONNX SHA-256: `6e5091a8fc18da3172634dfd3cd2b4fe612244b1dbe768a18405ea3eda4c118a`
- Logical seeds: 10000
- Evaluation seed range: `354000000..354009999`
- Actual games: 50000
- Candidate seat rotation: `[0, 1, 2, 3, 4]`
- Bidding rollout temperature: `1`
- Full-policy evaluation action selection: existing ONNX argmax selectors; the CLI does not expose a temperature option.
- Inference backend/device: ONNX Runtime CPU, `--inference-device cpu`
- Base commit used when generating data: `fd4aeba229c58b16b4b0c77a0f918b8e2917433c`

The latest artifact was identified from the run `state.json` / `run-summary.json` and the `latest/bidding` target, not from a guessed path.

## Reproduction

From the repo root after building this PR:

```bash
pnpm --filter @napoleon/self-play-cli build
```

Generate raw bidding decisions:

```bash
node apps/self-play-cli/dist/index.js non-playing-rollout --phase bidding --policy-onnx /home/eletim/napoleon_runs/non-playing-ppo-compact278-bidding-only-stratified-suit32-e01-lr1e4-100x2000/iterations/iter-000099/bidding/policy.onnx --policy-metadata /home/eletim/napoleon_runs/non-playing-ppo-compact278-bidding-only-stratified-suit32-e01-lr1e4-100x2000/iterations/iter-000099/bidding/policy.json --playing-onnx benchmarks/playing-policies/ppo-separated-v1000/policy.onnx --playing-metadata benchmarks/playing-policies/ppo-separated-v1000/policy.json --adjutant-onnx /home/eletim/napoleon_runs/non-playing-ppo-compact278-bidding-only-stratified-suit32-e01-lr1e4-100x2000/bootstrap/adjutant/policy.onnx --adjutant-metadata /home/eletim/napoleon_runs/non-playing-ppo-compact278-bidding-only-stratified-suit32-e01-lr1e4-100x2000/bootstrap/adjutant/policy.json --adjutant-artifact-id adjutant-bootstrap-seed-202 --exchange-onnx /home/eletim/napoleon_runs/non-playing-ppo-compact278-bidding-only-stratified-suit32-e01-lr1e4-100x2000/bootstrap/exchange/policy.onnx --exchange-metadata /home/eletim/napoleon_runs/non-playing-ppo-compact278-bidding-only-stratified-suit32-e01-lr1e4-100x2000/bootstrap/exchange/policy.json --exchange-artifact-id exchange-bootstrap-seed-202 --output /tmp/napoleon-issue-354/latest-bidding-rollout-10000 --start-seed 354000000 --games 10000 --games-per-shard 500 --temperature 1 --inference-device cpu --inference-max-batch-size 256 --artifact-id bidding-iter-000100 --playing-artifact-id ppo-separated-v1000 --progress-prefix "[issue354 rollout] "
```

Generate the matched full-policy-vs-RuleBased game evaluation:

```bash
node apps/self-play-cli/dist/index.js full-policy-evaluate --playing-onnx benchmarks/playing-policies/ppo-separated-v1000/policy.onnx --playing-metadata benchmarks/playing-policies/ppo-separated-v1000/policy.json --bidding-onnx /home/eletim/napoleon_runs/non-playing-ppo-compact278-bidding-only-stratified-suit32-e01-lr1e4-100x2000/iterations/iter-000099/bidding/policy.onnx --bidding-metadata /home/eletim/napoleon_runs/non-playing-ppo-compact278-bidding-only-stratified-suit32-e01-lr1e4-100x2000/iterations/iter-000099/bidding/policy.json --adjutant-onnx /home/eletim/napoleon_runs/non-playing-ppo-compact278-bidding-only-stratified-suit32-e01-lr1e4-100x2000/bootstrap/adjutant/policy.onnx --adjutant-metadata /home/eletim/napoleon_runs/non-playing-ppo-compact278-bidding-only-stratified-suit32-e01-lr1e4-100x2000/bootstrap/adjutant/policy.json --exchange-onnx /home/eletim/napoleon_runs/non-playing-ppo-compact278-bidding-only-stratified-suit32-e01-lr1e4-100x2000/bootstrap/exchange/policy.onnx --exchange-metadata /home/eletim/napoleon_runs/non-playing-ppo-compact278-bidding-only-stratified-suit32-e01-lr1e4-100x2000/bootstrap/exchange/policy.json --output /tmp/napoleon-issue-354/latest-bidding-evaluation-10000.json --start-seed 354000000 --games 10000 --inference-device cpu --inference-max-batch-size 256 --progress-prefix "[issue354 eval] "
```

Aggregate:

```bash
node apps/self-play-cli/dist/index.js bidding-diagnostics --rollout-dataset /tmp/napoleon-issue-354/latest-bidding-rollout-10000 --evaluation /tmp/napoleon-issue-354/latest-bidding-evaluation-10000.json --label "PPO bidding iter 100" --output-json /tmp/napoleon-issue-354/latest-bidding-diagnostics.json
```

## Bidding action distribution

Raw rollout decisions: 71931. PASS: 52856 (73.48%). Bid: 19075 (26.52%).

| action | count | rate |
|---|---:|---:|
| PASS | 52856 | 73.48% |
| 13 | 803 | 1.12% |
| 14 | 2033 | 2.83% |
| 15 | 3124 | 4.34% |
| 16 | 3383 | 4.70% |
| 17 | 3114 | 4.33% |
| 18 | 3280 | 4.56% |
| 19 | 3338 | 4.64% |

| suit | count | rate |
|---|---:|---:|
| S | 5139 | 26.94% |
| H | 4812 | 25.23% |
| D | 4763 | 24.97% |
| C | 4361 | 22.86% |

| target | S | H | D | C |
|---|---:|---:|---:|---:|
| 13 | 299 (1.57%) | 307 (1.61%) | 196 (1.03%) | 1 (0.01%) |
| 14 | 625 (3.28%) | 513 (2.69%) | 489 (2.56%) | 406 (2.13%) |
| 15 | 853 (4.47%) | 783 (4.10%) | 791 (4.15%) | 697 (3.65%) |
| 16 | 895 (4.69%) | 801 (4.20%) | 819 (4.29%) | 868 (4.55%) |
| 17 | 819 (4.29%) | 788 (4.13%) | 753 (3.95%) | 754 (3.95%) |
| 18 | 853 (4.47%) | 810 (4.25%) | 805 (4.22%) | 812 (4.26%) |
| 19 | 795 (4.17%) | 810 (4.25%) | 910 (4.77%) | 823 (4.31%) |

## Strongest suit relation

Strongest suit uses `RuleBasedAgent.evaluateHandForTrump` via `evaluateHandForTrump`, with the same tie break order `spades`, `hearts`, `diamonds`, `clubs`.

| strongest | count | pass rate | same-suit bid rate |
|---|---:|---:|---:|
| S | 18864 | 73.51% | 7.19% |
| H | 18478 | 73.58% | 6.92% |
| D | 18414 | 74.38% | 6.31% |
| C | 16175 | 72.32% | 6.32% |

Same-suit match rate among bids: 4819/19075 = 25.26%.

| strongest | S | H | D | C | PASS |
|---|---:|---:|---:|---:|---:|
| S | 1356 | 1252 | 1261 | 1128 | 13867 |
| H | 1312 | 1278 | 1200 | 1092 | 13596 |
| D | 1299 | 1139 | 1162 | 1118 | 13696 |
| C | 1172 | 1143 | 1140 | 1023 | 11697 |

| strongest | S | H | D | C | PASS |
|---|---:|---:|---:|---:|---:|
| S | 7.19% | 6.64% | 6.68% | 5.98% | 73.51% |
| H | 7.10% | 6.92% | 6.49% | 5.91% | 73.58% |
| D | 7.05% | 6.19% | 6.31% | 6.07% | 74.38% |
| C | 7.25% | 7.07% | 7.05% | 6.32% | 72.32% |

Score-bin PASS/target distributions are emitted in `/tmp/napoleon-issue-354/latest-bidding-diagnostics.json` under `strongestSuit.scoreBins`.

## Game result

Two game-result views are recorded because the existing reusable CLIs cover two related but different opponent configurations.

Raw rollout formation diagnostics, with the same frozen opponent setup as the training data generator (`mixed-frozen-bidding`, 50/50 RuleBased/conservative per non-candidate seat):

- all-pass immediate end rate: 2645/50000 = 5.29%
- candidate Napoleon formation rate: 17372/50000 = 34.74%
- candidate role counts: napoleon=13986, napoleon-adjutant=3386, adjutant=6389, citizen=23594, all-pass-starter=551, all-pass-other=2094
- declaration success rate: 414/17372 = 2.38%

Full-policy-vs-RuleBased evaluation, with the candidate full ONNX policy in one rotating seat and four RuleBased seats:

- completed games: 50000/50000
- all-pass rate: 0.00% (2 all-pass games in raw comparison denominator handling)
- candidate Napoleon rate: 0.00%
- candidate Adjutant rate: 30.16%
- candidate Citizen rate: 69.84%
- Napoleon contract success when candidate Napoleon: n/a
- Napoleon mean target when candidate Napoleon: n/a
- Napoleon mean point cards when candidate Napoleon: n/a
- candidate win rate: 59.45%
- illegal action count: 0
- rule-based fallback count: 0

## RuleBased comparison

This is the existing matched `full-policy-evaluate` comparison: one rotating full ONNX policy seat against four RuleBased seats, same seeds and seat rotation.

| agent | win rate | contract success | average point cards |
|---|---:|---:|---:|
| policy | 59.45% | 33.43% | 9.271 |
| ruleBased | 53.15% | 33.43% | 9.603 |
| delta | +6.30pp | +0.00pp | -0.332 |

The policy seat never became Napoleon in this RuleBased-only opponent setup. Its win-rate gain came from Adjutant/Citizen outcomes, not from winning declarations.

## Conclusion

1. Suit collapse is not present in the coarse suit totals: S/H/D/C bid shares are 26.94% / 25.23% / 24.97% / 22.86%.
2. Clubs are mildly under-selected overall and almost absent for target 13 (`13-C` = 1), but clubs are not globally collapsed at targets 14-19. Diamonds are not avoided.
3. Strongest-suit alignment is weak: same-suit match among bids is 25.26%, close to uniform random over four suits. PASS dominates every strongest-suit row at 72.32-74.38%.
4. There is clear over-PASS behavior in raw decisions: PASS is 73.48%. In the mixed frozen rollout, candidate Napoleon formation is still 34.74%, but declaration success is only 2.38%.
5. Target values are skewed away from 13/14 and toward 16-19. Target 13 is especially low at 1.12%, and `13-C` is effectively absent.
6. Against four RuleBased seats, the full ONNX policy wins more often than the RuleBased aggregate (+6.30pp), but it never becomes Napoleon; it wins as Adjutant/Citizen while RuleBased agents take the contracts.
7. The next learning change should target bidding calibration and strongest-suit conditioning before changing playing, adjutant, or exchange. The current issue is not a hard suit collapse, but a combination of high PASS, weak hand/suit alignment, and target/suit artifacts at low targets.
