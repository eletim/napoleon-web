# PPO separated v1000 playing policy

`ppo-separated-v1000` は、main PPO run `ppo-separated-e4-lr1e4-cpp-pool5050-1000` の最終 generation v1000 をrepo内に固定保存した Frozen playing policy artifact です。algorithmは `ppo-separated-v1` で、ActorとCriticを完全に分離した `playing-separated-actor-critic-v1` 構成です。

このartifactは1000 iterations完走後のcheckpointから作成しています。主な学習条件は、6000 games / iteration、epochs=4、learning rate=1e-4、PPO clip epsilon=0.2、C++ simulation backend、CUDA inferenceです。

self-play rolloutは、実際のC++ rollout実装と保存済みmanifestでは `current-plus-opponent-pool` として記録されています。各gameでcurrent policyが1席に入り、その席はgame indexでrotationします。残り4席は各席ごとに、`RuleBasedAgent` と frozen `rl-v740` を重み1:1のopponent poolから独立にsampleします。runの `config.json` には `rolloutRoster` が文字列で残っていますが、このREADMEでは `iterations/iter-999/selfplay/manifest.json` の `opponentPool` / `provenance.rosterSpec` と `packages/cpp-core/native/src/napoleon_rl_dataset_cli.cpp` の実装に基づいて記述しています。

v1000評価は `candidate-vs-opponent-pool` 条件で、同じく RuleBased / frozen `rl-v740` の重み1:1 opponent poolを使っています。保存済みsummaryでは 2000 games、winRate=65.65%でした。この値はその評価条件での確認済み結果であり、他条件で常に強いことを意味するものではありません。

このartifactに含まれるもの:

- `policy.onnx` - runtime用のActor-only ONNX。playing decisionではこのActorを使います。
- `policy.json` - Actor ONNX metadata。
- `checkpoint.pt` - 学習済みCriticを含む完全なPyTorch Actor/Critic checkpoint。
- `provenance.json` - 元run、generation、algorithm、hash、主要学習条件、v1000評価summaryの記録。

`policy.onnx` のruntime contractは、`model_input` `[batch, 6246]` float32 inputと `logits` `[batch, 53]` float32 outputです。このFrozen artifact化では学習や再評価は行わず、完走済みv1000 runの既存artifactをコピーして検証しています。

## `rl-v740` との違い

| 観点 | `rl-v740` | `ppo-separated-v1000` |
| --- | --- | --- |
| 出自 | 旧 `rl-v100-to-v1000` 系列 | main PPO run `ppo-separated-e4-lr1e4-cpp-pool5050-1000` |
| 世代 | v740 | v1000 |
| 元入力 | schema v1 / 6242 features | schema v2 / 6246 features |
| 現在の入力 | schema v2 / 6246 featuresへmigration済み | schema v2 / 6246 features |
| 学習系列 | provenanceに残る旧RL behavior policy | `ppo-separated-v1` |
| Actor/Critic | runtime用Actorのみ保存 | Actor/Critic完全分離checkpointを保存 |
| repo保存物 | `policy.onnx`, `policy.json`, `provenance.json` | `policy.onnx`, `policy.json`, `checkpoint.pt`, `provenance.json` |
| 主な用途 | 従来benchmark / Frozen opponent | 新しいFrozen Actor/Critic artifact |
| 特徴 | 旧policyをlogit維持でruntime互換migration | 現行schema上で1000 iterations学習したseparated PPO |
