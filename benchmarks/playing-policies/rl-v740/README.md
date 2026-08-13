# RL v740 playing benchmark policy

`rl-v740` は、旧RL系列 `rl-v100-to-v1000` の iteration / generation 740 で得られた playing policy です。repo内では従来の benchmark opponent / Frozen opponent として意図的に残している基準点であり、新しいpolicyの比較・評価で参照されます。

このpolicyの元artifactは playing input schema v1 / 6242 features でした。現在のruntimeは schema v2 / 6246 features を使うため、このrepoに保存している `policy.onnx` は schema v2 へmigration済みです。migrationでは旧6242次元ぶんの入力重みをそのままコピーし、追加された4次元 `selfRoleOneHot` の入力重みを0初期化しています。これにより、migration前から存在していた入力に対するlogitを変えないようにしています。

このartifactに含まれるもの:

- `policy.onnx` - runtime用のActor ONNX。
- `policy.json` - Actor ONNX metadata。
- `provenance.json` - 元artifact、migration、committed artifact hash、logit parityの記録。

このartifactには、今回追加した `ppo-separated-v1000` のような学習済みCritic checkpointは保存していません。学習方式や学習条件についても、`provenance.json` から追跡できない内容はここでは推測していません。

## `ppo-separated-v1000` との違い

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
