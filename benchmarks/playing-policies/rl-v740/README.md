# RL v740 playing benchmark policy

`rl-v740` は、旧RL系列 `rl-v100-to-v1000` の iteration / generation 740 で得られた playing policy です。repo内では従来の benchmark opponent / Frozen opponent として意図的に残している履歴基準点です。

このpolicyの元artifactは playing input schema v1 / 6242 features でした。Issue #193 時点のruntimeは schema v2 / 6246 features を使っていたため、このrepoに保存している `policy.onnx` は schema v2 へmigration済みです。Issue #234 以降の現行runtimeは bidding 10〜19 / all-pass 9 の schema v3 / 7653 features であり、このartifactは意図的に互換性検証で拒否されます。新ルールのbenchmark opponentとして使うには、現行schemaで再学習または再exportしたartifactを追加してください。

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
| repo保存入力 | schema v2 / 6246 featuresへmigration済み | schema v2 / 6246 features |
| 学習系列 | provenanceに残る旧RL behavior policy | `ppo-separated-v1` |
| Actor/Critic | runtime用Actorのみ保存 | Actor/Critic完全分離checkpointを保存 |
| repo保存物 | `policy.onnx`, `policy.json`, `provenance.json` | `policy.onnx`, `policy.json`, `checkpoint.pt`, `provenance.json` |
| 主な用途 | 履歴benchmark / 旧ルール互換性検証 | 旧ルールFrozen Actor/Critic artifact |
| 特徴 | 旧policyをlogit維持でschema v2へmigration | 旧schema上で1000 iterations学習したseparated PPO |
