# Final COM-AI vs RuleBase evaluation

PR #203 で積み上げた non-playing AI 開発の最終評価として、正式 builtin preset 同士を同一 game 内で 50/50 混成対戦させた結果です。学習・policy 変更・artifact 再選定は行っていません。

## 1. 最終AI構成

| preset | playing | bidding | nonPlaying |
| --- | --- | --- | --- |
| COM-AI | `ppo-separated-v1000` | `frozen-raise-v1` | `parameterized-adjutant-exchange-v1` |
| COM-RuleBase | `rule-based` | `rule-based` | `rule-based` |

legacy full-policy、旧 adjutant/exchange PPO、experimental artifact は使用していません。

## 2. 実験条件

- 完了 game: 50,000（失敗 0）
- game seed: 462000000..462049999
- assignment: seat ごとの独立 balanced Fisher–Yates shuffle（seed 462202203）
- logical manifest SHA-256: `5698296ecef88f3e2611bc5d2b487857bc37cea115055acd45bd4345a64603b5`
- assignment sequence SHA-256: `d2d559e33d1ec824d0052c2f44e2c23d55c7321d5f68ccefd846c2bf6e0b2cf9`
- 推論: CPU、正式 repo-managed artifact
- 95% CI / standard error: game を cluster とする sandwich 推定（同一 game 内 5 seats の相関を保持）
- All-Pass: exposure と reward 平均には含め、通常の win/loss および role 別 denominator からは除外
- mirrored 補助評価: 未実施（50/50 mixed 主評価を 50,000 games で完遂することを優先）

## 3. 50/50 assignment確認

| seat | COM-AI | COM-RuleBase | AI比率 |
| ---: | ---: | ---: | ---: |
| 0 | 25,000 | 25,000 | 50.00% |
| 1 | 25,000 | 25,000 | 50.00% |
| 2 | 25,000 | 25,000 | 50.00% |
| 3 | 25,000 | 25,000 | 50.00% |
| 4 | 25,000 | 25,000 | 50.00% |

All-Pass は 0 game（0.00%）でした。

## 4. 全体 COM-AI vs COM-RuleBase

| policy | player-game exposure | win denominator | win rate (SE; 95% CI) | mean relative reward (SE; 95% CI) | mean raw reward (SE; 95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| COM-AI | 125,000 | 125,000 | 54.09% (SE 0.13%; [53.84%, 54.33%]) | -0.1531 (SE 0.0086; [-0.1701, -0.1362]) | 6.9439 (SE 0.0406; [6.8643, 7.0236]) |
| COM-RuleBase | 125,000 | 125,000 | 50.69% (SE 0.10%; [50.49%, 50.89%]) | 0.1531 (SE 0.0087; [0.1361, 0.1701]) | 6.3138 (SE 0.0472; [6.2212, 6.4064]) |

COM-AI − COM-RuleBase の勝率差は 3.40%（95% CI [3.02%, 3.79%]）、mean relative reward 差は -0.3062（95% CI [-0.3401, -0.2723]）です。

win rate と reward は同じ指標ではありません。raw / relative reward は現行 Reward v3 の役職依存 payoff を使い、relative reward は各 game の5人平均を引いています。今回のように policy 間で役職獲得率が大きく違う場合、全体 win rate と mean relative reward が逆方向になることがあります。このため、全体表だけでなく次の role 内比較を主に用いて解釈します。

## 5. Napoleon / Adjutant / Citizen 別

| role | policy | n | win rate (95% CI) | mean relative reward (95% CI) |
| --- | --- | ---: | ---: | ---: |
| Napoleon | COM-AI | 13,231 | 55.95% [55.11%, 56.80%] | 4.7755 [4.6378, 4.9131] |
| Napoleon | COM-RuleBase | 36,769 | 37.78% [37.28%, 38.27%] | 1.9749 [1.8951, 2.0546] |
| Adjutant | COM-AI | 27,633 | 46.39% [45.81%, 46.98%] | -0.7490 [-0.7712, -0.7268] |
| Adjutant | COM-RuleBase | 17,814 | 41.14% [40.42%, 41.86%] | -0.5808 [-0.6086, -0.5530] |
| Citizen | COM-AI | 84,136 | 56.32% [55.83%, 56.81%] | -0.7324 [-0.7528, -0.7121] |
| Citizen | COM-RuleBase | 70,417 | 59.84% [59.34%, 60.35%] | -0.6125 [-0.6337, -0.5912] |

### Citizen の composition-fixed 追加解析

上の Citizen 生集計では COM-AI が 3.52 pp 低い、という観測事実は変わりません。ただし focal Citizen の policy 以外に、敵 Napoleon / Adjutant、味方 Citizen、game 全体の policy 構成、bidding 後の role / hand selection が同時に異なります。したがって、この生差は Citizen policy 単体の因果効果ではありません。以下ではまず敵 Napoleon-side composition を固定します（solo Napoleon game は除外）。

| Napoleon | Adjutant | focal Citizen | n | Citizen win rate (95% CI) | mean relative reward (95% CI) |
| --- | --- | --- | ---: | ---: | ---: |
| COM-AI | COM-AI | COM-AI | 12,816 | 40.21% [38.97%, 41.45%] | -1.2025 [-1.2482, -1.1567] |
| COM-AI | COM-AI | COM-RuleBase | 9,474 | 43.30% [41.94%, 44.66%] | -1.1056 [-1.1564, -1.0548] |
| COM-AI | COM-RuleBase | COM-AI | 7,379 | 41.50% [39.87%, 43.13%] | -1.1975 [-1.2586, -1.1363] |
| COM-AI | COM-RuleBase | COM-RuleBase | 6,052 | 46.96% [45.21%, 48.71%] | -1.0105 [-1.0769, -0.9440] |
| COM-RuleBase | COM-AI | COM-AI | 32,745 | 56.60% [55.83%, 57.38%] | -0.6470 [-0.6764, -0.6176] |
| COM-RuleBase | COM-AI | COM-RuleBase | 27,864 | 59.75% [58.96%, 60.55%] | -0.5392 [-0.5697, -0.5086] |
| COM-RuleBase | COM-RuleBase | COM-AI | 21,172 | 61.97% [61.02%, 62.91%] | -0.4714 [-0.5080, -0.4348] |
| COM-RuleBase | COM-RuleBase | COM-RuleBase | 18,839 | 65.99% [65.05%, 66.92%] | -0.3242 [-0.3608, -0.2875] |

さらに、focal 以外の Citizen 2席にいる COM-AI 人数も固定した参考解析です。`n < 1,000` は小標本の参考値として扱います。敵・味方 composition を固定しても bidding による focal role / hand selection は統制されないため、これも完全な因果比較ではありません。

| Napoleon | Adjutant | other Citizen AI | focal Citizen | n | Citizen win rate (95% CI) | mean relative reward (95% CI) | note |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| COM-AI | COM-AI | 0 | COM-AI | 2,332 | 43.40% [41.38%, 45.41%] | -1.1020 [-1.1768, -1.0272] |  |
| COM-AI | COM-AI | 0 | COM-RuleBase | 1,722 | 46.86% [42.78%, 50.95%] | -1.0000 [-1.1539, -0.8461] |  |
| COM-AI | COM-AI | 1 | COM-AI | 6,176 | 41.16% [39.42%, 42.90%] | -1.1700 [-1.2341, -1.1059] |  |
| COM-AI | COM-AI | 1 | COM-RuleBase | 4,664 | 43.40% [41.38%, 45.41%] | -1.1020 [-1.1768, -1.0272] |  |
| COM-AI | COM-AI | 2 | COM-AI | 4,308 | 37.12% [34.62%, 39.62%] | -1.3033 [-1.3950, -1.2117] |  |
| COM-AI | COM-AI | 2 | COM-RuleBase | 3,088 | 41.16% [39.42%, 42.90%] | -1.1700 [-1.2341, -1.1059] |  |
| COM-AI | COM-RuleBase | 0 | COM-AI | 1,540 | 46.88% [44.39%, 49.38%] | -1.0184 [-1.1133, -0.9236] |  |
| COM-AI | COM-RuleBase | 0 | COM-RuleBase | 1,248 | 55.05% [50.27%, 59.83%] | -0.7154 [-0.8981, -0.5327] |  |
| COM-AI | COM-RuleBase | 1 | COM-AI | 3,448 | 41.24% [38.92%, 43.57%] | -1.2099 [-1.2974, -1.1223] |  |
| COM-AI | COM-RuleBase | 1 | COM-RuleBase | 3,080 | 46.88% [44.39%, 49.38%] | -1.0184 [-1.1133, -0.9236] |  |
| COM-AI | COM-RuleBase | 2 | COM-AI | 2,391 | 38.39% [35.02%, 41.77%] | -1.2949 [-1.4209, -1.1688] |  |
| COM-AI | COM-RuleBase | 2 | COM-RuleBase | 1,724 | 41.24% [38.92%, 43.57%] | -1.2099 [-1.2974, -1.1223] |  |
| COM-RuleBase | COM-AI | 0 | COM-AI | 6,877 | 60.20% [59.04%, 61.36%] | -0.5240 [-0.5684, -0.4797] |  |
| COM-RuleBase | COM-AI | 0 | COM-RuleBase | 5,931 | 62.62% [60.49%, 64.75%] | -0.4407 [-0.5229, -0.3584] |  |
| COM-RuleBase | COM-AI | 1 | COM-AI | 16,358 | 56.93% [55.85%, 58.00%] | -0.6360 [-0.6768, -0.5952] |  |
| COM-RuleBase | COM-AI | 1 | COM-RuleBase | 13,754 | 60.20% [59.04%, 61.36%] | -0.5240 [-0.5684, -0.4797] |  |
| COM-RuleBase | COM-AI | 2 | COM-AI | 9,510 | 53.44% [51.70%, 55.17%] | -0.7547 [-0.8202, -0.6892] |  |
| COM-RuleBase | COM-AI | 2 | COM-RuleBase | 8,179 | 56.93% [55.85%, 58.00%] | -0.6360 [-0.6768, -0.5952] |  |
| COM-RuleBase | COM-RuleBase | 0 | COM-AI | 4,615 | 66.61% [65.25%, 67.97%] | -0.3004 [-0.3535, -0.2474] |  |
| COM-RuleBase | COM-RuleBase | 0 | COM-RuleBase | 4,347 | 68.94% [66.56%, 71.33%] | -0.2192 [-0.3128, -0.1256] |  |
| COM-RuleBase | COM-RuleBase | 1 | COM-AI | 10,524 | 62.45% [61.14%, 63.76%] | -0.4525 [-0.5032, -0.4019] |  |
| COM-RuleBase | COM-RuleBase | 1 | COM-RuleBase | 9,230 | 66.61% [65.25%, 67.97%] | -0.3004 [-0.3535, -0.2474] |  |
| COM-RuleBase | COM-RuleBase | 2 | COM-AI | 6,033 | 57.58% [55.42%, 59.74%] | -0.6351 [-0.7185, -0.5517] |  |
| COM-RuleBase | COM-RuleBase | 2 | COM-RuleBase | 5,262 | 62.45% [61.14%, 63.76%] | -0.4525 [-0.5032, -0.4019] |  |

敵 Napoleon/Adjutant を固定した focal AI−RuleBase の Citizen 勝率差は AI/AI: -3.09 pp、AI/RuleBase: -5.46 pp、RuleBase/AI: -3.15 pp、RuleBase/RuleBase: -4.02 pp でした。さらに other Citizen AI count まで固定し、両 policy とも n≥1,000 の 12 層では、AI が高い層 0、低い層 12 でした。全比較層で AI の観測勝率が低く、差の範囲は -8.16 pp から -2.24 pp でした。観測できる敵・味方 composition を固定しても差は解消しておらず、Citizen は正式 COM-AI の残存課題として認めます。ただし focal policy の効果に加えて seat / hand と bidding による role selection が残るため、Citizen 打牌 policy 単体が因果的に弱いとまでは断定しません。

## 6. Napoleon-side composition 別

| Napoleon | Adjutant | n | Napoleon-side win / contract success | mean contract margin (95% CI) |
| --- | --- | ---: | ---: | ---: |
| COM-AI | COM-AI | 7,430 | 58.48% [57.36%, 59.60%] | 0.0229 [-0.0441, 0.0898] |
| COM-AI | COM-RuleBase | 4,477 | 56.04% [54.59%, 57.50%] | -0.1135 [-0.1987, -0.0282] |
| COM-RuleBase | COM-AI | 20,203 | 41.95% [41.27%, 42.63%] | -1.2197 [-1.2655, -1.1740] |
| COM-RuleBase | COM-RuleBase | 13,337 | 36.14% [35.32%, 36.96%] | -1.7929 [-1.8516, -1.7342] |
| COM-AI | None (solo) | 1,324 | 41.47% [38.81%, 44.12%] | -1.1631 [-1.3187, -1.0076] |
| COM-RuleBase | None (solo) | 3,229 | 18.46% [17.12%, 19.80%] | -3.8111 [-3.9338, -3.6883] |

## 7. bidding / contract diagnostics

### Role acquisition（全 policy exposure 比）

| policy | exposure | Napoleon | Adjutant | Citizen | All-Pass/no-contract |
| --- | ---: | ---: | ---: | ---: | ---: |
| COM-AI | 125,000 | 10.58% | 22.11% | 67.31% | 0.00% |
| COM-RuleBase | 125,000 | 29.42% | 14.25% | 56.33% | 0.00% |

Napoleon になった player の contract 指標:

| policy | n | success rate (95% CI) | mean margin (95% CI) | mean target | Napoleon-side point cards |
| --- | ---: | ---: | ---: | ---: | ---: |
| COM-AI | 13,231 | 55.95% [55.11%, 56.80%] | -0.1419 [-0.1922, -0.0917] | 13.6181 | 13.4762 |
| COM-RuleBase | 36,769 | 37.78% [37.28%, 38.27%] | -1.6552 [-1.6906, -1.6198] | 14.2038 | 12.5486 |

役職獲得率と役職内成績は別々に示しています。したがって、COM-AI の全体差を単に『Napoleon になりやすい／なりにくい』こととは混同しません。

### Phase call sample audit

- COM-AI adjutant: game 8, seat 1; composition {"playing":"ppo-separated-v1000","bidding":"frozen-raise-v1","nonPlaying":"parameterized-adjutant-exchange-v1"}; calls bidding=3, adjutant=1, exchange=1, playing=10
- COM-AI exchange: game 8, seat 1; composition {"playing":"ppo-separated-v1000","bidding":"frozen-raise-v1","nonPlaying":"parameterized-adjutant-exchange-v1"}; calls bidding=3, adjutant=1, exchange=1, playing=10
- COM-AI bidding: game 14, seat 3; composition {"playing":"ppo-separated-v1000","bidding":"frozen-raise-v1","nonPlaying":"parameterized-adjutant-exchange-v1"}; calls bidding=2, adjutant=0, exchange=0, playing=10
- COM-AI playing: game 14, seat 3; composition {"playing":"ppo-separated-v1000","bidding":"frozen-raise-v1","nonPlaying":"parameterized-adjutant-exchange-v1"}; calls bidding=2, adjutant=0, exchange=0, playing=10
- COM-RuleBase bidding: game 13, seat 0; composition {"playing":"rule-based","bidding":"rule-based","nonPlaying":"rule-based"}; calls bidding=2, adjutant=1, exchange=1, playing=10
- COM-RuleBase adjutant: game 13, seat 0; composition {"playing":"rule-based","bidding":"rule-based","nonPlaying":"rule-based"}; calls bidding=2, adjutant=1, exchange=1, playing=10
- COM-RuleBase exchange: game 13, seat 0; composition {"playing":"rule-based","bidding":"rule-based","nonPlaying":"rule-based"}; calls bidding=2, adjutant=1, exchange=1, playing=10
- COM-RuleBase playing: game 13, seat 0; composition {"playing":"rule-based","bidding":"rule-based","nonPlaying":"rule-based"}; calls bidding=2, adjutant=1, exchange=1, playing=10

## 8. illegal / fallback / invariant

- illegal: 0
- fallback: 0
- invariant failure: 0
- other game failure: 0
- aggregate phase calls: COM-AI {"playingCalls":1250000,"biddingCalls":200839,"adjutantCalls":13231,"exchangeCalls":13231,"fallbackCount":0,"illegalCount":0}
- aggregate phase calls: COM-RuleBase {"playingCalls":1250000,"biddingCalls":193851,"adjutantCalls":36769,"exchangeCalls":36769,"fallbackCount":0,"illegalCount":0}

## 9. 結論

全体では COM-AI の勝率は 54.09%、COM-RuleBase は 50.69%で、差は +3.40 pp（95% CI 3.02% から 3.79%）でした。一方、mean relative reward はそれぞれ -0.1531 と 0.1531で、AI−RuleBase 差は -0.3062（95% CI -0.3401 から -0.2723）です。つまり全体勝率は AI が明確に高いものの、役職構成を反映する relative reward は逆方向でした。

Napoleon では COM-AI 55.95%、RuleBase 37.78%（差 +18.17 pp）、Adjutant では COM-AI 46.39%、RuleBase 41.14%（差 +5.25 pp）、Citizen では COM-AI 56.32%、RuleBase 59.84%（差 -3.52 pp）でした。Citizen の -3.52 pp は単純な role-conditioned 観測差であり、敵・味方 composition や bidding 後の role / hand selection を統制していないため、Citizen 打牌 policy 単体が弱いことを意味しません。逆に composition bias だけで全差を説明できるとも、この生集計だけからは断定しません。

敵 Napoleon/Adjutant を固定した focal AI−RuleBase の Citizen 勝率差は AI/AI: -3.09 pp、AI/RuleBase: -5.46 pp、RuleBase/AI: -3.15 pp、RuleBase/RuleBase: -4.02 pp でした。さらに other Citizen AI count まで固定し、両 policy とも n≥1,000 の 12 層では、AI が高い層 0、低い層 12 でした。全比較層で AI の観測勝率が低く、差の範囲は -8.16 pp から -2.24 pp でした。観測できる敵・味方 composition を固定しても差は解消しておらず、Citizen は正式 COM-AI の残存課題として認めます。ただし focal policy の効果に加えて seat / hand と bidding による role selection が残るため、Citizen 打牌 policy 単体が因果的に弱いとまでは断定しません。

最大の改善は Napoleon で見えます。COM-AI Napoleon は target を平均 0.5857 低く宣言し、Napoleon-side point cards は平均 +0.9276、contract margin は +1.5133 動きました。これは frozen bidding の契約選択と、Napoleon / Adjutant の non-playing・playing による契約実行の両方が寄与した可能性を示します。composition 比較でも Adjutant を RuleBase から AI に替えた組合せの成績が上がっています。全 phase が同時に異なるため、単一 phase の因果効果とは断定しません。

結論として、全体 win rate と Napoleon / Adjutant の role-conditioned 実戦成績では正式 COM-AI が上回りました。一方、全体 relative reward は逆方向です。Citizen は composition-fixed 追加解析を含めても観測研究であり、policy 単体の優劣をこの mixed evaluation だけで確定しません。正式 COM-AI は特に Napoleon-side を大きく強化した、という点を確かな成果とし、Citizen については独立した打牌単体評価とは実験条件を分けて扱うのが正確な締めです。

この評価は現時点の正式 COM-AI をそのまま測った締めの結果です。結果に応じた再学習や policy の差し替えは行っていません。
