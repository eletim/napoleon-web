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

Napoleon では COM-AI 55.95%、RuleBase 37.78%（差 +18.17 pp）、Adjutant では COM-AI 46.39%、RuleBase 41.14%（差 +5.25 pp）、Citizen では COM-AI 56.32%、RuleBase 59.84%（差 -3.52 pp）でした。Napoleon / Adjutant の差と Citizen の差を分けることで、AI の総合差がどの立場で生じたかを確認できます。

最大の改善は Napoleon で見えます。COM-AI Napoleon は target を平均 0.5857 低く宣言し、Napoleon-side point cards は平均 +0.9276、contract margin は +1.5133 動きました。これは frozen bidding の契約選択と、Napoleon / Adjutant の non-playing・playing による契約実行の両方が寄与した可能性を示します。composition 比較でも Adjutant を RuleBase から AI に替えた組合せの成績が上がっています。一方で AI Citizen の勝率は下がっており、playing を含む AI の優位は全役職に一様ではありません。全 phase が同時に異なるため、単一 phase の因果効果とは断定しません。

結論として、『作った AI は RuleBase を超えたか』には、全体 win rate と Napoleon / Adjutant の実戦成績では明確に yes です。しかし Citizen と relative reward では no であり、『全役職・全指標で全面的に超えた』とは言えません。正式 COM-AI は特に Napoleon-side を大きく強化した一方、Citizen performance と bidding による role acquisition の偏りを残した、というのがこの最終評価の正確な締めです。

この評価は現時点の正式 COM-AI をそのまま測った締めの結果です。結果に応じた再学習や policy の差し替えは行っていません。
