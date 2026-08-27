# Final COM-AI roster evaluation

旧50/50 mixed evaluationに代わるPR #203の正式最終評価です。1ゲーム内のCOM-AI人数を0〜5で固定し、各rosterを1,000 gamesずつ評価しました。player単位のAI対RB勝率ではなく、Napoleon-sideの構成と勝敗を主に読みます。

## 1. 正式AI構成

| preset | playing | bidding | nonPlaying |
| --- | --- | --- | --- |
| COM-AI | `ppo-separated-v1000` | `frozen-raise-v1` | `parameterized-adjutant-exchange-v1` |
| COM-RuleBase | `rule-based` | `rule-based` | `rule-based` |

legacy full-policy、旧adjutant/exchange PPO、experimental artifactは使用していません。

## 2. 6 roster条件

| roster | games | combinations | games / combination |
| --- | ---: | ---: | ---: |
| RB5 / AI0 | 1,000 | 1 | 1000 |
| RB4 / AI1 | 1,000 | 5 | 200 |
| RB3 / AI2 | 1,000 | 10 | 100 |
| RB2 / AI3 | 1,000 | 10 | 100 |
| RB1 / AI4 | 1,000 | 5 | 200 |
| RB0 / AI5 | 1,000 | 1 | 1000 |

game seedは `462600000..462605999`、scheduleは `lexicographic-seat-combinations-round-robin-v1` です。

## 3. Seat balancing

| roster | AI seats | scheduled | completed |
| --- | --- | ---: | ---: |
| RB5 / AI0 | none | 1000 | 1000 |
| RB4 / AI1 | 0 | 200 | 200 |
| RB4 / AI1 | 1 | 200 | 200 |
| RB4 / AI1 | 2 | 200 | 200 |
| RB4 / AI1 | 3 | 200 | 200 |
| RB4 / AI1 | 4 | 200 | 200 |
| RB3 / AI2 | 0,1 | 100 | 100 |
| RB3 / AI2 | 0,2 | 100 | 100 |
| RB3 / AI2 | 0,3 | 100 | 100 |
| RB3 / AI2 | 0,4 | 100 | 100 |
| RB3 / AI2 | 1,2 | 100 | 100 |
| RB3 / AI2 | 1,3 | 100 | 100 |
| RB3 / AI2 | 1,4 | 100 | 100 |
| RB3 / AI2 | 2,3 | 100 | 100 |
| RB3 / AI2 | 2,4 | 100 | 100 |
| RB3 / AI2 | 3,4 | 100 | 100 |
| RB2 / AI3 | 0,1,2 | 100 | 100 |
| RB2 / AI3 | 0,1,3 | 100 | 100 |
| RB2 / AI3 | 0,1,4 | 100 | 100 |
| RB2 / AI3 | 0,2,3 | 100 | 100 |
| RB2 / AI3 | 0,2,4 | 100 | 100 |
| RB2 / AI3 | 0,3,4 | 100 | 100 |
| RB2 / AI3 | 1,2,3 | 100 | 100 |
| RB2 / AI3 | 1,2,4 | 100 | 100 |
| RB2 / AI3 | 1,3,4 | 100 | 100 |
| RB2 / AI3 | 2,3,4 | 100 | 100 |
| RB1 / AI4 | 0,1,2,3 | 200 | 200 |
| RB1 / AI4 | 0,1,2,4 | 200 | 200 |
| RB1 / AI4 | 0,1,3,4 | 200 | 200 |
| RB1 / AI4 | 0,2,3,4 | 200 | 200 |
| RB1 / AI4 | 1,2,3,4 | 200 | 200 |
| RB0 / AI5 | 0,1,2,3,4 (all) | 1000 | 1000 |

## 4. Raw保存先 / manifest

- raw directory: `/tmp/napoleon-final-roster-eval`
- games.jsonl rows: 6,000
- logical manifest SHA-256: `9218f7bcc735018abb045dd83647ee6ad4371404af7eff2534d6527cc00eb0e7`
- schedule SHA-256: `c877fb1069e6544bf60981d901464c753f47e5aed5e9fc080abe45baaeefd3ff`
- files: `games.jsonl`, `summary.json`, `config.json`, `manifest.json`, `report.md`

## 5. Roster別1,000-game結果

| roster | games | Napoleon wins | Citizen wins | Napoleon win rate | All-Pass | Napoleon policy AI/RB | Adjutant policy AI/RB | solo | mean target | mean margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RB5 / AI0 | 1,000 | 296 | 704 | 29.60% | 0 | 0/1000 | 0/912 | 88 | 14.597 | -2.596 |
| RB4 / AI1 | 1,000 | 359 | 641 | 35.90% | 0 | 71/929 | 245/651 | 104 | 14.372 | -2.029 |
| RB3 / AI2 | 1,000 | 390 | 610 | 39.00% | 0 | 169/831 | 485/425 | 90 | 14.167 | -1.531 |
| RB2 / AI3 | 1,000 | 445 | 555 | 44.50% | 0 | 276/724 | 671/245 | 84 | 13.919 | -1.098 |
| RB1 / AI4 | 1,000 | 518 | 482 | 51.80% | 0 | 548/452 | 808/84 | 108 | 13.641 | -0.402 |
| RB0 / AI5 | 1,000 | 617 | 383 | 61.70% | 0 | 1000/0 | 906/0 | 94 | 13.308 | 0.266 |

## 6. Roster × Napoleon-side 6分類

| roster | classification | games | Napoleon wins | Citizen wins | Napoleon win rate | mean target | mean margin |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RB5 / AI0 | Napoleon=RB / Adjutant=RB | 912 | 277 | 635 | 30.37% | 14.607 | -2.487 |
| RB5 / AI0 | Napoleon=RB / Adjutant=AI | 0 | 0 | 0 | — | — | — |
| RB5 / AI0 | Napoleon=AI / Adjutant=RB | 0 | 0 | 0 | — | — | — |
| RB5 / AI0 | Napoleon=AI / Adjutant=AI | 0 | 0 | 0 | — | — | — |
| RB5 / AI0 | Solo Napoleon=RB | 88 | 19 | 69 | 21.59% | 14.489 | -3.727 |
| RB5 / AI0 | Solo Napoleon=AI | 0 | 0 | 0 | — | — | — |
| RB4 / AI1 | Napoleon=RB / Adjutant=RB | 592 | 214 | 378 | 36.15% | 14.443 | -1.966 |
| RB4 / AI1 | Napoleon=RB / Adjutant=AI | 245 | 98 | 147 | 40.00% | 14.314 | -1.657 |
| RB4 / AI1 | Napoleon=AI / Adjutant=RB | 59 | 29 | 30 | 49.15% | 14.136 | -0.932 |
| RB4 / AI1 | Napoleon=AI / Adjutant=AI | 0 | 0 | 0 | — | — | — |
| RB4 / AI1 | Solo Napoleon=RB | 92 | 11 | 81 | 11.96% | 14.283 | -4.326 |
| RB4 / AI1 | Solo Napoleon=AI | 12 | 7 | 5 | 58.33% | 13.917 | -0.500 |
| RB3 / AI2 | Napoleon=RB / Adjutant=RB | 318 | 125 | 193 | 39.31% | 14.365 | -1.569 |
| RB3 / AI2 | Napoleon=RB / Adjutant=AI | 444 | 176 | 268 | 39.64% | 14.171 | -1.435 |
| RB3 / AI2 | Napoleon=AI / Adjutant=RB | 107 | 46 | 61 | 42.99% | 13.850 | -0.907 |
| RB3 / AI2 | Napoleon=AI / Adjutant=AI | 41 | 19 | 22 | 46.34% | 13.927 | -0.341 |
| RB3 / AI2 | Solo Napoleon=RB | 69 | 10 | 59 | 14.49% | 13.971 | -4.232 |
| RB3 / AI2 | Solo Napoleon=AI | 21 | 14 | 7 | 66.67% | 13.810 | 0.381 |
| RB2 / AI3 | Napoleon=RB / Adjutant=RB | 145 | 52 | 93 | 35.86% | 14.110 | -1.697 |
| RB2 / AI3 | Napoleon=RB / Adjutant=AI | 523 | 245 | 278 | 46.85% | 13.971 | -1.010 |
| RB2 / AI3 | Napoleon=AI / Adjutant=RB | 100 | 48 | 52 | 48.00% | 13.900 | -0.620 |
| RB2 / AI3 | Napoleon=AI / Adjutant=AI | 148 | 77 | 71 | 52.03% | 13.635 | -0.345 |
| RB2 / AI3 | Solo Napoleon=RB | 56 | 11 | 45 | 19.64% | 13.857 | -3.375 |
| RB2 / AI3 | Solo Napoleon=AI | 28 | 12 | 16 | 42.86% | 13.643 | -0.786 |
| RB1 / AI4 | Napoleon=RB / Adjutant=RB | 0 | 0 | 0 | — | — | — |
| RB1 / AI4 | Napoleon=RB / Adjutant=AI | 407 | 194 | 213 | 47.67% | 13.907 | -0.577 |
| RB1 / AI4 | Napoleon=AI / Adjutant=RB | 84 | 53 | 31 | 63.10% | 13.726 | 0.310 |
| RB1 / AI4 | Napoleon=AI / Adjutant=AI | 401 | 236 | 165 | 58.85% | 13.429 | 0.057 |
| RB1 / AI4 | Solo Napoleon=RB | 45 | 7 | 38 | 15.56% | 13.600 | -3.422 |
| RB1 / AI4 | Solo Napoleon=AI | 63 | 28 | 35 | 44.44% | 13.190 | -0.984 |
| RB0 / AI5 | Napoleon=RB / Adjutant=RB | 0 | 0 | 0 | — | — | — |
| RB0 / AI5 | Napoleon=RB / Adjutant=AI | 0 | 0 | 0 | — | — | — |
| RB0 / AI5 | Napoleon=AI / Adjutant=RB | 0 | 0 | 0 | — | — | — |
| RB0 / AI5 | Napoleon=AI / Adjutant=AI | 906 | 581 | 325 | 64.13% | 13.315 | 0.455 |
| RB0 / AI5 | Solo Napoleon=RB | 0 | 0 | 0 | — | — | — |
| RB0 / AI5 | Solo Napoleon=AI | 94 | 36 | 58 | 38.30% | 13.245 | -1.553 |

全roster合計の構成別参考集計:

| classification | games | Napoleon wins | Citizen wins | Napoleon win rate | mean target | mean margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Napoleon=RB / Adjutant=RB | 1967 | 668 | 1299 | 33.96% | 14.482 | -2.124 |
| Napoleon=RB / Adjutant=AI | 1619 | 713 | 906 | 44.04% | 14.062 | -1.116 |
| Napoleon=AI / Adjutant=RB | 350 | 176 | 174 | 50.29% | 13.883 | -0.537 |
| Napoleon=AI / Adjutant=AI | 1496 | 913 | 583 | 61.03% | 13.394 | 0.247 |
| Solo Napoleon=RB | 350 | 58 | 292 | 16.57% | 14.117 | -3.889 |
| Solo Napoleon=AI | 218 | 97 | 121 | 44.50% | 13.372 | -1.046 |

## 7. Target / margin

targetとmarginは上のroster別・6分類表に併記しました。異なるrosterは独立seed集合であり、構成別の生差にはdeal/role selectionも含まれるため、単一phaseの因果効果とは断定しません。

## 8. illegal / fallback / invariant

- illegal: 0
- fallback: 0
- invariant failure: 0
- other game failure: 0
- All-Pass: 0

Sample phase audit:

- AI adjutant: game 1015, seat 0, composition={"playing":"ppo-separated-v1000","bidding":"frozen-raise-v1","nonPlaying":"parameterized-adjutant-exchange-v1"}, calls bidding=2, adjutant=1, exchange=1, playing=10, illegal=0, fallback=0
- AI bidding: game 1004, seat 4, composition={"playing":"ppo-separated-v1000","bidding":"frozen-raise-v1","nonPlaying":"parameterized-adjutant-exchange-v1"}, calls bidding=1, adjutant=0, exchange=0, playing=10, illegal=0, fallback=0
- AI exchange: game 1015, seat 0, composition={"playing":"ppo-separated-v1000","bidding":"frozen-raise-v1","nonPlaying":"parameterized-adjutant-exchange-v1"}, calls bidding=2, adjutant=1, exchange=1, playing=10, illegal=0, fallback=0
- AI playing: game 1004, seat 4, composition={"playing":"ppo-separated-v1000","bidding":"frozen-raise-v1","nonPlaying":"parameterized-adjutant-exchange-v1"}, calls bidding=1, adjutant=0, exchange=0, playing=10, illegal=0, fallback=0
- RB adjutant: game 0, seat 1, composition={"playing":"rule-based","bidding":"rule-based","nonPlaying":"rule-based"}, calls bidding=1, adjutant=1, exchange=1, playing=10, illegal=0, fallback=0
- RB bidding: game 0, seat 0, composition={"playing":"rule-based","bidding":"rule-based","nonPlaying":"rule-based"}, calls bidding=2, adjutant=0, exchange=0, playing=10, illegal=0, fallback=0
- RB exchange: game 0, seat 1, composition={"playing":"rule-based","bidding":"rule-based","nonPlaying":"rule-based"}, calls bidding=1, adjutant=1, exchange=1, playing=10, illegal=0, fallback=0
- RB playing: game 0, seat 0, composition={"playing":"rule-based","bidding":"rule-based","nonPlaying":"rule-based"}, calls bidding=2, adjutant=0, exchange=0, playing=10, illegal=0, fallback=0

## 9. 最終結論

Napoleon-side win rateは、RB5 / AI0 29.60%、RB4 / AI1 35.90%、RB3 / AI2 39.00%、RB2 / AI3 44.50%、RB1 / AI4 51.80%、RB0 / AI5 61.70%でした。今回の独立seed群ではAI人数が増える各段階で単調に上がり、AI0からAI5までの差は 32.10 percentage pointsでした。

全roster合計の記述集計では、Napoleon=AI / Adjutant=AI は 61.03%、Napoleon=AI / Adjutant=RB は 50.29%、Napoleon=RB / Adjutant=AI は 44.04%、Napoleon=RB / Adjutant=RB は 33.96%でした。Napoleon policyをAIにした構成、Adjutant policyをAIにした構成の双方で高い生勝率が観測され、AI+AIが最も高い値でした。

solo NapoleonはAI 44.50%、RB 16.57%でした。またroster平均では、AI0→AI5でdeclared targetが 14.597→13.308、contract marginが -2.596→0.266へ変化しました。

したがって、この固定roster評価では、正式COM-AIを増やした編成ほどNapoleon-side成績が良く、Napoleon・Adjutant・soloの構成別集計もCOM-AIの有効性と整合しています。一方、異なるrosterは同一dealの対比較ではなく、biddingによるrole selection、deal、Citizen側構成も同時に変わります。このため各差を特定phaseの因果効果やCitizen policy単体の優劣とは断定しません。

この結果を見てpolicy再学習、artifact再選定、reward/preset変更は行っていません。
