# Issue #450: actual adjutant-location oracle

## Verdict

The oracle does not materially improve exchange ranking. Actual adjutant location alone is
not the principal hidden latent variable behind compact396 error. The prescribed +0.10
top16 signal did not appear (+0.0040), so the experiment stopped at the existing 2k scale;
5k/10k training was not run.

Do not implement `P(adjutant location | visible state, called card)` as the next isolated
exchange project on this evidence. The more promising next investigation is expectation over
the wider hidden-card configuration, teacher target noise, or a sequential exchange policy.
The same location-only decomposition is likewise not justified for the adjutant selector yet.

## Semantics and provenance

Location is the owner of the called card before exchange. It is never changed when Napoleon
later buries that card. Owner seats are normalized clockwise relative to Napoleon:
`opponentSeat1..4`; a card in Napoleon's original hand or the three-card kitty maps to the
single `selfKittySolo` class because no external adjutant exists.

The training overlay deterministically reconstructs the original #438 and pseudo-fixed #442
hidden deals from their seeds and fixed-thirteen rules. Its SHA-256 is
`68c4487e71bc9968a70f59966cdedeac8dbeac67b2edf352f5ecc39acc663c3a`.
The full-gold overlay is derived through the C++ engine from each fixed source seed and is
bound to the original #446 manifest SHA-256
`4f38f95314aa6922c549c54c50532ae0b7aa95dd8fc2706a235d3506467d3da7`;
its SHA-256 is `ecf3d6200331c73af0ac94554be15004bcf7dc7c128c62ff9f8976e320d0d16e`.
The full-gold deal invariant is exact: each opponent class has 2,000 groups and the
self/kitty class has 2,600 (200 deals × 10/10/10/10/13 cards).

Training reused #438 1k (`657ce74c...4164`) + #442 1k (`e0ddcf29...c2cec`). The same
#448 audit exclusion removed 12 kitty-identity collisions, leaving 1,988 states / 568,568
candidates with zero guarded train/audit overlap. Both models use the same compact396 warm
start, seed 436, listwise temperature 0.5, and `[512,512,256,256]` architecture. The oracle
only adds five zero-initialized first-layer inputs.

## Fixed full-gold comparison

| metric | #448 compact396 | compact401 oracle | delta |
| --- | ---: | ---: | ---: |
| containment@1 | 0.0228 | 0.0258 | +0.0029 |
| containment@4 | 0.0758 | 0.0789 | +0.0031 |
| containment@8 | 0.1346 | 0.1344 | -0.0002 |
| containment@16 | 0.2249 | 0.2289 | +0.0040 |
| containment@24 | 0.2993 | 0.3040 | +0.0046 |
| containment@32 | 0.3623 | 0.3697 | +0.0075 |
| containment@48 | 0.4642 | 0.4760 | +0.0118 |
| containment@64 | 0.5534 | 0.5607 | +0.0073 |
| containment@96 | 0.6884 | 0.6915 | +0.0031 |
| containment@128 | 0.7910 | 0.7962 | +0.0052 |
| rank mean / median / p90 | 74.72 / 54 / 179 | 73.88 / 53 / 179 | -0.84 / -1 / 0 |
| pairwise accuracy | 0.6081 | 0.6080 | -0.0001 |
| exact / top3 / top5 | 0.0228 / 0.0611 / 0.0901 | 0.0258 / 0.0623 / 0.0943 | +0.0029 / +0.0011 / +0.0042 |
| scorer top1 margin regret | 4.8251 | 4.7908 | -0.0342 |
| practical K16 containment | 0.2576 | 0.2616 | +0.0040 |
| practical K16 margin regret | 1.0977 | 1.0871 | -0.0107 |

## Oracle class diagnostics

| class | groups | rank mean / median / p90 | top16 | top1 regret | practical containment / regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| opponentSeat1 | 2,000 | 74.96 / 53.5 / 184 | 0.2385 | 4.6880 | 0.2705 / 1.0870 |
| opponentSeat2 | 2,000 | 72.25 / 51 / 175 | 0.2385 | 4.7435 | 0.2690 / 1.0650 |
| opponentSeat3 | 2,000 | 74.76 / 54 / 182.1 | 0.2190 | 4.8295 | 0.2575 / 1.0735 |
| opponentSeat4 | 2,000 | 75.23 / 56 / 179.1 | 0.2200 | 4.8150 | 0.2505 / 1.0835 |
| selfKittySolo | 2,600 | 72.60 / 51 / 176.1 | 0.2285 | 4.8581 | 0.2608 / 1.1173 |
| all external | 8,000 | 74.30 / 54 / 180 | 0.2290 | 4.7690 | 0.2619 / 1.0773 |

Called-card origin remains more diagnostic than exact external seat: original-hand and kitty
groups rank substantially better than opponent-hand groups.

| called-card origin | groups | rank mean / median / p90 | top16 | practical containment / regret |
| --- | ---: | ---: | ---: | ---: |
| original hand | 2,000 | 60.81 / 41 / 153 | 0.2855 | 0.3180 / 1.0450 |
| kitty | 600 | 63.12 / 39.5 / 156.2 | 0.2883 | 0.3217 / 1.0100 |
| opponent hand | 8,000 | 77.96 / 58 / 185 | 0.2103 | 0.2430 / 1.1034 |

The complete JSON also reports every class × contract suit and class × target cell. Their
top16 deltas are mixed rather than coherent (suit cells range -0.0278 to +0.0388; the largest
target deltas occur in tiny target-15/16 cells of 39/13 groups), so they do not rescue the
latent-location hypothesis.

## Scope

This issue adds diagnostic loaders, overlays, metrics, and an experimental checkpoint only.
There is no runtime wiring, Frozen promotion, reward change, playing/bidding retraining,
adjutant model retraining, or adjutant 10k regeneration.
