# Issue #448 listwise exchange reranker (experimental)

This is the best fixed-holdout result from the bounded Issue #448 comparison. It is
preserved for reproducibility, but it is **not** approved for runtime wiring, Frozen
promotion, or adjutant dataset regeneration because the improvement over compact396 is
too small.

- Input: compact396 (343 visible exchange-state features + 53-card discard mask)
- Model: MLP `[512, 512, 256, 256] -> scalar score`
- Objective: state-wise 286-candidate listwise loss, temperature 0.5
- Warm start: Issue #438 compact396 checkpoint
- Raw training pool: 2,000 existing exchange states / 572,000 samples
- Strict audit exclusion: 12 kitty-identity collisions removed as complete 286-candidate
  groups
- Effective training pool: 1,988 states / 568,568 samples
- Checkpoint SHA-256:
  `5c859933a798935a1bc2a92115b4263f3ab29ee50e3c535cbf5558e33d968e3a`
- Fixed #446 holdout manifest SHA-256:
  `4f38f95314aa6922c549c54c50532ae0b7aa95dd8fc2706a235d3506467d3da7`
- Leakage guard: passed with zero overlap for deal seed, hidden deal, original hand,
  kitty, bidding history, and visible source identity

Fixed full-gold result:

| metric | compact396 baseline | experimental listwise |
| --- | ---: | ---: |
| top16 containment | 0.1981 | 0.2249 |
| top32 containment | 0.3258 | 0.3623 |
| top64 containment | 0.5110 | 0.5534 |
| practical K=16 containment | 0.2358 | 0.2576 |
| practical K=16 margin regret | 1.1235 | 1.0977 |
| gold-best rank mean / median / p90 | 81.31 / 62 / 189 | 74.72 / 54 / 179 |

The complete fixed-holdout metrics and leakage report are in `full-gold-report.json`.
See `diagnostics/issue448-exchange-reranker.md` for the comparison and decision.
