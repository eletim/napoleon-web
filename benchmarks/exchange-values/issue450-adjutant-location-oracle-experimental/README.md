# Issue #450 adjutant-location oracle (diagnostic only)

This artifact tests whether the hidden actual location of the called adjutant card explains
the compact396 exchange scorer's ranking error. It is not runtime-compatible and must not be
wired or promoted to Frozen.

- Input: compact396 + five-class actual pre-exchange location one-hot = compact401
- Architecture: `[512, 512, 256, 256] -> scalar contract margin`
- Training: the existing #438 1,000-state and #442 1,000x1 datasets; 12 fixed-audit identity
  collisions removed, leaving 1,988 states / 568,568 candidates
- Fixed audit: the exact #446/#448 200-state full-gold set, manifest SHA-256
  `4f38f95314aa6922c549c54c50532ae0b7aa95dd8fc2706a235d3506467d3da7`
- Checkpoint SHA-256: `3fe8ace7f81f6bae1e3089c6eeda0998e1e47660e5c1d8d1580f3f0b5d9dd632`

The top16 containment change is only `0.2249 -> 0.2289` (+0.0040), practical K16
containment is `0.2576 -> 0.2616`, and practical regret is `1.0977 -> 1.0871`.
This is far below the +0.10 support threshold, so actual adjutant location alone is not the
main hidden factor. No 5k/10k scale-up was run.

See `diagnostics/issue450-adjutant-location-oracle.md` for the comparison and conclusion.
