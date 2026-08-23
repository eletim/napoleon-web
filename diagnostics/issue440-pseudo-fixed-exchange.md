# Issue 440 pseudo-fixed exchange teacher diagnostic

Branch base: `feature/issue-202-non-playing-ai`

## Implementation

- Fixed unit: `originalHandCardIds` 10 + `kittyPickupCardIds` 3 per `fixedThirteenGroupId`.
- Per repeat: only the remaining 40 cards are seeded-shuffled and dealt to the four opponents.
- Bidding starts from a fresh initial bidding state for every repeat. No artificial bidding history is injected, and hidden hands are not reshuffled after bidding.
- Candidate policy: `frozen-raise-v1`, with PASS forbidden while any legal BID exists. If Frozen would prefer PASS, the candidate takes the highest-EV legal BID from Frozen's legal BID evaluations.
- Opponent policy: unchanged legal semantics, per-seat seeded 1:1:1 mix of `frozen-raise-v1`, `strong-rule-based-bidding-v1`, and `conservative-bidding-v1`.
- Accepted states require `contract.napoleonPlayerId == candidate`. Failed candidate-Napoleon deals are rejected; contracts are not rewritten.
- Exchange teacher uses existing #434-style 286 discard combinations and rollout labels, with compact396 input.

## Phase 1 Diagnostic

Command scale: `50 fixedThirteen groups x 10 accepted deals/group`, diagnostic-only.

- Accepted/rejected: 500 accepted, 0 rejected, acceptance rate 1.000.
- Rejection reasons: none.
- Target distribution: 13:139, 14:179, 15:125, 16:42, 17:12, 18:3.
- Suit distribution: spades:234, diamonds:160, hearts:71, clubs:35.
- Bidding history action count: min 5, max 34, mean 10.42, median 10.
- Unique bidding history hashes: 361 / 500.
- Opponent policy counts: Frozen 659, strong RuleBased 697, Conservative 644.
- Opponent policy ratios: Frozen 0.3295, strong RuleBased 0.3485, Conservative 0.3220.
- Same-group contract diversity: target mean unique 3.68, suit mean unique 1.40, target+suit mean unique 4.22.

Conclusion: distribution did not collapse to high targets. The 1:1:1 opponent mix is close enough at 2,000 opponent seats. Candidate acceptance is 100% in this diagnostic because candidate no-PASS Frozen can usually outbid naturally; this should be watched at larger scale but does not require contract rewriting.

## Phase 2 Mini Ablation

Full 1,000-state ablation was not run in this workspace because teacher rollout cost measured about 2.6-3.0 seconds per accepted state on CPU. Instead, equal 30-state smoke ablations were run:

| layout | states | samples | final Pearson | final pairwise | model margin regret | model relative reward regret | RuleBased margin regret | RuleBased relative reward regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 groups x 1 | 30 | 8,580 | 0.523 | 0.568 | 6.33 | 23.33 | 4.67 | 10.83 |
| 6 groups x 5 | 30 | 8,580 | -0.187 | 0.488 | 6.20 | 13.50 | 4.40 | 7.00 |
| 3 groups x 10 | 30 | 8,580 | 0.090 | 0.517 | 3.90 | 17.75 | 2.10 | 6.50 |

Split leakage guards passed for all runs:

- `sourceStateKey`: 0 cross-split leakage.
- `fixedThirteenGroupId`: 0 cross-split leakage.
- `dealSeed`: 0 cross-split leakage.
- `hiddenDealChecksum`: 0 cross-split leakage.
- `pickupHand`: 0 cross-split leakage.

Same-13 final observations:

- 30x1 has no repeated hidden deals per final group.
- 6x5 final group: teacher-best discard varied across all 5 hidden deals; model selected 1 unique discard.
- 3x10 final group: teacher-best discard varied across 9 of 10 hidden deals; model selected 4 unique discards.

## Baseline Comparison

#438 compact396 baseline:

- Pearson 0.409
- pairwise 0.592
- margin regret 4.60
- relative reward regret 12.30
- RuleBased margin regret 4.59

The 30-state smoke ablation is too small to claim improvement. It did not beat #438 pairwise or RuleBased comparison. The 3x10 smoke had lower model margin regret than #438 but its held-out split is only one 10-deal group, and RuleBased was stronger on that split.

## Recommendation

Adopt the pseudo-fixed generator as an experimental teacher generator, but do not promote this run's small-scale results as the formal teacher benchmark. The Phase 1 distribution is healthy enough to justify a full 1,000-state run outside this short workspace session.

Among the smoke ablations, 10 repeats/group showed the best model margin regret and the clearest same-13 teacher variation, but the split is too small. The next serious run should use the requested 1,000-state layouts: `1000x1`, `200x5`, and `100x10`.

Do not move to deputy+kitty joint optimization yet. First run the full pseudo-fixed exchange teacher ablation; if 100x10 still fails to beat #438 on pairwise and RuleBased margin regret, then proceed to joint deputy+exchange or teacher-target redesign.
