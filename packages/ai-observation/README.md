# @napoleon/ai-observation

This package converts automated `playing` card decisions into a fixed-length,
deterministic numeric schema for future Actor training and hidden-card ownership
supervision.

It currently covers only the `playing` phase and only `play-card` actions.
Past public bidding actions are preserved in every playing sample. Adjutant
selection, buried-card exchange, Python, neural-network models, PyTorch, ONNX,
dataset files, and batch self-play generation are not included.

## Schema

- Schema version: `1`
- Card order: 53 fixed ids
  - spades `A, K, Q, J, 10, 9, 8, 7, 6, 5, 4, 3, 2`
  - hearts `A, K, Q, J, 10, 9, 8, 7, 6, 5, 4, 3, 2`
  - diamonds `A, K, Q, J, 10, 9, 8, 7, 6, 5, 4, 3, 2`
  - clubs `A, K, Q, J, 10, 9, 8, 7, 6, 5, 4, 3, 2`
  - joker
- Actor action space: 53 card indices
- Relative players: 5 seats rotated so the acting player is index `0`
- Bidding history: fixed-length public action sequence with max length `117`
- Belief owner classes: `0..4` for relative players currently holding a card,
  `5` for not currently in any player's hand

Actor observations are built only from `PlayerObservation`, the table player
order, and public actions that occurred before the playing decision. They do not
consume `ActualCardState`, actual opponent hands, excluded card ids, or any other
complete-information field. Complete card locations are used only by
`encodeBeliefTarget` to build supervised labels.

## Bidding History

`biddingHistory` stores only real bidding decisions whose `phase` is `bidding`
and whose `step` is before the playing decision. The encoder does not reconstruct
history from `view.bidding`, because that field is `null` after bidding has
resolved.

- `actionTypeIndices`: `0` pass, `1` bid, `-1` empty
- `playerIndices`: `0..4` actor-relative player index, `-1` empty
- `suitIndices`: `0` spades, `1` hearts, `2` diamonds, `3` clubs, `-1` pass or
  empty
- `targetPointCards`: bid target card count, `0` for pass or empty
- `actionMask`: `1` real bidding action, `0` empty slot

The suit index order matches `trumpSuitOneHot` for model I/O. It is separate
from the internal bidding suit priority. An all-pass contract is represented as
the real five pass actions. The automatic 12-card spades contract is visible via
`contractTargetPointCards` and `trumpSuitOneHot`; it is not added as a synthetic
bid.

## Major Shapes

| Field | Shape |
| --- | --- |
| `trumpSuitOneHot` | `[4]` |
| `napoleonPlayerOneHot` | `[5]` |
| `revealedAdjutantPlayerOneHot` | `[6]` |
| `calledAdjutantCardMask` | `[53]` |
| `selfHandMask` | `[53]` |
| `legalPlayMask` | `[53]` |
| `handCountByPlayer` | `[5]` |
| `capturedPointCardMaskByPlayer` | `[5][53]` |
| `currentTrickCardIndices` | `[5]` |
| `currentTrickPlayerIndices` | `[5]` |
| `currentTrickSlotMask` | `[5]` |
| `completedTrickCardIndices` | `[50]` |
| `completedTrickPlayerIndices` | `[50]` |
| `completedTrickSlotMask` | `[50]` |
| `completedTrickWinnerIndices` | `[10]` |
| `completedTrickMask` | `[10]` |
| `biddingHistory.actionTypeIndices` | `[117]` |
| `biddingHistory.playerIndices` | `[117]` |
| `biddingHistory.suitIndices` | `[117]` |
| `biddingHistory.targetPointCards` | `[117]` |
| `biddingHistory.actionMask` | `[117]` |
| `latestBuriedEventPointCardMask` | `[53]` |
| `ownerClassByCard` | `[53]` |
| `hiddenOwnershipLossMask` | `[53]` |

Missing card and player slots use `-1` with a companion slot mask of `0`.

## Validation

The validators check fixed lengths, mask values, finite numeric values, and the
known schema version. Bidding history validation checks the fixed-length arrays
and mask values before samples are handed to JSONL or downstream training code.
