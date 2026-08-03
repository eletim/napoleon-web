# @napoleon/ai-observation

This package converts automated `playing` card decisions into a fixed-length,
deterministic numeric schema for future Actor training and hidden-card ownership
supervision.

It currently covers only the `playing` phase and only `play-card` actions.
Bidding, adjutant selection, buried-card exchange, Python, neural-network
models, PyTorch, ONNX, dataset files, and batch self-play generation are not
included.

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
- Belief owner classes: `0..4` for relative players currently holding a card,
  `5` for not currently in any player's hand

Actor observations are built only from `PlayerObservation` plus the table player
order. They do not consume `ActualCardState`, actual opponent hands, excluded card
ids, or any other complete-information field. Complete card locations are used
only by `encodeBeliefTarget` to build supervised labels.

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
| `latestBuriedEventPointCardMask` | `[53]` |
| `ownerClassByCard` | `[53]` |
| `hiddenOwnershipLossMask` | `[53]` |

Missing card and player slots use `-1` with a companion slot mask of `0`.
