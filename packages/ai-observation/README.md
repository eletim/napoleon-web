# @napoleon/ai-observation

This package converts automated `bidding`, `choosing-adjutant`, `exchanging`,
and `playing` decisions into fixed-length, deterministic numeric schemas for future Actor training and
hidden-card ownership supervision.

Playing samples cover only `play-card` actions. Bidding samples cover only
`pass` and `bid` actions. Adjutant samples cover only `choose-adjutant` actions.
Exchange samples cover only `discard-cards` actions. Python, neural-network
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
- Adjutant action space: 53 card indices
- Adjutant legal mask: 53-card mask generated from `choose-adjutant` legal actions
- Exchange discard target space: 53-card exactly 3-hot mask
- Exchange legal discard mask: 53-card mask equal to Napoleon's 13-card self hand
- Bidding schema version: `1`
- Bidding action space: 29 fixed indices
  - index `0`: pass
  - index `1..28`: 13 to 19 cards x 4 suits
  - bid index: `1 + (targetPointCards - 13) * 4 + suitIndex`
- Relative players: 5 seats rotated so the acting player is index `0`
- Bidding history: fixed-length public action sequence with max length `117`
- Belief owner classes: `0..4` for relative players currently holding a card,
  `5` for not currently in any player's hand

Actor observations are built only from `PlayerObservation`, the table player
order, and public actions that occurred before the acting decision. They do not
consume `ActualCardState`, actual opponent hands, excluded card ids, or any other
complete-information field. Complete card locations are used only by
`encodeBeliefTarget` to build supervised labels.

## Bidding History

`biddingHistory` stores only real bidding decisions whose `phase` is `bidding`
and whose `step` is before the playing or exchange decision. The encoder does not
reconstruct history from `view.bidding`, because that field is `null` after
bidding has resolved.

`encodePlayingObservation()` requires a `biddingHistory` argument and never
silently falls back to an empty history. Tests or tooling that need an explicit
empty fixed-length history can call `createEmptyEncodedBiddingHistory()` and pass
that value. Normal training sample generation uses `encodeBiddingHistory()` to
derive the real public bidding history from the automated game record, and
`encodeBiddingHistory()` validates its generated schema before returning it.
Exchange training sample generation uses the same `EncodedBiddingHistory` schema.
Adjutant training sample generation also uses this same public history schema.

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

## Bidding Observation

`EncodedBiddingObservation` uses only the acting player's `PlayerObservation`,
the absolute table player order, and public bidding history. It contains:

- `relativePlayerIds`: `[5]`, with the acting player at index `0`
- `selfHandMask`: `[53]`
- `legalBidMask`: `[29]`
- `starterPlayerIndex`: `0..4`
- `highestBidPresent`: `0/1`
- `highestBidPlayerIndex`: `-1` or `0..4`
- `highestBidSuitIndex`: `-1` or `0..3`
- `highestBidTargetPointCards`: `0` or `13..19`
- `consecutivePassCount`: `0..5`
- `biddingHistory`: existing fixed `[117]` public history fields

`legalBidMask` is generated only from `decision.observation.legalActions`; it
does not recalculate bidding rules. `actorTarget` is the RuleBasedAgent-selected
action encoded into `0..28`, and the validator requires
`legalBidMask[actorTarget] === 1`. The all-pass automatic spades-12 contract is
not part of the action space, so the fifth all-pass teacher action remains
`pass = 0`.

## Adjutant Observation

`EncodedAdjutantObservation` uses only Napoleon's adjutant-choice-time
`PlayerObservation`, the absolute table player order, and the existing public
`EncodedBiddingHistory`. It contains:

- `relativePlayerIds`: `[5]`, with Napoleon at index `0`
- `trumpSuitOneHot`: `[4]`
- `contractTargetPointCards`: `12..19`
- `selfHandMask`: `[53]`, exactly 10 cards
- `legalAdjutantMask`: `[53]`
- `specialCardIndices`: public special card indices for the resolved trump
- `biddingHistory`: existing fixed `[117]` public history fields

`legalAdjutantMask` is generated only from `decision.observation.legalActions`
entries whose type is `choose-adjutant`. It does not apply RuleBasedAgent's
candidate filtering, so current game-core rules encode all 53 cards as legal,
including Joker and cards in Napoleon's own 10-card hand. `actorTarget` is the
RuleBasedAgent-selected `choose-adjutant.cardId` encoded as the existing
`CARD_IDS` index, and validation requires it to be inside `legalAdjutantMask`.

The observation is before exchange: the buried three cards have not joined
Napoleon's hand, and the chosen card's true owner or resulting adjutant player is
not included.

## Exchange Observation

`EncodedExchangeObservation` uses only Napoleon's exchange-time
`PlayerObservation`, the absolute table player order, and the existing public
`EncodedBiddingHistory`. It contains:

- `relativePlayerIds`: `[5]`, with Napoleon at index `0`
- `contractTargetPointCards`: `12..19`
- `trumpSuitOneHot`: `[4]`
- `calledAdjutantCardMask`: `[53]`
- `selfHandMask`: `[53]`, exactly 13 cards
- `legalDiscardCardMask`: `[53]`, exactly equal to `selfHandMask`
- `handCountByPlayer`: `[5]`
- `specialCardIndices`: public special card indices for the resolved trump
- `biddingHistory`: existing fixed `[117]` public history fields

`actorTarget.discardTargetMask` is a separate `[53]` exactly 3-hot mask. The
three selected cards must be a subset of `legalDiscardCardMask`, and
`discard-cards.cardIds` order is encoded as an unordered set. The observation
does not include other players' hands, complete `ActualCardState`, unrevealed
adjutant ownership, buried-card provenance, or the teacher discard mask.

## Major Shapes

| Field | Shape |
| --- | --- |
| `trumpSuitOneHot` | `[4]` |
| `napoleonPlayerOneHot` | `[5]` |
| `revealedAdjutantPlayerOneHot` | `[6]` |
| `calledAdjutantCardMask` | `[53]` |
| `selfHandMask` | `[53]` |
| `legalAdjutantMask` | `[53]` |
| `legalDiscardCardMask` | `[53]` |
| `discardTargetMask` | `[53]` |
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

## Model Input

`encodePlayingModelInput()` converts an encoded playing observation into the
existing policy `model_input` vector. The output is a `Float32Array` with 6242
features: the 684-feature flat observation first, then the card/player/bidding
index fields one-hot encoded in the same order as the Python
`napoleon_ml.dataset.tensors.MODEL_INPUT_LAYOUT` schema. Empty index slots encode
as all-zero rows.

The non-playing phases expose the same API shape:

| Phase | Encoder | Wrapper | Features | Legal mask returned by wrapper |
| --- | --- | --- | ---: | --- |
| bidding | `encodeBiddingModelInput()` | `createBiddingModelInput()` | 2333 | `legalBidMask` |
| exchange | `encodeExchangeModelInput()` | `createExchangeModelInput()` | 2611 | `legalDiscardCardMask` |
| adjutant | `encodeAdjutantModelInput()` | `createAdjutantModelInput()` | 2553 | `legalAdjutantMask` |

Each phase exports its named layout (`BIDDING_MODEL_INPUT_LAYOUT`,
`EXCHANGE_MODEL_INPUT_LAYOUT`, `ADJUTANT_MODEL_INPUT_LAYOUT`), feature count,
and model input schema version. The layouts mirror Python's
`BIDDING_MODEL_INPUT_LAYOUT`, `EXCHANGE_MODEL_INPUT_LAYOUT`, and
`ADJUTANT_MODEL_INPUT_LAYOUT`, including all-zero rows for absent bidding
history slots. Exchange and adjutant special card indices must be concrete
card indices `0..52`.

The `create*ModelInput()` wrappers return `modelInput` together with the
observation's independent legal mask, so inference callers can pass both to an
ONNX policy and mask illegal logits without re-deriving legal actions. The
model input is built only from the encoded observation; teacher targets,
discard target masks, complete state, hidden hands, and unrevealed adjutant
ownership are not included.

## Validation

The validators check the schema version, fixed lengths, scalar ranges, finite
integer card/player indices, one-hot sums, mask values, contiguous trick slots,
completed-trick counts, latest buried-card event consistency, and that
`legalPlayMask` is a subset of the acting player's `selfHandMask`. Bidding
history validation checks fixed lengths, contiguous action masks, pass/bid/empty
slot consistency, relative player index ranges, suit index ranges, and bid target
ranges before samples are handed to JSONL or downstream training code. Exchange
validation additionally requires Napoleon's self hand to be 13 cards,
`discardCount` to be 3, `legalDiscardCardMask` to equal `selfHandMask`, and
`discardTargetMask` to be an exactly 3-card subset of that legal mask.
