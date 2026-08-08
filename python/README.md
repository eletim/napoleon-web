# napoleon-ml

A strict Python consumer for the self-play datasets that
`packages/training-data` (TypeScript) generates. It validates a generated
dataset directory end to end — `manifest.json`, every shard's raw bytes,
and every individual sample — and converts validated samples into
fixed-shape, fixed-dtype NumPy tensors. It also includes first CPU-only
PyTorch MLP baselines for predicting hidden card ownership and selecting a
legal play from `model_input`.

This package covers the boundary from a generated dataset directory to
validated NumPy arrays, an optional PyTorch `IterableDataset` adapter for
fixed-shape training batches, and the first supervised ownership-belief and
legal-play policy MLP baselines. It does not include TensorFlow, JAX, ONNX export, reinforcement
learning, a parallel `DataLoader`, dataset caching, or compression. See
[Not implemented](#not-implemented) below.

## Requirements

Python 3.11 or newer. The only required runtime dependency is `numpy`.
PyTorch is available through the `train` extra. ONNX export and parity checks
are available through the `export` extra.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "./python[dev]"
```

(Run the `pip install` from the repository root, as shown; if your shell is
already inside `python/`, use `-e ".[dev]"` instead.)

Development extras (`pytest`, `mypy`, `ruff`) are pulled in by `[dev]`.
For a runtime install that includes only the training adapter dependency:

```bash
python -m pip install -e "./python[train]"
```

For ONNX export and ONNX Runtime parity checks:

```bash
python -m pip install -e "./python[export]"
```

## Generating a dataset

Datasets are produced by the TypeScript CLI, from the repository root:

```bash
pnpm self-play:generate -- \
  --start-seed 0 \
  --games 100 \
  --output ./datasets/rule-based-v1 \
  --games-per-shard 10
```

See `apps/self-play-cli/README.md` and `packages/training-data/README.md`
for the generator itself; this package only reads what it produces.

## Loading a dataset

```python
from napoleon_ml.dataset import load_manifest, iter_samples, iter_tensorized_samples

manifest = load_manifest("./datasets/rule-based-v1")

for sample in iter_samples("./datasets/rule-based-v1"):
    ...  # a fully validated napoleon_ml.dataset.PlayingTrainingSample

for tensorized in iter_tensorized_samples("./datasets/rule-based-v1"):
    ...  # a napoleon_ml.dataset.TensorizedPlayingSample of NumPy arrays
```

Nothing here loads the whole dataset into memory: `iter_raw_samples`,
`iter_samples`, and `iter_tensorized_samples` are generators that read one
JSONL line at a time. Memory use does not grow with dataset size.

## What gets validated, and why

### Manifest validation

`load_manifest()` parses `manifest.json` with no implicit type coercion (a
JSON string `"1"` is never accepted where an integer is required, and a JSON
boolean is never accepted where an integer is required — Python's `bool` is
an `int` subclass, so a naive `isinstance(value, int)` check would silently
accept `True`/`False`; this package checks for that explicitly). It then
checks every invariant the TypeScript generator itself enforces
(`packages/training-data/src/validation.ts`): schema/generator/encoder
version identity, `startSeed`/`endSeed`/`gameCount` consistency, shard count
and shard file naming (`shard-00000.jsonl`, five-digit, in index order, no
gaps or overlaps in seed ranges), and the fixed `playerCount`/`cardCount`
values.

### `CARD_IDS` compatibility

`napoleon_ml.dataset.constants.EXPECTED_CARD_IDS` is a literal, hand-copied
transcription of `packages/ai-observation/src/cardIndex.ts`'s `CARD_IDS`
(53 entries, suit-major then rank order, `joker` last) — not re-derived by
guessing an ordering. `load_manifest()` checks that `manifest.cardIds`
matches this tuple exactly, in order, and that
`manifest.cardIdsSha256` is both a well-formed SHA-256 hex string and equal
to a value this package recomputes itself. That recomputation
(`validate.calculate_card_ids_sha256()`) must byte-for-byte match
`packages/training-data/src/serialization.ts`'s
`sha256Utf8(JSON.stringify(CARD_IDS))`: `JSON.stringify` uses compact
`,`/`:` separators with no extra whitespace, which is *not* what Python's
`json.dumps(..., default separators)` produces, so this package calls
`json.dumps(..., separators=(",", ":"))` explicitly to match.

### Shard integrity (SHA-256, byte length, streaming, binary mode)

Every shard file is opened with `Path.open("rb")` — binary mode, not text
mode. This matters because SHA-256 and byte length are properties of the
*exact bytes* the generator wrote, and text-mode I/O in Python normalizes
line endings and re-encodes text, which would silently change what gets
hashed and counted. Each raw line's bytes are fed into a running
`hashlib.sha256()` and a running byte counter before anything else happens
to them; only after that does the line get decoded and parsed.

Per-shard verification is split into two independent groups, checked
immediately after the shard's last line — not deferred to the end of the
whole dataset, and never silently skipped for a line that failed to parse:

- **Byte-identity checks** — SHA-256 and byte length. These are the only
  checks `verify_integrity=False` (the reader's `iter_raw_samples()` /
  `iter_samples()` / `iter_tensorized_samples()`) or the CLI's
  `--no-integrity-check` skip.
- **Structural checks** — line count, sample count, first sample seed, last
  sample seed, and game count. These always run, regardless of
  `verify_integrity`, so a shard with a deleted or duplicated line, a
  truncated file, or a seed range that doesn't match the manifest is
  rejected even when byte-identity verification is skipped.

Game count is computed by walking the shard's samples in order and
incrementing a counter each time the seed differs from the *immediately
preceding* sample's seed — not by collecting seeds into a `set` and taking
its size. A `set` would silently accept a shard whose seed sequence is
`0, 1, 0` (game 0's samples split around game 1's, i.e. seed 0 reappearing
after the dataset moved past it) as "2 unique seeds", the same count a
well-formed `0, 0, 1, 1` shard would produce. Counting consecutive-run
transitions instead gives that malformed shard a game count of 3, which
fails the comparison against `manifest.json`'s recorded `gameCount` for the
shard and is rejected.

A line is rejected (not silently dropped) if it: is empty; uses CRLF line
endings or contains a stray `\r`; is missing its trailing `\n` on the final
line of the file; is not valid UTF-8; is not valid JSON; is a JSON value
other than an object; contains a duplicate object key (Python's
`json.loads()` normally keeps the *last* value for a duplicate key — this
package uses `object_pairs_hook` to reject it instead); or contains a
literal `NaN`, `Infinity`, or `-Infinity` token (accepted by
`json.loads()` by default; rejected here via `parse_constant`).

### Sample schemas

`napoleon_ml.dataset.sample` defines typed, frozen dataclasses that mirror
`packages/ai-observation/src/createPlayingTrainingSample.ts`'s
`PlayingTrainingSample` (and its nested `EncodedPlayingObservation`,
`EncodedBiddingHistory`, `EncodedPlayAction`, `EncodedBeliefTarget`) field
for field, with the same fixed array lengths (`CARD_COUNT` = 53,
`PLAYER_COUNT` = 5, `MAX_BIDDING_ACTION_COUNT` = 117, and so on — see
`napoleon_ml.dataset.constants`). Parsing rejects an unknown JSON key, a
missing required key, and any value whose JSON type doesn't match the
schema. `napoleon_ml.dataset.validation.validate_sample()` then checks the
same numeric ranges, one-hot/mask invariants, and cross-field rules as
`packages/ai-observation`'s TypeScript validators — for example, that
`legalPlayMask` is a subset of `selfHandMask`, that the selected card index
is legal, and that every bidding-history slot's fields are consistent with
its action type.

The reader supports both TypeScript dataset manifest shapes:

| Manifest | Row sample type | Python sample dataclass |
| --- | --- | --- |
| v1 (`datasetSchemaVersion = 1`) | `playing-training-sample` | `PlayingTrainingSample` |
| v2 (`datasetSchemaVersion = 2`) | `bidding-training-sample` | `BiddingTrainingSample` |
| v2 (`datasetSchemaVersion = 2`) | `exchange-training-sample` | `ExchangeTrainingSample` |
| v2 (`datasetSchemaVersion = 2`) | `adjutant-training-sample` | `AdjutantTrainingSample` |

Manifest `sampleType` selects the parser, validator, and tensorizer. A v2
row whose `sampleType` does not match the manifest is rejected. Unknown keys,
missing keys, wrong fixed lengths, wrong mask values, illegal targets, and
schema-version drift are rejected before tensorization.

`iter_samples()` additionally enforces dataset-wide ordering that a single
sample can't check on its own: seeds are non-decreasing and form one
contiguous block per game (a seed reappearing after the dataset has moved
past it — which would indicate a game split across a shard boundary — is
rejected), and `step` strictly increases within a seed's block.

## NumPy tensors

`napoleon_ml.dataset.tensors.tensorize_sample()` converts one validated
sample into the matching tensorized dataclass. The four fixed `model_input`
lengths are:

| Sample type | Tensorized dataclass | `model_input` shape |
| --- | --- | --- |
| `playing-training-sample` | `TensorizedPlayingSample` | `(6242,)` |
| `bidding-training-sample` | `TensorizedBiddingSample` | `(2333,)` |
| `exchange-training-sample` | `TensorizedExchangeSample` | `(2611,)` |
| `adjutant-training-sample` | `TensorizedAdjutantSample` | `(2553,)` |

The PyTorch DataLoader supports all four sample types. The policy model and
ONNX export remain playing-only in this version.

For playing, `tensorize_sample()` returns a `TensorizedPlayingSample`:

| Field | Shape | dtype |
| --- | --- | --- |
| `flat_observation` | `(684,)` | `float32` |
| `model_input` | `(6242,)` | `float32` |
| `legal_play_mask` | `(53,)` | `uint8` |
| `actor_target` | scalar | `int64` |
| `belief_target` | `(53,)` | `int64` |
| `belief_hidden_ownership_loss_mask` | `(53,)` | `uint8` |

`TensorizedPlayingSample.observation` (a `PlayingObservationTensors`) holds
every field of the observation individually: binary masks as `uint8`,
one-hot vectors and small scalars as `float32`, and card/player/trick
*index* fields (`currentTrickCardIndices`, the bidding-history index
arrays, `specialCardIndices`, and so on) as `int64` — deliberately **not**
folded into `flat_observation`. A raw category index (e.g. "card 37") cast
straight into a float vector would silently imply an ordinal relationship
between cards that isn't real; `model_input` (below) is where those are
one-hot encoded instead.

`validate_tensorized_sample()` (run automatically by `tensorize_sample()`)
checks that `flat_observation` is one-dimensional, `float32`,
C-contiguous, and free of `NaN`/`Infinity`; that `actor_target` is `int64`
and in `[0, 52]`; that `legal_play_mask` has shape `(53,)` and is `1` at
the `actor_target` index; that `belief_target` has shape `(53,)`, dtype
`int64`, and values in `[0, 5]`; and that `model_input` (below) meets its
own shape/dtype/value invariants.

### Flat feature layout

`flat_observation`'s field order is fixed and explicit —
`napoleon_ml.dataset.tensors.FLAT_OBSERVATION_LAYOUT` is a tuple of
`FeatureSlice(name, start, stop, shape, dtype)` covering every one of its
684 positions with no gap and no overlap (checked at import time). To
inspect it:

```python
from napoleon_ml.dataset.tensors import FLAT_OBSERVATION_LAYOUT

for feature in FLAT_OBSERVATION_LAYOUT:
    print(feature.name, feature.start, feature.stop, feature.shape)
```

### Model input (`model_input`, schema version `MODEL_INPUT_SCHEMA_VERSION` = 1)

`flat_observation` deliberately leaves out every category-*index* field, so
it alone is not a complete model input — a policy that also needs to know
which cards are in the current trick, say, gets nothing from it. `model_input`
is: `flat_observation`'s 684 positions, followed by every one of those index
fields, one-hot encoded, in a second fixed-order block. It is what a
first-version MLP policy should actually train and infer on; `flat_observation`
is kept only for existing consumers that already depend on it.

The appended block, in order, one-hot encodes:

| Field | Shape | Classes |
| --- | --- | --- |
| `specialCardIndicesOneHot` (oruma, yoromeki, seiJack, uraJack) | `(4, 53)` | card id, 0–52 |
| `currentTrickCardIndicesOneHot` | `(5, 53)` | card id, 0–52 |
| `completedTrickCardIndicesOneHot` | `(50, 53)` | card id, 0–52 |
| `currentTrickPlayerIndicesOneHot` | `(5, 5)` | player, 0–4 |
| `completedTrickPlayerIndicesOneHot` | `(50, 5)` | player, 0–4 |
| `completedTrickWinnerIndicesOneHot` | `(10, 5)` | player, 0–4 |
| `biddingHistoryActionTypeIndicesOneHot` | `(117, 2)` | pass=0 / bid=1 |
| `biddingHistoryPlayerIndicesOneHot` | `(117, 5)` | player, 0–4 |
| `biddingHistorySuitIndicesOneHot` | `(117, 4)` | suit, 0–3 |
| `biddingHistoryTargetPointCardsOneHot` | `(117, 7)` | target point cards, 13–19 |

684 + (4 + 5 + 50) × 53 + (5 + 50 + 10) × 5 + 117 × (2 + 5 + 4 + 7) = 6242.

An empty slot one-hot-encodes to an all-zero region, never to a one-hot at
some placeholder class — so "no value" can never be mistaken for a real
class 0. This covers both kinds of "empty" this schema uses: the `-1`
sentinel (card/player index fields — see `EMPTY_CARD_INDEX`/
`EMPTY_PLAYER_INDEX`/`EMPTY_BIDDING_ACTION_TYPE`/`EMPTY_BIDDING_SUIT_INDEX`
in `napoleon_ml.dataset.constants`) and, for
`biddingHistoryTargetPointCardsOneHot` specifically, a `pass` or empty
slot's `targetPointCards` of `0` (which is outside the valid `13`–`19`
bid-target range and so is never one-hot encoded, whether or not it's `-1`).
Callers never need to branch on which "empty" applies — one-hot encoding
already reduces it to "no bit set".

`legalPlayMask` is included in `model_input` (inside `flat_observation`'s
684 positions, unchanged) and is *also* kept as the independent
`TensorizedPlayingSample.legal_play_mask` array, so inference code can mask
illegal actions out of a policy's output without re-deriving the mask from
`model_input`.

Deliberately excluded from `model_input` (and from `flat_observation`):
`schemaVersion`, `seed`, `step`, and the player-id strings (dataset
bookkeeping, not part of the game state a policy conditions on), and the
sample's `actorTarget` and `beliefTarget` (the training label and the
hidden-ownership ground truth — including a real player's hand would leak
information no player can actually observe).

`napoleon_ml.dataset.tensors.MODEL_INPUT_LAYOUT` is
`FLAT_OBSERVATION_LAYOUT` followed by the ten slices above (also available
alone as `MODEL_INPUT_ONEHOT_LAYOUT`), covering all 6242 positions of
`model_input` with no gap, overlap, or duplicate name (checked at import
time, same as `FLAT_OBSERVATION_LAYOUT`):

```python
from napoleon_ml.dataset.tensors import MODEL_INPUT_LAYOUT

for feature in MODEL_INPUT_LAYOUT:
    print(feature.name, feature.start, feature.stop, feature.shape)
```

The non-playing phases expose equivalent named layouts:

| Constant | Feature count | Notes |
| --- | --- | --- |
| `BIDDING_MODEL_INPUT_LAYOUT` | 2333 | self hand, legal bid mask, bidding state, shared bidding history |
| `EXCHANGE_MODEL_INPUT_LAYOUT` | 2611 | includes `handCountByPlayer (5,)` and excludes discard target |
| `ADJUTANT_MODEL_INPUT_LAYOUT` | 2553 | adjutant legal mask plus shared bidding history |

All layout slices are explicit `FeatureSlice` entries. The tensorizer does
not recursively flatten arbitrary JSON. Training labels (`actorTarget` and
`discardTargetMask`), hidden ownership, other players' private hands,
complete `actualState` information, buried-card provenance, and unrevealed
adjutant ownership are deliberately excluded from every `model_input`.

## Train/validation/test splits

`napoleon_ml.dataset.split.split_for_seed(seed)` assigns each *game* (not
each sample) to a split, deterministically, by `seed % 100`:

- `0 <= seed % 100 < 80` → `train`
- `80 <= seed % 100 < 90` → `validation`
- otherwise → `test`

(Configurable via `SplitConfig(train=..., validation=..., test=...)`, whose
three values must be non-negative integers summing to 100.)

**Splitting is per game, never per sample.** A single game contributes
around 50 sequential decisions, all describing the same hidden game state
from the same player's perspective at consecutive points in time. Assigning
those ~50 samples to splits independently (e.g. by hashing each sample) would
let a model see most of a game during training and be "tested" on the
few decisions that happened to land in the validation split — those
decisions are not independent of what the model already learned about that
exact game, so the validation score would be optimistic in a way that
doesn't reflect performance on a truly unseen game. Keying the split on
`seed` alone, and only on `seed`, keeps every decision from one game on the
same side of the split.

`split_for_seed` never uses Python's built-in `hash()` (which is randomly
salted per process for strings — not for ints, but this package doesn't
rely on that detail either) or any other source of run-to-run
non-determinism: the same `(seed, config)` always returns the same split,
in any process, on any machine.

## PyTorch DataLoader

Install the training extra, then create a split-filtered loader. For
`playing-training-sample` datasets, the existing playing-specific API remains:

```python
from napoleon_ml.dataset import DatasetSplit, SplitConfig
from napoleon_ml.dataset.pytorch import create_playing_dataloader

loader = create_playing_dataloader(
    "./datasets/rule-based-v1",
    split=DatasetSplit.TRAIN,
    split_config=SplitConfig(train=80, validation=10, test=10),
    batch_size=32,
)

for batch in loader:
    model_input = batch["model_input"]
    actor_target = batch["actor_target"]
    legal_play_mask = batch["legal_play_mask"]
    belief_target = batch["belief_target"]
    mask = batch["belief_hidden_ownership_loss_mask"]
    seed = batch["seed"]
    step = batch["step"]
```

Each batch is a dictionary of PyTorch tensors with fixed shapes and dtypes:

| Field | Batch shape | dtype |
| --- | --- | --- |
| `model_input` | `(batch, 6242)` | `torch.float32` |
| `actor_target` | `(batch,)` | `torch.int64` |
| `legal_play_mask` | `(batch, 53)` | `torch.bool` |
| `belief_target` | `(batch, 53)` | `torch.int64` |
| `belief_hidden_ownership_loss_mask` | `(batch, 53)` | `torch.bool` by default (`torch.uint8` optional) |
| `seed` | `(batch,)` | `torch.int64` |
| `step` | `(batch,)` | `torch.int64` |

`actor_target` is the supervised play label, and every row is checked while
streaming so `legal_play_mask[row, actor_target[row]]` is true. These policy
fields are exposed alongside the existing ownership-belief fields; they do
not change the dataset, observation, or `model_input` schema.

For manifest-selected loading, use `create_training_dataloader()`. It reads
`manifest.sampleType` and returns the matching batch shape; phase-specific
helpers (`create_bidding_dataloader`, `create_exchange_dataloader`,
`create_adjutant_dataloader`) reject datasets whose manifest sample type does
not match.

| Sample type | Batch fields |
| --- | --- |
| `bidding-training-sample` | `model_input (batch, 2333) float32`, `legal_bid_mask (batch, 29) bool`, `actor_target (batch,) int64` |
| `exchange-training-sample` | `model_input (batch, 2611) float32`, `legal_discard_card_mask (batch, 53) bool`, `discard_target_mask (batch, 53) bool` |
| `adjutant-training-sample` | `model_input (batch, 2553) float32`, `legal_adjutant_mask (batch, 53) bool`, `actor_target (batch,) int64` |

`mask_dtype` is only a playing-dataset compatibility option for
`belief_hidden_ownership_loss_mask`. Non-playing DataLoader masks are always
`torch.bool`, and `create_training_dataloader()` rejects non-playing
`mask_dtype` overrides.

All DataLoader variants stream through `iter_tensorized_samples()` and filter
by `split_for_seed()`, so they do not load the whole dataset into memory. The
same dataset, split, and split config produce the same sample order each
time, and the dataset can be iterated again for a new epoch because each
`__iter__()` call reopens the shard stream from the beginning. An empty split
simply yields no batches. This first version only supports `num_workers=0`;
all DataLoader factories reject `num_workers=1`, and direct worker-process
iteration raises `DatasetError`.

## Inspecting a dataset

```bash
python -m napoleon_ml.cli.inspect_dataset ./datasets/rule-based-v1
```

or, after the editable install above, the console script:

```bash
napoleon-inspect-dataset ./datasets/rule-based-v1
```

Flags:

- `--no-integrity-check` — skip shard SHA-256/byte-length re-verification
  only. Structural checks (line count, sample count, seed range, game
  count) and semantic sample validation still run regardless. Full
  verification runs unless this flag is passed explicitly.
- `--json` — print a machine-readable JSON report to stdout instead of the
  human-readable text report. Progress and error messages always go to
  stderr, so `--json` output is safe to pipe.

The command validates the entire dataset while streaming it and prints a
summary: dataset/schema metadata, integrity status, per-split game/sample
counts, an actor legal-action tally with the top selected cards, a belief
owner-class histogram, and the tensor shapes/dtypes described above. Only
small fixed-size counters and histograms are accumulated in memory (a
53-slot card histogram, a 6-slot belief histogram, three per-split
counters) — the dataset itself is never buffered. Before reporting success,
the CLI also cross-checks the actual number of samples it streamed against
`manifest.json`'s `sampleCount` on its own, independently of the reader's
per-shard checks. Exit code is `0` on success; `1` if the manifest, a
shard, or a sample fails validation, if the streamed sample count doesn't
match the manifest, or if the dataset directory doesn't exist (the specific
problem is always printed to stderr, without a Python stack trace). An
unexpected internal error (a bug, not a dataset problem) is allowed to
raise normally with its full traceback rather than being swallowed.

## Ownership MLP baseline

Install the `train` or `dev` extra, then train the CPU-only baseline:

```bash
napoleon-train-ownership-mlp ./datasets/rule-based-v1 \
  --output ./models/ownership-mlp.pt \
  --epochs 3 \
  --batch-size 32 \
  --seed 0
```

The model consumes only `model_input` and emits logits with shape
`(batch, 53, 6)`: one owner class for each card (`relative_player_0` through
`relative_player_4`, plus `not_in_hand`). Training computes cross entropy
only where `belief_hidden_ownership_loss_mask` is true, so known cards are
excluded from the loss and the primary masked accuracy. The output report
also includes owner-class accuracy, game-progress accuracy by step bucket,
and a comparison against an untrained baseline that always predicts
`not_in_hand`.

Evaluate a saved checkpoint without training:

```bash
napoleon-evaluate-ownership-mlp ./datasets/rule-based-v1 \
  --checkpoint ./models/ownership-mlp.pt \
  --split test
```

Checkpoints store the model state, model/training settings, dataset schema
version, playing encoder schema version, model input schema version, and the
`CARD_IDS` SHA-256 hash. Loading refuses a checkpoint whose saved schema
metadata or card-id hash does not match the current package and dataset.
For reproducibility, `--seed` fixes model initialization and the DataLoader
keeps the deterministic shard/seed sample order; no shuffle option is
exposed.

## Policy MLP baseline

Install the `train` or `dev` extra, then train the CPU-only behavior-cloning
baseline on the rule-based agent's selected plays:

```bash
napoleon-train-policy-mlp ./datasets/rule-based-v1 \
  --output ./models/policy-mlp.pt \
  --epochs 3 \
  --batch-size 32 \
  --seed 0
```

The model consumes only `model_input` and emits logits with shape
`(batch, 53)`: one logit per card id. Training masks illegal card logits
with `legal_play_mask` before cross entropy, and evaluation and inference
use the same mask before top-1 selection, so an illegal card is never
available as a prediction. `napoleon_ml.policy.select_policy_action()` is
the inference helper for turning logits plus a legal mask into selected
card indices.

Evaluate a saved checkpoint without training:

```bash
napoleon-evaluate-policy-mlp ./datasets/rule-based-v1 \
  --checkpoint ./models/policy-mlp.pt \
  --split test
```

Export a saved checkpoint for TypeScript-side inference:

```bash
napoleon-export-policy-onnx ./datasets/rule-based-v1 \
  --checkpoint ./models/policy-mlp.pt \
  --output ./models/policy-mlp.onnx \
  --metadata-output ./models/policy-mlp.json
```

The ONNX model has one input named `model_input` with shape `(batch, 6242)`
and dtype `float32`, and one output named `logits` with shape `(batch, 53)`
and dtype `float32`; the batch dimension is dynamic. The JSON metadata records
the dataset schema version, playing encoder schema version, model input schema
version, `CARD_IDS` SHA-256 hash, ONNX input/output names, shapes, dtypes, ONNX
opset, and the policy model config.

Before writing a usable export, the command refuses a checkpoint whose saved
schema metadata, `CARD_IDS` hash, model input size, state dict, or output logit
shape is inconsistent with the current package and dataset. It then runs a
CPU ONNX Runtime smoke check on the first validated dataset sample and verifies
that PyTorch and ONNX Runtime logits match within tolerance and that
`legal_play_mask`-masked top-1 selection picks the same card.

The report includes top-1 accuracy and counts for all positions, forced
positions with exactly one legal card, and non-forced positions with
multiple legal cards. It also reports the expected accuracy of a legal-card
uniform random baseline over the same buckets. Checkpoints store the model
state, model/training settings, dataset schema version, playing encoder
schema version, model input schema version, and the `CARD_IDS` SHA-256 hash.
Loading refuses a checkpoint whose saved schema metadata or card-id hash
does not match the current package and dataset. For reproducibility,
`--seed` fixes model initialization and the DataLoader keeps the
deterministic shard/seed sample order; no shuffle option is exposed.

## Tests

```bash
# Fast unit tests only (no subprocess, no Node/pnpm required):
python -m pytest -m "not integration"

# Everything, including the TypeScript cross-language integration test:
python -m pytest
```

Unit tests (`tests/unit/`) build small fixtures in-process or in a pytest
`tmp_path` — no TypeScript-generated JSONL is committed to this repository.

## Cross-language integration test

`tests/integration/test_typescript_dataset.py` (marked
`@pytest.mark.integration`) runs `pnpm self-play:generate` via
`subprocess.run()` (argument list, no shell, `cwd` set to the repository
root) to generate a real 3-game, 2-shard dataset into a temporary
directory, loads it with this package, and checks the shard count, game
count, sample count, seed range, per-game sample count, shard hashes and
byte lengths, `cardIds`/`cardIdsSha256`, that every sample and every
tensorization succeeds, that every tensor shape and dtype is consistent
across all 150 samples, that every actor target is legal, and that seeds
0–2 are all assigned to the `train` split. It requires `pnpm` (and Node) to
be available and does not skip quietly if they are missing; the temporary
directory is removed after the test regardless of outcome, and no dataset
output is committed.

## Not implemented

TensorFlow, JAX, behavior cloning, actor-critic or other reinforcement
learning, GPU code, TensorBoard, shuffle, a parallel
`DataLoader`, dataset caching, gzip/compression, and a database or web UI.
