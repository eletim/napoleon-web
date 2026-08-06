# napoleon-ml

A strict Python consumer for the self-play datasets that
`packages/training-data` (TypeScript) generates. It validates a generated
dataset directory end to end — `manifest.json`, every shard's raw bytes,
and every individual sample — and converts validated samples into
fixed-shape, fixed-dtype NumPy tensors.

This package only covers the boundary from a generated dataset directory to
validated NumPy arrays. It does not include PyTorch, TensorFlow, JAX, any
neural-network model, training loop, ONNX export, reinforcement learning, a
parallel `DataLoader`, dataset caching, or compression. See
[Not implemented](#not-implemented) below.

## Requirements

Python 3.11 or newer. The only runtime dependency is `numpy`.

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

### Sample schema v1

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

`iter_samples()` additionally enforces dataset-wide ordering that a single
sample can't check on its own: seeds are non-decreasing and form one
contiguous block per game (a seed reappearing after the dataset has moved
past it — which would indicate a game split across a shard boundary — is
rejected), and `step` strictly increases within a seed's block.

## NumPy tensors

`napoleon_ml.dataset.tensors.tensorize_sample()` converts one validated
`PlayingTrainingSample` into a `TensorizedPlayingSample`:

| Field | Shape | dtype |
| --- | --- | --- |
| `flat_observation` | `(684,)` | `float32` |
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
between cards that isn't real; a future consumer that wants those as model
input should one-hot encode them explicitly instead.

`validate_tensorized_sample()` (run automatically by `tensorize_sample()`)
checks that `flat_observation` is one-dimensional, `float32`,
C-contiguous, and free of `NaN`/`Infinity`; that `actor_target` is `int64`
and in `[0, 52]`; that `legal_play_mask` has shape `(53,)` and is `1` at
the `actor_target` index; and that `belief_target` has shape `(53,)`,
dtype `int64`, and values in `[0, 5]`.

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

PyTorch, TensorFlow, JAX, any neural-network model or training loop,
behavior cloning, actor-critic or other reinforcement learning, ONNX
export, GPU code, checkpoints, TensorBoard, a parallel `DataLoader`,
dataset caching, gzip/compression, and a database or web UI. This package
stops at validated NumPy arrays.
