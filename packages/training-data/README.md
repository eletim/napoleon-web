# @napoleon/training-data

This package writes deterministic rule-based self-play datasets for future neural-network training.

Generation is sequential and single-process: seeds are processed in ascending order, one game record is converted to `PlayingTrainingSample` rows, and those rows are streamed to JSONL before the next game starts. It does not implement parallel workers, resume, compression, Python, PyTorch, or training.

## Format

Output is a directory containing `manifest.json` and `shard-00000.jsonl`, `shard-00001.jsonl`, and so on.

- One JSONL line is one `PlayingTrainingSample`.
- JSONL uses UTF-8 and `\n` line endings.
- Games are never split across shards.
- Rows are ordered by seed, then by `decision.step`.
- Existing output directories are rejected.
- Files are first written to a temporary sibling directory and renamed into place only after all validation succeeds.

## Manifest

The manifest records dataset schema versions, the playing encoder schema version, agent metadata, seed ranges, game and sample counts, shard metadata, card ids, and hashes. It does not contain timestamps, elapsed time, process ids, absolute paths, or temporary directory names.

`cardIdsSha256` is the SHA-256 of the UTF-8 bytes of `JSON.stringify(CARD_IDS)`.
Each shard SHA-256 is computed from the exact UTF-8 bytes written to that JSONL file. `byteLength` is accumulated from those same chunks.

Datasets generated with the same generator version and arguments are byte-identical across output directories.
