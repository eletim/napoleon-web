# @napoleon/self-play-cli

CLI for generating deterministic rule-based self-play datasets.

Run from the repository root after dependencies are installed:

```bash
pnpm self-play:generate -- \
  --start-seed 0 \
  --games 10 \
  --output ./datasets/example \
  --games-per-shard 5
```

No individual package build is required before running this command. The root
`pnpm self-play:generate` script builds `@napoleon/self-play-cli` and its
workspace dependencies before starting generation, so it works from a clean
checkout with no relevant `dist/` directories.

Arguments:

- `--start-seed <uint32>`: first seed to generate.
- `--games <positive integer>`: number of consecutive games.
- `--output <path>`: output dataset directory. It must not already exist.
- `--games-per-shard <positive integer>`: games per JSONL shard, default `100`.
- `--help`: print usage.

Seeds are processed as `startSeed`, `startSeed + 1`, through `startSeed + games - 1`.
The final seed must fit in uint32. Games are never split across shards.

Generated files are written under the specified `--output` directory, such as a
path below `datasets/`. The output directory must not already exist; generation
fails instead of overwriting an existing dataset.

Progress is printed to stderr and does not affect `manifest.json` or JSONL bytes.
On success the CLI prints a short summary to stdout. On input errors it prints a
message to stderr and exits with code `1`.
