# RL v740 playing benchmark policy

This directory contains the repo-managed frozen RL v740 playing policy used by standard benchmark evaluation.

The source policy was a schema v1 / 6242-feature artifact from `rl-v100-to-v1000` iteration 740. It was migrated to the current schema v2 / 6246-feature runtime by copying old input weights and zero-initializing the appended `selfRoleOneHot` columns.

Runtime files:

- `policy.onnx`
- `policy.json`
- `provenance.json`

See `provenance.json` for source hashes, committed artifact hashes, and logit parity results.
