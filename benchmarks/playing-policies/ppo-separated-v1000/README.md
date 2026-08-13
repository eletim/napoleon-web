# PPO separated v1000 playing policy

This directory contains the repo-managed frozen PPO separated v1000 playing
policy from the completed main run
`ppo-separated-e4-lr1e4-cpp-pool5050-1000`.

Runtime files:

- `policy.onnx` - Actor-only ONNX runtime artifact for playing decisions.
- `policy.json` - Actor ONNX metadata.
- `checkpoint.pt` - PyTorch Actor/Critic checkpoint, including the trained
  Critic for analysis or future migration work.
- `provenance.json` - Source paths, hashes, training conditions, and copied
  artifact hashes.

The ONNX artifact preserves the existing runtime contract: one
`model_input` `[batch, 6246]` float32 input and one `logits` `[batch, 53]`
float32 output. No training or re-evaluation was performed while freezing this
repo artifact; the files were copied from the completed v1000 run and verified.
