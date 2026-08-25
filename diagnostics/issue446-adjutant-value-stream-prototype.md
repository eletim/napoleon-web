# Issue #446 adjutant value stream prototype report

## Status

This branch implements the scalable compact396 proposal-scoring bridge and compact290
adjutant-value trainer needed for Issue #446, but the requested 10,000-state training
run and 200-state full-gold holdout are not marked complete from this environment.

Reason: the current C++ rollout path used by the stream generator still uses the
RuleBased CPU game policy path for bidding/playing decisions. Issue #446 fixes
bidding to `frozen-raise-v1` and playing to `ppo-separated-v1000`; wiring those ONNX
policies into the terminal rollout path must be completed before the 10k/200 results
should be treated as the requested production run.

## Implemented

- C++ binary stream generator:
  - `napoleon_adjutant_value_stream_cli`
  - mode `proposal`: compact396-scored proposal teacher labels
  - mode `full-gold`: full `53 x 286` gold labels plus proposal containment metrics
- Python compact396 scorer driver:
  - `napoleon-generate-adjutant-value-dataset`
  - loads `benchmarks/exchange-values/exchange-compact396-value-v1/checkpoint.pt`
  - receives one source state's `53 x 286 x 396` compact inputs from C++
  - returns compact396 top-k discard indices
- Python compact290 adjutant value trainer:
  - `napoleon-train-adjutant-value-mlp`
  - default `[512,512,256,256] -> scalar contractMargin`
  - Huber regression
  - state-level split
  - full-gold evaluation path

## Smoke Runs

Proposal teacher smoke:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --project python --extra train python \
  -m napoleon_ml.cli.generate_adjutant_value_dataset \
  --cpp-cli /tmp/napoleon-issue446-build/napoleon_adjutant_value_stream_cli \
  --exchange-checkpoint benchmarks/exchange-values/exchange-compact396-value-v1/checkpoint.pt \
  --output-directory /tmp/issue446-smoke-proposal5 \
  --mode proposal \
  --states 5 \
  --max-deal-attempts 125 \
  --start-seed 446000000 \
  --agent-seed 446 \
  --device cpu \
  --score-batch-size 4096
```

Result:

- source states: 5
- samples: 265
- terminal rollouts: 6,791

Full-gold smoke:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --project python --extra train python \
  -m napoleon_ml.cli.generate_adjutant_value_dataset \
  --cpp-cli /tmp/napoleon-issue446-build/napoleon_adjutant_value_stream_cli \
  --exchange-checkpoint benchmarks/exchange-values/exchange-compact396-value-v1/checkpoint.pt \
  --output-directory /tmp/issue446-smoke-gold2 \
  --mode full-gold \
  --states 2 \
  --max-deal-attempts 50 \
  --start-seed 446100000 \
  --agent-seed 446 \
  --device cpu \
  --score-batch-size 4096
```

Result:

- source states: 2
- samples: 106
- terminal rollouts: 33,019

Training smoke:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --project python --extra train python \
  -m napoleon_ml.cli.train_adjutant_value_mlp \
  /tmp/issue446-smoke-proposal5 \
  --output-directory /tmp/issue446-smoke-adjutant-value \
  --full-gold-directory /tmp/issue446-smoke-gold2 \
  --epochs 3 \
  --batch-size 64 \
  --hidden-dims 32,16 \
  --patience 3 \
  --device cpu
```

Result:

- checkpoint: `/tmp/issue446-smoke-adjutant-value/checkpoint.pt`
- full-gold model-selected regret mean: 2.0 on 2 smoke states
- full-gold proposal-best regret mean: 1.716981 on 106 candidates
- full-gold top-k containment:
  - K=4: 0.000000
  - K=8: 0.009434
  - K=16: 0.028302
  - K=32: 0.141509
  - K=64: 0.292453

The smoke numbers are not quality claims; they only validate the data path.

## Compute Envelope

The requested production run is approximately:

- main proposal dataset: `10,000 x 53 x (top16 + RB + 8 diversity)` terminal rollouts
- full gold holdout: `200 x 53 x 286` terminal rollouts
- compact396 scoring: `10,200 x 53 x 286 = 154,611,600` MLP candidate scores

CUDA is available only outside the sandbox on this host:

- `torch.cuda.is_available() == true`
- GPU: `NVIDIA GeForce RTX 4060`

The stream bridge is designed to use that GPU for compact396 scoring while keeping
terminal rollouts in C++.

## Remaining Blockers Before Ready

1. Wire the stream generator's rollout policies to the fixed Issue #446 policies:
   - bidding: `frozen-raise-v1`
   - playing: `ppo-separated-v1000`
2. Run the actual `10,000` accepted-state proposal dataset.
3. Run the fully separated `200` accepted-state full-gold holdout, or at least the
   issue-allowed first `50` states with measured expansion plan.
4. Train the default `[512,512,256,256]` compact290 model on the 10k dataset.
5. Report production metrics and only then decide whether 50k/100k scale-up is valid.

## Verification

```bash
cmake -S packages/cpp-core/native -B /tmp/napoleon-issue446-build
cmake --build /tmp/napoleon-issue446-build --target napoleon_adjutant_value_stream_cli napoleon_core_self_test
/tmp/napoleon-issue446-build/napoleon_core_self_test
UV_CACHE_DIR=/tmp/uv-cache uv run --project python --extra train python -m compileall \
  python/src/napoleon_ml/cli/generate_adjutant_value_dataset.py \
  python/src/napoleon_ml/cli/train_adjutant_value_mlp.py \
  python/src/napoleon_ml/adjutant_value
```
