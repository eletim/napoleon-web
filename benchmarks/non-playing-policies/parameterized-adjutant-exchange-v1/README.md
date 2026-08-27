# Parameterized adjutant + exchange policy v1

`policy.json` is the repo-managed, human-readable source of truth for
the 95-weight policy selected by Issue #452 and independently verified by Issue
#454. It contains 35 adjutant weights and 60 exchange weights under feature schema
v1. `feature-schema.json` is the exact schema used by the evaluator.

The logical candidate artifact SHA-256 is
`a6e97b72160338d3f0ce831f5b1422f86dafb419ff8e458b79741440b2433faa`.
The unchanged source parameter SHA-256 is
`d364aef0c48a1832bd6602d254d0440f6cb2e2cb50492cfb53934e0378a84d69`.

The server phase-policy registry wires this JSON artifact directly into the
adjutant and exchange runtime as one `nonPlaying` policy. Loading fails closed
when its schema, weights, hashes, or provenance are invalid; there is no silent
fallback. This directory intentionally contains no ONNX or PyTorch conversion. See
`benchmarks/exchange-values/issue454-independent-verification/REPORT.md` for the
10,000-game independent paired verification and adoption decision.
