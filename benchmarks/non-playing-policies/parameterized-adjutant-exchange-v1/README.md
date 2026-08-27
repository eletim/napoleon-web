# Parameterized adjutant + exchange policy v1 candidate

`policy.json` is the repo-managed, human-readable source-of-truth candidate for
the 95-weight policy selected by Issue #452 and independently verified by Issue
#454. It contains 35 adjutant weights and 60 exchange weights under feature schema
v1. `feature-schema.json` is the exact schema used by the evaluator.

The logical candidate artifact SHA-256 is
`34f90a4f767c678365ad33e19206c088ea1ef359f02c2aa7787da37857f9e59c`.
The unchanged source parameter SHA-256 is
`d364aef0c48a1832bd6602d254d0440f6cb2e2cb50492cfb53934e0378a84d69`.

This directory does not wire the policy into runtime and does not contain an
ONNX or PyTorch conversion. See
`benchmarks/exchange-values/issue454-independent-verification/REPORT.md` for the
10,000-game independent paired verification and adoption decision.
