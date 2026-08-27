"""Merge detailed exchange audit shards and bind them to the fixed #446 holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

ADJUTANT_COUNT = 53
DISCARD_COUNT = 286
AUDIT_FILES = {
    "stateFeatures": "exchange-state-features.f32",
    "candidateMask": "exchange-candidate-mask.u8",
    "contractMargin": "exchange-contract-margin.f32",
    "relativeReward": "exchange-relative-reward.f32",
    "ruleBasedCandidate": "exchange-rule-based-candidate.u32",
    "goldCandidate": "exchange-gold-candidate.u32",
}
FIXED_COMPARISON_FILES = (
    "features.f32",
    "contract-margin.f32",
    "relative-reward.f32",
    "candidate-card.u8",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-holdout", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--replacement-state",
        action="append",
        default=[],
        metavar="INDEX=DATASET",
        help="replace one regenerated state with a one-state dataset",
    )
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()
    fixed_manifest_path = args.fixed_holdout / "manifest.json"
    fixed_manifest_bytes = fixed_manifest_path.read_bytes()
    fixed_manifest = json.loads(fixed_manifest_bytes)
    if fixed_manifest.get("mode") != "full-gold" or int(fixed_manifest["sourceStateCount"]) != 200:
        raise ValueError("fixed holdout must be the #446 200-state full-gold artifact.")
    expected_diagnostics = fixed_manifest["sourceDiagnostics"]

    shard_manifests: list[dict[str, object]] = []
    state_sources: list[tuple[Path, int]] = []
    replacement_records: list[dict[str, object]] = []
    actual_diagnostics: list[dict[str, object]] = []
    for shard in args.shards:
        manifest = json.loads((shard / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("mode") != "full-gold" or not manifest.get("exchangeAudit", {}).get(
            "enabled"
        ):
            raise ValueError(f"{shard}: detailed full-gold exchange audit is not enabled.")
        shard_manifests.append(manifest)
        for local_index, diagnostic in enumerate(manifest["sourceDiagnostics"]):
            row = dict(diagnostic)
            row["shardSourceIndex"] = int(diagnostic["sourceIndex"])
            row["sourceIndex"] = len(actual_diagnostics)
            actual_diagnostics.append(row)
            state_sources.append((shard, local_index))
    for specification in args.replacement_state:
        raw_index, separator, raw_path = specification.partition("=")
        if not separator:
            raise ValueError("--replacement-state must be INDEX=DATASET.")
        index = int(raw_index)
        replacement = Path(raw_path)
        manifest = json.loads((replacement / "manifest.json").read_text(encoding="utf-8"))
        if int(manifest.get("sourceStateCount", 0)) != 1:
            raise ValueError(f"{replacement}: replacement dataset must contain one state.")
        diagnostic = dict(manifest["sourceDiagnostics"][0])
        diagnostic["shardSourceIndex"] = int(actual_diagnostics[index]["shardSourceIndex"])
        diagnostic["sourceIndex"] = index
        actual_diagnostics[index] = diagnostic
        state_sources[index] = (replacement, 0)
        replacement_records.append(
            {
                "sourceIndex": index,
                "path": str(replacement),
                "manifestSha256": _sha256(replacement / "manifest.json"),
            }
        )
    if len(actual_diagnostics) != len(expected_diagnostics):
        raise ValueError("regenerated source-state count differs from fixed holdout.")
    for index, (expected, actual) in enumerate(
        zip(expected_diagnostics, actual_diagnostics, strict=True)
    ):
        for key in (
            "seed",
            "napoleonSeatIndex",
            "contractSuit",
            "contractTarget",
            "ruleBasedAdjutantIndex",
        ):
            if expected[key] != actual[key]:
                raise ValueError(f"fixed holdout source {index} differs at {key}.")

    for file_name in FIXED_COMPARISON_FILES:
        expected_sha = _sha256(args.fixed_holdout / file_name)
        actual_sha = _statewise_sha256(state_sources, file_name)
        if expected_sha != actual_sha:
            raise ValueError(
                f"regenerated {file_name} differs from fixed holdout: "
                f"{actual_sha} != {expected_sha}."
            )

    output = args.output_directory
    output.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, object]] = {}
    for key, file_name in AUDIT_FILES.items():
        destination = output / file_name
        with destination.open("wb") as target:
            for source_dir, local_index in state_sources:
                _copy_state(source_dir, file_name, local_index, target)
        files[key] = {
            "path": file_name,
            "byteLength": destination.stat().st_size,
            "sha256": _sha256(destination),
        }

    manifest = {
        "artifactType": "issue446-fixed-exchange-full-gold-audit-v1",
        "fixedHoldout": {
            "path": str(args.fixed_holdout),
            "manifestSha256": hashlib.sha256(fixed_manifest_bytes).hexdigest(),
            "sourceStateCount": int(fixed_manifest["sourceStateCount"]),
            "adjutantCandidateCount": ADJUTANT_COUNT,
            "discardCandidateCount": DISCARD_COUNT,
            "terminalRolloutCount": int(fixed_manifest["terminalRolloutCount"]),
            "binaryEquality": {
                name: _sha256(args.fixed_holdout / name) for name in FIXED_COMPARISON_FILES
            },
        },
        "sourceStateCount": len(actual_diagnostics),
        "groupCount": len(actual_diagnostics) * ADJUTANT_COUNT,
        "candidateCount": len(actual_diagnostics) * ADJUTANT_COUNT * DISCARD_COUNT,
        "stateFeatureCount": 343,
        "candidateMaskFeatureCount": 53,
        "files": files,
        "sourceDiagnostics": actual_diagnostics,
        "shards": [
            {
                "path": str(path),
                "manifestSha256": _sha256(path / "manifest.json"),
                "sourceStateCount": int(manifest["sourceStateCount"]),
            }
            for path, manifest in zip(args.shards, shard_manifests, strict=True)
        ],
        "replacementStates": replacement_records,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["fixedHoldout"], indent=2, sort_keys=True))
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _statewise_sha256(state_sources: list[tuple[Path, int]], file_name: str) -> str:
    digest = hashlib.sha256()
    for source_dir, local_index in state_sources:
        for chunk in _state_chunks(source_dir, file_name, local_index):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_state(source_dir: Path, file_name: str, local_index: int, target: BinaryIO) -> None:
    for chunk in _state_chunks(source_dir, file_name, local_index):
        target.write(chunk)


def _state_chunks(source_dir: Path, file_name: str, local_index: int) -> Iterator[bytes]:
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    state_count = int(manifest["sourceStateCount"])
    path = source_dir / file_name
    byte_length = path.stat().st_size
    if byte_length % state_count:
        raise ValueError(f"{path}: byte length is not divisible by state count.")
    state_bytes = byte_length // state_count
    if not 0 <= local_index < state_count:
        raise ValueError(f"{path}: state index is out of range.")
    with path.open("rb") as handle:
        handle.seek(local_index * state_bytes)
        remaining = state_bytes
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                raise EOFError(f"unexpected EOF in {path}.")
            remaining -= len(chunk)
            yield chunk


if __name__ == "__main__":
    raise SystemExit(main())
