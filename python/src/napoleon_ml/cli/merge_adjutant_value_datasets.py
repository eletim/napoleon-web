"""Merge Issue #446 adjutant-value dataset shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

FEATURE_COUNT = 290
ADJUTANT_COUNT = 53


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("shards", nargs="+", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    merge_adjutant_value_datasets(args.shards, args.output_directory)
    return 0


def merge_adjutant_value_datasets(shards: Sequence[Path], output_directory: Path) -> None:
    if not shards:
        raise ValueError("at least one shard is required.")
    manifests = [_load_manifest(path) for path in shards]
    _validate_compatible(manifests)

    output_directory.mkdir(parents=True, exist_ok=True)
    for name in ("features.f32", "contract-margin.f32", "relative-reward.f32", "candidate-card.u8"):
        with (output_directory / name).open("wb") as out:
            for shard in shards:
                with (shard / name).open("rb") as src:
                    shutil.copyfileobj(src, out, length=1024 * 1024)

    source_offset = 0
    diagnostics: list[dict[str, Any]] = []
    with (output_directory / "state-index.u32").open("wb") as out:
        for shard, manifest in zip(shards, manifests, strict=True):
            sample_count = int(manifest["sampleCount"])
            state_indices = np.memmap(
                shard / "state-index.u32",
                mode="r",
                dtype="<u4",
                shape=(sample_count,),
            )
            adjusted = np.asarray(state_indices, dtype=np.uint32) + np.uint32(source_offset)
            out.write(adjusted.astype("<u4", copy=False).tobytes(order="C"))
            for diagnostic in manifest.get("sourceDiagnostics", []):
                item = dict(diagnostic)
                item["sourceIndex"] = int(item["sourceIndex"]) + source_offset
                diagnostics.append(item)
            source_offset += int(manifest["sourceStateCount"])

    merged_manifest = dict(manifests[0])
    merged_manifest["sourceStateCount"] = sum(int(item["sourceStateCount"]) for item in manifests)
    merged_manifest["sampleCount"] = sum(int(item["sampleCount"]) for item in manifests)
    merged_manifest["terminalRolloutCount"] = sum(
        int(item["terminalRolloutCount"]) for item in manifests
    )
    merged_manifest["sourceDiagnostics"] = diagnostics
    merged_manifest["mergedShardCount"] = len(shards)
    merged_manifest["mergedShards"] = [
        {
            "path": str(shard),
            "manifestSha256": _sha256(shard / "manifest.json"),
            "sourceStateCount": int(manifest["sourceStateCount"]),
            "sampleCount": int(manifest["sampleCount"]),
            "terminalRolloutCount": int(manifest["terminalRolloutCount"]),
        }
        for shard, manifest in zip(shards, manifests, strict=True)
    ]
    (output_directory / "manifest.json").write_text(
        json.dumps(merged_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_manifest(directory: Path) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    sample_count = int(manifest["sampleCount"])
    source_count = int(manifest["sourceStateCount"])
    if sample_count != source_count * ADJUTANT_COUNT:
        raise ValueError(f"{directory}: sampleCount must be sourceStateCount*53.")
    if int(manifest["featureCount"]) != FEATURE_COUNT:
        raise ValueError(f"{directory}: featureCount must be {FEATURE_COUNT}.")
    return manifest


def _validate_compatible(manifests: Sequence[dict[str, Any]]) -> None:
    keys = (
        "datasetSchemaVersion",
        "sampleType",
        "teacherId",
        "mode",
        "featureCount",
        "stateFeatureCount",
        "candidateCountPerState",
        "runtimeOrder",
        "policyPath",
        "proposal",
    )
    first = manifests[0]
    for manifest in manifests[1:]:
        for key in keys:
            if manifest.get(key) != first.get(key):
                raise ValueError(f"incompatible shard manifest field: {key}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
