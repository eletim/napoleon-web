"""Export a fixed-hand bidding margin checkpoint to a runtime ONNX artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from napoleon_ml.bidding_q.fixed_hand_margin_training import (
    FixedHandMarginCheckpointError,
    load_fixed_hand_margin_checkpoint,
)
from napoleon_ml.bidding_q.margin_training import export_bidding_margin_onnx
from napoleon_ml.dataset.tensors import BIDDING_MODEL_INPUT_FEATURE_COUNT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--source-note", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (FixedHandMarginCheckpointError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    model, raw = load_fixed_hand_margin_checkpoint(args.checkpoint)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    onnx_path = output / "margin.onnx"
    metadata_path = output / "margin.json"
    checkpoint_sha256 = _sha256_file(args.checkpoint)
    metadata = _metadata_from_checkpoint(
        raw,
        artifact_id=args.artifact_id,
        display_name=args.display_name,
        source_checkpoint_sha256=checkpoint_sha256,
        source_note=args.source_note,
    )
    parity = export_bidding_margin_onnx(
        model=model,
        metadata=metadata,
        onnx_path=onnx_path,
        metadata_path=metadata_path,
        sample_model_input=np.zeros(BIDDING_MODEL_INPUT_FEATURE_COUNT, dtype=np.float32),
    )
    manifest = {
        "artifactId": args.artifact_id,
        "displayName": args.display_name,
        "onnxPath": str(onnx_path),
        "metadataPath": str(metadata_path),
        "sourceCheckpointSha256": checkpoint_sha256,
        "onnxParity": parity,
    }
    (output / "export-report.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _metadata_from_checkpoint(
    raw: dict[str, object],
    *,
    artifact_id: str,
    display_name: str,
    source_checkpoint_sha256: str,
    source_note: str,
) -> dict[str, object]:
    metadata = dict(raw)
    metadata.pop("modelState", None)
    metadata.update(
        {
            "artifactId": artifact_id,
            "displayName": display_name,
            "sourceCheckpointSha256": source_checkpoint_sha256,
        }
    )
    if source_note:
        metadata["sourceNote"] = source_note
    return metadata


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
