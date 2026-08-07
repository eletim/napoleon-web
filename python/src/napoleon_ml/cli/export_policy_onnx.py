"""Export a saved legal-play policy MLP checkpoint to ONNX."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from napoleon_ml.cli._policy_common import handle_cli_error, load_checked_manifest
from napoleon_ml.policy.onnx_export import export_policy_checkpoint_to_onnx


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metadata-output",
        type=Path,
        help="JSON metadata output path. Defaults to the ONNX path with .json suffix.",
    )
    parser.add_argument("--no-integrity-check", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return _run(args)
    except Exception as error:
        return handle_cli_error(error)


def _run(args: argparse.Namespace) -> int:
    metadata_output = args.metadata_output or args.output.with_suffix(".json")
    manifest = load_checked_manifest(args.dataset_directory, command_label="export-policy-onnx")
    report = export_policy_checkpoint_to_onnx(
        dataset_directory=args.dataset_directory,
        checkpoint_path=args.checkpoint,
        onnx_path=args.output,
        metadata_path=metadata_output,
        manifest=manifest,
        verify_integrity=not args.no_integrity_check,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"onnx: {report.onnx_path}")
        print(f"metadata: {report.metadata_path}")
        print("parity_sample:")
        print(f"  seed: {report.sample_seed}")
        print(f"  step: {report.sample_step}")
        print(f"  max_abs_logit_diff: {report.max_abs_logit_diff:.8f}")
        print(f"  pytorch_selected_card_index: {report.pytorch_selected_card_index}")
        print(f"  onnx_selected_card_index: {report.onnx_selected_card_index}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
