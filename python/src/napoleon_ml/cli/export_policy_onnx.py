"""Export a saved policy MLP checkpoint to ONNX."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

from napoleon_ml.adjutant.ppo import NON_PLAYING_RL_SAMPLE_TYPE as ADJUTANT_RL_SAMPLE_TYPE
from napoleon_ml.bidding.ppo import NON_PLAYING_RL_SAMPLE_TYPE as BIDDING_RL_SAMPLE_TYPE
from napoleon_ml.dataset.errors import DatasetError
from napoleon_ml.exchange.ppo import NON_PLAYING_RL_SAMPLE_TYPE as EXCHANGE_RL_SAMPLE_TYPE
from napoleon_ml.nonplaying_onnx_export import (
    PolicyType,
    export_adjutant_rl_checkpoint_to_onnx,
    export_bidding_rl_checkpoint_to_onnx,
    export_exchange_rl_checkpoint_to_onnx,
    export_nonplaying_checkpoint_to_onnx,
)
from napoleon_ml.policy.onnx_export import (
    export_policy_checkpoint_to_onnx,
    export_policy_critic_checkpoint_to_onnx,
)

from ._policy_common import load_checked_manifest


class _NonPlayingExportReport(Protocol):
    def to_dict(self) -> dict[str, object]:
        ...


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument(
        "--policy-type",
        choices=("playing", "bidding", "exchange", "adjutant", "critic"),
        default="playing",
        help="Checkpoint/export contract to use. Defaults to the existing playing exporter.",
    )
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
        return _handle_export_cli_error(error)


def _run(args: argparse.Namespace) -> int:
    metadata_output = args.metadata_output or args.output.with_suffix(".json")
    if args.policy_type == "critic":
        critic_report = export_policy_critic_checkpoint_to_onnx(
            checkpoint_path=args.checkpoint,
            onnx_path=args.output,
            metadata_path=metadata_output,
        )

        if args.json:
            print(json.dumps(critic_report.to_dict(), indent=2, sort_keys=True))
        else:
            print(f"onnx: {critic_report.onnx_path}")
            print(f"metadata: {critic_report.metadata_path}")
            print("parity_sample:")
            print(f"  max_abs_value_diff: {critic_report.max_abs_value_diff:.8f}")
            print(f"  pytorch_value: {critic_report.pytorch_value:.8f}")
            print(f"  onnx_value: {critic_report.onnx_value:.8f}")

        return 0

    sample_type = _dataset_sample_type(args.dataset_directory)
    if args.policy_type == "bidding" and sample_type == BIDDING_RL_SAMPLE_TYPE:
        bidding_rl_report = export_bidding_rl_checkpoint_to_onnx(
            dataset_directory=args.dataset_directory,
            checkpoint_path=args.checkpoint,
            onnx_path=args.output,
            metadata_path=metadata_output,
        )
        _print_nonplaying_report(bidding_rl_report, as_json=args.json)
        return 0

    if args.policy_type == "adjutant" and sample_type == ADJUTANT_RL_SAMPLE_TYPE:
        adjutant_rl_report = export_adjutant_rl_checkpoint_to_onnx(
            dataset_directory=args.dataset_directory,
            checkpoint_path=args.checkpoint,
            onnx_path=args.output,
            metadata_path=metadata_output,
        )
        _print_nonplaying_report(adjutant_rl_report, as_json=args.json)
        return 0

    if args.policy_type == "exchange" and sample_type == EXCHANGE_RL_SAMPLE_TYPE:
        exchange_rl_report = export_exchange_rl_checkpoint_to_onnx(
            dataset_directory=args.dataset_directory,
            checkpoint_path=args.checkpoint,
            onnx_path=args.output,
            metadata_path=metadata_output,
        )
        _print_nonplaying_report(exchange_rl_report, as_json=args.json)
        return 0

    manifest = load_checked_manifest(args.dataset_directory, command_label="export-policy-onnx")
    if args.policy_type == "playing":
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

    nonplaying_report = export_nonplaying_checkpoint_to_onnx(
        policy_type=cast(PolicyType, args.policy_type),
        dataset_directory=args.dataset_directory,
        checkpoint_path=args.checkpoint,
        onnx_path=args.output,
        metadata_path=metadata_output,
        manifest=manifest,
        verify_integrity=not args.no_integrity_check,
    )

    _print_nonplaying_report(nonplaying_report, as_json=args.json)

    return 0


def _dataset_sample_type(dataset_directory: Path) -> object:
    manifest_path = dataset_directory / "manifest.json"
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    return raw.get("sampleType")


def _print_nonplaying_report(report: _NonPlayingExportReport, *, as_json: bool) -> None:
    report_dict = report.to_dict()
    if as_json:
        print(json.dumps(report_dict, indent=2, sort_keys=True))
        return
    print(f"onnx: {report_dict['onnxPath']}")
    print(f"metadata: {report_dict['metadataPath']}")
    parity_raw = report_dict["paritySample"]
    if not isinstance(parity_raw, dict):
        raise ValueError("non-playing export report paritySample must be a dictionary.")
    parity = parity_raw
    print("parity_sample:")
    print(f"  seed: {parity['seed']}")
    print(f"  step: {parity['step']}")
    print(f"  max_abs_logit_diff: {parity['maxAbsLogitDiff']:.8f}")
    print(f"  pytorch_selection: {parity['pytorchSelection']}")
    print(f"  onnx_selection: {parity['onnxSelection']}")


def _handle_export_cli_error(error: Exception) -> int:
    if isinstance(error, DatasetError | ValueError):
        print(f"error: {error}", file=sys.stderr)
        return 1

    raise error


if __name__ == "__main__":
    raise SystemExit(main())
