"""Evaluate an exchange scorer on the fixed Issue #446 full-gold holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from napoleon_ml.exchange_value import (
    combine_exchange_counterfactual_datasets,
    load_exchange_counterfactual_dataset,
    load_exchange_value_checkpoint,
)
from napoleon_ml.exchange_value.full_gold_audit import (
    audit_training_leakage_report,
    exchange_full_gold_report,
    exclude_audit_overlaps,
    load_exchange_full_gold_audit,
    predict_full_gold_scores,
)
from napoleon_ml.policy.device import resolve_torch_device


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_directory", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--group-batch-size", type=int, default=16)
    parser.add_argument("--training-dataset", action="append", type=Path)
    parser.add_argument("--exclude-audit-overlaps", action="store_true")
    args = parser.parse_args()
    audit = load_exchange_full_gold_audit(args.audit_directory)
    model, _checkpoint = load_exchange_value_checkpoint(args.checkpoint)
    device = resolve_torch_device(args.device)
    model.to(device.torch_device)
    scores = predict_full_gold_scores(
        audit, model, device=device.torch_device, group_batch_size=args.group_batch_size
    )
    checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    report = exchange_full_gold_report(
        audit, scores, scorer_name=f"{args.checkpoint}:{checkpoint_sha}"
    )
    if args.training_dataset is not None:
        training_components = tuple(
            load_exchange_counterfactual_dataset(path, load_legacy_model_input=False)
            for path in args.training_dataset
        )
        training_dataset = (
            training_components[0]
            if len(training_components) == 1
            else combine_exchange_counterfactual_datasets(training_components)
        )
        if args.exclude_audit_overlaps:
            training_dataset, exclusion = exclude_audit_overlaps(audit, training_dataset)
            report["trainingAuditExclusion"] = exclusion
        report["leakageGuard"] = audit_training_leakage_report(audit, training_dataset)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
