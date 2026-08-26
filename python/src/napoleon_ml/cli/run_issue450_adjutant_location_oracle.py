"""Run the paired compact396 vs diagnostic compact401 Issue #450 experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from napoleon_ml.exchange_value import (
    ExchangeValueTrainConfig,
    combine_exchange_counterfactual_datasets,
    load_exchange_counterfactual_dataset,
    save_exchange_value_artifact,
    train_exchange_value_model,
)
from napoleon_ml.exchange_value.full_gold_audit import (
    exclude_audit_overlaps,
    load_exchange_full_gold_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_directories", nargs="+", type=Path)
    parser.add_argument("--oracle-location-overlay", type=Path, required=True)
    parser.add_argument("--fixed-audit", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    components = tuple(load_exchange_counterfactual_dataset(
        path, load_legacy_model_input=False,
        oracle_location_overlay=args.oracle_location_overlay,
    ) for path in args.dataset_directories)
    dataset = (
        components[0]
        if len(components) == 1
        else combine_exchange_counterfactual_datasets(components)
    )
    dataset, exclusion = exclude_audit_overlaps(
        load_exchange_full_gold_audit(args.fixed_audit), dataset
    )
    common = ExchangeValueTrainConfig(
        seed=436, epochs=args.epochs, batch_size=1024,
        hidden_dims=(512, 512, 256, 256), input_variant="compact396",
        loss="huber", pointwise_loss_weight=0.0, listwise_loss_weight=1.0,
        listwise_temperature=0.5, pairwise_state_batch_size=4, patience=6,
        device=args.device, warm_start_checkpoint=str(args.baseline_checkpoint),
    )
    configs = {
        "compact396": common,
        "compact401-oracle-location": replace(common, input_variant="compact401-oracle-location"),
    }
    summary: dict[str, object] = {
        "datasetDirectories": [str(path) for path in args.dataset_directories],
        "oracleLocationOverlay": str(args.oracle_location_overlay),
        "sourceStateCount": dataset.source_state_count,
        "sampleCount": dataset.sample_count,
        "fixedAuditExclusion": exclusion,
        "runs": {},
    }
    for name, config in configs.items():
        result = train_exchange_value_model(dataset, config)
        artifact = save_exchange_value_artifact(
            args.output_directory / name, result=result, dataset=dataset
        )
        summary["runs"][name] = {
            "config": config.to_dict(), "bestEpoch": result.best_epoch,
            "validation": result.validation_report, "final": result.final_report,
            "artifact": artifact,
        }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "training-comparison.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
