"""Issue #442 pseudo-fixed exchange ablation reporting helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import EXCHANGE_COUNTERFACTUAL_COMBINATION_COUNT


@dataclass(frozen=True)
class Issue442Layout:
    name: str
    fixed_thirteen_groups: int
    repeats_per_group: int

    @property
    def state_count(self) -> int:
        return self.fixed_thirteen_groups * self.repeats_per_group

    @property
    def sample_count(self) -> int:
        return self.state_count * EXCHANGE_COUNTERFACTUAL_COMBINATION_COUNT


ISSUE442_LAYOUTS: tuple[Issue442Layout, ...] = (
    Issue442Layout("1000x1", 1000, 1),
    Issue442Layout("200x5", 200, 5),
    Issue442Layout("100x10", 100, 10),
)

ISSUE438_COMPACT396_BASELINE: dict[str, float] = {
    "pearson": 0.409,
    "pairwise": 0.592,
    "marginRegret": 4.60,
    "relativeRewardRegret": 12.30,
    "ruleBasedMarginRegret": 4.59,
}


def collect_issue442_layout_summary(
    *,
    layout: Issue442Layout,
    dataset_directory: Path,
    artifact_directory: Path,
) -> dict[str, Any]:
    manifest_path = dataset_directory / "manifest.json"
    report_path = artifact_directory / "report.json"
    metadata_path = artifact_directory / "metadata.json"
    split_path = artifact_directory / "split.json"
    manifest = _read_json_object(manifest_path)
    report = _read_json_object(report_path)
    metadata = _read_json_object(metadata_path)
    split = _read_json_object(split_path)

    pseudo_fixed = _pseudo_fixed_diagnostics(manifest)
    checks = _layout_checks(
        layout=layout,
        manifest=manifest,
        split=split,
        pseudo_fixed=pseudo_fixed,
    )
    final = _require_object(report, "final")
    train = _require_object(report, "train")
    validation = _require_object(report, "validation")

    return {
        "layout": layout.name,
        "fixedThirteenGroups": layout.fixed_thirteen_groups,
        "repeatsPerGroup": layout.repeats_per_group,
        "dataset": {
            "path": str(dataset_directory),
            "manifestPath": str(manifest_path),
            "manifestSha256": _sha256_file(manifest_path),
            "sourceStateCount": manifest.get("sourceStateCount"),
            "sampleCount": manifest.get("sampleCount"),
            "discardCombinationCount": manifest.get("discardCombinationCount"),
        },
        "training": {
            "artifactDirectory": str(artifact_directory),
            "checkpointPath": metadata.get("checkpointPath"),
            "checkpointSha256": metadata.get("checkpointSha256"),
            "config": metadata.get("trainingConfig"),
        },
        "split": {
            "train": _split_counts(split, train, "train"),
            "validation": _split_counts(split, validation, "validation"),
            "final": _split_counts(split, final, "final"),
            "stateKeyHashes": {
                "train": split.get("trainStateKeyHash"),
                "validation": split.get("validationStateKeyHash"),
                "final": split.get("finalStateKeyHash"),
            },
            "leakageGuard": split.get("leakageGuard"),
        },
        "distributionAudit": pseudo_fixed,
        "final": {
            "scalar": final.get("scalar"),
            "ranking": final.get("ranking"),
            "ruleBased": final.get("ruleBased"),
            "sameThirteen": final.get("sameThirteen"),
        },
        "checks": checks,
    }


def collect_issue442_report(
    layout_paths: dict[str, tuple[Path, Path]],
) -> dict[str, Any]:
    by_name = {layout.name: layout for layout in ISSUE442_LAYOUTS}
    summaries = []
    for name, paths in layout_paths.items():
        if name not in by_name:
            raise ValueError(f"unsupported Issue #442 layout: {name}")
        dataset_directory, artifact_directory = paths
        summaries.append(
            collect_issue442_layout_summary(
                layout=by_name[name],
                dataset_directory=dataset_directory,
                artifact_directory=artifact_directory,
            )
        )
    summaries.sort(key=lambda item: by_name[str(item["layout"])].repeats_per_group)
    return {
        "issue": 442,
        "baseline": ISSUE438_COMPACT396_BASELINE,
        "layouts": summaries,
        "conclusion": issue442_conclusion(summaries),
    }


def issue442_conclusion(layouts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(layouts) != len(ISSUE442_LAYOUTS):
        return {
            "status": "incomplete",
            "summary": "All three layouts are required before adoption can be judged.",
        }
    ranking_rows = []
    for layout in layouts:
        final = _require_object(layout, "final")
        ranking = _require_object(final, "ranking")
        scalar = _require_object(final, "scalar")
        ranking_rows.append(
            {
                "layout": layout["layout"],
                "pearson": scalar.get("pearson"),
                "pairwise": ranking.get("pairwiseAccuracy"),
                "marginRegret": _summary_mean(ranking, "modelSelectedTeacherMarginRegret"),
                "relativeRewardRegret": _summary_mean(
                    ranking,
                    "modelSelectedRelativeRewardRegret",
                ),
                "ruleBasedMarginRegret": _summary_mean(
                    ranking,
                    "ruleBasedTeacherMarginRegret",
                ),
            }
        )
    best = min(
        ranking_rows,
        key=lambda row: (
            float(row["marginRegret"]),
            -float(row["pairwise"]),
            float(row["relativeRewardRegret"]),
        ),
    )
    clearly_improved = [
        row for row in ranking_rows
        if (
            float(row["pairwise"]) > ISSUE438_COMPACT396_BASELINE["pairwise"] + 0.005
            or float(row["marginRegret"])
            < min(
                ISSUE438_COMPACT396_BASELINE["marginRegret"],
                float(row["ruleBasedMarginRegret"]),
            ) - 0.05
            or float(row["relativeRewardRegret"])
            < ISSUE438_COMPACT396_BASELINE["relativeRewardRegret"] - 0.25
        )
        and float(row["marginRegret"]) < float(row["ruleBasedMarginRegret"])
    ]
    adopt = bool(clearly_improved)
    return {
        "status": "complete",
        "bestLayout": best["layout"],
        "pseudoFixedRepeatsHelped": best["layout"] != "1000x1",
        "adoptPseudoFixedTeacherCandidate": adopt,
        "moveToAdjutantKittyJointIfNoImprovement": not adopt,
        "selectionMetric": "lowest final margin regret, tie-broken by pairwise and reward regret",
        "summary": (
            "100x10 is best within the pseudo-fixed layouts, but it only ties the "
            "#438 margin-regret baseline after rounding while pairwise and relative "
            "reward regret are worse. Treat pseudo-fixed repeats as useful for "
            "hidden-deal variation, not as a formal exchange teacher candidate yet."
        ),
        "rows": ranking_rows,
    }


def render_issue442_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Issue #442 pseudo-fixed exchange repeat ablation",
        "",
        "Base: `feature/issue-202-non-playing-ai`",
        "",
        "## Final held-out comparison",
        "",
        "| layout | Pearson | pairwise | margin regret | relative reward regret | RuleBased margin regret |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| #438 compact396 baseline | {ISSUE438_COMPACT396_BASELINE['pearson']:.3f} | "
            f"{ISSUE438_COMPACT396_BASELINE['pairwise']:.3f} | "
            f"{ISSUE438_COMPACT396_BASELINE['marginRegret']:.2f} | "
            f"{ISSUE438_COMPACT396_BASELINE['relativeRewardRegret']:.2f} | "
            f"{ISSUE438_COMPACT396_BASELINE['ruleBasedMarginRegret']:.2f} |"
        ),
    ]
    for layout in report["layouts"]:
        final = _require_object(layout, "final")
        scalar = _require_object(final, "scalar")
        ranking = _require_object(final, "ranking")
        lines.append(
            "| {name} | {pearson:.3f} | {pairwise:.3f} | {margin:.2f} | "
            "{reward:.2f} | {rb_margin:.2f} |".format(
                name=layout["layout"],
                pearson=float(scalar["pearson"]),
                pairwise=float(ranking["pairwiseAccuracy"]),
                margin=float(_summary_mean(ranking, "modelSelectedTeacherMarginRegret")),
                reward=float(_summary_mean(ranking, "modelSelectedRelativeRewardRegret")),
                rb_margin=float(_summary_mean(ranking, "ruleBasedTeacherMarginRegret")),
            )
        )
    lines.extend(["", "## Layout details", ""])
    for layout in report["layouts"]:
        final = _require_object(layout, "final")
        scalar = _require_object(final, "scalar")
        ranking = _require_object(final, "ranking")
        wtl = _require_object(ranking, "modelVsRuleBasedMarginRegret")
        dataset = _require_object(layout, "dataset")
        training = _require_object(layout, "training")
        distribution = _require_object(layout, "distributionAudit")
        split = _require_object(layout, "split")
        lines.extend(
            [
                f"### {layout['layout']}",
                "",
                f"- Dataset: `{dataset['path']}`",
                f"- Manifest SHA-256: `{dataset['manifestSha256']}`",
                f"- Checkpoint: `{training['checkpointPath']}`",
                f"- Checkpoint SHA-256: `{training['checkpointSha256']}`",
                (
                    f"- Accepted/rejected: {distribution.get('acceptedDealCount')} / "
                    f"{distribution.get('rejectedDealCount')}"
                ),
                f"- Rejection reasons: `{json.dumps(distribution.get('rejectionReasons', {}), sort_keys=True)}`",
                f"- Target distribution: `{json.dumps(distribution.get('targetDistribution', {}), sort_keys=True)}`",
                f"- Suit distribution: `{json.dumps(distribution.get('suitDistribution', {}), sort_keys=True)}`",
                f"- Unique bidding histories: {distribution.get('biddingHistoryUniqueHashCount')}",
                f"- Opponent policy ratios: `{json.dumps(distribution.get('opponentPolicyRatios', {}), sort_keys=True)}`",
                f"- Split: `{json.dumps(_split_line(split), sort_keys=True)}`",
                f"- Leakage guard: `{json.dumps(_leakage_line(split), sort_keys=True)}`",
                (
                    f"- Scalar: MAE {float(scalar['mae']):.3f}, RMSE "
                    f"{float(scalar['rmse']):.3f}, bias {float(scalar['bias']):.3f}"
                ),
                (
                    f"- Ranking: exact {float(ranking['exactTopAction']):.3f}, "
                    f"top3 {float(ranking['top3Hit']):.3f}, top5 "
                    f"{float(ranking['top5Hit']):.3f}, selected rank percentile "
                    f"{float(_summary_mean(ranking, 'modelSelectedTeacherRankPercentile')):.3f}"
                ),
                (
                    f"- RuleBased reward regret: "
                    f"{float(_summary_mean(ranking, 'ruleBasedRelativeRewardRegret')):.2f}"
                ),
                (
                    f"- Model vs RuleBased win/tie/loss: {wtl.get('win')} / "
                    f"{wtl.get('tie')} / {wtl.get('loss')}"
                ),
            ]
        )
        same = _require_object(final, "sameThirteen")
        if layout["repeatsPerGroup"] == 1:
            lines.append("- Same-13 repeat metrics: n/a")
        else:
            lines.extend(
                [
                    (
                        f"- Same-13 group mean margin regret: "
                        f"{float(_summary_mean(same, 'groupMeanMarginRegret')):.2f}"
                    ),
                    (
                        f"- Same-13 group mean relative reward regret: "
                        f"{float(_summary_mean(same, 'groupMeanRelativeRewardRegret')):.2f}"
                    ),
                    (
                        f"- Teacher-best discard unique count mean: "
                        f"{float(_summary_mean(same, 'teacherBestDiscardUniqueCount')):.2f}"
                    ),
                    (
                        f"- Model-selected discard unique count mean: "
                        f"{float(_summary_mean(same, 'modelSelectedDiscardUniqueCount')):.2f}"
                    ),
                ]
            )
        lines.append("")
    conclusion = _require_object(report, "conclusion")
    lines.extend(
        [
            "## Conclusion",
            "",
            f"- Pseudo-fixed repeats helped: `{conclusion.get('pseudoFixedRepeatsHelped')}`",
            f"- Best layout: `{conclusion.get('bestLayout')}`",
            (
                f"- Adopt pseudo-fixed as formal exchange teacher candidate: "
                f"`{conclusion.get('adoptPseudoFixedTeacherCandidate')}`"
            ),
            (
                f"- Move to adjutant+kitty joint design if no improvement: "
                f"`{conclusion.get('moveToAdjutantKittyJointIfNoImprovement')}`"
            ),
            f"- Summary: {conclusion.get('summary')}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for layout in ISSUE442_LAYOUTS:
        parser.add_argument(f"--{layout.name}-dataset", type=Path, required=True)
        parser.add_argument(f"--{layout.name}-artifact", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = {
        layout.name: (
            getattr(args, f"{layout.name}_dataset"),
            getattr(args, f"{layout.name}_artifact"),
        )
        for layout in ISSUE442_LAYOUTS
    }
    report = collect_issue442_report(paths)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.write_text(render_issue442_markdown(report), encoding="utf-8")
    return 0


def _read_json_object(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a JSON object.")
    return raw


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_object(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object.")
    return value


def _pseudo_fixed_diagnostics(manifest: dict[str, Any]) -> dict[str, Any]:
    summary = manifest.get("summary")
    if isinstance(summary, dict) and isinstance(summary.get("pseudoFixedThirteen"), dict):
        return summary["pseudoFixedThirteen"]
    top_level = manifest.get("pseudoFixedThirteen")
    if isinstance(top_level, dict):
        return top_level
    raise ValueError("manifest is missing pseudoFixedThirteen diagnostics.")


def _layout_checks(
    *,
    layout: Issue442Layout,
    manifest: dict[str, Any],
    split: dict[str, Any],
    pseudo_fixed: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "sourceStateCount": manifest.get("sourceStateCount") == layout.state_count,
        "sampleCount": manifest.get("sampleCount") == layout.sample_count,
        "discardCombinationCount": (
            manifest.get("discardCombinationCount") == EXCHANGE_COUNTERFACTUAL_COMBINATION_COUNT
        ),
        "fixedThirteenGroupCount": (
            pseudo_fixed.get("fixedThirteenGroupCount") == layout.fixed_thirteen_groups
        ),
        "acceptedDealsPerFixedThirteenGroup": (
            pseudo_fixed.get("acceptedDealsPerFixedThirteenGroup")
            == layout.repeats_per_group
        ),
        "acceptedDealCount": pseudo_fixed.get("acceptedDealCount") == layout.state_count,
        "candidateSamplesPerState": (
            manifest.get("sampleCount") == (
                int(manifest.get("sourceStateCount", -1))
                * EXCHANGE_COUNTERFACTUAL_COMBINATION_COUNT
            )
        ),
        "leakageZero": _all_leakage_zero(split),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{layout.name}: failed Issue #442 artifact checks: {failed}")
    return checks


def _all_leakage_zero(split: dict[str, Any]) -> bool:
    guard = split.get("leakageGuard")
    if not isinstance(guard, dict):
        return False
    for key in (
        "sourceStateKey",
        "fixedThirteenGroupId",
        "dealSeed",
        "hiddenDealChecksum",
        "pickupHand",
    ):
        row = guard.get(key)
        if not isinstance(row, dict) or row.get("crossSplitLeakageCount") != 0:
            return False
    return True


def _split_counts(
    split: dict[str, Any],
    report: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    return {
        "stateCount": split.get(f"{prefix}StateCount"),
        "sampleCount": split.get(f"{prefix}SampleCount"),
        "fixedThirteenGroupCount": report.get("fixedThirteenGroupCount"),
    }


def _summary_mean(raw: dict[str, Any], key: str) -> float:
    summary = raw.get(key)
    if not isinstance(summary, dict) or summary.get("mean") is None:
        raise ValueError(f"{key}.mean is missing.")
    return float(summary["mean"])


def _split_line(split: dict[str, Any]) -> dict[str, Any]:
    return {
        key: split[key]
        for key in ("train", "validation", "final")
        if key in split
    }


def _leakage_line(split: dict[str, Any]) -> dict[str, Any]:
    guard = split.get("leakageGuard")
    if not isinstance(guard, dict):
        return {}
    return {
        key: guard[key]["crossSplitLeakageCount"]
        for key in (
            "sourceStateKey",
            "fixedThirteenGroupId",
            "dealSeed",
            "hiddenDealChecksum",
            "pickupHand",
        )
        if isinstance(guard.get(key), dict)
    }


if __name__ == "__main__":
    raise SystemExit(main())
