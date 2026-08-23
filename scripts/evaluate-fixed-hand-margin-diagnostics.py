#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from napoleon_ml.bidding_q.fixed_hand_margin_training import (
    FixedHandMarginSample,
    load_fixed_hand_margin_checkpoint,
    predict_fixed_hand_margin_samples,
)
from napoleon_ml.bidding_q.margin_training import gaussian_success_probability
from napoleon_ml.bidding_q.pass_outcome_training import _raw_float, _std_from_raw
from napoleon_ml.policy.device import resolve_torch_device


SUITS = ("spades", "hearts", "diamonds", "clubs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue421-selection", type=Path, default=Path("/tmp/issue421-selected-states.json"))
    parser.add_argument("--issue421-report", type=Path, default=Path("/tmp/issue421-report.json"))
    parser.add_argument("--issue422-selection", type=Path, default=Path("/tmp/issue422-selected-states.json"))
    parser.add_argument("--issue422-report", type=Path, default=Path("/tmp/issue422-report.json"))
    parser.add_argument("--model", action="append", required=True, help="label=/path/to/checkpoint.pt")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    issue421_selection = json.loads(args.issue421_selection.read_text(encoding="utf-8"))
    issue421_report = json.loads(args.issue421_report.read_text(encoding="utf-8"))
    issue422_selection = json.loads(args.issue422_selection.read_text(encoding="utf-8"))
    issue422_report = json.loads(args.issue422_report.read_text(encoding="utf-8"))
    models = [parse_model_arg(value) for value in args.model]
    result = {
        "issue421Raise": {},
        "issue422Opening": {},
        "models": {label: str(path) for label, path in models},
    }
    for label, path in models:
        result["issue421Raise"][label] = evaluate_issue421(path, issue421_selection, issue421_report)
        result["issue422Opening"][label] = evaluate_issue422(path, issue422_selection, issue422_report)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "issue421": {label: result["issue421Raise"][label]["summary"] for label, _ in models},
        "issue422": {label: result["issue422Opening"][label]["summary"] for label, _ in models},
    }, indent=2, sort_keys=True))
    return 0


def evaluate_issue421(
    checkpoint: Path,
    selection: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    empirical_by_state = {row["stateKey"]: row for row in report["allStates"]}
    states = selection["states"]
    samples = [
        diagnostic_sample(
            state=state,
            action_index=int(state["bestRaise"]["actionIndex"]),
            target=int(state["bestRaise"]["target"]),
            suit=str(state["bestRaise"]["suit"]),
        )
        for state in states
    ]
    p_win, mu, sigma = predict_selected(checkpoint, samples)
    rows = []
    for index, state in enumerate(states):
        empirical = empirical_by_state[state["stateKey"]]
        target = int(state["bestRaise"]["target"])
        predicted_raise_ev = napoleon_relative_ev(float(p_win[index]), target)
        predicted_pass_ev = float(state["evPass"])
        predicted_delta = predicted_raise_ev - predicted_pass_ev
        empirical_delta = float(empirical["empiricalDelta"])
        rows.append({
            "stateKey": state["stateKey"],
            "action": state["bestRaise"]["actionLabel"],
            "currentBid": state["currentBidLabel"],
            "mu": float(mu[index]),
            "sigma": float(sigma[index]),
            "pWin": float(p_win[index]),
            "predictedRaiseEv": predicted_raise_ev,
            "predictedPassEv": predicted_pass_ev,
            "predictedDelta": predicted_delta,
            "empiricalRaiseEv": float(empirical["empiricalRaiseMean"]),
            "empiricalPassEv": float(empirical["empiricalPassMean"]),
            "empiricalDelta": empirical_delta,
            "decision": "RAISE" if predicted_delta > 0 else "PASS",
            "empiricalDecision": "RAISE" if empirical_delta > 0 else "PASS",
        })
    return {"summary": delta_summary(rows), "rows": rows}


def evaluate_issue422(
    checkpoint: Path,
    selection: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    empirical_by_state = {row["stateKey"]: row for row in report["rows"]}
    states = selection["states"]
    samples = [
        diagnostic_sample(
            state=state,
            action_index=int(state["bestOpeningBid"]["actionIndex"]),
            target=int(state["bestOpeningBid"]["target"]),
            suit=str(state["bestOpeningBid"]["suit"]),
        )
        for state in states
    ]
    p_win, mu, sigma = predict_selected(checkpoint, samples)
    rows = []
    for index, state in enumerate(states):
        empirical = empirical_by_state[state["stateKey"]]
        target = int(state["bestOpeningBid"]["target"])
        predicted_ev = napoleon_relative_ev(float(p_win[index]), target)
        empirical_ev = float(empirical["empiricalEv"])
        rows.append({
            "stateKey": state["stateKey"],
            "action": state["bestOpeningBid"]["actionLabel"],
            "mu": float(mu[index]),
            "sigma": float(sigma[index]),
            "pWin": float(p_win[index]),
            "predictedEv": predicted_ev,
            "empiricalEv": empirical_ev,
            "decision": "BID" if predicted_ev > 0 else "PASS",
            "empiricalDecision": "BID" if empirical_ev > 0 else "PASS",
        })
    return {"summary": opening_summary(rows), "rows": rows}


def predict_selected(checkpoint: Path, samples: list[FixedHandMarginSample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = resolve_torch_device("cpu")
    model, raw = load_fixed_hand_margin_checkpoint(checkpoint)
    predictions = predict_fixed_hand_margin_samples(
        model,
        samples,
        device=device,
        standardization=_std_from_raw(raw),
        variant=str(raw.get("variant", "M2")),  # type: ignore[arg-type]
        constant_sigma=_raw_float(raw, "constantSigma", 1.0),
    )
    rows = np.arange(len(samples))
    actions = np.asarray([sample.forced_action_index for sample in samples], dtype=np.int64)
    mu = predictions["mean"][rows, actions]
    sigma = predictions["sigma"][rows, actions]
    p_win = gaussian_success_probability(mu, sigma)
    return p_win, mu, sigma


def diagnostic_sample(
    *,
    state: dict[str, Any],
    action_index: int,
    target: int,
    suit: str,
) -> FixedHandMarginSample:
    return FixedHandMarginSample(
        fixed_hand_id=str(state["stateKey"]),
        hand_ids=(),
        forced_action_index=action_index,
        forced_target_point_cards=target,
        forced_suit=suit,
        model_input=np.asarray(state["modelInput"], dtype=np.float32),
        rollout_count=1,
        empirical_margin_mean=0.0,
        empirical_margin_std=1.0,
        empirical_win_rate=0.0,
        source_state_key=str(state["stateKey"]),
    )


def delta_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predicted = np.asarray([row["predictedDelta"] for row in rows], dtype=np.float64)
    empirical = np.asarray([row["empiricalDelta"] for row in rows], dtype=np.float64)
    residual = predicted - empirical
    false_pass = [row for row in rows if row["predictedDelta"] < 0 < row["empiricalDelta"]]
    false_raise = [row for row in rows if row["predictedDelta"] > 0 > row["empiricalDelta"]]
    return {
        "count": len(rows),
        "signAccuracy": float(np.mean((predicted > 0) == (empirical > 0))),
        "falsePASS": len(false_pass),
        "falseRaise": len(false_raise),
        "raiseRate": float(np.mean(predicted > 0)),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "pearson": pearson(predicted, empirical),
        "spearman": spearman(predicted, empirical),
    }


def opening_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predicted = np.asarray([row["predictedEv"] for row in rows], dtype=np.float64)
    empirical = np.asarray([row["empiricalEv"] for row in rows], dtype=np.float64)
    residual = predicted - empirical
    false_pass = [row for row in rows if row["predictedEv"] < 0 < row["empiricalEv"]]
    false_bid = [row for row in rows if row["predictedEv"] > 0 > row["empiricalEv"]]
    return {
        "count": len(rows),
        "signAccuracy": float(np.mean((predicted > 0) == (empirical > 0))),
        "falsePASS": len(false_pass),
        "falseBID": len(false_bid),
        "bidRate": float(np.mean(predicted > 0)),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "pearson": pearson(predicted, empirical),
        "spearman": spearman(predicted, empirical),
    }


def napoleon_relative_ev(p_win: float, target: int) -> float:
    return p_win * (7.0 * target / 4.0) + (1.0 - p_win) * (-3.0 * target / 4.0)


def pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else None


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    return pearson(rankdata(left), rankdata(right))


def rankdata(values: np.ndarray) -> np.ndarray:
    order = sorted(enumerate(values.tolist()), key=lambda item: item[1])
    ranks = np.zeros(values.shape[0], dtype=np.float64)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and order[end][1] == order[index][1]:
            end += 1
        rank = (index + end - 1) / 2.0
        for offset in range(index, end):
            ranks[order[offset][0]] = rank
        index = end
    return ranks


def parse_model_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise SystemExit("--model must be label=/path/to/checkpoint.pt")
    label, path = value.split("=", 1)
    return label, Path(path)


if __name__ == "__main__":
    raise SystemExit(main())
