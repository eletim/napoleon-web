"""Fixed #446 full-gold exchange ranking audit and failure analysis."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from napoleon_ml.dataset.constants import EXPECTED_CARD_IDS

from .dataset import ExchangeCounterfactualDataset
from .tactical import compact396_tactical_features, compact406_value_input

ADJUTANT_COUNT = 53
DISCARD_COUNT = 286
STATE_FEATURE_COUNT = 343
CARD_COUNT = 53
CONTAINMENT_K = (1, 4, 8, 16, 24, 32, 48, 64, 96, 128)
PROPOSAL_BUDGETS = (16, 24, 32, 48, 64)
ISSUE446_BASELINE = {
    "containment": {"4": 0.0654, "8": 0.1139, "16": 0.1981, "32": 0.3258, "64": 0.5110},
    "practicalProposalContainment": 0.2358,
    "proposalBestMarginRegret": 1.1235,
}


@dataclass(frozen=True)
class ExchangeFullGoldAudit:
    directory: Path
    manifest: dict[str, Any]
    state_features: np.memmap
    candidate_masks: np.memmap
    contract_margins: np.memmap
    relative_rewards: np.memmap
    rule_based_candidates: np.memmap
    gold_candidates: np.memmap

    @property
    def group_count(self) -> int:
        return int(self.manifest["groupCount"])


def load_exchange_full_gold_audit(directory: Path | str) -> ExchangeFullGoldAudit:
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("artifactType") != "issue446-fixed-exchange-full-gold-audit-v1":
        raise ValueError("exchange full-gold audit artifactType mismatch.")
    group_count = int(manifest["groupCount"])
    files = manifest["files"]

    def mapped(name: str, dtype: str, shape: tuple[int, ...]) -> np.memmap:
        path = root / str(files[name]["path"])
        expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
        if path.stat().st_size != expected:
            raise ValueError(f"{path}: expected {expected} bytes, got {path.stat().st_size}.")
        return np.memmap(path, mode="r", dtype=dtype, shape=shape)

    return ExchangeFullGoldAudit(
        directory=root,
        manifest=manifest,
        state_features=mapped("stateFeatures", "<f4", (group_count, STATE_FEATURE_COUNT)),
        candidate_masks=mapped("candidateMask", "u1", (group_count, DISCARD_COUNT, CARD_COUNT)),
        contract_margins=mapped("contractMargin", "<f4", (group_count, DISCARD_COUNT)),
        relative_rewards=mapped("relativeReward", "<f4", (group_count, DISCARD_COUNT)),
        rule_based_candidates=mapped("ruleBasedCandidate", "<u4", (group_count,)),
        gold_candidates=mapped("goldCandidate", "<u4", (group_count,)),
    )


def predict_full_gold_scores(
    audit: ExchangeFullGoldAudit,
    model: torch.nn.Module,
    *,
    device: torch.device,
    group_batch_size: int = 16,
) -> np.ndarray:
    input_dim = int(model.config.input_dim)
    if input_dim not in (396, 406):
        raise ValueError(f"full-gold scorer input must be 396 or 406, got {input_dim}.")
    model.eval()
    result = np.empty((audit.group_count, DISCARD_COUNT), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, audit.group_count, group_batch_size):
            end = min(start + group_batch_size, audit.group_count)
            states = np.asarray(audit.state_features[start:end])
            masks = np.asarray(audit.candidate_masks[start:end], dtype=np.float32)
            compact396 = np.concatenate(
                (
                    np.broadcast_to(
                        states[:, None, :], (end - start, DISCARD_COUNT, STATE_FEATURE_COUNT)
                    ),
                    masks,
                ),
                axis=2,
            ).reshape(-1, 396)
            inputs = compact406_value_input(compact396) if input_dim == 406 else compact396
            prediction = model(torch.as_tensor(inputs, dtype=torch.float32, device=device)).reshape(
                end - start, DISCARD_COUNT
            )
            result[start:end] = prediction.detach().cpu().numpy()
    return result


def exchange_full_gold_report(
    audit: ExchangeFullGoldAudit,
    scores: np.ndarray,
    *,
    scorer_name: str,
) -> dict[str, object]:
    if scores.shape != (audit.group_count, DISCARD_COUNT):
        raise ValueError("score matrix shape mismatch.")
    ranks: list[int] = []
    top1_margin_regrets: list[float] = []
    top1_reward_regrets: list[float] = []
    rule_based_margin_regrets: list[float] = []
    rule_based_reward_regrets: list[float] = []
    oracle_margin: dict[int, list[float]] = {k: [] for k in CONTAINMENT_K}
    oracle_reward: dict[int, list[float]] = {k: [] for k in CONTAINMENT_K}
    containment: Counter[int] = Counter()
    practical: dict[int, dict[str, list[float] | int]] = {
        k: {"contained": 0, "margin": [], "reward": [], "candidateCount": []}
        for k in PROPOSAL_BUDGETS
    }
    pairwise_correct = 0.0
    pairwise_total = 0
    gold_tactical: list[np.ndarray] = []
    top1_tactical: list[np.ndarray] = []
    rb_tactical: list[np.ndarray] = []
    failures: list[dict[str, object]] = []
    by_suit: dict[str, list[int]] = defaultdict(list)
    by_target: dict[str, list[int]] = defaultdict(list)
    by_called_location: dict[str, list[int]] = defaultdict(list)
    by_original_point_count: dict[str, list[int]] = defaultdict(list)
    by_kitty_point_count: dict[str, list[int]] = defaultdict(list)
    by_original_trump_count: dict[str, list[int]] = defaultdict(list)
    by_kitty_trump_count: dict[str, list[int]] = defaultdict(list)
    source_diagnostics = audit.manifest["sourceDiagnostics"]

    for group_index in range(audit.group_count):
        source_index = group_index // ADJUTANT_COUNT
        adjutant_index = group_index % ADJUTANT_COUNT
        source = source_diagnostics[source_index]
        margins = np.asarray(audit.contract_margins[group_index], dtype=np.float64)
        rewards = np.asarray(audit.relative_rewards[group_index], dtype=np.float64)
        group_scores = np.asarray(scores[group_index], dtype=np.float64)
        order = np.lexsort((np.arange(DISCARD_COUNT), -group_scores))
        gold = int(audit.gold_candidates[group_index])
        rb = int(audit.rule_based_candidates[group_index])
        rank = int(np.flatnonzero(order == gold)[0]) + 1
        ranks.append(rank)
        by_suit[str(source["contractSuit"])].append(rank)
        by_target[str(source["contractTarget"])].append(rank)
        top1 = int(order[0])
        gold_margin = float(margins[gold])
        gold_reward = float(rewards[gold])
        top1_margin_regrets.append(gold_margin - float(margins[top1]))
        top1_reward_regrets.append(gold_reward - float(rewards[top1]))
        rule_based_margin_regrets.append(gold_margin - float(margins[rb]))
        rule_based_reward_regrets.append(gold_reward - float(rewards[rb]))

        state = np.asarray(audit.state_features[group_index])
        called_index = int(np.argmax(state[106:159]))
        called_location = (
            "originalHand"
            if state[called_index] == 1.0
            else "kitty"
            if state[53 + called_index] == 1.0
            else "opponentHand"
        )
        point_indices = [
            index
            for index, card_id in enumerate(EXPECTED_CARD_IDS)
            if card_id.split("-")[-1] in {"A", "K", "Q", "J", "10"}
        ]
        suit_index = int(np.argmax(state[159:163]))
        trump_slice = slice(suit_index * 13, suit_index * 13 + 13)
        by_called_location[called_location].append(rank)
        by_original_point_count[str(int(state[point_indices].sum()))].append(rank)
        by_kitty_point_count[str(int(state[53 + np.asarray(point_indices)].sum()))].append(rank)
        by_original_trump_count[str(int(state[trump_slice].sum()))].append(rank)
        by_kitty_trump_count[
            str(int(state[53 + trump_slice.start : 53 + trump_slice.stop].sum()))
        ].append(rank)

        compact = np.concatenate(
            (
                np.broadcast_to(
                    np.asarray(audit.state_features[group_index]), (3, STATE_FEATURE_COUNT)
                ),
                np.asarray(audit.candidate_masks[group_index, [gold, top1, rb]], dtype=np.float32),
            ),
            axis=1,
        )
        tactical = compact396_tactical_features(compact)
        gold_tactical.append(tactical[0])
        top1_tactical.append(tactical[1])
        rb_tactical.append(tactical[2])

        for k in CONTAINMENT_K:
            selected = order[:k]
            containment[k] += int(gold in selected)
            best = _best_candidate(selected, margins, rewards)
            oracle_margin[k].append(gold_margin - float(margins[best]))
            oracle_reward[k].append(gold_reward - float(rewards[best]))
        local_source_index = int(source["shardSourceIndex"])
        for k in PROPOSAL_BUDGETS:
            selected = _practical_indices(
                order[:k],
                rb,
                state_index=local_source_index,
                adjutant_index=adjutant_index,
                diversity_count=8,
            )
            practical[k]["contained"] = int(practical[k]["contained"]) + int(gold in selected)
            best = _best_candidate(selected, margins, rewards)
            practical[k]["margin"].append(gold_margin - float(margins[best]))  # type: ignore[union-attr]
            practical[k]["reward"].append(gold_reward - float(rewards[best]))  # type: ignore[union-attr]
            practical[k]["candidateCount"].append(len(selected))  # type: ignore[union-attr]

        truth_delta = margins[:, None] - margins[None, :]
        score_delta = group_scores[:, None] - group_scores[None, :]
        upper = np.triu(np.ones((DISCARD_COUNT, DISCARD_COUNT), dtype=bool), k=1)
        valid = upper & (truth_delta != 0.0)
        pairwise_total += int(valid.sum())
        pairwise_correct += float(
            np.sum(
                np.where(
                    score_delta[valid] == 0.0,
                    0.5,
                    np.sign(score_delta[valid]) == np.sign(truth_delta[valid]),
                )
            )
        )
        if rank > 16:
            failures.append(
                {
                    "groupIndex": group_index,
                    "sourceIndex": source_index,
                    "seed": source["seed"],
                    "adjutantIndex": adjutant_index,
                    "goldRank": rank,
                    "goldMargin": gold_margin,
                    "top1Margin": float(margins[top1]),
                    "ruleBasedMargin": float(margins[rb]),
                    "goldCandidate": gold,
                    "top1Candidate": top1,
                    "ruleBasedCandidate": rb,
                }
            )

    rank_values = np.asarray(ranks, dtype=np.float64)
    gold_tactical_array = np.stack(gold_tactical)
    top1_tactical_array = np.stack(top1_tactical)
    rb_tactical_array = np.stack(rb_tactical)
    return {
        "scorer": scorer_name,
        "fixedHoldout": audit.manifest["fixedHoldout"],
        "groupCount": audit.group_count,
        "issue446Baseline": ISSUE446_BASELINE,
        "containment": {str(k): containment[k] / audit.group_count for k in CONTAINMENT_K},
        "goldBestRank": {
            **_summary(rank_values),
            "exact": float(np.mean(rank_values == 1)),
            "top3": float(np.mean(rank_values <= 3)),
            "top5": float(np.mean(rank_values <= 5)),
            "bins": _rank_bins(rank_values),
        },
        "pairwiseAccuracy": pairwise_correct / pairwise_total,
        "pairwiseCount": pairwise_total,
        "top1": {
            "marginRegret": _summary(np.asarray(top1_margin_regrets)),
            "relativeRewardRegret": _summary(np.asarray(top1_reward_regrets)),
        },
        "ruleBased": {
            "marginRegret": _summary(np.asarray(rule_based_margin_regrets)),
            "relativeRewardRegret": _summary(np.asarray(rule_based_reward_regrets)),
        },
        "topKOracle": {
            str(k): {
                "marginRegret": _summary(np.asarray(oracle_margin[k])),
                "relativeRewardRegret": _summary(np.asarray(oracle_reward[k])),
            }
            for k in CONTAINMENT_K
        },
        "practicalProposal": {
            str(k): {
                "containment": int(practical[k]["contained"]) / audit.group_count,
                "marginRegret": _summary(np.asarray(practical[k]["margin"])),
                "relativeRewardRegret": _summary(np.asarray(practical[k]["reward"])),
                "candidateCount": _summary(np.asarray(practical[k]["candidateCount"])),
            }
            for k in PROPOSAL_BUDGETS
        },
        "failureAnalysis": {
            "rankByContractSuit": {
                key: _summary(np.asarray(value)) for key, value in sorted(by_suit.items())
            },
            "rankByContractTarget": {
                key: _summary(np.asarray(value)) for key, value in sorted(by_target.items())
            },
            "rankByCalledAdjutantLocation": {
                key: _summary(np.asarray(value))
                for key, value in sorted(by_called_location.items())
            },
            "rankByOriginalPointCardCount": {
                key: _summary(np.asarray(value))
                for key, value in sorted(by_original_point_count.items())
            },
            "rankByKittyPointCardCount": {
                key: _summary(np.asarray(value))
                for key, value in sorted(by_kitty_point_count.items())
            },
            "rankByOriginalTrumpCount": {
                key: _summary(np.asarray(value))
                for key, value in sorted(by_original_trump_count.items())
            },
            "rankByKittyTrumpCount": {
                key: _summary(np.asarray(value))
                for key, value in sorted(by_kitty_trump_count.items())
            },
            "tacticalFeatureNames": list(
                (
                    "buriedPointCardCount/3",
                    "buriedTrumpCount/3",
                    "remainingTrumpCount/13",
                    "retainedPointCardCount/13",
                    "buriedJoker",
                    "buriedOruma",
                    "buriedSeiJack",
                    "buriedUraJack",
                    "buriedYoromeki",
                    "buriedCalledAdjutant",
                )
            ),
            "goldBestTacticalMean": gold_tactical_array.mean(axis=0).tolist(),
            "scorerTop1TacticalMean": top1_tactical_array.mean(axis=0).tolist(),
            "ruleBasedTacticalMean": rb_tactical_array.mean(axis=0).tolist(),
            "goldBestTactical": _tactical_summary(gold_tactical_array),
            "scorerTop1Tactical": _tactical_summary(top1_tactical_array),
            "ruleBasedTactical": _tactical_summary(rb_tactical_array),
            "goldMinusTop1TacticalMean": (gold_tactical_array - top1_tactical_array)
            .mean(axis=0)
            .tolist(),
            "goldMinusRuleBasedTacticalMean": (gold_tactical_array - rb_tactical_array)
            .mean(axis=0)
            .tolist(),
            "goldVsRuleBasedExact": float(
                np.mean(
                    np.asarray(audit.gold_candidates) == np.asarray(audit.rule_based_candidates)
                )
            ),
            "worstMisses": sorted(
                failures, key=lambda row: (-int(row["goldRank"]), int(row["groupIndex"]))
            )[:100],
        },
    }


def audit_training_leakage_report(
    audit: ExchangeFullGoldAudit,
    training_dataset: ExchangeCounterfactualDataset,
) -> dict[str, object]:
    """Check raw and encoded source identities across training and fixed audit."""

    first_training_samples = {}
    for sample in training_dataset.raw_samples:
        first_training_samples.setdefault(sample.source_state_key, sample)
    audit_sources = audit.manifest["sourceDiagnostics"]
    audit_seeds = {int(source["seed"]) for source in audit_sources}
    training_seeds = {sample.deal_seed for sample in first_training_samples.values()}
    audit_hidden = {str(source["hiddenDealChecksum"]) for source in audit_sources}
    training_hidden = {sample.hidden_deal_checksum for sample in first_training_samples.values()}
    audit_bidding = {str(source["biddingHistoryHash"]) for source in audit_sources}
    training_bidding = {sample.bidding_history_hash for sample in first_training_samples.values()}
    audit_original = {
        tuple(np.flatnonzero(audit.state_features[index * ADJUTANT_COUNT, :53]))
        for index in range(len(audit_sources))
    }
    training_original = {
        tuple(np.flatnonzero(sample.compact_exchange_state_input[:53]))
        for sample in first_training_samples.values()
        if sample.compact_exchange_state_input is not None
    }
    audit_kitty = {
        tuple(np.flatnonzero(audit.state_features[index * ADJUTANT_COUNT, 53:106]))
        for index in range(len(audit_sources))
    }
    training_kitty = {
        tuple(np.flatnonzero(sample.compact_exchange_state_input[53:106]))
        for sample in first_training_samples.values()
        if sample.compact_exchange_state_input is not None
    }

    audit_visible = {
        _visible_signature(audit.state_features[index * ADJUTANT_COUNT])
        for index in range(len(audit_sources))
    }
    training_visible = {
        _visible_signature(sample.compact_exchange_state_input)
        for sample in first_training_samples.values()
        if sample.compact_exchange_state_input is not None
    }
    overlaps = {
        "dealSeed": len(audit_seeds & training_seeds),
        "hiddenDealChecksum": len(audit_hidden & training_hidden),
        "originalHandIdentity": len(audit_original & training_original),
        "kittyIdentity": len(audit_kitty & training_kitty),
        "biddingHistoryIdentity": len(audit_bidding & training_bidding),
        "visibleSourceIdentity": len(audit_visible & training_visible),
    }
    return {
        "status": "passed" if all(value == 0 for value in overlaps.values()) else "failed",
        "trainingSourceStateCount": len(first_training_samples),
        "auditSourceStateCount": len(audit_sources),
        "crossDatasetOverlapCount": overlaps,
        "note": (
            "visibleSourceIdentity hashes original hand, kitty, contract, starter, and the "
            "compact public bidding-history table after masking the called-card slot."
        ),
    }


def exclude_audit_overlaps(
    audit: ExchangeFullGoldAudit,
    training_dataset: ExchangeCounterfactualDataset,
) -> tuple[ExchangeCounterfactualDataset, dict[str, object]]:
    """Remove complete training groups sharing any guarded identity with the audit."""

    audit_sources = audit.manifest["sourceDiagnostics"]
    guarded = {
        "dealSeed": {int(source["seed"]) for source in audit_sources},
        "hiddenDealChecksum": {str(source["hiddenDealChecksum"]) for source in audit_sources},
        "biddingHistoryIdentity": {str(source["biddingHistoryHash"]) for source in audit_sources},
        "originalHandIdentity": {
            tuple(np.flatnonzero(audit.state_features[index * ADJUTANT_COUNT, :53]))
            for index in range(len(audit_sources))
        },
        "kittyIdentity": {
            tuple(np.flatnonzero(audit.state_features[index * ADJUTANT_COUNT, 53:106]))
            for index in range(len(audit_sources))
        },
        "visibleSourceIdentity": {
            _visible_signature(audit.state_features[index * ADJUTANT_COUNT])
            for index in range(len(audit_sources))
        },
    }
    first_samples = {}
    for sample in training_dataset.raw_samples:
        first_samples.setdefault(sample.source_state_key, sample)
    excluded: dict[str, list[str]] = {}
    reason_counts: Counter[str] = Counter()
    for key, sample in first_samples.items():
        identities = {
            "dealSeed": sample.deal_seed,
            "hiddenDealChecksum": sample.hidden_deal_checksum,
            "biddingHistoryIdentity": sample.bidding_history_hash,
            "originalHandIdentity": tuple(np.flatnonzero(sample.compact_exchange_state_input[:53])),
            "kittyIdentity": tuple(np.flatnonzero(sample.compact_exchange_state_input[53:106])),
            "visibleSourceIdentity": _visible_signature(sample.compact_exchange_state_input),
        }
        reasons = [name for name, value in identities.items() if value in guarded[name]]
        if reasons:
            excluded[key] = reasons
            reason_counts.update(reasons)
    retained = tuple(
        sample for sample in training_dataset.raw_samples if sample.source_state_key not in excluded
    )
    report = {
        "status": "passed",
        "inputSourceStateCount": training_dataset.source_state_count,
        "retainedSourceStateCount": len(first_samples) - len(excluded),
        "excludedSourceStateCount": len(excluded),
        "excludedByIdentity": dict(sorted(reason_counts.items())),
        "excludedSourceStateKeys": sorted(excluded),
    }
    manifest = {
        **training_dataset.manifest,
        "sourceStateCount": report["retainedSourceStateCount"],
        "sampleCount": len(retained),
        "fixedAuditExclusion": report,
    }
    return (
        ExchangeCounterfactualDataset(
            directory=training_dataset.directory,
            manifest=manifest,
            raw_samples=retained,
        ),
        report,
    )


def _visible_signature(values: np.ndarray) -> str:
    source = np.asarray(values, dtype="<f4").copy()
    source[106:159] = 0.0
    return hashlib.sha256(source.tobytes()).hexdigest()


def _best_candidate(indices: np.ndarray, margins: np.ndarray, rewards: np.ndarray) -> int:
    return int(indices[np.lexsort((indices, -rewards[indices], -margins[indices]))[0]])


def _practical_indices(
    top: np.ndarray,
    rule_based: int,
    *,
    state_index: int,
    adjutant_index: int,
    diversity_count: int,
) -> np.ndarray:
    selected = list(dict.fromkeys([*(int(value) for value in top), rule_based]))
    excluded = set(selected)
    constant = (((state_index + 17) * 2246822519) & 0xFFFFFFFF) ^ (
        ((adjutant_index + 31) * 3266489917) & 0xFFFFFFFF
    )
    remaining = [index for index in range(DISCARD_COUNT) if index not in excluded]
    remaining.sort(
        key=lambda index: (
            (((index + 1) * 2654435761) & 0xFFFFFFFF) ^ constant,
            index,
        )
    )
    selected.extend(remaining[:diversity_count])
    return np.asarray(selected, dtype=np.int64)


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def _tactical_summary(features: np.ndarray) -> dict[str, object]:
    point_counts = np.rint(features[:, 0] * 3.0).astype(np.int64)
    trump_counts = np.rint(features[:, 1] * 3.0).astype(np.int64)
    return {
        "buriedPointCardCount": {
            "mean": float(np.mean(point_counts)),
            "distribution": {str(value): int(np.sum(point_counts == value)) for value in range(4)},
        },
        "buriedTrumpCount": {
            "mean": float(np.mean(trump_counts)),
            "distribution": {str(value): int(np.sum(trump_counts == value)) for value in range(4)},
        },
        "buriedSpecialRate": {
            "joker": float(np.mean(features[:, 4])),
            "oruma": float(np.mean(features[:, 5])),
            "seiJack": float(np.mean(features[:, 6])),
            "uraJack": float(np.mean(features[:, 7])),
            "yoromeki": float(np.mean(features[:, 8])),
            "calledAdjutant": float(np.mean(features[:, 9])),
        },
    }


def _rank_bins(ranks: np.ndarray) -> dict[str, int]:
    boundaries = ((1, 1), (2, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 286))
    return {
        f"{left}-{right}": int(np.sum((ranks >= left) & (ranks <= right)))
        for left, right in boundaries
    }
