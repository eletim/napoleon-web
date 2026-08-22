"""Loader and aggregation for Issue #368 bidding Q counterfactual datasets."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from napoleon_ml.dataset.constants import (
    BIDDING_ACTION_COUNT,
    BIDDING_HISTORY_SUIT_ORDER,
    MAX_BIDDING_TARGET_POINT_CARDS,
    MIN_BIDDING_TARGET_POINT_CARDS,
)
from napoleon_ml.dataset.tensors import (
    BIDDING_MODEL_INPUT_FEATURE_COUNT,
    BIDDING_MODEL_INPUT_SCHEMA_VERSION,
)
from napoleon_ml.dataset.validation import calculate_card_ids_sha256

BIDDING_Q_DATASET_SAMPLE_TYPE = "bidding-q-monte-carlo-counterfactual-sample"
BIDDING_Q_DATASET_SCHEMA_VERSION = 2
BIDDING_Q_SUPPORTED_DATASET_SCHEMA_VERSIONS = (1, 2)
BIDDING_Q_SAMPLE_SCHEMA_VERSION = 2
BIDDING_Q_SUPPORTED_SAMPLE_SCHEMA_VERSIONS = (1, 2)
BIDDING_Q_ACTION_MAPPING_ID = (
    "bidding-action-index-v1-pass-then-13-19-spades-hearts-diamonds-clubs"
)
BIDDING_Q_REWARD_ID = "bidding-q-contract-result-loss-minus-one-v1"
BIDDING_Q_MODEL_INPUT_SCHEMA_VERSION = BIDDING_MODEL_INPUT_SCHEMA_VERSION
BIDDING_Q_MODEL_INPUT_FEATURE_COUNT = BIDDING_MODEL_INPUT_FEATURE_COUNT
BIDDING_Q_SUITS: tuple[str, ...] = tuple(BIDDING_HISTORY_SUIT_ORDER)
BIDDING_Q_TARGETS: tuple[int, ...] = tuple(
    range(MIN_BIDDING_TARGET_POINT_CARDS, MAX_BIDDING_TARGET_POINT_CARDS + 1)
)
SuitName = Literal["spades", "hearts", "diamonds", "clubs"]


class BiddingQDatasetError(ValueError):
    """Raised when a bidding Q dataset is incompatible or malformed."""


@dataclass(frozen=True)
class BiddingQDatasetManifest:
    dataset_directory: Path
    dataset_schema_version: int
    sample_type: str
    sample_schema_version: int
    sample_count: int
    source_states: int
    forced_state_action_pairs: int
    reward_id: str
    reward_version: int
    action_mapping_id: str
    model_input_feature_count: int
    model_input_schema_version: int
    manifest_sha256: str
    raw: dict[str, object]


@dataclass(frozen=True)
class BiddingQRawSample:
    state_key: str
    model_input: np.ndarray
    legal_bid_mask: np.ndarray
    forced_action_index: int
    terminal_reward: float
    raw_terminal_reward: float
    repeat_index: int
    rollout_seed: int
    source_seed: int
    source_game_seed: int
    candidate_seat_index: int
    acting_player_index: int
    bidding_step: int
    strongest_suit: SuitName
    strongest_suit_score: float
    forced_action_type: Literal["pass", "bid"]
    forced_target_point_cards: int | None
    forced_suit: SuitName | None
    terminal_role: str
    contract_success: bool
    result_type: str
    final_role: str
    candidate_final_team: str | None
    napoleon_side_point_cards: int | None
    coalition_side_point_cards: int | None
    candidate_team_point_cards: int | None
    team_point_cards_regression_mask: bool
    final_declared_target: int | None
    final_declared_suit: SuitName | None
    contract_margin: int | None
    opponent_configuration_key: str | None


@dataclass(frozen=True)
class BiddingQAggregatedExample:
    state_key: str
    model_input: np.ndarray
    legal_bid_mask: np.ndarray
    action_index: int
    target_mean: float
    target_variance: float
    repeat_count: int
    source_seed: int
    candidate_seat_index: int
    acting_player_index: int
    bidding_step: int
    strongest_suit: SuitName
    strongest_suit_score: float
    action_type: Literal["pass", "bid"]
    target_point_cards: int | None
    suit: SuitName | None


@dataclass(frozen=True)
class BiddingQDataset:
    manifest: BiddingQDatasetManifest
    raw_samples: tuple[BiddingQRawSample, ...]
    examples: tuple[BiddingQAggregatedExample, ...]


@dataclass(frozen=True)
class BiddingQSplit:
    train_state_keys: frozenset[str]
    validation_state_keys: frozenset[str]
    train_examples: tuple[BiddingQAggregatedExample, ...]
    validation_examples: tuple[BiddingQAggregatedExample, ...]
    train_ratio: float
    validation_ratio: float
    seed: int


def load_bidding_q_dataset(
    dataset_directory: Path | str,
    *,
    verify_integrity: bool = True,
) -> BiddingQDataset:
    """Load and aggregate an Issue #368 bidding Q counterfactual dataset."""

    directory = Path(dataset_directory)
    manifest = load_bidding_q_manifest(directory)
    raw_samples = tuple(
        iter_bidding_q_raw_samples(directory, manifest, verify_integrity=verify_integrity)
    )
    if len(raw_samples) != manifest.sample_count:
        raise BiddingQDatasetError(
            f"sample count mismatch: manifest={manifest.sample_count}, loaded={len(raw_samples)}."
        )
    examples = tuple(aggregate_bidding_q_samples(raw_samples))
    if len(examples) != manifest.forced_state_action_pairs:
        raise BiddingQDatasetError(
            "forced state-action pair count mismatch: "
            f"manifest={manifest.forced_state_action_pairs}, aggregated={len(examples)}."
        )
    state_count = len({example.state_key for example in examples})
    if state_count != manifest.source_states:
        raise BiddingQDatasetError(
            "source state count mismatch: "
            f"manifest={manifest.source_states}, aggregated={state_count}."
        )
    return BiddingQDataset(manifest=manifest, raw_samples=raw_samples, examples=examples)


def load_bidding_q_manifest(dataset_directory: Path | str) -> BiddingQDatasetManifest:
    directory = Path(dataset_directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise BiddingQDatasetError(f"manifest.json not found in {directory}.")
    content = manifest_path.read_bytes()
    try:
        raw = json.loads(content.decode("utf-8"))
    except json.JSONDecodeError as error:
        raise BiddingQDatasetError(f"manifest.json is not valid JSON: {error}") from error
    if not isinstance(raw, dict):
        raise BiddingQDatasetError("manifest.json must contain a JSON object.")
    manifest = _parse_manifest(
        raw,
        dataset_directory=directory,
        manifest_sha256=hashlib.sha256(content).hexdigest(),
    )
    _validate_manifest_files(directory, raw)
    return manifest


def iter_bidding_q_raw_samples(
    dataset_directory: Path | str,
    manifest: BiddingQDatasetManifest | None = None,
    *,
    verify_integrity: bool = True,
) -> Iterable[BiddingQRawSample]:
    directory = Path(dataset_directory)
    resolved_manifest = manifest if manifest is not None else load_bidding_q_manifest(directory)
    raw_shards = _require_list(resolved_manifest.raw.get("shards"), "manifest.shards")
    for shard_index, raw_shard in enumerate(raw_shards):
        if not isinstance(raw_shard, dict):
            raise BiddingQDatasetError(f"manifest.shards[{shard_index}] must be an object.")
        yield from _iter_shard(directory, raw_shard, verify_integrity=verify_integrity)


def aggregate_bidding_q_samples(
    samples: Iterable[BiddingQRawSample],
) -> list[BiddingQAggregatedExample]:
    groups: dict[tuple[str, int], list[BiddingQRawSample]] = defaultdict(list)
    for sample in samples:
        groups[(sample.state_key, sample.forced_action_index)].append(sample)

    examples: list[BiddingQAggregatedExample] = []
    for (state_key, action_index), group in sorted(groups.items(), key=lambda item: item[0]):
        first = group[0]
        for sample in group[1:]:
            if not np.array_equal(sample.model_input, first.model_input):
                raise BiddingQDatasetError(
                    f"modelInput mismatch within repeats for {state_key}:{action_index}."
                )
            if not np.array_equal(sample.legal_bid_mask, first.legal_bid_mask):
                raise BiddingQDatasetError(
                    f"legalBidMask mismatch within repeats for {state_key}:{action_index}."
                )
        rewards = np.asarray([sample.terminal_reward for sample in group], dtype=np.float64)
        examples.append(
            BiddingQAggregatedExample(
                state_key=state_key,
                model_input=first.model_input,
                legal_bid_mask=first.legal_bid_mask,
                action_index=action_index,
                target_mean=float(rewards.mean()),
                target_variance=float(rewards.var(ddof=0)),
                repeat_count=len(group),
                source_seed=first.source_seed,
                candidate_seat_index=first.candidate_seat_index,
                acting_player_index=first.acting_player_index,
                bidding_step=first.bidding_step,
                strongest_suit=first.strongest_suit,
                strongest_suit_score=first.strongest_suit_score,
                action_type=first.forced_action_type,
                target_point_cards=first.forced_target_point_cards,
                suit=first.forced_suit,
            )
        )
    return examples


def create_bidding_q_split(
    examples: Iterable[BiddingQAggregatedExample],
    *,
    train_ratio: float,
    seed: int,
) -> BiddingQSplit:
    if train_ratio <= 0.0 or train_ratio >= 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}.")
    example_tuple = tuple(examples)
    state_keys = sorted({example.state_key for example in example_tuple})
    if len(state_keys) < 2:
        raise ValueError("stateKey split requires at least two states.")
    keyed = sorted(state_keys, key=lambda value: _stable_split_key(seed, value))
    train_count = max(1, min(len(keyed) - 1, int(round(len(keyed) * train_ratio))))
    train_state_keys = frozenset(keyed[:train_count])
    validation_state_keys = frozenset(keyed[train_count:])
    train_examples = tuple(
        example for example in example_tuple if example.state_key in train_state_keys
    )
    validation_examples = tuple(
        example for example in example_tuple if example.state_key in validation_state_keys
    )
    if {example.state_key for example in train_examples} & {
        example.state_key for example in validation_examples
    }:
        raise AssertionError("stateKey split leaked between train and validation.")
    return BiddingQSplit(
        train_state_keys=train_state_keys,
        validation_state_keys=validation_state_keys,
        train_examples=train_examples,
        validation_examples=validation_examples,
        train_ratio=train_ratio,
        validation_ratio=1.0 - train_ratio,
        seed=seed,
    )


def action_target(action_index: int) -> int | None:
    if action_index == 0:
        return None
    _validate_action_index(action_index)
    return MIN_BIDDING_TARGET_POINT_CARDS + (action_index - 1) // len(BIDDING_Q_SUITS)


def action_suit(action_index: int) -> SuitName | None:
    if action_index == 0:
        return None
    _validate_action_index(action_index)
    return BIDDING_Q_SUITS[(action_index - 1) % len(BIDDING_Q_SUITS)]  # type: ignore[return-value]


def _parse_manifest(
    raw: dict[str, object],
    *,
    dataset_directory: Path,
    manifest_sha256: str,
) -> BiddingQDatasetManifest:
    dataset_schema_version = _require_int(
        raw.get("datasetSchemaVersion"), "manifest.datasetSchemaVersion"
    )
    if dataset_schema_version not in BIDDING_Q_SUPPORTED_DATASET_SCHEMA_VERSIONS:
        raise BiddingQDatasetError("bidding Q dataset schema version mismatch.")
    if raw.get("sampleType") != BIDDING_Q_DATASET_SAMPLE_TYPE:
        raise BiddingQDatasetError("bidding Q sample type mismatch.")
    sample_schema_version = _require_int(
        raw.get("sampleSchemaVersion"), "manifest.sampleSchemaVersion"
    )
    if sample_schema_version not in BIDDING_Q_SUPPORTED_SAMPLE_SCHEMA_VERSIONS:
        raise BiddingQDatasetError("bidding Q sample schema version mismatch.")
    compact = _require_dict(raw.get("compactObservation"), "manifest.compactObservation")
    if compact.get("phase") != "bidding":
        raise BiddingQDatasetError("bidding Q compact observation phase mismatch.")
    if compact.get("modelInputFeatureCount") != BIDDING_Q_MODEL_INPUT_FEATURE_COUNT:
        raise BiddingQDatasetError("bidding Q compact278 feature count mismatch.")
    if compact.get("modelInputSchemaVersion") != BIDDING_Q_MODEL_INPUT_SCHEMA_VERSION:
        raise BiddingQDatasetError("bidding Q model input schema version mismatch.")
    action_mapping = _require_dict(raw.get("actionMapping"), "manifest.actionMapping")
    if action_mapping.get("id") != BIDDING_Q_ACTION_MAPPING_ID:
        raise BiddingQDatasetError("bidding Q action mapping mismatch.")
    if action_mapping.get("actionCount") != BIDDING_ACTION_COUNT:
        raise BiddingQDatasetError("bidding Q action count mismatch.")
    reward = _require_dict(raw.get("reward"), "manifest.reward")
    if reward.get("id") != BIDDING_Q_REWARD_ID:
        raise BiddingQDatasetError("bidding Q reward id mismatch.")
    if raw.get("cardIdsSha256") != calculate_card_ids_sha256():
        raise BiddingQDatasetError("bidding Q card id hash mismatch.")
    return BiddingQDatasetManifest(
        dataset_directory=dataset_directory,
        dataset_schema_version=dataset_schema_version,
        sample_type=BIDDING_Q_DATASET_SAMPLE_TYPE,
        sample_schema_version=sample_schema_version,
        sample_count=_require_int(raw.get("sampleCount"), "manifest.sampleCount"),
        source_states=_require_int(raw.get("sourceStates"), "manifest.sourceStates"),
        forced_state_action_pairs=_require_int(
            raw.get("forcedStateActionPairs"), "manifest.forcedStateActionPairs"
        ),
        reward_id=str(reward["id"]),
        reward_version=_require_int(reward.get("version"), "manifest.reward.version"),
        action_mapping_id=str(action_mapping["id"]),
        model_input_feature_count=BIDDING_Q_MODEL_INPUT_FEATURE_COUNT,
        model_input_schema_version=BIDDING_Q_MODEL_INPUT_SCHEMA_VERSION,
        manifest_sha256=manifest_sha256,
        raw=raw,
    )


def _validate_manifest_files(directory: Path, raw: dict[str, object]) -> None:
    expected = {"manifest.json", "summary.json"}
    for shard in _require_list(raw.get("shards"), "manifest.shards"):
        if not isinstance(shard, dict):
            raise BiddingQDatasetError("manifest.shards entries must be objects.")
        file_name = _require_str(shard.get("file"), "manifest.shards[].file")
        if "/" in file_name or file_name.startswith("."):
            raise BiddingQDatasetError(f"unsafe shard filename: {file_name}.")
        expected.add(file_name)
    actual = {path.name for path in directory.iterdir() if path.is_file()}
    if actual != expected:
        raise BiddingQDatasetError(
            f"dataset files mismatch: expected {sorted(expected)}, got {sorted(actual)}."
        )


def _iter_shard(
    directory: Path,
    shard: dict[str, object],
    *,
    verify_integrity: bool,
) -> Iterable[BiddingQRawSample]:
    file_name = _require_str(shard.get("file"), "shard.file")
    path = directory / file_name
    hasher = hashlib.sha256()
    byte_length = 0
    sample_count = 0
    first_seed: int | None = None
    last_seed: int | None = None
    state_keys: set[str] = set()
    with path.open("rb") as file:
        for line_number, raw_line in enumerate(file, start=1):
            context = f"{file_name}:{line_number}"
            hasher.update(raw_line)
            byte_length += len(raw_line)
            if not raw_line.endswith(b"\n"):
                raise BiddingQDatasetError(f"{context}: missing final LF.")
            try:
                raw_sample = json.loads(raw_line.decode("utf-8"))
            except json.JSONDecodeError as error:
                raise BiddingQDatasetError(f"{context}: invalid JSON: {error}") from error
            if not isinstance(raw_sample, dict):
                raise BiddingQDatasetError(f"{context}: sample must be an object.")
            sample = _parse_sample(raw_sample, context=context)
            if first_seed is None:
                first_seed = sample.source_seed
            last_seed = sample.source_seed
            state_keys.add(sample.state_key)
            sample_count += 1
            yield sample
    if sample_count != _require_int(shard.get("sampleCount"), "shard.sampleCount"):
        raise BiddingQDatasetError(f"{file_name}: sampleCount mismatch.")
    if first_seed != _require_int(shard.get("startSeed"), "shard.startSeed"):
        raise BiddingQDatasetError(f"{file_name}: startSeed mismatch.")
    if last_seed != _require_int(shard.get("endSeed"), "shard.endSeed"):
        raise BiddingQDatasetError(f"{file_name}: endSeed mismatch.")
    if len(state_keys) != _require_int(shard.get("gameCount"), "shard.gameCount"):
        raise BiddingQDatasetError(f"{file_name}: gameCount/stateKey count mismatch.")
    if verify_integrity:
        if byte_length != _require_int(shard.get("byteLength"), "shard.byteLength"):
            raise BiddingQDatasetError(f"{file_name}: byteLength mismatch.")
        if hasher.hexdigest() != _require_str(shard.get("sha256"), "shard.sha256"):
            raise BiddingQDatasetError(f"{file_name}: sha256 mismatch.")


def _parse_sample(raw: dict[str, object], *, context: str) -> BiddingQRawSample:
    if raw.get("sampleType") != BIDDING_Q_DATASET_SAMPLE_TYPE:
        raise BiddingQDatasetError(f"{context}: sampleType mismatch.")
    sample_schema_version = _require_int(raw.get("schemaVersion"), f"{context}.schemaVersion")
    if sample_schema_version not in BIDDING_Q_SUPPORTED_SAMPLE_SCHEMA_VERSIONS:
        raise BiddingQDatasetError(f"{context}: schemaVersion mismatch.")
    model_input = np.asarray(
        _require_number_list(raw.get("modelInput"), f"{context}.modelInput"),
        dtype=np.float32,
    )
    if model_input.shape != (BIDDING_Q_MODEL_INPUT_FEATURE_COUNT,):
        raise BiddingQDatasetError(f"{context}: modelInput must have length 278.")
    legal_bid_mask = np.asarray(
        _require_number_list(raw.get("legalBidMask"), f"{context}.legalBidMask"),
        dtype=np.float32,
    )
    if legal_bid_mask.shape != (BIDDING_ACTION_COUNT,):
        raise BiddingQDatasetError(f"{context}: legalBidMask must have length 29.")
    action_index = _require_int(raw.get("forcedActionIndex"), f"{context}.forcedActionIndex")
    _validate_action_index(action_index)
    if legal_bid_mask[action_index] != 1.0:
        raise BiddingQDatasetError(f"{context}: forced action is illegal.")
    forced_action = _require_dict(raw.get("forcedAction"), f"{context}.forcedAction")
    action_type = _require_str(forced_action.get("type"), f"{context}.forcedAction.type")
    if action_type not in ("pass", "bid"):
        raise BiddingQDatasetError(f"{context}: forced action type mismatch.")
    semantic_target = action_target(action_index)
    semantic_suit = action_suit(action_index)
    if action_type == "pass" and action_index != 0:
        raise BiddingQDatasetError(f"{context}: PASS semantic/action index mismatch.")
    if action_type == "bid":
        if _require_int(
            forced_action.get("targetPointCards"), f"{context}.forcedAction.targetPointCards"
        ) != semantic_target:
            raise BiddingQDatasetError(f"{context}: forced target mismatch.")
        if _require_str(forced_action.get("suit"), f"{context}.forcedAction.suit") != semantic_suit:
            raise BiddingQDatasetError(f"{context}: forced suit mismatch.")
    reward = _require_float(raw.get("terminalReward"), f"{context}.terminalReward")
    raw_reward = _require_float(raw.get("rawTerminalReward"), f"{context}.rawTerminalReward")
    if not math.isclose(reward, raw_reward, rel_tol=0.0, abs_tol=0.0):
        raise BiddingQDatasetError(f"{context}: reward transform must be identity.")
    strongest_suit = _require_suit(raw.get("strongestSuit"), f"{context}.strongestSuit")
    provenance = _require_dict(raw.get("provenance"), f"{context}.provenance")
    if provenance.get("replayMatchedModelInput") is not True:
        raise BiddingQDatasetError(f"{context}: replay modelInput parity missing.")
    if provenance.get("replayMatchedLegalBidMask") is not True:
        raise BiddingQDatasetError(f"{context}: replay legalBidMask parity missing.")
    if provenance.get("forcedOnce") is not True:
        raise BiddingQDatasetError(f"{context}: forcedOnce missing.")
    terminal_role = _require_str(raw.get("terminalRole"), f"{context}.terminalRole")
    final_role = (
        _require_str(raw.get("finalRole"), f"{context}.finalRole")
        if sample_schema_version >= 2
        else terminal_role
    )
    if final_role != terminal_role:
        raise BiddingQDatasetError(f"{context}: finalRole must match terminalRole.")
    candidate_final_team = (
        _require_str(raw.get("candidateFinalTeam"), f"{context}.candidateFinalTeam")
        if sample_schema_version >= 2
        else None
    )
    team_point_cards_regression_mask = (
        _require_bool(
            raw.get("teamPointCardsRegressionMask"),
            f"{context}.teamPointCardsRegressionMask",
        )
        if sample_schema_version >= 2
        else False
    )
    return BiddingQRawSample(
        state_key=_require_str(raw.get("stateKey"), f"{context}.stateKey"),
        model_input=model_input,
        legal_bid_mask=legal_bid_mask,
        forced_action_index=action_index,
        terminal_reward=reward,
        raw_terminal_reward=raw_reward,
        repeat_index=_require_int(raw.get("repeatIndex"), f"{context}.repeatIndex"),
        rollout_seed=_require_int(raw.get("rolloutSeed"), f"{context}.rolloutSeed"),
        source_seed=_require_int(raw.get("sourceSeed"), f"{context}.sourceSeed"),
        source_game_seed=_require_int(raw.get("sourceGameSeed"), f"{context}.sourceGameSeed"),
        candidate_seat_index=_require_int(
            raw.get("candidateSeatIndex"), f"{context}.candidateSeatIndex"
        ),
        acting_player_index=_require_int(
            raw.get("actingPlayerIndex"), f"{context}.actingPlayerIndex"
        ),
        bidding_step=_require_int(raw.get("biddingStep"), f"{context}.biddingStep"),
        strongest_suit=strongest_suit,
        strongest_suit_score=_require_float(
            raw.get("strongestSuitScore"), f"{context}.strongestSuitScore"
        ),
        forced_action_type=action_type,  # type: ignore[arg-type]
        forced_target_point_cards=semantic_target,
        forced_suit=semantic_suit,
        terminal_role=terminal_role,
        contract_success=_require_bool(raw.get("contractSuccess"), f"{context}.contractSuccess"),
        result_type=_require_str(raw.get("resultType"), f"{context}.resultType"),
        final_role=final_role,
        candidate_final_team=candidate_final_team,
        napoleon_side_point_cards=_optional_int(
            raw.get("napoleonSidePointCards"), f"{context}.napoleonSidePointCards"
        ),
        coalition_side_point_cards=_optional_int(
            raw.get("coalitionSidePointCards"), f"{context}.coalitionSidePointCards"
        ),
        candidate_team_point_cards=_optional_int(
            raw.get("candidateTeamPointCards"), f"{context}.candidateTeamPointCards"
        ),
        team_point_cards_regression_mask=team_point_cards_regression_mask,
        final_declared_target=_optional_int(
            raw.get("finalDeclaredTarget"), f"{context}.finalDeclaredTarget"
        ),
        final_declared_suit=_optional_suit(
            raw.get("finalDeclaredSuit"), f"{context}.finalDeclaredSuit"
        ),
        contract_margin=_optional_int(raw.get("contractMargin"), f"{context}.contractMargin"),
        opponent_configuration_key=(
            _require_str(
                raw.get("opponentConfigurationKey"), f"{context}.opponentConfigurationKey"
            )
            if sample_schema_version >= 2
            else None
        ),
    )


def _stable_split_key(seed: int, state_key: str) -> str:
    return hashlib.sha256(f"{seed}:{state_key}".encode()).hexdigest()


def _validate_action_index(action_index: int) -> None:
    if action_index < 0 or action_index >= BIDDING_ACTION_COUNT:
        raise BiddingQDatasetError(f"action index out of range: {action_index}.")


def _require_dict(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise BiddingQDatasetError(f"{path} must be an object.")
    return value


def _require_list(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise BiddingQDatasetError(f"{path} must be a list.")
    return value


def _require_number_list(value: object, path: str) -> list[float]:
    items = _require_list(value, path)
    result: list[float] = []
    for index, item in enumerate(items):
        if isinstance(item, bool) or not isinstance(item, int | float) or not math.isfinite(item):
            raise BiddingQDatasetError(f"{path}[{index}] must be a finite number.")
        result.append(float(item))
    return result


def _require_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BiddingQDatasetError(f"{path} must be an integer.")
    return value


def _optional_int(value: object, path: str) -> int | None:
    if value is None:
        return None
    return _require_int(value, path)


def _require_float(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise BiddingQDatasetError(f"{path} must be a finite number.")
    return float(value)


def _require_str(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise BiddingQDatasetError(f"{path} must be a string.")
    return value


def _require_bool(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise BiddingQDatasetError(f"{path} must be a boolean.")
    return value


def _require_suit(value: object, path: str) -> SuitName:
    if value not in BIDDING_Q_SUITS:
        raise BiddingQDatasetError(f"{path} must be one of {BIDDING_Q_SUITS}.")
    return value  # type: ignore[return-value]


def _optional_suit(value: object, path: str) -> SuitName | None:
    if value is None:
        return None
    return _require_suit(value, path)
