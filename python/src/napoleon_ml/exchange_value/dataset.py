"""Loader and group split guard for exchange counterfactual value datasets."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from napoleon_ml.dataset.constants import CARD_COUNT, EXPECTED_CARD_IDS
from napoleon_ml.dataset.tensors import EXCHANGE_MODEL_INPUT_FEATURE_COUNT

EXCHANGE_COUNTERFACTUAL_SAMPLE_TYPE = "exchange-counterfactual-value-v1"
EXCHANGE_COUNTERFACTUAL_COMBINATION_COUNT = 286
EXCHANGE_VALUE_INPUT_FEATURE_COUNT = EXCHANGE_MODEL_INPUT_FEATURE_COUNT + CARD_COUNT
EXCHANGE_COMPACT_STATE_FEATURE_COUNT = 343
EXCHANGE_COMPACT_VALUE_INPUT_FEATURE_COUNT = EXCHANGE_COMPACT_STATE_FEATURE_COUNT + CARD_COUNT
EXCHANGE_VALUE_INPUT_VARIANTS = ("legacy2724", "compact396")
ExchangeValueInputVariant = str


@dataclass(frozen=True)
class ExchangeCounterfactualSample:
    source_state_key: str
    fixed_hand_id: str
    deal_seed: int
    source_index: int
    candidate_index: int
    candidate_key: str
    original_hand_card_ids: tuple[str, ...] | None
    kitty_pickup_card_ids: tuple[str, ...] | None
    pickup_hand_card_ids: tuple[str, ...]
    model_input: np.ndarray
    compact_exchange_state_input: np.ndarray | None
    legal_discard_card_mask: np.ndarray
    candidate_discard_card_ids: tuple[str, ...]
    candidate_discard_mask: np.ndarray
    buried_point_card_count: int
    buried_trump_count: int
    buried_special_cards: dict[str, bool]
    contract_margin: float
    contract_success: bool
    napoleon_relative_reward: float
    is_rule_based_action: bool
    rule_based_candidate_index: int
    hidden_deal_checksum: str

    @property
    def value_input(self) -> np.ndarray:
        return np.concatenate((self.model_input, self.candidate_discard_mask)).astype(np.float32)

    @property
    def compact_value_input(self) -> np.ndarray:
        if self.compact_exchange_state_input is None:
            raise ValueError(
                "compact396 input is unavailable; regenerate the exchange counterfactual "
                "dataset with compactExchangeStateInput."
            )
        return np.concatenate(
            (self.compact_exchange_state_input, self.candidate_discard_mask)
        ).astype(np.float32)

    @property
    def pickup_hand_key(self) -> str:
        return "|".join(self.pickup_hand_card_ids)

    def value_input_for_variant(self, input_variant: ExchangeValueInputVariant) -> np.ndarray:
        if input_variant == "legacy2724":
            return self.value_input
        if input_variant == "compact396":
            return self.compact_value_input
        raise ValueError(f"unsupported exchange value input variant: {input_variant}.")


@dataclass(frozen=True)
class ExchangeCounterfactualDataset:
    directory: Path
    manifest: dict[str, Any]
    raw_samples: tuple[ExchangeCounterfactualSample, ...]

    @property
    def source_state_count(self) -> int:
        return len({sample.source_state_key for sample in self.raw_samples})

    @property
    def sample_count(self) -> int:
        return len(self.raw_samples)


@dataclass(frozen=True)
class ExchangeValueSplit:
    train_state_keys: frozenset[str]
    validation_state_keys: frozenset[str]
    final_state_keys: frozenset[str]
    train_samples: tuple[ExchangeCounterfactualSample, ...]
    validation_samples: tuple[ExchangeCounterfactualSample, ...]
    final_samples: tuple[ExchangeCounterfactualSample, ...]
    train_state_key_hash: str
    validation_state_key_hash: str
    final_state_key_hash: str
    leakage_guard: dict[str, object]


def load_exchange_counterfactual_dataset(
    dataset_directory: Path | str,
) -> ExchangeCounterfactualDataset:
    directory = Path(dataset_directory)
    manifest_path = directory / "manifest.json"
    manifest = _read_json_object(manifest_path, context="manifest.json")
    _validate_manifest_shape(manifest)
    samples: list[ExchangeCounterfactualSample] = []
    for shard in manifest["shards"]:
        shard_path = directory / _require_str(shard, "file")
        expected_sha = _require_str(shard, "sha256")
        expected_bytes = _require_int(shard, "byteLength")
        expected_samples = _require_int(shard, "sampleCount")
        content = shard_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected_sha:
            raise ValueError(f"{shard_path}: SHA-256 mismatch.")
        if len(content) != expected_bytes:
            raise ValueError(f"{shard_path}: byte length mismatch.")
        shard_rows = content.splitlines()
        if len(shard_rows) != expected_samples:
            raise ValueError(
                f"{shard_path}: sample count mismatch: "
                f"expected {expected_samples}, got {len(shard_rows)}."
            )
        for line_number, line in enumerate(shard_rows, start=1):
            if not line:
                raise ValueError(f"{shard_path}:{line_number}: empty line.")
            raw = json.loads(line.decode("utf-8"))
            if not isinstance(raw, dict):
                raise ValueError(f"{shard_path}:{line_number}: sample must be an object.")
            samples.append(_parse_sample(raw, context=f"{shard_path.name}:{line_number}"))
    dataset = ExchangeCounterfactualDataset(
        directory=directory,
        manifest=manifest,
        raw_samples=tuple(samples),
    )
    _validate_dataset(dataset)
    return dataset


def create_exchange_value_split(
    dataset: ExchangeCounterfactualDataset,
    *,
    seed: int,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    final_ratio: float = 0.1,
    train_state_count: int | None = None,
) -> ExchangeValueSplit:
    if train_ratio <= 0.0 or validation_ratio <= 0.0 or final_ratio <= 0.0:
        raise ValueError("split ratios must all be positive.")
    total = train_ratio + validation_ratio + final_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1.0, got {total}.")
    state_keys = _state_keys_in_source_order(dataset.raw_samples)
    components = _identity_components(dataset.raw_samples, state_keys)
    shuffled_components = sorted(
        components,
        key=lambda keys: hashlib.sha256(
            f"{seed}:{','.join(sorted(keys))}".encode()
        ).hexdigest(),
    )
    validation_count = max(1, int(round(len(state_keys) * validation_ratio)))
    final_count = max(1, int(round(len(state_keys) * final_ratio)))
    if validation_count + final_count >= len(state_keys):
        raise ValueError("dataset is too small for non-empty train/validation/final splits.")
    validation_set, remaining_components = _take_components(shuffled_components, validation_count)
    final_set, train_components = _take_components(remaining_components, final_count)
    if not train_components:
        raise ValueError("identity grouping left no train components.")
    all_train_keys = {key for component in train_components for key in component}
    train_candidates = [key for key in state_keys if key not in validation_set | final_set]
    if set(train_candidates) != all_train_keys:
        raise AssertionError("train component accounting mismatch.")
    selected_train_count = train_state_count or len(train_candidates)
    if selected_train_count <= 0 or selected_train_count > len(train_candidates):
        raise ValueError(
            f"train_state_count must be in [1,{len(train_candidates)}], "
            f"got {selected_train_count}."
        )
    train_set = frozenset(train_candidates[:selected_train_count])
    if train_set & validation_set or train_set & final_set or validation_set & final_set:
        raise AssertionError("sourceStateKey leakage between splits.")
    samples_by_split = {
        "train": tuple(s for s in dataset.raw_samples if s.source_state_key in train_set),
        "validation": tuple(
            s for s in dataset.raw_samples if s.source_state_key in validation_set
        ),
        "final": tuple(s for s in dataset.raw_samples if s.source_state_key in final_set),
    }
    guard = _leakage_guard(
        train=samples_by_split["train"],
        validation=samples_by_split["validation"],
        final=samples_by_split["final"],
    )
    return ExchangeValueSplit(
        train_state_keys=train_set,
        validation_state_keys=validation_set,
        final_state_keys=final_set,
        train_samples=samples_by_split["train"],
        validation_samples=samples_by_split["validation"],
        final_samples=samples_by_split["final"],
        train_state_key_hash=state_key_hash(train_set),
        validation_state_key_hash=state_key_hash(validation_set),
        final_state_key_hash=state_key_hash(final_set),
        leakage_guard=guard,
    )


def _identity_components(
    samples: tuple[ExchangeCounterfactualSample, ...],
    state_keys: tuple[str, ...],
) -> list[frozenset[str]]:
    parent = {key: key for key in state_keys}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    identity_owner: dict[tuple[str, str], str] = {}
    for sample in samples:
        for identity in (
            ("dealSeed", str(sample.deal_seed)),
            ("hiddenDealChecksum", sample.hidden_deal_checksum),
            ("fixedHandId", sample.fixed_hand_id),
            ("pickupHand", sample.pickup_hand_key),
        ):
            owner = identity_owner.setdefault(identity, sample.source_state_key)
            union(owner, sample.source_state_key)
    components: dict[str, set[str]] = defaultdict(set)
    for key in state_keys:
        components[find(key)].add(key)
    return [frozenset(keys) for keys in components.values()]


def _take_components(
    components: list[frozenset[str]],
    target_state_count: int,
) -> tuple[frozenset[str], list[frozenset[str]]]:
    selected: set[str] = set()
    remaining: list[frozenset[str]] = []
    for component in components:
        if len(selected) < target_state_count:
            selected.update(component)
        else:
            remaining.append(component)
    return frozenset(selected), remaining


def state_key_hash(state_keys: frozenset[str]) -> str:
    return hashlib.sha256("\n".join(sorted(state_keys)).encode("utf-8")).hexdigest()


def dataset_provenance(dataset: ExchangeCounterfactualDataset) -> dict[str, object]:
    manifest = dataset.manifest
    return {
        "path": str(dataset.directory),
        "sampleType": manifest.get("sampleType"),
        "sampleCount": manifest.get("sampleCount"),
        "sourceStateCount": manifest.get("sourceStateCount"),
        "startSeed": manifest.get("startSeed"),
        "endSeed": manifest.get("endSeed"),
        "sourceCommit": manifest.get("sourceCommit"),
        "biddingPolicy": manifest.get("biddingPolicy"),
        "adjutantPolicy": manifest.get("adjutantPolicy"),
        "playingPolicy": manifest.get("playingPolicy"),
        "cardIdsSha256": manifest.get("cardIdsSha256"),
    }


def _read_json_object(path: Path, *, context: str) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be a JSON object.")
    return raw


def _validate_manifest_shape(manifest: dict[str, Any]) -> None:
    if manifest.get("sampleType") != EXCHANGE_COUNTERFACTUAL_SAMPLE_TYPE:
        raise ValueError(
            f"manifest.sampleType must be {EXCHANGE_COUNTERFACTUAL_SAMPLE_TYPE!r}."
        )
    if manifest.get("format") != "jsonl":
        raise ValueError("manifest.format must be jsonl.")
    if manifest.get("discardCombinationCount") != EXCHANGE_COUNTERFACTUAL_COMBINATION_COUNT:
        raise ValueError("manifest.discardCombinationCount must be 286.")
    if manifest.get("cardIds") != list(EXPECTED_CARD_IDS):
        raise ValueError("manifest.cardIds mismatch.")
    if manifest.get("cardCount") != CARD_COUNT:
        raise ValueError("manifest.cardCount mismatch.")
    if manifest.get("modelInput", {}).get("featureCount") != EXCHANGE_MODEL_INPUT_FEATURE_COUNT:
        raise ValueError("manifest.modelInput.featureCount must be 2671.")
    compact = manifest.get("compactExchangeValueInput")
    if compact is not None:
        if not isinstance(compact, dict):
            raise ValueError("manifest.compactExchangeValueInput must be an object.")
        if compact.get("stateFeatureCount") != EXCHANGE_COMPACT_STATE_FEATURE_COUNT:
            raise ValueError("manifest.compactExchangeValueInput.stateFeatureCount must be 343.")
        if compact.get("featureCount") != EXCHANGE_COMPACT_VALUE_INPUT_FEATURE_COUNT:
            raise ValueError("manifest.compactExchangeValueInput.featureCount must be 396.")


def _parse_sample(raw: dict[str, Any], *, context: str) -> ExchangeCounterfactualSample:
    if raw.get("sampleType") != EXCHANGE_COUNTERFACTUAL_SAMPLE_TYPE:
        raise ValueError(f"{context}: sampleType mismatch.")
    model_input = _float_array(raw.get("modelInput"), EXCHANGE_MODEL_INPUT_FEATURE_COUNT, context)
    compact_state_input = _optional_float_array(
        raw.get("compactExchangeStateInput"),
        EXCHANGE_COMPACT_STATE_FEATURE_COUNT,
        context,
        "compactExchangeStateInput",
    )
    discard_mask = _float_array(raw.get("candidateDiscardMask"), CARD_COUNT, context)
    legal_mask = _float_array(raw.get("legalDiscardCardMask"), CARD_COUNT, context)
    if int(discard_mask.sum()) != 3:
        raise ValueError(f"{context}: candidateDiscardMask must contain exactly 3 ones.")
    if not bool(np.all(discard_mask <= legal_mask)):
        raise ValueError(f"{context}: candidateDiscardMask must be legal.")
    return ExchangeCounterfactualSample(
        source_state_key=_require_str(raw, "sourceStateKey"),
        fixed_hand_id=_require_str(raw, "fixedHandId"),
        deal_seed=_require_int(raw, "dealSeed"),
        source_index=_require_int(raw, "sourceIndex"),
        candidate_index=_require_int(raw, "candidateIndex"),
        candidate_key=_require_str(raw, "candidateKey"),
        original_hand_card_ids=_optional_str_tuple(
            raw.get("originalHandCardIds"),
            10,
            context,
            "originalHandCardIds",
        ),
        kitty_pickup_card_ids=_optional_str_tuple(
            raw.get("kittyPickupCardIds"),
            3,
            context,
            "kittyPickupCardIds",
        ),
        pickup_hand_card_ids=_str_tuple(raw.get("pickupHandCardIds"), 13, context),
        model_input=model_input,
        compact_exchange_state_input=compact_state_input,
        legal_discard_card_mask=legal_mask,
        candidate_discard_card_ids=_str_tuple(raw.get("candidateDiscardCardIds"), 3, context),
        candidate_discard_mask=discard_mask,
        buried_point_card_count=_require_int(raw, "buriedPointCardCount"),
        buried_trump_count=_require_int(raw, "buriedTrumpCount"),
        buried_special_cards=_special_flags(raw.get("buriedSpecialCards"), context),
        contract_margin=float(_require_number(raw, "contractMargin")),
        contract_success=_require_bool(raw, "contractSuccess"),
        napoleon_relative_reward=float(_require_number(raw, "napoleonRelativeReward")),
        is_rule_based_action=_require_bool(raw, "isRuleBasedAction"),
        rule_based_candidate_index=_require_int(raw, "ruleBasedCandidateIndex"),
        hidden_deal_checksum=_require_str(raw, "hiddenDealChecksum"),
    )


def _validate_dataset(dataset: ExchangeCounterfactualDataset) -> None:
    if dataset.sample_count != int(dataset.manifest.get("sampleCount", -1)):
        raise ValueError("manifest.sampleCount does not match JSONL rows.")
    by_source: dict[str, list[ExchangeCounterfactualSample]] = defaultdict(list)
    for sample in dataset.raw_samples:
        by_source[sample.source_state_key].append(sample)
    if len(by_source) != int(dataset.manifest.get("sourceStateCount", -1)):
        raise ValueError("manifest.sourceStateCount does not match unique source states.")
    for key, rows in by_source.items():
        if len(rows) != EXCHANGE_COUNTERFACTUAL_COMBINATION_COUNT:
            raise ValueError(f"{key}: expected 286 candidates, got {len(rows)}.")
        candidate_indices = {sample.candidate_index for sample in rows}
        if candidate_indices != set(range(EXCHANGE_COUNTERFACTUAL_COMBINATION_COUNT)):
            raise ValueError(f"{key}: candidate indices must be 0..285.")
        if sum(1 for sample in rows if sample.is_rule_based_action) != 1:
            raise ValueError(f"{key}: expected exactly one RuleBased candidate.")
        compact_inputs = [
            sample.compact_exchange_state_input for sample in rows
            if sample.compact_exchange_state_input is not None
        ]
        if compact_inputs and len(compact_inputs) != len(rows):
            raise ValueError(f"{key}: compactExchangeStateInput must be present for every row.")
        if compact_inputs:
            first = compact_inputs[0]
            if not all(bool(np.array_equal(first, value)) for value in compact_inputs[1:]):
                raise ValueError(f"{key}: compactExchangeStateInput must be source-stable.")


def _state_keys_in_source_order(
    samples: tuple[ExchangeCounterfactualSample, ...],
) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for sample in samples:
        if sample.source_state_key not in seen:
            seen.add(sample.source_state_key)
            result.append(sample.source_state_key)
    return tuple(result)


def _leakage_guard(
    *,
    train: tuple[ExchangeCounterfactualSample, ...],
    validation: tuple[ExchangeCounterfactualSample, ...],
    final: tuple[ExchangeCounterfactualSample, ...],
) -> dict[str, object]:
    split_samples = {"train": train, "validation": validation, "final": final}
    checks: dict[str, object] = {}
    for name, key_fn in (
        ("sourceStateKey", lambda s: s.source_state_key),
        ("dealSeed", lambda s: str(s.deal_seed)),
        ("hiddenDealChecksum", lambda s: s.hidden_deal_checksum),
        ("fixedHandId", lambda s: s.fixed_hand_id),
        ("pickupHand", lambda s: s.pickup_hand_key),
    ):
        owners: dict[str, str] = {}
        duplicate_count = 0
        for split, samples in split_samples.items():
            for sample in samples:
                key = key_fn(sample)
                owner = owners.setdefault(key, split)
                if owner != split:
                    duplicate_count += 1
        if duplicate_count:
            raise ValueError(f"{name} leakage across splits: {duplicate_count} samples.")
        checks[name] = {
            "uniqueCount": len(owners),
            "crossSplitLeakageCount": 0,
        }
    checks["status"] = "passed"
    return checks


def _float_array(value: object, length: int, context: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{context}: expected length {length} numeric array.")
    return np.asarray([float(item) for item in value], dtype=np.float32)


def _optional_float_array(
    value: object,
    length: int,
    context: str,
    field_name: str,
) -> np.ndarray | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{context}: {field_name} must be length {length} numeric array.")
    return np.asarray([float(item) for item in value], dtype=np.float32)


def _str_tuple(value: object, length: int, context: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or len(value) != length
        or not all(isinstance(item, str) for item in value)
    ):
        raise ValueError(f"{context}: expected length {length} string array.")
    return tuple(value)


def _optional_str_tuple(
    value: object,
    length: int,
    context: str,
    field_name: str,
) -> tuple[str, ...] | None:
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or len(value) != length
        or not all(isinstance(item, str) for item in value)
    ):
        raise ValueError(f"{context}: {field_name} must be length {length} string array.")
    return tuple(value)


def _special_flags(value: object, context: str) -> dict[str, bool]:
    keys = ("joker", "oruma", "yoromeki", "seiJack", "uraJack", "calledAdjutant")
    if not isinstance(value, dict):
        raise ValueError(f"{context}: buriedSpecialCards must be an object.")
    return {key: _require_bool(value, key) for key in keys}


def _require_str(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string.")
    return value


def _require_int(raw: dict[str, Any], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer.")
    return value


def _require_number(raw: dict[str, Any], key: str) -> int | float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be a number.")
    return value


def _require_bool(raw: dict[str, Any], key: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean.")
    return value
