"""Playing observation variant metadata shared by RL datasets and policies."""

from __future__ import annotations

from .constants import (
    COMPLETE_INFO_COMPACT_PLAYING_OBSERVATION_VARIANT,
    COMPLETE_INFO_PLAYING_ENCODER_SCHEMA_VERSION,
    COMPLETE_INFO_PLAYING_MODEL_INPUT_FEATURE_COUNT,
    COMPLETE_INFO_PLAYING_MODEL_INPUT_SCHEMA_VERSION,
    MODEL_INPUT_FEATURE_COUNT,
    PLAYING_ENCODER_SCHEMA_VERSION,
    PLAYING_MODEL_INPUT_SCHEMA_VERSION,
    PUBLIC_PLAYING_OBSERVATION_VARIANT,
)

PlayingObservationVariant = str


def normalize_playing_observation_variant(value: object | None) -> str:
    if value is None:
        return PUBLIC_PLAYING_OBSERVATION_VARIANT
    if value in {
        PUBLIC_PLAYING_OBSERVATION_VARIANT,
        COMPLETE_INFO_COMPACT_PLAYING_OBSERVATION_VARIANT,
    }:
        return str(value)
    raise ValueError(f"unsupported playing observation variant: {value!r}")


def playing_encoder_schema_version_for_variant(variant: object | None) -> int:
    normalized = normalize_playing_observation_variant(variant)
    if normalized == PUBLIC_PLAYING_OBSERVATION_VARIANT:
        return PLAYING_ENCODER_SCHEMA_VERSION
    return COMPLETE_INFO_PLAYING_ENCODER_SCHEMA_VERSION


def playing_model_input_schema_version_for_variant(variant: object | None) -> int:
    normalized = normalize_playing_observation_variant(variant)
    if normalized == PUBLIC_PLAYING_OBSERVATION_VARIANT:
        return PLAYING_MODEL_INPUT_SCHEMA_VERSION
    return COMPLETE_INFO_PLAYING_MODEL_INPUT_SCHEMA_VERSION


def model_input_feature_count_for_variant(variant: object | None) -> int:
    normalized = normalize_playing_observation_variant(variant)
    if normalized == PUBLIC_PLAYING_OBSERVATION_VARIANT:
        return MODEL_INPUT_FEATURE_COUNT
    return COMPLETE_INFO_PLAYING_MODEL_INPUT_FEATURE_COUNT
