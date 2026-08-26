"""Diagnostic-only actual adjutant-location overlays for Issue #450."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ADJUTANT_LOCATION_CLASS_NAMES = (
    "opponentSeat1",
    "opponentSeat2",
    "opponentSeat3",
    "opponentSeat4",
    "selfKittySolo",
)
ADJUTANT_LOCATION_CLASS_COUNT = len(ADJUTANT_LOCATION_CLASS_NAMES)
ORACLE_INPUT_VARIANT = "compact401-oracle-location"


def relative_adjutant_location_class(owner_seat: int | None, napoleon_seat: int) -> int:
    """Return the five-class Napoleon-relative pre-exchange location."""
    if not 0 <= napoleon_seat < 5:
        raise ValueError("napoleon_seat must be in [0,4].")
    if owner_seat is None or owner_seat == napoleon_seat:
        return 4
    if not 0 <= owner_seat < 5:
        raise ValueError("owner_seat must be in [0,4] or None.")
    return (owner_seat - napoleon_seat) % 5 - 1


def location_one_hot(class_index: int) -> np.ndarray:
    if not 0 <= class_index < ADJUTANT_LOCATION_CLASS_COUNT:
        raise ValueError("adjutant location class index must be in [0,4].")
    result = np.zeros(ADJUTANT_LOCATION_CLASS_COUNT, dtype=np.float32)
    result[class_index] = 1.0
    return result


def load_training_location_overlay(
    path: Path | str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    overlay_path = Path(path)
    raw = json.loads(overlay_path.read_text(encoding="utf-8"))
    if raw.get("artifactType") != "issue450-exchange-training-location-overlay-v1":
        raise ValueError("training oracle overlay artifactType mismatch.")
    if raw.get("classNames") != list(ADJUTANT_LOCATION_CLASS_NAMES):
        raise ValueError("training oracle overlay classNames mismatch.")
    entries = raw.get("entries")
    if not isinstance(entries, dict) or len(entries) != int(raw.get("sourceStateCount", -1)):
        raise ValueError("training oracle overlay entry count mismatch.")
    return entries, {
        "path": str(overlay_path),
        "sha256": hashlib.sha256(overlay_path.read_bytes()).hexdigest(),
        "sourceStateCount": len(entries),
        "datasetManifests": raw.get("datasetManifests"),
    }


def load_full_gold_location_overlay(
    path: Path | str,
    *,
    manifest_sha256: str,
    source_seeds: tuple[int, ...],
) -> dict[str, Any]:
    overlay_path = Path(path)
    raw = json.loads(overlay_path.read_text(encoding="utf-8"))
    if raw.get("artifactType") != "issue450-fixed-full-gold-location-overlay-v1":
        raise ValueError("full-gold oracle overlay artifactType mismatch.")
    if raw.get("fixedHoldoutManifestSha256") != manifest_sha256:
        raise ValueError("full-gold oracle overlay fixed manifest SHA-256 mismatch.")
    if raw.get("classNames") != list(ADJUTANT_LOCATION_CLASS_NAMES):
        raise ValueError("full-gold oracle overlay classNames mismatch.")
    values = np.asarray(raw.get("classIndices"), dtype=np.int64)
    if values.ndim != 2 or values.shape != (len(source_seeds), 53):
        raise ValueError("full-gold oracle overlay must have shape (states,53).")
    if raw.get("sourceSeeds") != list(source_seeds):
        raise ValueError("full-gold oracle overlay source seed order mismatch.")
    if bool(np.any((values < 0) | (values >= ADJUTANT_LOCATION_CLASS_COUNT))):
        raise ValueError("full-gold oracle overlay contains an invalid class index.")
    return {
        **raw,
        "classIndicesArray": values,
        "path": str(overlay_path),
        "sha256": hashlib.sha256(overlay_path.read_bytes()).hexdigest(),
    }
