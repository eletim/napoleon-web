"""Byte-strict JSON decoding.

The standard library's ``json.loads()`` silently keeps the *last* value for a
duplicate object key, and by default accepts the non-standard ``NaN``,
``Infinity``, and ``-Infinity`` tokens. Both behaviors are unsafe for a
dataset-integrity boundary: a duplicate key can hide a legitimate value
behind a corrupted or malicious one, and a stray ``NaN``/``Infinity`` would
otherwise pass through every downstream numeric-range check unnoticed.
``loads_strict`` rejects both.
"""

from __future__ import annotations

import json


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}

    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")

        result[key] = value

    return result


def _reject_constant(constant: str) -> float:
    raise ValueError(f"JSON must not contain {constant}.")


def loads_strict(text: str) -> object:
    """Parse ``text`` as JSON, rejecting duplicate object keys and NaN/Infinity."""

    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_constant,
    )
