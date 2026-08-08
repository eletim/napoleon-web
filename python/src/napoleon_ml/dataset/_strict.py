"""Strict, non-coercing JSON value extraction helpers.

``json.loads()`` hands back loosely typed values, and Python's ``bool`` is an
``int`` subclass, so a bare ``isinstance(value, int)`` also accepts ``True``
and ``False``. None of the helpers below ever coerce a value (no
``int(value)``, ``str(value)``); they only check that the value already has
the expected type and raise a caller-supplied error otherwise.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

ErrorFactory = Callable[[str], Exception]


def describe_type(value: object) -> str:
    if isinstance(value, bool):
        return "bool"
    if value is None:
        return "null"
    return type(value).__name__


def require_dict(value: object, *, path: str, error: ErrorFactory) -> dict[str, object]:
    if not isinstance(value, dict):
        raise error(f"{path} must be a JSON object, got {describe_type(value)}.")
    return value


def require_list(value: object, *, path: str, error: ErrorFactory) -> list[object]:
    if not isinstance(value, list):
        raise error(f"{path} must be a JSON array, got {describe_type(value)}.")
    return value


def require_str(value: object, *, path: str, error: ErrorFactory) -> str:
    if not isinstance(value, str):
        raise error(f"{path} must be a string, got {describe_type(value)}.")
    return value


def require_int(value: object, *, path: str, error: ErrorFactory) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise error(f"{path} must be an integer, got {describe_type(value)}.")
    return value


def require_number(value: object, *, path: str, error: ErrorFactory) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise error(f"{path} must be a number, got {describe_type(value)}.")
    return float(value)


def require_key(obj: Mapping[str, object], key: str, *, path: str, error: ErrorFactory) -> object:
    if key not in obj:
        raise error(f"{path} is missing required key {key!r}.")
    return obj[key]


def reject_unknown_keys(
    obj: Mapping[str, object],
    allowed: frozenset[str],
    *,
    path: str,
    error: ErrorFactory,
) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise error(f"{path} has unknown key(s): {unknown!r}.")


def require_exact_keys(
    obj: Mapping[str, object],
    allowed: frozenset[str],
    *,
    path: str,
    error: ErrorFactory,
) -> None:
    for key in sorted(allowed):
        require_key(obj, key, path=path, error=error)
    reject_unknown_keys(obj, allowed, path=path, error=error)


def require_str_tuple(value: object, *, path: str, error: ErrorFactory) -> tuple[str, ...]:
    items = require_list(value, path=path, error=error)
    return tuple(
        require_str(item, path=f"{path}[{index}]", error=error) for index, item in enumerate(items)
    )


def require_int_tuple(value: object, *, path: str, error: ErrorFactory) -> tuple[int, ...]:
    items = require_list(value, path=path, error=error)
    return tuple(
        require_int(item, path=f"{path}[{index}]", error=error) for index, item in enumerate(items)
    )


def require_fixed_length_int_tuple(
    value: object, *, path: str, length: int, error: ErrorFactory
) -> tuple[int, ...]:
    items = require_int_tuple(value, path=path, error=error)
    if len(items) != length:
        raise error(f"{path} must have length {length}, got {len(items)}.")
    return items


def require_int_in_range(
    value: object, *, path: str, minimum: int, maximum: int, error: ErrorFactory
) -> int:
    integer = require_int(value, path=path, error=error)
    if integer < minimum or integer > maximum:
        raise error(f"{path} must be between {minimum} and {maximum}, got {integer}.")
    return integer


def sequence_length_error(
    name: str, sequence: Sequence[object], expected_length: int, *, error: ErrorFactory
) -> None:
    if len(sequence) != expected_length:
        raise error(f"{name} must have length {expected_length}, got {len(sequence)}.")
