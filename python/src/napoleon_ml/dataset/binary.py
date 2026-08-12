"""Binary tensor-ready reader for playing self-play datasets."""

from __future__ import annotations

import gzip
import hashlib
import json
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import torch
from torch import Tensor

from .constants import CARD_COUNT, PLAYER_COUNT, SELF_ROLE_COUNT
from .errors import DatasetError, ShardIntegrityError
from .manifest import DatasetManifest, DatasetShardManifest
from .playing_variants import (
    model_input_feature_count_for_variant,
    normalize_playing_observation_variant,
)
from .split import DatasetSplit, SplitConfig, split_for_seed


class PlayingSelfPlayBinaryBatch(TypedDict):
    model_input: Tensor
    legal_play_mask: Tensor
    selected_card_index: Tensor
    behavior_log_probability: Tensor
    terminal_reward: Tensor
    seed: Tensor
    step: Tensor
    acting_player_index: Tensor
    self_role_index: Tensor


_MAGIC = b"NPSPBD01"
_FIELD_DTYPES: dict[str, np.dtype[Any]] = {
    "modelInput": np.dtype("<f4"),
    "legalPlayMask": np.dtype("u1"),
    "selectedCardIndex": np.dtype("u1"),
    "behaviorLogProbability": np.dtype("<f4"),
    "terminalReward": np.dtype("i1"),
    "seed": np.dtype("<u4"),
    "step": np.dtype("<u2"),
    "actingPlayerIndex": np.dtype("u1"),
    "selfRoleIndex": np.dtype("u1"),
}


def _field_shapes(manifest: DatasetManifest) -> dict[str, tuple[int, ...]]:
    return {
        "modelInput": (
            model_input_feature_count_for_variant(manifest.playing_observation_variant),
        ),
        "legalPlayMask": (CARD_COUNT,),
        "selectedCardIndex": (),
        "behaviorLogProbability": (),
        "terminalReward": (),
        "seed": (),
        "step": (),
        "actingPlayerIndex": (),
        "selfRoleIndex": (),
    }


_HEADER_PREFIX_LENGTH = len(_MAGIC) + 4
_SUPPORTED_COMPRESSIONS = frozenset({"none", "gzip"})


def iter_binary_playing_self_play_batches(
    dataset_directory: Path | str,
    manifest: DatasetManifest,
    *,
    split: DatasetSplit | None,
    split_config: SplitConfig,
    batch_size: int,
    verify_integrity: bool,
    drop_last: bool = False,
) -> Iterator[PlayingSelfPlayBinaryBatch]:
    """Yield pre-batched tensors from a v4 binary self-play dataset."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise DatasetError(f"batch_size must be a positive integer, got {batch_size!r}.")

    directory = Path(dataset_directory)
    pending: dict[str, np.ndarray] | None = None
    for shard in manifest.shards:
        arrays = _read_shard(
            directory / shard.file,
            shard,
            manifest,
            verify_integrity=verify_integrity,
        )
        keep = _split_mask(arrays["seed"], split=split, split_config=split_config)
        indices = np.nonzero(keep)[0]
        offset = 0

        if pending is not None:
            needed = batch_size - _batch_array_count(pending)
            take = min(needed, int(indices.shape[0]))
            offset = take
            if take > 0:
                pending = _concat_batch_arrays(pending, _slice_arrays(arrays, indices[:take]))
            if _batch_array_count(pending) == batch_size:
                yield _torch_batch(pending)
                pending = None

        while offset < int(indices.shape[0]):
            selected = indices[offset : offset + batch_size]
            if int(selected.shape[0]) == batch_size:
                yield _torch_batch(_slice_arrays(arrays, selected))
                offset += batch_size
                continue
            pending = _slice_arrays(arrays, selected)
            offset = int(indices.shape[0])

    if pending is not None and not drop_last:
        yield _torch_batch(pending)


def _read_shard(
    path: Path,
    shard: DatasetShardManifest,
    manifest: DatasetManifest,
    *,
    verify_integrity: bool,
) -> dict[str, np.ndarray]:
    data = path.read_bytes()
    if len(data) != shard.byte_length:
        raise ShardIntegrityError(
            f"{shard.file}: byte length mismatch: expected {shard.byte_length}, got {len(data)}."
        )
    if verify_integrity:
        sha256 = hashlib.sha256(data).hexdigest()
        if sha256 != shard.sha256:
            raise ShardIntegrityError(
                f"{shard.file}: SHA-256 mismatch: expected {shard.sha256}, got {sha256}."
            )
    if len(data) < _HEADER_PREFIX_LENGTH or data[: len(_MAGIC)] != _MAGIC:
        raise ShardIntegrityError(f"{shard.file}: invalid binary shard magic.")

    header_length = struct.unpack_from("<I", data, len(_MAGIC))[0]
    header_start = _HEADER_PREFIX_LENGTH
    header_stop = header_start + header_length
    if header_stop > len(data):
        raise ShardIntegrityError(f"{shard.file}: header length exceeds file size.")

    try:
        header = json.loads(data[header_start:header_stop].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ShardIntegrityError(f"{shard.file}: invalid binary shard header: {error}") from error

    _validate_header(header, shard, manifest)
    compression = header["compression"]
    raw_payload = data[header_stop:]
    if compression == "gzip":
        try:
            payload = bytearray(gzip.decompress(raw_payload))
        except OSError as error:
            raise ShardIntegrityError(
                f"{shard.file}: gzip payload cannot be decompressed."
            ) from error
    elif compression == "none":
        payload = bytearray(raw_payload)
    else:
        raise ShardIntegrityError(
            f"{shard.file}: unsupported binary shard compression: {compression!r}."
        )

    expected_length = int(header["uncompressedByteLength"])
    if len(payload) != expected_length:
        raise ShardIntegrityError(
            f"{shard.file}: uncompressed byte length mismatch: "
            f"expected {expected_length}, got {len(payload)}."
        )

    arrays = _arrays_from_payload(payload, header, manifest)
    _validate_arrays(arrays, shard)
    return arrays


def _validate_header(
    header: object,
    shard: DatasetShardManifest,
    manifest: DatasetManifest,
) -> None:
    if not isinstance(header, dict):
        raise ShardIntegrityError(f"{shard.file}: binary shard header must be an object.")
    expected_scalars = {
        "shardSchemaVersion": 1,
        "sampleType": "playing-self-play-sample",
        "sampleSchemaVersion": 4,
        "sampleCount": shard.sample_count,
        "modelInputFeatureCount": model_input_feature_count_for_variant(
            manifest.playing_observation_variant
        ),
        "cardCount": CARD_COUNT,
        "byteOrder": "little-endian",
    }
    for key, expected in expected_scalars.items():
        if header.get(key) != expected:
            raise ShardIntegrityError(
                f"{shard.file}: header {key} mismatch: expected {expected!r}, "
                f"got {header.get(key)!r}."
            )
    if header.get("compression") not in _SUPPORTED_COMPRESSIONS:
        raise ShardIntegrityError(
            f"{shard.file}: header compression mismatch: expected one of "
            f"{sorted(_SUPPORTED_COMPRESSIONS)!r}, got {header.get('compression')!r}."
        )
    if not isinstance(header.get("uncompressedByteLength"), int):
        raise ShardIntegrityError(f"{shard.file}: header uncompressedByteLength is invalid.")
    fields = header.get("fields")
    if not isinstance(fields, list):
        raise ShardIntegrityError(f"{shard.file}: header fields must be a list.")
    if len(fields) != len(_FIELD_DTYPES):
        raise ShardIntegrityError(f"{shard.file}: header field count mismatch.")
    variant = normalize_playing_observation_variant(manifest.playing_observation_variant)
    if header.get("playingObservationVariant", variant) != variant:
        raise ShardIntegrityError(
            f"{shard.file}: header playingObservationVariant mismatch."
        )


def _arrays_from_payload(
    payload: bytearray,
    header: dict[str, object],
    manifest: DatasetManifest,
) -> dict[str, np.ndarray]:
    raw_sample_count = header["sampleCount"]
    if not isinstance(raw_sample_count, int):
        raise ShardIntegrityError("binary shard header sampleCount is invalid.")
    sample_count = raw_sample_count
    arrays: dict[str, np.ndarray] = {}
    fields = cast(list[object], header["fields"])
    field_shapes = _field_shapes(manifest)

    for raw_field in fields:
        if not isinstance(raw_field, dict):
            raise ShardIntegrityError("binary shard header field must be an object.")
        name = raw_field.get("name")
        dtype_name = raw_field.get("dtype")
        shape_raw = raw_field.get("shape")
        byte_offset = raw_field.get("byteOffset")
        byte_length = raw_field.get("byteLength")
        if not isinstance(name, str) or name not in _FIELD_DTYPES:
            raise ShardIntegrityError(f"binary shard field has invalid name: {name!r}.")
        if dtype_name != _dtype_name(name):
            raise ShardIntegrityError(f"binary shard field {name} dtype mismatch.")
        if not isinstance(shape_raw, list) or tuple(shape_raw) != field_shapes[name]:
            raise ShardIntegrityError(f"binary shard field {name} shape mismatch.")
        if not isinstance(byte_offset, int) or not isinstance(byte_length, int):
            raise ShardIntegrityError(f"binary shard field {name} byte range is invalid.")

        dtype: np.dtype[Any] = _FIELD_DTYPES[name]
        shape = (sample_count, *field_shapes[name])
        expected_byte_length = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if byte_length != expected_byte_length:
            raise ShardIntegrityError(f"binary shard field {name} byteLength mismatch.")
        if byte_offset < 0 or byte_offset + byte_length > len(payload):
            raise ShardIntegrityError(f"binary shard field {name} byte range exceeds payload.")
        arrays[name] = np.frombuffer(
            payload,
            dtype=dtype,
            count=expected_byte_length // dtype.itemsize,
            offset=byte_offset,
        ).reshape(shape)

    return arrays


def _validate_arrays(arrays: dict[str, np.ndarray], shard: DatasetShardManifest) -> None:
    required = set(_FIELD_DTYPES)
    if set(arrays) != required:
        raise ShardIntegrityError(f"{shard.file}: binary fields mismatch.")
    if int(arrays["seed"].shape[0]) <= 0:
        raise ShardIntegrityError(f"{shard.file}: binary shard must contain at least one sample.")
    if int(arrays["seed"][0]) != shard.start_seed or int(arrays["seed"][-1]) != shard.end_seed:
        raise ShardIntegrityError(f"{shard.file}: seed range mismatch.")

    legal = arrays["legalPlayMask"]
    selected = arrays["selectedCardIndex"]
    if bool(((legal != 0) & (legal != 1)).any()):
        raise ShardIntegrityError(f"{shard.file}: legalPlayMask must contain only 0/1.")
    if bool(legal.sum(axis=1).min() <= 0):
        raise ShardIntegrityError(f"{shard.file}: every row needs at least one legal card.")
    if bool((selected >= CARD_COUNT).any()):
        raise ShardIntegrityError(f"{shard.file}: selectedCardIndex out of range.")
    if bool((legal[np.arange(selected.shape[0]), selected] != 1).any()):
        raise ShardIntegrityError(f"{shard.file}: selectedCardIndex must be legal.")
    if bool(~np.isfinite(arrays["modelInput"]).all()):
        raise ShardIntegrityError(f"{shard.file}: modelInput contains NaN or Infinity.")
    if bool(~np.isfinite(arrays["behaviorLogProbability"]).all()):
        raise ShardIntegrityError(
            f"{shard.file}: behaviorLogProbability contains NaN or Infinity."
        )
    if bool((arrays["behaviorLogProbability"] > 1e-12).any()):
        raise ShardIntegrityError(f"{shard.file}: behaviorLogProbability must be <= 0.")
    if bool(((arrays["terminalReward"] != 1) & (arrays["terminalReward"] != -1)).any()):
        raise ShardIntegrityError(f"{shard.file}: terminalReward must be +/-1.")
    if bool(arrays["actingPlayerIndex"].max() >= PLAYER_COUNT):
        raise ShardIntegrityError(f"{shard.file}: actingPlayerIndex out of range.")
    if bool(arrays["selfRoleIndex"].max() >= SELF_ROLE_COUNT):
        raise ShardIntegrityError(f"{shard.file}: selfRoleIndex out of range.")


def _split_mask(
    seeds: np.ndarray,
    *,
    split: DatasetSplit | None,
    split_config: SplitConfig,
) -> np.ndarray:
    if split is None:
        return np.ones(seeds.shape, dtype=np.bool_)
    return np.fromiter(
        (split_for_seed(int(seed), split_config) == split for seed in seeds),
        dtype=np.bool_,
        count=int(seeds.shape[0]),
    )


def _slice_arrays(
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    return {name: value[indices] for name, value in arrays.items()}


def _concat_batch_arrays(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {name: np.concatenate([left[name], right[name]], axis=0) for name in left}


def _batch_array_count(arrays: dict[str, np.ndarray]) -> int:
    return int(arrays["seed"].shape[0])


def _torch_batch(
    arrays: dict[str, np.ndarray],
) -> PlayingSelfPlayBinaryBatch:
    return {
        "model_input": torch.from_numpy(arrays["modelInput"]),
        "legal_play_mask": torch.from_numpy(arrays["legalPlayMask"]).to(dtype=torch.bool),
        "selected_card_index": torch.from_numpy(
            arrays["selectedCardIndex"].astype(np.int64, copy=False)
        ),
        "behavior_log_probability": torch.from_numpy(arrays["behaviorLogProbability"]),
        "terminal_reward": torch.from_numpy(
            arrays["terminalReward"].astype(np.float32, copy=False)
        ),
        "seed": torch.from_numpy(arrays["seed"].astype(np.int64, copy=False)),
        "step": torch.from_numpy(arrays["step"].astype(np.int64, copy=False)),
        "acting_player_index": torch.from_numpy(
            arrays["actingPlayerIndex"].astype(np.int64, copy=False)
        ),
        "self_role_index": torch.from_numpy(
            arrays["selfRoleIndex"].astype(np.int64, copy=False)
        ),
    }


def _dtype_name(name: str) -> str:
    dtype = _FIELD_DTYPES[name]
    if dtype == np.dtype("<f4"):
        return "float32"
    if dtype == np.dtype("<u4"):
        return "uint32"
    if dtype == np.dtype("<u2"):
        return "uint16"
    if dtype == np.dtype("u1"):
        return "uint8"
    if dtype == np.dtype("i1"):
        return "int8"
    raise DatasetError(f"Unsupported binary dtype for {name}: {dtype}.")
