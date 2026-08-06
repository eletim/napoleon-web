from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from napoleon_ml.dataset.constants import EXPECTED_CARD_IDS
from napoleon_ml.dataset.errors import SampleValidationError, ShardIntegrityError
from napoleon_ml.dataset.reader import iter_raw_samples, iter_samples
from napoleon_ml.dataset.validation import calculate_card_ids_sha256

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "valid_sample.json"


def _load_valid_sample() -> dict[str, Any]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_single_shard_dataset(
    directory: Path,
    shard_bytes: bytes,
    *,
    game_count: int,
    sample_count: int,
    games_per_shard: int,
    start_seed: int,
    end_seed: int,
    shard_game_count: int,
    shard_sample_count: int | None = None,
    shard_sha256: str | None = None,
    shard_byte_length: int | None = None,
) -> None:
    shard_path = directory / "shard-00000.jsonl"
    shard_path.write_bytes(shard_bytes)

    manifest = {
        "datasetSchemaVersion": 1,
        "generatorVersion": 1,
        "playingEncoderSchemaVersion": 1,
        "format": "jsonl",
        "sampleType": "playing-training-sample",
        "agent": {"type": "rule-based", "version": 1},
        "startSeed": start_seed,
        "endSeed": end_seed,
        "gameCount": game_count,
        "sampleCount": sample_count,
        "gamesPerShard": games_per_shard,
        "shardCount": 1,
        "playerCount": 5,
        "cardCount": 53,
        "cardIds": list(EXPECTED_CARD_IDS),
        "cardIdsSha256": calculate_card_ids_sha256(),
        "shards": [
            {
                "file": "shard-00000.jsonl",
                "startSeed": start_seed,
                "endSeed": end_seed,
                "gameCount": shard_game_count,
                "sampleCount": shard_sample_count
                if shard_sample_count is not None
                else sample_count,
                "byteLength": shard_byte_length
                if shard_byte_length is not None
                else len(shard_bytes),
                "sha256": shard_sha256 if shard_sha256 is not None else _sha256_hex(shard_bytes),
            }
        ],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _one_line(sample: dict[str, Any]) -> bytes:
    return (json.dumps(sample) + "\n").encode("utf-8")


def test_iter_raw_samples_reads_valid_shard(tmp_path: Path) -> None:
    sample = _load_valid_sample()
    shard_bytes = _one_line(sample)
    _write_single_shard_dataset(
        tmp_path,
        shard_bytes,
        game_count=1,
        sample_count=1,
        games_per_shard=1,
        start_seed=0,
        end_seed=0,
        shard_game_count=1,
    )

    samples = list(iter_raw_samples(tmp_path))

    assert len(samples) == 1
    assert samples[0]["seed"] == 0


def test_sha256_mismatch_rejected(tmp_path: Path) -> None:
    shard_bytes = _one_line({"seed": 0})
    _write_single_shard_dataset(
        tmp_path,
        shard_bytes,
        game_count=1,
        sample_count=1,
        games_per_shard=1,
        start_seed=0,
        end_seed=0,
        shard_game_count=1,
        shard_sha256="0" * 64,
    )

    with pytest.raises(ShardIntegrityError, match="SHA-256 mismatch"):
        list(iter_raw_samples(tmp_path))


def test_byte_length_mismatch_rejected(tmp_path: Path) -> None:
    shard_bytes = _one_line({"seed": 0})
    _write_single_shard_dataset(
        tmp_path,
        shard_bytes,
        game_count=1,
        sample_count=1,
        games_per_shard=1,
        start_seed=0,
        end_seed=0,
        shard_game_count=1,
        shard_byte_length=len(shard_bytes) + 1,
    )

    with pytest.raises(ShardIntegrityError, match="byte length mismatch"):
        list(iter_raw_samples(tmp_path))


def test_line_count_mismatch_rejected(tmp_path: Path) -> None:
    shard_bytes = _one_line({"seed": 0})
    _write_single_shard_dataset(
        tmp_path,
        shard_bytes,
        game_count=1,
        sample_count=2,
        games_per_shard=1,
        start_seed=0,
        end_seed=0,
        shard_game_count=1,
        shard_sample_count=2,
    )

    with pytest.raises(ShardIntegrityError, match="line count mismatch"):
        list(iter_raw_samples(tmp_path))


def test_empty_line_rejected(tmp_path: Path) -> None:
    shard_bytes = _one_line({"seed": 0}) + b"\n"
    _write_single_shard_dataset(
        tmp_path,
        shard_bytes,
        game_count=1,
        sample_count=2,
        games_per_shard=1,
        start_seed=0,
        end_seed=0,
        shard_game_count=1,
        shard_sample_count=2,
    )

    with pytest.raises(ShardIntegrityError, match="line is empty"):
        list(iter_raw_samples(tmp_path))


def test_crlf_line_ending_rejected(tmp_path: Path) -> None:
    shard_bytes = json.dumps({"seed": 0}).encode("utf-8") + b"\r\n"
    _write_single_shard_dataset(
        tmp_path,
        shard_bytes,
        game_count=1,
        sample_count=1,
        games_per_shard=1,
        start_seed=0,
        end_seed=0,
        shard_game_count=1,
    )

    with pytest.raises(ShardIntegrityError, match="CRLF"):
        list(iter_raw_samples(tmp_path))


def test_missing_final_newline_rejected(tmp_path: Path) -> None:
    shard_bytes = json.dumps({"seed": 0}).encode("utf-8")  # no trailing \n
    _write_single_shard_dataset(
        tmp_path,
        shard_bytes,
        game_count=1,
        sample_count=1,
        games_per_shard=1,
        start_seed=0,
        end_seed=0,
        shard_game_count=1,
    )

    with pytest.raises(ShardIntegrityError, match="trailing newline"):
        list(iter_raw_samples(tmp_path))


def test_invalid_utf8_rejected(tmp_path: Path) -> None:
    shard_bytes = b"\xff\xfe not utf-8\n"
    _write_single_shard_dataset(
        tmp_path,
        shard_bytes,
        game_count=1,
        sample_count=1,
        games_per_shard=1,
        start_seed=0,
        end_seed=0,
        shard_game_count=1,
    )

    with pytest.raises(ShardIntegrityError, match="not valid UTF-8"):
        list(iter_raw_samples(tmp_path))


def test_malformed_json_rejected(tmp_path: Path) -> None:
    shard_bytes = b"{not json}\n"
    _write_single_shard_dataset(
        tmp_path,
        shard_bytes,
        game_count=1,
        sample_count=1,
        games_per_shard=1,
        start_seed=0,
        end_seed=0,
        shard_game_count=1,
    )

    with pytest.raises(ShardIntegrityError, match="not valid JSON"):
        list(iter_raw_samples(tmp_path))


def test_json_non_object_rejected(tmp_path: Path) -> None:
    shard_bytes = b"[1, 2, 3]\n"
    _write_single_shard_dataset(
        tmp_path,
        shard_bytes,
        game_count=1,
        sample_count=1,
        games_per_shard=1,
        start_seed=0,
        end_seed=0,
        shard_game_count=1,
    )

    with pytest.raises(ShardIntegrityError, match="JSON object"):
        list(iter_raw_samples(tmp_path))


def test_duplicate_json_key_rejected(tmp_path: Path) -> None:
    shard_bytes = b'{"seed": 0, "seed": 1}\n'
    _write_single_shard_dataset(
        tmp_path,
        shard_bytes,
        game_count=1,
        sample_count=1,
        games_per_shard=1,
        start_seed=0,
        end_seed=0,
        shard_game_count=1,
    )

    with pytest.raises(ShardIntegrityError, match="duplicate JSON key"):
        list(iter_raw_samples(tmp_path))


def test_nan_rejected(tmp_path: Path) -> None:
    shard_bytes = b'{"seed": 0, "value": NaN}\n'
    _write_single_shard_dataset(
        tmp_path,
        shard_bytes,
        game_count=1,
        sample_count=1,
        games_per_shard=1,
        start_seed=0,
        end_seed=0,
        shard_game_count=1,
    )

    with pytest.raises(ShardIntegrityError, match="not valid JSON"):
        list(iter_raw_samples(tmp_path))


def test_infinity_rejected(tmp_path: Path) -> None:
    shard_bytes = b'{"seed": 0, "value": Infinity}\n'
    _write_single_shard_dataset(
        tmp_path,
        shard_bytes,
        game_count=1,
        sample_count=1,
        games_per_shard=1,
        start_seed=0,
        end_seed=0,
        shard_game_count=1,
    )

    with pytest.raises(ShardIntegrityError, match="not valid JSON"):
        list(iter_raw_samples(tmp_path))


def test_verify_integrity_false_skips_hash_check(tmp_path: Path) -> None:
    shard_bytes = _one_line({"seed": 0})
    _write_single_shard_dataset(
        tmp_path,
        shard_bytes,
        game_count=1,
        sample_count=1,
        games_per_shard=1,
        start_seed=0,
        end_seed=0,
        shard_game_count=1,
        shard_sha256="0" * 64,
    )

    samples = list(iter_raw_samples(tmp_path, verify_integrity=False))

    assert len(samples) == 1


def test_verify_integrity_false_skips_byte_length_check(tmp_path: Path) -> None:
    shard_bytes = _one_line({"seed": 0})
    _write_single_shard_dataset(
        tmp_path,
        shard_bytes,
        game_count=1,
        sample_count=1,
        games_per_shard=1,
        start_seed=0,
        end_seed=0,
        shard_game_count=1,
        shard_byte_length=len(shard_bytes) + 1,
    )

    samples = list(iter_raw_samples(tmp_path, verify_integrity=False))

    assert len(samples) == 1


def test_deleted_line_rejected_in_both_modes(tmp_path: Path) -> None:
    # The shard on disk has had a line deleted since it was generated (only
    # 1 line remains), but its SHA-256/byte length in the manifest are left
    # matching the (now-truncated) bytes actually on disk -- as if an
    # attacker recomputed the hash to match the tampered file. Only the
    # manifest's stale sampleCount (2, from before the deletion) can catch
    # this, so it must be rejected in both integrity modes: structural
    # checks (line/sample count) are never skipped, unlike the hash checks.
    shard_bytes = _one_line({"seed": 0})
    _write_single_shard_dataset(
        tmp_path,
        shard_bytes,
        game_count=1,
        sample_count=2,
        games_per_shard=1,
        start_seed=0,
        end_seed=0,
        shard_game_count=1,
        shard_sample_count=2,
    )

    with pytest.raises(ShardIntegrityError, match="line count mismatch"):
        list(iter_raw_samples(tmp_path))

    with pytest.raises(ShardIntegrityError, match="line count mismatch"):
        list(iter_raw_samples(tmp_path, verify_integrity=False))


def test_sha256_only_mismatch_succeeds_with_no_integrity_check(tmp_path: Path) -> None:
    # Content, line count, sample count, and seeds are all correct -- only
    # the recorded SHA-256 is wrong. Normal mode must fail; no-integrity mode
    # must succeed, since structural checks all pass on their own.
    shard_bytes = _one_line({"seed": 0})
    _write_single_shard_dataset(
        tmp_path,
        shard_bytes,
        game_count=1,
        sample_count=1,
        games_per_shard=1,
        start_seed=0,
        end_seed=0,
        shard_game_count=1,
        shard_sha256="0" * 64,
    )

    with pytest.raises(ShardIntegrityError, match="SHA-256 mismatch"):
        list(iter_raw_samples(tmp_path))

    samples = list(iter_raw_samples(tmp_path, verify_integrity=False))
    assert len(samples) == 1


def test_game_count_uses_seed_transitions_not_unique_seed_set(tmp_path: Path) -> None:
    # Seeds 0, 1, 0, 1 form 4 consecutive runs (game 0 and game 1 interleave
    # instead of each occupying one contiguous block) but only 2 *unique*
    # seeds. A set-based unique-seed count would accept this shard as
    # "2 games" -- the same count a well-formed 0, 0, 1, 1 shard would
    # produce. Counting seed transitions instead gives 4, which must be
    # rejected against the manifest's gameCount of 2 -- in both integrity
    # modes, since game count is a structural check.
    shard_bytes = (
        _one_line({"seed": 0})
        + _one_line({"seed": 1})
        + _one_line({"seed": 0})
        + _one_line({"seed": 1})
    )
    _write_single_shard_dataset(
        tmp_path,
        shard_bytes,
        game_count=2,
        sample_count=4,
        games_per_shard=2,
        start_seed=0,
        end_seed=1,
        shard_game_count=2,
        shard_sample_count=4,
    )

    with pytest.raises(ShardIntegrityError, match="game count mismatch"):
        list(iter_raw_samples(tmp_path))

    with pytest.raises(ShardIntegrityError, match="game count mismatch"):
        list(iter_raw_samples(tmp_path, verify_integrity=False))


def test_iter_samples_full_pipeline(tmp_path: Path) -> None:
    sample = _load_valid_sample()
    shard_bytes = _one_line(sample)
    _write_single_shard_dataset(
        tmp_path,
        shard_bytes,
        game_count=1,
        sample_count=1,
        games_per_shard=1,
        start_seed=0,
        end_seed=0,
        shard_game_count=1,
    )

    samples = list(iter_samples(tmp_path))

    assert len(samples) == 1
    assert samples[0].seed == 0
    assert samples[0].step == 12


def test_iter_samples_accepts_ascending_seed_and_step(tmp_path: Path) -> None:
    base = _load_valid_sample()
    first = dict(base, seed=0, step=1)
    second = dict(base, seed=0, step=2)
    third = dict(base, seed=1, step=1)
    shard_bytes = _one_line(first) + _one_line(second) + _one_line(third)
    _write_single_shard_dataset(
        tmp_path,
        shard_bytes,
        game_count=2,
        sample_count=3,
        games_per_shard=2,
        start_seed=0,
        end_seed=1,
        shard_game_count=2,
    )

    samples = list(iter_samples(tmp_path))

    assert [(sample.seed, sample.step) for sample in samples] == [(0, 1), (0, 2), (1, 1)]


def test_iter_samples_rejects_non_increasing_step_within_seed(tmp_path: Path) -> None:
    base = _load_valid_sample()
    first = dict(base, seed=0, step=5)
    second = dict(base, seed=0, step=5)  # step does not increase
    shard_bytes = _one_line(first) + _one_line(second)
    _write_single_shard_dataset(
        tmp_path,
        shard_bytes,
        game_count=1,
        sample_count=2,
        games_per_shard=1,
        start_seed=0,
        end_seed=0,
        shard_game_count=1,
    )

    with pytest.raises(SampleValidationError, match="step must increase"):
        list(iter_samples(tmp_path))


def test_iter_samples_rejects_seed_reappearance(tmp_path: Path) -> None:
    base = _load_valid_sample()
    first = dict(base, seed=0, step=1)
    second = dict(base, seed=1, step=1)
    third = dict(base, seed=0, step=2)  # seed 0 reappears after moving to seed 1
    shard_bytes = _one_line(first) + _one_line(second) + _one_line(third)
    _write_single_shard_dataset(
        tmp_path,
        shard_bytes,
        game_count=2,
        sample_count=3,
        games_per_shard=2,
        start_seed=0,
        end_seed=1,
        shard_game_count=2,
    )

    with pytest.raises(SampleValidationError, match="breaks dataset seed ordering"):
        list(iter_samples(tmp_path))


def test_iter_samples_rejects_seed_gap(tmp_path: Path) -> None:
    base = _load_valid_sample()
    first = dict(base, seed=0, step=1)
    second = dict(base, seed=2, step=1)  # jumps from 0 to 2, skipping 1
    shard_bytes = _one_line(first) + _one_line(second)
    _write_single_shard_dataset(
        tmp_path,
        shard_bytes,
        game_count=3,
        sample_count=2,
        games_per_shard=3,
        start_seed=0,
        end_seed=2,
        shard_game_count=3,
    )

    with pytest.raises(SampleValidationError, match="breaks dataset seed ordering"):
        list(iter_samples(tmp_path))
