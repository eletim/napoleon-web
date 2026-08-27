"""CRN evaluation and pycma orchestration primitives for Issue #452."""

from __future__ import annotations

import json
import math
import pickle
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import cma  # type: ignore[import-untyped]
import numpy as np

FEATURE_SCHEMA_VERSION = 1
PARAMETER_COUNT = 95
TRAIN_SEED_BASE = 452_000_000
VALIDATION_SEED_BASE = 552_000_000
FINAL_SEED_BASE = 652_000_000
DIAGNOSTIC_SEED_BASE = 752_000_000
VERIFICATION_SEED_BASE = 954_000_000
SEED_RANGE_STRIDE = 100_000


@dataclass(frozen=True)
class SeedManifest:
    pool: str
    seeds: tuple[int, ...]
    sha256: str
    discovery_start: int


def _seed_identity(seeds: Sequence[int]) -> bytes:
    return "".join(f"{seed}\n" for seed in seeds).encode("ascii")


def seed_manifest(pool: str, seeds: Iterable[int], discovery_start: int) -> SeedManifest:
    materialized = tuple(int(seed) for seed in seeds)
    if not materialized:
        raise ValueError("seed manifest must not be empty")
    if len(set(materialized)) != len(materialized):
        raise ValueError("seed manifest contains duplicates")
    if any(seed < 0 or seed > 0xFFFFFFFF for seed in materialized):
        raise ValueError("seed must fit uint32")
    return SeedManifest(
        pool=pool,
        seeds=materialized,
        sha256=sha256(_seed_identity(materialized)).hexdigest(),
        discovery_start=discovery_start,
    )


def assert_disjoint_seed_manifests(manifests: Sequence[SeedManifest]) -> None:
    owners: dict[int, str] = {}
    for manifest in manifests:
        for seed in manifest.seeds:
            previous = owners.get(seed)
            if previous is not None:
                raise ValueError(f"seed overlap between {previous} and {manifest.pool}: {seed}")
            owners[seed] = manifest.pool


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_seed_manifest(path: Path, manifest: SeedManifest) -> None:
    save_json(path, {**asdict(manifest), "seeds": list(manifest.seeds)})


def write_seed_file(path: Path, manifest: SeedManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_seed_identity(manifest.seeds))


def discover_seeds(
    evaluator: Path,
    *,
    start: int,
    count: int,
    cwd: Path,
    max_attempts: int | None = None,
) -> tuple[int, ...]:
    command = [
        str(evaluator),
        "--discover-start",
        str(start),
        "--discover-count",
        str(count),
        "--max-attempts",
        str(max_attempts or count * 20),
    ]
    completed = subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True)
    seeds = tuple(int(line) for line in completed.stdout.splitlines() if line.strip())
    if len(seeds) != count:
        raise RuntimeError(f"seed discovery returned {len(seeds)} seeds, expected {count}")
    return seeds


class EvaluationServer:
    """One ONNX-loaded process evaluating every candidate on one exact seed manifest."""

    def __init__(self, evaluator: Path, manifest: SeedManifest, *, cwd: Path, work: Path) -> None:
        self._seed_file = work / f"{manifest.pool}-{manifest.sha256[:12]}.seeds"
        write_seed_file(self._seed_file, manifest)
        self._process = subprocess.Popen(
            [str(evaluator), "--seeds", str(self._seed_file), "--server"],
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.manifest = manifest

    def evaluate(
        self,
        parameters: Sequence[float] | None,
        *,
        detailed: bool = False,
    ) -> dict[str, Any]:
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("evaluation server pipes are unavailable")
        if parameters is None:
            payload = "rule-based"
        else:
            if len(parameters) != PARAMETER_COUNT:
                raise ValueError(f"expected {PARAMETER_COUNT} parameters")
            if not all(math.isfinite(float(value)) for value in parameters):
                raise ValueError("parameters must be finite")
            payload = ",".join(format(float(value), ".17g") for value in parameters)
        mode = "detailed" if detailed else "summary"
        self._process.stdin.write(f"{mode}\t{payload}\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            return_code = self._process.poll()
            raise RuntimeError(f"evaluation server stopped unexpectedly (exit={return_code})")
        report = cast(dict[str, Any], json.loads(line))
        if report["seedHash"] != _cpp_seed_hash(self.manifest.seeds):
            raise RuntimeError("evaluation server used a different seed batch")
        if report["gameCount"] != len(self.manifest.seeds):
            raise RuntimeError("evaluation server game count differs from seed batch")
        return report

    def close(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
        return_code = self._process.wait(timeout=30)
        if return_code != 0:
            raise RuntimeError(f"evaluation server exited with {return_code}")

    def __enter__(self) -> EvaluationServer:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is not None:
            self._process.terminate()
            self._process.wait(timeout=30)
            return
        self.close()


def _cpp_seed_hash(seeds: Sequence[int]) -> str:
    value = 1469598103934665603
    for byte in _seed_identity(seeds):
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def create_cma_strategy(
    *,
    parameter_count: int = PARAMETER_COUNT,
    sigma: float,
    population_size: int,
    optimizer_seed: int,
    initial_mean: Sequence[float] | None = None,
) -> cma.CMAEvolutionStrategy:
    if parameter_count <= 0 or sigma <= 0 or population_size <= 1:
        raise ValueError("invalid CMA-ES configuration")
    mean = np.zeros(parameter_count) if initial_mean is None else np.asarray(initial_mean)
    if mean.shape != (parameter_count,) or not np.isfinite(mean).all():
        raise ValueError("invalid CMA-ES initial mean")
    return cma.CMAEvolutionStrategy(
        mean,
        sigma,
        {
            "bounds": [-40.0, 40.0],
            "popsize": population_size,
            "seed": optimizer_seed,
            "verbose": -9,
        },
    )


def save_optimizer_state(path: Path, strategy: cma.CMAEvolutionStrategy) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(strategy, protocol=pickle.HIGHEST_PROTOCOL))


def load_optimizer_state(path: Path) -> cma.CMAEvolutionStrategy:
    strategy = pickle.loads(path.read_bytes())  # noqa: S301 - trusted local experiment artifact
    if not isinstance(strategy, cma.CMAEvolutionStrategy):
        raise TypeError("optimizer state is not CMAEvolutionStrategy")
    return strategy


def parameter_artifact(
    schema: dict[str, Any],
    initial: Sequence[float],
    learned: Sequence[float],
    *,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    if schema.get("parameterCount") != PARAMETER_COUNT:
        raise ValueError("feature schema parameter count mismatch")
    if len(initial) != PARAMETER_COUNT or len(learned) != PARAMETER_COUNT:
        raise ValueError("parameter artifact dimension mismatch")
    rows = []
    for definition, initial_value, learned_value in zip(
        schema["features"], initial, learned, strict=True
    ):
        rows.append(
            {
                **definition,
                "initialWeight": float(initial_value),
                "weight": float(learned_value),
                "delta": float(learned_value) - float(initial_value),
            }
        )
    payload = {
        "artifactType": "parameterized-adjutant-exchange-policy",
        "featureSchemaVersion": FEATURE_SCHEMA_VERSION,
        "parameterCount": PARAMETER_COUNT,
        "weights": rows,
        "provenance": provenance,
    }
    payload["sha256"] = parameter_artifact_checksum(payload)
    return payload


def parameter_artifact_checksum(payload: dict[str, Any]) -> str:
    canonical_payload = {key: value for key, value in payload.items() if key != "sha256"}
    canonical = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return sha256(canonical).hexdigest()


def load_parameter_artifact(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_checksum = payload.get("sha256")
    if not isinstance(expected_checksum, str) or expected_checksum != parameter_artifact_checksum(
        payload
    ):
        raise ValueError("parameter artifact checksum mismatch")
    if payload.get("parameterCount") != PARAMETER_COUNT:
        raise ValueError("parameter artifact count mismatch")
    result = np.asarray([row["weight"] for row in payload["weights"]], dtype=np.float64)
    if result.shape != (PARAMETER_COUNT,) or not np.isfinite(result).all():
        raise ValueError("invalid parameter artifact weights")
    return result


def _sample_variance(values: Sequence[float]) -> float:
    return float(np.var(np.asarray(values, dtype=np.float64), ddof=1)) if len(values) > 1 else 0.0


def paired_comparison(
    learned_report: dict[str, Any], baseline_report: dict[str, Any]
) -> dict[str, Any]:
    learned_rows = learned_report.get("perSeed")
    baseline_rows = baseline_report.get("perSeed")
    if not isinstance(learned_rows, list) or not isinstance(baseline_rows, list):
        raise ValueError("detailed per-seed reports are required")
    learned_seeds = [int(row["seed"]) for row in learned_rows]
    baseline_seeds = [int(row["seed"]) for row in baseline_rows]
    if len(set(learned_seeds)) != len(learned_seeds):
        raise ValueError("learned report contains duplicate seeds")
    if len(set(baseline_seeds)) != len(baseline_seeds):
        raise ValueError("baseline report contains duplicate seeds")
    if learned_seeds != baseline_seeds:
        raise ValueError("paired reports must use the same ordered seed sequence")
    baseline_by_seed = {int(row["seed"]): row for row in baseline_rows}
    differences = []
    wins = ties = losses = 0
    for learned in learned_rows:
        seed = int(learned["seed"])
        baseline = baseline_by_seed.get(seed)
        if baseline is None:
            raise ValueError(f"baseline missing seed {seed}")
        difference = float(learned["relativeReward"]) - float(baseline["relativeReward"])
        differences.append(difference)
        wins += difference > 0
        ties += difference == 0
        losses += difference < 0
    if len(differences) != len(baseline_rows):
        raise ValueError("paired reports have different seed sets")
    mean = float(np.mean(differences))
    variance = _sample_variance(differences)
    standard_error = math.sqrt(variance / len(differences))
    return {
        "gameCount": len(differences),
        "meanDifference": mean,
        "standardError": standard_error,
        "ci95": [mean - 1.96 * standard_error, mean + 1.96 * standard_error],
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "perSeedDifferences": differences,
    }


def paired_block_comparisons(
    learned_report: dict[str, Any],
    baseline_report: dict[str, Any],
    *,
    block_size: int,
) -> list[dict[str, Any]]:
    """Calculate paired statistics for consecutive, pre-declared seed blocks."""
    if block_size <= 0:
        raise ValueError("block size must be positive")
    learned_rows = learned_report.get("perSeed")
    baseline_rows = baseline_report.get("perSeed")
    if not isinstance(learned_rows, list) or not isinstance(baseline_rows, list):
        raise ValueError("detailed per-seed reports are required")
    if len(learned_rows) != len(baseline_rows):
        raise ValueError("paired reports must use the same ordered seed sequence")
    if len(learned_rows) % block_size != 0:
        raise ValueError("game count must be divisible by block size")
    blocks = []
    for start in range(0, len(learned_rows), block_size):
        stop = start + block_size
        comparison = paired_comparison(
            {"perSeed": learned_rows[start:stop]},
            {"perSeed": baseline_rows[start:stop]},
        )
        comparison.pop("perSeedDifferences")
        blocks.append(
            {
                "block": start // block_size,
                "startIndex": start,
                "stopIndexExclusive": stop,
                "firstSeed": int(learned_rows[start]["seed"]),
                "lastSeed": int(learned_rows[stop - 1]["seed"]),
                **comparison,
            }
        )
    return blocks


def variance_diagnostic(
    common_differences: Sequence[float], independent_differences: Sequence[float]
) -> dict[str, Any]:
    common_variance = _sample_variance(common_differences)
    independent_variance = _sample_variance(independent_differences)
    reduction = 1.0 - common_variance / independent_variance if independent_variance > 0 else 0.0
    return {
        "batchCount": len(common_differences),
        "commonSeedDifferenceVariance": common_variance,
        "independentSeedDifferenceVariance": independent_variance,
        "varianceRatioCommonOverIndependent": (
            common_variance / independent_variance if independent_variance > 0 else None
        ),
        "varianceReductionFraction": reduction,
        "commonSeedDifferences": list(common_differences),
        "independentSeedDifferences": list(independent_differences),
    }
