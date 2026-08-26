from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from napoleon_ml.cli.optimize_parameterized_policy import (
    _seed_manifests_under,
    _validate_resume_state,
)
from napoleon_ml.parameterized_policy.optimization import (
    PARAMETER_COUNT,
    assert_disjoint_seed_manifests,
    create_cma_strategy,
    load_optimizer_state,
    load_parameter_artifact,
    paired_comparison,
    parameter_artifact,
    save_json,
    save_optimizer_state,
    seed_manifest,
    variance_diagnostic,
)


def test_seed_manifest_is_deterministic_and_pools_are_disjoint() -> None:
    first = seed_manifest("train", [10, 20, 30], 10)
    repeated = seed_manifest("train", [10, 20, 30], 10)
    validation = seed_manifest("validation", [40, 50], 40)
    assert first.sha256 == repeated.sha256
    assert_disjoint_seed_manifests([first, validation])
    with pytest.raises(ValueError, match="seed overlap"):
        assert_disjoint_seed_manifests([first, seed_manifest("final", [30, 60], 30)])


def _quadratic_generation(strategy: Any) -> list[list[float]]:
    candidates = strategy.ask()
    fitness = [float(np.dot(candidate, candidate)) for candidate in candidates]
    strategy.tell(candidates, fitness)
    return [np.asarray(candidate).tolist() for candidate in candidates]


def test_cma_optimizer_resume_matches_uninterrupted_run(tmp_path: Path) -> None:
    uninterrupted = create_cma_strategy(
        parameter_count=4, sigma=1.0, population_size=6, optimizer_seed=452
    )
    expected_first = _quadratic_generation(uninterrupted)
    expected_second = _quadratic_generation(uninterrupted)

    resumed = create_cma_strategy(
        parameter_count=4, sigma=1.0, population_size=6, optimizer_seed=452
    )
    actual_first = _quadratic_generation(resumed)
    state_path = tmp_path / "optimizer.pkl"
    save_optimizer_state(state_path, resumed)
    restored = load_optimizer_state(state_path)
    actual_second = _quadratic_generation(restored)
    assert np.asarray(actual_first) == pytest.approx(np.asarray(expected_first))
    assert np.asarray(actual_second) == pytest.approx(np.asarray(expected_second))
    assert restored.mean == pytest.approx(uninterrupted.mean)
    assert restored.sigma == pytest.approx(uninterrupted.sigma)


def test_parameter_artifact_round_trip(tmp_path: Path) -> None:
    features = [
        {"index": index, "block": "adjutant", "name": f"f{index}", "scale": 1.0}
        for index in range(PARAMETER_COUNT)
    ]
    schema = {"parameterCount": PARAMETER_COUNT, "features": features}
    initial = np.zeros(PARAMETER_COUNT)
    learned = np.arange(PARAMETER_COUNT, dtype=np.float64) / 10
    artifact = parameter_artifact(
        schema, initial.tolist(), learned.tolist(), provenance={"test": True}
    )
    path = tmp_path / "parameters.json"
    save_json(path, artifact)
    assert load_parameter_artifact(path) == pytest.approx(learned)
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["sha256"] == artifact["sha256"]
    parsed["weights"][0]["weight"] = 999.0
    path.write_text(json.dumps(parsed), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_parameter_artifact(path)


def test_resume_identity_and_recorded_seed_manifests_are_enforced(tmp_path: Path) -> None:
    validation = seed_manifest("validation", [40, 50], 40)
    config = {"population": 16, "runtimeIdentity": {"evaluator": "abc"}}
    state = {
        "config": config,
        "validationSeedManifest": {"sha256": validation.sha256},
    }
    _validate_resume_state(state, config, validation)
    with pytest.raises(ValueError, match="population"):
        _validate_resume_state(state, {**config, "population": 32}, validation)
    with pytest.raises(ValueError, match="validationSeedManifest"):
        _validate_resume_state(state, config, seed_manifest("validation", [60, 70], 60))

    train_path = tmp_path / "main" / "seeds" / "train.json"
    final_path = tmp_path / "final" / "seeds" / "final.json"
    save_json(train_path, {**asdict(seed_manifest("train", [1, 2], 1)), "seeds": [1, 2]})
    save_json(final_path, {**asdict(seed_manifest("final", [3, 4], 3)), "seeds": [3, 4]})
    reserved = _seed_manifests_under(tmp_path, exclude=tmp_path / "final")
    assert [manifest.pool for manifest in reserved] == ["train"]
    assert_disjoint_seed_manifests([*reserved, seed_manifest("new-final", [3, 4], 3)])
    with pytest.raises(ValueError, match="seed overlap"):
        assert_disjoint_seed_manifests([*reserved, seed_manifest("new-final", [2, 4], 2)])


def test_paired_comparison_and_variance_diagnostic() -> None:
    learned = {
        "perSeed": [
            {"seed": 1, "relativeReward": 2.0},
            {"seed": 2, "relativeReward": 1.0},
            {"seed": 3, "relativeReward": -1.0},
        ]
    }
    baseline = {
        "perSeed": [
            {"seed": 1, "relativeReward": 1.0},
            {"seed": 2, "relativeReward": 1.0},
            {"seed": 3, "relativeReward": 0.0},
        ]
    }
    paired = paired_comparison(learned, baseline)
    assert paired["meanDifference"] == pytest.approx(0.0)
    assert (paired["wins"], paired["ties"], paired["losses"]) == (1, 1, 1)
    diagnostic = variance_diagnostic([0.1, 0.2, 0.15], [-1.0, 1.0, 0.0])
    assert (
        diagnostic["commonSeedDifferenceVariance"] < diagnostic["independentSeedDifferenceVariance"]
    )
    assert diagnostic["varianceReductionFraction"] > 0
