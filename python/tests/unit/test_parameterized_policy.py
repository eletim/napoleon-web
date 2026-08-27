from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from napoleon_ml.cli.optimize_parameterized_policy import (
    ISSUE452_PARAMETER_SHA256,
    _assert_feature_schema_parity,
    _plateau_reached,
    _restore_checkpoint_progress,
    _seed_manifests_under,
    _validate_cli_args,
    _validate_final_provenance,
    _validate_resume_state,
)
from napoleon_ml.parameterized_policy.optimization import (
    PARAMETER_COUNT,
    assert_disjoint_seed_manifests,
    create_cma_strategy,
    load_optimizer_state,
    load_parameter_artifact,
    paired_block_comparisons,
    paired_comparison,
    parameter_artifact,
    parameter_artifact_checksum,
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


def test_issue452_frozen_winner_and_feature_schema_are_unchanged() -> None:
    repo = Path(__file__).resolve().parents[3]
    root = repo / "benchmarks/exchange-values/issue452-parameterized-policy/main"
    artifact = json.loads((root / "best-parameters.json").read_text(encoding="utf-8"))
    schema = json.loads((root / "feature-schema.json").read_text(encoding="utf-8"))

    assert artifact["sha256"] == ISSUE452_PARAMETER_SHA256
    assert parameter_artifact_checksum(artifact) == ISSUE452_PARAMETER_SHA256
    assert len(load_parameter_artifact(root / "best-parameters.json")) == PARAMETER_COUNT
    assert sum(row["block"] == "adjutant" for row in artifact["weights"]) == 35
    assert sum(row["block"] == "exchange" for row in artifact["weights"]) == 60
    _assert_feature_schema_parity(schema, schema, artifact)

    promoted_path = (
        repo
        / "benchmarks/non-playing-policies/parameterized-adjutant-exchange-v1/policy.json"
    )
    promoted = json.loads(promoted_path.read_text(encoding="utf-8"))
    assert promoted["parameterSha256"] == ISSUE452_PARAMETER_SHA256
    assert parameter_artifact_checksum(promoted) == promoted["sha256"]
    assert load_parameter_artifact(promoted_path) == pytest.approx(
        load_parameter_artifact(root / "best-parameters.json")
    )
    assert promoted["runtimeWiringIncluded"] is False


def test_issue454_verification_seeds_are_disjoint_from_every_issue452_pool() -> None:
    repo = Path(__file__).resolve().parents[3]
    reserved = _seed_manifests_under(
        repo / "benchmarks/exchange-values/issue452-parameterized-policy"
    )
    manifest_path = (
        repo
        / "benchmarks/exchange-values/issue454-independent-verification/seeds"
        / "independent-verification.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    verification = seed_manifest(
        payload["pool"], payload["seeds"], payload["discovery_start"]
    )

    assert len(reserved) == 122
    assert sum(len(manifest.seeds) for manifest in reserved) == 57_100
    assert len(verification.seeds) == 10_000
    assert verification.sha256 == payload["sha256"]
    assert verification.sha256 == (
        "0408442b7a3fa7dbb0521e4d15755262dd3af21aaf7478e1f5201c71312bff56"
    )
    assert_disjoint_seed_manifests([*reserved, verification])

    report = json.loads(
        (
            repo
            / "benchmarks/exchange-values/issue454-independent-verification"
            / "verification-report.json"
        ).read_text(encoding="utf-8")
    )
    for policy in (report["learned"], report["existingRuleBased"]):
        assert policy["invariantCheckCount"] == 120_000
        assert policy["invariantFailureCount"] == 0
        assert policy["illegalCount"] == 0
        assert policy["fallbackCount"] == 0


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

    strategy = create_cma_strategy(
        parameter_count=4, sigma=1.0, population_size=6, optimizer_seed=452
    )
    checkpoint = {
        "history": [],
        "optimizerGeneration": 0,
        "plateauState": {
            "bestSeenValidationFitness": 1.25,
            "noImprovementGenerations": 7,
        },
    }
    assert _restore_checkpoint_progress(checkpoint, strategy) == (0, 7, 1.25)
    assert _plateau_reached(45, 20, 12, 12)
    assert not _plateau_reached(19, 20, 12, 12)
    with pytest.raises(ValueError, match="generation mismatch"):
        _restore_checkpoint_progress({**checkpoint, "history": [{}]}, strategy)

    train_path = tmp_path / "main" / "seeds" / "train.json"
    final_path = tmp_path / "final" / "seeds" / "final.json"
    save_json(train_path, {**asdict(seed_manifest("train", [1, 2], 1)), "seeds": [1, 2]})
    save_json(final_path, {**asdict(seed_manifest("final", [3, 4], 3)), "seeds": [3, 4]})
    reserved = _seed_manifests_under(tmp_path, exclude=tmp_path / "final")
    assert [manifest.pool for manifest in reserved] == ["train"]
    assert_disjoint_seed_manifests([*reserved, seed_manifest("new-final", [3, 4], 3)])
    with pytest.raises(ValueError, match="seed overlap"):
        assert_disjoint_seed_manifests([*reserved, seed_manifest("new-final", [2, 4], 2)])


def test_final_provenance_binds_runtime_run_state_and_seed_pools(tmp_path: Path) -> None:
    validation = seed_manifest("validation", [10, 11], 10)
    train = seed_manifest("train-generation-000", [20, 21], 20)
    weights = [0.0] * PARAMETER_COUNT
    state = {
        "config": {"runtimeIdentity": {"evaluator": "abc"}},
        "history": [{"trainSeedHash": train.sha256}],
        "incumbentWeights": weights,
        "validationSeedManifest": {"sha256": validation.sha256},
    }
    save_json(tmp_path / "run-state.json", state)
    artifact = {
        "weights": [{"weight": value} for value in weights],
        "provenance": {
            "runState": "run-state.json",
            "validationSeedHash": validation.sha256,
            "completedGenerations": 1,
        },
    }
    assert _validate_final_provenance(
        tmp_path / "best-parameters.json",
        artifact,
        [validation, train],
        {"evaluator": "abc"},
    ) == state
    with pytest.raises(ValueError, match="missing artifact-bound pools"):
        _validate_final_provenance(
            tmp_path / "best-parameters.json",
            artifact,
            [validation],
            {"evaluator": "abc"},
        )
    with pytest.raises(ValueError, match="runtime identity mismatch"):
        _validate_final_provenance(
            tmp_path / "best-parameters.json",
            artifact,
            [validation, train],
            {"evaluator": "different"},
        )


def test_optimize_rejects_non_positive_generation_count() -> None:
    _validate_cli_args(argparse.Namespace(command="optimize", population=8, generations=1))
    with pytest.raises(SystemExit, match="generations must be >= 1"):
        _validate_cli_args(argparse.Namespace(command="optimize", population=8, generations=0))
    with pytest.raises(SystemExit, match="generations must be >= 1"):
        _validate_cli_args(argparse.Namespace(command="optimize", population=8, generations=-1))


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
    blocks = paired_block_comparisons(learned, baseline, block_size=1)
    assert [block["meanDifference"] for block in blocks] == [1.0, 0.0, -1.0]
    with pytest.raises(ValueError, match="same ordered seed sequence"):
        paired_comparison(learned, {"perSeed": list(reversed(baseline["perSeed"]))})
    with pytest.raises(ValueError, match="same ordered seed sequence"):
        paired_block_comparisons(
            learned,
            {"perSeed": [*baseline["perSeed"], {"seed": 4, "relativeReward": 0.0}]},
            block_size=1,
        )
    diagnostic = variance_diagnostic([0.1, 0.2, 0.15], [-1.0, 1.0, 0.0])
    assert (
        diagnostic["commonSeedDifferenceVariance"] < diagnostic["independentSeedDifferenceVariance"]
    )
    assert diagnostic["varianceReductionFraction"] > 0
