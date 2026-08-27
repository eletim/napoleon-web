"""Optimize and evaluate the Issue #452 parameterized joint policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np

from napoleon_ml.parameterized_policy.optimization import (
    DIAGNOSTIC_SEED_BASE,
    FINAL_SEED_BASE,
    PARAMETER_COUNT,
    SEED_RANGE_STRIDE,
    TRAIN_SEED_BASE,
    VALIDATION_SEED_BASE,
    VERIFICATION_SEED_BASE,
    EvaluationServer,
    SeedManifest,
    assert_disjoint_seed_manifests,
    create_cma_strategy,
    discover_seeds,
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
    write_seed_manifest,
)

ISSUE452_PARAMETER_SHA256 = "d364aef0c48a1832bd6602d254d0440f6cb2e2cb50492cfb53934e0378a84d69"
ISSUE454_VERIFICATION_GAMES = 10_000


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_evaluator(repo: Path) -> Path:
    return repo / "packages/cpp-core/build-ort-release/napoleon_parameterized_eval_cli"


def _read_evaluator_json(evaluator: Path, flag: str, repo: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(evaluator), flag], cwd=repo, check=True, text=True, capture_output=True
    )
    return cast(dict[str, Any], json.loads(completed.stdout))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_identity(repo: Path, evaluator: Path) -> dict[str, str]:
    files = {
        "evaluator": evaluator,
        "biddingMargin": repo / "benchmarks/bidding-margin-policies/frozen-raise-v1/margin.onnx",
        "playingPolicy": repo / "benchmarks/playing-policies/ppo-separated-v1000/policy.onnx",
        "playingCritic": repo / "benchmarks/playing-policies/ppo-separated-v1000/critic.onnx",
    }
    return {name: _file_sha256(path) for name, path in files.items()}


def _read_seed_manifest(path: Path) -> SeedManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = seed_manifest(
        str(payload["pool"]), payload["seeds"], int(payload["discovery_start"])
    )
    if manifest.sha256 != payload.get("sha256"):
        raise ValueError(f"seed manifest checksum mismatch: {path}")
    return manifest


def _seed_manifests_under(root: Path, *, exclude: Path | None = None) -> list[SeedManifest]:
    excluded = exclude.resolve() if exclude is not None else None
    manifests = []
    for path in sorted(root.resolve().rglob("seeds/*.json")):
        if excluded is not None and path.is_relative_to(excluded):
            continue
        manifests.append(_read_seed_manifest(path))
    return manifests


def _validate_resume_state(
    state: dict[str, Any],
    expected_config: dict[str, Any],
    validation: SeedManifest,
) -> None:
    saved_config = state.get("config")
    if not isinstance(saved_config, dict):
        raise ValueError("resume state has no configuration")
    mismatches = [
        key for key, expected in expected_config.items() if saved_config.get(key) != expected
    ]
    saved_validation = state.get("validationSeedManifest")
    if (
        not isinstance(saved_validation, dict)
        or saved_validation.get("sha256") != validation.sha256
    ):
        mismatches.append("validationSeedManifest")
    if mismatches:
        raise ValueError("resume experiment identity mismatch: " + ", ".join(mismatches))


def _restore_checkpoint_progress(
    state: dict[str, Any], strategy: Any
) -> tuple[int, int, float]:
    history = state.get("history")
    if not isinstance(history, list):
        raise ValueError("resume state has no history")
    start_generation = len(history)
    saved_optimizer_generation = int(state.get("optimizerGeneration", -1))
    if (
        int(strategy.countiter) != start_generation
        or saved_optimizer_generation != start_generation
    ):
        raise ValueError(
            "optimizer checkpoint generation mismatch: "
            f"CMA={strategy.countiter}, state={saved_optimizer_generation}, "
            f"history={start_generation}"
        )
    plateau_state = state.get("plateauState")
    if not isinstance(plateau_state, dict):
        raise ValueError("resume state has no plateau state")
    no_improvement = int(plateau_state["noImprovementGenerations"])
    best_seen = float(plateau_state["bestSeenValidationFitness"])
    if no_improvement < 0 or not math.isfinite(best_seen):
        raise ValueError("resume plateau state is invalid")
    return start_generation, no_improvement, best_seen


def _validate_final_provenance(
    parameter_path: Path,
    artifact: dict[str, Any],
    manifests: Sequence[SeedManifest],
    runtime_identity: dict[str, str],
) -> dict[str, Any]:
    provenance = artifact.get("provenance")
    if not isinstance(provenance, dict) or not isinstance(provenance.get("runState"), str):
        raise ValueError("parameter artifact has no bound run state")
    state_path = (parameter_path.resolve().parent / provenance["runState"]).resolve()
    state = cast(dict[str, Any], json.loads(state_path.read_text(encoding="utf-8")))
    config = state.get("config")
    if not isinstance(config, dict) or config.get("runtimeIdentity") != runtime_identity:
        raise ValueError("final evaluation runtime identity mismatch")
    validation = state.get("validationSeedManifest")
    validation_hash = provenance.get("validationSeedHash")
    if not isinstance(validation, dict) or validation.get("sha256") != validation_hash:
        raise ValueError("parameter artifact validation provenance mismatch")
    history = state.get("history")
    if not isinstance(history, list) or provenance.get("completedGenerations") != len(history):
        raise ValueError("parameter artifact generation provenance mismatch")
    expected_hashes = {str(validation_hash)}
    expected_hashes.update(str(row["trainSeedHash"]) for row in history)
    recorded_hashes = {manifest.sha256 for manifest in manifests}
    missing_hashes = expected_hashes - recorded_hashes
    if missing_hashes:
        raise ValueError(
            "seed manifest root is missing artifact-bound pools: "
            + ", ".join(sorted(missing_hashes))
        )
    artifact_weights = np.asarray(
        [row["weight"] for row in artifact.get("weights", [])], dtype=np.float64
    )
    incumbent_weights = np.asarray(state.get("incumbentWeights", []), dtype=np.float64)
    if not np.array_equal(artifact_weights, incumbent_weights):
        raise ValueError("parameter artifact does not match its bound run-state incumbent")
    return state


def _plateau_reached(
    generation_count: int,
    minimum_generations: int,
    no_improvement_generations: int,
    patience: int,
) -> bool:
    return (
        generation_count >= minimum_generations
        and no_improvement_generations >= patience
    )


def _discover_manifest(
    evaluator: Path,
    repo: Path,
    output: Path,
    *,
    pool: str,
    start: int,
    games: int,
) -> SeedManifest:
    manifest = seed_manifest(
        pool,
        discover_seeds(evaluator, start=start, count=games, cwd=repo),
        start,
    )
    write_seed_manifest(output / "seeds" / f"{pool}.json", manifest)
    return manifest


def _actual(initial: np.ndarray, delta: Sequence[float]) -> np.ndarray:
    values = initial + np.asarray(delta, dtype=np.float64)
    if values.shape != (PARAMETER_COUNT,) or not np.isfinite(values).all():
        raise ValueError("invalid CMA candidate")
    return values


def _optimize(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    evaluator = args.evaluator.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    work = output / "work"
    work.mkdir(parents=True, exist_ok=True)
    schema = _read_evaluator_json(evaluator, "--schema", repo)
    initial_payload = _read_evaluator_json(evaluator, "--initial-parameters", repo)
    initial = np.asarray(initial_payload["weights"], dtype=np.float64)
    save_json(output / "feature-schema.json", schema)
    save_json(output / "initial-parameters.json", initial_payload)
    runtime_identity = _runtime_identity(repo, evaluator)
    resume_identity = {
        "optimizer": "pycma.CMAEvolutionStrategy",
        "population": args.population,
        "gamesPerCandidate": args.games_per_candidate,
        "validationGames": args.validation_games,
        "initialSigma": args.sigma,
        "boundsOnDelta": [-40.0, 40.0],
        "optimizerSeed": args.optimizer_seed,
        "seedOffset": args.seed_offset,
        "minimumGenerations": args.minimum_generations,
        "plateauPatience": args.plateau_patience,
        "plateauDelta": args.plateau_delta,
        "bidding": "frozen-raise-v1",
        "playing": "ppo-separated-v1000",
        "commonRandomNumbersWithinGeneration": True,
        "runtimeIdentity": runtime_identity,
    }

    validation = _discover_manifest(
        evaluator,
        repo,
        output,
        pool="validation",
        start=VALIDATION_SEED_BASE + args.seed_offset,
        games=args.validation_games,
    )
    if args.resume:
        state = json.loads((output / "run-state.json").read_text(encoding="utf-8"))
        _validate_resume_state(state, resume_identity, validation)
        strategy = load_optimizer_state(output / "optimizer-state.pkl")
        history: list[dict[str, Any]] = state["history"]
        incumbent = np.asarray(state["incumbentWeights"], dtype=np.float64)
        incumbent_validation = float(state["incumbentValidationFitness"])
        start_generation, no_improvement, best_seen = _restore_checkpoint_progress(
            state, strategy
        )
    else:
        warm_start = (
            load_parameter_artifact(args.warm_start.resolve())
            if args.warm_start is not None
            else initial.copy()
        )
        strategy = create_cma_strategy(
            sigma=args.sigma,
            population_size=args.population,
            optimizer_seed=args.optimizer_seed,
            initial_mean=(warm_start - initial).tolist(),
        )
        history = []
        incumbent = warm_start
        incumbent_validation = -math.inf
        start_generation = 0

    all_manifests = _seed_manifests_under(output)
    assert_disjoint_seed_manifests(all_manifests)
    with EvaluationServer(evaluator, validation, cwd=repo, work=work) as validation_server:
        if not args.resume:
            initial_validation = validation_server.evaluate(initial.tolist())
            rule_validation = validation_server.evaluate(None)
            warm_start_validation = validation_server.evaluate(incumbent.tolist())
            incumbent_validation = float(warm_start_validation["meanRelativeReward"])
            save_json(
                output / "initial-rule-based-approximation.json",
                {
                    "parameterizedInitial": initial_validation,
                    "existingRuleBased": rule_validation,
                    "meanDifference": initial_validation["meanRelativeReward"]
                    - rule_validation["meanRelativeReward"],
                    "warmStart": warm_start_validation,
                },
            )
            no_improvement = 0
            best_seen = incumbent_validation
        already_plateaued = (
            args.resume
            and _plateau_reached(
                start_generation,
                args.minimum_generations,
                no_improvement,
                args.plateau_patience,
            )
        )
        end_generation = start_generation if already_plateaued else args.generations
        for generation in range(start_generation, end_generation):
            started = time.monotonic()
            train = _discover_manifest(
                evaluator,
                repo,
                output,
                pool=f"train-generation-{generation:03d}",
                start=TRAIN_SEED_BASE + args.seed_offset + generation * SEED_RANGE_STRIDE,
                games=args.games_per_candidate,
            )
            all_manifests.append(train)
            assert_disjoint_seed_manifests(all_manifests)
            deltas = strategy.ask()
            fitnesses: list[float] = []
            reports: list[dict[str, Any]] = []
            with EvaluationServer(evaluator, train, cwd=repo, work=work) as train_server:
                for delta in deltas:
                    report = train_server.evaluate(_actual(initial, delta).tolist())
                    reports.append(report)
                    fitnesses.append(float(report["meanRelativeReward"]))
            strategy.tell(deltas, [-fitness for fitness in fitnesses])
            best_index = int(np.argmax(fitnesses))
            generation_best = _actual(initial, deltas[best_index])
            validation_report = validation_server.evaluate(generation_best.tolist())
            validation_fitness = float(validation_report["meanRelativeReward"])
            if validation_fitness > incumbent_validation:
                incumbent = generation_best
                incumbent_validation = validation_fitness
            improved = incumbent_validation > best_seen + args.plateau_delta
            if improved:
                best_seen = incumbent_validation
                no_improvement = 0
            else:
                no_improvement += 1
            elapsed = time.monotonic() - started
            generation_row = {
                "generation": generation,
                "trainSeedHash": train.sha256,
                "population": len(deltas),
                "gamesPerCandidate": args.games_per_candidate,
                "populationMeanFitness": float(np.mean(fitnesses)),
                "populationFitnessStd": float(np.std(fitnesses, ddof=1)),
                "generationBestFitness": fitnesses[best_index],
                "generationBestValidationFitness": validation_fitness,
                "incumbentValidationFitness": incumbent_validation,
                "parameterNorm": float(np.linalg.norm(incumbent - initial)),
                "sigma": float(strategy.sigma),
                "elapsedSeconds": elapsed,
                "illegalCount": sum(int(report["illegalCount"]) for report in reports),
                "fallbackCount": sum(int(report["fallbackCount"]) for report in reports),
            }
            history.append(generation_row)
            run_state = {
                "config": {
                    **resume_identity,
                    "requestedGenerations": args.generations,
                },
                "history": history,
                "incumbentWeights": incumbent.tolist(),
                "incumbentValidationFitness": incumbent_validation,
                "optimizerGeneration": int(strategy.countiter),
                "plateauState": {
                    "bestSeenValidationFitness": best_seen,
                    "noImprovementGenerations": no_improvement,
                },
                "validationSeedManifest": asdict(validation),
            }
            save_optimizer_state(output / "optimizer-state.pkl", strategy)
            save_json(output / "learning-curve.json", {"history": history})
            save_json(output / "run-state.json", run_state)
            print(json.dumps(generation_row, sort_keys=True), flush=True)
            if (
                generation + 1 >= args.minimum_generations
                and no_improvement >= args.plateau_patience
            ):
                save_json(
                    output / "plateau-stop.json",
                    {
                        "stoppedAfterGeneration": generation,
                        "patience": args.plateau_patience,
                        "minimumImprovement": args.plateau_delta,
                    },
                )
                break

    artifact = parameter_artifact(
        schema,
        initial.tolist(),
        incumbent.tolist(),
        provenance={
            "optimizer": "pycma.CMAEvolutionStrategy",
            "validationSeedHash": validation.sha256,
            "incumbentValidationFitness": incumbent_validation,
            "completedGenerations": len(history),
            "runState": "run-state.json",
        },
    )
    save_json(output / "best-parameters.json", artifact)


def _variance(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    evaluator = args.evaluator.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    work = output / "work"
    work.mkdir(parents=True, exist_ok=True)
    initial = np.asarray(
        _read_evaluator_json(evaluator, "--initial-parameters", repo)["weights"],
        dtype=np.float64,
    )
    learned = load_parameter_artifact(args.parameters.resolve())
    common_differences: list[float] = []
    independent_differences: list[float] = []
    manifests: list[SeedManifest] = []
    for batch in range(args.batches):
        base = DIAGNOSTIC_SEED_BASE + args.seed_offset + batch * 3 * SEED_RANGE_STRIDE
        common = _discover_manifest(
            evaluator, repo, output, pool=f"common-{batch:02d}", start=base, games=args.games
        )
        independent_a = _discover_manifest(
            evaluator,
            repo,
            output,
            pool=f"independent-a-{batch:02d}",
            start=base + SEED_RANGE_STRIDE,
            games=args.games,
        )
        independent_b = _discover_manifest(
            evaluator,
            repo,
            output,
            pool=f"independent-b-{batch:02d}",
            start=base + 2 * SEED_RANGE_STRIDE,
            games=args.games,
        )
        manifests.extend([common, independent_a, independent_b])
        assert_disjoint_seed_manifests(manifests)
        with EvaluationServer(evaluator, common, cwd=repo, work=work) as server:
            common_a = server.evaluate(learned.tolist())["meanRelativeReward"]
            common_b = server.evaluate(initial.tolist())["meanRelativeReward"]
        with EvaluationServer(evaluator, independent_a, cwd=repo, work=work) as server:
            independent_a_fitness = server.evaluate(learned.tolist())["meanRelativeReward"]
        with EvaluationServer(evaluator, independent_b, cwd=repo, work=work) as server:
            independent_b_fitness = server.evaluate(initial.tolist())["meanRelativeReward"]
        common_differences.append(float(common_a - common_b))
        independent_differences.append(float(independent_a_fitness - independent_b_fitness))
    report = variance_diagnostic(common_differences, independent_differences)
    report.update(
        {
            "parameterA": str(args.parameters),
            "parameterB": "initial-rule-based-approximation",
            "gamesPerBatchPerParameter": args.games,
            "allSeedPoolsDisjoint": True,
        }
    )
    save_json(output / "variance-diagnostic.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _selection_tendencies(report: dict[str, Any]) -> dict[str, Any]:
    rows = report["perSeed"]
    called = Counter(row["calledAdjutantCard"] for row in rows)
    buried = Counter(card for row in rows for card in row["buriedCards"].split(","))
    return {
        "calledAdjutantCardCounts": dict(called.most_common()),
        "buriedCardCounts": dict(buried.most_common()),
    }


def _buried_content_distribution(report: dict[str, Any]) -> dict[str, Any]:
    rows = report["perSeed"]
    rank_counts: Counter[str] = Counter()
    suit_counts: Counter[str] = Counter()
    point_count = 0
    trump_count = 0
    for row in rows:
        trump = str(row["contractSuit"])
        for card in str(row["buriedCards"]).split(","):
            if card == "joker":
                rank_counts["joker"] += 1
                suit_counts["joker"] += 1
                continue
            suit, rank = card.split("-", 1)
            rank_counts[rank] += 1
            suit_counts[suit] += 1
            point_count += rank in {"10", "J", "Q", "K", "A"}
            trump_count += suit == trump
    game_count = len(rows)
    return {
        "rankCounts": dict(sorted(rank_counts.items())),
        "suitCounts": dict(sorted(suit_counts.items())),
        "pointCardCount": point_count,
        "meanPointCardsPerGame": point_count / game_count,
        "trumpCardCount": trump_count,
        "meanTrumpCardsPerGame": trump_count / game_count,
    }


def _assert_feature_schema_parity(
    runtime_schema: dict[str, Any],
    recorded_schema: dict[str, Any],
    parameter_artifact_payload: dict[str, Any],
) -> None:
    if runtime_schema != recorded_schema:
        raise ValueError("#452 recorded feature schema differs from evaluator schema")
    feature_keys = set(runtime_schema["features"][0])
    artifact_features = [
        {key: row[key] for key in feature_keys} for row in parameter_artifact_payload["weights"]
    ]
    if artifact_features != runtime_schema["features"]:
        raise ValueError("parameter feature definitions differ from evaluator schema")
    if parameter_artifact_payload.get("featureSchemaVersion") != runtime_schema.get(
        "schemaVersion"
    ):
        raise ValueError("parameter feature schema version differs from evaluator schema")


def _weight_vector_sha256(weights: Sequence[float]) -> str:
    canonical = json.dumps(list(weights), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _verification(args: argparse.Namespace) -> None:
    """Independently re-evaluate the frozen #452 winner; never optimize or select weights."""
    repo = args.repo.resolve()
    evaluator = args.evaluator.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    work = output / "work"
    work.mkdir(parents=True, exist_ok=True)
    parameter_relative_path = Path(
        "benchmarks/exchange-values/issue452-parameterized-policy/main/best-parameters.json"
    )
    parameter_path = repo / parameter_relative_path
    parameter_payload = json.loads(parameter_path.read_text(encoding="utf-8"))
    learned = load_parameter_artifact(parameter_path)
    if parameter_payload.get("sha256") != ISSUE452_PARAMETER_SHA256:
        raise ValueError("verification parameter is not the frozen #452 winner")

    recorded_schema_path = parameter_path.parent / "feature-schema.json"
    runtime_schema = _read_evaluator_json(evaluator, "--schema", repo)
    recorded_schema = json.loads(recorded_schema_path.read_text(encoding="utf-8"))
    _assert_feature_schema_parity(runtime_schema, recorded_schema, parameter_payload)

    manifest_root = repo / "benchmarks/exchange-values/issue452-parameterized-policy"
    reserved_manifests = _seed_manifests_under(manifest_root, exclude=output)
    reserved_seed_count = sum(len(manifest.seeds) for manifest in reserved_manifests)
    reserved_unique_seed_count = len(
        {seed for manifest in reserved_manifests for seed in manifest.seeds}
    )
    if (
        len(reserved_manifests) != 122
        or reserved_seed_count != 57_100
        or reserved_unique_seed_count != 57_100
    ):
        raise ValueError("incomplete or overlapping frozen #452 seed manifest inventory")
    assert_disjoint_seed_manifests(reserved_manifests)
    runtime_identity = _runtime_identity(repo, evaluator)
    source_state_path = parameter_path.parent / parameter_payload["provenance"]["runState"]
    source_state = cast(
        dict[str, Any], json.loads(source_state_path.read_text(encoding="utf-8"))
    )
    source_runtime_identity = source_state["config"]["runtimeIdentity"]
    bound_state = _validate_final_provenance(
        parameter_path, parameter_payload, reserved_manifests, source_runtime_identity
    )
    policy_dependency_keys = {"biddingMargin", "playingPolicy", "playingCritic"}
    if any(
        runtime_identity[key] != source_runtime_identity[key] for key in policy_dependency_keys
    ):
        raise ValueError("verification bidding/playing dependency identity mismatch")
    verification_manifest = _discover_manifest(
        evaluator,
        repo,
        output,
        pool="independent-verification",
        start=VERIFICATION_SEED_BASE,
        games=ISSUE454_VERIFICATION_GAMES,
    )
    assert_disjoint_seed_manifests([*reserved_manifests, verification_manifest])

    with EvaluationServer(evaluator, verification_manifest, cwd=repo, work=work) as server:
        learned_report = server.evaluate(learned.tolist(), detailed=True)
        baseline_report = server.evaluate(None, detailed=True)
    if learned_report["seedHash"] != baseline_report["seedHash"]:
        raise RuntimeError("paired policies did not use the same seed sequence")
    paired = paired_comparison(learned_report, baseline_report)
    blocks = paired_block_comparisons(
        learned_report, baseline_report, block_size=args.block_size
    )
    failures = sum(
        int(policy[key])
        for policy in (learned_report, baseline_report)
        for key in ("illegalCount", "fallbackCount", "invariantFailureCount")
    )
    adopted = paired["meanDifference"] > 0 and paired["ci95"][0] > 0 and failures == 0
    report = {
        "artifactType": "parameterized-policy-independent-verification",
        "issue": 454,
        "featureSchemaVersion": runtime_schema["schemaVersion"],
        "parameterCount": PARAMETER_COUNT,
        "bestParameterSha256": parameter_payload["sha256"],
        "weightVectorSha256": _weight_vector_sha256(learned.tolist()),
        "verificationSeedManifest": asdict(verification_manifest),
        "seedAudit": {
            "reservedManifestRoot": str(manifest_root.relative_to(repo)),
            "reservedManifestCount": len(reserved_manifests),
            "reservedSeedCount": reserved_seed_count,
            "reservedUniqueSeedCount": reserved_unique_seed_count,
            "verificationSeedCount": len(verification_manifest.seeds),
            "overlapCount": 0,
        },
        "fixedSemantics": {
            "bidding": "frozen-raise-v1",
            "playing": "ppo-separated-v1000",
            "reward": "current-relative-reward",
            "phaseOrder": ["bidding", "adjutant", "kitty", "exchange", "playing"],
            "pairedPlayingRandomStream": True,
        },
        "dependencyProvenance": {
            **runtime_identity,
            "sourceIssue452Evaluator": source_runtime_identity["evaluator"],
            "verificationEvaluator": runtime_identity["evaluator"],
        },
        "optimizerProvenance": {
            "sourceIssue": 452,
            "sourceParameterPath": str(parameter_relative_path),
            "parameterArtifact": parameter_payload["provenance"],
            "runConfiguration": bound_state["config"],
        },
        "learned": learned_report,
        "existingRuleBased": baseline_report,
        "pairedComparison": paired,
        "blockSize": args.block_size,
        "pairedBlockComparisons": blocks,
        "learnedSelectionTendencies": _selection_tendencies(learned_report),
        "ruleBasedSelectionTendencies": _selection_tendencies(baseline_report),
        "learnedBuriedContentDistribution": _buried_content_distribution(learned_report),
        "ruleBasedBuriedContentDistribution": _buried_content_distribution(baseline_report),
        "adoptionCriteria": {
            "positivePairedMean": paired["meanDifference"] > 0,
            "ci95LowerBoundAboveZero": paired["ci95"][0] > 0,
            "allFailureCountsZero": failures == 0,
            "adoptedAsFormalArtifactCandidate": adopted,
        },
        "runtimeWiringIncluded": False,
    }
    report_path = output / "verification-report.json"
    save_json(report_path, report)

    if adopted and args.artifact_output is not None:
        artifact_output = args.artifact_output.resolve()
        artifact_output.mkdir(parents=True, exist_ok=True)
        promoted = {
            "artifactType": "parameterized-adjutant-exchange-policy-candidate",
            "artifactVersion": 1,
            "featureSchemaVersion": runtime_schema["schemaVersion"],
            "parameterCount": PARAMETER_COUNT,
            "parameterSha256": parameter_payload["sha256"],
            "weightVectorSha256": report["weightVectorSha256"],
            "weights": parameter_payload["weights"],
            "optimizerProvenance": report["optimizerProvenance"],
            "verificationProvenance": {
                "sourceIssue": 454,
                "seedManifest": str(args.output / "seeds/independent-verification.json"),
                "seedManifestSha256": verification_manifest.sha256,
                "seedManifestFileSha256": _file_sha256(
                    output / "seeds/independent-verification.json"
                ),
                "report": str(args.output / "verification-report.json"),
                "reportFileSha256": _file_sha256(report_path),
                "games": ISSUE454_VERIFICATION_GAMES,
                "pairedMeanDifference": paired["meanDifference"],
                "pairedStandardError": paired["standardError"],
                "pairedCi95": paired["ci95"],
            },
            "dependencyProvenance": report["dependencyProvenance"],
            "runtimeWiringIncluded": False,
        }
        promoted["sha256"] = parameter_artifact_checksum(promoted)
        save_json(artifact_output / "policy.json", promoted)
        save_json(artifact_output / "feature-schema.json", runtime_schema)

    print(
        json.dumps(
            {
                "adoptedAsFormalArtifactCandidate": adopted,
                "learnedMeanRelativeReward": learned_report["meanRelativeReward"],
                "ruleBasedMeanRelativeReward": baseline_report["meanRelativeReward"],
                "pairedComparison": {k: v for k, v in paired.items() if k != "perSeedDifferences"},
                "pairedBlockComparisons": blocks,
                "verificationSeedSha256": verification_manifest.sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _final(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    evaluator = args.evaluator.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    work = output / "work"
    work.mkdir(parents=True, exist_ok=True)
    parameter_path = args.parameters.resolve()
    learned = load_parameter_artifact(parameter_path)
    artifact = json.loads(parameter_path.read_text(encoding="utf-8"))
    manifest_root = (
        args.seed_manifest_root.resolve()
        if args.seed_manifest_root is not None
        else args.parameters.resolve().parent.parent
    )
    reserved_manifests = _seed_manifests_under(manifest_root, exclude=output)
    if not reserved_manifests:
        raise ValueError(f"no reserved seed manifests found under {manifest_root}")
    assert_disjoint_seed_manifests(reserved_manifests)
    bound_state = _validate_final_provenance(
        parameter_path,
        artifact,
        reserved_manifests,
        _runtime_identity(repo, evaluator),
    )
    final_manifest = _discover_manifest(
        evaluator, repo, output, pool="final-holdout", start=FINAL_SEED_BASE, games=args.games
    )
    assert_disjoint_seed_manifests([*reserved_manifests, final_manifest])
    with EvaluationServer(evaluator, final_manifest, cwd=repo, work=work) as server:
        learned_report = server.evaluate(learned.tolist(), detailed=True)
        baseline_report = server.evaluate(None, detailed=True)
    paired = paired_comparison(learned_report, baseline_report)
    largest_changes = sorted(
        artifact["weights"], key=lambda row: abs(float(row["delta"])), reverse=True
    )[:20]
    report = {
        "featureSchemaVersion": 1,
        "bestParameterSha256": artifact["sha256"],
        "finalSeedManifest": asdict(final_manifest),
        "seedOverlapWithReservedTrainValidationRanges": False,
        "provenanceAudit": {
            "boundRunState": artifact["provenance"]["runState"],
            "boundTrainGenerationCount": len(bound_state["history"]),
            "auditedReservedManifestCount": len(reserved_manifests),
            "runtimeIdentity": bound_state["config"]["runtimeIdentity"],
        },
        "learned": learned_report,
        "existingRuleBased": baseline_report,
        "pairedComparison": paired,
        "largestWeightChanges": largest_changes,
        "learnedSelectionTendencies": _selection_tendencies(learned_report),
        "ruleBasedSelectionTendencies": _selection_tendencies(baseline_report),
    }
    save_json(output / "final-evaluation.json", report)
    print(
        json.dumps(
            {
                "learnedMeanRelativeReward": learned_report["meanRelativeReward"],
                "ruleBasedMeanRelativeReward": baseline_report["meanRelativeReward"],
                "pairedComparison": paired,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _parser() -> argparse.ArgumentParser:
    repo = _repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--evaluator", type=Path, default=_default_evaluator(repo))
    subparsers = parser.add_subparsers(dest="command", required=True)

    optimize = subparsers.add_parser("optimize")
    optimize.add_argument("--output", type=Path, required=True)
    optimize.add_argument("--population", type=int, default=12)
    optimize.add_argument("--generations", type=int, default=8)
    optimize.add_argument("--games-per-candidate", type=int, default=200)
    optimize.add_argument("--validation-games", type=int, default=500)
    optimize.add_argument("--sigma", type=float, default=2.0)
    optimize.add_argument("--optimizer-seed", type=int, default=452)
    optimize.add_argument("--seed-offset", type=int, default=0)
    optimize.add_argument("--minimum-generations", type=int, default=5)
    optimize.add_argument("--plateau-patience", type=int, default=12)
    optimize.add_argument("--plateau-delta", type=float, default=0.01)
    optimize.add_argument("--resume", action="store_true")
    optimize.add_argument("--warm-start", type=Path)
    optimize.set_defaults(run=_optimize)

    variance = subparsers.add_parser("variance")
    variance.add_argument("--parameters", type=Path, required=True)
    variance.add_argument("--output", type=Path, required=True)
    variance.add_argument("--batches", type=int, default=10)
    variance.add_argument("--games", type=int, default=200)
    variance.add_argument("--seed-offset", type=int, default=0)
    variance.set_defaults(run=_variance)

    final = subparsers.add_parser("final")
    final.add_argument("--parameters", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    final.add_argument("--games", type=int, default=5000)
    final.add_argument("--seed-manifest-root", type=Path)
    final.set_defaults(run=_final)

    verification = subparsers.add_parser("verification")
    verification.add_argument("--output", type=Path, required=True)
    verification.add_argument("--artifact-output", type=Path)
    verification.add_argument("--block-size", type=int, default=1_000)
    verification.set_defaults(run=_verification)
    return parser


def _validate_cli_args(args: argparse.Namespace) -> None:
    if getattr(args, "population", 2) < 2:
        raise SystemExit("population must be >= 2")
    if args.command == "optimize" and args.generations < 1:
        raise SystemExit("generations must be >= 1")
    if args.command == "verification":
        if (
            args.block_size <= 0
            or ISSUE454_VERIFICATION_GAMES % args.block_size != 0
        ):
            raise SystemExit("verification games must be divisible by a positive block size")


def main() -> None:
    args = _parser().parse_args()
    _validate_cli_args(args)
    args.run(args)


if __name__ == "__main__":
    main()
