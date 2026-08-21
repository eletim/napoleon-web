from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest

from napoleon_ml.bidding.ppo import advantage_normalization_metadata
from napoleon_ml.cli.run_nonplaying_rl import (
    _iterative_config_from_args,
    _provided_iterative_config_keys,
    build_parser,
)
from napoleon_ml.nonplaying_rl_orchestrator import (
    DEFAULT_BIDDING_ADVANTAGE_NORMALIZATION,
    DEFAULT_BIDDING_ENTROPY_COEFFICIENT,
    DEFAULT_BIDDING_MINIBATCH_STRATEGY,
    DEFAULT_ITERATIVE_BATCH_SIZE,
    DEFAULT_ITERATIVE_EVALUATION_GAMES,
    NONPLAYING_ALL_PASS_RULE_ID,
    NONPLAYING_REWARD_ID,
    NONPLAYING_REWARD_TYPE,
    NONPLAYING_REWARD_VERSION,
    NONPLAYING_TERMINAL_REWARD_TRANSFORM_ID,
    NONPLAYING_TERMINAL_REWARD_TRANSFORM_TYPE,
    NONPLAYING_TERMINAL_REWARD_TRANSFORM_VERSION,
    NonPlayingIterativeRlRunConfig,
    NonPlayingRlOrchestratorError,
    NonPlayingRlRunConfig,
    _patch_cpp_bidding_manifest,
    _validate_iterative_resume_config,
    run_iterative_nonplaying_rl_pipeline,
    run_nonplaying_rl_pipeline,
)


def test_nonplaying_rl_pipeline_smoke_writes_summary(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    playing_onnx = tmp_path / "playing.onnx"
    playing_metadata = tmp_path / "playing.json"
    playing_onnx.write_bytes(b"onnx")
    playing_metadata.write_text("{}\n", encoding="utf-8")
    stages: list[str] = []

    monkeypatch.setattr(
        "napoleon_ml.nonplaying_rl_orchestrator._build_typescript_helpers",
        lambda: stages.append("typescript-build"),
    )

    def fake_rollout(*args: object, **kwargs: object) -> dict[str, object]:
        phase = kwargs["phase"]
        stages.append(f"{phase}-rollout")
        return {
            "phase": phase,
            "gameCount": 1,
            "sampleCount": 1,
            "shardCount": 1,
        }

    def fake_train(
        config: NonPlayingRlRunConfig,
        phase: str,
        dataset_dir: Path,
        checkpoint_path: Path,
        training_seed: int,
    ) -> dict[str, object]:
        stages.append(f"{phase}-train")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(b"checkpoint")
        return {
            "sampleCount": 1,
            "optimizerStepCount": 1,
            "outputCheckpointPath": str(checkpoint_path),
            "seed": training_seed,
        }

    def fake_export(
        phase: str,
        dataset_dir: Path,
        checkpoint_path: Path,
        onnx_path: Path,
        metadata_path: Path,
    ) -> dict[str, object]:
        stages.append(f"{phase}-export")
        onnx_path.write_bytes(b"onnx")
        metadata_path.write_text("{}\n", encoding="utf-8")
        return {
            "policyType": phase,
            "onnxPath": str(onnx_path),
            "metadataPath": str(metadata_path),
        }

    monkeypatch.setattr(
        "napoleon_ml.nonplaying_rl_orchestrator.export_seeded_nonplaying_bootstrap_policy_to_onnx",
        lambda **kwargs: {"policyType": kwargs["policy_type"]},
    )
    monkeypatch.setattr(
        "napoleon_ml.nonplaying_rl_orchestrator._run_nonplaying_rollout",
        fake_rollout,
    )
    monkeypatch.setattr("napoleon_ml.nonplaying_rl_orchestrator._train_phase", fake_train)
    monkeypatch.setattr(
        "napoleon_ml.nonplaying_rl_orchestrator._export_trained_phase",
        fake_export,
    )
    monkeypatch.setattr(
        "napoleon_ml.nonplaying_rl_orchestrator._run_full_policy_evaluation",
        lambda *args, **kwargs: {
            "scheduledGames": 5,
            "completedGames": 5,
            "failedGames": 0,
            "fallbackCount": 0,
            "illegalActionCount": 0,
            "policyAgentDecisionCounts": {
                "biddingOnnxCallCount": 5,
                "adjutantOnnxCallCount": 5,
                "exchangeOnnxCallCount": 15,
                "playingOnnxCallCount": 100,
                "ruleBasedFallbackDecisionCount": 0,
            },
        },
    )

    output_dir = tmp_path / "run"
    summary = run_nonplaying_rl_pipeline(
        NonPlayingRlRunConfig(
            output_dir=output_dir,
            games=1,
            evaluation_games=1,
            games_per_shard=1,
            bidding_hidden_dims=(16, 12),
            training_device="auto",
            playing_policy_onnx=playing_onnx,
            playing_policy_metadata=playing_metadata,
        )
    )

    assert (output_dir / "run-summary.json").is_file()
    written = json.loads((output_dir / "run-summary.json").read_text(encoding="utf-8"))
    assert written["settings"]["trainingDevice"] == "auto"
    assert written["settings"]["biddingHiddenDims"] == [16, 12]
    assert written["settings"]["biddingMinibatchStrategy"] == DEFAULT_BIDDING_MINIBATCH_STRATEGY
    assert written["evaluation"]["fallbackCount"] == 0
    assert written["evaluation"]["illegalActionCount"] == 0
    assert written["artifactPaths"]["bidding"]["onnxPath"].endswith("bidding/policy.onnx")
    assert summary["artifactPaths"] == written["artifactPaths"]
    assert stages == [
        "typescript-build",
        "bidding-rollout",
        "bidding-train",
        "bidding-export",
        "adjutant-rollout",
        "adjutant-train",
        "adjutant-export",
        "exchange-rollout",
        "exchange-train",
        "exchange-export",
    ]


def test_cpp_bidding_manifest_patch_preserves_rule_based_phase_metadata(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    policy_onnx = tmp_path / "bidding.onnx"
    policy_metadata = tmp_path / "bidding.json"
    playing_onnx = tmp_path / "playing.onnx"
    playing_metadata = tmp_path / "playing.json"
    policy_onnx.write_bytes(b"bidding-onnx")
    policy_metadata.write_text('{"policyType":"bidding"}\n', encoding="utf-8")
    playing_onnx.write_bytes(b"playing-onnx")
    playing_metadata.write_text('{"policyType":"playing"}\n', encoding="utf-8")
    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "behaviorPolicy": {},
                "fixedPlayingPolicy": {},
                "nonLearningAgents": {
                    "bidding": {"type": "mixed-frozen-bidding"},
                    "choosingAdjutant": {"type": "rule-based", "version": 1},
                    "exchanging": {"type": "rule-based", "version": 1},
                    "playing": {"type": "playing-onnx"},
                },
                "diagnostics": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = NonPlayingRlRunConfig(
        output_dir=tmp_path / "run",
        games=1,
        playing_policy_onnx=playing_onnx,
        playing_policy_metadata=playing_metadata,
        playing_policy_artifact_id="ppo-separated-v1000",
    )

    _patch_cpp_bidding_manifest(
        config,
        dataset_dir=dataset_dir,
        policy_onnx=policy_onnx,
        policy_metadata=policy_metadata,
        start_seed=7,
        artifact_id="candidate-bidding",
    )

    patched = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    non_learning = patched["nonLearningAgents"]
    assert non_learning["choosingAdjutant"] == {"type": "rule-based", "version": 1}
    assert non_learning["exchanging"] == {"type": "rule-based", "version": 1}
    assert non_learning["bidding"]["type"] == "mixed-frozen-bidding"
    assert non_learning["playing"]["artifactId"] == "ppo-separated-v1000"


def test_iterative_nonplaying_rl_resumes_and_chains_checkpoints(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    playing_onnx = tmp_path / "playing.onnx"
    playing_metadata = tmp_path / "playing.json"
    playing_onnx.write_bytes(b"playing-onnx")
    playing_metadata.write_text("{}\n", encoding="utf-8")
    build_count = 0
    parent_checkpoints: list[tuple[str, str | None]] = []

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def fake_build() -> None:
        nonlocal build_count
        build_count += 1

    def fake_bootstrap(**kwargs: object) -> dict[str, object]:
        onnx_path = Path(cast(str | Path, kwargs["onnx_path"]))
        metadata_path = Path(cast(str | Path, kwargs["metadata_path"]))
        policy_type = str(kwargs["policy_type"])
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        onnx_path.write_bytes(f"{policy_type}-bootstrap-onnx".encode())
        metadata_path.write_text(
            json.dumps({"policyType": policy_type, "bootstrap": True}) + "\n",
            encoding="utf-8",
        )
        return {"policyType": policy_type}

    def fake_rollout(*args: object, **kwargs: object) -> dict[str, object]:
        phase = str(kwargs["phase"])
        dataset_dir = Path(cast(str | Path, kwargs["dataset_dir"]))
        policy_onnx = Path(cast(str | Path, kwargs["policy_onnx"]))
        policy_metadata = Path(cast(str | Path, kwargs["policy_metadata"]))
        start_seed = cast(int, kwargs["start_seed"])
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "samples.jsonl").write_text("{}\n", encoding="utf-8")
        manifest = {
            "startSeed": start_seed,
            "gameCount": 1,
            "gameCountUnit": "logical-seeds",
            "logicalSeedCount": 1,
            "actualGameCount": 5,
            "rolloutPolicyTopology": "candidate-x1-frozen-x4-v1",
            "rotationOffsets": [0, 1, 2, 3, 4],
            "reward": {
                "type": NONPLAYING_REWARD_TYPE,
                "version": NONPLAYING_REWARD_VERSION,
                "id": NONPLAYING_REWARD_ID,
            },
            "terminalRewardTransform": _terminal_reward_transform(),
            "allPassRule": {
                "id": NONPLAYING_ALL_PASS_RULE_ID,
                "starterPayoff": 0,
                "otherPayoff": 0,
            },
            "sampleCount": 1,
            "behaviorPolicy": {
                "artifactId": kwargs["artifact_id"],
                "onnxSha256": sha256(policy_onnx),
                "metadataSha256": sha256(policy_metadata),
            },
            "fixedPlayingPolicy": {"artifactId": "ppo-separated-v1000"},
        }
        if phase == "bidding":
            manifest["nonLearningAgents"] = {
                "bidding": {
                    "type": "mixed-frozen-bidding",
                    "mixingRuleVersion": "per-seat-seeded-rule-based-conservative-50-50-v1",
                    "selectionUnit": "game-seat",
                    "ruleBasedWeight": 0.5,
                    "conservativeWeight": 0.5,
                    "policies": {
                        "ruleBased": {
                            "type": "rule-based-bidding",
                            "id": "rule-based-bidding-v1",
                        },
                        "conservative": {
                            "type": "conservative-bidding",
                            "id": "conservative-bidding-v1",
                        },
                    },
                }
            }
            manifest["diagnostics"] = {
                "frozenBiddingOpponentMix": {
                    "mixingRuleVersion": "per-seat-seeded-rule-based-conservative-50-50-v1",
                    "ruleBasedSeatCount": 10,
                    "conservativeSeatCount": 10,
                    "seatAssignments": [
                        {
                            "policy": {
                                "type": "rule-based-bidding",
                                "id": "rule-based-bidding-v1",
                            }
                        }
                        for _ in range(10)
                    ]
                    + [
                        {
                            "policy": {
                                "type": "conservative-bidding",
                                "id": "conservative-bidding-v1",
                            }
                        }
                        for _ in range(10)
                    ],
                }
            }
        (dataset_dir / "manifest.json").write_text(
            json.dumps(manifest) + "\n",
            encoding="utf-8",
        )
        return {
            "phase": phase,
            "gameCount": 1,
            "gameCountUnit": "logical-seeds",
            "logicalSeedCount": 1,
            "actualGameCount": 5,
            "sampleCount": 1,
            "shardCount": 1,
            "diagnostics": {
                "candidateSeatCount": 1,
                "frozenSeatCount": 4,
                "candidateRotationSeatCount": 5,
                "actualGameCount": 5,
                "logicalSeedCount": 1,
                "rotationOffsets": [0, 1, 2, 3, 4],
            },
        }

    def fake_train(
        config: NonPlayingRlRunConfig,
        phase: str,
        dataset_dir: Path,
        checkpoint_path: Path,
        training_seed: int,
        parent_checkpoint: Path | None = None,
    ) -> dict[str, object]:
        parent_checkpoints.append((phase, str(parent_checkpoint) if parent_checkpoint else None))
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(
            f"{phase}-checkpoint-{training_seed}-{parent_checkpoint}".encode()
        )
        return {
            "sampleCount": 1,
            "optimizerStepCount": 1,
            "meanReward": 0.25,
            "meanTotalLoss": 0.5,
            "meanValueLoss": 0.125,
            "clippedFraction": 0.0,
            "outputCheckpointPath": str(checkpoint_path),
        }

    def fake_export(
        phase: str,
        dataset_dir: Path,
        checkpoint_path: Path,
        onnx_path: Path,
        metadata_path: Path,
    ) -> dict[str, object]:
        onnx_path.write_bytes(f"{phase}-onnx-{checkpoint_path.read_bytes().decode()}".encode())
        metadata_path.write_text(
            json.dumps({"policyType": phase, "checkpoint": str(checkpoint_path)}) + "\n",
            encoding="utf-8",
        )
        return {
            "policyType": phase,
            "onnxPath": str(onnx_path),
            "metadataPath": str(metadata_path),
        }

    def fake_evaluation(*args: object, **kwargs: object) -> dict[str, object]:
        result = {
            "scheduledGames": 1,
            "completedGames": 1,
            "failedGames": 0,
            "fallbackCount": 0,
            "illegalActionCount": 0,
            "policyAgentDecisionCounts": {
                "biddingOnnxCallCount": 1,
                "adjutantOnnxCallCount": 1,
                "exchangeOnnxCallCount": 1,
            },
        }
        Path(cast(str | Path, args[2])).write_text(
            json.dumps(result) + "\n",
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(
        "napoleon_ml.nonplaying_rl_orchestrator._build_typescript_helpers",
        fake_build,
    )
    monkeypatch.setattr(
        "napoleon_ml.nonplaying_rl_orchestrator.export_seeded_nonplaying_bootstrap_policy_to_onnx",
        fake_bootstrap,
    )
    monkeypatch.setattr(
        "napoleon_ml.nonplaying_rl_orchestrator._run_nonplaying_rollout",
        fake_rollout,
    )
    monkeypatch.setattr("napoleon_ml.nonplaying_rl_orchestrator._train_phase", fake_train)
    monkeypatch.setattr(
        "napoleon_ml.nonplaying_rl_orchestrator._export_trained_phase",
        fake_export,
    )
    monkeypatch.setattr(
        "napoleon_ml.nonplaying_rl_orchestrator._run_full_policy_evaluation",
        fake_evaluation,
    )

    output_dir = tmp_path / "iterative"
    config = NonPlayingIterativeRlRunConfig(
        output_dir=output_dir,
        iterations=3,
        games_per_iteration=1,
        evaluation_interval=1,
        evaluation_games=1,
        games_per_shard=1,
        epochs=1,
        batch_size=4,
        hidden_dim=8,
        hidden_layers=1,
        learning_rate=1e-4,
        bidding_entropy_coefficient=0.01,
        playing_policy_onnx=playing_onnx,
        playing_policy_metadata=playing_metadata,
    )

    first_summary = run_iterative_nonplaying_rl_pipeline(
        config,
        stop_after_iterations=2,
    )
    assert first_summary["completedIterationCount"] == 2
    (output_dir / "iterations" / "iter-000002").mkdir(parents=True)
    (output_dir / "iterations" / "iter-000002" / "partial.txt").write_text(
        "partial\n",
        encoding="utf-8",
    )

    resumed_summary = run_iterative_nonplaying_rl_pipeline(config, resume=True)

    assert resumed_summary["completedIterationCount"] == 3
    stored_config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert stored_config["trainingDevice"] == "cpu"
    assert stored_config["biddingEntropyCoefficient"] == 0.01
    assert stored_config["biddingMinibatchStrategy"] == "random"
    assert stored_config["biddingAdvantageNormalization"] == (
        advantage_normalization_metadata(DEFAULT_BIDDING_ADVANTAGE_NORMALIZATION)
    )
    assert build_count == 2
    assert list((output_dir / "iterations").glob("iter-000002.incomplete-*"))

    iter0 = json.loads(
        (output_dir / "iterations" / "iter-000000" / "iteration.json").read_text(
            encoding="utf-8"
        )
    )
    iter1 = json.loads(
        (output_dir / "iterations" / "iter-000001" / "iteration.json").read_text(
            encoding="utf-8"
        )
    )
    iter2 = json.loads(
        (output_dir / "iterations" / "iter-000002" / "iteration.json").read_text(
            encoding="utf-8"
        )
    )
    for phase in ("bidding", "adjutant", "exchange"):
        assert iter1["phases"][phase]["inputCheckpointSha256"] == (
            iter0["phases"][phase]["artifact"]["checkpointSha256"]
        )
        assert iter2["phases"][phase]["inputCheckpointSha256"] == (
            iter1["phases"][phase]["artifact"]["checkpointSha256"]
        )
        assert iter1["phases"][phase]["behaviorArtifact"]["onnxSha256"] == (
            iter0["phases"][phase]["artifact"]["onnxSha256"]
        )
        assert iter2["phases"][phase]["behaviorArtifact"]["onnxSha256"] == (
            iter1["phases"][phase]["artifact"]["onnxSha256"]
        )

    assert parent_checkpoints[0:3] == [
        ("bidding", None),
        ("adjutant", None),
        ("exchange", None),
    ]
    assert all(parent is not None for _phase, parent in parent_checkpoints[3:])


def test_iterative_bidding_only_freezes_downstream_artifacts_and_resumes(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    playing_onnx = tmp_path / "playing.onnx"
    playing_metadata = tmp_path / "playing.json"
    playing_onnx.write_bytes(b"playing-onnx")
    playing_metadata.write_text("{}\n", encoding="utf-8")
    stages: list[str] = []
    evaluation_artifacts: list[dict[str, dict[str, str]]] = []

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr(
        "napoleon_ml.nonplaying_rl_orchestrator._build_typescript_helpers",
        lambda: stages.append("typescript-build"),
    )

    def fake_bootstrap(**kwargs: object) -> dict[str, object]:
        phase = str(kwargs["policy_type"])
        onnx_path = Path(cast(str | Path, kwargs["onnx_path"]))
        metadata_path = Path(cast(str | Path, kwargs["metadata_path"]))
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        onnx_path.write_bytes(f"{phase}-bootstrap-onnx".encode())
        metadata_path.write_text(json.dumps({"policyType": phase}) + "\n", encoding="utf-8")
        stages.append(f"{phase}-bootstrap")
        return {"policyType": phase}

    def fake_rollout(*args: object, **kwargs: object) -> dict[str, object]:
        phase = str(kwargs["phase"])
        fixed_artifacts = cast(dict[str, dict[str, str]] | None, kwargs.get("fixed_artifacts"))
        assert phase == "bidding"
        assert fixed_artifacts is not None
        assert set(fixed_artifacts) == {"adjutant", "exchange"}
        stages.append(f"{phase}-rollout")
        dataset_dir = Path(cast(str | Path, kwargs["dataset_dir"]))
        policy_onnx = Path(cast(str | Path, kwargs["policy_onnx"]))
        policy_metadata = Path(cast(str | Path, kwargs["policy_metadata"]))
        start_seed = cast(int, kwargs["start_seed"])
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "samples.jsonl").write_text("{}\n", encoding="utf-8")
        (dataset_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "startSeed": start_seed,
                    "gameCount": 1,
                    "gameCountUnit": "logical-seeds",
                    "logicalSeedCount": 1,
                    "actualGameCount": 5,
                    "rolloutPolicyTopology": "candidate-x1-frozen-x4-v1",
                    "rotationOffsets": [0, 1, 2, 3, 4],
                    "reward": {
                        "type": NONPLAYING_REWARD_TYPE,
                        "version": NONPLAYING_REWARD_VERSION,
                        "id": NONPLAYING_REWARD_ID,
                    },
                    "terminalRewardTransform": _terminal_reward_transform(),
                    "allPassRule": {
                        "id": NONPLAYING_ALL_PASS_RULE_ID,
                        "starterPayoff": 0,
                        "otherPayoff": 0,
                    },
                    "behaviorPolicy": {
                        "onnxSha256": sha256(policy_onnx),
                        "metadataSha256": sha256(policy_metadata),
                    },
                    "fixedPlayingPolicy": {"artifactId": "ppo-separated-v1000"},
                    "nonLearningAgents": {
                        "bidding": {
                            "type": "mixed-frozen-bidding",
                            "mixingRuleVersion": (
                                "per-seat-seeded-rule-based-conservative-50-50-v1"
                            ),
                            "selectionUnit": "game-seat",
                            "ruleBasedWeight": 0.5,
                            "conservativeWeight": 0.5,
                            "policies": {
                                "ruleBased": {
                                    "type": "rule-based-bidding",
                                    "id": "rule-based-bidding-v1",
                                },
                                "conservative": {
                                    "type": "conservative-bidding",
                                    "id": "conservative-bidding-v1",
                                },
                            },
                        }
                    },
                    "diagnostics": {
                        "frozenBiddingOpponentMix": {
                            "mixingRuleVersion": (
                                "per-seat-seeded-rule-based-conservative-50-50-v1"
                            ),
                            "ruleBasedSeatCount": 10,
                            "conservativeSeatCount": 10,
                            "seatAssignments": [
                                {
                                    "policy": {
                                        "type": "rule-based-bidding",
                                        "id": "rule-based-bidding-v1",
                                    }
                                }
                                for _ in range(10)
                            ]
                            + [
                                {
                                    "policy": {
                                        "type": "conservative-bidding",
                                        "id": "conservative-bidding-v1",
                                    }
                                }
                                for _ in range(10)
                            ],
                        }
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {"phase": phase, "sampleCount": 1, "gameCount": 1}

    def fake_train(
        config: NonPlayingRlRunConfig,
        phase: str,
        dataset_dir: Path,
        checkpoint_path: Path,
        training_seed: int,
        parent_checkpoint: Path | None = None,
    ) -> dict[str, object]:
        assert phase == "bidding"
        stages.append(f"{phase}-train")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(f"{phase}-checkpoint-{training_seed}".encode())
        return {
            "sampleCount": 1,
            "meanReward": 0.1,
            "meanTotalLoss": 0.2,
            "meanValueLoss": 0.3,
            "clippedFraction": 0.0,
        }

    def fake_export(
        phase: str,
        dataset_dir: Path,
        checkpoint_path: Path,
        onnx_path: Path,
        metadata_path: Path,
    ) -> dict[str, object]:
        assert phase == "bidding"
        stages.append(f"{phase}-export")
        onnx_path.write_bytes(f"{phase}-onnx-{checkpoint_path.read_text()}".encode())
        metadata_path.write_text(json.dumps({"policyType": phase}) + "\n", encoding="utf-8")
        return {"policyType": phase}

    def fake_evaluation(
        config: NonPlayingRlRunConfig,
        artifacts: dict[str, dict[str, str]],
        evaluation_path: Path,
        start_seed: int | None = None,
    ) -> dict[str, object]:
        evaluation_artifacts.append(artifacts)
        result = {
            "scheduledGames": 1,
            "completedGames": 1,
            "failedGames": 0,
            "fallbackCount": 0,
            "illegalActionCount": 0,
            "policyArtifacts": artifacts,
            "policyAgentDecisionCounts": {
                "biddingOnnxCallCount": 1,
                "adjutantOnnxCallCount": 1,
                "exchangeOnnxCallCount": 1,
            },
        }
        evaluation_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
        return result

    monkeypatch.setattr(
        "napoleon_ml.nonplaying_rl_orchestrator.export_seeded_nonplaying_bootstrap_policy_to_onnx",
        fake_bootstrap,
    )
    monkeypatch.setattr(
        "napoleon_ml.nonplaying_rl_orchestrator._run_nonplaying_rollout",
        fake_rollout,
    )
    monkeypatch.setattr("napoleon_ml.nonplaying_rl_orchestrator._train_phase", fake_train)
    monkeypatch.setattr(
        "napoleon_ml.nonplaying_rl_orchestrator._export_trained_phase",
        fake_export,
    )
    monkeypatch.setattr(
        "napoleon_ml.nonplaying_rl_orchestrator._run_full_policy_evaluation",
        fake_evaluation,
    )

    output_dir = tmp_path / "bidding-only"
    config = NonPlayingIterativeRlRunConfig(
        output_dir=output_dir,
        train_mode="bidding-only",
        iterations=2,
        games_per_iteration=1,
        evaluation_interval=1,
        evaluation_games=1,
        games_per_shard=1,
        hidden_dim=8,
        hidden_layers=1,
        playing_policy_onnx=playing_onnx,
        playing_policy_metadata=playing_metadata,
    )

    first_summary = run_iterative_nonplaying_rl_pipeline(config, stop_after_iterations=1)
    resumed_summary = run_iterative_nonplaying_rl_pipeline(config, resume=True)

    assert first_summary["completedIterationCount"] == 1
    assert resumed_summary["completedIterationCount"] == 2
    stored_config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert stored_config["trainMode"] == "bidding-only"
    assert stored_config["trainPhases"] == ["bidding"]
    assert stored_config["frozenPhases"] == ["adjutant", "exchange"]
    assert set(stored_config["frozenArtifacts"]) == {"adjutant", "exchange"}

    iter0 = json.loads(
        (output_dir / "iterations" / "iter-000000" / "iteration.json").read_text(
            encoding="utf-8"
        )
    )
    iter1 = json.loads(
        (output_dir / "iterations" / "iter-000001" / "iteration.json").read_text(
            encoding="utf-8"
        )
    )
    assert iter0["trainPhases"] == ["bidding"]
    assert iter0["phases"]["adjutant"]["mode"] == "frozen"
    assert iter0["phases"]["exchange"]["mode"] == "frozen"
    assert not (output_dir / "iterations" / "iter-000000" / "adjutant").exists()
    assert not (output_dir / "iterations" / "iter-000000" / "exchange").exists()
    assert (
        iter0["phases"]["bidding"]["artifact"]["checkpointSha256"]
        != iter1["phases"]["bidding"]["artifact"]["checkpointSha256"]
    )
    for phase in ("adjutant", "exchange"):
        assert iter0["phases"][phase]["artifact"]["onnxSha256"] == (
            iter1["phases"][phase]["artifact"]["onnxSha256"]
        )
        assert iter0["phases"][phase]["artifact"]["onnxSha256"] == (
            stored_config["frozenArtifacts"][phase]["onnxSha256"]
        )
    assert evaluation_artifacts[-1]["bidding"]["artifactId"] == "bidding-iter-000002"
    assert evaluation_artifacts[-1]["adjutant"]["artifactId"] == "adjutant-bootstrap-seed-202"
    assert evaluation_artifacts[-1]["exchange"]["artifactId"] == "exchange-bootstrap-seed-202"
    assert "adjutant-train" not in stages
    assert "exchange-train" not in stages


def test_iterative_resume_rejects_train_mode_mismatch(tmp_path: Path) -> None:
    playing_onnx = tmp_path / "playing.onnx"
    playing_metadata = tmp_path / "playing.json"
    playing_onnx.write_bytes(b"playing-onnx")
    playing_metadata.write_text("{}\n", encoding="utf-8")
    stored_config = NonPlayingIterativeRlRunConfig(
        output_dir=tmp_path / "run",
        train_mode="bidding-only",
        playing_policy_onnx=playing_onnx,
        playing_policy_metadata=playing_metadata,
    ).file_dict()
    requested_config = NonPlayingIterativeRlRunConfig(
        output_dir=tmp_path / "run",
        train_mode="multi-phase",
        playing_policy_onnx=playing_onnx,
        playing_policy_metadata=playing_metadata,
    ).file_dict()

    with pytest.raises(NonPlayingRlOrchestratorError, match="train"):
        _validate_iterative_resume_config(
            stored_config,
            requested_config,
            provided_config_keys={"trainMode"},
        )


def test_iterative_bidding_only_rejects_cpp_backend_until_frozen_downstream_supported(
    tmp_path: Path,
) -> None:
    playing_onnx = tmp_path / "playing.onnx"
    playing_metadata = tmp_path / "playing.json"
    playing_onnx.write_bytes(b"playing-onnx")
    playing_metadata.write_text("{}\n", encoding="utf-8")
    config = NonPlayingIterativeRlRunConfig(
        output_dir=tmp_path / "run",
        train_mode="bidding-only",
        simulation_backend="cpp",
        playing_policy_onnx=playing_onnx,
        playing_policy_metadata=playing_metadata,
    )

    with pytest.raises(NonPlayingRlOrchestratorError, match="frozen adjutant"):
        run_iterative_nonplaying_rl_pipeline(config)


def test_iterative_resume_rejects_legacy_self_play_schema(tmp_path: Path) -> None:
    playing_onnx = tmp_path / "playing.onnx"
    playing_metadata = tmp_path / "playing.json"
    playing_onnx.write_bytes(b"playing-onnx")
    playing_metadata.write_text("{}\n", encoding="utf-8")
    requested_config = NonPlayingIterativeRlRunConfig(
        output_dir=tmp_path / "run",
        iterations=3,
        games_per_iteration=1,
        playing_policy_onnx=playing_onnx,
        playing_policy_metadata=playing_metadata,
    ).file_dict()
    legacy_config = dict(requested_config)
    legacy_config["schemaVersion"] = 1
    legacy_config.pop("rolloutPolicyTopology")
    legacy_config.pop("gamesPerIterationUnit")
    legacy_config.pop("actualGamesPerIteration")
    legacy_config.pop("rotationOffsets")

    with pytest.raises(NonPlayingRlOrchestratorError, match="schemaVersion mismatch"):
        _validate_iterative_resume_config(
            legacy_config,
            requested_config,
            provided_config_keys=set(),
        )


def test_iterative_resume_rejects_pre_mix_schema(tmp_path: Path) -> None:
    playing_onnx = tmp_path / "playing.onnx"
    playing_metadata = tmp_path / "playing.json"
    playing_onnx.write_bytes(b"playing-onnx")
    playing_metadata.write_text("{}\n", encoding="utf-8")
    requested_config = NonPlayingIterativeRlRunConfig(
        output_dir=tmp_path / "run",
        iterations=3,
        games_per_iteration=1,
        playing_policy_onnx=playing_onnx,
        playing_policy_metadata=playing_metadata,
    ).file_dict()
    legacy_config = dict(requested_config)
    legacy_config["schemaVersion"] = 3
    legacy_config.pop("biddingFrozenOpponentMixRuleVersion")
    legacy_config.pop("biddingFrozenOpponentPolicyIds")

    with pytest.raises(NonPlayingRlOrchestratorError, match="schemaVersion mismatch"):
        _validate_iterative_resume_config(
            legacy_config,
            requested_config,
            provided_config_keys=set(),
        )


@pytest.mark.parametrize(
    ("legacy_version", "legacy_id"),
    [
        (1, "non-playing-terminal-role-reward-v1"),
        (2, "non-playing-terminal-role-reward-v2"),
    ],
)
def test_iterative_resume_rejects_older_reward_config(
    tmp_path: Path,
    legacy_version: int,
    legacy_id: str,
) -> None:
    playing_onnx = tmp_path / "playing.onnx"
    playing_metadata = tmp_path / "playing.json"
    playing_onnx.write_bytes(b"playing-onnx")
    playing_metadata.write_text("{}\n", encoding="utf-8")
    requested_config = NonPlayingIterativeRlRunConfig(
        output_dir=tmp_path / "run",
        iterations=3,
        games_per_iteration=1,
        playing_policy_onnx=playing_onnx,
        playing_policy_metadata=playing_metadata,
    ).file_dict()
    legacy_config = dict(requested_config)
    legacy_config["reward"] = {
        "type": NONPLAYING_REWARD_TYPE,
        "version": legacy_version,
        "id": legacy_id,
    }

    with pytest.raises(NonPlayingRlOrchestratorError, match="reward"):
        _validate_iterative_resume_config(
            legacy_config,
            requested_config,
            provided_config_keys=set(),
        )


def test_iterative_resume_rejects_missing_terminal_reward_transform(
    tmp_path: Path,
) -> None:
    playing_onnx = tmp_path / "playing.onnx"
    playing_metadata = tmp_path / "playing.json"
    playing_onnx.write_bytes(b"playing-onnx")
    playing_metadata.write_text("{}\n", encoding="utf-8")
    requested_config = NonPlayingIterativeRlRunConfig(
        output_dir=tmp_path / "run",
        iterations=3,
        games_per_iteration=1,
        playing_policy_onnx=playing_onnx,
        playing_policy_metadata=playing_metadata,
    ).file_dict()
    legacy_config = dict(requested_config)
    legacy_config.pop("terminalRewardTransform")

    with pytest.raises(NonPlayingRlOrchestratorError, match="terminalRewardTransform"):
        _validate_iterative_resume_config(
            legacy_config,
            requested_config,
            provided_config_keys=set(),
        )


def test_iterative_resume_allows_omitted_games_per_iteration(tmp_path: Path) -> None:
    playing_onnx = tmp_path / "playing.onnx"
    playing_metadata = tmp_path / "playing.json"
    playing_onnx.write_bytes(b"playing-onnx")
    playing_metadata.write_text("{}\n", encoding="utf-8")
    stored_config = NonPlayingIterativeRlRunConfig(
        output_dir=tmp_path / "run",
        games_per_iteration=200,
        playing_policy_onnx=playing_onnx,
        playing_policy_metadata=playing_metadata,
    ).file_dict()
    requested_config = NonPlayingIterativeRlRunConfig(
        output_dir=tmp_path / "run",
        playing_policy_onnx=playing_onnx,
        playing_policy_metadata=playing_metadata,
    ).file_dict()

    _validate_iterative_resume_config(
        stored_config,
        requested_config,
        provided_config_keys=set(),
    )


def test_iterative_resume_rejects_explicit_games_per_iteration_mismatch(
    tmp_path: Path,
) -> None:
    playing_onnx = tmp_path / "playing.onnx"
    playing_metadata = tmp_path / "playing.json"
    playing_onnx.write_bytes(b"playing-onnx")
    playing_metadata.write_text("{}\n", encoding="utf-8")
    stored_config = NonPlayingIterativeRlRunConfig(
        output_dir=tmp_path / "run",
        games_per_iteration=200,
        playing_policy_onnx=playing_onnx,
        playing_policy_metadata=playing_metadata,
    ).file_dict()
    requested_config = NonPlayingIterativeRlRunConfig(
        output_dir=tmp_path / "run",
        games_per_iteration=201,
        playing_policy_onnx=playing_onnx,
        playing_policy_metadata=playing_metadata,
    ).file_dict()

    with pytest.raises(NonPlayingRlOrchestratorError, match="gamesPerIteration"):
        _validate_iterative_resume_config(
            stored_config,
            requested_config,
            provided_config_keys={"gamesPerIteration"},
        )


def test_iterative_resume_rejects_explicit_bidding_entropy_mismatch(
    tmp_path: Path,
) -> None:
    playing_onnx = tmp_path / "playing.onnx"
    playing_metadata = tmp_path / "playing.json"
    playing_onnx.write_bytes(b"playing-onnx")
    playing_metadata.write_text("{}\n", encoding="utf-8")
    stored_config = NonPlayingIterativeRlRunConfig(
        output_dir=tmp_path / "run",
        bidding_entropy_coefficient=0.01,
        playing_policy_onnx=playing_onnx,
        playing_policy_metadata=playing_metadata,
    ).file_dict()
    requested_config = NonPlayingIterativeRlRunConfig(
        output_dir=tmp_path / "run",
        bidding_entropy_coefficient=0.02,
        playing_policy_onnx=playing_onnx,
        playing_policy_metadata=playing_metadata,
    ).file_dict()

    with pytest.raises(NonPlayingRlOrchestratorError, match="biddingEntropyCoefficient"):
        _validate_iterative_resume_config(
            stored_config,
            requested_config,
            provided_config_keys={"biddingEntropyCoefficient"},
        )


def test_iterative_resume_rejects_explicit_bidding_hidden_dims_mismatch(
    tmp_path: Path,
) -> None:
    playing_onnx = tmp_path / "playing.onnx"
    playing_metadata = tmp_path / "playing.json"
    playing_onnx.write_bytes(b"playing-onnx")
    playing_metadata.write_text("{}\n", encoding="utf-8")
    stored_config = NonPlayingIterativeRlRunConfig(
        output_dir=tmp_path / "run",
        bidding_hidden_dims=(512, 256, 256),
        playing_policy_onnx=playing_onnx,
        playing_policy_metadata=playing_metadata,
    ).file_dict()
    requested_config = NonPlayingIterativeRlRunConfig(
        output_dir=tmp_path / "run",
        bidding_hidden_dims=(128, 128),
        playing_policy_onnx=playing_onnx,
        playing_policy_metadata=playing_metadata,
    ).file_dict()

    with pytest.raises(NonPlayingRlOrchestratorError, match="biddingHiddenDims"):
        _validate_iterative_resume_config(
            stored_config,
            requested_config,
            provided_config_keys={"biddingHiddenDims"},
        )


def test_iterative_resume_loads_stored_bidding_hidden_dims_when_omitted(
    tmp_path: Path,
) -> None:
    playing_onnx = tmp_path / "playing.onnx"
    playing_metadata = tmp_path / "playing.json"
    playing_onnx.write_bytes(b"playing-onnx")
    playing_metadata.write_text("{}\n", encoding="utf-8")
    stored_config = NonPlayingIterativeRlRunConfig(
        output_dir=tmp_path / "run",
        bidding_hidden_dims=(512, 256, 256),
        playing_policy_onnx=playing_onnx,
        playing_policy_metadata=playing_metadata,
    ).file_dict()
    requested_config = NonPlayingIterativeRlRunConfig(
        output_dir=tmp_path / "run",
        playing_policy_onnx=playing_onnx,
        playing_policy_metadata=playing_metadata,
    ).file_dict()

    _validate_iterative_resume_config(
        stored_config,
        requested_config,
        provided_config_keys=set(),
    )


def test_iterative_resume_defaults_missing_simulation_backend_to_typescript(
    tmp_path: Path,
) -> None:
    playing_onnx = tmp_path / "playing.onnx"
    playing_metadata = tmp_path / "playing.json"
    playing_onnx.write_bytes(b"playing-onnx")
    playing_metadata.write_text("{}\n", encoding="utf-8")
    stored_config = NonPlayingIterativeRlRunConfig(
        output_dir=tmp_path / "run",
        playing_policy_onnx=playing_onnx,
        playing_policy_metadata=playing_metadata,
    ).file_dict()
    stored_config.pop("simulationBackend")
    requested_config = NonPlayingIterativeRlRunConfig(
        output_dir=tmp_path / "run",
        simulation_backend="typescript",
        playing_policy_onnx=playing_onnx,
        playing_policy_metadata=playing_metadata,
    ).file_dict()

    _validate_iterative_resume_config(
        stored_config,
        requested_config,
        provided_config_keys=set(),
    )


def test_iterative_resume_allows_training_device_mismatch(tmp_path: Path) -> None:
    playing_onnx = tmp_path / "playing.onnx"
    playing_metadata = tmp_path / "playing.json"
    playing_onnx.write_bytes(b"playing-onnx")
    playing_metadata.write_text("{}\n", encoding="utf-8")
    stored_config = NonPlayingIterativeRlRunConfig(
        output_dir=tmp_path / "run",
        training_device="cpu",
        playing_policy_onnx=playing_onnx,
        playing_policy_metadata=playing_metadata,
    ).file_dict()
    requested_config = NonPlayingIterativeRlRunConfig(
        output_dir=tmp_path / "run",
        training_device="auto",
        playing_policy_onnx=playing_onnx,
        playing_policy_metadata=playing_metadata,
    ).file_dict()

    _validate_iterative_resume_config(
        stored_config,
        requested_config,
        provided_config_keys={"trainingDevice"},
    )


def test_iterative_resume_rejects_bidding_advantage_normalization_mismatch(
    tmp_path: Path,
) -> None:
    playing_onnx = tmp_path / "playing.onnx"
    playing_metadata = tmp_path / "playing.json"
    playing_onnx.write_bytes(b"playing-onnx")
    playing_metadata.write_text("{}\n", encoding="utf-8")
    stored_config = NonPlayingIterativeRlRunConfig(
        output_dir=tmp_path / "run",
        bidding_advantage_normalization="dataset",
        playing_policy_onnx=playing_onnx,
        playing_policy_metadata=playing_metadata,
    ).file_dict()
    requested_config = NonPlayingIterativeRlRunConfig(
        output_dir=tmp_path / "run",
        bidding_advantage_normalization="none",
        playing_policy_onnx=playing_onnx,
        playing_policy_metadata=playing_metadata,
    ).file_dict()

    with pytest.raises(NonPlayingRlOrchestratorError, match="biddingAdvantageNormalization"):
        _validate_iterative_resume_config(
            stored_config,
            requested_config,
            provided_config_keys=set(),
        )


def test_iterative_resume_defaults_missing_bidding_minibatch_strategy_to_random(
    tmp_path: Path,
) -> None:
    playing_onnx = tmp_path / "playing.onnx"
    playing_metadata = tmp_path / "playing.json"
    playing_onnx.write_bytes(b"playing-onnx")
    playing_metadata.write_text("{}\n", encoding="utf-8")
    stored_config = NonPlayingIterativeRlRunConfig(
        output_dir=tmp_path / "run",
        bidding_minibatch_strategy="random",
        playing_policy_onnx=playing_onnx,
        playing_policy_metadata=playing_metadata,
    ).file_dict()
    stored_config.pop("biddingMinibatchStrategy")
    requested_config = NonPlayingIterativeRlRunConfig(
        output_dir=tmp_path / "run",
        bidding_minibatch_strategy="random",
        playing_policy_onnx=playing_onnx,
        playing_policy_metadata=playing_metadata,
    ).file_dict()

    _validate_iterative_resume_config(
        stored_config,
        requested_config,
        provided_config_keys=set(),
    )


def test_iterative_resume_rejects_bidding_minibatch_strategy_mismatch(
    tmp_path: Path,
) -> None:
    playing_onnx = tmp_path / "playing.onnx"
    playing_metadata = tmp_path / "playing.json"
    playing_onnx.write_bytes(b"playing-onnx")
    playing_metadata.write_text("{}\n", encoding="utf-8")
    stored_config = NonPlayingIterativeRlRunConfig(
        output_dir=tmp_path / "run",
        bidding_minibatch_strategy="random",
        playing_policy_onnx=playing_onnx,
        playing_policy_metadata=playing_metadata,
    ).file_dict()
    requested_config = NonPlayingIterativeRlRunConfig(
        output_dir=tmp_path / "run",
        bidding_minibatch_strategy="strongest-suit-balanced",
        playing_policy_onnx=playing_onnx,
        playing_policy_metadata=playing_metadata,
    ).file_dict()

    with pytest.raises(NonPlayingRlOrchestratorError, match="biddingMinibatchStrategy"):
        _validate_iterative_resume_config(
            stored_config,
            requested_config,
            provided_config_keys=set(),
        )


def test_iterative_cli_honors_explicit_one_shot_default_values() -> None:
    argv = [
        "--output-dir",
        "/tmp/non-playing",
        "--iterations",
        "10",
        "--evaluation-games",
        "5",
        "--epochs",
        "1",
        "--batch-size",
        "32",
        "--learning-rate",
        "1e-3",
        "--bidding-entropy-coefficient",
        "0.01",
        "--bidding-hidden-dims",
        "512,256,256",
        "--bidding-advantage-normalization",
        "dataset",
        "--bidding-minibatch-strategy",
        "strongest-suit-balanced",
        "--training-device",
        "auto",
    ]
    args = build_parser().parse_args(argv)
    config = _iterative_config_from_args(
        args,
        provided_config_keys=_provided_iterative_config_keys(argv),
    )

    assert config.evaluation_games == 5
    assert config.epochs == 1
    assert config.batch_size == 32
    assert config.learning_rate == 1e-3
    assert config.bidding_entropy_coefficient == 0.01
    assert config.resolved_bidding_hidden_dims == (512, 256, 256)
    assert config.bidding_advantage_normalization == "dataset"
    assert config.bidding_minibatch_strategy == "strongest-suit-balanced"
    assert config.training_device == "auto"
    assert "biddingHiddenDims" in _provided_iterative_config_keys(argv)
    assert "biddingMinibatchStrategy" in _provided_iterative_config_keys(argv)


def test_iterative_cli_accepts_cpp_simulation_backend() -> None:
    argv = [
        "--output-dir",
        "/tmp/non-playing",
        "--iterations",
        "10",
        "--simulation-backend",
        "cpp",
    ]
    args = build_parser().parse_args(argv)
    config = _iterative_config_from_args(
        args,
        provided_config_keys=_provided_iterative_config_keys(argv),
    )

    assert config.simulation_backend == "cpp"
    assert config.file_dict()["simulationBackend"] == "cpp"
    assert config.as_one_shot_config().simulation_backend == "cpp"
    assert "simulationBackend" in _provided_iterative_config_keys(argv)


def test_iterative_cli_uses_iterative_defaults_when_values_are_omitted() -> None:
    argv = [
        "--output-dir",
        "/tmp/non-playing",
        "--iterations",
        "10",
    ]
    args = build_parser().parse_args(argv)
    config = _iterative_config_from_args(
        args,
        provided_config_keys=_provided_iterative_config_keys(argv),
    )

    assert config.evaluation_games == DEFAULT_ITERATIVE_EVALUATION_GAMES
    assert config.batch_size == DEFAULT_ITERATIVE_BATCH_SIZE
    assert config.bidding_entropy_coefficient == DEFAULT_BIDDING_ENTROPY_COEFFICIENT
    assert config.bidding_advantage_normalization == DEFAULT_BIDDING_ADVANTAGE_NORMALIZATION
    assert config.bidding_minibatch_strategy == DEFAULT_BIDDING_MINIBATCH_STRATEGY


def _terminal_reward_transform() -> dict[str, object]:
    return {
        "type": NONPLAYING_TERMINAL_REWARD_TRANSFORM_TYPE,
        "version": NONPLAYING_TERMINAL_REWARD_TRANSFORM_VERSION,
        "id": NONPLAYING_TERMINAL_REWARD_TRANSFORM_ID,
        "sourceRewardId": NONPLAYING_REWARD_ID,
        "baseline": "meanRawRewardAllPlayers",
        "formula": "relative_reward_i = raw_reward_i - mean(raw_reward_all_players)",
    }
