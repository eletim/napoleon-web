from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from napoleon_ml.nonplaying_rl_orchestrator import (
    NonPlayingRlRunConfig,
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
            playing_policy_onnx=playing_onnx,
            playing_policy_metadata=playing_metadata,
        )
    )

    assert (output_dir / "run-summary.json").is_file()
    written = json.loads((output_dir / "run-summary.json").read_text(encoding="utf-8"))
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
