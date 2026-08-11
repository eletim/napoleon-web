"""C++ RL rollout binary dataset -> current Python trainer reader."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch

from napoleon_ml.dataset import load_manifest
from napoleon_ml.dataset.constants import EXPECTED_CARD_IDS, MODEL_INPUT_FEATURE_COUNT
from napoleon_ml.dataset.pytorch import create_playing_self_play_dataloader
from napoleon_ml.dataset.split import DatasetSplit, SplitConfig
from napoleon_ml.dataset.validation import calculate_card_ids_sha256

_REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.integration
def test_cpp_generated_rl_binary_dataset_loads_current_python_trainer_batch() -> None:
    with tempfile.TemporaryDirectory(prefix="napoleon-cpp-rl-dataset-") as tmp_dir_name:
        tmp_root = Path(tmp_dir_name)
        output_directory = tmp_root / "dataset"

        _build_cpp_core()
        result = subprocess.run(
            [
                str(_REPO_ROOT / "packages/cpp-core/build/napoleon_rl_dataset_cli"),
                "--output",
                str(output_directory),
                "--start-seed",
                "0",
                "--games",
                "3",
                "--games-per-shard",
                "2",
                "--roster-seed",
                "17",
            ],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )

        assert result.returncode == 0, (
            f"napoleon_rl_dataset_cli failed (exit {result.returncode}):\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        cli_summary = json.loads(result.stdout)
        assert cli_summary["format"] == "playing-self-play-binary-v1"

        manifest = load_manifest(output_directory)

        assert manifest.dataset_schema_version == 4
        assert manifest.format == "playing-self-play-binary-v1"
        assert manifest.sample_type == "playing-self-play-sample"
        assert manifest.sample_schema_version == 4
        assert manifest.tensor_schema is not None
        assert manifest.tensor_schema.compression == "none"
        assert manifest.tensor_schema.fields[0].name == "modelInput"
        assert manifest.tensor_schema.fields[0].shape == (MODEL_INPUT_FEATURE_COUNT,)
        assert manifest.tensor_schema.fields[1].name == "legalPlayMask"
        assert manifest.tensor_schema.fields[1].shape == (53,)
        assert [field.name for field in manifest.tensor_schema.fields] == [
            "modelInput",
            "legalPlayMask",
            "selectedCardIndex",
            "behaviorLogProbability",
            "terminalReward",
            "seed",
            "step",
            "actingPlayerIndex",
            "selfRoleIndex",
        ]
        assert manifest.playing_encoder_schema_version == 2
        assert manifest.playing_model_input_schema_version == 2
        assert manifest.card_ids == EXPECTED_CARD_IDS
        assert manifest.card_ids_sha256 == calculate_card_ids_sha256()
        assert manifest.sample_count == cli_summary["sampleCount"]
        assert manifest.sample_count == 30
        assert manifest.shard_count == 2

        assert manifest.behavior_policy is not None
        assert manifest.behavior_policy.type == "playing-onnx"
        assert isinstance(manifest.behavior_policy.metadata, dict)
        metadata = manifest.behavior_policy.metadata
        assert metadata["producer"] == "cpp-rl-dataset-cli"
        assert metadata["sampleAttribution"] == "current-policy-only"
        assert metadata["rawCacheCompatible"] is True
        assert metadata["rosterSpec"] == {
            "kind": "current-plus-opponent-pool",
            "currentSeatRotation": "game-index-mod-player-count",
            "opponentPool": ["rule-based", "frozen-onnx"],
        }

        assert manifest.rollout_roster is not None
        assert manifest.rollout_roster.assignment == "rotate-by-seed"
        assert [seat.source for seat in manifest.rollout_roster.seats] == [
            "current-policy",
            "rule-based",
            "frozen-onnx",
            "rule-based",
            "frozen-onnx",
        ]

        loader = create_playing_self_play_dataloader(
            output_directory,
            split=DatasetSplit.TRAIN,
            split_config=SplitConfig(train=100, validation=0, test=0),
            batch_size=7,
        )
        batches = list(loader)
        loaded_sample_count = sum(int(batch["seed"].shape[0]) for batch in batches)
        assert loaded_sample_count == manifest.sample_count

        first = batches[0]
        _assert_cpp_rl_batch_contract(first)

        all_seeds = torch.cat([batch["seed"] for batch in batches])
        all_steps = torch.cat([batch["step"] for batch in batches])
        all_actors = torch.cat([batch["acting_player_index"] for batch in batches])
        all_roles = torch.cat([batch["self_role_index"] for batch in batches])
        assert set(all_seeds.tolist()) == {0, 1, 2}
        assert all((0 <= value <= 4) for value in all_actors.tolist())
        assert all((0 <= value <= 3) for value in all_roles.tolist())
        assert all(value > 0 for value in all_steps.tolist())

        # One current-policy seat is rotated by seed; only that seat's ten
        # playing decisions per game become training samples.
        actor_by_seed = {
            int(seed): sorted(set(all_actors[all_seeds == seed].tolist()))
            for seed in all_seeds.unique(sorted=True)
        }
        assert actor_by_seed == {0: [0], 1: [1], 2: [2]}


def _build_cpp_core() -> None:
    result = subprocess.run(
        ["pnpm", "--filter", "@napoleon/cpp-core", "build"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, (
        f"pnpm cpp-core build failed (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _assert_cpp_rl_batch_contract(batch: dict[str, Any]) -> None:
    assert batch["model_input"].shape == (7, MODEL_INPUT_FEATURE_COUNT)
    assert batch["model_input"].dtype == torch.float32
    assert torch.isfinite(batch["model_input"]).all()
    assert batch["legal_play_mask"].shape == (7, 53)
    assert batch["legal_play_mask"].dtype == torch.bool
    assert batch["selected_card_index"].shape == (7,)
    assert batch["selected_card_index"].dtype == torch.int64
    assert batch["behavior_log_probability"].shape == (7,)
    assert batch["behavior_log_probability"].dtype == torch.float32
    assert torch.isfinite(batch["behavior_log_probability"]).all()
    assert (batch["behavior_log_probability"] <= 0).all()
    assert batch["terminal_reward"].shape == (7,)
    assert batch["terminal_reward"].dtype == torch.float32
    assert set(batch["terminal_reward"].tolist()) <= {-1.0, 1.0}
    assert batch["legal_play_mask"][torch.arange(7), batch["selected_card_index"]].all()
    assert batch["seed"].dtype == torch.int64
    assert batch["step"].dtype == torch.int64
    assert batch["acting_player_index"].dtype == torch.int64
    assert batch["self_role_index"].shape == (7,)
    assert batch["self_role_index"].dtype == torch.int64
