from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import cast

import pytest
import torch

from napoleon_ml.cli.export_policy_onnx import main as export_main
from napoleon_ml.cli.train_policy_mlp import main as train_supervised_main
from napoleon_ml.dataset.constants import CARD_COUNT, EXPECTED_CARD_IDS
from napoleon_ml.dataset.pytorch import create_playing_self_play_dataloader
from napoleon_ml.dataset.reader import load_manifest
from napoleon_ml.dataset.split import DatasetSplit, SplitConfig
from napoleon_ml.dataset.validation import calculate_card_ids_sha256
from napoleon_ml.policy.checkpoint import load_policy_checkpoint
from napoleon_ml.policy.reinforce import (
    ReinforceTrainReport,
    ReinforceTrainSettings,
    train_policy_reinforce,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FIXTURE_PATH = Path(__file__).parents[1] / "unit" / "fixtures" / "valid_sample.json"


@pytest.mark.integration
def test_policy_reinforce_real_self_play_to_updated_onnx_smoke(tmp_path: Path) -> None:
    supervised_dataset = tmp_path / "supervised"
    supervised_dataset.mkdir()
    _write_supervised_dataset(supervised_dataset, seeds=(0, 1, 2))

    input_checkpoint = tmp_path / "input-playing.pt"
    behavior_onnx = tmp_path / "behavior.onnx"
    behavior_metadata = tmp_path / "behavior.json"
    self_play_dataset = tmp_path / "self-play"
    updated_checkpoint = tmp_path / "updated-playing.pt"
    updated_checkpoint_repro = tmp_path / "updated-playing-repro.pt"
    updated_onnx = tmp_path / "updated.onnx"
    updated_metadata = tmp_path / "updated.json"

    assert (
        train_supervised_main(
            [
                str(supervised_dataset),
                "--output",
                str(input_checkpoint),
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--hidden-dim",
                "8",
                "--hidden-layers",
                "1",
                "--train-ratio",
                "100",
                "--validation-ratio",
                "0",
                "--test-ratio",
                "0",
                "--seed",
                "7",
                "--json",
            ]
        )
        == 0
    )
    assert (
        export_main(
            [
                str(supervised_dataset),
                "--checkpoint",
                str(input_checkpoint),
                "--output",
                str(behavior_onnx),
                "--metadata-output",
                str(behavior_metadata),
                "--json",
            ]
        )
        == 0
    )

    _build_typescript_policy_self_play_packages()
    _generate_real_playing_self_play_dataset(
        onnx_path=behavior_onnx,
        metadata_path=behavior_metadata,
        output_directory=self_play_dataset,
    )

    manifest = load_manifest(self_play_dataset)
    assert manifest.game_count == 2
    assert manifest.sample_count > 0
    assert manifest.behavior_policy is not None
    assert manifest.behavior_policy.onnx_sha256 == _sha256_file(behavior_onnx)
    assert manifest.behavior_policy.metadata_sha256 == _sha256_file(behavior_metadata)

    settings = ReinforceTrainSettings(
        seed=99,
        epochs=1,
        batch_size=16,
        learning_rate=0.001,
        verify_integrity=True,
    )
    report = _run_reinforce(
        input_checkpoint=input_checkpoint,
        self_play_dataset=self_play_dataset,
        output_checkpoint=updated_checkpoint,
        settings=settings,
    )

    assert report.sample_count == manifest.sample_count
    assert report.optimizer_step_count > 0
    assert report.changed_parameter_count > 0
    assert report.parameter_delta_norm > 0
    assert report.forced_sample_count > 0
    assert report.non_forced_sample_count > 0
    assert report.max_behavior_log_probability_parity_error <= 1e-4

    reloaded_model, reloaded_checkpoint = load_policy_checkpoint(
        updated_checkpoint,
        manifest=load_manifest(supervised_dataset),
    )
    provenance = cast(dict[str, object], reloaded_checkpoint["rl_provenance"])
    assert provenance["algorithm"] == "reinforce-v1"
    assert provenance["optimizerSteps"] == report.optimizer_step_count
    assert reloaded_model.config.hidden_dim == 8

    export_report = export_main(
        [
            str(supervised_dataset),
            "--checkpoint",
            str(updated_checkpoint),
            "--output",
            str(updated_onnx),
            "--metadata-output",
            str(updated_metadata),
            "--json",
        ]
    )
    assert export_report == 0
    assert updated_onnx.is_file()
    assert updated_metadata.is_file()

    repro_report = _run_reinforce(
        input_checkpoint=input_checkpoint,
        self_play_dataset=self_play_dataset,
        output_checkpoint=updated_checkpoint_repro,
        settings=settings,
    )
    assert repro_report.optimizer_step_count == report.optimizer_step_count
    _assert_checkpoint_parameters_equal(updated_checkpoint, updated_checkpoint_repro)


def _run_reinforce(
    *,
    input_checkpoint: Path,
    self_play_dataset: Path,
    output_checkpoint: Path,
    settings: ReinforceTrainSettings,
) -> ReinforceTrainReport:
    loader = create_playing_self_play_dataloader(
        self_play_dataset,
        split=DatasetSplit.TRAIN,
        split_config=SplitConfig(train=100, validation=0, test=0),
        batch_size=settings.batch_size,
        verify_integrity=settings.verify_integrity,
    )
    return train_policy_reinforce(
        input_checkpoint=input_checkpoint,
        self_play_dataset_directory=self_play_dataset,
        output_checkpoint=output_checkpoint,
        manifest=load_manifest(self_play_dataset),
        dataloader=loader,
        settings=settings,
    )


def _build_typescript_policy_self_play_packages() -> None:
    for package in ("@napoleon/policy-onnx", "@napoleon/training-data..."):
        result = subprocess.run(
            ["pnpm", "--filter", package, "build"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, (
            f"pnpm build failed for {package} (exit {result.returncode}):\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def _generate_real_playing_self_play_dataset(
    *,
    onnx_path: Path,
    metadata_path: Path,
    output_directory: Path,
) -> None:
    script = """
import { loadPolicyOnnxModel } from "./packages/policy-onnx/dist/index.js";
import { generatePlayingSelfPlayDataset } from "./packages/training-data/dist/index.js";

const onnxPath = process.argv[1];
const metadataPath = process.argv[2];
const outputDirectory = process.argv[3];
const policy = await loadPolicyOnnxModel({ onnxPath, metadataPath });

await generatePlayingSelfPlayDataset({
  outputDirectory,
  playingPolicy: policy,
  playingPolicyArtifact: {
    onnxPath,
    metadataPath,
    artifactId: "python-reinforce-e2e-policy"
  },
  startSeed: 101,
  gameCount: 2,
  gamesPerShard: 1,
  temperature: 1.0
});
"""
    result = subprocess.run(
        [
            "node",
            "--input-type=module",
            "--eval",
            script,
            str(onnx_path),
            str(metadata_path),
            str(output_directory),
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"node playing self-play generator failed (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _write_supervised_dataset(directory: Path, *, seeds: tuple[int, ...]) -> None:
    base = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    samples = [dict(base, seed=seed, step=1) for seed in seeds]
    shard_bytes = b"".join((json.dumps(sample) + "\n").encode("utf-8") for sample in samples)
    (directory / "shard-00000.jsonl").write_bytes(shard_bytes)
    manifest = {
        "datasetSchemaVersion": 1,
        "generatorVersion": 1,
        "playingEncoderSchemaVersion": 3,
        "format": "jsonl",
        "sampleType": "playing-training-sample",
        "agent": {"type": "rule-based", "version": 1},
        "startSeed": min(seeds),
        "endSeed": max(seeds),
        "gameCount": len(seeds),
        "sampleCount": len(samples),
        "gamesPerShard": len(seeds),
        "shardCount": 1,
        "playerCount": 5,
        "cardCount": CARD_COUNT,
        "cardIds": list(EXPECTED_CARD_IDS),
        "cardIdsSha256": calculate_card_ids_sha256(),
        "shards": [
            {
                "file": "shard-00000.jsonl",
                "startSeed": min(seeds),
                "endSeed": max(seeds),
                "gameCount": len(seeds),
                "sampleCount": len(samples),
                "byteLength": len(shard_bytes),
                "sha256": hashlib.sha256(shard_bytes).hexdigest(),
            }
        ],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _assert_checkpoint_parameters_equal(left_path: Path, right_path: Path) -> None:
    left = torch.load(left_path, map_location="cpu", weights_only=True)
    right = torch.load(right_path, map_location="cpu", weights_only=True)
    left_state = cast(dict[str, torch.Tensor], left["model_state"])
    right_state = cast(dict[str, torch.Tensor], right["model_state"])
    assert left_state.keys() == right_state.keys()
    for key in left_state:
        assert torch.equal(left_state[key], right_state[key])


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
