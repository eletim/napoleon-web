"""Cross-language integration test: TypeScript generator -> Python loader.

Marked ``@pytest.mark.integration`` since it shells out to
``pnpm self-play:generate`` and therefore needs Node/pnpm on ``PATH``.
``pytest -m "not integration"`` skips it for a fast inner loop; plain
``pytest`` (the repository's full test suite) always runs it -- this test
must fail loudly, not skip quietly, if ``pnpm`` is unavailable.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest
import torch

from napoleon_ml.dataset import (
    iter_samples,
    iter_tensorized_samples,
    load_manifest,
)
from napoleon_ml.dataset.constants import EXPECTED_CARD_IDS
from napoleon_ml.dataset.pytorch import (
    create_playing_dataloader,
    create_playing_self_play_dataloader,
    create_training_dataloader,
)
from napoleon_ml.dataset.sample import (
    AdjutantTrainingSample,
    BiddingTrainingSample,
    ExchangeTrainingSample,
)
from napoleon_ml.dataset.split import DatasetSplit, SplitConfig, split_for_seed
from napoleon_ml.dataset.tensors import (
    ADJUTANT_MODEL_INPUT_FEATURE_COUNT,
    BIDDING_MODEL_INPUT_FEATURE_COUNT,
    EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
    MODEL_INPUT_FEATURE_COUNT,
    TensorizedPlayingSample,
)
from napoleon_ml.dataset.validation import calculate_card_ids_sha256

_REPO_ROOT = Path(__file__).resolve().parents[3]
_GAME_COUNT = 3
_GAMES_PER_SHARD = 2
_SAMPLES_PER_GAME = 50
_MULTIPHASE_SAMPLE_TYPES = {
    "bidding-training-sample": BIDDING_MODEL_INPUT_FEATURE_COUNT,
    "exchange-training-sample": EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
    "adjutant-training-sample": ADJUTANT_MODEL_INPUT_FEATURE_COUNT,
}


@pytest.mark.integration
def test_typescript_generated_dataset_loads_and_tensorizes_cleanly() -> None:
    with tempfile.TemporaryDirectory(prefix="napoleon-python-integration-") as tmp_dir_name:
        tmp_root = Path(tmp_dir_name)
        output_directory = tmp_root / "dataset"

        result = subprocess.run(
            [
                "pnpm",
                "self-play:generate",
                "--",
                "--start-seed",
                "0",
                "--games",
                str(_GAME_COUNT),
                "--output",
                str(output_directory),
                "--games-per-shard",
                str(_GAMES_PER_SHARD),
            ],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )

        assert result.returncode == 0, (
            f"pnpm self-play:generate failed (exit {result.returncode}):\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

        # --- manifest ---------------------------------------------------

        manifest = load_manifest(output_directory)

        assert manifest.shard_count == 2
        assert manifest.game_count == _GAME_COUNT
        assert manifest.sample_count == _GAME_COUNT * _SAMPLES_PER_GAME
        assert manifest.start_seed == 0
        assert manifest.end_seed == 2
        assert manifest.card_ids == EXPECTED_CARD_IDS
        assert manifest.card_ids_sha256 == calculate_card_ids_sha256()

        # load_manifest() already re-verified every shard's SHA-256, byte
        # length, and line count against these manifest entries; assert
        # their shape here too so a future refactor that weakens that
        # guarantee is caught by this test directly, not just silently.
        for shard in manifest.shards:
            assert len(shard.sha256) == 64
            assert shard.byte_length > 0

        # --- every sample parses and validates ---------------------------

        sample_count_by_seed: dict[int, int] = {}
        total_samples = 0

        for sample in iter_samples(output_directory):
            sample_count_by_seed[sample.seed] = sample_count_by_seed.get(sample.seed, 0) + 1
            total_samples += 1

        assert total_samples == _GAME_COUNT * _SAMPLES_PER_GAME
        assert sample_count_by_seed == {0: 50, 1: 50, 2: 50}

        # --- every sample tensorizes cleanly and consistently -----------

        flat_shapes: set[tuple[int, ...]] = set()
        flat_dtypes: set[str] = set()
        legal_mask_shapes: set[tuple[int, ...]] = set()
        belief_target_shapes: set[tuple[int, ...]] = set()
        belief_target_dtypes: set[str] = set()
        seed_to_split: dict[int, DatasetSplit] = {}
        total_tensorized = 0

        for tensorized in iter_tensorized_samples(output_directory):
            assert isinstance(tensorized, TensorizedPlayingSample)
            total_tensorized += 1

            flat_shapes.add(tuple(tensorized.flat_observation.shape))
            flat_dtypes.add(str(tensorized.flat_observation.dtype))
            legal_mask_shapes.add(tuple(tensorized.legal_play_mask.shape))
            belief_target_shapes.add(tuple(tensorized.belief_target.shape))
            belief_target_dtypes.add(str(tensorized.belief_target.dtype))

            actor_target = int(tensorized.actor_target)
            assert tensorized.legal_play_mask[actor_target] == 1

            split = split_for_seed(tensorized.seed)
            seed_to_split.setdefault(tensorized.seed, split)
            assert seed_to_split[tensorized.seed] == split  # same seed -> same split, always

        assert total_tensorized == _GAME_COUNT * _SAMPLES_PER_GAME
        assert flat_shapes == {(684,)}
        assert flat_dtypes == {"float32"}
        assert legal_mask_shapes == {(53,)}
        assert belief_target_shapes == {(53,)}
        assert belief_target_dtypes == {"int64"}
        assert seed_to_split == {
            0: DatasetSplit.TRAIN,
            1: DatasetSplit.TRAIN,
            2: DatasetSplit.TRAIN,
        }

        # --- PyTorch DataLoader can read one real generated batch --------

        loader = create_playing_dataloader(
            output_directory,
            split=DatasetSplit.TRAIN,
            batch_size=8,
        )
        batch = next(iter(loader))

        assert batch["model_input"].shape == (8, MODEL_INPUT_FEATURE_COUNT)
        assert str(batch["model_input"].dtype) == "torch.float32"
        assert batch["actor_target"].shape == (8,)
        assert str(batch["actor_target"].dtype) == "torch.int64"
        assert batch["legal_play_mask"].shape == (8, 53)
        assert str(batch["legal_play_mask"].dtype) == "torch.bool"
        assert batch["legal_play_mask"][torch.arange(8), batch["actor_target"]].all()
        assert batch["belief_target"].shape == (8, 53)
        assert str(batch["belief_target"].dtype) == "torch.int64"
        assert batch["belief_hidden_ownership_loss_mask"].shape == (8, 53)
        assert str(batch["belief_hidden_ownership_loss_mask"].dtype) == "torch.bool"
        assert batch["seed"].shape == (8,)
        assert batch["step"].shape == (8,)

        # --- no stray files outside the declared output directory -------

        assert {entry.name for entry in tmp_root.iterdir()} == {"dataset"}

    assert not tmp_root.exists()


@pytest.mark.integration
def test_typescript_generated_multiphase_datasets_load_and_tensorize_cleanly() -> None:
    with tempfile.TemporaryDirectory(prefix="napoleon-python-multiphase-") as tmp_dir_name:
        tmp_root = Path(tmp_dir_name)
        _build_training_data_package()

        for sample_type, feature_count in _MULTIPHASE_SAMPLE_TYPES.items():
            output_directory = tmp_root / sample_type
            _generate_multiphase_dataset(sample_type=sample_type, output_directory=output_directory)

            manifest = load_manifest(output_directory)
            assert manifest.dataset_schema_version == 2
            assert manifest.sample_type == sample_type
            assert manifest.encoder_schema_version == 1
            assert manifest.card_ids == EXPECTED_CARD_IDS
            assert manifest.card_ids_sha256 == calculate_card_ids_sha256()

            samples = list(iter_samples(output_directory))
            assert len(samples) == manifest.sample_count
            assert len(samples) > 0
            non_playing_samples: list[
                BiddingTrainingSample | ExchangeTrainingSample | AdjutantTrainingSample
            ] = []
            for sample in samples:
                assert isinstance(
                    sample,
                    (BiddingTrainingSample, ExchangeTrainingSample, AdjutantTrainingSample),
                )
                non_playing_samples.append(sample)

            assert {sample.sample_type for sample in non_playing_samples} == {sample_type}
            assert min(sample.step for sample in non_playing_samples) >= 1

            tensorized_samples = list(iter_tensorized_samples(output_directory))
            assert len(tensorized_samples) == manifest.sample_count
            assert {tuple(sample.model_input.shape) for sample in tensorized_samples} == {
                (feature_count,)
            }
            assert {str(sample.model_input.dtype) for sample in tensorized_samples} == {"float32"}

            loader = create_training_dataloader(
                output_directory,
                split=DatasetSplit.TRAIN,
                batch_size=1,
            )
            batch = next(iter(loader))
            assert batch["model_input"].shape == (1, feature_count)
            assert str(batch["model_input"].dtype) == "torch.float32"

            if sample_type == "bidding-training-sample":
                assert batch["legal_bid_mask"].shape == (1, 29)
                assert str(batch["legal_bid_mask"].dtype) == "torch.bool"
                assert batch["actor_target"].shape == (1,)
                assert str(batch["actor_target"].dtype) == "torch.int64"
                assert batch["legal_bid_mask"][torch.arange(1), batch["actor_target"]].all()
            elif sample_type == "exchange-training-sample":
                assert batch["legal_discard_card_mask"].shape == (1, 53)
                assert str(batch["legal_discard_card_mask"].dtype) == "torch.bool"
                assert batch["discard_target_mask"].shape == (1, 53)
                assert str(batch["discard_target_mask"].dtype) == "torch.bool"
                assert batch["legal_discard_card_mask"].sum(dim=1).tolist() == [13]
                assert batch["discard_target_mask"].sum(dim=1).tolist() == [3]
                assert torch.logical_or(
                    torch.logical_not(batch["discard_target_mask"]),
                    batch["legal_discard_card_mask"],
                ).all()
            elif sample_type == "adjutant-training-sample":
                assert batch["legal_adjutant_mask"].shape == (1, 53)
                assert str(batch["legal_adjutant_mask"].dtype) == "torch.bool"
                assert batch["actor_target"].shape == (1,)
                assert str(batch["actor_target"].dtype) == "torch.int64"
                assert batch["legal_adjutant_mask"][torch.arange(1), batch["actor_target"]].all()
            else:
                raise AssertionError(f"Unexpected sample type: {sample_type}")

    assert not tmp_root.exists()


@pytest.mark.integration
def test_typescript_generated_playing_self_play_dataset_loads_rl_batch() -> None:
    with tempfile.TemporaryDirectory(prefix="napoleon-python-self-play-v3-") as tmp_dir_name:
        tmp_root = Path(tmp_dir_name)
        output_directory = tmp_root / "playing-self-play"
        _build_training_data_package()
        _generate_playing_self_play_dataset(output_directory=output_directory)

        manifest = load_manifest(output_directory)

        assert manifest.dataset_schema_version == 4
        assert manifest.generator_version == 1
        assert manifest.format == "playing-self-play-binary-v1"
        assert manifest.sample_type == "playing-self-play-sample"
        assert manifest.sample_schema_version == 4
        assert manifest.tensor_schema is not None
        assert manifest.playing_encoder_schema_version == 2
        assert manifest.playing_model_input_schema_version == 2
        assert manifest.sampling_algorithm == "masked-categorical"
        assert manifest.temperature == 1.25
        assert manifest.reward is not None
        assert manifest.reward.type == "terminal-team-win"
        assert manifest.reward.version == 1
        assert manifest.non_playing_agent is not None
        assert manifest.non_playing_agent.type == "rule-based"
        assert manifest.behavior_policy is not None
        assert manifest.behavior_policy.artifact_id == "python-integration-policy"
        assert manifest.card_ids == EXPECTED_CARD_IDS
        assert manifest.card_ids_sha256 == calculate_card_ids_sha256()

        loader = create_playing_self_play_dataloader(
            output_directory,
            split=DatasetSplit.TRAIN,
            split_config=SplitConfig(train=100, validation=0, test=0),
            batch_size=4,
        )
        batches = list(loader)
        batch = batches[0]

        assert batch["model_input"].shape == (4, MODEL_INPUT_FEATURE_COUNT)
        assert batch["model_input"].dtype == torch.float32
        assert batch["legal_play_mask"].shape == (4, 53)
        assert batch["legal_play_mask"].dtype == torch.bool
        assert batch["selected_card_index"].shape == (4,)
        assert batch["selected_card_index"].dtype == torch.int64
        assert batch["behavior_log_probability"].shape == (4,)
        assert batch["behavior_log_probability"].dtype == torch.float32
        assert torch.isfinite(batch["behavior_log_probability"]).all()
        assert batch["terminal_reward"].shape == (4,)
        assert batch["terminal_reward"].dtype == torch.float32
        assert set(batch["terminal_reward"].tolist()) <= {-1.0, 1.0}
        assert batch["legal_play_mask"][torch.arange(4), batch["selected_card_index"]].all()
        assert batch["seed"].dtype == torch.int64
        assert batch["step"].dtype == torch.int64

        loaded_sample_count = sum(int(next(iter(batch.values())).shape[0]) for batch in batches)
        assert loaded_sample_count == manifest.sample_count
        assert all((batch["behavior_log_probability"] <= 0).all() for batch in batches)

    assert not tmp_root.exists()


def _build_training_data_package() -> None:
    result = subprocess.run(
        ["pnpm", "--filter", "@napoleon/training-data...", "build"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, (
        f"pnpm training-data build failed (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _generate_multiphase_dataset(*, sample_type: str, output_directory: Path) -> None:
    script = """
import { generateRuleBasedDataset } from "./packages/training-data/dist/index.js";

await generateRuleBasedDataset({
  startSeed: 7,
  gameCount: 1,
  gamesPerShard: 1,
  sampleType: process.argv[1],
  outputDirectory: process.argv[2]
});
"""
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script, sample_type, str(output_directory)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, (
        f"node multiphase generator failed for {sample_type} (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _generate_playing_self_play_dataset(*, output_directory: Path) -> None:
    script = """
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { generatePlayingSelfPlayDataset } from "./packages/training-data/dist/index.js";

const outputDirectory = process.argv[1];
const artifactDirectory = join(dirname(outputDirectory), "artifacts");
await mkdir(artifactDirectory, { recursive: true });
const onnxPath = join(artifactDirectory, "policy.onnx");
const metadataPath = join(artifactDirectory, "policy.json");
await writeFile(onnxPath, new Uint8Array([1, 2, 3, 4]));
await writeFile(metadataPath, JSON.stringify({ metadataSchemaVersion: 1 }) + "\\n", "utf8");
const logits = Float32Array.from(Array.from({ length: 53 }, (_, index) => (index % 13) / 5));

await generatePlayingSelfPlayDataset({
  outputDirectory,
  playingPolicy: {
    metadata: { metadataSchemaVersion: 1, source: "python-integration" },
    predictLogits: async () => Float32Array.from(logits)
  },
  playingPolicyArtifact: {
    onnxPath,
    metadataPath,
    artifactId: "python-integration-policy"
  },
  startSeed: 21,
  gameCount: 3,
  gamesPerShard: 2,
  temperature: 1.25
});
"""
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script, str(output_directory)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, (
        f"node playing self-play generator failed (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
