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

from napoleon_ml.dataset import iter_samples, iter_tensorized_samples, load_manifest
from napoleon_ml.dataset.constants import EXPECTED_CARD_IDS
from napoleon_ml.dataset.pytorch import create_playing_dataloader
from napoleon_ml.dataset.split import DatasetSplit, split_for_seed
from napoleon_ml.dataset.tensors import MODEL_INPUT_FEATURE_COUNT
from napoleon_ml.dataset.validation import calculate_card_ids_sha256

_REPO_ROOT = Path(__file__).resolve().parents[3]
_GAME_COUNT = 3
_GAMES_PER_SHARD = 2
_SAMPLES_PER_GAME = 50


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
        assert batch["belief_target"].shape == (8, 53)
        assert str(batch["belief_target"].dtype) == "torch.int64"
        assert batch["belief_hidden_ownership_loss_mask"].shape == (8, 53)
        assert str(batch["belief_hidden_ownership_loss_mask"].dtype) == "torch.bool"
        assert batch["seed"].shape == (8,)
        assert batch["step"].shape == (8,)

        # --- no stray files outside the declared output directory -------

        assert {entry.name for entry in tmp_root.iterdir()} == {"dataset"}

    assert not tmp_root.exists()
