"""One-command non-playing RL rollout, PPO, ONNX export, and evaluation."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar, cast

from napoleon_ml.adjutant.model import AdjutantMlpConfig
from napoleon_ml.adjutant.ppo import AdjutantPpoTrainSettings, train_adjutant_ppo
from napoleon_ml.bidding.model import BiddingMlpConfig
from napoleon_ml.bidding.ppo import BiddingPpoTrainSettings, train_bidding_ppo
from napoleon_ml.exchange.model import ExchangeMlpConfig
from napoleon_ml.exchange.ppo import ExchangePpoTrainSettings, train_exchange_ppo
from napoleon_ml.nonplaying_onnx_export import (
    PolicyType,
    export_adjutant_rl_checkpoint_to_onnx,
    export_bidding_rl_checkpoint_to_onnx,
    export_exchange_rl_checkpoint_to_onnx,
    export_seeded_nonplaying_bootstrap_policy_to_onnx,
)

DEFAULT_GAMES = 20
DEFAULT_EVALUATION_GAMES = 5
DEFAULT_GAMES_PER_SHARD = 20
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_HIDDEN_DIM = 128
DEFAULT_HIDDEN_LAYERS = 2
DEFAULT_DROPOUT = 0.0
DEFAULT_PPO_CLIP_EPSILON = 0.2
DEFAULT_VALUE_LOSS_COEFFICIENT = 0.5
DEFAULT_TEMPERATURE = 1.0
DEFAULT_INFERENCE_DEVICE: Literal["cpu", "auto", "cuda"] = "cpu"
DEFAULT_INFERENCE_MAX_BATCH_SIZE = 256
DEFAULT_SEED = 202
DEFAULT_PLAYING_POLICY_ARTIFACT_ID = "ppo-separated-v1000"
DEFAULT_PLAYING_POLICY_ONNX = Path("benchmarks/playing-policies/ppo-separated-v1000/policy.onnx")
DEFAULT_PLAYING_POLICY_METADATA = Path(
    "benchmarks/playing-policies/ppo-separated-v1000/policy.json"
)
SUPPORTED_INFERENCE_DEVICES = ("cpu", "auto", "cuda")

PhaseName = Literal["bidding", "adjutant", "exchange"]
_T = TypeVar("_T")


class NonPlayingRlOrchestratorError(RuntimeError):
    """Raised when the non-playing RL pipeline cannot continue safely."""


@dataclass(frozen=True)
class NonPlayingRlRunConfig:
    output_dir: Path
    games: int = DEFAULT_GAMES
    evaluation_games: int = DEFAULT_EVALUATION_GAMES
    games_per_shard: int | None = None
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    hidden_layers: int = DEFAULT_HIDDEN_LAYERS
    dropout: float = DEFAULT_DROPOUT
    ppo_clip_epsilon: float = DEFAULT_PPO_CLIP_EPSILON
    value_loss_coefficient: float = DEFAULT_VALUE_LOSS_COEFFICIENT
    seed: int = DEFAULT_SEED
    temperature: float = DEFAULT_TEMPERATURE
    inference_device: Literal["cpu", "auto", "cuda"] = DEFAULT_INFERENCE_DEVICE
    inference_max_batch_size: int = DEFAULT_INFERENCE_MAX_BATCH_SIZE
    playing_policy_onnx: Path = DEFAULT_PLAYING_POLICY_ONNX
    playing_policy_metadata: Path = DEFAULT_PLAYING_POLICY_METADATA
    playing_policy_artifact_id: str = DEFAULT_PLAYING_POLICY_ARTIFACT_ID
    build_typescript: bool = True
    overwrite: bool = False

    def normalized(self) -> NonPlayingRlRunConfig:
        return NonPlayingRlRunConfig(
            output_dir=self.output_dir.expanduser().resolve(),
            games=self.games,
            evaluation_games=self.evaluation_games,
            games_per_shard=self.games_per_shard,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            ppo_clip_epsilon=self.ppo_clip_epsilon,
            value_loss_coefficient=self.value_loss_coefficient,
            seed=self.seed,
            temperature=self.temperature,
            inference_device=self.inference_device,
            inference_max_batch_size=self.inference_max_batch_size,
            playing_policy_onnx=_resolve_repo_path(self.playing_policy_onnx),
            playing_policy_metadata=_resolve_repo_path(self.playing_policy_metadata),
            playing_policy_artifact_id=self.playing_policy_artifact_id,
            build_typescript=self.build_typescript,
            overwrite=self.overwrite,
        )

    @property
    def effective_games_per_shard(self) -> int:
        return self.games_per_shard if self.games_per_shard is not None else self.games

    def settings_dict(self) -> dict[str, object]:
        return {
            "games": self.games,
            "evaluationGames": self.evaluation_games,
            "gamesPerShard": self.effective_games_per_shard,
            "epochs": self.epochs,
            "batchSize": self.batch_size,
            "learningRate": self.learning_rate,
            "hiddenDim": self.hidden_dim,
            "hiddenLayers": self.hidden_layers,
            "dropout": self.dropout,
            "ppoClipEpsilon": self.ppo_clip_epsilon,
            "valueLossCoefficient": self.value_loss_coefficient,
            "seed": self.seed,
            "temperature": self.temperature,
            "inferenceDevice": self.inference_device,
            "inferenceMaxBatchSize": self.inference_max_batch_size,
            "playingPolicyOnnx": str(self.playing_policy_onnx),
            "playingPolicyMetadata": str(self.playing_policy_metadata),
            "playingPolicyArtifactId": self.playing_policy_artifact_id,
        }


def run_nonplaying_rl_pipeline(config: NonPlayingRlRunConfig) -> dict[str, object]:
    config = config.normalized()
    _validate_config(config)
    _prepare_output_dir(config)
    if config.build_typescript:
        _stage("typescript-build", _build_typescript_helpers)

    started = time.monotonic()
    summary: dict[str, object] = {
        "schemaVersion": 1,
        "runType": "non-playing-rl-smoke",
        "settings": config.settings_dict(),
        "artifacts": {
            "playing": {
                "onnxPath": str(config.playing_policy_onnx),
                "metadataPath": str(config.playing_policy_metadata),
            }
        },
        "phases": {},
    }
    final_artifacts: dict[PhaseName, dict[str, str]] = {}

    for offset, phase in enumerate(("bidding", "adjutant", "exchange")):
        phase_summary, artifact = _run_phase(config, cast(PhaseName, phase), offset)
        cast(dict[str, object], summary["phases"])[phase] = phase_summary
        final_artifacts[cast(PhaseName, phase)] = artifact

    evaluation_path = config.output_dir / "evaluation.json"
    evaluation_summary = _stage(
        "full-policy-evaluation",
        lambda: _run_full_policy_evaluation(config, final_artifacts, evaluation_path),
    )
    summary["evaluation"] = evaluation_summary
    summary["artifactPaths"] = {
        "bidding": final_artifacts["bidding"],
        "adjutant": final_artifacts["adjutant"],
        "exchange": final_artifacts["exchange"],
        "evaluation": str(evaluation_path),
        "runSummary": str(config.output_dir / "run-summary.json"),
    }
    summary["completedAtUnixSeconds"] = int(time.time())
    summary["elapsedSeconds"] = time.monotonic() - started
    _atomic_write_json(config.output_dir / "run-summary.json", summary)
    _print_completion(summary)
    return summary


def _run_phase(
    config: NonPlayingRlRunConfig,
    phase: PhaseName,
    offset: int,
) -> tuple[dict[str, object], dict[str, str]]:
    phase_dir = config.output_dir / phase
    dataset_dir = phase_dir / "dataset"
    checkpoint_path = phase_dir / "checkpoint.pt"
    onnx_path = phase_dir / "policy.onnx"
    metadata_path = phase_dir / "policy.json"
    bootstrap_dir = phase_dir / "bootstrap"
    bootstrap_onnx = bootstrap_dir / "policy.onnx"
    bootstrap_metadata = bootstrap_dir / "policy.json"
    rollout_seed = config.seed + offset * 100_000
    training_seed = config.seed + offset * 100_000 + 1

    bootstrap_report = _stage(
        f"{phase}-bootstrap-export",
        lambda: export_seeded_nonplaying_bootstrap_policy_to_onnx(
            policy_type=cast(PolicyType, phase),
            onnx_path=bootstrap_onnx,
            metadata_path=bootstrap_metadata,
            seed=training_seed,
            hidden_dim=config.hidden_dim,
            hidden_layers=config.hidden_layers,
            dropout=config.dropout,
        ),
    )
    rollout_summary = _stage(
        f"{phase}-rollout",
        lambda: _run_nonplaying_rollout(
            config,
            phase=phase,
            policy_onnx=bootstrap_onnx,
            policy_metadata=bootstrap_metadata,
            dataset_dir=dataset_dir,
            start_seed=rollout_seed,
        ),
    )
    train_report = _stage(
        f"{phase}-train",
        lambda: _train_phase(config, phase, dataset_dir, checkpoint_path, training_seed),
    )
    export_report = _stage(
        f"{phase}-export",
        lambda: _export_trained_phase(
            phase,
            dataset_dir,
            checkpoint_path,
            onnx_path,
            metadata_path,
        ),
    )

    artifact = {
        "checkpointPath": str(checkpoint_path),
        "onnxPath": str(onnx_path),
        "metadataPath": str(metadata_path),
    }
    return (
        {
            "bootstrap": bootstrap_report,
            "rollout": rollout_summary,
            "train": train_report,
            "export": export_report,
            "artifact": artifact,
        },
        artifact,
    )


def _run_nonplaying_rollout(
    config: NonPlayingRlRunConfig,
    *,
    phase: PhaseName,
    policy_onnx: Path,
    policy_metadata: Path,
    dataset_dir: Path,
    start_seed: int,
) -> dict[str, object]:
    return _run_node_json(
        [
            "node",
            str(_repo_root() / "apps/self-play-cli/dist/index.js"),
            "non-playing-rollout",
            "--phase",
            phase,
            "--policy-onnx",
            str(policy_onnx),
            "--policy-metadata",
            str(policy_metadata),
            "--playing-onnx",
            str(config.playing_policy_onnx),
            "--playing-metadata",
            str(config.playing_policy_metadata),
            "--output",
            str(dataset_dir),
            "--start-seed",
            str(start_seed),
            "--games",
            str(config.games),
            "--games-per-shard",
            str(config.effective_games_per_shard),
            "--temperature",
            repr(config.temperature),
            "--inference-device",
            config.inference_device,
            "--inference-max-batch-size",
            str(config.inference_max_batch_size),
            "--artifact-id",
            f"{phase}-bootstrap-seed-{start_seed}",
            "--playing-artifact-id",
            config.playing_policy_artifact_id,
            "--progress-prefix",
            f"[{phase} rollout] ",
        ],
        cwd=_repo_root(),
    )


def _train_phase(
    config: NonPlayingRlRunConfig,
    phase: PhaseName,
    dataset_dir: Path,
    checkpoint_path: Path,
    training_seed: int,
) -> dict[str, object]:
    if phase == "bidding":
        return train_bidding_ppo(
            dataset_directory=dataset_dir,
            output_checkpoint_path=checkpoint_path,
            settings=BiddingPpoTrainSettings(
                seed=training_seed,
                epochs=config.epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                ppo_clip_epsilon=config.ppo_clip_epsilon,
                value_loss_coefficient=config.value_loss_coefficient,
                parent_actor_checkpoint=None,
            ),
            model_config=BiddingMlpConfig(
                hidden_dim=config.hidden_dim,
                hidden_layers=config.hidden_layers,
                dropout=config.dropout,
            ),
        ).to_dict()
    if phase == "adjutant":
        return train_adjutant_ppo(
            dataset_directory=dataset_dir,
            output_checkpoint_path=checkpoint_path,
            settings=AdjutantPpoTrainSettings(
                seed=training_seed,
                epochs=config.epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                ppo_clip_epsilon=config.ppo_clip_epsilon,
                value_loss_coefficient=config.value_loss_coefficient,
                parent_actor_checkpoint=None,
            ),
            model_config=AdjutantMlpConfig(
                hidden_dim=config.hidden_dim,
                hidden_layers=config.hidden_layers,
                dropout=config.dropout,
            ),
        ).to_dict()
    return train_exchange_ppo(
        dataset_directory=dataset_dir,
        output_checkpoint_path=checkpoint_path,
        settings=ExchangePpoTrainSettings(
            seed=training_seed,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            ppo_clip_epsilon=config.ppo_clip_epsilon,
            value_loss_coefficient=config.value_loss_coefficient,
            parent_actor_checkpoint=None,
        ),
        model_config=ExchangeMlpConfig(
            hidden_dim=config.hidden_dim,
            hidden_layers=config.hidden_layers,
            dropout=config.dropout,
        ),
    ).to_dict()


def _export_trained_phase(
    phase: PhaseName,
    dataset_dir: Path,
    checkpoint_path: Path,
    onnx_path: Path,
    metadata_path: Path,
) -> dict[str, object]:
    if phase == "bidding":
        report = export_bidding_rl_checkpoint_to_onnx(
            dataset_directory=dataset_dir,
            checkpoint_path=checkpoint_path,
            onnx_path=onnx_path,
            metadata_path=metadata_path,
        )
    elif phase == "adjutant":
        report = export_adjutant_rl_checkpoint_to_onnx(
            dataset_directory=dataset_dir,
            checkpoint_path=checkpoint_path,
            onnx_path=onnx_path,
            metadata_path=metadata_path,
        )
    else:
        report = export_exchange_rl_checkpoint_to_onnx(
            dataset_directory=dataset_dir,
            checkpoint_path=checkpoint_path,
            onnx_path=onnx_path,
            metadata_path=metadata_path,
        )
    return report.to_dict()


def _run_full_policy_evaluation(
    config: NonPlayingRlRunConfig,
    artifacts: dict[PhaseName, dict[str, str]],
    evaluation_path: Path,
) -> dict[str, object]:
    return _run_node_json(
        [
            "node",
            str(_repo_root() / "apps/self-play-cli/dist/index.js"),
            "full-policy-evaluate",
            "--playing-onnx",
            str(config.playing_policy_onnx),
            "--playing-metadata",
            str(config.playing_policy_metadata),
            "--bidding-onnx",
            artifacts["bidding"]["onnxPath"],
            "--bidding-metadata",
            artifacts["bidding"]["metadataPath"],
            "--adjutant-onnx",
            artifacts["adjutant"]["onnxPath"],
            "--adjutant-metadata",
            artifacts["adjutant"]["metadataPath"],
            "--exchange-onnx",
            artifacts["exchange"]["onnxPath"],
            "--exchange-metadata",
            artifacts["exchange"]["metadataPath"],
            "--output",
            str(evaluation_path),
            "--start-seed",
            str(config.seed + 300_000),
            "--games",
            str(config.evaluation_games),
            "--inference-device",
            config.inference_device,
            "--inference-max-batch-size",
            str(config.inference_max_batch_size),
            "--progress-prefix",
            "[full-policy eval] ",
        ],
        cwd=_repo_root(),
    )


def _stage(name: str, fn: Callable[[], _T]) -> _T:
    print(f"[stage] {name}", flush=True)
    try:
        return fn()
    except Exception as error:
        if isinstance(error, NonPlayingRlOrchestratorError) and str(error).startswith("stage "):
            raise
        raise NonPlayingRlOrchestratorError(f"stage '{name}' failed: {error}") from error


def _validate_config(config: NonPlayingRlRunConfig) -> None:
    _validate_positive_int(config.games, "games")
    _validate_positive_int(config.evaluation_games, "evaluation-games")
    _validate_positive_int(config.effective_games_per_shard, "games-per-shard")
    _validate_positive_int(config.epochs, "epochs")
    _validate_positive_int(config.batch_size, "batch-size")
    _validate_positive_int(config.hidden_dim, "hidden-dim")
    _validate_positive_int(config.hidden_layers, "hidden-layers")
    _validate_positive_int(config.inference_max_batch_size, "inference-max-batch-size")
    _validate_positive_float(config.learning_rate, "learning-rate")
    _validate_positive_float(config.ppo_clip_epsilon, "ppo-clip-epsilon")
    _validate_positive_float(config.temperature, "temperature")
    if config.value_loss_coefficient < 0.0:
        raise NonPlayingRlOrchestratorError("value-loss-coefficient must be non-negative.")
    if config.dropout < 0.0 or config.dropout >= 1.0:
        raise NonPlayingRlOrchestratorError("dropout must be in [0.0, 1.0).")
    if config.inference_device not in SUPPORTED_INFERENCE_DEVICES:
        raise NonPlayingRlOrchestratorError(
            f"inference-device must be one of {', '.join(SUPPORTED_INFERENCE_DEVICES)}."
        )
    _ensure_file(config.playing_policy_onnx, "playing policy ONNX")
    _ensure_file(config.playing_policy_metadata, "playing policy metadata")


def _prepare_output_dir(config: NonPlayingRlRunConfig) -> None:
    if config.output_dir.exists():
        if not config.output_dir.is_dir():
            raise NonPlayingRlOrchestratorError(
                f"output-dir exists and is not a directory: {config.output_dir}"
            )
        if any(config.output_dir.iterdir()):
            if not config.overwrite:
                raise NonPlayingRlOrchestratorError(
                    f"output-dir is not empty; choose a new path or pass --overwrite: "
                    f"{config.output_dir}"
                )
            shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)


def _build_typescript_helpers() -> None:
    result = subprocess.run(
        ["pnpm", "--filter", "@napoleon/self-play-cli...", "build"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    if result.returncode != 0:
        raise NonPlayingRlOrchestratorError(
            f"TypeScript helper build failed with exit {result.returncode}."
        )


def _run_node_json(command: Sequence[str], *, cwd: Path) -> dict[str, object]:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    if result.returncode != 0:
        raise NonPlayingRlOrchestratorError(
            f"command failed with exit {result.returncode}: {command}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise NonPlayingRlOrchestratorError(
            f"subprocess did not return JSON: {result.stdout!r}"
        ) from error
    if not isinstance(parsed, dict):
        raise NonPlayingRlOrchestratorError("subprocess JSON output must be an object.")
    return cast(dict[str, object], parsed)


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)


def _print_completion(summary: dict[str, object]) -> None:
    evaluation = cast(dict[str, object], summary["evaluation"])
    counts = cast(dict[str, object], evaluation["policyAgentDecisionCounts"])
    print(
        "[complete] "
        f"runSummary={cast(dict[str, object], summary['artifactPaths'])['runSummary']} "
        f"completed={evaluation['completedGames']}/{evaluation['scheduledGames']} "
        f"fallback={evaluation['fallbackCount']} "
        f"illegal={evaluation['illegalActionCount']} "
        f"bidding={counts['biddingOnnxCallCount']} "
        f"adjutant={counts['adjutantOnnxCallCount']} "
        f"exchange={counts['exchangeOnnxCallCount']}",
        flush=True,
    )


def _validate_positive_int(value: int, label: str) -> None:
    if isinstance(value, bool) or value <= 0:
        raise NonPlayingRlOrchestratorError(f"{label} must be a positive integer.")


def _validate_positive_float(value: float, label: str) -> None:
    if value <= 0.0:
        raise NonPlayingRlOrchestratorError(f"{label} must be positive.")


def _ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise NonPlayingRlOrchestratorError(f"{label} file does not exist: {path}")


def _resolve_repo_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (_repo_root() / expanded).resolve()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]
