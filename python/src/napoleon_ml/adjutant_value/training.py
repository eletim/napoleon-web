"""Candidate-value MLP training for Issue #446 adjutant joint teacher datasets."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from napoleon_ml.policy.device import resolve_torch_device

ADJUTANT_VALUE_FEATURE_COUNT = 290
ADJUTANT_CANDIDATE_COUNT = 53
ADJUTANT_VALUE_MODEL_TYPE = "adjutant-joint-value-mlp"
ADJUTANT_VALUE_CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AdjutantValueMlpConfig:
    input_dim: int = ADJUTANT_VALUE_FEATURE_COUNT
    hidden_dims: tuple[int, ...] = (512, 512, 256, 256)
    dropout: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "architectureId": "adjutant-joint-value-mlp-v1",
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "output": "scalarContractMargin",
        }


class AdjutantValueMlp(nn.Module):
    def __init__(self, config: AdjutantValueMlpConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        input_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, model_input: Tensor) -> Tensor:
        if model_input.ndim != 2:
            raise ValueError("model_input must have shape (batch, features).")
        if model_input.shape[1] != self.config.input_dim:
            raise ValueError(
                f"model_input feature count must be {self.config.input_dim}, "
                f"got {model_input.shape[1]}."
            )
        return self.network(model_input).squeeze(-1)


@dataclass(frozen=True)
class LoadedAdjutantValueDataset:
    directory: Path
    manifest: dict[str, Any]
    features: np.memmap
    margins: np.memmap
    rewards: np.memmap
    state_indices: np.memmap
    candidate_cards: np.memmap
    source_state_count: int
    sample_count: int


class _TorchAdjutantValueDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, dataset: LoadedAdjutantValueDataset, indices: np.ndarray) -> None:
        self.dataset = dataset
        self.indices = indices.astype(np.int64, copy=True)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        row = int(self.indices[index])
        features = np.array(self.dataset.features[row], dtype=np.float32, copy=True)
        margin = np.float32(self.dataset.margins[row])
        return torch.from_numpy(features), torch.tensor(margin, dtype=torch.float32)


def train_adjutant_value_model(
    dataset_directory: Path,
    *,
    output_directory: Path,
    full_gold_directory: Path | None = None,
    seed: int = 446,
    epochs: int = 30,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    hidden_dims: tuple[int, ...] = (512, 512, 256, 256),
    dropout: float = 0.0,
    huber_delta: float = 1.0,
    weight_decay: float = 1e-4,
    patience: int = 6,
    device_name: str = "auto",
) -> dict[str, object]:
    dataset = load_adjutant_value_dataset(dataset_directory)
    split = split_by_state(dataset, seed=seed)
    device = resolve_torch_device(device_name)
    torch.manual_seed(seed)
    model = AdjutantValueMlp(
        AdjutantValueMlpConfig(hidden_dims=hidden_dims, dropout=dropout)
    ).to(device.torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    train_loader = DataLoader(
        _TorchAdjutantValueDataset(dataset, split["train"]),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        _TorchAdjutantValueDataset(dataset, split["validation"]),
        batch_size=batch_size,
        shuffle=False,
    )

    best_state: dict[str, Tensor] | None = None
    best_validation = math.inf
    best_epoch = 0
    stale_epochs = 0
    epoch_reports: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device.torch_device,
            huber_delta=huber_delta,
        )
        validation_loss = _run_epoch(
            model,
            validation_loader,
            optimizer=None,
            device=device.torch_device,
            huber_delta=huber_delta,
        )
        epoch_reports.append(
            {"epoch": epoch, "trainHuber": train_loss, "validationHuber": validation_loss}
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint state.")
    model.load_state_dict(best_state)

    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_directory / "checkpoint.pt"
    checkpoint = {
        "checkpointSchemaVersion": ADJUTANT_VALUE_CHECKPOINT_SCHEMA_VERSION,
        "modelType": ADJUTANT_VALUE_MODEL_TYPE,
        "modelConfig": model.config.to_dict(),
        "modelState": best_state,
        "trainingConfig": {
            "seed": seed,
            "epochs": epochs,
            "batchSize": batch_size,
            "learningRate": learning_rate,
            "hiddenDims": list(hidden_dims),
            "dropout": dropout,
            "loss": "huber",
            "huberDelta": huber_delta,
            "weightDecay": weight_decay,
            "patience": patience,
        },
    }
    torch.save(checkpoint, checkpoint_path)

    reports = {
        "bestEpoch": best_epoch,
        "bestValidationHuber": best_validation,
        "epochs": epoch_reports,
        "split": {name: int(values.shape[0]) for name, values in split.items()},
        "approxTeacher": {
            name: evaluate_model_on_indices(model, dataset, values, device=device.torch_device)
            for name, values in split.items()
        },
    }
    if full_gold_directory is not None:
        full_gold = load_adjutant_value_dataset(full_gold_directory)
        reports["fullGold"] = evaluate_full_gold(
            model,
            full_gold,
            device=device.torch_device,
        )

    report_path = output_directory / "report.json"
    report_path.write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata = {
        "artifactType": "adjutant-joint-value-checkpoint",
        "checkpointPath": str(checkpoint_path),
        "checkpointSha256": sha256(checkpoint_path),
        "reportPath": str(report_path),
        "reportSha256": sha256(report_path),
        "datasetPath": str(dataset_directory),
        "datasetManifestSha256": sha256(dataset_directory / "manifest.json"),
        "fullGoldPath": None if full_gold_directory is None else str(full_gold_directory),
        "device": device.to_metadata(),
    }
    metadata_path = output_directory / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"metadata": metadata, "report": reports}


def load_adjutant_value_dataset(directory: Path | str) -> LoadedAdjutantValueDataset:
    path = Path(directory)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    sample_count = int(manifest["sampleCount"])
    source_state_count = int(manifest["sourceStateCount"])
    feature_count = int(manifest["featureCount"])
    if feature_count != ADJUTANT_VALUE_FEATURE_COUNT:
        raise ValueError(f"featureCount must be {ADJUTANT_VALUE_FEATURE_COUNT}.")
    expected_samples = source_state_count * ADJUTANT_CANDIDATE_COUNT
    if sample_count != expected_samples:
        raise ValueError(
            f"sampleCount must be sourceStateCount*53, got {sample_count} vs {expected_samples}."
        )
    return LoadedAdjutantValueDataset(
        directory=path,
        manifest=manifest,
        features=np.memmap(
            path / "features.f32",
            mode="r",
            dtype="<f4",
            shape=(sample_count, ADJUTANT_VALUE_FEATURE_COUNT),
        ),
        margins=np.memmap(
            path / "contract-margin.f32", mode="r", dtype="<f4", shape=(sample_count,)
        ),
        rewards=np.memmap(
            path / "relative-reward.f32", mode="r", dtype="<f4", shape=(sample_count,)
        ),
        state_indices=np.memmap(
            path / "state-index.u32", mode="r", dtype="<u4", shape=(sample_count,)
        ),
        candidate_cards=np.memmap(
            path / "candidate-card.u8", mode="r", dtype="u1", shape=(sample_count,)
        ),
        source_state_count=source_state_count,
        sample_count=sample_count,
    )


def split_by_state(dataset: LoadedAdjutantValueDataset, *, seed: int) -> dict[str, np.ndarray]:
    if dataset.source_state_count < 3:
        raise ValueError("at least 3 source states are required for train/validation/final splits.")
    rng = np.random.default_rng(seed)
    states = np.arange(dataset.source_state_count, dtype=np.uint32)
    rng.shuffle(states)
    validation_count = max(1, int(round(dataset.source_state_count * 0.1)))
    final_count = max(1, int(round(dataset.source_state_count * 0.1)))
    if validation_count + final_count >= dataset.source_state_count:
        validation_count = max(1, dataset.source_state_count // 5)
        final_count = max(1, dataset.source_state_count // 5)
    validation_states = set(int(value) for value in states[:validation_count])
    final_states = set(
        int(value) for value in states[validation_count : validation_count + final_count]
    )
    train_states = set(int(value) for value in states[validation_count + final_count :])
    result: dict[str, np.ndarray] = {}
    state_indices = np.asarray(dataset.state_indices, dtype=np.uint32)
    for name, state_set in (
        ("train", train_states),
        ("validation", validation_states),
        ("final", final_states),
    ):
        mask = np.fromiter((int(value) in state_set for value in state_indices), dtype=bool)
        result[name] = np.nonzero(mask)[0].astype(np.int64)
    return result


def evaluate_model_on_indices(
    model: AdjutantValueMlp,
    dataset: LoadedAdjutantValueDataset,
    indices: np.ndarray,
    *,
    device: torch.device,
) -> dict[str, object]:
    predictions = predict(model, dataset, indices, device=device)
    truth = np.asarray(dataset.margins[indices], dtype=np.float64)
    return {
        "scalar": scalar_metrics(predictions, truth),
        "ranking": ranking_metrics(
            predictions,
            truth,
            np.asarray(dataset.state_indices[indices], dtype=np.int64),
        ),
    }


def evaluate_full_gold(
    model: AdjutantValueMlp,
    dataset: LoadedAdjutantValueDataset,
    *,
    device: torch.device,
) -> dict[str, object]:
    indices = np.arange(dataset.sample_count, dtype=np.int64)
    predictions = predict(model, dataset, indices, device=device)
    truth = np.asarray(dataset.margins, dtype=np.float64)
    state_indices = np.asarray(dataset.state_indices, dtype=np.int64)
    ranking = ranking_metrics(predictions, truth, state_indices)
    source_diagnostics = dataset.manifest.get("sourceDiagnostics", [])
    decomposition = full_gold_decomposition(predictions, truth, source_diagnostics)
    proposal = proposal_gold_metrics(source_diagnostics)
    return {
        "scalar": scalar_metrics(predictions, truth),
        "ranking": ranking,
        "proposal": proposal,
        "decomposition": decomposition,
    }


def predict(
    model: AdjutantValueMlp,
    dataset: LoadedAdjutantValueDataset,
    indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    model.eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, int(indices.shape[0]), batch_size):
            rows = indices[start : start + batch_size]
            features = torch.from_numpy(
                np.array(dataset.features[rows], dtype=np.float32, copy=True)
            ).to(device)
            batches.append(model(features).detach().cpu().numpy())
    return np.concatenate(batches).astype(np.float64)


def _run_epoch(
    model: AdjutantValueMlp,
    loader: DataLoader[tuple[Tensor, Tensor]],
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    huber_delta: float,
) -> float:
    model.train(optimizer is not None)
    total_loss = 0.0
    total_samples = 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for features, target in loader:
            features = features.to(device=device, dtype=torch.float32)
            target = target.to(device=device, dtype=torch.float32)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(features)
            loss = torch.nn.functional.huber_loss(prediction, target, delta=huber_delta)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            batch_size = int(target.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
    if total_samples == 0:
        raise ValueError("empty split.")
    return total_loss / total_samples


def scalar_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = prediction - truth
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "bias": float(np.mean(error)),
        "pearson": pearson(prediction, truth),
    }


def ranking_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    state_indices: np.ndarray,
) -> dict[str, object]:
    exact = 0
    top3 = 0
    top5 = 0
    regrets: list[float] = []
    pairwise_correct = 0
    pairwise_total = 0
    for state in sorted(set(int(value) for value in state_indices)):
        rows = np.nonzero(state_indices == state)[0]
        if rows.shape[0] != ADJUTANT_CANDIDATE_COUNT:
            continue
        state_truth = truth[rows]
        state_prediction = prediction[rows]
        gold_order = np.argsort(-state_truth)
        model_order = np.argsort(-state_prediction)
        selected = int(model_order[0])
        gold = int(gold_order[0])
        exact += selected == gold
        top3 += selected in set(int(value) for value in gold_order[:3])
        top5 += selected in set(int(value) for value in gold_order[:5])
        regrets.append(float(state_truth[gold] - state_truth[selected]))
        for left in range(ADJUTANT_CANDIDATE_COUNT):
            for right in range(left + 1, ADJUTANT_CANDIDATE_COUNT):
                truth_delta = state_truth[left] - state_truth[right]
                if truth_delta == 0:
                    continue
                pred_delta = state_prediction[left] - state_prediction[right]
                pairwise_correct += (truth_delta > 0) == (pred_delta > 0)
                pairwise_total += 1
    state_count = len(regrets)
    return {
        "stateCount": state_count,
        "exact": 0.0 if state_count == 0 else exact / state_count,
        "top3": 0.0 if state_count == 0 else top3 / state_count,
        "top5": 0.0 if state_count == 0 else top5 / state_count,
        "selectedRegret": summarize(regrets),
        "pairwise": 0.0 if pairwise_total == 0 else pairwise_correct / pairwise_total,
    }


def full_gold_decomposition(
    prediction: np.ndarray,
    truth: np.ndarray,
    source_diagnostics: Any,
) -> dict[str, object]:
    rb_rb: list[float] = []
    rb_opt: list[float] = []
    model_opt: list[float] = []
    gold_opt: list[float] = []
    rb_regret: list[float] = []
    for state_index, diagnostic in enumerate(source_diagnostics):
        start = state_index * ADJUTANT_CANDIDATE_COUNT
        stop = start + ADJUTANT_CANDIDATE_COUNT
        state_truth = truth[start:stop]
        state_prediction = prediction[start:stop]
        if state_truth.shape[0] != ADJUTANT_CANDIDATE_COUNT:
            continue
        rb_index = int(diagnostic["ruleBasedAdjutantIndex"])
        model_index = int(np.argmax(state_prediction))
        gold_index = int(np.argmax(state_truth))
        rb_rb.append(float(diagnostic["rbAdjRbExchangeMargin"]))
        rb_opt.append(float(state_truth[rb_index]))
        model_opt.append(float(state_truth[model_index]))
        gold_opt.append(float(state_truth[gold_index]))
        rb_regret.append(float(state_truth[gold_index] - state_truth[rb_index]))
    return {
        "rbAdjRbExchange": summarize(rb_rb),
        "rbAdjOptimizedExchange": summarize(rb_opt),
        "modelAdjOptimizedExchange": summarize(model_opt),
        "goldAdjGoldExchange": summarize(gold_opt),
        "ruleBasedAdjutantRegret": summarize(rb_regret),
    }


def proposal_gold_metrics(source_diagnostics: Any) -> dict[str, object]:
    totals = {
        "top4": 0,
        "top8": 0,
        "top16": 0,
        "top32": 0,
        "top64": 0,
        "top16PlusRuleBased": 0,
        "fullProposal": 0,
        "ruleBasedExchange": 0,
    }
    regret_top16_sum = 0.0
    regret_top16_plus_rb_sum = 0.0
    regret_full_proposal_sum = 0.0
    count = 0
    for diagnostic in source_diagnostics:
        containment = diagnostic.get("proposalGoldContainment", {})
        for key in totals:
            totals[key] += int(containment.get(key, 0))
        regret_top16_sum += float(diagnostic.get("proposalBestRegretTop16Sum", 0.0))
        regret_top16_plus_rb_sum += float(
            diagnostic.get("proposalBestRegretTop16PlusRuleBasedSum", 0.0)
        )
        regret_full_proposal_sum += float(diagnostic.get("proposalBestRegretSum", 0.0))
        count += ADJUTANT_CANDIDATE_COUNT
    return {
        "candidateCount": count,
        "topKContainment": {
            key: 0.0 if count == 0 else value / count for key, value in totals.items()
        },
        "regretMean": {
            "top16": 0.0 if count == 0 else regret_top16_sum / count,
            "top16PlusRuleBased": 0.0 if count == 0 else regret_top16_plus_rb_sum / count,
            "fullProposalTop16RuleBasedDiversity": (
                0.0 if count == 0 else regret_full_proposal_sum / count
            ),
        },
        "ruleBasedAdditiveEffect": {
            "containmentDelta": 0.0
            if count == 0
            else (totals["top16PlusRuleBased"] - totals["top16"]) / count,
            "regretReduction": 0.0
            if count == 0
            else (regret_top16_sum - regret_top16_plus_rb_sum) / count,
        },
        "diversityRandomAdditiveEffect": {
            "containmentDelta": 0.0
            if count == 0
            else (totals["fullProposal"] - totals["top16PlusRuleBased"]) / count,
            "regretReduction": 0.0
            if count == 0
            else (regret_top16_plus_rb_sum - regret_full_proposal_sum) / count,
        },
    }


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0:
        return 0.0
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denom = float(
        np.sqrt(
            np.sum(left_centered * left_centered) * np.sum(right_centered * right_centered)
        )
    )
    if denom == 0.0:
        return 0.0
    return float(np.sum(left_centered * right_centered) / denom)


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.shape[0]),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
