"""ONNX export and runtime parity checks for non-playing policy checkpoints."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal, NamedTuple, cast

import numpy as np
import torch
from torch import Tensor, nn

from napoleon_ml.adjutant.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION as ADJUTANT_CHECKPOINT_SCHEMA_VERSION,
)
from napoleon_ml.adjutant.checkpoint import load_adjutant_checkpoint
from napoleon_ml.adjutant.metrics import select_adjutant_action
from napoleon_ml.adjutant.ppo import (
    ADJUTANT_ACTOR_CRITIC_MODEL_ARCHITECTURE,
    ADJUTANT_PPO_ALGORITHM,
    ADJUTANT_PPO_CHECKPOINT_SCHEMA_VERSION,
    iter_non_playing_adjutant_rl_samples,
    load_adjutant_logits_checkpoint,
)
from napoleon_ml.adjutant.ppo import (
    NON_PLAYING_RL_SAMPLE_TYPE as ADJUTANT_PPO_SAMPLE_TYPE,
)
from napoleon_ml.bidding.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION as BIDDING_CHECKPOINT_SCHEMA_VERSION,
)
from napoleon_ml.bidding.checkpoint import load_bidding_checkpoint
from napoleon_ml.bidding.metrics import select_bidding_action
from napoleon_ml.bidding.ppo import (
    BIDDING_ACTOR_CRITIC_MODEL_ARCHITECTURE,
    BIDDING_PPO_ALGORITHM,
    BIDDING_PPO_CHECKPOINT_SCHEMA_VERSION,
    NON_PLAYING_RL_SAMPLE_TYPE,
    iter_non_playing_bidding_rl_samples,
    load_bidding_logits_checkpoint,
    load_bidding_ppo_checkpoint,
)
from napoleon_ml.dataset.constants import (
    ADJUTANT_DATASET_SAMPLE_TYPE,
    ADJUTANT_ENCODER_SCHEMA_VERSION,
    BIDDING_ACTION_COUNT,
    BIDDING_DATASET_SAMPLE_TYPE,
    BIDDING_ENCODER_SCHEMA_VERSION,
    CARD_COUNT,
    EXCHANGE_DATASET_SAMPLE_TYPE,
    EXCHANGE_ENCODER_SCHEMA_VERSION,
    MULTIPHASE_DATASET_SCHEMA_VERSION,
)
from napoleon_ml.dataset.manifest import DatasetManifest
from napoleon_ml.dataset.reader import iter_tensorized_samples
from napoleon_ml.dataset.tensors import (
    ADJUTANT_MODEL_INPUT_FEATURE_COUNT,
    ADJUTANT_MODEL_INPUT_SCHEMA_VERSION,
    BIDDING_MODEL_INPUT_FEATURE_COUNT,
    BIDDING_MODEL_INPUT_SCHEMA_VERSION,
    EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
    EXCHANGE_MODEL_INPUT_SCHEMA_VERSION,
    TensorizedAdjutantSample,
    TensorizedBiddingSample,
    TensorizedExchangeSample,
)
from napoleon_ml.dataset.validation import calculate_card_ids_sha256
from napoleon_ml.exchange.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION as EXCHANGE_CHECKPOINT_SCHEMA_VERSION,
)
from napoleon_ml.exchange.checkpoint import load_exchange_checkpoint
from napoleon_ml.exchange.metrics import DISCARD_COUNT, select_exchange_discards
from napoleon_ml.exchange.ppo import (
    EXCHANGE_ACTOR_CRITIC_MODEL_ARCHITECTURE,
    EXCHANGE_DECISION_MODE,
    EXCHANGE_PPO_ALGORITHM,
    EXCHANGE_PPO_CHECKPOINT_SCHEMA_VERSION,
    iter_non_playing_exchange_rl_samples,
    load_exchange_logits_checkpoint,
    load_exchange_ppo_checkpoint,
)
from napoleon_ml.exchange.ppo import (
    NON_PLAYING_RL_SAMPLE_TYPE as EXCHANGE_PPO_SAMPLE_TYPE,
)

PolicyType = Literal["bidding", "exchange", "adjutant"]

ONNX_INPUT_NAME = "model_input"
ONNX_OUTPUT_NAME = "logits"
ONNX_OPSET_VERSION = 18
NONPLAYING_ONNX_METADATA_SCHEMA_VERSION = 1
NONPLAYING_POLICY_CHECKPOINT_SCHEMA_VERSION = 1
_FLOAT32_DTYPE = "float32"
_BATCH_DIMENSION = "batch"
_PARITY_RTOL = 1e-4
_PARITY_ATOL = 1e-5


class NonPlayingOnnxExportError(ValueError):
    """Raised when a non-playing checkpoint cannot be exported safely."""


@dataclass(frozen=True)
class NonPlayingOnnxExportReport:
    policy_type: PolicyType
    onnx_path: Path
    metadata_path: Path
    sample_seed: int
    sample_step: int
    max_abs_logit_diff: float
    pytorch_selection: int | tuple[int, ...]
    onnx_selection: int | tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "policyType": self.policy_type,
            "onnxPath": str(self.onnx_path),
            "metadataPath": str(self.metadata_path),
            "paritySample": {
                "seed": self.sample_seed,
                "step": self.sample_step,
                "maxAbsLogitDiff": self.max_abs_logit_diff,
                "pytorchSelection": self.pytorch_selection,
                "onnxSelection": self.onnx_selection,
                "selectionParity": self.pytorch_selection == self.onnx_selection,
            },
        }


class _OnnxParityResult(NamedTuple):
    max_abs_logit_diff: float
    pytorch_selection: int | tuple[int, ...]
    onnx_selection: int | tuple[int, ...]


@dataclass(frozen=True)
class _NonPlayingPolicySpec:
    policy_type: PolicyType
    artifact_type: str
    sample_type: str
    checkpoint_schema_version: int
    encoder_schema_version: int
    model_input_schema_version: int
    model_input_feature_count: int
    output_count: int
    encoder_checkpoint_key: str
    model_input_checkpoint_key: str
    output_count_checkpoint_key: str
    load_checkpoint: Callable[[Path | str, DatasetManifest], tuple[nn.Module, dict[str, object]]]
    sample_class: type[Any]
    legal_mask_attr: str
    selection_kind: Literal["single", "top3"]
    discard_count: int | None = None


def _load_bidding(
    path: Path | str, manifest: DatasetManifest
) -> tuple[nn.Module, dict[str, object]]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and checkpoint.get("model_architecture") == (
        BIDDING_ACTOR_CRITIC_MODEL_ARCHITECTURE
    ):
        return load_bidding_ppo_checkpoint(path)
    return load_bidding_checkpoint(path, manifest=manifest)


def _load_exchange(
    path: Path | str, manifest: DatasetManifest
) -> tuple[nn.Module, dict[str, object]]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and checkpoint.get("model_architecture") == (
        EXCHANGE_ACTOR_CRITIC_MODEL_ARCHITECTURE
    ):
        return load_exchange_ppo_checkpoint(path)
    return load_exchange_checkpoint(path, manifest=manifest)


def _load_adjutant(
    path: Path | str, manifest: DatasetManifest
) -> tuple[nn.Module, dict[str, object]]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and checkpoint.get("model_architecture") == (
        ADJUTANT_ACTOR_CRITIC_MODEL_ARCHITECTURE
    ):
        return load_adjutant_logits_checkpoint(path)
    return load_adjutant_checkpoint(path, manifest=manifest)


_SPECS: dict[PolicyType, _NonPlayingPolicySpec] = {
    "bidding": _NonPlayingPolicySpec(
        policy_type="bidding",
        artifact_type="napoleon-bidding-policy-onnx",
        sample_type=BIDDING_DATASET_SAMPLE_TYPE,
        checkpoint_schema_version=BIDDING_CHECKPOINT_SCHEMA_VERSION,
        encoder_schema_version=BIDDING_ENCODER_SCHEMA_VERSION,
        model_input_schema_version=BIDDING_MODEL_INPUT_SCHEMA_VERSION,
        model_input_feature_count=BIDDING_MODEL_INPUT_FEATURE_COUNT,
        output_count=BIDDING_ACTION_COUNT,
        encoder_checkpoint_key="bidding_encoder_schema_version",
        model_input_checkpoint_key="model_input_schema_version",
        output_count_checkpoint_key="action_count",
        load_checkpoint=_load_bidding,
        sample_class=TensorizedBiddingSample,
        legal_mask_attr="legal_bid_mask",
        selection_kind="single",
    ),
    "exchange": _NonPlayingPolicySpec(
        policy_type="exchange",
        artifact_type="napoleon-exchange-policy-onnx",
        sample_type=EXCHANGE_DATASET_SAMPLE_TYPE,
        checkpoint_schema_version=NONPLAYING_POLICY_CHECKPOINT_SCHEMA_VERSION,
        encoder_schema_version=EXCHANGE_ENCODER_SCHEMA_VERSION,
        model_input_schema_version=EXCHANGE_MODEL_INPUT_SCHEMA_VERSION,
        model_input_feature_count=EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
        output_count=CARD_COUNT,
        encoder_checkpoint_key="encoder_schema_version",
        model_input_checkpoint_key="exchange_model_input_schema_version",
        output_count_checkpoint_key="action_count",
        load_checkpoint=_load_exchange,
        sample_class=TensorizedExchangeSample,
        legal_mask_attr="legal_discard_card_mask",
        selection_kind="top3",
        discard_count=DISCARD_COUNT,
    ),
    "adjutant": _NonPlayingPolicySpec(
        policy_type="adjutant",
        artifact_type="napoleon-adjutant-policy-onnx",
        sample_type=ADJUTANT_DATASET_SAMPLE_TYPE,
        checkpoint_schema_version=ADJUTANT_CHECKPOINT_SCHEMA_VERSION,
        encoder_schema_version=ADJUTANT_ENCODER_SCHEMA_VERSION,
        model_input_schema_version=ADJUTANT_MODEL_INPUT_SCHEMA_VERSION,
        model_input_feature_count=ADJUTANT_MODEL_INPUT_FEATURE_COUNT,
        output_count=CARD_COUNT,
        encoder_checkpoint_key="adjutant_encoder_schema_version",
        model_input_checkpoint_key="adjutant_model_input_schema_version",
        output_count_checkpoint_key="action_count",
        load_checkpoint=_load_adjutant,
        sample_class=TensorizedAdjutantSample,
        legal_mask_attr="legal_adjutant_mask",
        selection_kind="single",
    ),
}


def export_nonplaying_checkpoint_to_onnx(
    *,
    policy_type: PolicyType,
    dataset_directory: Path | str,
    checkpoint_path: Path | str,
    onnx_path: Path | str,
    metadata_path: Path | str,
    manifest: DatasetManifest,
    verify_integrity: bool = True,
) -> NonPlayingOnnxExportReport:
    spec = _SPECS[policy_type]
    checkpoint_file = Path(checkpoint_path)
    output = Path(onnx_path)
    metadata_output = Path(metadata_path)
    _validate_export_paths(
        checkpoint_path=checkpoint_file,
        onnx_path=output,
        metadata_path=metadata_output,
    )
    _validate_manifest_for_export(spec, manifest)

    model, checkpoint = spec.load_checkpoint(checkpoint_file, manifest)
    _validate_checkpoint_metadata_for_export(spec, checkpoint, model=model)
    _validate_model_for_export(spec, model)

    samples = iter_tensorized_samples(dataset_directory, verify_integrity=verify_integrity)
    sample = next(samples, None)
    if sample is None:
        raise NonPlayingOnnxExportError(
            f"{spec.policy_type} dataset contains no samples for ONNX parity check."
        )
    if not isinstance(sample, spec.sample_class):
        raise NonPlayingOnnxExportError(
            f"{spec.policy_type} ONNX export requires a {spec.sample_type} dataset."
        )
    if verify_integrity:
        for _ in samples:
            pass

    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.zeros((2, spec.model_input_feature_count), dtype=torch.float32)
    staged_output = _temporary_sibling(output)
    staged_metadata = _temporary_sibling(metadata_output)
    try:
        with torch.no_grad():
            _export_onnx(model=model, dummy_input=dummy_input, output=staged_output)

        metadata = build_nonplaying_onnx_metadata(
            policy_type=policy_type,
            model=model,
            checkpoint=checkpoint,
        )
        parity = _check_onnx_runtime_parity(
            spec=spec,
            model=model,
            onnx_path=staged_output,
            sample=sample,
            metadata=metadata,
        )
        staged_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staged_output.replace(output)
        staged_metadata.replace(metadata_output)
    except Exception:
        staged_output.unlink(missing_ok=True)
        staged_metadata.unlink(missing_ok=True)
        raise

    return NonPlayingOnnxExportReport(
        policy_type=policy_type,
        onnx_path=output,
        metadata_path=metadata_output,
        sample_seed=sample.seed,
        sample_step=sample.step,
        max_abs_logit_diff=parity.max_abs_logit_diff,
        pytorch_selection=parity.pytorch_selection,
        onnx_selection=parity.onnx_selection,
    )


def export_bidding_rl_checkpoint_to_onnx(
    *,
    dataset_directory: Path | str,
    checkpoint_path: Path | str,
    onnx_path: Path | str,
    metadata_path: Path | str,
) -> NonPlayingOnnxExportReport:
    """Export a bidding PPO checkpoint using a #206 non-playing RL dataset sample."""

    spec = _SPECS["bidding"]
    checkpoint_file = Path(checkpoint_path)
    output = Path(onnx_path)
    metadata_output = Path(metadata_path)
    _validate_export_paths(
        checkpoint_path=checkpoint_file,
        onnx_path=output,
        metadata_path=metadata_output,
    )
    model, checkpoint = load_bidding_logits_checkpoint(checkpoint_file)
    _validate_checkpoint_metadata_for_export(spec, checkpoint, model=model)
    _validate_model_for_export(spec, model)

    sample = next(iter_non_playing_bidding_rl_samples(dataset_directory), None)
    if sample is None:
        raise NonPlayingOnnxExportError("bidding RL dataset contains no samples.")

    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.zeros((2, spec.model_input_feature_count), dtype=torch.float32)
    staged_output = _temporary_sibling(output)
    staged_metadata = _temporary_sibling(metadata_output)
    try:
        with torch.no_grad():
            _export_onnx(model=model, dummy_input=dummy_input, output=staged_output)
        metadata = build_nonplaying_onnx_metadata(
            policy_type="bidding",
            model=model,
            checkpoint=checkpoint,
        )
        parity = _check_onnx_runtime_parity(
            spec=spec,
            model=model,
            onnx_path=staged_output,
            sample=sample,
            metadata=metadata,
        )
        staged_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staged_output.replace(output)
        staged_metadata.replace(metadata_output)
    except Exception:
        staged_output.unlink(missing_ok=True)
        staged_metadata.unlink(missing_ok=True)
        raise

    return NonPlayingOnnxExportReport(
        policy_type="bidding",
        onnx_path=output,
        metadata_path=metadata_output,
        sample_seed=sample.seed,
        sample_step=sample.step,
        max_abs_logit_diff=parity.max_abs_logit_diff,
        pytorch_selection=parity.pytorch_selection,
        onnx_selection=parity.onnx_selection,
    )


def export_adjutant_rl_checkpoint_to_onnx(
    *,
    dataset_directory: Path | str,
    checkpoint_path: Path | str,
    onnx_path: Path | str,
    metadata_path: Path | str,
) -> NonPlayingOnnxExportReport:
    """Export an adjutant PPO checkpoint using a non-playing RL dataset sample."""

    spec = _SPECS["adjutant"]
    checkpoint_file = Path(checkpoint_path)
    output = Path(onnx_path)
    metadata_output = Path(metadata_path)
    _validate_export_paths(
        checkpoint_path=checkpoint_file,
        onnx_path=output,
        metadata_path=metadata_output,
    )
    model, checkpoint = load_adjutant_logits_checkpoint(checkpoint_file)
    _validate_checkpoint_metadata_for_export(spec, checkpoint, model=model)
    _validate_model_for_export(spec, model)

    sample = next(iter_non_playing_adjutant_rl_samples(dataset_directory), None)
    if sample is None:
        raise NonPlayingOnnxExportError("adjutant RL dataset contains no samples.")

    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.zeros((2, spec.model_input_feature_count), dtype=torch.float32)
    staged_output = _temporary_sibling(output)
    staged_metadata = _temporary_sibling(metadata_output)
    try:
        with torch.no_grad():
            _export_onnx(model=model, dummy_input=dummy_input, output=staged_output)
        metadata = build_nonplaying_onnx_metadata(
            policy_type="adjutant",
            model=model,
            checkpoint=checkpoint,
        )
        parity = _check_onnx_runtime_parity(
            spec=spec,
            model=model,
            onnx_path=staged_output,
            sample=sample,
            metadata=metadata,
        )
        staged_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staged_output.replace(output)
        staged_metadata.replace(metadata_output)
    except Exception:
        staged_output.unlink(missing_ok=True)
        staged_metadata.unlink(missing_ok=True)
        raise

    return NonPlayingOnnxExportReport(
        policy_type="adjutant",
        onnx_path=output,
        metadata_path=metadata_output,
        sample_seed=sample.seed,
        sample_step=sample.step,
        max_abs_logit_diff=parity.max_abs_logit_diff,
        pytorch_selection=parity.pytorch_selection,
        onnx_selection=parity.onnx_selection,
    )


def export_exchange_rl_checkpoint_to_onnx(
    *,
    dataset_directory: Path | str,
    checkpoint_path: Path | str,
    onnx_path: Path | str,
    metadata_path: Path | str,
) -> NonPlayingOnnxExportReport:
    """Export a sequential-card exchange PPO checkpoint using a non-playing RL sample."""

    spec = _SPECS["exchange"]
    checkpoint_file = Path(checkpoint_path)
    output = Path(onnx_path)
    metadata_output = Path(metadata_path)
    _validate_export_paths(
        checkpoint_path=checkpoint_file,
        onnx_path=output,
        metadata_path=metadata_output,
    )
    model, checkpoint = load_exchange_logits_checkpoint(checkpoint_file)
    _validate_checkpoint_metadata_for_export(spec, checkpoint, model=model)
    _validate_model_for_export(spec, model)

    sample = next(iter_non_playing_exchange_rl_samples(dataset_directory), None)
    if sample is None:
        raise NonPlayingOnnxExportError("exchange RL dataset contains no samples.")

    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.zeros((2, spec.model_input_feature_count), dtype=torch.float32)
    staged_output = _temporary_sibling(output)
    staged_metadata = _temporary_sibling(metadata_output)
    try:
        with torch.no_grad():
            _export_onnx(model=model, dummy_input=dummy_input, output=staged_output)
        metadata = build_nonplaying_onnx_metadata(
            policy_type="exchange",
            model=model,
            checkpoint=checkpoint,
        )
        parity = _check_onnx_runtime_parity(
            spec=spec,
            model=model,
            onnx_path=staged_output,
            sample=sample,
            metadata=metadata,
        )
        staged_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staged_output.replace(output)
        staged_metadata.replace(metadata_output)
    except Exception:
        staged_output.unlink(missing_ok=True)
        staged_metadata.unlink(missing_ok=True)
        raise

    return NonPlayingOnnxExportReport(
        policy_type="exchange",
        onnx_path=output,
        metadata_path=metadata_output,
        sample_seed=sample.seed,
        sample_step=sample.step,
        max_abs_logit_diff=parity.max_abs_logit_diff,
        pytorch_selection=parity.pytorch_selection,
        onnx_selection=parity.onnx_selection,
    )


def build_nonplaying_onnx_metadata(
    *,
    policy_type: PolicyType,
    model: nn.Module,
    checkpoint: dict[str, object],
) -> dict[str, object]:
    spec = _SPECS[policy_type]
    _validate_checkpoint_metadata_for_export(spec, checkpoint, model=model)
    model_config = _model_config_dict(model)
    compatibility_metadata = _checkpoint_compatibility_metadata(
        spec,
        checkpoint=checkpoint,
        model_config=model_config,
    )
    metadata: dict[str, object] = {
        "metadataSchemaVersion": NONPLAYING_ONNX_METADATA_SCHEMA_VERSION,
        "artifactType": spec.artifact_type,
        "policyType": spec.policy_type,
        "checkpointSchemaVersion": spec.checkpoint_schema_version,
        "datasetSchemaVersion": MULTIPHASE_DATASET_SCHEMA_VERSION,
        "encoderSchemaVersion": spec.encoder_schema_version,
        "modelInputSchemaVersion": spec.model_input_schema_version,
        "modelInputFeatureCount": spec.model_input_feature_count,
        "outputLogitCount": spec.output_count,
        "actionCount": spec.output_count,
        "cardIdsSha256": calculate_card_ids_sha256(),
        "inputName": ONNX_INPUT_NAME,
        "outputName": ONNX_OUTPUT_NAME,
        "inputShape": [_BATCH_DIMENSION, spec.model_input_feature_count],
        "outputShape": [_BATCH_DIMENSION, spec.output_count],
        "inputDtype": _FLOAT32_DTYPE,
        "outputDtype": _FLOAT32_DTYPE,
        "onnx": {
            "opsetVersion": ONNX_OPSET_VERSION,
            "inputs": [
                {
                    "name": ONNX_INPUT_NAME,
                    "shape": [_BATCH_DIMENSION, spec.model_input_feature_count],
                    "dtype": _FLOAT32_DTYPE,
                }
            ],
            "outputs": [
                {
                    "name": ONNX_OUTPUT_NAME,
                    "shape": [_BATCH_DIMENSION, spec.output_count],
                    "dtype": _FLOAT32_DTYPE,
                }
            ],
        },
        "modelConfig": model_config,
        "checkpointSeed": checkpoint["seed"],
        "checkpointCompatibilityMetadata": compatibility_metadata,
    }
    if spec.discard_count is not None:
        metadata["discardCount"] = spec.discard_count
    if spec.policy_type == "exchange":
        metadata["decisionMode"] = checkpoint.get("decision_mode", "top3-set-v1")
    return metadata


def validate_nonplaying_onnx_metadata(metadata: dict[str, Any]) -> None:
    policy_type = metadata.get("policyType")
    if policy_type not in _SPECS:
        raise NonPlayingOnnxExportError(f"metadata policyType is unsupported: {policy_type!r}.")

    spec = _SPECS[cast(PolicyType, policy_type)]
    expected: dict[str, object] = {
        "metadataSchemaVersion": NONPLAYING_ONNX_METADATA_SCHEMA_VERSION,
        "artifactType": spec.artifact_type,
        "policyType": spec.policy_type,
        "checkpointSchemaVersion": spec.checkpoint_schema_version,
        "datasetSchemaVersion": MULTIPHASE_DATASET_SCHEMA_VERSION,
        "encoderSchemaVersion": spec.encoder_schema_version,
        "modelInputSchemaVersion": spec.model_input_schema_version,
        "modelInputFeatureCount": spec.model_input_feature_count,
        "outputLogitCount": spec.output_count,
        "actionCount": spec.output_count,
        "cardIdsSha256": calculate_card_ids_sha256(),
        "inputName": ONNX_INPUT_NAME,
        "outputName": ONNX_OUTPUT_NAME,
        "inputShape": [_BATCH_DIMENSION, spec.model_input_feature_count],
        "outputShape": [_BATCH_DIMENSION, spec.output_count],
        "inputDtype": _FLOAT32_DTYPE,
        "outputDtype": _FLOAT32_DTYPE,
    }
    if spec.discard_count is not None:
        expected["discardCount"] = spec.discard_count
    if spec.policy_type == "exchange":
        decision_mode = metadata.get("decisionMode", "top3-set-v1")
        if decision_mode not in ("top3-set-v1", EXCHANGE_DECISION_MODE):
            raise NonPlayingOnnxExportError(
                f"metadata decisionMode is unsupported: {decision_mode!r}."
            )

    for key, value in expected.items():
        actual = metadata.get(key)
        if actual != value:
            raise NonPlayingOnnxExportError(
                f"metadata {key} mismatch: expected {value!r}, got {actual!r}."
            )

    onnx = metadata.get("onnx")
    if not isinstance(onnx, dict):
        raise NonPlayingOnnxExportError("metadata onnx must be a dictionary.")
    if onnx.get("opsetVersion") != ONNX_OPSET_VERSION:
        raise NonPlayingOnnxExportError(
            "metadata onnx opsetVersion mismatch: "
            f"expected {ONNX_OPSET_VERSION!r}, got {onnx.get('opsetVersion')!r}."
        )
    _validate_io_metadata(
        onnx.get("inputs"),
        expected_name=ONNX_INPUT_NAME,
        expected_shape=(_BATCH_DIMENSION, spec.model_input_feature_count),
        expected_dtype=_FLOAT32_DTYPE,
        label="input",
    )
    _validate_io_metadata(
        onnx.get("outputs"),
        expected_name=ONNX_OUTPUT_NAME,
        expected_shape=(_BATCH_DIMENSION, spec.output_count),
        expected_dtype=_FLOAT32_DTYPE,
        label="output",
    )


def _temporary_sibling(path: Path) -> Path:
    with NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as file:
        return Path(file.name)


def _validate_export_paths(*, checkpoint_path: Path, onnx_path: Path, metadata_path: Path) -> None:
    resolved_checkpoint = checkpoint_path.resolve()
    resolved_onnx = onnx_path.resolve()
    resolved_metadata = metadata_path.resolve()

    if resolved_onnx == resolved_metadata:
        raise NonPlayingOnnxExportError(
            "ONNX output and metadata output must be different paths."
        )
    if _is_nested_path(resolved_onnx, resolved_metadata):
        raise NonPlayingOnnxExportError(
            "ONNX output and metadata output must not be nested under each other."
        )
    if resolved_onnx == resolved_checkpoint:
        raise NonPlayingOnnxExportError(
            "ONNX output and checkpoint input must be different paths."
        )
    if resolved_metadata == resolved_checkpoint:
        raise NonPlayingOnnxExportError(
            "metadata output and checkpoint input must be different paths."
        )
    _validate_writable_artifact_path(onnx_path, label="ONNX output")
    _validate_writable_artifact_path(metadata_path, label="metadata output")


def _is_nested_path(first: Path, second: Path) -> bool:
    return first != second and (first.is_relative_to(second) or second.is_relative_to(first))


def _validate_writable_artifact_path(path: Path, *, label: str) -> None:
    if path.exists() and not path.is_file():
        raise NonPlayingOnnxExportError(
            f"{label} must be a file path, got existing non-file path: {path}."
        )


def _validate_manifest_for_export(spec: _NonPlayingPolicySpec, manifest: DatasetManifest) -> None:
    expected_values = {
        "dataset_schema_version": MULTIPHASE_DATASET_SCHEMA_VERSION,
        "sample_type": spec.sample_type,
        "encoder_schema_version": spec.encoder_schema_version,
        "card_ids_sha256": calculate_card_ids_sha256(),
    }
    actual_values = {
        "dataset_schema_version": manifest.dataset_schema_version,
        "sample_type": manifest.sample_type,
        "encoder_schema_version": manifest.encoder_schema_version,
        "card_ids_sha256": manifest.card_ids_sha256,
    }
    for key, expected in expected_values.items():
        actual = actual_values[key]
        if actual != expected:
            raise NonPlayingOnnxExportError(
                f"{spec.policy_type} manifest {key} mismatch: "
                f"expected {expected!r}, got {actual!r}."
            )


def _validate_checkpoint_metadata_for_export(
    spec: _NonPlayingPolicySpec,
    checkpoint: dict[str, object],
    *,
    model: nn.Module,
) -> None:
    model_config = _model_config_dict(model)
    if (
        spec.policy_type == "bidding"
        and checkpoint.get("model_architecture") == BIDDING_ACTOR_CRITIC_MODEL_ARCHITECTURE
    ):
        _validate_bidding_ppo_checkpoint_for_export(checkpoint, model_config=model_config)
        return
    if (
        spec.policy_type == "adjutant"
        and checkpoint.get("model_architecture") == ADJUTANT_ACTOR_CRITIC_MODEL_ARCHITECTURE
    ):
        _validate_adjutant_ppo_checkpoint_for_export(checkpoint, model_config=model_config)
        return
    if (
        spec.policy_type == "exchange"
        and checkpoint.get("model_architecture") == EXCHANGE_ACTOR_CRITIC_MODEL_ARCHITECTURE
    ):
        _validate_exchange_ppo_checkpoint_for_export(checkpoint, model_config=model_config)
        return
    expected_values: dict[str, object] = {
        "checkpoint_schema_version": (
            EXCHANGE_CHECKPOINT_SCHEMA_VERSION
            if spec.policy_type == "exchange"
            else spec.checkpoint_schema_version
        ),
        "dataset_schema_version": MULTIPHASE_DATASET_SCHEMA_VERSION,
        "sample_type": spec.sample_type,
        spec.encoder_checkpoint_key: spec.encoder_schema_version,
        spec.model_input_checkpoint_key: spec.model_input_schema_version,
        spec.output_count_checkpoint_key: spec.output_count,
        "card_ids_sha256": calculate_card_ids_sha256(),
        "model_config": model_config,
    }
    if spec.discard_count is not None:
        expected_values["discard_count"] = spec.discard_count

    for key, expected in expected_values.items():
        actual = checkpoint.get(key)
        if actual != expected:
            raise NonPlayingOnnxExportError(
                f"{spec.policy_type} checkpoint {key} mismatch: "
                f"expected {expected!r}, got {actual!r}."
            )

    seed = checkpoint.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise NonPlayingOnnxExportError(
            f"{spec.policy_type} checkpoint seed must be an integer."
        )
    input_dim = model_config.get("input_dim")
    if input_dim != spec.model_input_feature_count:
        raise NonPlayingOnnxExportError(
            f"{spec.policy_type} checkpoint model_config.input_dim mismatch: "
            f"expected {spec.model_input_feature_count}, got {input_dim!r}."
        )


def _validate_bidding_ppo_checkpoint_for_export(
    checkpoint: dict[str, object],
    *,
    model_config: dict[str, object],
) -> None:
    expected_values: dict[str, object] = {
        "checkpoint_schema_version": BIDDING_PPO_CHECKPOINT_SCHEMA_VERSION,
        "model_architecture": BIDDING_ACTOR_CRITIC_MODEL_ARCHITECTURE,
        "algorithm": BIDDING_PPO_ALGORITHM,
        "sample_type": NON_PLAYING_RL_SAMPLE_TYPE,
        "bidding_encoder_schema_version": BIDDING_ENCODER_SCHEMA_VERSION,
        "model_input_schema_version": BIDDING_MODEL_INPUT_SCHEMA_VERSION,
        "model_input_feature_count": BIDDING_MODEL_INPUT_FEATURE_COUNT,
        "action_count": BIDDING_ACTION_COUNT,
        "card_ids_sha256": calculate_card_ids_sha256(),
        "model_config": model_config,
    }
    for key, expected in expected_values.items():
        actual = checkpoint.get(key)
        if actual != expected:
            raise NonPlayingOnnxExportError(
                f"bidding PPO checkpoint {key} mismatch: "
                f"expected {expected!r}, got {actual!r}."
            )
    seed = checkpoint.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise NonPlayingOnnxExportError("bidding PPO checkpoint seed must be an integer.")


def _validate_adjutant_ppo_checkpoint_for_export(
    checkpoint: dict[str, object],
    *,
    model_config: dict[str, object],
) -> None:
    expected_values: dict[str, object] = {
        "checkpoint_schema_version": ADJUTANT_PPO_CHECKPOINT_SCHEMA_VERSION,
        "model_architecture": ADJUTANT_ACTOR_CRITIC_MODEL_ARCHITECTURE,
        "algorithm": ADJUTANT_PPO_ALGORITHM,
        "sample_type": ADJUTANT_PPO_SAMPLE_TYPE,
        "adjutant_encoder_schema_version": ADJUTANT_ENCODER_SCHEMA_VERSION,
        "model_input_schema_version": ADJUTANT_MODEL_INPUT_SCHEMA_VERSION,
        "model_input_feature_count": ADJUTANT_MODEL_INPUT_FEATURE_COUNT,
        "action_count": CARD_COUNT,
        "card_ids_sha256": calculate_card_ids_sha256(),
        "model_config": model_config,
    }
    for key, expected in expected_values.items():
        actual = checkpoint.get(key)
        if actual != expected:
            raise NonPlayingOnnxExportError(
                f"adjutant PPO checkpoint {key} mismatch: "
                f"expected {expected!r}, got {actual!r}."
            )
    seed = checkpoint.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise NonPlayingOnnxExportError("adjutant PPO checkpoint seed must be an integer.")


def _validate_exchange_ppo_checkpoint_for_export(
    checkpoint: dict[str, object],
    *,
    model_config: dict[str, object],
) -> None:
    expected_values: dict[str, object] = {
        "checkpoint_schema_version": EXCHANGE_PPO_CHECKPOINT_SCHEMA_VERSION,
        "model_architecture": EXCHANGE_ACTOR_CRITIC_MODEL_ARCHITECTURE,
        "algorithm": EXCHANGE_PPO_ALGORITHM,
        "sample_type": EXCHANGE_PPO_SAMPLE_TYPE,
        "phase_scope": "exchange-only",
        "decision_mode": EXCHANGE_DECISION_MODE,
        "exchange_encoder_schema_version": EXCHANGE_ENCODER_SCHEMA_VERSION,
        "model_input_schema_version": EXCHANGE_MODEL_INPUT_SCHEMA_VERSION,
        "model_input_feature_count": EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
        "action_count": CARD_COUNT,
        "card_ids_sha256": calculate_card_ids_sha256(),
        "model_config": model_config,
    }
    for key, expected in expected_values.items():
        actual = checkpoint.get(key)
        if actual != expected:
            raise NonPlayingOnnxExportError(
                f"exchange PPO checkpoint {key} mismatch: "
                f"expected {expected!r}, got {actual!r}."
            )
    seed = checkpoint.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise NonPlayingOnnxExportError("exchange PPO checkpoint seed must be an integer.")


def _checkpoint_compatibility_metadata(
    spec: _NonPlayingPolicySpec,
    *,
    checkpoint: dict[str, object],
    model_config: dict[str, object],
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "checkpointSchemaVersion": spec.checkpoint_schema_version,
        "datasetSchemaVersion": checkpoint["dataset_schema_version"],
        "sampleType": checkpoint["sample_type"],
        "encoderSchemaVersion": checkpoint.get(
            spec.encoder_checkpoint_key,
            checkpoint.get("exchange_encoder_schema_version"),
        ),
        "modelInputSchemaVersion": checkpoint[spec.model_input_checkpoint_key],
        "modelInputFeatureCount": spec.model_input_feature_count,
        "outputCount": checkpoint[spec.output_count_checkpoint_key],
        "cardIdsSha256": checkpoint["card_ids_sha256"],
        "modelConfig": model_config,
        "seed": checkpoint["seed"],
    }
    if spec.discard_count is not None:
        metadata["discardCount"] = checkpoint.get("discard_count", spec.discard_count)
    if spec.policy_type == "exchange" and "decision_mode" in checkpoint:
        metadata["decisionMode"] = checkpoint["decision_mode"]
    return metadata


def _model_config_dict(model: nn.Module) -> dict[str, object]:
    config_owner = cast(Any, model)
    config = config_owner.config
    value = config.to_dict()
    if not isinstance(value, dict):
        raise NonPlayingOnnxExportError("model config must serialize to a dictionary.")
    return cast(dict[str, object], value)


def _validate_model_for_export(spec: _NonPlayingPolicySpec, model: nn.Module) -> None:
    with torch.no_grad():
        output = model(
            torch.zeros((2, spec.model_input_feature_count), dtype=torch.float32)
        )

    if tuple(output.shape) != (2, spec.output_count):
        raise NonPlayingOnnxExportError(
            f"{spec.policy_type} model must emit shape (batch, {spec.output_count}), "
            f"got {tuple(output.shape)}."
        )
    if output.dtype != torch.float32:
        raise NonPlayingOnnxExportError(
            f"{spec.policy_type} model must emit torch.float32 logits, got {output.dtype}."
        )


def _export_onnx(*, model: nn.Module, dummy_input: torch.Tensor, output: Path) -> None:
    with redirect_stdout(StringIO()):
        torch.onnx.export(
            model,
            (dummy_input,),
            output,
            input_names=[ONNX_INPUT_NAME],
            output_names=[ONNX_OUTPUT_NAME],
            dynamic_shapes={ONNX_INPUT_NAME: {0: torch.export.Dim(_BATCH_DIMENSION)}},
            opset_version=ONNX_OPSET_VERSION,
            dynamo=True,
            external_data=False,
        )


def _check_onnx_runtime_parity(
    *,
    spec: _NonPlayingPolicySpec,
    model: nn.Module,
    onnx_path: Path,
    sample: Any,
    metadata: dict[str, object],
) -> _OnnxParityResult:
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]
    except ImportError as error:
        raise NonPlayingOnnxExportError(
            "onnxruntime is required for non-playing ONNX parity checks."
        ) from error

    model_input_np = sample.model_input.astype(np.float32, copy=True).reshape(
        1, spec.model_input_feature_count
    )
    legal_mask_np = cast(np.ndarray, getattr(sample, spec.legal_mask_attr)).astype(
        np.bool_,
        copy=True,
    ).reshape(1, spec.output_count)
    with torch.no_grad():
        model_input_torch = torch.from_numpy(model_input_np)
        legal_mask_torch = torch.from_numpy(legal_mask_np)
        pytorch_logits = model(model_input_torch)
        pytorch_selection = _select(
            spec,
            pytorch_logits,
            legal_mask_torch,
            decision_mode=metadata.get("decisionMode"),
        )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    _validate_onnx_runtime_io(spec, session)
    validate_nonplaying_onnx_metadata(metadata)
    _validate_metadata_matches_onnx_runtime_io(session, metadata)

    outputs = session.run([ONNX_OUTPUT_NAME], {ONNX_INPUT_NAME: model_input_np})
    onnx_logits_np = cast(np.ndarray, outputs[0])
    if onnx_logits_np.shape != (1, spec.output_count):
        raise NonPlayingOnnxExportError(
            f"ONNX output must have shape (1, {spec.output_count}), got {onnx_logits_np.shape}."
        )
    if onnx_logits_np.dtype != np.float32:
        raise NonPlayingOnnxExportError(
            f"ONNX output must have dtype float32, got {onnx_logits_np.dtype}."
        )

    try:
        np.testing.assert_allclose(
            pytorch_logits.detach().numpy(),
            onnx_logits_np,
            rtol=_PARITY_RTOL,
            atol=_PARITY_ATOL,
        )
    except AssertionError as error:
        raise NonPlayingOnnxExportError(
            "PyTorch and ONNX Runtime logits differ beyond tolerance."
        ) from error

    onnx_selection = _select(
        spec,
        torch.from_numpy(onnx_logits_np),
        legal_mask_torch,
        decision_mode=metadata.get("decisionMode"),
    )
    pytorch_value = _selection_value(spec, pytorch_selection)
    onnx_value = _selection_value(spec, onnx_selection)
    if pytorch_value != onnx_value:
        raise NonPlayingOnnxExportError(
            "PyTorch and ONNX Runtime masked selections differ: "
            f"PyTorch={pytorch_value}, ONNX={onnx_value}."
        )

    _validate_selection_is_legal(spec, selection=onnx_selection, legal_mask=legal_mask_torch)

    return _OnnxParityResult(
        max_abs_logit_diff=float(
            np.max(np.abs(pytorch_logits.detach().numpy() - onnx_logits_np))
        ),
        pytorch_selection=pytorch_value,
        onnx_selection=onnx_value,
    )


def _select(
    spec: _NonPlayingPolicySpec,
    logits: Tensor,
    legal_mask: Tensor,
    *,
    decision_mode: object = None,
) -> Tensor:
    if spec.policy_type == "bidding":
        return select_bidding_action(logits, legal_mask)
    if spec.policy_type == "exchange":
        if decision_mode == EXCHANGE_DECISION_MODE:
            masked = logits.masked_fill(
                ~legal_mask.to(dtype=torch.bool),
                torch.finfo(logits.dtype).min,
            )
            return torch.argmax(masked, dim=1)
        return select_exchange_discards(logits, legal_mask)
    return select_adjutant_action(logits, legal_mask)


def _selection_value(spec: _NonPlayingPolicySpec, selection: Tensor) -> int | tuple[int, ...]:
    if selection.ndim == 2 and selection.shape[1] == DISCARD_COUNT:
        selected = tuple(sorted(int(index) for index in selection[0].tolist()))
        if len(selected) != DISCARD_COUNT or len(set(selected)) != DISCARD_COUNT:
            raise NonPlayingOnnxExportError(
                f"{spec.policy_type} selection must contain {DISCARD_COUNT} distinct cards."
            )
        return selected

    return int(selection.item())


def _validate_selection_is_legal(
    spec: _NonPlayingPolicySpec,
    *,
    selection: Tensor,
    legal_mask: Tensor,
) -> None:
    if selection.ndim == 2 and selection.shape[1] == DISCARD_COUNT:
        if selection.shape != (1, DISCARD_COUNT):
            raise NonPlayingOnnxExportError(
                f"{spec.policy_type} selection must have shape (1, {DISCARD_COUNT})."
            )
        if not bool(legal_mask.gather(1, selection).all().item()):
            raise NonPlayingOnnxExportError(f"{spec.policy_type} selected an illegal card.")
        return

    selected_index = int(selection.item())
    if not bool(legal_mask[0, selected_index].item()):
        raise NonPlayingOnnxExportError(f"{spec.policy_type} selected an illegal action.")


def _validate_onnx_runtime_io(spec: _NonPlayingPolicySpec, session: Any) -> None:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1:
        raise NonPlayingOnnxExportError(f"ONNX model must have one input, got {len(inputs)}.")
    if len(outputs) != 1:
        raise NonPlayingOnnxExportError(f"ONNX model must have one output, got {len(outputs)}.")

    input_meta = inputs[0]
    output_meta = outputs[0]
    if input_meta.name != ONNX_INPUT_NAME:
        raise NonPlayingOnnxExportError(
            f"ONNX input name mismatch: expected {ONNX_INPUT_NAME!r}, got {input_meta.name!r}."
        )
    if output_meta.name != ONNX_OUTPUT_NAME:
        raise NonPlayingOnnxExportError(
            f"ONNX output name mismatch: expected {ONNX_OUTPUT_NAME!r}, got {output_meta.name!r}."
        )
    if (
        len(input_meta.shape) != 2
        or not isinstance(input_meta.shape[0], str)
        or input_meta.shape[1] != spec.model_input_feature_count
    ):
        raise NonPlayingOnnxExportError(
            "ONNX input shape mismatch: expected dynamic batch and "
            f"{spec.model_input_feature_count} features, got {input_meta.shape!r}."
        )
    if (
        len(output_meta.shape) != 2
        or not isinstance(output_meta.shape[0], str)
        or output_meta.shape[1] != spec.output_count
    ):
        raise NonPlayingOnnxExportError(
            "ONNX output shape mismatch: expected dynamic batch and "
            f"{spec.output_count} logits, got {output_meta.shape!r}."
        )
    if input_meta.type != "tensor(float)":
        raise NonPlayingOnnxExportError(
            f"ONNX input dtype mismatch: expected tensor(float), got {input_meta.type}."
        )
    if output_meta.type != "tensor(float)":
        raise NonPlayingOnnxExportError(
            f"ONNX output dtype mismatch: expected tensor(float), got {output_meta.type}."
        )


def _validate_metadata_matches_onnx_runtime_io(session: Any, metadata: dict[str, object]) -> None:
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    if metadata["inputName"] != input_meta.name or metadata["outputName"] != output_meta.name:
        raise NonPlayingOnnxExportError("metadata input/output names do not match ONNX graph.")
    _validate_shape_matches_runtime(metadata["inputShape"], input_meta.shape, label="input")
    _validate_shape_matches_runtime(metadata["outputShape"], output_meta.shape, label="output")
    if metadata["inputDtype"] != _FLOAT32_DTYPE or input_meta.type != "tensor(float)":
        raise NonPlayingOnnxExportError("metadata input dtype does not match ONNX graph.")
    if metadata["outputDtype"] != _FLOAT32_DTYPE or output_meta.type != "tensor(float)":
        raise NonPlayingOnnxExportError("metadata output dtype does not match ONNX graph.")


def _validate_shape_matches_runtime(
    metadata_shape: object,
    runtime_shape: list[Any],
    *,
    label: str,
) -> None:
    if (
        not isinstance(metadata_shape, list)
        or len(metadata_shape) != 2
        or metadata_shape[0] != _BATCH_DIMENSION
        or not isinstance(runtime_shape[0], str)
        or metadata_shape[1] != runtime_shape[1]
    ):
        raise NonPlayingOnnxExportError(
            f"metadata {label} shape does not match ONNX graph: "
            f"metadata={metadata_shape!r}, graph={runtime_shape!r}."
        )


def _validate_io_metadata(
    value: object,
    *,
    expected_name: str,
    expected_shape: tuple[str, int],
    expected_dtype: str,
    label: str,
) -> None:
    if not isinstance(value, list) or len(value) != 1:
        raise NonPlayingOnnxExportError(f"metadata onnx {label}s must contain one item.")

    item = value[0]
    if not isinstance(item, dict):
        raise NonPlayingOnnxExportError(f"metadata onnx {label} item must be a dictionary.")

    expected_shape_list = [expected_shape[0], expected_shape[1]]
    if item.get("name") != expected_name:
        raise NonPlayingOnnxExportError(
            f"metadata onnx {label} name mismatch: "
            f"expected {expected_name!r}, got {item.get('name')!r}."
        )
    if item.get("shape") != expected_shape_list:
        raise NonPlayingOnnxExportError(
            "metadata onnx "
            f"{label} shape mismatch: expected {expected_shape_list!r}, got {item.get('shape')!r}."
        )
    if item.get("dtype") != expected_dtype:
        raise NonPlayingOnnxExportError(
            "metadata onnx "
            f"{label} dtype mismatch: expected {expected_dtype!r}, got {item.get('dtype')!r}."
        )
