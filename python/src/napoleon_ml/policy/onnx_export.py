"""ONNX export and runtime parity checks for policy checkpoints."""

from __future__ import annotations

import json
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import torch

from napoleon_ml.dataset.constants import (
    CARD_COUNT,
    DATASET_SCHEMA_VERSION,
    PLAYING_ENCODER_SCHEMA_VERSION,
)
from napoleon_ml.dataset.manifest import DatasetManifest
from napoleon_ml.dataset.reader import iter_tensorized_samples
from napoleon_ml.dataset.tensors import (
    MODEL_INPUT_FEATURE_COUNT,
    MODEL_INPUT_SCHEMA_VERSION,
    TensorizedPlayingSample,
)
from napoleon_ml.dataset.validation import calculate_card_ids_sha256
from napoleon_ml.policy.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    PolicyCheckpointCompatibilityError,
    load_policy_checkpoint,
)
from napoleon_ml.policy.metrics import select_policy_action
from napoleon_ml.policy.model import PolicyMlpModel

ONNX_INPUT_NAME = "model_input"
ONNX_OUTPUT_NAME = "logits"
ONNX_OPSET_VERSION = 18
POLICY_ONNX_METADATA_SCHEMA_VERSION = 1
_FLOAT32_DTYPE = "float32"
_BATCH_DIMENSION = "batch"
_PARITY_RTOL = 1e-4
_PARITY_ATOL = 1e-5


@dataclass(frozen=True)
class PolicyOnnxExportReport:
    onnx_path: Path
    metadata_path: Path
    sample_seed: int
    sample_step: int
    max_abs_logit_diff: float
    pytorch_selected_card_index: int
    onnx_selected_card_index: int

    def to_dict(self) -> dict[str, object]:
        return {
            "onnxPath": str(self.onnx_path),
            "metadataPath": str(self.metadata_path),
            "paritySample": {
                "seed": self.sample_seed,
                "step": self.sample_step,
                "maxAbsLogitDiff": self.max_abs_logit_diff,
                "pytorchSelectedCardIndex": self.pytorch_selected_card_index,
                "onnxSelectedCardIndex": self.onnx_selected_card_index,
            },
        }


class _OnnxParityResult(NamedTuple):
    max_abs_logit_diff: float
    pytorch_selected_card_index: int
    onnx_selected_card_index: int


def export_policy_checkpoint_to_onnx(
    *,
    dataset_directory: Path | str,
    checkpoint_path: Path | str,
    onnx_path: Path | str,
    metadata_path: Path | str,
    manifest: DatasetManifest,
    verify_integrity: bool = True,
) -> PolicyOnnxExportReport:
    model, checkpoint = load_policy_checkpoint(checkpoint_path, manifest=manifest)
    _validate_model_for_export(model)

    samples = iter_tensorized_samples(dataset_directory, verify_integrity=verify_integrity)
    sample = next(samples, None)
    if sample is None:
        raise PolicyCheckpointCompatibilityError(
            "dataset contains no samples for ONNX parity check."
        )
    if verify_integrity:
        for _ in samples:
            pass

    model.eval()
    output = Path(onnx_path)
    metadata_output = Path(metadata_path)
    if output.resolve() == metadata_output.resolve():
        raise PolicyCheckpointCompatibilityError(
            "ONNX output and metadata output must be different paths."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.zeros((1, MODEL_INPUT_FEATURE_COUNT), dtype=torch.float32)
    with torch.no_grad():
        _export_onnx(model=model, dummy_input=dummy_input, output=output)

    parity = _check_onnx_runtime_parity(model=model, onnx_path=output, sample=sample)
    metadata = build_policy_onnx_metadata(model=model, checkpoint=checkpoint)
    metadata_output.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return PolicyOnnxExportReport(
        onnx_path=output,
        metadata_path=metadata_output,
        sample_seed=sample.seed,
        sample_step=sample.step,
        max_abs_logit_diff=parity.max_abs_logit_diff,
        pytorch_selected_card_index=parity.pytorch_selected_card_index,
        onnx_selected_card_index=parity.onnx_selected_card_index,
    )


def build_policy_onnx_metadata(
    *,
    model: PolicyMlpModel,
    checkpoint: dict[str, object],
) -> dict[str, object]:
    _validate_model_for_export(model)
    _validate_checkpoint_metadata_for_export(checkpoint)

    return {
        "metadataSchemaVersion": POLICY_ONNX_METADATA_SCHEMA_VERSION,
        "checkpointSchemaVersion": CHECKPOINT_SCHEMA_VERSION,
        "datasetSchemaVersion": DATASET_SCHEMA_VERSION,
        "playingEncoderSchemaVersion": PLAYING_ENCODER_SCHEMA_VERSION,
        "modelInputSchemaVersion": MODEL_INPUT_SCHEMA_VERSION,
        "cardIdsSha256": calculate_card_ids_sha256(),
        "onnx": {
            "opsetVersion": ONNX_OPSET_VERSION,
            "inputs": [
                {
                    "name": ONNX_INPUT_NAME,
                    "shape": [_BATCH_DIMENSION, MODEL_INPUT_FEATURE_COUNT],
                    "dtype": _FLOAT32_DTYPE,
                }
            ],
            "outputs": [
                {
                    "name": ONNX_OUTPUT_NAME,
                    "shape": [_BATCH_DIMENSION, CARD_COUNT],
                    "dtype": _FLOAT32_DTYPE,
                }
            ],
        },
        "policyModel": model.config.to_dict(),
    }


def validate_policy_onnx_metadata(metadata: dict[str, Any]) -> None:
    expected = {
        "metadataSchemaVersion": POLICY_ONNX_METADATA_SCHEMA_VERSION,
        "checkpointSchemaVersion": CHECKPOINT_SCHEMA_VERSION,
        "datasetSchemaVersion": DATASET_SCHEMA_VERSION,
        "playingEncoderSchemaVersion": PLAYING_ENCODER_SCHEMA_VERSION,
        "modelInputSchemaVersion": MODEL_INPUT_SCHEMA_VERSION,
        "cardIdsSha256": calculate_card_ids_sha256(),
    }

    for key, value in expected.items():
        actual = metadata.get(key)
        if actual != value:
            raise PolicyCheckpointCompatibilityError(
                f"metadata {key} mismatch: expected {value!r}, got {actual!r}."
            )

    onnx = metadata.get("onnx")
    if not isinstance(onnx, dict):
        raise PolicyCheckpointCompatibilityError("metadata onnx must be a dictionary.")

    _validate_io_metadata(
        onnx.get("inputs"),
        expected_name=ONNX_INPUT_NAME,
        expected_shape=(_BATCH_DIMENSION, MODEL_INPUT_FEATURE_COUNT),
        expected_dtype=_FLOAT32_DTYPE,
        label="input",
    )
    _validate_io_metadata(
        onnx.get("outputs"),
        expected_name=ONNX_OUTPUT_NAME,
        expected_shape=(_BATCH_DIMENSION, CARD_COUNT),
        expected_dtype=_FLOAT32_DTYPE,
        label="output",
    )


def _validate_checkpoint_metadata_for_export(checkpoint: dict[str, object]) -> None:
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "playing_encoder_schema_version": PLAYING_ENCODER_SCHEMA_VERSION,
        "model_input_schema_version": MODEL_INPUT_SCHEMA_VERSION,
        "card_ids_sha256": calculate_card_ids_sha256(),
    }

    for key, value in expected.items():
        actual = checkpoint.get(key)
        if actual != value:
            raise PolicyCheckpointCompatibilityError(
                f"checkpoint {key} mismatch: expected {value!r}, got {actual!r}."
            )


def _validate_model_for_export(model: PolicyMlpModel) -> None:
    if model.config.input_dim != MODEL_INPUT_FEATURE_COUNT:
        raise PolicyCheckpointCompatibilityError(
            "checkpoint model_config.input_dim mismatch: "
            f"expected {MODEL_INPUT_FEATURE_COUNT}, got {model.config.input_dim}."
        )

    with torch.no_grad():
        output = model(torch.zeros((2, MODEL_INPUT_FEATURE_COUNT), dtype=torch.float32))

    if tuple(output.shape) != (2, CARD_COUNT):
        raise PolicyCheckpointCompatibilityError(
            f"policy model must emit shape (batch, {CARD_COUNT}), got {tuple(output.shape)}."
        )

    if output.dtype != torch.float32:
        raise PolicyCheckpointCompatibilityError(
            f"policy model must emit torch.float32 logits, got {output.dtype}."
        )


def _export_onnx(*, model: PolicyMlpModel, dummy_input: torch.Tensor, output: Path) -> None:
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
    model: PolicyMlpModel,
    onnx_path: Path,
    sample: TensorizedPlayingSample,
) -> _OnnxParityResult:
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]
    except ImportError as error:
        raise PolicyCheckpointCompatibilityError(
            "onnxruntime is required for policy ONNX parity checks."
        ) from error

    model_input_np = sample.model_input.astype(np.float32, copy=True).reshape(
        1, MODEL_INPUT_FEATURE_COUNT
    )
    legal_mask_np = sample.legal_play_mask.astype(np.bool_, copy=True).reshape(1, CARD_COUNT)
    with torch.no_grad():
        model_input_torch = torch.from_numpy(model_input_np)
        legal_mask_torch = torch.from_numpy(legal_mask_np)
        pytorch_logits = model(model_input_torch)
        pytorch_selected = select_policy_action(pytorch_logits, legal_mask_torch)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    _validate_onnx_runtime_io(session)
    outputs = session.run([ONNX_OUTPUT_NAME], {ONNX_INPUT_NAME: model_input_np})
    onnx_logits_np = cast(np.ndarray, outputs[0])
    if onnx_logits_np.shape != (1, CARD_COUNT):
        raise PolicyCheckpointCompatibilityError(
            f"ONNX output must have shape (1, {CARD_COUNT}), got {onnx_logits_np.shape}."
        )
    if onnx_logits_np.dtype != np.float32:
        raise PolicyCheckpointCompatibilityError(
            f"ONNX output must have dtype float32, got {onnx_logits_np.dtype}."
        )

    onnx_logits = torch.from_numpy(onnx_logits_np)
    try:
        np.testing.assert_allclose(
            pytorch_logits.detach().numpy(),
            onnx_logits_np,
            rtol=_PARITY_RTOL,
            atol=_PARITY_ATOL,
        )
    except AssertionError as error:
        raise PolicyCheckpointCompatibilityError(
            "PyTorch and ONNX Runtime logits differ beyond tolerance."
        ) from error

    onnx_selected = select_policy_action(onnx_logits, legal_mask_torch)
    pytorch_selected_index = int(pytorch_selected.item())
    onnx_selected_index = int(onnx_selected.item())
    if onnx_selected_index != pytorch_selected_index:
        raise PolicyCheckpointCompatibilityError(
            "PyTorch and ONNX Runtime masked selections differ: "
            f"PyTorch={pytorch_selected_index}, ONNX={onnx_selected_index}."
        )

    return _OnnxParityResult(
        max_abs_logit_diff=float(
            np.max(np.abs(pytorch_logits.detach().numpy() - onnx_logits_np))
        ),
        pytorch_selected_card_index=pytorch_selected_index,
        onnx_selected_card_index=onnx_selected_index,
    )


def _validate_onnx_runtime_io(session: Any) -> None:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1:
        raise PolicyCheckpointCompatibilityError(
            f"ONNX model must have one input, got {len(inputs)}."
        )
    if len(outputs) != 1:
        raise PolicyCheckpointCompatibilityError(
            f"ONNX model must have one output, got {len(outputs)}."
        )

    input_meta = inputs[0]
    output_meta = outputs[0]
    if input_meta.name != ONNX_INPUT_NAME:
        raise PolicyCheckpointCompatibilityError(
            f"ONNX input name mismatch: expected {ONNX_INPUT_NAME!r}, got {input_meta.name!r}."
        )
    if output_meta.name != ONNX_OUTPUT_NAME:
        raise PolicyCheckpointCompatibilityError(
            f"ONNX output name mismatch: expected {ONNX_OUTPUT_NAME!r}, got {output_meta.name!r}."
        )
    if (
        len(input_meta.shape) != 2
        or not isinstance(input_meta.shape[0], str)
        or input_meta.shape[1] != MODEL_INPUT_FEATURE_COUNT
    ):
        raise PolicyCheckpointCompatibilityError(
            "ONNX input shape mismatch: expected dynamic batch and "
            f"{MODEL_INPUT_FEATURE_COUNT} features, got {input_meta.shape!r}."
        )
    if (
        len(output_meta.shape) != 2
        or not isinstance(output_meta.shape[0], str)
        or output_meta.shape[1] != CARD_COUNT
    ):
        raise PolicyCheckpointCompatibilityError(
            "ONNX output shape mismatch: expected dynamic batch and "
            f"{CARD_COUNT} logits, got {output_meta.shape!r}."
        )
    if input_meta.type != "tensor(float)":
        raise PolicyCheckpointCompatibilityError(
            f"ONNX input dtype mismatch: expected tensor(float), got {input_meta.type}."
        )
    if output_meta.type != "tensor(float)":
        raise PolicyCheckpointCompatibilityError(
            f"ONNX output dtype mismatch: expected tensor(float), got {output_meta.type}."
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
        raise PolicyCheckpointCompatibilityError(f"metadata onnx {label}s must contain one item.")

    item = value[0]
    if not isinstance(item, dict):
        raise PolicyCheckpointCompatibilityError(
            f"metadata onnx {label} item must be a dictionary."
        )

    expected_shape_list = [expected_shape[0], expected_shape[1]]
    if item.get("name") != expected_name:
        raise PolicyCheckpointCompatibilityError(
            f"metadata onnx {label} name mismatch: "
            f"expected {expected_name!r}, got {item.get('name')!r}."
        )
    if item.get("shape") != expected_shape_list:
        raise PolicyCheckpointCompatibilityError(
            "metadata onnx "
            f"{label} shape mismatch: expected {expected_shape_list!r}, got {item.get('shape')!r}."
        )
    if item.get("dtype") != expected_dtype:
        raise PolicyCheckpointCompatibilityError(
            "metadata onnx "
            f"{label} dtype mismatch: expected {expected_dtype!r}, got {item.get('dtype')!r}."
        )
