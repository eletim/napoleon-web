"""Diagnose PyTorch vs ONNX Runtime policy parity across ONNX batch sizes."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from napoleon_ml.cli._policy_common import handle_cli_error, load_checked_manifest
from napoleon_ml.dataset.reader import iter_tensorized_playing_self_play_samples
from napoleon_ml.dataset.tensors import TensorizedPlayingSelfPlaySample
from napoleon_ml.policy.actor_critic import load_checkpoint_for_actor_critic
from napoleon_ml.policy.device import SUPPORTED_TORCH_DEVICES, resolve_torch_device
from napoleon_ml.policy.onnx_export import validate_policy_onnx_metadata
from napoleon_ml.policy.reinforce import masked_selected_log_probability


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("self_play_dataset_directory", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--device", choices=SUPPORTED_TORCH_DEVICES, default="cpu")
    parser.add_argument("--execution-provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--batch-sizes", default="1,2,8,16,32,64,128,256")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except Exception as error:
        return handle_cli_error(error)


def _run(args: argparse.Namespace) -> int:
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]
    except ImportError as error:
        raise RuntimeError(
            "onnxruntime is required for ONNX parity diagnostics; install napoleon-ml[export]."
        ) from error

    if args.max_samples <= 0:
        raise ValueError(f"--max-samples must be positive, got {args.max_samples}.")

    manifest = load_checked_manifest(
        args.self_play_dataset_directory,
        command_label="diagnose-policy-onnx-parity",
    )
    raw_temperature = args.temperature if args.temperature is not None else manifest.temperature
    if raw_temperature is None:
        raise ValueError("self-play manifest temperature is required.")
    temperature = float(raw_temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature!r}.")

    device = resolve_torch_device(args.device)
    metadata_raw = json.loads(args.metadata.read_text(encoding="utf-8"))
    if not isinstance(metadata_raw, dict):
        raise ValueError("policy ONNX metadata must be a JSON object.")
    validate_policy_onnx_metadata(metadata_raw)
    loaded = load_checkpoint_for_actor_critic(args.checkpoint, manifest=manifest)
    model = loaded.behavior_model
    model.to(device.torch_device)
    model.eval()

    samples = list(
        _take_samples(args.self_play_dataset_directory, max_samples=int(args.max_samples))
    )
    if not samples:
        raise ValueError("self-play dataset contains no samples.")

    model_input_np = np.stack(
        [sample.model_input.astype(np.float32, copy=False) for sample in samples]
    )
    legal_mask_np = np.stack(
        [sample.legal_play_mask.astype(np.bool_, copy=False) for sample in samples]
    )
    selected_np = np.array([int(sample.selected_card_index) for sample in samples], dtype=np.int64)

    model_input = torch.from_numpy(model_input_np).to(device.torch_device)
    legal_mask = torch.from_numpy(legal_mask_np).to(device.torch_device)
    selected = torch.from_numpy(selected_np).to(device.torch_device)
    with torch.no_grad():
        pytorch_logits = model(model_input)
        pytorch_selected_logp = masked_selected_log_probability(
            pytorch_logits,
            selected,
            legal_mask,
            temperature=temperature,
        )
    pytorch_logits_np = pytorch_logits.detach().cpu().numpy()
    pytorch_selected_logp_np = pytorch_selected_logp.detach().cpu().numpy()

    providers = (
        ["CUDAExecutionProvider"]
        if args.execution_provider == "cuda"
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(str(args.onnx), providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    batch_reports = []
    for batch_size in _parse_batch_sizes(args.batch_sizes):
        onnx_logits_np = _run_onnx_batches(
            session,
            input_name=input_name,
            output_name=output_name,
            model_input=model_input_np,
            batch_size=batch_size,
        )
        onnx_selected_logp_np = _masked_selected_log_probability_np(
            onnx_logits_np,
            selected_np,
            legal_mask_np,
            temperature=temperature,
        )
        logits_abs = np.abs(onnx_logits_np - pytorch_logits_np)
        selected_abs = np.abs(onnx_selected_logp_np - pytorch_selected_logp_np)
        legal_logp_abs = _legal_log_probability_abs_error_np(
            pytorch_logits_np,
            onnx_logits_np,
            legal_mask_np,
            temperature=temperature,
        )
        batch_reports.append(
            {
                "batchSize": batch_size,
                "logits": _distribution(logits_abs.reshape(-1)),
                "selectedLegalActionLogProbability": _distribution(selected_abs),
                "legalMaskLogProbability": _distribution(legal_logp_abs),
                "argmaxMismatchCount": int(
                    np.sum(
                        _masked_argmax(pytorch_logits_np, legal_mask_np)
                        != _masked_argmax(onnx_logits_np, legal_mask_np)
                    )
                ),
                "forcedActionNonZeroCount": int(
                    np.sum(
                        np.abs(onnx_selected_logp_np[legal_mask_np.sum(axis=1) == 1]) > 0.0
                    )
                ),
            }
        )

    report = {
        "sampleCount": len(samples),
        "checkpoint": str(args.checkpoint),
        "onnx": str(args.onnx),
        "metadata": str(args.metadata),
        "torchDevice": device.to_metadata(),
        "executionProvider": args.execution_provider,
        "temperature": temperature,
        "batches": batch_reports,
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _take_samples(directory: Path, *, max_samples: int) -> list[TensorizedPlayingSelfPlaySample]:
    samples: list[TensorizedPlayingSelfPlaySample] = []
    for sample in iter_tensorized_playing_self_play_samples(directory):
        samples.append(sample)
        if len(samples) >= max_samples:
            break
    return samples


def _parse_batch_sizes(value: str) -> list[int]:
    sizes = []
    for raw in value.split(","):
        size = int(raw.strip())
        if size <= 0:
            raise ValueError(f"batch sizes must be positive, got {size}.")
        sizes.append(size)
    return sizes


def _run_onnx_batches(
    session: Any,
    *,
    input_name: str,
    output_name: str,
    model_input: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    for start in range(0, model_input.shape[0], batch_size):
        batch = model_input[start : start + batch_size]
        output = session.run([output_name], {input_name: batch})[0]
        outputs.append(np.asarray(output, dtype=np.float32))
    return np.concatenate(outputs, axis=0)


def _masked_selected_log_probability_np(
    logits: np.ndarray,
    selected: np.ndarray,
    legal_mask: np.ndarray,
    *,
    temperature: float,
) -> np.ndarray:
    logp = _masked_log_probabilities_np(logits, legal_mask, temperature=temperature)
    selected_logp = logp[np.arange(logits.shape[0]), selected]
    return np.where(legal_mask.sum(axis=1) == 1, 0.0, selected_logp)


def _legal_log_probability_abs_error_np(
    left_logits: np.ndarray,
    right_logits: np.ndarray,
    legal_mask: np.ndarray,
    *,
    temperature: float,
) -> np.ndarray:
    left = _masked_log_probabilities_np(left_logits, legal_mask, temperature=temperature)
    right = _masked_log_probabilities_np(right_logits, legal_mask, temperature=temperature)
    return cast(np.ndarray, np.abs(left[legal_mask] - right[legal_mask]))


def _masked_log_probabilities_np(
    logits: np.ndarray,
    legal_mask: np.ndarray,
    *,
    temperature: float,
) -> np.ndarray:
    masked = logits / temperature
    masked = np.where(legal_mask, masked, -np.inf)
    max_values = np.max(masked, axis=1, keepdims=True)
    exp = np.where(legal_mask, np.exp(masked - max_values), 0.0)
    log_sum = np.log(np.sum(exp, axis=1, keepdims=True)) + max_values
    return cast(np.ndarray, masked - log_sum)


def _masked_argmax(logits: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    masked = np.where(legal_mask, logits, -np.inf)
    return cast(np.ndarray, np.argmax(masked, axis=1))


def _distribution(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"max": 0.0, "mean": 0.0, "p99": 0.0, "p999": 0.0}
    return {
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "p99": float(np.quantile(values, 0.99, method="higher")),
        "p999": float(np.quantile(values, 0.999, method="higher")),
    }


if __name__ == "__main__":
    raise SystemExit(main())
