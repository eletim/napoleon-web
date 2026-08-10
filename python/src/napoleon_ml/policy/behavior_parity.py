"""Behavior policy provenance and log-probability parity diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from napoleon_ml.dataset.manifest import DatasetManifest
from napoleon_ml.policy.checkpoint import PolicyCheckpointCompatibilityError
from napoleon_ml.policy.model import PolicyActorCriticModel, PolicyMlpModel
from napoleon_ml.policy.onnx_export import build_policy_onnx_metadata

STRICT_BEHAVIOR_LOG_PROB_PARITY_RTOL = 1e-4
STRICT_BEHAVIOR_LOG_PROB_PARITY_ATOL = 1e-5
BATCHED_CUDA_BEHAVIOR_LOG_PROB_MAX_ABS = 5e-3
BATCHED_CUDA_BEHAVIOR_LOG_PROB_P99_ABS = 2e-3
BATCHED_CUDA_BEHAVIOR_LOG_PROB_P999_ABS = 4e-3


@dataclass(frozen=True)
class BehaviorParityTolerance:
    mode: str
    rtol: float
    atol: float
    max_abs_error: float
    p99_abs_error: float | None = None
    p999_abs_error: float | None = None
    distribution_guard_min_count: int = 100

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "rtol": self.rtol,
            "atol": self.atol,
            "maxAbsError": self.max_abs_error,
            "p99AbsError": self.p99_abs_error,
            "p999AbsError": self.p999_abs_error,
            "distributionGuardMinCount": self.distribution_guard_min_count,
        }


STRICT_BEHAVIOR_PARITY_TOLERANCE = BehaviorParityTolerance(
    mode="strict",
    rtol=STRICT_BEHAVIOR_LOG_PROB_PARITY_RTOL,
    atol=STRICT_BEHAVIOR_LOG_PROB_PARITY_ATOL,
    max_abs_error=STRICT_BEHAVIOR_LOG_PROB_PARITY_ATOL,
    p99_abs_error=None,
    p999_abs_error=None,
)

BATCHED_CUDA_BEHAVIOR_PARITY_TOLERANCE = BehaviorParityTolerance(
    mode="batched-cuda",
    rtol=STRICT_BEHAVIOR_LOG_PROB_PARITY_RTOL,
    atol=STRICT_BEHAVIOR_LOG_PROB_PARITY_ATOL,
    max_abs_error=BATCHED_CUDA_BEHAVIOR_LOG_PROB_MAX_ABS,
    p99_abs_error=BATCHED_CUDA_BEHAVIOR_LOG_PROB_P99_ABS,
    p999_abs_error=BATCHED_CUDA_BEHAVIOR_LOG_PROB_P999_ABS,
)


@dataclass(frozen=True)
class BehaviorPolicyProvenanceDiagnostic:
    source_checkpoint_sha256: str
    expected_metadata_sha256: str
    manifest_metadata_sha256: str
    manifest_onnx_sha256: str
    metadata_sha256_matches_expected: bool
    metadata_file_sha256_checked: bool
    onnx_file_sha256_checked: bool
    metadata_file_path: Path | None
    onnx_file_path: Path | None

    def to_dict(self) -> dict[str, object]:
        return {
            "sourceCheckpointSha256": self.source_checkpoint_sha256,
            "expectedMetadataSha256": self.expected_metadata_sha256,
            "manifestMetadataSha256": self.manifest_metadata_sha256,
            "manifestOnnxSha256": self.manifest_onnx_sha256,
            "metadataSha256MatchesExpected": self.metadata_sha256_matches_expected,
            "metadataFileSha256Checked": self.metadata_file_sha256_checked,
            "onnxFileSha256Checked": self.onnx_file_sha256_checked,
            "metadataFilePath": (
                str(self.metadata_file_path) if self.metadata_file_path is not None else None
            ),
            "onnxFilePath": str(self.onnx_file_path) if self.onnx_file_path is not None else None,
        }


@dataclass
class BehaviorParityDiagnostics:
    tolerance: BehaviorParityTolerance
    execution_provider: str | None = None
    max_observed_batch_size: int | None = None
    sample_count: int = 0
    strict_failed_count: int = 0
    max_abs_error: float = 0.0
    mean_abs_error_sum: float = 0.0
    nonfinite_count: int = 0
    _errors: list[float] = field(default_factory=list)

    def update(self, *, selected_log_probability: Tensor, behavior_log_probability: Tensor) -> None:
        selected = selected_log_probability.detach()
        behavior = behavior_log_probability.detach().to(
            device=selected.device,
            dtype=selected.dtype,
        )
        errors = torch.abs(selected - behavior)
        finite = torch.isfinite(errors)
        finite_errors = errors[finite]
        batch_size = int(errors.numel())
        self.sample_count += batch_size
        self.nonfinite_count += int((~finite).sum().item())
        if int(finite_errors.numel()) > 0:
            error_list = [float(value) for value in finite_errors.cpu().tolist()]
            self._errors.extend(error_list)
            self.mean_abs_error_sum += float(finite_errors.sum().item())
            self.max_abs_error = max(self.max_abs_error, max(error_list))

        close = torch.isclose(
            selected,
            behavior,
            rtol=self.tolerance.rtol,
            atol=self.tolerance.atol,
        )
        self.strict_failed_count += int((~close).sum().item())

    def mean_abs_error(self) -> float:
        if not self._errors:
            return 0.0
        return self.mean_abs_error_sum / len(self._errors)

    def p99_abs_error(self) -> float:
        return _nearest_rank_quantile(self._errors, 0.99)

    def p999_abs_error(self) -> float:
        return _nearest_rank_quantile(self._errors, 0.999)

    def failed(self) -> bool:
        if self.nonfinite_count > 0:
            return True
        if self.strict_failed_count == 0:
            return False
        if self.tolerance.mode == "strict":
            return True
        if self.max_abs_error > self.tolerance.max_abs_error:
            return True
        if (
            self.tolerance.p99_abs_error is not None
            and self.sample_count >= self.tolerance.distribution_guard_min_count
            and self.p99_abs_error() > self.tolerance.p99_abs_error
        ):
            return True
        p999_min_count = self.tolerance.distribution_guard_min_count * 10
        if (
            self.tolerance.p999_abs_error is not None
            and self.sample_count >= p999_min_count
            and self.p999_abs_error() > self.tolerance.p999_abs_error
        ):
            return True
        return False

    def failure_detail(self) -> str:
        if self.nonfinite_count > 0:
            return f"{self.nonfinite_count} non-finite behavior parity errors"
        if self.tolerance.mode == "strict":
            return (
                f"{self.strict_failed_count} samples exceed rtol={self.tolerance.rtol} "
                f"and atol={self.tolerance.atol}"
            )
        if self.max_abs_error > self.tolerance.max_abs_error:
            return (
                f"max abs error {self.max_abs_error:.8g} exceeds "
                f"{self.tolerance.max_abs_error:.8g}"
            )
        if (
            self.tolerance.p99_abs_error is not None
            and self.sample_count >= self.tolerance.distribution_guard_min_count
            and self.p99_abs_error() > self.tolerance.p99_abs_error
        ):
            return (
                f"p99 abs error {self.p99_abs_error():.8g} exceeds "
                f"{self.tolerance.p99_abs_error:.8g}"
            )
        if (
            self.tolerance.p999_abs_error is not None
            and self.sample_count >= self.tolerance.distribution_guard_min_count * 10
            and self.p999_abs_error() > self.tolerance.p999_abs_error
        ):
            return (
                f"p99.9 abs error {self.p999_abs_error():.8g} exceeds "
                f"{self.tolerance.p999_abs_error:.8g}"
            )
        return "behavior parity failed"

    def to_dict(self) -> dict[str, object]:
        return {
            "sampleCount": self.sample_count,
            "tolerance": self.tolerance.to_dict(),
            "executionProvider": self.execution_provider,
            "maxObservedBatchSize": self.max_observed_batch_size,
            "strictFailedCount": self.strict_failed_count,
            "nonfiniteCount": self.nonfinite_count,
            "maxAbsError": self.max_abs_error,
            "meanAbsError": self.mean_abs_error(),
            "p99AbsError": self.p99_abs_error(),
            "p999AbsError": self.p999_abs_error(),
            "passed": not self.failed(),
        }


def select_behavior_parity_tolerance(
    *,
    execution_provider: str | None,
    max_observed_batch_size: int | None,
) -> BehaviorParityTolerance:
    if execution_provider == "cuda" and max_observed_batch_size is not None:
        if max_observed_batch_size > 1:
            return BATCHED_CUDA_BEHAVIOR_PARITY_TOLERANCE
    return STRICT_BEHAVIOR_PARITY_TOLERANCE


def validate_behavior_policy_provenance(
    *,
    model: PolicyMlpModel | PolicyActorCriticModel,
    checkpoint: dict[str, object],
    manifest: DatasetManifest,
    self_play_dataset_directory: Path,
    source_checkpoint_sha256: str,
) -> BehaviorPolicyProvenanceDiagnostic:
    behavior_policy = manifest.behavior_policy
    if behavior_policy is None:
        raise ValueError("self-play manifest behaviorPolicy metadata is required.")

    expected_metadata = build_policy_onnx_metadata(
        model=model,
        checkpoint=checkpoint,
        source_checkpoint_sha256=source_checkpoint_sha256,
    )
    if behavior_policy.metadata != expected_metadata:
        raise PolicyCheckpointCompatibilityError(
            "self-play behaviorPolicy.metadata does not match the input checkpoint "
            "provenance. このtrajectoryは別policyから生成された可能性がある"
        )

    expected_metadata_sha256 = _sha256_bytes(_canonical_metadata_bytes(expected_metadata))
    metadata_sha256_matches_expected = (
        behavior_policy.metadata_sha256 == expected_metadata_sha256
    )
    if not metadata_sha256_matches_expected:
        raise PolicyCheckpointCompatibilityError(
            "self-play behaviorPolicy.metadataSha256 does not match the expected "
            "input checkpoint export metadata SHA. "
            "このtrajectoryは別policyから生成された可能性がある"
        )

    metadata_path = _resolve_behavior_artifact_path(
        self_play_dataset_directory,
        behavior_policy.metadata_file_name,
    )
    onnx_path = _resolve_behavior_artifact_path(
        self_play_dataset_directory,
        behavior_policy.onnx_file_name,
    )
    metadata_file_checked = False
    onnx_file_checked = False
    if metadata_path is not None:
        metadata_file_checked = True
        actual_metadata_sha256 = _sha256_file(metadata_path)
        if actual_metadata_sha256 != behavior_policy.metadata_sha256:
            raise PolicyCheckpointCompatibilityError(
                "self-play behavior metadata file SHA does not match manifest. "
                "このtrajectoryは別policyから生成された可能性がある"
            )
    if onnx_path is not None:
        onnx_file_checked = True
        actual_onnx_sha256 = _sha256_file(onnx_path)
        if actual_onnx_sha256 != behavior_policy.onnx_sha256:
            raise PolicyCheckpointCompatibilityError(
                "self-play behavior ONNX file SHA does not match manifest. "
                "このtrajectoryは別policyから生成された可能性がある"
            )

    return BehaviorPolicyProvenanceDiagnostic(
        source_checkpoint_sha256=source_checkpoint_sha256,
        expected_metadata_sha256=expected_metadata_sha256,
        manifest_metadata_sha256=behavior_policy.metadata_sha256,
        manifest_onnx_sha256=behavior_policy.onnx_sha256,
        metadata_sha256_matches_expected=metadata_sha256_matches_expected,
        metadata_file_sha256_checked=metadata_file_checked,
        onnx_file_sha256_checked=onnx_file_checked,
        metadata_file_path=metadata_path,
        onnx_file_path=onnx_path,
    )


def _nearest_rank_quantile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _canonical_metadata_bytes(metadata: dict[str, object]) -> bytes:
    return (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _resolve_behavior_artifact_path(directory: Path, file_name: str) -> Path | None:
    path = Path(file_name)
    candidates = [path] if path.is_absolute() else [directory / path, directory.parent / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None
