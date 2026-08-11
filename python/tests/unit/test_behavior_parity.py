from __future__ import annotations

import pytest
import torch

from napoleon_ml.dataset.constants import CARD_COUNT
from napoleon_ml.policy.behavior_parity import (
    BATCHED_CUDA_BEHAVIOR_PARITY_TOLERANCE,
    STRICT_BEHAVIOR_PARITY_TOLERANCE,
    BehaviorParityDiagnostics,
)


def test_batched_cuda_small_max_outlier_warns_without_hard_failure() -> None:
    errors = [0.0] * 4091 + [0.0028779507] * 4 + [0.0051143169]
    diagnostics = _diagnostics_from_errors(
        errors,
        execution_provider="cuda",
        max_observed_batch_size=32,
    )

    assert diagnostics.failed() is False
    assert diagnostics.warning() is True
    assert diagnostics.max_abs_error == pytest.approx(0.0051143169)
    assert diagnostics.p99_abs_error() <= 0.002
    assert diagnostics.p999_abs_error() <= 0.004

    artifact = diagnostics.to_dict()
    assert artifact["passed"] is True
    assert artifact["severity"] == "warning"
    assert artifact["warningCount"] == 1
    assert artifact["hardFailureCount"] == 0
    warnings = artifact["warnings"]
    assert isinstance(warnings, list)
    assert len(warnings) == 1
    assert str(warnings[0]).startswith("max abs error 0.005114")


def test_batched_cuda_warning_does_not_depend_on_strict_failure_count() -> None:
    diagnostics = _diagnostics_from_selected_and_behavior(
        [-1000.0],
        [-1000.006],
        execution_provider="cuda",
        max_observed_batch_size=32,
    )

    assert diagnostics.strict_failed_count == 0
    assert diagnostics.failed() is False
    assert diagnostics.warning() is True
    assert diagnostics.to_dict()["severity"] == "warning"


@pytest.mark.parametrize(
    ("errors", "detail"),
    [
        ([0.0] * 99 + [0.011], "max abs error"),
        ([0.0] * 98 + [0.003, 0.003], "p99 abs error"),
        ([0.0] * 998 + [0.0045, 0.0045], "p99.9 abs error"),
    ],
)
def test_batched_cuda_hard_failures_remain_strict(
    errors: list[float],
    detail: str,
) -> None:
    diagnostics = _diagnostics_from_errors(
        errors,
        execution_provider="cuda",
        max_observed_batch_size=32,
    )

    assert diagnostics.failed() is True
    assert diagnostics.warning() is False
    assert detail in diagnostics.failure_detail()
    artifact = diagnostics.to_dict()
    assert artifact["passed"] is False
    assert artifact["severity"] == "error"
    assert artifact["warningCount"] == 0
    hard_failure_count = artifact["hardFailureCount"]
    assert isinstance(hard_failure_count, int)
    assert hard_failure_count >= 1


def test_batched_cuda_hard_max_does_not_depend_on_strict_failure_count() -> None:
    diagnostics = _diagnostics_from_selected_and_behavior(
        [-1000.0],
        [-1000.011],
        execution_provider="cuda",
        max_observed_batch_size=32,
    )

    assert diagnostics.strict_failed_count == 0
    assert diagnostics.failed() is True
    assert "max abs error" in diagnostics.failure_detail()
    assert diagnostics.to_dict()["severity"] == "error"


def test_nonfinite_error_hard_fails() -> None:
    diagnostics = _diagnostics_from_errors(
        [float("inf")],
        execution_provider="cuda",
        max_observed_batch_size=32,
    )

    assert diagnostics.failed() is True
    assert "non-finite" in diagnostics.failure_detail()
    assert diagnostics.to_dict()["severity"] == "error"


def test_forced_action_invalid_behavior_log_probability_hard_fails() -> None:
    diagnostics = _diagnostics_from_errors(
        [0.001],
        execution_provider="cuda",
        max_observed_batch_size=32,
        forced=True,
    )

    assert diagnostics.failed() is True
    assert "forced-action" in diagnostics.failure_detail()
    assert diagnostics.to_dict()["severity"] == "error"


def test_strict_parity_still_hard_fails_for_small_drift() -> None:
    diagnostics = _diagnostics_from_errors(
        [0.001],
        execution_provider="cuda",
        max_observed_batch_size=1,
    )

    assert diagnostics.tolerance.mode == "strict"
    assert diagnostics.failed() is True
    assert diagnostics.warning() is False
    assert "samples exceed rtol" in diagnostics.failure_detail()


def _diagnostics_from_errors(
    errors: list[float],
    *,
    execution_provider: str,
    max_observed_batch_size: int,
    forced: bool = False,
) -> BehaviorParityDiagnostics:
    tolerance = (
        BATCHED_CUDA_BEHAVIOR_PARITY_TOLERANCE
        if execution_provider == "cuda" and max_observed_batch_size > 1
        else STRICT_BEHAVIOR_PARITY_TOLERANCE
    )
    selected = torch.zeros((len(errors),), dtype=torch.float32)
    behavior = -torch.tensor(errors, dtype=torch.float32)
    legal_mask = torch.zeros((len(errors), CARD_COUNT), dtype=torch.bool)
    legal_mask[:, 0] = True
    if not forced:
        legal_mask[:, 1] = True
    diagnostics = BehaviorParityDiagnostics(
        tolerance=tolerance,
        execution_provider=execution_provider,
        max_observed_batch_size=max_observed_batch_size,
    )
    diagnostics.update(
        selected_log_probability=selected,
        behavior_log_probability=behavior,
        legal_mask=legal_mask,
    )
    return diagnostics


def _diagnostics_from_selected_and_behavior(
    selected_values: list[float],
    behavior_values: list[float],
    *,
    execution_provider: str,
    max_observed_batch_size: int,
) -> BehaviorParityDiagnostics:
    tolerance = (
        BATCHED_CUDA_BEHAVIOR_PARITY_TOLERANCE
        if execution_provider == "cuda" and max_observed_batch_size > 1
        else STRICT_BEHAVIOR_PARITY_TOLERANCE
    )
    selected = torch.tensor(selected_values, dtype=torch.float32)
    behavior = torch.tensor(behavior_values, dtype=torch.float32)
    legal_mask = torch.zeros((len(selected_values), CARD_COUNT), dtype=torch.bool)
    legal_mask[:, 0] = True
    legal_mask[:, 1] = True
    diagnostics = BehaviorParityDiagnostics(
        tolerance=tolerance,
        execution_provider=execution_provider,
        max_observed_batch_size=max_observed_batch_size,
    )
    diagnostics.update(
        selected_log_probability=selected,
        behavior_log_probability=behavior,
        legal_mask=legal_mask,
    )
    return diagnostics
