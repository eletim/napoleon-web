from __future__ import annotations

import pytest
import torch

from napoleon_ml.dataset.constants import CARD_COUNT
from napoleon_ml.policy.behavior_parity import (
    BATCHED_CUDA_BEHAVIOR_PARITY_TOLERANCE,
    STRICT_BEHAVIOR_PARITY_TOLERANCE,
    BehaviorParityDiagnostics,
)


def test_batched_cuda_strict_failed_count_is_diagnostic_only() -> None:
    diagnostics = _diagnostics_from_errors(
        [0.001] * 100,
        execution_provider="cuda",
        max_observed_batch_size=32,
    )

    assert diagnostics.strict_failed_count == 100
    assert diagnostics.failed() is False
    assert diagnostics.warning() is False
    assert diagnostics.to_dict()["severity"] == "pass"


def test_batched_cuda_iter_361_known_diagnostics_continue_as_pass() -> None:
    errors = [0.0] * 998 + [0.0041306] * 2
    diagnostics = _diagnostics_from_errors(
        errors,
        execution_provider="cuda",
        max_observed_batch_size=179,
    )

    assert diagnostics.failed() is False
    assert diagnostics.warning() is False
    assert diagnostics.p999_abs_error() == pytest.approx(0.0041306)
    assert diagnostics.to_dict()["severity"] == "pass"


def test_batched_cuda_iter_420_known_diagnostics_continue_as_pass() -> None:
    errors = (
        [0.0001522570] * 4055
        + [0.0021209717] * 36
        + [0.0033550262] * 4
        + [0.0039141178]
    )
    diagnostics = _diagnostics_from_errors(
        errors,
        execution_provider="cuda",
        max_observed_batch_size=179,
    )

    assert diagnostics.failed() is False
    assert diagnostics.warning() is False
    assert diagnostics.mean_abs_error() == pytest.approx(0.0001736, rel=0.02)
    assert diagnostics.p99_abs_error() == pytest.approx(0.0021209717)
    assert diagnostics.p999_abs_error() == pytest.approx(0.0033550262)
    assert diagnostics.max_abs_error == pytest.approx(0.0039141178)
    assert diagnostics.to_dict()["severity"] == "pass"


def test_batched_cuda_small_max_outlier_passes_without_hard_failure() -> None:
    errors = [0.0] * 4091 + [0.0028779507] * 4 + [0.0051143169]
    diagnostics = _diagnostics_from_errors(
        errors,
        execution_provider="cuda",
        max_observed_batch_size=32,
    )

    assert diagnostics.failed() is False
    assert diagnostics.warning() is False
    assert diagnostics.max_abs_error == pytest.approx(0.0051143169)
    assert diagnostics.p99_abs_error() <= 0.005
    assert diagnostics.p999_abs_error() <= 0.010

    artifact = diagnostics.to_dict()
    assert artifact["passed"] is True
    assert artifact["severity"] == "pass"
    assert artifact["warningCount"] == 0
    assert artifact["hardFailureCount"] == 0


def test_batched_cuda_iter_390_known_diagnostics_continue_as_pass() -> None:
    errors = (
        [0.0001169613] * 4055
        + [0.0015007257] * 36
        + [0.0031688213] * 4
        + [0.0111589432]
    )
    diagnostics = _diagnostics_from_errors(
        errors,
        execution_provider="cuda",
        max_observed_batch_size=180,
    )

    assert diagnostics.failed() is False
    assert diagnostics.warning() is False
    assert diagnostics.mean_abs_error() == pytest.approx(0.000135, rel=0.01)
    assert diagnostics.p99_abs_error() == pytest.approx(0.0015007257)
    assert diagnostics.p999_abs_error() == pytest.approx(0.0031688213)
    assert diagnostics.max_abs_error == pytest.approx(0.0111589432)

    artifact = diagnostics.to_dict()
    assert artifact["passed"] is True
    assert artifact["severity"] == "pass"
    assert artifact["warningCount"] == 0
    assert artifact["hardFailureCount"] == 0
    assert artifact["maxAbsError"] == pytest.approx(0.0111589432)
    assert artifact["p99AbsError"] == pytest.approx(0.0015007257)
    assert artifact["p999AbsError"] == pytest.approx(0.0031688213)
    checks = _numeric_checks_by_metric(artifact)
    assert checks["maxAbsError"]["status"] == "pass"
    assert checks["maxAbsError"]["passThreshold"] == pytest.approx(0.020)
    assert checks["maxAbsError"]["hardFailThreshold"] == pytest.approx(0.100)


def test_batched_cuda_max_abs_boundary_passes_at_warning_threshold() -> None:
    diagnostics = _diagnostics_from_errors(
        [0.0] * 99 + [0.020],
        execution_provider="cuda",
        max_observed_batch_size=32,
    )

    assert diagnostics.failed() is False
    assert diagnostics.warning() is False
    assert diagnostics.max_abs_error == pytest.approx(0.020)
    assert diagnostics.to_dict()["severity"] == "pass"


def test_batched_cuda_max_abs_boundary_warns_at_hard_threshold() -> None:
    diagnostics = _diagnostics_from_errors(
        [0.0] * 99 + [0.100],
        execution_provider="cuda",
        max_observed_batch_size=32,
    )

    assert diagnostics.failed() is False
    assert diagnostics.warning() is True
    assert diagnostics.max_abs_error == pytest.approx(0.100)
    artifact = diagnostics.to_dict()
    assert artifact["severity"] == "warning"
    assert artifact["warningCount"] == 1
    assert artifact["hardFailureCount"] == 0


@pytest.mark.parametrize(
    ("metric", "errors", "expected_value"),
    [
        ("meanAbsError", [0.002] * 100, 0.002),
        ("p99AbsError", [0.0] * 98 + [0.006] * 2, 0.006),
        ("p999AbsError", [0.0] * 998 + [0.012] * 2, 0.012),
        ("maxAbsError", [0.0] * 99 + [0.021], 0.021),
    ],
)
def test_batched_cuda_warning_bands_are_machine_readable(
    metric: str,
    errors: list[float],
    expected_value: float,
) -> None:
    diagnostics = _diagnostics_from_errors(
        errors,
        execution_provider="cuda",
        max_observed_batch_size=32,
    )

    assert diagnostics.failed() is False
    assert diagnostics.warning() is True

    artifact = diagnostics.to_dict()
    assert artifact["passed"] is True
    assert artifact["severity"] == "warning"
    assert artifact["warningCount"] == 1
    checks = _numeric_checks_by_metric(artifact)
    assert checks[metric]["value"] == pytest.approx(expected_value)
    assert checks[metric]["status"] == "warning"
    assert checks[metric]["evaluated"] is True
    warnings = artifact["warnings"]
    assert isinstance(warnings, list)
    assert metric.replace("AbsError", "").lower().replace("p999", "p99.9") in str(
        warnings[0],
    ).replace(" ", "").lower()


def test_batched_cuda_p999_boundary_passes_at_warning_threshold() -> None:
    diagnostics = _diagnostics_from_errors(
        [0.0] * 998 + [0.010] * 2,
        execution_provider="cuda",
        max_observed_batch_size=32,
    )

    assert diagnostics.failed() is False
    assert diagnostics.warning() is False
    assert diagnostics.p999_abs_error() == pytest.approx(0.010)
    assert diagnostics.to_dict()["severity"] == "pass"


def test_batched_cuda_warning_does_not_depend_on_strict_failure_count() -> None:
    diagnostics = _diagnostics_from_selected_and_behavior(
        [-1000.0],
        [-1000.002],
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
        ([0.0031] * 100, "mean abs error"),
        ([0.0] * 98 + [0.0101, 0.0101], "p99 abs error"),
        ([0.0] * 998 + [0.0201, 0.0201], "p99.9 abs error"),
        ([0.0] * 99 + [0.1001], "max abs error"),
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
        [-2000.0] * 100,
        [-2000.0] * 99 + [-2000.101],
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


def _numeric_checks_by_metric(artifact: dict[str, object]) -> dict[str, dict[str, object]]:
    checks = artifact["numericChecks"]
    assert isinstance(checks, list)
    result: dict[str, dict[str, object]] = {}
    for check in checks:
        assert isinstance(check, dict)
        metric = check["metric"]
        assert isinstance(metric, str)
        result[metric] = check
    return result


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
