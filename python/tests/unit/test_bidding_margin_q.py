from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from test_bidding_role_q import _write_role_q_dataset

from napoleon_ml.bidding_q import (
    BIDDING_MARGIN_HETEROSCEDASTIC_ARCHITECTURE_ID,
    BiddingMarginHeteroscedasticModel,
    BiddingMarginHeteroscedasticModelConfig,
    BiddingMarginTrainConfig,
    gaussian_margin_nll,
    gaussian_success_probability,
    interval_coverage,
    load_bidding_margin_checkpoint,
    load_bidding_q_dataset,
    margin_sigma_from_log_variance,
    risk_aware_ranking,
    save_bidding_margin_artifact,
    sigma_diagnostics,
    state_key_hash,
    train_bidding_margin_model,
)
from napoleon_ml.bidding_q.margin_training import create_margin_split
from napoleon_ml.dataset.constants import BIDDING_ACTION_COUNT
from napoleon_ml.dataset.tensors import BIDDING_MODEL_INPUT_FEATURE_COUNT


def test_margin_model_forward_shape_and_sigma_positive() -> None:
    model = BiddingMarginHeteroscedasticModel(
        BiddingMarginHeteroscedasticModelConfig(hidden_dims=(8,))
    )
    mean, log_variance = model(
        torch.zeros((2, BIDDING_MODEL_INPUT_FEATURE_COUNT), dtype=torch.float32)
    )

    assert mean.shape == (2, BIDDING_ACTION_COUNT)
    assert log_variance.shape == (2, BIDDING_ACTION_COUNT)
    assert torch.isfinite(log_variance).all()
    assert torch.gt(margin_sigma_from_log_variance(log_variance), 0.0).all()
    assert (
        model.config.to_dict()["architectureId"] == BIDDING_MARGIN_HETEROSCEDASTIC_ARCHITECTURE_ID
    )


def test_gaussian_nll_selected_action_only_and_no_contract_mask() -> None:
    mean = torch.zeros((2, BIDDING_ACTION_COUNT), dtype=torch.float32, requires_grad=True)
    log_variance = torch.zeros((2, BIDDING_ACTION_COUNT), dtype=torch.float32, requires_grad=True)
    action_index = torch.tensor([3, 4], dtype=torch.long)
    target = torch.tensor([2.0, 100.0], dtype=torch.float32)
    mask = torch.tensor([True, False], dtype=torch.bool)

    loss = gaussian_margin_nll(mean, log_variance, action_index, target, mask)
    assert loss.item() == pytest.approx(2.0)
    loss.backward()  # type: ignore[no-untyped-call]

    assert mean.grad is not None
    assert log_variance.grad is not None
    assert mean.grad.nonzero().tolist() == [[0, 3]]
    assert log_variance.grad.nonzero().tolist() == [[0, 3]]

    zero_loss = gaussian_margin_nll(
        mean.detach(),
        log_variance.detach(),
        action_index,
        target,
        torch.tensor([False, False], dtype=torch.bool),
    )
    assert zero_loss.item() == pytest.approx(0.0)


def test_extreme_scale_is_clamped_and_finite() -> None:
    config = BiddingMarginHeteroscedasticModelConfig(
        hidden_dims=(8,), log_variance_min=-2.0, log_variance_max=2.0
    )
    model = BiddingMarginHeteroscedasticModel(config)
    with torch.no_grad():
        model.log_variance_head.bias.fill_(1000.0)
    _mean, log_variance = model(
        torch.zeros((1, BIDDING_MODEL_INPUT_FEATURE_COUNT), dtype=torch.float32)
    )
    assert log_variance.max().item() <= 2.0
    loss = gaussian_margin_nll(
        torch.zeros((1, BIDDING_ACTION_COUNT)),
        log_variance,
        torch.tensor([0]),
        torch.tensor([1000.0]),
        torch.tensor([True]),
    )
    assert torch.isfinite(loss)


def test_fixed_validation_split_has_no_state_key_leakage(tmp_path: Path) -> None:
    _write_role_q_dataset(tmp_path / "dataset", repeats=1, state_count=12)
    dataset = load_bidding_q_dataset(tmp_path / "dataset")
    validation_keys = sorted([f"state-{index}" for index in range(2)])
    validation_file = tmp_path / "validation-state-keys.json"
    validation_file.write_text(json.dumps({"stateKeys": validation_keys}), encoding="utf-8")

    split = create_margin_split(
        dataset,
        BiddingMarginTrainConfig(
            hidden_dims=(8,),
            train_state_count=5,
            validation_state_keys_path=str(validation_file),
        ),
    )

    assert split.validation_state_key_hash == state_key_hash(validation_keys)
    assert split.train_state_keys.isdisjoint(split.validation_state_keys)


def test_margin_diagnostics_interval_sigma_bucket_and_risk_ranking(tmp_path: Path) -> None:
    _write_role_q_dataset(tmp_path / "dataset", repeats=1, state_count=8)
    dataset = load_bidding_q_dataset(tmp_path / "dataset")
    samples = tuple(sample for sample in dataset.raw_samples if sample.result_type == "standard")
    truth = np.asarray(
        [
            sample.napoleon_side_point_cards - sample.final_declared_target
            for sample in samples
            if sample.napoleon_side_point_cards is not None
            and sample.final_declared_target is not None
        ],
        dtype=np.float64,
    )
    mean = truth + np.linspace(-2.0, 2.0, truth.size)
    sigma = np.linspace(0.5, 3.0, truth.size)

    diagnostics = sigma_diagnostics(samples, truth=truth, mean=mean, sigma=sigma)
    assert cast(dict[str, object], diagnostics["sharpness"])["overall"] is not None
    buckets = cast(list[dict[str, object]], diagnostics["sigmaBuckets"])
    assert [bucket["bucket"] for bucket in buckets] == ["low", "mid", "high"]
    coverage = interval_coverage(truth=truth, mean=mean, sigma=sigma)
    assert 0.0 <= cast(float, coverage["oneSigma"]) <= 1.0
    probability = gaussian_success_probability(mean, sigma)
    assert np.all((probability >= 0.0) & (probability <= 1.0))

    score_mean = np.zeros((len(dataset.raw_samples), BIDDING_ACTION_COUNT), dtype=np.float64)
    score_sigma = np.ones_like(score_mean)
    for index, sample in enumerate(dataset.raw_samples):
        if sample.result_type == "standard" and sample.contract_margin is not None:
            score_mean[index, sample.forced_action_index] = sample.contract_margin
    ranking = risk_aware_ranking(dataset.raw_samples, mean=score_mean, sigma=score_sigma)
    assert cast(dict[str, object], ranking["0.0"])["pairwiseRankingAccuracy"] is not None


def test_margin_training_checkpoint_and_onnx_parity(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    _write_role_q_dataset(tmp_path / "dataset", repeats=1, state_count=16)
    dataset = load_bidding_q_dataset(tmp_path / "dataset")
    validation_keys = sorted([f"state-{index}" for index in range(4)])
    validation_file = tmp_path / "validation-state-keys.json"
    validation_file.write_text(json.dumps({"stateKeys": validation_keys}), encoding="utf-8")
    config = BiddingMarginTrainConfig(
        seed=380,
        epochs=6,
        batch_size=8,
        learning_rate=5e-3,
        hidden_dims=(16, 8),
        train_state_count=8,
        validation_state_keys_path=str(validation_file),
        patience=3,
        device="cpu",
    )

    first = train_bidding_margin_model(dataset, config)
    second = train_bidding_margin_model(dataset, config)

    assert first.epoch_reports == second.epoch_reports
    assert 1 <= first.best_epoch <= len(first.epoch_reports) <= config.epochs
    validation = first.validation_report
    mean = cast(dict[str, object], validation["mean"])
    sigma = cast(dict[str, object], validation["sigma"])
    success = cast(dict[str, object], validation["gaussianSuccess"])
    baselines = cast(dict[str, object], validation["baselines"])
    assert cast(dict[str, object], mean["overall"])["mae"] is not None
    assert cast(dict[str, object], sigma["intervalCoverage"])["oneSigma"] is not None
    assert success["rocAuc"] is not None
    assert cast(dict[str, object], baselines["globalResidualStd"])["nll"] is not None

    artifact = save_bidding_margin_artifact(tmp_path / "artifact", result=first, dataset=dataset)
    parity = cast(dict[str, object], artifact["onnxParity"])
    assert parity["checked"] is True
    assert cast(float, parity["maxAbsDiff"]) <= 1e-5
    loaded_model, checkpoint = load_bidding_margin_checkpoint(Path(str(artifact["checkpointPath"])))
    assert checkpoint["modelType"] == "bidding-margin-heteroscedastic"
    loaded_mean, loaded_log_variance = loaded_model(
        torch.zeros((1, BIDDING_MODEL_INPUT_FEATURE_COUNT), dtype=torch.float32)
    )
    assert loaded_mean.shape == (1, BIDDING_ACTION_COUNT)
    assert loaded_log_variance.shape == (1, BIDDING_ACTION_COUNT)
