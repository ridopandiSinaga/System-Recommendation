"""Validation-only experiment invariants, without importing TensorFlow."""
import numpy as np
import pytest

from scripts.colab_neural_ablation import (
    CANDIDATES, FROZEN_INPUT_SHA256, correction_summary, experiment_protocol, fit_scalar_intercept,
)


def test_frozen_experiment_excludes_test_and_pairs_initialization_ablation():
    assert set(FROZEN_INPUT_SHA256) == {
        "train_books.csv", "train_ratings.csv", "validation_ratings.csv",
    }
    assert len({candidate["name"] for candidate in CANDIDATES}) == 8
    neural = [row for row in CANDIDATES if row["kind"] == "neural"]
    zero_head = [row for row in neural if row["head_initialization"] == "zeros"]
    assert {row["l2"] for row in zero_head} == {1e-4, 1e-5, 1e-6, 1e-7, 0.0}
    random_head = [row for row in neural if row["head_initialization"] != "zeros"]
    assert len(random_head) == 1
    assert random_head[0]["l2"] == 1e-6


def test_train_fitted_scalar_is_raw_mse_optimal_even_outside_rating_bounds():
    targets = np.array([0.0, 0.3, 1.0])
    baseline = np.array([-0.2, 0.2, 1.4])
    scalar = fit_scalar_intercept(targets, baseline, np)
    best_mse = np.mean((targets - baseline - scalar) ** 2)
    for perturbation in (-0.5, -0.01, 0.01, 0.5):
        assert np.mean((targets - baseline - scalar - perturbation) ** 2) > best_mse
    # Clipping before fitting is a different control and must not be substituted.
    assert scalar != pytest.approx(fit_scalar_intercept(targets, np.clip(baseline, 0, 1), np))


@pytest.mark.parametrize("targets,baseline", [([], []), ([1], [1, 2]), ([np.nan], [0])])
def test_scalar_rejects_invalid_training_arrays(targets, baseline):
    with pytest.raises(ValueError):
        fit_scalar_intercept(targets, baseline, np)


def test_correction_statistics_use_rating_points_and_preserve_variability():
    constant = correction_summary([0.01, 0.01, 0.01], np)
    varying = correction_summary([-0.01, 0.01, 0.03], np)
    assert constant["mean_rating"] == pytest.approx(0.09)
    assert constant["std_rating"] == pytest.approx(0)
    assert varying["mean_rating"] == pytest.approx(constant["mean_rating"])
    assert varying["std_rating"] > 0
    assert varying["min_rating"] == pytest.approx(-0.09)
    assert varying["max_rating"] == pytest.approx(0.27)


def test_default_ablation_retains_eight_candidates_and_original_stopping_protocol():
    protocol = experiment_protocol("ablation", 10)
    assert protocol["candidates"] == list(CANDIDATES)
    assert protocol["candidate_count"] == 8
    assert protocol["max_neural_epochs_each"] == 10
    assert protocol["patience"] == 3
    assert protocol["early_stopping_enabled"] is True
    assert protocol["snapshot_interval_epochs"] == 2


def test_warmup_is_two_predeclared_fixed_twelve_epoch_runs():
    protocol = experiment_protocol("warmup", 12)
    assert [row["name"] for row in protocol["candidates"]] == [
        "residual_l2_0_zero_head_fixed12", "residual_l2_1e7_zero_head_fixed12",
    ]
    assert [row["l2"] for row in protocol["candidates"]] == [0, 1e-7]
    assert all(row["kind"] == "neural" for row in protocol["candidates"])
    assert protocol["max_neural_epochs_each"] == protocol["min_neural_epochs_each"] == 12
    assert protocol["patience"] == 12
    assert protocol["early_stopping_enabled"] is False
    assert protocol["snapshot_interval_epochs"] == 4
    assert protocol["previous_controls_rerun"] is False


@pytest.mark.parametrize("preset,epochs", [("ablation", 12), ("ablation", 0),
                                          ("warmup", 10), ("warmup", 13), ("unknown", 12)])
def test_preset_cannot_silently_change_predeclared_epoch_limit(preset, epochs):
    with pytest.raises(ValueError):
        experiment_protocol(preset, epochs)
