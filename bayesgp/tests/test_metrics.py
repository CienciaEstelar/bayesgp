"""Tests for bayesgp.metrics."""

import numpy as np
import pytest

from bayesgp.metrics import (
    calibration_summary,
    coverage_at,
    pit_values,
    regression_metrics,
)


class TestRegressionMetrics:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        m = regression_metrics(y, y)
        assert m["r2"] == pytest.approx(1.0)
        assert m["rmse"] == pytest.approx(0.0)
        assert m["mae"] == pytest.approx(0.0)

    def test_with_uncertainty_includes_nll(self):
        y = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.1, 1.9, 3.05])
        std = np.array([0.1, 0.1, 0.1])
        m = regression_metrics(y, pred, std)
        assert "nll" in m
        assert "coverage_95" in m

    def test_prefix(self):
        y = np.array([1.0, 2.0])
        m = regression_metrics(y, y, prefix="val_")
        assert "val_r2" in m
        assert "val_rmse" in m


class TestCoverageAt:
    def test_well_calibrated_gaussian(self):
        """Sample from N(0,1), predict mean=0 std=1; coverage should match level."""
        rng = np.random.default_rng(0)
        n = 20000
        y_true = rng.normal(0.0, 1.0, size=n)
        y_pred = np.zeros(n)
        y_std = np.ones(n)
        for level in (0.5, 0.9, 0.95):
            cov = coverage_at(y_true, y_pred, y_std, level)
            assert cov == pytest.approx(level, abs=0.02)

    def test_overconfident_undercovers(self):
        """If we underestimate std by 10x, coverage at 95% drops far below 0.95."""
        rng = np.random.default_rng(0)
        n = 5000
        y_true = rng.normal(0.0, 1.0, size=n)
        y_pred = np.zeros(n)
        y_std_too_small = np.full(n, 0.1)
        cov = coverage_at(y_true, y_pred, y_std_too_small, 0.95)
        assert cov < 0.5

    def test_invalid_level(self):
        with pytest.raises(ValueError):
            coverage_at(np.array([1.0]), np.array([1.0]), np.array([1.0]), 1.5)


class TestPITValues:
    def test_pit_uniform_under_calibration(self):
        """PIT values of well-calibrated predictions are uniform on [0, 1]."""
        rng = np.random.default_rng(0)
        n = 10000
        y_true = rng.normal(0.0, 1.0, size=n)
        y_pred = np.zeros(n)
        y_std = np.ones(n)
        pits = pit_values(y_true, y_pred, y_std)
        # Kolmogorov-Smirnov distance to uniform should be small
        sorted_pits = np.sort(pits)
        uniform_grid = np.linspace(0, 1, n)
        ks = np.max(np.abs(sorted_pits - uniform_grid))
        assert ks < 0.02

    def test_pit_in_unit_interval(self):
        y = np.array([1.0, 2.0, 3.0])
        pits = pit_values(y, y, np.ones(3))
        assert np.all(pits >= 0) and np.all(pits <= 1)


class TestCalibrationSummary:
    def test_returns_all_levels(self):
        rng = np.random.default_rng(0)
        n = 1000
        y = rng.normal(size=n)
        summary = calibration_summary(y, np.zeros(n), np.ones(n))
        for key in ("coverage_50", "coverage_68", "coverage_90", "coverage_95", "coverage_99"):
            assert key in summary
