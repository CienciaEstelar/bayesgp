"""Tests for bayesgp.datasets."""

import numpy as np

from bayesgp.datasets import make_oscillatory, make_quadratic, make_step_with_noise


class TestMakeQuadratic:
    def test_shape_and_default(self):
        d = make_quadratic(n=50, seed=0)
        assert d.X.shape == (50, 1)
        assert d.y.shape == (50,)

    def test_reproducible(self):
        a = make_quadratic(n=20, seed=42)
        b = make_quadratic(n=20, seed=42)
        np.testing.assert_array_equal(a.y, b.y)

    def test_outliers_increase_variance(self):
        clean = make_quadratic(n=200, seed=0, outlier_fraction=0.0, noise_std=0.05)
        dirty = make_quadratic(
            n=200, seed=0, outlier_fraction=0.1, outlier_scale=200.0, noise_std=0.05
        )
        # Compare residuals from the underlying signal, not raw y std
        signal = (dirty.X.ravel()) ** 2
        clean_resid = (clean.y - signal).std()
        dirty_resid = (dirty.y - signal).std()
        assert dirty_resid > clean_resid * 5


class TestMakeOscillatory:
    def test_shape(self):
        d = make_oscillatory(n=100, seed=1)
        assert d.X.shape == (100, 1)
        assert d.y.shape == (100,)


class TestMakeStep:
    def test_jump_at_zero(self):
        d = make_step_with_noise(n=400, noise_std=0.001, seed=0)
        left = d.y[d.X.ravel() < -0.5].mean()
        right = d.y[d.X.ravel() > 0.5].mean()
        assert left < 0
        assert right > 0
