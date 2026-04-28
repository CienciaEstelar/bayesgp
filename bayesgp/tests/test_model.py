"""Tests for bayesgp.model.BayesianEnsembleGP."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from bayesgp.datasets import make_oscillatory, make_quadratic
from bayesgp.metrics import coverage_at, regression_metrics
from bayesgp.model import BayesianEnsembleGP, EnsembleConfig


@pytest.fixture(scope="module")
def quadratic_data():
    return make_quadratic(n=80, seed=0)


@pytest.fixture(scope="module")
def fitted_model(quadratic_data):
    cfg = EnsembleConfig(
        kernel_families=("matern", "rbf"),
        matern_nu_values=(1.5, 2.5),
        n_restarts_optimizer=2,
        random_state=0,
    )
    model = BayesianEnsembleGP(cfg)
    model.fit(quadratic_data.X, quadratic_data.y)
    return model


class TestFitting:
    def test_fits_without_error(self, fitted_model):
        assert fitted_model._is_fitted

    def test_weights_sum_to_one(self, fitted_model):
        assert fitted_model.weights_.sum() == pytest.approx(1.0, abs=1e-10)

    def test_weights_nonnegative(self, fitted_model):
        assert (fitted_model.weights_ >= 0).all()

    def test_n_models_consistent(self, fitted_model):
        # 2 matern values + 1 rbf + 1 polynomial fallback = 4
        assert len(fitted_model.weights_) == 4
        assert len(fitted_model.model_names_) == 4

    def test_polynomial_fallback_added(self, fitted_model):
        assert "polynomial_fallback" in fitted_model.model_names_

    def test_no_fallback_when_disabled(self, quadratic_data):
        cfg = EnsembleConfig(
            kernel_families=("rbf",),
            add_polynomial_fallback=False,
            n_restarts_optimizer=1,
            random_state=0,
        )
        model = BayesianEnsembleGP(cfg).fit(quadratic_data.X, quadratic_data.y)
        assert "polynomial_fallback" not in model.model_names_


class TestPrediction:
    def test_predict_returns_mean_and_std(self, fitted_model, quadratic_data):
        mean, std = fitted_model.predict(quadratic_data.X)
        assert mean.shape == (len(quadratic_data.X),)
        assert std.shape == (len(quadratic_data.X),)
        assert (std > 0).all()

    def test_predict_without_std(self, fitted_model, quadratic_data):
        mean = fitted_model.predict(quadratic_data.X, return_std=False)
        assert mean.shape == (len(quadratic_data.X),)

    def test_recovers_quadratic(self, fitted_model, quadratic_data):
        """On training data, R^2 should be near 1."""
        pred, _ = fitted_model.predict(quadratic_data.X)
        m = regression_metrics(quadratic_data.y, pred)
        assert m["r2"] > 0.99

    def test_uncertainty_grows_with_extrapolation(self, fitted_model):
        """Std at x in training range < std far outside it."""
        x_in = np.array([[0.0]])
        x_out = np.array([[20.0]])
        _, s_in = fitted_model.predict(x_in)
        _, s_out = fitted_model.predict(x_out)
        assert s_out[0] > s_in[0]

    def test_predict_validates_n_features(self, fitted_model):
        with pytest.raises(ValueError, match="features"):
            fitted_model.predict(np.array([[1.0, 2.0]]))


class TestCalibration:
    def test_held_out_coverage(self, quadratic_data):
        """On a held-out set from the same distribution, 95% coverage
        should be roughly 0.95 (allow generous tolerance for small n)."""
        rng = np.random.default_rng(1)
        n = len(quadratic_data.X)
        idx = rng.permutation(n)
        cut = int(0.7 * n)
        tr, te = idx[:cut], idx[cut:]

        cfg = EnsembleConfig(
            kernel_families=("matern",),
            matern_nu_values=(2.5,),
            n_restarts_optimizer=3,
            add_polynomial_fallback=False,
            random_state=0,
        )
        model = BayesianEnsembleGP(cfg).fit(
            quadratic_data.X[tr], quadratic_data.y[tr]
        )
        pred, std = model.predict(quadratic_data.X[te])
        cov = coverage_at(quadratic_data.y[te], pred, std, 0.95)
        # Loose tolerance for small held-out sample
        assert 0.7 < cov <= 1.0


class TestSampling:
    def test_sample_y_shape(self, fitted_model):
        X = np.array([[0.0], [1.0], [2.0]])
        samples = fitted_model.sample_y(X, n_samples=5, random_state=0)
        assert samples.shape == (5, 3)

    def test_sample_y_deterministic(self, fitted_model):
        X = np.array([[0.0]])
        s1 = fitted_model.sample_y(X, n_samples=3, random_state=42)
        s2 = fitted_model.sample_y(X, n_samples=3, random_state=42)
        np.testing.assert_allclose(s1, s2)


class TestPersistence:
    def test_save_load_roundtrip(self, fitted_model, quadratic_data):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            fitted_model.save(path)
            assert path.exists()

            loaded = BayesianEnsembleGP.load(path)
            mean_orig, std_orig = fitted_model.predict(quadratic_data.X[:5])
            mean_load, std_load = loaded.predict(quadratic_data.X[:5])
            np.testing.assert_allclose(mean_orig, mean_load, rtol=1e-12)
            np.testing.assert_allclose(std_orig, std_load, rtol=1e-12)


class TestRobustnessToOutliers:
    def test_handles_outliers(self):
        """With 5% outliers, ensemble should still produce reasonable R^2."""
        data = make_quadratic(
            n=100, outlier_fraction=0.05, outlier_scale=10.0, seed=0
        )
        cfg = EnsembleConfig(
            kernel_families=("matern",),
            matern_nu_values=(2.5,),
            scaler="robust",
            n_restarts_optimizer=2,
            random_state=0,
        )
        model = BayesianEnsembleGP(cfg).fit(data.X, data.y)
        pred, _ = model.predict(data.X)
        # True signal is x^2; relax R^2 because outliers add variance
        true_signal = data.X.ravel() ** 2
        r2_clean = regression_metrics(true_signal, pred)["r2"]
        assert r2_clean > 0.9


class TestMultiDimensional:
    def test_2d_input(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(-1, 1, size=(60, 2))
        y = X[:, 0] ** 2 + 0.5 * X[:, 1] + rng.normal(0, 0.05, size=60)
        cfg = EnsembleConfig(
            kernel_families=("rbf",),
            n_restarts_optimizer=2,
            random_state=0,
            add_polynomial_fallback=False,
        )
        model = BayesianEnsembleGP(cfg).fit(X, y)
        pred, std = model.predict(X)
        assert pred.shape == (60,)
        assert std.shape == (60,)
        assert regression_metrics(y, pred)["r2"] > 0.9


class TestAllModelsFailGracefully:
    def test_uniform_weights_when_all_fail(self):
        """Construct an ensemble and force LMLs to -inf, check fallback."""
        cfg = EnsembleConfig(
            kernel_families=("rbf",),
            n_restarts_optimizer=1,
            random_state=0,
        )
        model = BayesianEnsembleGP(cfg)
        # Test the static method directly
        lmls = np.array([-np.inf, -np.inf])
        weights = model._softmax_weights(lmls)
        np.testing.assert_allclose(weights, [0.5, 0.5])
