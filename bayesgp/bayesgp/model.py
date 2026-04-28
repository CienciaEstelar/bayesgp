"""
bayesgp.model
=============

Bayesian Model Averaging over a portfolio of Gaussian Process regressors.

Each candidate model is fitted independently to the training data; posterior
predictive moments are combined using log-marginal-likelihood weights.

This is the textbook BMA approach (Hoeting et al., 1999; Madigan & Raftery,
1994) restricted to GP base models. It is intended as a robust alternative
to selecting a single kernel a priori, particularly when the smoothness
properties of the underlying function are unknown.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Kernel,
    Matern,
    RationalQuadratic,
    RBF,
    WhiteKernel,
)
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.validation import check_array, check_X_y

from bayesgp.kernels import PolynomialKernel, parse_kernel_expression

__all__ = ["EnsembleConfig", "BayesianEnsembleGP"]

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass
class EnsembleConfig:
    """
    Configuration for ``BayesianEnsembleGP``.

    Parameters
    ----------
    kernel_families : iterable of str
        Names of stationary kernel families to include. Allowed values:
        ``"matern"``, ``"rbf"``, ``"rational_quadratic"``.
    matern_nu_values : iterable of float
        Smoothness parameters for the Matern family. One model per value.
    custom_expressions : iterable of str
        Optional user-supplied kernel expressions, parsed via
        ``bayesgp.kernels.parse_kernel_expression``.
    add_polynomial_fallback : bool
        If True, append a non-stationary polynomial kernel to the ensemble
        whenever all other kernels are stationary. This improves
        extrapolation behaviour at the cost of added flexibility.
    n_restarts_optimizer : int
        ``n_restarts_optimizer`` passed to each underlying GP.
    alpha : float
        Diagonal jitter added to the kernel matrix for numerical stability.
    scaler : {"standard", "robust"}
        Type of input/output rescaling.
    n_jobs : int
        Number of parallel workers for ensemble fitting. -1 uses all cores.
    random_state : int or None
        Seed used by every base GP for reproducibility.
    """

    kernel_families: Iterable[str] = field(
        default_factory=lambda: ("matern", "rbf", "rational_quadratic")
    )
    matern_nu_values: Iterable[float] = field(default_factory=lambda: (0.5, 1.5, 2.5))
    custom_expressions: Iterable[str] = field(default_factory=tuple)
    add_polynomial_fallback: bool = True
    n_restarts_optimizer: int = 10
    alpha: float = 1e-8
    scaler: str = "standard"
    n_jobs: int = 1
    random_state: int | None = 42


class BayesianEnsembleGP:
    """
    Bayesian Model Averaging over a portfolio of Gaussian Process regressors.

    Each candidate kernel produces an independent GP. After fitting, the
    log marginal likelihoods of the base models are converted into BMA
    weights via a numerically stable softmax. Predictions are computed as
    a weighted mixture of Gaussians, with predictive variance correctly
    decomposed into within-model and between-model components.

    Parameters
    ----------
    config : EnsembleConfig, optional
        Configuration object. A default instance is used if not supplied.

    Attributes
    ----------
    weights_ : numpy.ndarray of shape (n_models,)
        BMA posterior weights after fitting.
    log_marginal_likelihoods_ : numpy.ndarray of shape (n_models,)
        Log marginal likelihood of each base model.
    model_names_ : list of str
        Human-readable identifiers of each base model.

    References
    ----------
    Hoeting, J. A., Madigan, D., Raftery, A. E., & Volinsky, C. T. (1999).
    Bayesian Model Averaging: A Tutorial. Statistical Science, 14(4), 382.

    Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for
    Machine Learning. MIT Press.
    """

    def __init__(self, config: EnsembleConfig | None = None):
        self.config = config or EnsembleConfig()
        self._models: list[dict[str, Any]] = []
        self._is_fitted = False

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted; call .fit(X, y) first."
            )

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def _make_base_structure(self, main_kernel: Kernel) -> Kernel:
        """Standard wrapping: amplitude * main_kernel + white_noise."""
        return (
            ConstantKernel(1.0, (1e-4, 1e4)) * main_kernel
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-9, 1e-1))
        )

    def _build_models(self) -> list[dict[str, Any]]:
        models: list[dict[str, Any]] = []
        seed = self.config.random_state

        def add(kernel: Kernel, name: str) -> None:
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=self.config.n_restarts_optimizer,
                alpha=self.config.alpha,
                normalize_y=False,  # we scale externally
                random_state=seed,
            )
            models.append({"name": name, "kernel": kernel, "gp": gp})

        for fam in self.config.kernel_families:
            fam_low = fam.lower()
            if fam_low == "matern":
                for nu in self.config.matern_nu_values:
                    k = self._make_base_structure(
                        Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=nu)
                    )
                    add(k, f"matern_nu={nu}")
            elif fam_low == "rbf":
                k = self._make_base_structure(
                    RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
                )
                add(k, "rbf")
            elif fam_low in {"rational_quadratic", "rq"}:
                k = self._make_base_structure(
                    RationalQuadratic(
                        length_scale=1.0,
                        alpha=0.1,
                        alpha_bounds=(1e-2, 1e2),
                    )
                )
                add(k, "rational_quadratic")
            else:
                raise ValueError(f"Unknown kernel family: {fam!r}")

        for i, expr in enumerate(self.config.custom_expressions):
            k = parse_kernel_expression(expr)
            add(k, f"custom_{i}")

        if self.config.add_polynomial_fallback and all(
            m["kernel"].is_stationary() for m in models
        ):
            add(PolynomialKernel(degree=2, c=1.0), "polynomial_fallback")

        if not models:
            raise ValueError("No models constructed; check EnsembleConfig.")

        return models

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fit_single(model: dict[str, Any], X: np.ndarray, y: np.ndarray):
        """Fit one base GP and return (model, log_marginal_likelihood)."""
        try:
            model["gp"].fit(X, y)
            return model, model["gp"].log_marginal_likelihood()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Fit failed for %s: %s", model["name"], exc)
            return model, -np.inf

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianEnsembleGP":
        """
        Fit each base GP and compute BMA weights from log marginal likelihoods.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training inputs.
        y : array-like of shape (n_samples,) or (n_samples, 1)
            Training targets.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y, multi_output=False, ensure_2d=True)
        self._n_features_in_ = X.shape[1]

        if self.config.scaler == "standard":
            self._x_scaler = StandardScaler()
            self._y_scaler = StandardScaler()
        elif self.config.scaler == "robust":
            self._x_scaler = RobustScaler()
            self._y_scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler: {self.config.scaler!r}")

        Xs = self._x_scaler.fit_transform(X)
        ys = self._y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

        self._models = self._build_models()

        results = Parallel(n_jobs=self.config.n_jobs, backend="threading")(
            delayed(self._fit_single)(m, Xs, ys) for m in self._models
        )
        self._models = [r[0] for r in results]
        lmls = np.array([r[1] for r in results], dtype=float)

        self.log_marginal_likelihoods_ = lmls
        self.weights_ = self._softmax_weights(lmls)
        self.model_names_ = [m["name"] for m in self._models]

        self._is_fitted = True

        logger.info(
            "Fitted %d models. Top weights: %s",
            len(self._models),
            ", ".join(
                f"{n}={w:.3f}"
                for n, w in sorted(
                    zip(self.model_names_, self.weights_),
                    key=lambda t: -t[1],
                )[:3]
            ),
        )
        return self

    @staticmethod
    def _softmax_weights(lmls: np.ndarray) -> np.ndarray:
        """Numerically stable softmax with sensible fallback for all -inf."""
        finite = np.isfinite(lmls)
        if not finite.any():
            logger.warning("All base models failed; falling back to uniform weights.")
            return np.full(len(lmls), 1.0 / len(lmls))
        weights = np.zeros_like(lmls)
        max_finite = lmls[finite].max()
        weights[finite] = np.exp(lmls[finite] - max_finite)
        weights /= weights.sum()
        return weights

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict(
        self, X: np.ndarray, return_std: bool = True
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """
        Predict the BMA posterior mean and (optionally) standard deviation.

        The mixture variance is computed as
            Var[y*] = sum_k w_k * Var_k[y*] + sum_k w_k * (mu_k - mu_bar)^2
        i.e. the law of total variance applied to the model index as a
        latent categorical variable.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        return_std : bool

        Returns
        -------
        mean : array of shape (n_samples,)
        std : array of shape (n_samples,)   (only if return_std)
        """
        self._check_fitted()
        X = check_array(X, ensure_2d=True)
        if X.shape[1] != self._n_features_in_:
            raise ValueError(
                f"Expected {self._n_features_in_} features, got {X.shape[1]}"
            )

        Xs = self._x_scaler.transform(X)

        means = np.empty((len(self._models), Xs.shape[0]))
        variances = np.empty_like(means)
        for i, m in enumerate(self._models):
            mu, sigma = m["gp"].predict(Xs, return_std=True)
            means[i] = mu.ravel()
            variances[i] = sigma.ravel() ** 2

        w = self.weights_.reshape(-1, 1)
        mean_scaled = (w * means).sum(axis=0)

        if return_std:
            within = (w * variances).sum(axis=0)
            between = (w * (means - mean_scaled) ** 2).sum(axis=0)
            std_scaled = np.sqrt(within + between)
            y_scale = float(self._y_scaler.scale_[0])
            std = std_scaled * y_scale
        # back-transform mean
        mean = self._y_scaler.inverse_transform(mean_scaled.reshape(-1, 1)).ravel()

        if return_std:
            return mean, std
        return mean

    def sample_y(
        self, X: np.ndarray, n_samples: int = 10, random_state: int | None = None
    ) -> np.ndarray:
        """
        Draw samples from the BMA posterior predictive distribution.

        Implementation: choose model index ~ Categorical(weights_), then
        sample from that model's posterior. This gives proper mixture
        samples rather than Gaussian approximation samples.

        Returns
        -------
        samples : array of shape (n_samples, n_query)
        """
        self._check_fitted()
        X = check_array(X, ensure_2d=True)
        rng = np.random.default_rng(random_state)
        Xs = self._x_scaler.transform(X)
        chosen = rng.choice(len(self._models), size=n_samples, p=self.weights_)
        out = np.empty((n_samples, Xs.shape[0]))
        for s, idx in enumerate(chosen):
            sample_scaled = self._models[idx]["gp"].sample_y(
                Xs, n_samples=1, random_state=rng.integers(0, 2**31 - 1)
            ).ravel()
            out[s] = self._y_scaler.inverse_transform(
                sample_scaled.reshape(-1, 1)
            ).ravel()
        return out

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        """Persist the fitted ensemble using joblib."""
        self._check_fitted()
        joblib.dump(self, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "BayesianEnsembleGP":
        """Load a previously-saved ensemble."""
        obj = joblib.load(str(path))
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}: {type(obj)}")
        return obj
