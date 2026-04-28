"""
bayesgp.datasets
================

Synthetic datasets used by examples, tests, and benchmarks.

These are generic regression problems chosen for their well-known shape;
they do not depend on any specific physical interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SyntheticDataset",
    "make_quadratic",
    "make_oscillatory",
    "make_step_with_noise",
]


@dataclass(frozen=True)
class SyntheticDataset:
    """A synthetic regression dataset with optional ground-truth function."""

    X: np.ndarray
    y: np.ndarray
    name: str
    description: str = ""

    def __repr__(self) -> str:
        return (
            f"SyntheticDataset(name={self.name!r}, "
            f"n={len(self.X)}, n_features={self.X.shape[1]})"
        )


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_quadratic(
    n: int = 200,
    x_min: float = -3.0,
    x_max: float = 3.0,
    a: float = 1.0,
    noise_std: float = 0.05,
    outlier_fraction: float = 0.0,
    outlier_scale: float = 20.0,
    seed: int | None = 0,
) -> SyntheticDataset:
    """
    1D quadratic ``y = a * x^2 + noise``.

    Optionally contaminates a fraction of the targets with large-amplitude
    Gaussian outliers to test robustness.
    """
    rng = _rng(seed)
    X = np.linspace(x_min, x_max, n).reshape(-1, 1)
    y = a * X.ravel() ** 2 + rng.normal(0.0, noise_std, size=n)
    if outlier_fraction > 0:
        k = int(outlier_fraction * n)
        idx = rng.choice(n, size=k, replace=False)
        y[idx] += rng.normal(0.0, outlier_scale * noise_std, size=k)
    return SyntheticDataset(
        X=X,
        y=y,
        name="quadratic",
        description=f"y = {a}*x^2 + N(0, {noise_std}^2)",
    )


def make_oscillatory(
    n: int = 200,
    x_min: float = 0.0,
    x_max: float = 10.0,
    noise_std: float = 0.1,
    seed: int | None = 0,
) -> SyntheticDataset:
    """1D oscillatory ``y = sin(x) + 0.5*sin(3x) + noise``."""
    rng = _rng(seed)
    X = np.linspace(x_min, x_max, n).reshape(-1, 1)
    y = (
        np.sin(X.ravel())
        + 0.5 * np.sin(3.0 * X.ravel())
        + rng.normal(0.0, noise_std, size=n)
    )
    return SyntheticDataset(
        X=X,
        y=y,
        name="oscillatory",
        description="y = sin(x) + 0.5 sin(3x) + noise",
    )


def make_step_with_noise(
    n: int = 200,
    x_min: float = -2.0,
    x_max: float = 2.0,
    noise_std: float = 0.05,
    seed: int | None = 0,
) -> SyntheticDataset:
    """
    Discontinuous step function. Useful for stress-testing smooth GP priors,
    which will typically over-smooth this signal.
    """
    rng = _rng(seed)
    X = np.linspace(x_min, x_max, n).reshape(-1, 1)
    y = np.where(X.ravel() < 0, -1.0, 1.0) + rng.normal(0.0, noise_std, size=n)
    return SyntheticDataset(
        X=X, y=y, name="step", description="step at x=0, additive noise"
    )
