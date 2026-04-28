"""
bayesgp.metrics
===============

Standard pointwise regression metrics plus calibration diagnostics for
probabilistic predictions:

    - r2, rmse, mae          (point-prediction quality)
    - nll                    (mean negative log-likelihood under the
                              predicted Gaussian)
    - coverage_at            (empirical coverage of central predictive
                              intervals; should match the nominal level
                              for a well-calibrated model)
    - pit_values             (Probability Integral Transform values, useful
                              for visual calibration diagnostics)
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

__all__ = [
    "regression_metrics",
    "coverage_at",
    "pit_values",
    "calibration_summary",
]

_LOG_2PI = math.log(2.0 * math.pi)
_EPS = 1e-12


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray | None = None,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Compute point and (optionally) probabilistic regression metrics.

    Parameters
    ----------
    y_true : array of shape (n,)
    y_pred : array of shape (n,)
    y_std : array of shape (n,), optional
        Predictive standard deviation. If provided, NLL and 95% coverage
        are reported in addition to point metrics.
    prefix : str
        String prepended to each metric key; useful when collating metrics
        from several splits.

    Returns
    -------
    dict
        Metric name to value.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    out: Dict[str, float] = {
        f"{prefix}r2": float(r2_score(y_true, y_pred)),
        f"{prefix}rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}mae": float(mean_absolute_error(y_true, y_pred)),
    }

    if y_std is not None:
        s = np.maximum(np.asarray(y_std).ravel(), _EPS)
        z = (y_true - y_pred) / s
        nll = 0.5 * np.mean(z**2 + _LOG_2PI + 2.0 * np.log(s))
        out[f"{prefix}nll"] = float(nll)
        out[f"{prefix}coverage_95"] = float(coverage_at(y_true, y_pred, s, 0.95))

    return out


def coverage_at(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    level: float,
) -> float:
    """
    Empirical coverage of the central ``level``-credible interval under the
    predicted Gaussian.

    A perfectly calibrated model produces coverage equal to ``level``.
    Returns a value in [0, 1].
    """
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level}")
    z = norm.ppf(0.5 + level / 2)
    s = np.maximum(np.asarray(y_std).ravel(), _EPS)
    return float(np.mean(np.abs(y_true - y_pred) <= z * s))


def pit_values(
    y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray
) -> np.ndarray:
    """
    Probability Integral Transform values F(y_true) under the predictive
    Gaussian N(y_pred, y_std^2).

    Under perfect calibration these values are uniform on [0, 1]. Plot a
    histogram to detect over- or under-dispersion.
    """
    s = np.maximum(np.asarray(y_std).ravel(), _EPS)
    return norm.cdf(y_true, loc=y_pred, scale=s)


def calibration_summary(
    y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray
) -> Dict[str, float]:
    """
    Coverage at several nominal levels. Useful for diagnostic tables.
    """
    return {
        f"coverage_{int(p * 100)}": coverage_at(y_true, y_pred, y_std, p)
        for p in (0.50, 0.68, 0.90, 0.95, 0.99)
    }
