"""
Benchmark: ``BayesianEnsembleGP`` vs single-kernel sklearn ``GaussianProcessRegressor``.

For each dataset we compare:
    - point-prediction quality (RMSE)
    - calibration (95% coverage; nominal value is 0.95)
    - log-likelihood quality (NLL; lower is better)

The single-kernel baselines are chosen to be the most common defaults
(RBF and Matern-2.5). The ensemble is the bayesgp default (Matern with
nu in {0.5, 1.5, 2.5} + RBF + RationalQuadratic + Polynomial fallback).

Run:
    python benchmarks/run_benchmark.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    RBF,
    WhiteKernel,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from bayesgp import (
    BayesianEnsembleGP,
    EnsembleConfig,
    SyntheticDataset,
    make_oscillatory,
    make_quadratic,
    make_step_with_noise,
    regression_metrics,
)


@dataclass
class BenchmarkRow:
    dataset: str
    method: str
    rmse: float
    nll: float
    coverage_95: float


# --------------------------------------------------------------------------- #
# Sklearn single-kernel baseline: standard pipeline matching what bayesgp does
# (external scaling), so the comparison is fair.
# --------------------------------------------------------------------------- #
def _fit_single_kernel(main_kernel, X_train, y_train):
    full_kernel = (
        ConstantKernel(1.0, (1e-4, 1e4)) * main_kernel
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-9, 1e-1))
    )
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    Xs = x_scaler.transform(X_train)
    ys = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
    gp = GaussianProcessRegressor(
        kernel=full_kernel, n_restarts_optimizer=5, alpha=1e-8, random_state=0
    )
    gp.fit(Xs, ys)

    def predict(X):
        Xs = x_scaler.transform(X)
        mu_s, sigma_s = gp.predict(Xs, return_std=True)
        mu = y_scaler.inverse_transform(mu_s.reshape(-1, 1)).ravel()
        sigma = sigma_s * float(y_scaler.scale_[0])
        return mu, sigma

    return predict


def _evaluate(predict_fn: Callable, X_test, y_test) -> dict:
    pred, std = predict_fn(X_test)
    return regression_metrics(y_test, pred, std)


def run_one(dataset: SyntheticDataset, seed: int = 0) -> list[BenchmarkRow]:
    X_tr, X_te, y_tr, y_te = train_test_split(
        dataset.X, dataset.y, test_size=0.3, random_state=seed
    )

    rows: list[BenchmarkRow] = []

    # Baselines
    for name, kernel in [
        ("sklearn_RBF", RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))),
        (
            "sklearn_Matern25",
            Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5),
        ),
    ]:
        pred = _fit_single_kernel(kernel, X_tr, y_tr)
        m = _evaluate(pred, X_te, y_te)
        rows.append(
            BenchmarkRow(dataset.name, name, m["rmse"], m["nll"], m["coverage_95"])
        )

    # bayesgp default ensemble
    cfg = EnsembleConfig(n_restarts_optimizer=5, random_state=0)
    bgp = BayesianEnsembleGP(cfg).fit(X_tr, y_tr)
    pred_fn = lambda X: bgp.predict(X)
    m = _evaluate(pred_fn, X_te, y_te)
    rows.append(
        BenchmarkRow(dataset.name, "bayesgp_ensemble", m["rmse"], m["nll"], m["coverage_95"])
    )

    return rows


def main() -> None:
    datasets = [
        make_quadratic(n=120, seed=0),
        make_quadratic(n=120, seed=0, outlier_fraction=0.05, outlier_scale=10.0),
        make_oscillatory(n=120, seed=0),
        make_step_with_noise(n=120, seed=0),
    ]
    # rename the contaminated quadratic for readability
    datasets[1] = SyntheticDataset(
        X=datasets[1].X,
        y=datasets[1].y,
        name="quadratic_outliers",
        description=datasets[1].description,
    )

    all_rows: list[BenchmarkRow] = []
    for ds in datasets:
        all_rows.extend(run_one(ds))

    # Pretty print
    header = f"{'dataset':<22} {'method':<22} {'RMSE':>10} {'NLL':>10} {'Cov95':>8}"
    print(header)
    print("-" * len(header))
    for r in all_rows:
        print(
            f"{r.dataset:<22} {r.method:<22} "
            f"{r.rmse:>10.4f} {r.nll:>10.3f} {r.coverage_95:>8.2f}"
        )


if __name__ == "__main__":
    main()
