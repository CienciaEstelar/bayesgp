"""
Basic usage of BayesianEnsembleGP on a 1D quadratic with noise.

Demonstrates fitting, prediction with uncertainty, and posterior sampling.

Run:
    python examples/basic_quadratic.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from bayesgp import (
    BayesianEnsembleGP,
    EnsembleConfig,
    make_quadratic,
    regression_metrics,
)


def main() -> None:
    # Synthetic data: y = x^2 + small noise on [-3, 3]
    data = make_quadratic(n=80, x_min=-3.0, x_max=3.0, noise_std=0.1, seed=0)

    # Fit the default ensemble
    cfg = EnsembleConfig(n_restarts_optimizer=5, random_state=0)
    model = BayesianEnsembleGP(cfg).fit(data.X, data.y)

    print("BMA weights:")
    for name, w in zip(model.model_names_, model.weights_):
        print(f"  {name:<25s} {w:.4f}")

    # Predict on a fine grid (extending past training range to see uncertainty grow)
    X_grid = np.linspace(-5, 5, 400).reshape(-1, 1)
    mean, std = model.predict(X_grid)

    # Quality metrics on the training set
    pred_tr, _ = model.predict(data.X)
    metrics = regression_metrics(data.y, pred_tr, prefix="train_")
    print("\nTraining metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Posterior samples (proper mixture sampling)
    samples = model.sample_y(X_grid, n_samples=20, random_state=42)

    fig, ax = plt.subplots(figsize=(8, 5))
    for s in samples:
        ax.plot(X_grid, s, color="steelblue", alpha=0.15, linewidth=1)
    ax.fill_between(
        X_grid.ravel(),
        mean - 1.96 * std,
        mean + 1.96 * std,
        color="steelblue",
        alpha=0.25,
        label="95% credible band",
    )
    ax.plot(X_grid, mean, color="steelblue", linewidth=2, label="BMA mean")
    ax.plot(X_grid, X_grid.ravel() ** 2, "k--", linewidth=1, label="Ground truth $x^2$")
    ax.scatter(data.X, data.y, color="black", s=12, label="Training data")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("BayesianEnsembleGP on quadratic with extrapolation")
    ax.legend()
    fig.tight_layout()
    fig.savefig("basic_quadratic.png", dpi=150)
    print("\nSaved basic_quadratic.png")


if __name__ == "__main__":
    main()
