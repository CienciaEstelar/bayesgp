"""
Calibration diagnostics for BayesianEnsembleGP.

Plots:
    1. Coverage at multiple nominal levels (should follow the diagonal).
    2. Histogram of PIT values (should be uniform under perfect calibration).

Run:
    python examples/calibration_diagnostics.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from bayesgp import (
    BayesianEnsembleGP,
    EnsembleConfig,
    coverage_at,
    make_oscillatory,
    pit_values,
)


def main() -> None:
    data = make_oscillatory(n=200, noise_std=0.15, seed=0)
    X_tr, X_te, y_tr, y_te = train_test_split(
        data.X, data.y, test_size=0.4, random_state=0
    )

    model = BayesianEnsembleGP(
        EnsembleConfig(n_restarts_optimizer=5, random_state=0)
    ).fit(X_tr, y_tr)

    pred, std = model.predict(X_te)

    nominal_levels = np.linspace(0.05, 0.99, 20)
    empirical = np.array([coverage_at(y_te, pred, std, p) for p in nominal_levels])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Calibration curve
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    axes[0].plot(nominal_levels, empirical, "o-", label="bayesgp ensemble")
    axes[0].set_xlabel("Nominal coverage")
    axes[0].set_ylabel("Empirical coverage")
    axes[0].set_title("Calibration curve (held-out set)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # PIT histogram
    pits = pit_values(y_te, pred, std)
    axes[1].hist(pits, bins=15, edgecolor="black", density=True, color="steelblue")
    axes[1].axhline(1.0, color="k", linestyle="--", linewidth=1, label="uniform target")
    axes[1].set_xlabel("PIT value")
    axes[1].set_ylabel("Density")
    axes[1].set_title("PIT histogram")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig("calibration_diagnostics.png", dpi=150)
    print("Saved calibration_diagnostics.png")


if __name__ == "__main__":
    main()
