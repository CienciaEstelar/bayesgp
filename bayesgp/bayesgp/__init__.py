"""
bayesgp
=======

Bayesian Model Averaging over a portfolio of Gaussian Process regressors,
with calibrated uncertainty quantification.

Public API
----------
- ``BayesianEnsembleGP`` and ``EnsembleConfig``: the model.
- ``regression_metrics``, ``coverage_at``, ``pit_values``: diagnostics.
- ``make_quadratic``, ``make_oscillatory``, ``make_step_with_noise``:
  synthetic datasets.
- ``PolynomialKernel``, ``parse_kernel_expression``: custom kernels.
"""

from bayesgp.datasets import (
    SyntheticDataset,
    make_oscillatory,
    make_quadratic,
    make_step_with_noise,
)
from bayesgp.kernels import (
    KERNEL_REGISTRY,
    PolynomialKernel,
    parse_kernel_expression,
)
from bayesgp.metrics import (
    calibration_summary,
    coverage_at,
    pit_values,
    regression_metrics,
)
from bayesgp.model import BayesianEnsembleGP, EnsembleConfig

__version__ = "0.1.0"

__all__ = [
    "BayesianEnsembleGP",
    "EnsembleConfig",
    "PolynomialKernel",
    "KERNEL_REGISTRY",
    "parse_kernel_expression",
    "SyntheticDataset",
    "make_quadratic",
    "make_oscillatory",
    "make_step_with_noise",
    "regression_metrics",
    "coverage_at",
    "pit_values",
    "calibration_summary",
    "__version__",
]
