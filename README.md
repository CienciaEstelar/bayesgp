# bayesgp

Bayesian Model Averaging over a portfolio of Gaussian Process regressors, with calibrated uncertainty quantification.

## What this does

`bayesgp` fits several Gaussian Process regressors with different kernels to the same data, then combines their predictions using log-marginal-likelihood weights. This is the standard Bayesian Model Averaging recipe (Hoeting et al. 1999) applied to GPs.

Why bother:

- When the smoothness of the underlying function is unknown, single-kernel GPs can be miscalibrated. The ensemble adapts automatically.
- BMA weights are a clean diagnostic: if one kernel dominates with weight ≈ 1, the data clearly favours it; if weights spread, the data is ambiguous and the wider mixture variance reflects that honestly.
- The mixture predictive variance is computed correctly via the law of total variance (within-model + between-model), not as a naive average of per-model variances.

## What this does *not* do

- It does not learn the kernel structure (cf. Duvenaud's Automatic Statistician). The portfolio is fixed at construction.
- It does not implement variational or sparse GPs. Each base model is exact and scales as `O(n^3)`. For `n > ~5000` use a different library.
- It does not solve any open problem in physics, cosmology, or finance.

## Installation

```bash
git clone https://github.com/CienciaEstelar/bayesgp
cd bayesgp
pip install -e .
```

Dependencies: `numpy`, `scipy`, `scikit-learn`, `joblib`. Optional: `matplotlib`, `pytest`.

## Quick start

```python
import numpy as np
from bayesgp import BayesianEnsembleGP, EnsembleConfig, make_quadratic

data = make_quadratic(n=80, seed=0)
model = BayesianEnsembleGP(EnsembleConfig(random_state=0)).fit(data.X, data.y)

X_new = np.linspace(-5, 5, 100).reshape(-1, 1)
mean, std = model.predict(X_new)

# Inspect which kernels the data prefers
for name, w in zip(model.model_names_, model.weights_):
    print(f"{name:25s} {w:.3f}")
```

## Public API

| Symbol | Purpose |
|---|---|
| `BayesianEnsembleGP` | Main estimator. `fit`, `predict`, `sample_y`, `save`, `load`. |
| `EnsembleConfig` | Configuration dataclass: kernel families, scaler, parallelism, seed. |
| `PolynomialKernel` | Non-stationary kernel for extrapolation. |
| `parse_kernel_expression` | Whitelist-based parser for user-supplied kernel expressions. |
| `regression_metrics`, `coverage_at`, `pit_values`, `calibration_summary` | Probabilistic regression diagnostics. |
| `make_quadratic`, `make_oscillatory`, `make_step_with_noise` | Synthetic datasets. |

## Tests

```bash
pytest tests/ -v
```

43 unit tests covering kernel construction, expression parsing security, BMA weight stability, mixture variance correctness, calibration diagnostics, persistence, multi-dimensional inputs, and graceful failure modes.

## Benchmarks

```bash
python benchmarks/run_benchmark.py
```

Compares `bayesgp` against single-kernel sklearn baselines (`RBF`, `Matern-2.5`) on four synthetic datasets. Reports RMSE, NLL, and 95%-coverage. Headline result: the ensemble matches the best single kernel on smooth problems and improves NLL on problems with mixed regularity (step functions).

## References

- Hoeting, J. A., Madigan, D., Raftery, A. E., & Volinsky, C. T. (1999). *Bayesian Model Averaging: A Tutorial.* Statistical Science, **14**(4), 382–417.
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning.* MIT Press.
- Duvenaud, D. (2014). *Automatic Model Construction with Gaussian Processes.* PhD thesis, Cambridge.

## License

MIT.

## Citation

If you use this code, please cite it via the Zenodo DOI in `CITATION.cff`.
