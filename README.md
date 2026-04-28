<div align="center">

# bayesgp

**Bayesian Model Averaging over a portfolio of Gaussian Process regressors, with calibrated uncertainty quantification.**

[![tests](https://github.com/CienciaEstelar/bayesgp/actions/workflows/tests.yml/badge.svg)](https://github.com/CienciaEstelar/bayesgp/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

🇬🇧 [**English**](#english) &nbsp;·&nbsp; 🇨🇱🇨🇱 [**Español**](#español)

</div>

---

<a id="english"></a>

# 🇬🇧 English

## What this does

`bayesgp` fits several Gaussian Process regressors with different kernels to the same data, then combines their predictions using log-marginal-likelihood weights. This is the standard Bayesian Model Averaging recipe (Hoeting et al. 1999) applied to GPs.

**Why bother:**

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

<br>

<div align="center">

⸻ &nbsp;&nbsp; 🇨🇱🇨🇱 [Versión en Español](#español) &nbsp;&nbsp; ⸻

</div>

---

<a id="español"></a>

# 🇪🇸 Español

**Promediado Bayesiano de Modelos sobre un portafolio de regresores de Procesos Gaussianos, con cuantificación de incertidumbre calibrada.**

## Qué hace

`bayesgp` ajusta varios regresores de Procesos Gaussianos con kernels distintos sobre los mismos datos, y luego combina sus predicciones usando pesos derivados de la log-verosimilitud marginal. Esta es la receta estándar de Bayesian Model Averaging (Hoeting et al. 1999) aplicada a GPs.

**Por qué importa:**

- Cuando la suavidad de la función subyacente es desconocida, un GP de kernel único puede estar mal calibrado. El ensamble se adapta automáticamente.
- Los pesos BMA son un diagnóstico limpio: si un kernel domina con peso ≈ 1, los datos lo prefieren claramente; si los pesos se distribuyen, los datos son ambiguos y la varianza de mezcla más amplia refleja eso honestamente.
- La varianza predictiva de la mezcla se calcula correctamente mediante la ley de la varianza total (intra-modelo + inter-modelo), no como un promedio ingenuo de las varianzas individuales.

## Qué *no* hace

- No aprende la estructura del kernel (cf. Automatic Statistician de Duvenaud). El portafolio se fija al construir el modelo.
- No implementa GPs variacionales ni sparse. Cada modelo base es exacto y escala como `O(n^3)`. Para `n > ~5000` usar otra librería.
- No resuelve ningún problema abierto en física, cosmología ni finanzas.

## Instalación

```bash
git clone https://github.com/CienciaEstelar/bayesgp
cd bayesgp
pip install -e .
```

Dependencias: `numpy`, `scipy`, `scikit-learn`, `joblib`. Opcionales: `matplotlib`, `pytest`.

## Inicio rápido

```python
import numpy as np
from bayesgp import BayesianEnsembleGP, EnsembleConfig, make_quadratic

datos = make_quadratic(n=80, seed=0)
modelo = BayesianEnsembleGP(EnsembleConfig(random_state=0)).fit(datos.X, datos.y)

X_nuevo = np.linspace(-5, 5, 100).reshape(-1, 1)
media, std = modelo.predict(X_nuevo)

# Inspeccionar qué kernels prefieren los datos
for nombre, w in zip(modelo.model_names_, modelo.weights_):
    print(f"{nombre:25s} {w:.3f}")
```

## API pública

| Símbolo | Propósito |
|---|---|
| `BayesianEnsembleGP` | Estimador principal. `fit`, `predict`, `sample_y`, `save`, `load`. |
| `EnsembleConfig` | Dataclass de configuración: familias de kernels, escalado, paralelismo, semilla. |
| `PolynomialKernel` | Kernel no estacionario para extrapolación. |
| `parse_kernel_expression` | Parser con lista blanca para expresiones de kernels suministradas por el usuario. |
| `regression_metrics`, `coverage_at`, `pit_values`, `calibration_summary` | Diagnósticos de regresión probabilística. |
| `make_quadratic`, `make_oscillatory`, `make_step_with_noise` | Datasets sintéticos. |

## Tests

```bash
pytest tests/ -v
```

43 tests unitarios que cubren construcción de kernels, seguridad del parser de expresiones, estabilidad de los pesos BMA, correctitud de la varianza de mezcla, diagnósticos de calibración, serialización, entradas multi-dimensionales, y modos de fallo manejados.

## Benchmarks

```bash
python benchmarks/run_benchmark.py
```

Compara `bayesgp` contra baselines de sklearn de kernel único (`RBF`, `Matern-2.5`) sobre cuatro datasets sintéticos. Reporta RMSE, NLL y cobertura al 95%. Resultado principal: el ensamble iguala al mejor kernel único en problemas suaves y mejora NLL en problemas con regularidad mixta (funciones escalón).

## Referencias

- Hoeting, J. A., Madigan, D., Raftery, A. E., & Volinsky, C. T. (1999). *Bayesian Model Averaging: A Tutorial.* Statistical Science, **14**(4), 382–417.
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning.* MIT Press.
- Duvenaud, D. (2014). *Automatic Model Construction with Gaussian Processes.* Tesis doctoral, Cambridge.

## Licencia

MIT.

## Citación

Si usas este código, por favor cítalo mediante el DOI de Zenodo en `CITATION.cff`.

<br>

<div align="center">

⸻ &nbsp;&nbsp; 🇬🇧 [Back to English](#english) &nbsp;&nbsp; ⸻

<br>

Made with rigor by [@CienciaEstelar](https://github.com/CienciaEstelar)

</div>
