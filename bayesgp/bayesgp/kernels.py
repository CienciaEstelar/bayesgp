"""
bayesgp.kernels
===============

Custom kernels and a safe expression parser for kernel construction.
"""

from __future__ import annotations

import re
from typing import Callable, Dict

import numpy as np
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Kernel,
    Matern,
    RationalQuadratic,
    RBF,
    WhiteKernel,
)

__all__ = [
    "PolynomialKernel",
    "KERNEL_REGISTRY",
    "parse_kernel_expression",
]


class PolynomialKernel(Kernel):
    """
    Polynomial kernel  K(x, y) = (x . y + c)^degree.

    Non-stationary; useful as an extrapolation fallback when stationary
    kernels (RBF, Matern) saturate to a constant beyond the training range.

    Parameters
    ----------
    degree : int
        Polynomial degree. Must be >= 0. Default 2.
    c : float
        Bias term inside the polynomial. Default 1.0.
    """

    def __init__(self, degree: int = 2, c: float = 1.0):
        if degree < 0:
            raise ValueError(f"degree must be >= 0, got {degree}")
        self.degree = int(degree)
        self.c = float(c)

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        Y = X if Y is None else np.atleast_2d(Y)
        K = (X @ Y.T + self.c) ** self.degree
        if eval_gradient:
            return K, np.empty((X.shape[0], X.shape[0], 0))
        return K

    def diag(self, X):
        X = np.atleast_2d(X)
        return (np.einsum("ij,ij->i", X, X) + self.c) ** self.degree

    def is_stationary(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"Polynomial(degree={self.degree}, c={self.c})"


KERNEL_REGISTRY: Dict[str, Callable[..., Kernel]] = {
    "RBF": RBF,
    "Matern": Matern,
    "RationalQuadratic": RationalQuadratic,
    "RQ": RationalQuadratic,
    "ExpSineSquared": ExpSineSquared,
    "WhiteKernel": WhiteKernel,
    "White": WhiteKernel,
    "ConstantKernel": ConstantKernel,
    "Constant": ConstantKernel,
    "DotProduct": DotProduct,
    "Polynomial": PolynomialKernel,
}

_FORBIDDEN_CHARS = re.compile(r"[^\w\s\+\-\*\/\(\),.:=eE0-9]")
_TOKEN_RE = re.compile(r"(?P<kernel_id>[A-Za-z][A-Za-z0-9_]*)\s*(?:\((?P<call_args>[^()]*)\))?")


def parse_kernel_expression(expr: str) -> Kernel:
    """
    Parse a user-supplied kernel expression into a scikit-learn Kernel object.

    Only kernels listed in ``KERNEL_REGISTRY`` are accessible. The expression
    is rewritten to look up names in the registry rather than evaluating them
    as free variables, so arbitrary code execution is not possible even if
    ``eval`` is used internally.

    Parameters
    ----------
    expr : str
        Expression like
        ``"ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5)"``.

    Returns
    -------
    Kernel

    Raises
    ------
    ValueError
        If the expression contains forbidden characters or unknown names,
        or if evaluation does not produce a Kernel instance.
    """
    if not expr or not expr.strip():
        raise ValueError("Empty kernel expression")
    if _FORBIDDEN_CHARS.search(expr):
        raise ValueError(f"Forbidden characters in expression: {expr!r}")

    def rewrite(match: "re.Match[str]") -> str:
        name = match.group("kernel_id")
        args = match.group("call_args") or ""
        if name not in KERNEL_REGISTRY:
            raise ValueError(f"Unknown kernel: {name!r}")
        return f'KERNEL_REGISTRY["{name}"]({args})'

    rewritten = _TOKEN_RE.sub(rewrite, expr)
    try:
        # Restricted globals: only the registry. No __builtins__ are exposed.
        result = eval(  # noqa: S307 - whitelist-controlled
            rewritten,
            {"__builtins__": {}, "KERNEL_REGISTRY": KERNEL_REGISTRY},
            {},
        )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to evaluate kernel expression {expr!r}: {exc}") from exc

    if not isinstance(result, Kernel):
        raise TypeError(f"Expression did not produce a Kernel: {result!r}")
    return result
