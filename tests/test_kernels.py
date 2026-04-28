"""Tests for bayesgp.kernels."""

import numpy as np
import pytest

from bayesgp.kernels import KERNEL_REGISTRY, PolynomialKernel, parse_kernel_expression


class TestPolynomialKernel:
    def test_evaluation_shape(self):
        k = PolynomialKernel(degree=2, c=1.0)
        X = np.array([[1.0], [2.0], [3.0]])
        K = k(X)
        assert K.shape == (3, 3)
        # K(1, 1) = (1*1 + 1)^2 = 4
        assert K[0, 0] == pytest.approx(4.0)
        # K(2, 3) = (2*3 + 1)^2 = 49
        assert K[1, 2] == pytest.approx(49.0)

    def test_diag_consistent(self):
        k = PolynomialKernel(degree=3, c=0.5)
        X = np.random.default_rng(0).normal(size=(10, 2))
        diag = k.diag(X)
        K = k(X)
        np.testing.assert_allclose(diag, np.diag(K), rtol=1e-12)

    def test_non_stationary(self):
        assert not PolynomialKernel().is_stationary()

    def test_negative_degree_raises(self):
        with pytest.raises(ValueError):
            PolynomialKernel(degree=-1)

    def test_eval_gradient_returns_empty(self):
        k = PolynomialKernel(degree=2)
        X = np.array([[1.0], [2.0]])
        K, grad = k(X, eval_gradient=True)
        assert K.shape == (2, 2)
        assert grad.shape == (2, 2, 0)


class TestParseKernelExpression:
    def test_basic_rbf(self):
        k = parse_kernel_expression("RBF(length_scale=1.0)")
        from sklearn.gaussian_process.kernels import RBF
        assert isinstance(k, RBF)

    def test_composition(self):
        expr = "ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.01)"
        k = parse_kernel_expression(expr)
        # Sum kernels are produced by sklearn; just check it's a Kernel
        from sklearn.gaussian_process.kernels import Kernel
        assert isinstance(k, Kernel)

    def test_unknown_kernel_raises(self):
        with pytest.raises(ValueError, match="Unknown kernel"):
            parse_kernel_expression("MalwareKernel(1.0)")

    def test_forbidden_chars_raise(self):
        with pytest.raises(ValueError, match="Forbidden characters"):
            parse_kernel_expression("RBF(1.0); import os")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_kernel_expression("")

    def test_no_arbitrary_code_execution(self):
        """The parser must not allow access to builtins."""
        with pytest.raises((ValueError, NameError)):
            parse_kernel_expression("__import__('os')")
