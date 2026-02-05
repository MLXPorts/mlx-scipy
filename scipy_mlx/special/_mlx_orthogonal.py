"""
Small MLX implementations for orthogonal-polynomial utilities used by SciPy.

This module exists to avoid importing SciPy's full `_orthogonal` machinery
(which depends on many special functions) while still providing the Gauss
quadrature primitives needed by `scipy_mlx.integrate`.
"""

from __future__ import annotations

import mlx.core as mx


def roots_legendre(n: int, mu: bool = False):
    """
    Gauss-Legendre quadrature.

    Parameters
    ----------
    n : int
        Number of quadrature points.
    mu : bool, optional
        If True, also return the zeroth moment of the weight function (2.0).

    Returns
    -------
    x : mx.array
        Quadrature nodes on [-1, 1], shape (n,).
    w : mx.array
        Quadrature weights, shape (n,).
    mu0 : mx.array, optional
        Zeroth moment (2.0) if `mu=True`.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")

    # Golub-Welsch for Legendre: build symmetric tridiagonal Jacobi matrix.
    k = mx.arange(1, n, dtype=mx.float32)
    four = mx.array(4.0)
    one = mx.array(1.0)
    beta = mx.divide(k, mx.sqrt(mx.subtract(mx.multiply(four, mx.multiply(k, k)), one)))
    J = mx.add(mx.diag(beta, 1), mx.diag(beta, -1))

    # MLX linalg is CPU-only in many builds; run explicitly on CPU stream.
    x, V = mx.linalg.eigh(J, stream=mx.cpu)
    v0 = V[0]
    two = mx.array(2.0)
    w = mx.multiply(two, mx.multiply(v0, v0))

    if mu:
        return x, w, two
    return x, w

