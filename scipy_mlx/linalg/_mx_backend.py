"""
Minimal MLX-only linear algebra backend.

This is a fallback used when SciPy-style BLAS/LAPACK extension modules are not
available. It exposes a small subset of `scipy.linalg` using `mx.linalg`.
"""

from __future__ import annotations

from typing import Any, Tuple

import mlx.core as mx


class LinAlgError(RuntimeError):
    pass

class LinAlgWarning(UserWarning):
    pass

def _unimplemented(name: str):
    def _fn(*args, **kwargs):
        raise NotImplementedError(f"scipy_mlx.linalg.{name} is not yet implemented for MLX.")
    _fn.__name__ = name
    return _fn


def norm(x, ord=None, axis=None, keepdims=False):
    return mx.linalg.norm(mx.array(x), ord=ord, axis=axis, keepdims=keepdims)

def issymmetric(a, atol=0, rtol=1e-5):
    a = mx.array(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("expected a square 2D array")
    return mx.allclose(a, mx.transpose(a), atol=atol, rtol=rtol)


def ishermitian(a, atol=0, rtol=1e-5):
    a = mx.array(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("expected a square 2D array")
    ah = mx.transpose(mx.conjugate(a))
    return mx.allclose(a, ah, atol=atol, rtol=rtol)


def get_lapack_funcs(names, arrays=(), dtype=None):
    if isinstance(names, str):
        return _unimplemented(f"lapack.{names}")
    return tuple(_unimplemented(f"lapack.{n}") for n in names)


def get_blas_funcs(names, arrays=(), dtype=None):
    if isinstance(names, str):
        return _unimplemented(f"blas.{names}")
    return tuple(_unimplemented(f"blas.{n}") for n in names)


def solve_banded(*args, **kwargs):
    raise NotImplementedError("scipy_mlx.linalg.solve_banded is not yet implemented for MLX.")


def cholesky_banded(*args, **kwargs):
    raise NotImplementedError("scipy_mlx.linalg.cholesky_banded is not yet implemented for MLX.")


def cho_solve_banded(*args, **kwargs):
    raise NotImplementedError("scipy_mlx.linalg.cho_solve_banded is not yet implemented for MLX.")


def pinv(a, rcond=None):
    return mx.linalg.pinv(mx.array(a), stream=mx.cpu)


def toeplitz(c, r=None):
    c = mx.reshape(mx.array(c), (-1,))
    if r is None:
        r = mx.conjugate(c)
    r = mx.reshape(mx.array(r), (-1,))
    i = mx.expand_dims(mx.arange(int(c.shape[0])), axis=1)
    j = mx.expand_dims(mx.arange(int(r.shape[0])), axis=0)
    k = mx.subtract(j, i)
    mask = mx.greater_equal(k, mx.array(0))
    idx_r = mx.where(mask, k, mx.zeros_like(k))
    idx_c = mx.where(mask, mx.zeros_like(k), mx.multiply(mx.array(-1), k))
    vals_r = mx.take(r, idx_r)
    vals_c = mx.take(c, idx_c)
    return mx.where(mask, vals_r, vals_c)


def hankel(c, r=None):
    c = mx.reshape(mx.array(c), (-1,))
    m = int(c.shape[0])
    if r is None:
        # Default: zeros with r[0] = c[-1]
        r = mx.concatenate([mx.reshape(c[-1], (1,)), mx.zeros((max(m - 1, 0),), dtype=c.dtype)])
    r = mx.reshape(mx.array(r), (-1,))
    n = int(r.shape[0])
    vals = mx.concatenate([c, r[1:]])
    i = mx.expand_dims(mx.arange(m), axis=1)
    j = mx.expand_dims(mx.arange(n), axis=0)
    idx = mx.add(i, j)
    return mx.take(vals, idx)


def qr_insert(*args, **kwargs):
    raise NotImplementedError("scipy_mlx.linalg.qr_insert is not yet implemented for MLX.")


def qr_delete(*args, **kwargs):
    raise NotImplementedError("scipy_mlx.linalg.qr_delete is not yet implemented for MLX.")


def qr_update(*args, **kwargs):
    raise NotImplementedError("scipy_mlx.linalg.qr_update is not yet implemented for MLX.")

def cholesky(a, lower=False, overwrite_a=False, check_finite=True):
    # Many MLX linalg ops are CPU-only; run explicitly on the CPU stream.
    L = mx.linalg.cholesky(mx.array(a), stream=mx.cpu)
    if lower:
        return L
    return mx.transpose(L)


def inv(a):
    return mx.linalg.inv(mx.array(a), stream=mx.cpu)


def solve(a, b):
    return mx.linalg.solve(mx.array(a), mx.array(b), stream=mx.cpu)


def svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    return mx.linalg.svd(
        mx.array(a),
        full_matrices=full_matrices,
        compute_uv=compute_uv,
        stream=mx.cpu,
    )


def qr(a, mode="reduced"):
    q, r = mx.linalg.qr(mx.array(a), stream=mx.cpu)
    return q, r


def eig(a):
    return mx.linalg.eig(mx.array(a), stream=mx.cpu)


def eigh(a, lower=True, eigvals_only=False):
    w, v = mx.linalg.eigh(mx.array(a), stream=mx.cpu)
    return w if eigvals_only else (w, v)


def lu_factor(a, overwrite_a=False, check_finite=True):
    p, L, U = mx.linalg.lu(mx.array(a), stream=mx.cpu)
    lu = mx.add(U, mx.tril(L, k=-1))
    return lu, p


def lu_solve(lu_and_piv, b, trans=0, overwrite_b=False, check_finite=True):
    lu, piv = lu_and_piv
    lu = mx.array(lu)
    b = mx.array(b)
    n = int(lu.shape[0])
    L = mx.add(mx.tril(lu, k=-1), mx.eye(n, dtype=lu.dtype))
    U = mx.triu(lu)
    # Apply row permutation to RHS.
    b_perm = b[piv]
    y = mx.linalg.solve(L, b_perm, stream=mx.cpu)
    x = mx.linalg.solve(U, y, stream=mx.cpu)
    return x


def solve_triangular(a, b, lower=False, trans=0, unit_diagonal=False,
                     overwrite_b=False, check_finite=True):
    # MLX doesn't expose a specialized triangular solver; fall back to solve.
    return mx.linalg.solve(mx.array(a), mx.array(b), stream=mx.cpu)


def cho_factor(a, lower=False, overwrite_a=False, check_finite=True):
    c = cholesky(a, lower=lower, overwrite_a=overwrite_a, check_finite=check_finite)
    return c, lower


def cho_solve(c_and_lower, b, overwrite_b=False, check_finite=True):
    c, lower = c_and_lower
    c = mx.array(c)
    b = mx.array(b)
    if lower:
        y = mx.linalg.solve(c, b, stream=mx.cpu)
        x = mx.linalg.solve(mx.transpose(mx.conjugate(c)), y, stream=mx.cpu)
        return x
    y = mx.linalg.solve(mx.transpose(mx.conjugate(c)), b, stream=mx.cpu)
    x = mx.linalg.solve(c, y, stream=mx.cpu)
    return x


def lstsq(a, b, rcond=None):
    # Basic least squares via normal equations. Not numerically ideal but avoids LAPACK.
    a = mx.array(a)
    b = mx.array(b)
    at = mx.transpose(a)
    ata = mx.matmul(at, a)
    atb = mx.matmul(at, b)
    x = mx.linalg.solve(ata, atb, stream=mx.cpu)
    # residuals / rank / s are placeholders
    return x, None, None, None


def orthogonal_procrustes(A, B):
    A = mx.array(A)
    B = mx.array(B)
    m = mx.matmul(mx.transpose(mx.conjugate(A)), B)
    U, s, Vh = mx.linalg.svd(m, full_matrices=False, stream=mx.cpu)
    R = mx.matmul(U, Vh)
    scale = mx.sum(s)
    return R, scale


__all__ = [
    "LinAlgError",
    "LinAlgWarning",
    "norm",
    "cholesky",
    "issymmetric",
    "ishermitian",
    "get_lapack_funcs",
    "get_blas_funcs",
    "solve_banded",
    "cholesky_banded",
    "cho_solve_banded",
    "pinv",
    "toeplitz",
    "hankel",
    "qr_insert",
    "qr_delete",
    "qr_update",
    "inv",
    "solve",
    "svd",
    "qr",
    "eig",
    "eigh",
    "lu_factor",
    "lu_solve",
    "solve_triangular",
    "cho_factor",
    "cho_solve",
    "lstsq",
    "orthogonal_procrustes",
]
