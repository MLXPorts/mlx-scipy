"""
Minimal replacement for `numpy.linalg` for the MLX port.

Delegates to `mlx.core.linalg` first, then to `scipy_mlx.linalg` for
functionality not present in MLX.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.core.linalg as _mx_la

import scipy_mlx.linalg as _sp_la


LinAlgError = getattr(_sp_la, "LinAlgError", RuntimeError)


def cond(x, p=None):
    """Compute a simple 2-norm condition number (best-effort)."""
    if p not in (None, 2):
        raise NotImplementedError("Only 2-norm cond is supported in the MLX shim.")
    s = _mx_la.svd(mx.array(x), compute_uv=False)
    return s[0] / s[-1]


def __getattr__(name: str) -> Any:
    if hasattr(_mx_la, name):
        return getattr(_mx_la, name)
    if hasattr(_sp_la, name):
        return getattr(_sp_la, name)
    raise AttributeError(name)


__all__ = [
    "LinAlgError",
    "cond",
]

