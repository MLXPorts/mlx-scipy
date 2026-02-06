"""
MLX-backed NumPy-compatibility namespace.

This package exists solely to avoid importing external NumPy while keeping
legacy SciPy/benchmark/test modules importable during the MLX port.

It is **not** a full NumPy implementation. New code should prefer using
`import mlx.core as mx` directly.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

# Ensure global MLX compatibility shims are installed on `mlx.core`.
from scipy_mlx._lib import _mlx_compat as _mlx_compat  # noqa: F401

# Common NumPy dtype aliases that appear in legacy tests/benchmarks.
double = mx.float64
cdouble = getattr(mx, "complex128", mx.complex64)
ndarray = mx.array

# Expose common numpy submodules.
from . import exceptions, fft, lib, linalg, ma, polynomial, random, testing  # noqa: E402,F401


def __getattr__(name: str) -> Any:
    # First try `mlx.core` directly.
    if hasattr(mx, name):
        return getattr(mx, name)
    # Fall back to shims defined in `_mlx_compat` (e.g. `dot`, `asarray`, ...).
    if hasattr(_mlx_compat, name):
        return getattr(_mlx_compat, name)
    raise AttributeError(name)


__all__ = [
    "mx",
    "ndarray",
    "double",
    "cdouble",
    "exceptions",
    "fft",
    "lib",
    "linalg",
    "ma",
    "polynomial",
    "random",
    "testing",
]

