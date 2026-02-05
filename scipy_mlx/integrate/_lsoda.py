"""
MLX stub for SciPy's compiled LSODA wrapper (`integrate._lsoda`).
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx


types = SimpleNamespace(
    intvar=SimpleNamespace(dtype=mx.int32),
)


def lsoda(*args, **kwargs):
    raise NotImplementedError("scipy_mlx.integrate.lsoda is not yet implemented for MLX.")

