"""
MLX stub for SciPy's compiled dopri/dop853 wrapper (`integrate._dop`).
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx


types = SimpleNamespace(
    intvar=SimpleNamespace(dtype=mx.int32),
)


def dopri5(*args, **kwargs):
    raise NotImplementedError("scipy_mlx.integrate.dopri5 is not yet implemented for MLX.")


def dop853(*args, **kwargs):
    raise NotImplementedError("scipy_mlx.integrate.dop853 is not yet implemented for MLX.")

