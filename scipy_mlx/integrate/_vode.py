"""
MLX stub for SciPy's compiled VODE/ZVODE wrapper (`integrate._vode`).

The pure-Python `integrate._ode` module expects a small `types` namespace at
import time. We provide that, while leaving the solver entry points
unimplemented.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx


types = SimpleNamespace(
    intvar=SimpleNamespace(dtype=mx.int32),
)


def dvode(*args, **kwargs):
    raise NotImplementedError("scipy_mlx.integrate.dvode is not yet implemented for MLX.")


def zvode(*args, **kwargs):
    raise NotImplementedError("scipy_mlx.integrate.zvode is not yet implemented for MLX.")

