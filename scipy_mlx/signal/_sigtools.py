"""
Stub for SciPy's compiled `scipy.signal._sigtools` extension.

Provides a minimal subset of symbols referenced by pure-python signal code.
Most functions raise `NotImplementedError` in this MLX port.
"""

from __future__ import annotations

from typing import Any, Callable

import mlx.core as mx


def _unimplemented(name: str) -> Callable[..., Any]:
    def _fn(*args, **kwargs):
        raise NotImplementedError(f"scipy_mlx.signal.{name} is not implemented for MLX.")
    _fn.__name__ = name
    return _fn


_correlateND = _unimplemented("_correlateND")
_convolve2d = _unimplemented("_convolve2d")
_medfilt2d = _unimplemented("_medfilt2d")
_linear_filter = _unimplemented("_linear_filter")
_remez = _unimplemented("_remez")

__all__ = [
    "_correlateND",
    "_convolve2d",
    "_medfilt2d",
    "_linear_filter",
    "_remez",
]

