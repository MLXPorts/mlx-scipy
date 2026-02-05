"""
MLX stub for SciPy's compiled Dierckx spline helpers (`interpolate._dierckx`).

SciPy uses a compiled extension for evaluating B-splines and constructing
collocation matrices. This MLX port does not ship that extension; the high-level
Python APIs remain importable but will raise when the backend is required.
"""

from __future__ import annotations

from typing import Any, Callable


def _unimplemented(name: str) -> Callable[..., Any]:
    def _fn(*args, **kwargs):
        raise NotImplementedError(f"scipy_mlx.interpolate._dierckx.{name} is not yet implemented for MLX.")
    _fn.__name__ = name
    return _fn


def __getattr__(name: str) -> Any:  # pragma: no cover
    return _unimplemented(name)

