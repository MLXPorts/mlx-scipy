"""
MLX stub for SciPy's compiled statistics helper extension (`stats._stats`).

Upstream SciPy uses compiled routines for performance and special-function
support in distributions. This MLX port provides a placeholder so imports can
proceed.
"""

from __future__ import annotations

from typing import Any, Callable


def _unimplemented(name: str) -> Callable[..., Any]:
    def _fn(*args, **kwargs):
        raise NotImplementedError(f"scipy_mlx.stats._stats.{name} is not yet implemented for MLX.")
    _fn.__name__ = name
    return _fn


def __getattr__(name: str) -> Any:  # pragma: no cover
    return _unimplemented(name)

