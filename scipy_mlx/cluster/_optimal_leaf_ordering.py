"""
MLX stub for SciPy's compiled `cluster._optimal_leaf_ordering` extension.

Upstream SciPy provides an optimized implementation for optimal leaf ordering.
This MLX port does not include compiled extensions; the public Python API can
still be imported, but the accelerated kernel is not available.
"""

from __future__ import annotations

from typing import Any, Callable


def _unimplemented(name: str) -> Callable[..., Any]:
    def _fn(*args, **kwargs):
        raise NotImplementedError(
            "scipy_mlx.cluster._optimal_leaf_ordering is not yet implemented for MLX."
        )

    _fn.__name__ = name
    return _fn


def __getattr__(name: str) -> Any:  # pragma: no cover
    return _unimplemented(name)

