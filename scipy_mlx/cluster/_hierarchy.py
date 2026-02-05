"""
MLX stub for SciPy's compiled `cluster._hierarchy` extension.

SciPy's hierarchical clustering algorithms rely on compiled code for speed.
This repository targets MLX and does not ship Cython extensions; we provide a
minimal stub so imports succeed. Calling functions here will raise
`NotImplementedError`.
"""

from __future__ import annotations

from typing import Any, Callable


def _unimplemented(name: str) -> Callable[..., Any]:
    def _fn(*args, **kwargs):
        raise NotImplementedError(
            f"scipy_mlx.cluster._hierarchy.{name} is not yet implemented for MLX."
        )

    _fn.__name__ = name
    return _fn


def __getattr__(name: str) -> Any:  # pragma: no cover
    return _unimplemented(name)

