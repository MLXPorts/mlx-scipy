"""
MLX stub for SciPy's spatial distance pybind extension.

Upstream SciPy provides a compiled extension (pybind11) containing fast
implementations of `cdist`/`pdist` for many metrics. In this MLX-first port,
we keep Python implementations where available and provide stubs for the
compiled entry points so that imports succeed.

Any attempt to call the missing compiled functions will raise
`NotImplementedError`.
"""

from __future__ import annotations

from typing import Any, Callable


def _unimplemented(name: str) -> Callable[..., Any]:
    def _fn(*args, **kwargs):
        raise NotImplementedError(
            f"scipy_mlx.spatial._distance_pybind.{name} is not yet implemented for MLX."
        )

    _fn.__name__ = name
    return _fn


def __getattr__(name: str) -> Any:  # pragma: no cover
    # Provide any `cdist_*`/`pdist_*` symbol on-demand.
    return _unimplemented(name)

