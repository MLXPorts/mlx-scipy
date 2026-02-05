"""
MLX stub for SciPy's compiled FITPACK extension (`interpolate._fitpack`).

SciPy's spline wrappers are built on FITPACK, which is provided via compiled
extensions. This MLX port does not ship those extensions, so we provide a stub
module to keep imports working. Any attempt to call these symbols will raise
`NotImplementedError`.
"""

from __future__ import annotations

from typing import Any, Callable


def _unimplemented(name: str) -> Callable[..., Any]:
    def _fn(*args, **kwargs):
        raise NotImplementedError(
            f"scipy_mlx.interpolate._fitpack.{name} is not yet implemented for MLX."
        )

    _fn.__name__ = name
    return _fn


def __getattr__(name: str) -> Any:  # pragma: no cover
    return _unimplemented(name)

