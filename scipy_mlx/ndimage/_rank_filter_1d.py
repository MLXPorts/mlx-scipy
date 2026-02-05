"""
MLX stub for SciPy's compiled ndimage rank filter extension (`ndimage._rank_filter_1d`).
"""

from __future__ import annotations

from typing import Any, Callable


def _unimplemented(name: str) -> Callable[..., Any]:
    def _fn(*args, **kwargs):
        raise NotImplementedError(
            f"scipy_mlx.ndimage._rank_filter_1d.{name} is not yet implemented for MLX."
        )

    _fn.__name__ = name
    return _fn


def __getattr__(name: str) -> Any:  # pragma: no cover
    return _unimplemented(name)

